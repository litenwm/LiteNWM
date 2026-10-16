#!/usr/bin/env python3
"""Nested fixed-threshold gate on frozen universal scorer outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tools.scoring.screen_recon_target_bank_models_v22 import load_bank
from tools.scoring.train_recon_cem_state_gate_v6 import method_summary, optional
from tools.scoring.train_recon_dense_meta_dinov3_v1 import file_sha256, write_json, write_npz


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def checks(metrics: dict) -> dict[str, bool]:
    return {
        "coverage_at_least_0.40": metrics["coverage"] >= 0.40,
        "dual_at_least_0.90": optional(metrics["dual_precision"], 0.0) >= 0.90,
        "wilson_at_least_0.80": optional(metrics["dual_wilson_lower_95"], 0.0) >= 0.80,
        "harm_at_most_0.10": optional(metrics["harm_rate"], 1.0) <= 0.10,
    }


def load_score(root: Path, letter: str) -> dict[str, np.ndarray]:
    path = root / f"bank_{letter}" / "universal_scores.npz"
    with np.load(path, allow_pickle=False) as payload:
        return {
            "record": payload["record_key"].astype(str),
            "mean": payload["mean"][:, 1:].astype(np.float32),
            "std": payload["std"][:, 1:].astype(np.float32),
            "dual": payload["dual_probability"][:, 1:].astype(np.float32),
            "harm": payload["harm_probability"][:, 1:].astype(np.float32),
            "rank": payload["rank_logit"][:, 1:].astype(np.float32),
            "sha256": file_sha256(path),
        }


def candidate_confidence(
    score: dict[str, np.ndarray],
    valid: np.ndarray,
    policy: dict[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    predicted_improvement = -(score["mean"][..., 0] + score["mean"][..., 3])
    predicted_uncertainty = score["std"][..., 0] + score["std"][..., 3]
    value = (
        score["dual"]
        - policy["harm_weight"] * score["harm"]
        + policy["mean_weight"] * predicted_improvement
        - policy["uncertainty_weight"] * predicted_uncertainty
        + policy["rank_weight"] * score["rank"]
    )
    value = np.where(valid, value, -np.inf)
    candidate = value.argmax(1)
    row = np.arange(len(candidate))
    return candidate, value[row, candidate]


def selected_for(
    score: dict[str, np.ndarray],
    valid: np.ndarray,
    policy: dict[str, float],
) -> np.ndarray:
    candidate, confidence = candidate_confidence(score, valid, policy)
    promote = np.isfinite(confidence) & (confidence >= policy["threshold"])
    return np.where(promote, candidate + 1, 0).astype(np.int64)


def calibrate(
    banks: list[dict], scores: list[dict], train_names: set[str]
) -> dict:
    candidates = []
    for harm_weight in (0.5, 1.0, 2.0, 4.0):
        for mean_weight in (0.0, 0.25, 0.5, 1.0, 2.0):
            for uncertainty_weight in (0.0, 0.25, 0.5, 1.0, 2.0):
                for rank_weight in (0.0, 0.05, 0.10, 0.25):
                    base = {
                        "harm_weight": harm_weight,
                        "mean_weight": mean_weight,
                        "uncertainty_weight": uncertainty_weight,
                        "rank_weight": rank_weight,
                    }
                    confidence_values = []
                    for bank, score in zip(banks, scores):
                        if bank["name"] in train_names:
                            confidence_values.append(
                                candidate_confidence(score, bank["valid"], base)[1]
                            )
                    confidence = np.concatenate(confidence_values)
                    thresholds = np.unique(
                        np.quantile(confidence, np.linspace(0.0, 0.75, 31))
                    )
                    for threshold in thresholds:
                        policy = {**base, "threshold": float(threshold)}
                        per_bank = {}
                        for bank, score in zip(banks, scores):
                            if bank["name"] not in train_names:
                                continue
                            selected = selected_for(score, bank["valid"], policy)
                            metrics = method_summary(
                                selected, bank["delta"], bank["absolute"]
                            )
                            per_bank[bank["name"]] = {
                                "metrics": metrics, "checks": checks(metrics),
                            }
                        strict = all(
                            all(value["checks"].values()) for value in per_bank.values()
                        )
                        candidates.append({
                            "policy": policy, "per_bank": per_bank, "strict": strict,
                        })
    feasible = [value for value in candidates if value["strict"]]
    def key(value: dict):
        metrics = [item["metrics"] for item in value["per_bank"].values()]
        return (
            sum(sum(item["checks"].values()) for item in value["per_bank"].values()),
            min(optional(metric["dual_wilson_lower_95"], 0.0) for metric in metrics),
            min(optional(metric["dual_precision"], 0.0) for metric in metrics),
            -max(optional(metric["harm_rate"], 1.0) for metric in metrics),
            min(metric["coverage"] for metric in metrics),
            sum(metric["coverage"] for metric in metrics),
        )
    best = max(feasible if feasible else candidates, key=key)
    return {
        "strict_feasible": bool(feasible),
        "strict_feasible_policies": len(feasible),
        "searched_policies": len(candidates),
        "policy": best["policy"],
        "calibration_banks": best["per_bank"],
    }


def main() -> None:
    args = parse_args()
    result_path = args.output_dir / "result.json"
    if result_path.exists():
        raise FileExistsError(result_path)
    banks = [load_bank(args.root, value) for value in "abcdef"]
    scores = [load_score(args.root, value) for value in "abcdef"]
    for bank, score in zip(banks, scores):
        if not np.array_equal(bank["record"], score["record"]):
            raise RuntimeError("frozen score identity mismatch")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    selected_oof = np.zeros(600, dtype=np.int64)
    folds = {}
    cohort = np.concatenate([
        np.full(100, bank["name"], dtype="<U16") for bank in banks
    ])
    for outer_index, outer in enumerate(banks):
        train_names = {bank["name"] for bank in banks if bank["name"] != outer["name"]}
        calibration = calibrate(banks, scores, train_names)
        selected = selected_for(
            scores[outer_index], outer["valid"], calibration["policy"]
        )
        selected_oof[outer_index * 100 : (outer_index + 1) * 100] = selected
        metrics = method_summary(selected, outer["delta"], outer["absolute"])
        folds[outer["name"]] = {
            "outer_metrics": metrics,
            "outer_checks": checks(metrics),
            "inner_calibration": calibration,
        }
    strict = all(all(value["outer_checks"].values()) for value in folds.values())
    delta = np.concatenate([bank["delta"] for bank in banks])
    absolute = np.concatenate([bank["absolute"] for bank in banks])
    prediction_path = args.output_dir / "selected_index.npz"
    write_npz(
        prediction_path,
        record_key=np.concatenate([bank["record"] for bank in banks]),
        cohort=cohort,
        selected_index=selected_oof,
    )
    worst = {
        "coverage": min(value["outer_metrics"]["coverage"] for value in folds.values()),
        "dual_precision": min(optional(value["outer_metrics"]["dual_precision"], 0.0) for value in folds.values()),
        "dual_wilson_lower_95": min(optional(value["outer_metrics"]["dual_wilson_lower_95"], 0.0) for value in folds.values()),
        "harm_rate": max(optional(value["outer_metrics"]["harm_rate"], 1.0) for value in folds.values()),
    }
    report = {
        "schema": "recon_frozen_universal_gate_v23",
        "status": (
            "all_six_outer_banks_passed_cem_replacement_may_train"
            if strict else "frozen_universal_nested_gate_failed_continue_base_research"
        ),
        "evidence_role": "target_development_nested_five_bank_fixed_threshold",
        "states": 600, "routes": 150, "banks": 6,
        "online_causal": True,
        "future_route_quota": False,
        "model_retrained": False,
        "outer_folds": folds,
        "worst_outer_fold": worst,
        "pooled_nested_oof": method_summary(selected_oof, delta, absolute),
        "strict_all_outer_folds": strict,
        "selected_index": str(prediction_path),
        "selected_index_sha256": file_sha256(prediction_path),
        "score_sha256": {bank["name"]: score["sha256"] for bank, score in zip(banks, scores)},
        "final_reserve_accessed": False,
        "final_reserve_remains_sealed": True,
        "cem": False,
    }
    write_json(result_path, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
