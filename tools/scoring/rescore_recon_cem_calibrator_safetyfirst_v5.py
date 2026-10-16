#!/usr/bin/env python3
"""Rescore frozen route-disjoint CEM OOF predictions with a safety-first rule."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from tools.scoring.train_recon_cem_candidate_calibrator_v4 import (  # noqa: E402
    METRIC_NAMES,
    absolute_summary,
    file_sha256,
    load_cohort,
    safety,
    sigmoid,
    write_json,
    write_npz,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v4-result", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def optional_rate(value: float | None, default: float) -> float:
    return default if value is None else float(value)


def selected_for_policy(
    prediction: dict[str, np.ndarray],
    valid: np.ndarray,
    policy: dict[str, float],
) -> np.ndarray:
    p_dual = sigmoid(prediction["dual_logit"])
    p_harm = sigmoid(prediction["harm_logit"])
    mean = prediction["mean"]
    quality_utility = (
        mean[..., 0]
        + 0.25 * mean[..., 1]
        + 0.25 * np.maximum(mean[..., 2], 0.0)
        + 0.50 * mean[..., 3]
    )
    safety_risk = p_harm - p_dual
    score = (
        quality_utility
        + float(policy["safety_rank_weight"]) * safety_risk
    )
    allowed = (
        valid
        & (p_dual >= float(policy["dual_threshold"]))
        & (p_harm <= float(policy["harm_threshold"]))
        & (
            p_dual - p_harm
            >= float(policy["dual_harm_gap_threshold"])
        )
        & (mean[..., 0] < -float(policy["ate_margin"]))
        & (mean[..., 3] < -float(policy["objective_margin"]))
    )
    score = np.where(allowed, score, np.inf)
    candidate = score.argmin(axis=1)
    exists = np.isfinite(score[np.arange(len(score)), candidate])
    return np.where(exists, candidate + 1, 0).astype(np.int64)


def policy_evaluation(
    *,
    selected: np.ndarray,
    cohorts: list[dict[str, Any]],
    cohort_name: np.ndarray,
    delta: np.ndarray,
) -> dict[str, Any]:
    per_cohort: dict[str, Any] = {}
    safety_all = True
    quality_all = True
    penalty = 0.0
    total_coverage = 0.0
    for cohort in cohorts:
        mask = cohort_name == cohort["name"]
        local_selected = selected[mask]
        metrics = safety(local_selected, delta[mask])
        absolute = absolute_summary(
            local_selected, cohort["absolute_final"]
        )
        baseline = json.loads(
            Path(cohort["result_path"]).read_text(encoding="utf-8")
        )["methods"]["universal_d_cem_false"]
        checks = {
            "coverage_at_least_0.40": metrics["coverage"] >= 0.40,
            "dual_at_least_0.90": optional_rate(
                metrics["dual_precision"], 0.0
            )
            >= 0.90,
            "wilson_at_least_0.80": optional_rate(
                metrics["dual_wilson_lower_95"], 0.0
            )
            >= 0.80,
            "harm_at_most_0.10": optional_rate(
                metrics["harm_rate"], 1.0
            )
            <= 0.10,
        }
        quality_checks = {
            "means_not_worse": all(
                absolute[name]["mean_m"]
                <= baseline["absolute"][name]["mean_m"] + 1.0e-12
                for name in METRIC_NAMES
            ),
            "p90_not_worse": all(
                absolute[name]["p90_m"]
                <= baseline["absolute"][name]["p90_m"] + 1.0e-12
                for name in METRIC_NAMES
            ),
        }
        safety_all &= all(checks.values())
        quality_all &= all(quality_checks.values())
        coverage = metrics["coverage"]
        dual = optional_rate(metrics["dual_precision"], 0.0)
        wilson = optional_rate(metrics["dual_wilson_lower_95"], 0.0)
        harm = optional_rate(metrics["harm_rate"], 1.0)
        penalty += (
            12.0 * max(0.0, 0.40 - coverage)
            + 10.0 * max(0.0, 0.90 - dual)
            + 10.0 * max(0.0, 0.80 - wilson)
            + 10.0 * max(0.0, harm - 0.10)
            + (0.0 if quality_checks["means_not_worse"] else 0.25)
            + (0.0 if quality_checks["p90_not_worse"] else 0.25)
        )
        total_coverage += coverage
        per_cohort[cohort["name"]] = {
            "safety": metrics,
            "absolute": absolute,
            "safety_checks": checks,
            "quality_checks": quality_checks,
        }
    return {
        "per_cohort": per_cohort,
        "strict_safety_all_cohorts": safety_all,
        "quality_all_cohorts": quality_all,
        "strict_feasible": safety_all and quality_all,
        "penalty": float(penalty),
        "coverage_sum": float(total_coverage),
    }


def main() -> None:
    args = parse_args()
    result_path = args.output_dir / "result.json"
    if result_path.exists():
        raise FileExistsError(result_path)
    v4 = json.loads(args.v4_result.read_text(encoding="utf-8"))
    if v4.get("schema") != "recon_cem_candidate_calibrator_v4":
        raise RuntimeError("input is not the expected frozen v4 result")
    cohorts = [
        load_cohort(
            name=value["name"],
            feature_path=Path(value["features"]),
            score_path=Path(value["scores"]),
            result_path=Path(value["result"]),
        )
        for value in v4["inputs"]
    ]
    with np.load(v4["oof_predictions"], allow_pickle=False) as payload:
        prediction = {
            "mean": payload["mean"].astype(np.float32),
            "dual_logit": payload["dual_logit"].astype(np.float32),
            "harm_logit": payload["harm_logit"].astype(np.float32),
        }
        oof_record_key = payload["record_key"].astype(str)
        cohort_name = payload["cohort"].astype(str)
        route_group = payload["route_group"].astype(str)
        fold_id = payload["fold_id"].astype(np.int64)
    expected_record_key = np.concatenate(
        [value["record_key"] for value in cohorts]
    )
    if not np.array_equal(oof_record_key, expected_record_key):
        raise RuntimeError("v4 OOF/cohort identity mismatch")
    valid = np.concatenate([value["valid"] for value in cohorts])
    delta = np.concatenate([value["delta"] for value in cohorts])

    policies: list[dict[str, Any]] = []
    for safety_rank_weight in (0.0, 0.10, 0.25, 0.50, 1.0, 2.0, 4.0):
        for dual_threshold in np.linspace(0.40, 0.95, 12):
            for harm_threshold in (
                0.05,
                0.10,
                0.15,
                0.20,
                0.25,
                0.30,
                0.40,
                0.50,
            ):
                for gap_threshold in (-0.10, 0.0, 0.10, 0.20, 0.30):
                    for ate_margin in (0.0, 0.005, 0.010, 0.020):
                        for objective_margin in (
                            0.0,
                            0.005,
                            0.010,
                            0.020,
                        ):
                            policy = {
                                "safety_rank_weight": safety_rank_weight,
                                "dual_threshold": float(dual_threshold),
                                "harm_threshold": harm_threshold,
                                "dual_harm_gap_threshold": gap_threshold,
                                "ate_margin": ate_margin,
                                "objective_margin": objective_margin,
                            }
                            selected = selected_for_policy(
                                prediction, valid, policy
                            )
                            evaluation = policy_evaluation(
                                selected=selected,
                                cohorts=cohorts,
                                cohort_name=cohort_name,
                                delta=delta,
                            )
                            policies.append(
                                {
                                    "policy": policy,
                                    "selected": selected,
                                    **evaluation,
                                }
                            )
    feasible = [value for value in policies if value["strict_feasible"]]
    if feasible:
        best = max(
            feasible,
            key=lambda value: (
                value["coverage_sum"],
                -value["penalty"],
            ),
        )
        tier = "strict_all_three_oof_cohorts"
    else:
        best = min(
            policies,
            key=lambda value: (
                value["penalty"],
                -value["coverage_sum"],
            ),
        )
        tier = "best_available_oof"

    args.output_dir.mkdir(parents=True, exist_ok=False)
    selection_path = args.output_dir / "oof_selection.npz"
    write_npz(
        selection_path,
        record_key=oof_record_key,
        cohort=cohort_name,
        route_group=route_group,
        fold_id=fold_id,
        selected_index=best["selected"],
    )
    report = {
        "schema": "recon_cem_candidate_safetyfirst_policy_v5",
        "status": "frozen_v4_route_disjoint_oof_rescored",
        "evidence_role": "development_only_three_spent_cohorts",
        "cem": True,
        "states": int(len(oof_record_key)),
        "routes": int(len(np.unique(route_group))),
        "policy_search": {
            "tier": tier,
            "strict_feasible_policies": len(feasible),
            "searched_policies": len(policies),
            "policy": best["policy"],
            "per_cohort": best["per_cohort"],
            "strict_safety_all_cohorts": best[
                "strict_safety_all_cohorts"
            ],
            "quality_all_cohorts": best["quality_all_cohorts"],
            "strict_feasible": best["strict_feasible"],
            "penalty": best["penalty"],
            "coverage_sum": best["coverage_sum"],
        },
        "source_v4_result": str(args.v4_result),
        "source_v4_result_sha256": file_sha256(args.v4_result),
        "source_v4_oof": v4["oof_predictions"],
        "source_v4_oof_sha256": file_sha256(Path(v4["oof_predictions"])),
        "selection": str(selection_path),
        "selection_sha256": file_sha256(selection_path),
        "model_checkpoints_reused_without_retraining": v4["checkpoints"],
        "next_confirmation_accessed": False,
        "formal_claim_allowed": False,
    }
    write_json(result_path, report)
    print(
        json.dumps(
            {
                "result": str(result_path),
                "result_sha256": file_sha256(result_path),
                "policy_search": report["policy_search"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
