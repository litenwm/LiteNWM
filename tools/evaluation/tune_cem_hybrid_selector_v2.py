#!/usr/bin/env python3
"""Tune a fixed-pool-fallback CEM selector on spent cem-cal development data.

This script intentionally opens already-spent cem-cal outcomes.  Its output is
development evidence only: it searches a small, predeclared grid of stricter
frozen-scorer gates and always falls back to the original CEM=false selection.
It never removes a fixed-pool promotion merely to inflate precision.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from tools.evaluation.evaluate_scand_fresh_outer_v1 import method_metrics  # noqa: E402
from tools.scoring.train_recon_dense_meta_dinov3_v1 import (  # noqa: E402
    file_sha256,
    write_json,
    write_npz,
)


METRICS = ("ate", "rpe", "endpoint", "objective")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", choices=("recon", "scand"), required=True)
    parser.add_argument("--cem-result", type=Path, required=True)
    parser.add_argument("--selected", type=Path, required=True)
    parser.add_argument("--frozen-scores", type=Path, required=True)
    parser.add_argument("--training-result", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260721)
    return parser.parse_args()


def policy_for_domain(training: dict[str, Any], domain: str) -> dict[str, float]:
    return training["arms"]["D"]["calibration"]["frontier"]["domains"][domain][
        "policy"
    ]


def compact_metrics(selected: np.ndarray, absolute: np.ndarray) -> dict[str, Any]:
    row = np.arange(len(selected))
    incumbent = absolute[:, 0]
    chosen = absolute[row, selected]
    delta = chosen - incumbent
    promote = selected != 0
    count = int(promote.sum())
    dual = int(
        ((delta[promote, 0] < -1.0e-7) & (delta[promote, 3] < -1.0e-7)).sum()
    )
    harm = int(
        ((delta[promote, 0] > 1.0e-7) | (delta[promote, 3] > 1.0e-7)).sum()
    )
    return {
        "promotions": count,
        "coverage": float(count / len(selected)),
        "dual_success": dual,
        "dual_precision": float(dual / count) if count else None,
        "harm_count": harm,
        "harm_rate": float(harm / count) if count else None,
        "mean": chosen.mean(axis=0),
        "p90": np.quantile(chosen, 0.9, axis=0),
    }


def constraints(
    value: dict[str, Any],
    baseline: dict[str, Any],
    *,
    changed: int,
) -> dict[str, bool]:
    return {
        "coverage_not_lower": value["coverage"] + 1.0e-12 >= baseline["coverage"],
        "dual_precision_not_lower": (
            value["dual_precision"] is not None
            and baseline["dual_precision"] is not None
            and value["dual_precision"] + 1.0e-12 >= baseline["dual_precision"]
        ),
        "harm_rate_not_higher": (
            value["harm_rate"] is not None
            and baseline["harm_rate"] is not None
            and value["harm_rate"] <= baseline["harm_rate"] + 1.0e-12
        ),
        "all_means_not_higher": bool(
            np.all(value["mean"] <= baseline["mean"] + 1.0e-12)
        ),
        "all_p90_not_higher": bool(
            np.all(value["p90"] <= baseline["p90"] + 1.0e-12)
        ),
        "at_least_one_mean_strictly_better": bool(
            np.any(value["mean"] < baseline["mean"] - 1.0e-7)
        ),
        "cem_changes_at_least_one_window": changed > 0,
    }


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = json.loads(args.cem_result.read_text(encoding="utf-8"))
    if (
        result.get("evidence_role")
        != "development_only_cem_cal_spent_after_this_evaluation"
        or result.get("domain") != args.domain
        or not result.get("cem")
        or not result.get("cem_cal_now_spent")
    ):
        raise RuntimeError("input is not a spent CEM development result")
    outcome_path = Path(result["outcomes"])
    if file_sha256(outcome_path) != result["outcomes_sha256"]:
        raise RuntimeError("spent CEM outcomes changed")
    with np.load(outcome_path, allow_pickle=False) as payload:
        absolute = payload["combined_absolute_metrics"].astype(np.float32)
        trajectory = payload["trajectory"].astype(str)
        route = payload["route_group"].astype(str)
        record_key = payload["record_key"].astype(str)
        baseline = payload[
            "universal_d_cem_false_selected_index"
        ].astype(np.int64)
    with np.load(args.selected, allow_pickle=False) as payload:
        if not np.array_equal(payload["record_key"].astype(str), record_key):
            raise RuntimeError("selection/outcome identity mismatch")
    with np.load(args.frozen_scores, allow_pickle=False) as payload:
        energy = payload["energy"].astype(np.float32)
        mean = payload["mean"].astype(np.float32)
        std = payload["std"].astype(np.float32)
        p_dual = payload["dual_probability"].astype(np.float32)
        p_harm = payload["harm_probability"].astype(np.float32)
        valid = payload["valid"].astype(bool)
    if energy.shape != (100, 17) or absolute.shape != (100, 33, 4):
        raise RuntimeError("CEM hybrid development shape changed")

    training = json.loads(args.training_result.read_text(encoding="utf-8"))
    policy = policy_for_domain(training, args.domain)
    row = np.arange(100)
    candidate = np.argmin(energy, axis=1).astype(np.int64)
    exists = np.isfinite(energy[row, candidate]) & (candidate != 0)
    combined_candidate = 16 + candidate
    z = float(policy["z"])
    ucb_ate = mean[row, candidate, 0] + z * std[row, candidate, 0]
    ucb_objective = mean[row, candidate, 3] + z * std[row, candidate, 3]
    candidate_energy = energy[row, candidate]

    baseline_metrics = compact_metrics(baseline, absolute)
    grid = itertools.product(
        (0.00, 0.03, 0.06, 0.10),
        (1.00, 0.80, 0.60, 0.40),
        (0.000, 0.005, 0.010, 0.020),
        (0.000, 0.005, 0.010, 0.020),
        (0.00, -0.03, -0.06, -0.10, -0.16),
    )
    records: list[dict[str, Any]] = []
    best: tuple[Any, ...] | None = None
    best_record: dict[str, Any] | None = None
    for dual_offset, harm_scale, ate_extra, objective_extra, energy_cap in grid:
        allow = (
            exists
            & valid[row, candidate]
            & (
                p_dual[row, candidate]
                >= min(0.999, float(policy["dual_threshold"]) + dual_offset)
            )
            & (
                p_harm[row, candidate]
                <= float(policy["harm_threshold"]) * harm_scale
            )
            & (
                ucb_ate
                < -(float(policy["margin_ate"]) + ate_extra)
            )
            & (
                ucb_objective
                < -(float(policy["margin_objective"]) + objective_extra)
            )
            & (candidate_energy <= energy_cap)
        )
        selected = baseline.copy()
        selected[allow] = combined_candidate[allow]
        changed = int((selected != baseline).sum())
        value = compact_metrics(selected, absolute)
        check = constraints(value, baseline_metrics, changed=changed)
        feasible = all(check.values())
        mean_gain = baseline_metrics["mean"] - value["mean"]
        p90_gain = baseline_metrics["p90"] - value["p90"]
        score = (
            int(feasible),
            int(check["dual_precision_not_lower"]),
            int(check["harm_rate_not_higher"]),
            int(check["coverage_not_lower"]),
            int(check["all_means_not_higher"]),
            int(check["all_p90_not_higher"]),
            float(mean_gain[[0, 2, 3]].sum()),
            float(mean_gain[1]),
            float(p90_gain.sum()),
            -changed,
        )
        record = {
            "parameters": {
                "dual_threshold_offset": dual_offset,
                "harm_threshold_scale": harm_scale,
                "ate_margin_extra_m": ate_extra,
                "objective_margin_extra_m": objective_extra,
                "conservative_energy_cap": energy_cap,
            },
            "changed_windows": changed,
            "metrics": {
                key: (
                    value[key].tolist()
                    if isinstance(value[key], np.ndarray)
                    else value[key]
                )
                for key in value
            },
            "constraints": check,
            "feasible": feasible,
            "score": list(score),
        }
        records.append(record)
        if best is None or score > best:
            best = score
            best_record = {**record, "selected": selected}
    assert best_record is not None
    selected = best_record.pop("selected")
    selected_path = args.output_dir / "selected_index.npz"
    write_npz(
        selected_path,
        record_key=record_key,
        universal_d_cem_false_selected_index=baseline,
        cem_hybrid_v2_selected_combined_index=selected,
        cem_candidate_local_index=candidate,
    )
    baseline_full = method_metrics(
        baseline,
        absolute,
        trajectory,
        route,
        replicates=args.bootstrap_replicates,
        seed=args.seed,
    )
    selected_full = method_metrics(
        selected,
        absolute,
        trajectory,
        route,
        replicates=args.bootstrap_replicates,
        seed=args.seed + 1,
    )
    report = {
        "schema": "cem_hybrid_selector_v2_spent_development",
        "status": "outcome_tuned_on_spent_cem_cal",
        "evidence_role": "in_sample_development_only_not_outer_evidence",
        "domain": args.domain,
        "cem": True,
        "formal_claim_allowed": False,
        "search_space_size": len(records),
        "selection_rule": (
            "Always retain the frozen CEM=false selection unless the best "
            "CEM candidate passes the stricter frozen-scorer risk gate."
        ),
        "fixed_pool_promotion_never_removed": True,
        "best": best_record,
        "methods": {
            "universal_d_cem_false": baseline_full,
            "cem_hybrid_v2": selected_full,
        },
        "selected_index": str(selected_path),
        "selected_index_sha256": file_sha256(selected_path),
        "inputs": {
            "cem_result": str(args.cem_result),
            "cem_result_sha256": file_sha256(args.cem_result),
            "selected_actions_sha256": file_sha256(args.selected),
            "frozen_scores_sha256": file_sha256(args.frozen_scores),
            "training_result_sha256": file_sha256(args.training_result),
            "outcomes_sha256": file_sha256(outcome_path),
        },
        "outcomes_already_opened": True,
        "hyperparameters_selected_with_outcomes": True,
    }
    write_json(args.output_dir / "result.json", report)
    print(
        json.dumps(
            {
                "domain": args.domain,
                "best": best_record,
                "baseline": {
                    "promotions": baseline_full["promotions"],
                    "dual": baseline_full["dual_precision"],
                    "wilson": baseline_full["dual_wilson_lower_95"],
                    "harm": baseline_full["harm_rate_per_promotion"],
                    "absolute": baseline_full["absolute"],
                },
                "cem_hybrid_v2": {
                    "promotions": selected_full["promotions"],
                    "dual": selected_full["dual_precision"],
                    "wilson": selected_full["dual_wilson_lower_95"],
                    "harm": selected_full["harm_rate_per_promotion"],
                    "absolute": selected_full["absolute"],
                },
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
