#!/usr/bin/env python3
"""Fit one conservative add-only CEM selector across two spent RECON cohorts."""

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

from tools.evaluation.tune_cem_hybrid_selector_v2 import (  # noqa: E402
    compact_metrics,
    constraints,
)
from tools.scoring.train_recon_dense_meta_dinov3_v1 import (  # noqa: E402
    file_sha256,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cem-results", type=Path, nargs=2, required=True)
    parser.add_argument("--frozen-scores", type=Path, nargs=2, required=True)
    parser.add_argument("--training-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_cohort(result_path: Path, score_path: Path) -> dict[str, Any]:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if (
        result.get("domain") != "recon"
        or not result.get("cem")
        or not result.get("confirmation_now_spent", result.get("cem_cal_now_spent"))
    ):
        raise RuntimeError(f"not a spent RECON CEM cohort: {result_path}")
    with np.load(result["outcomes"], allow_pickle=False) as payload:
        absolute = payload["combined_absolute_metrics"].astype(np.float32)
        if "universal_d_cem_false_selected_index" in payload.files:
            baseline = payload[
                "universal_d_cem_false_selected_index"
            ].astype(np.int64)
        else:
            freeze = json.loads(
                Path(result["freeze"]).read_text(encoding="utf-8")
            )
            with np.load(
                freeze["selected_index"], allow_pickle=False
            ) as selected:
                baseline = selected[
                    "universal_d_cem_false_selected_index"
                ].astype(np.int64)
    with np.load(score_path, allow_pickle=False) as payload:
        score = {
            "energy": payload["energy"].astype(np.float32),
            "mean": payload["mean"].astype(np.float32),
            "std": payload["std"].astype(np.float32),
            "p_dual": payload["dual_probability"].astype(np.float32),
            "p_harm": payload["harm_probability"].astype(np.float32),
            "valid": payload["valid"].astype(bool),
        }
    if absolute.shape != (100, 33, 4) or score["energy"].shape != (100, 17):
        raise RuntimeError("joint CEM tuning shape changed")
    return {
        "result": result,
        "absolute": absolute,
        "baseline": baseline,
        "baseline_metrics": compact_metrics(baseline, absolute),
        **score,
    }


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    training = json.loads(args.training_result.read_text(encoding="utf-8"))
    policy = training["arms"]["D"]["calibration"]["frontier"]["domains"][
        "recon"
    ]["policy"]
    cohorts = [
        load_cohort(result, score)
        for result, score in zip(args.cem_results, args.frozen_scores)
    ]
    for cohort in cohorts:
        row = np.arange(100)
        candidate = np.argmin(cohort["energy"], axis=1).astype(np.int64)
        cohort["candidate"] = candidate
        cohort["exists"] = (
            np.isfinite(cohort["energy"][row, candidate])
            & (candidate != 0)
        )
        z = float(policy["z"])
        cohort["ucb_ate"] = (
            cohort["mean"][row, candidate, 0]
            + z * cohort["std"][row, candidate, 0]
        )
        cohort["ucb_objective"] = (
            cohort["mean"][row, candidate, 3]
            + z * cohort["std"][row, candidate, 3]
        )

    grid = itertools.product(
        (0.00, 0.03, 0.06, 0.10, 0.15),
        (1.00, 0.80, 0.60, 0.40, 0.25),
        (0.000, 0.005, 0.010, 0.020, 0.030),
        (0.000, 0.005, 0.010, 0.020, 0.030),
        (0.00, -0.03, -0.06, -0.10, -0.16, -0.24, -0.32),
    )
    best_key: tuple[Any, ...] | None = None
    best: dict[str, Any] | None = None
    searched = 0
    feasible_count = 0
    for dual_offset, harm_scale, ate_extra, objective_extra, energy_cap in grid:
        searched += 1
        per_cohort = []
        all_feasible = True
        total_changed = 0
        total_gain = 0.0
        worst_gain = np.inf
        for cohort in cohorts:
            row = np.arange(100)
            candidate = cohort["candidate"]
            allow = (
                (cohort["baseline"] == 0)
                & cohort["exists"]
                & cohort["valid"][row, candidate]
                & (
                    cohort["p_dual"][row, candidate]
                    >= min(
                        0.999,
                        float(policy["dual_threshold"]) + dual_offset,
                    )
                )
                & (
                    cohort["p_harm"][row, candidate]
                    <= float(policy["harm_threshold"]) * harm_scale
                )
                & (
                    cohort["ucb_ate"]
                    < -(float(policy["margin_ate"]) + ate_extra)
                )
                & (
                    cohort["ucb_objective"]
                    < -(
                        float(policy["margin_objective"])
                        + objective_extra
                    )
                )
                & (
                    cohort["energy"][row, candidate] <= energy_cap
                )
            )
            selected = cohort["baseline"].copy()
            selected[allow] = 16 + candidate[allow]
            changed = int(allow.sum())
            value = compact_metrics(selected, cohort["absolute"])
            check = constraints(
                value, cohort["baseline_metrics"], changed=changed
            )
            # Add-only cannot remove coverage; require at least one accepted
            # CEM action on each cohort and no degradation on any metric.
            feasible = all(check.values())
            all_feasible &= feasible
            total_changed += changed
            gain = float(
                (
                    cohort["baseline_metrics"]["mean"] - value["mean"]
                ).sum()
            )
            total_gain += gain
            worst_gain = min(worst_gain, gain)
            per_cohort.append(
                {
                    "changed_windows": changed,
                    "feasible": feasible,
                    "constraints": check,
                    "metrics": {
                        key: (
                            item.tolist()
                            if isinstance(item, np.ndarray)
                            else item
                        )
                        for key, item in value.items()
                    },
                }
            )
        if all_feasible:
            feasible_count += 1
        key = (
            int(all_feasible),
            int(all(value["constraints"]["dual_precision_not_lower"] for value in per_cohort)),
            int(all(value["constraints"]["harm_rate_not_higher"] for value in per_cohort)),
            int(all(value["constraints"]["all_means_not_higher"] for value in per_cohort)),
            int(all(value["constraints"]["all_p90_not_higher"] for value in per_cohort)),
            float(worst_gain),
            float(total_gain),
            int(total_changed),
        )
        if best_key is None or key > best_key:
            best_key = key
            best = {
                "parameters": {
                    "dual_threshold_offset": dual_offset,
                    "harm_threshold_scale": harm_scale,
                    "ate_margin_extra_m": ate_extra,
                    "objective_margin_extra_m": objective_extra,
                    "conservative_energy_cap": energy_cap,
                    "replacement_mode": "add_only_when_fixed_pool_incumbent",
                },
                "per_cohort": per_cohort,
                "all_cohort_feasible": all_feasible,
                "total_changed_windows": total_changed,
                "score": list(key),
            }
    assert best is not None
    report = {
        "schema": "cem_joint_addonly_selector_v3_spent_development",
        "status": "joint_outcome_tuned_on_two_spent_recon_cohorts",
        "evidence_role": "development_only_requires_new_frozen_confirmation",
        "domain": "recon",
        "cem": True,
        "formal_claim_allowed": False,
        "selection_rule": (
            "Keep every frozen fixed-pool selection unchanged; only when it "
            "selects incumbent may a CEM challenger passing the joint strict "
            "risk gate be added."
        ),
        "fixed_pool_action_never_replaced": True,
        "searched_policies": searched,
        "feasible_policies": feasible_count,
        "best": best,
        "inputs": [
            {
                "cem_result": str(result),
                "cem_result_sha256": file_sha256(result),
                "frozen_scores": str(score),
                "frozen_scores_sha256": file_sha256(score),
            }
            for result, score in zip(args.cem_results, args.frozen_scores)
        ],
        "training_result": str(args.training_result),
        "training_result_sha256": file_sha256(args.training_result),
        "outcomes_used_for_tuning": True,
        "next_confirmation_accessed": False,
    }
    write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
