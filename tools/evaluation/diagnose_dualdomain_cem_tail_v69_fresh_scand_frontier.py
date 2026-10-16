#!/usr/bin/env python3
"""Spent-only diagnostic of v69 separability on the fresh SCAND final tail."""

from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from tools.evaluation.build_recon100_k17_source_v1 import metric_labels  # noqa: E402
from tools.evaluation.export_recon_dense_dinov3_features_v1 import load_npz  # noqa: E402
from tools.scoring.run_train_dualdomain_cem_tail_v66_candidate_level import (  # noqa: E402
    choose_candidate_level_cem_frontier,
)


BASE = (
    ROOT
    / "outputs/cem_development_v1/"
    "dualdomain_cem_tail_v69_scand_route_crossfit"
)


def main() -> None:
    score = load_npz(
        BASE
        / "replays/scand_fresh_outer/proposal_aware_v69/"
        "iteration_02/frozen_scores.npz"
    )
    selected = load_npz(
        BASE / "replays/scand_fresh_outer/proposal_aware_v69/selected_actions.npz"
    )
    outcomes = load_npz(
        ROOT
        / "outputs/scand_domain_adaptation_v1/fresh_outer_v1/"
        "outcomes/fresh_outer_outcomes.npz"
    )
    labels = metric_labels(
        selected["final_cem_feature_waypoint_units"].astype("float32"),
        outcomes["ground_truth_local_m"].astype("float32"),
        0.25,
        0.4,
        0.6,
    )
    prediction = {
        key: score[key]
        for key in ("mean", "std", "rank_logit")
    }
    result = choose_candidate_level_cem_frontier(
        {"scand": {
            "prediction": prediction,
            "p_dual": score["dual_probability"],
            "p_harm": score["harm_probability"],
            "valid": score["valid"].astype(bool),
            "targets": labels["targets"],
        }},
        {
            "minimum_coverage": 0.4,
            "target_dual_precision": 0.85,
            "target_wilson_lower_95": 0.80,
            "maximum_harm_rate": 0.15,
        },
        shared_policy=False,
    )["domains"]["scand"]
    frontier = result.pop("frontier")
    result.pop("selected")
    eligible = [
        value for value in frontier
        if value["metrics"]["coverage"] >= 0.4
    ]
    best_dual = max(
        eligible,
        key=lambda value: (
            value["metrics"]["dual_precision"] or 0.0,
            -(value["metrics"]["harm_rate"] or 1.0),
        ),
    )
    print(json.dumps({
        "schema": "v69_fresh_scand_frontier_spent_diagnostic",
        "evidence_role": "spent_diagnostic_only_no_policy_change_allowed",
        "frozen_policy_result": result,
        "best_policy_at_coverage_at_least_0_40": {
            "policy": best_dual["policy"],
            "metrics": best_dual["metrics"],
            "feasible": best_dual["feasible"],
        },
        "frontier_policies": len(frontier),
        "outcomes_opened_after_all_replays_frozen": True,
        "reserve_accessed": False,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
