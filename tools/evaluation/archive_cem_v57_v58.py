#!/usr/bin/env python3
"""Create the immutable v57/v58 development archive and one strategy draft."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from tools.scoring.train_recon_dense_meta_dinov3_v1 import (  # noqa: E402
    file_sha256,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ef-result", type=Path, required=True)
    parser.add_argument("--attribution", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = read_json(args.ef_result)
    attribution = read_json(args.attribution)
    if (
        result.get("status")
        != "frozen_one_time_EF_development_failed_substantial_gain_rule"
        or attribution.get("status")
        != "spent_EF_attribution_complete_no_new_cohort_run"
        or result.get("production_safe_claim_allowed")
        or attribution.get("production_safe_claim_allowed")
        or result.get("reserve_accessed")
        or attribution.get("reserve_accessed")
    ):
        raise RuntimeError("v57/v58 evidence role is invalid")
    pre_path = Path(str(result["preoutcome_freeze"]))
    if file_sha256(pre_path) != result["preoutcome_freeze_sha256"]:
        raise RuntimeError("pre-outcome freeze changed")
    pre = read_json(pre_path)
    reserve_path = Path(str(pre["reserve_freeze"]))
    reserve = read_json(reserve_path)
    if (
        file_sha256(reserve_path) != pre["reserve_freeze_sha256"]
        or reserve.get("labels_opened")
        or reserve.get("outcomes_accessed")
    ):
        raise RuntimeError("reserve is no longer sealed")

    comparison = attribution["pooled_relative_to_addonly"]["unified_union"]
    quality_delta = comparison["absolute_delta"]
    all_quality_nondegraded = all(
        quality_delta[metric][summary] <= 1.0e-12
        for metric in ("ate", "rpe", "endpoint", "objective")
        for summary in ("mean_m", "p90_m")
    )
    tail_gain = max(
        -quality_delta[metric]["p90_m"]
        for metric in ("ate", "rpe", "endpoint", "objective")
    )
    descriptive_pareto = bool(
        comparison["coverage_delta"] >= -1.0e-12
        and comparison["dual_delta"] >= -1.0e-12
        and comparison["harm_delta"] <= 1.0e-12
        and all_quality_nondegraded
        and tail_gain >= 0.005
    )
    strategy = None
    if descriptive_pareto:
        strategy = {
            "name": "v58_unified_union_single_gate_draft",
            "status": "draft_only_not_run_on_new_cohort",
            "rule": (
                "Run the frozen 3x16 safe-elite CEM. Apply one shared candidate "
                "gate and the same three-scorer LCB energy to both original K17 "
                "and final CEM candidates. Select the lowest-energy safe action "
                "over the union, otherwise incumbent."
            ),
            "fixed_action_privilege": False,
            "state_gate": False,
            "outcome_used_by_online_selection": False,
            "why_drafted": (
                "On spent E/F it preserved coverage/dual/harm exactly and was "
                "noninferior on all eight pooled mean/p90 summaries, including "
                "material Endpoint and Objective p90 reductions."
            ),
            "why_not_validated": (
                "The rule was chosen after outcome-aware E/F attribution. It "
                "must be preregistered and frozen before any new cohort."
            ),
            "new_cohort_run_authorized": False,
        }

    code_paths = [
        ROOT / "tools/scoring/calibrate_cem_candidate_gate_lobo_v57b.py",
        ROOT / "tools/planning/run_cem_lobo_gate_v57.py",
        ROOT / "tools/planning/freeze_cem_lobo_gate_v57_contract.py",
        ROOT / "tools/planning/freeze_cem_lobo_gate_v57_ef.py",
        ROOT / "tools/evaluation/summarize_cem_lobo_gate_v57_ef.py",
        ROOT / "tools/evaluation/diagnose_cem_primary_union_v58.py",
    ]
    archive = {
        "schema": "cem_v57_v58_spent_EF_archive",
        "status": "current_frozen_results_archived_and_attribution_closed",
        "evidence_role": "development_only_not_validation_not_production_safe",
        "addonly_result": {
            "path": str(args.ef_result),
            "sha256": file_sha256(args.ef_result),
            "substantial_gain_checks": result["substantial_gain_checks"],
            "pooled_funnel": result["pooled_funnel"],
            "predicted_vs_true_utility": result[
                "predicted_vs_true_utility"
            ],
            "automatic_acceptance_audit": result[
                "automatic_acceptance_audit"
            ],
        },
        "attribution_diagnostic": {
            "path": str(args.attribution),
            "sha256": file_sha256(args.attribution),
            "pooled_methods": attribution["pooled_methods"],
            "pooled_relative_to_addonly": attribution[
                "pooled_relative_to_addonly"
            ],
            "unified_union_replacement_subset": attribution[
                "pooled_unified_union_replacement_subset"
            ],
        },
        "preoutcome_freeze": {
            "path": str(pre_path),
            "sha256": file_sha256(pre_path),
            "E_F_labels_opened_at_freeze": False,
            "E_F_outcomes_accessed_at_freeze": False,
            "total_states": pre["total_states"],
            "total_routes": pre["total_routes"],
            "route_overlap": len(pre["E_F_route_overlap"]),
        },
        "reserve": {
            "path": str(reserve_path),
            "sha256": file_sha256(reserve_path),
            "labels_opened": False,
            "outcomes_accessed": False,
        },
        "descriptive_pareto_rule": {
            "coverage_dual_harm_nondegraded": True,
            "all_eight_mean_p90_nondegraded": True,
            "at_least_one_p90_improvement_m": 0.005,
            "passed": descriptive_pareto,
            "observed_largest_p90_improvement_m": tail_gain,
            "not_a_preregistered_significance_test": True,
        },
        "single_candidate_strategy_draft": strategy,
        "code_sha256": {
            str(path.relative_to(ROOT)): file_sha256(path) for path in code_paths
        },
        "post_E_F_tuning_performed": False,
        "new_cohort_run_or_opened": False,
        "reserve_accessed": False,
        "production_safe_claim_allowed": False,
    }
    write_json(args.output, archive)
    print(json.dumps(archive, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
