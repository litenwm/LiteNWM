#!/usr/bin/env python3
"""Pool frozen B/C v56 development results and freeze the conclusion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from tools.evaluation.evaluate_cem_safe_addonly_v56 import (  # noqa: E402
    NAMES,
    added_subset_report,
)
from tools.evaluation.evaluate_scand_fresh_outer_v1 import (  # noqa: E402
    method_metrics,
)
from tools.scoring.train_recon_dense_meta_dinov3_v1 import (  # noqa: E402
    file_sha256,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, action="append", required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=20000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261057)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(args.result) != 2:
        raise ValueError("v56 pooled result requires exactly two frozen cohorts")
    if args.output.exists():
        raise FileExistsError(args.output)
    preregistration = json.loads(
        args.preregistration.read_text(encoding="utf-8")
    )
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in args.result]
    if any(
        report.get("schema") != "recon_cem_safe_addonly_v56_development_result"
        or report.get("reserve_accessed")
        or report.get("state_gate_used")
        or report.get("fixed_pool_action_replacement_allowed")
        for report in reports
    ):
        raise RuntimeError("invalid v56 cohort result")

    absolute_parts = []
    false_parts = []
    hybrid_parts = []
    added_parts = []
    trajectory_parts = []
    route_parts = []
    cohort_names = []
    audit_bindings = []
    closure_logs = {}
    route_sets = []
    for path, report in zip(args.result, reports):
        outcome_path = Path(report["outcomes"])
        if file_sha256(outcome_path) != report["outcomes_sha256"]:
            raise RuntimeError(f"outcome artifact changed: {outcome_path}")
        freeze_path = Path(report["freeze"])
        freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
        if file_sha256(freeze_path) != report["freeze_sha256"]:
            raise RuntimeError(f"selector freeze changed: {freeze_path}")
        cem_freeze_path = Path(freeze["cem_freeze"])
        cem_freeze = json.loads(cem_freeze_path.read_text(encoding="utf-8"))
        with np.load(outcome_path, allow_pickle=False) as payload:
            absolute_parts.append(payload["combined_absolute_metrics"].astype(np.float32))
            false_parts.append(payload["safe_cem_false_selected_index"].astype(np.int64))
            hybrid_parts.append(payload["hybrid_combined_selected_index"].astype(np.int64))
            added_parts.append(payload["cem_added_window"].astype(bool))
            trajectory_parts.append(payload["trajectory"].astype(str))
            route_parts.append(payload["route_group"].astype(str))
        cohort = freeze_path.parents[1].name
        cohort_names.append(cohort)
        routes = set(route_parts[-1])
        route_sets.append(routes)
        closure_logs[cohort] = {
            "outcome_free_iteration_audit": cem_freeze["iteration_audit"],
            "predicted_vs_true_elite_utility": report[
                "iteration_elite_predicted_vs_true"
            ],
        }
        audit_bindings.append(
            {
                "cohort": cohort,
                "result": str(path),
                "result_sha256": file_sha256(path),
                "freeze": str(freeze_path),
                "freeze_sha256": file_sha256(freeze_path),
                "selected_index_sha256": freeze["selected_index_sha256"],
                "cem_freeze_sha256": freeze["cem_freeze_sha256"],
                "cem_contract_sha256": freeze["cem_contract_sha256"],
            }
        )
    if route_sets[0] & route_sets[1]:
        raise RuntimeError("B/C route overlap detected")

    absolute = np.concatenate(absolute_parts)
    cem_false = np.concatenate(false_parts)
    hybrid = np.concatenate(hybrid_parts)
    added = np.concatenate(added_parts)
    trajectory = np.concatenate(trajectory_parts)
    route = np.concatenate(route_parts)
    methods = {
        "safe_cem_false": method_metrics(
            cem_false,
            absolute,
            trajectory,
            route,
            replicates=args.bootstrap_replicates,
            seed=args.bootstrap_seed,
        ),
        "safe_fixed_plus_cem_addonly": method_metrics(
            hybrid,
            absolute,
            trajectory,
            route,
            replicates=args.bootstrap_replicates,
            seed=args.bootstrap_seed + 100,
        ),
    }
    reference = methods["safe_cem_false"]
    value = methods["safe_fixed_plus_cem_addonly"]
    relative = {
        "coverage_delta": float(
            value["promotion_fraction"] - reference["promotion_fraction"]
        ),
        "dual_delta": float(value["dual_precision"] - reference["dual_precision"]),
        "harm_delta": float(
            value["harm_rate_per_promotion"]
            - reference["harm_rate_per_promotion"]
        ),
        "absolute_delta_hybrid_minus_cem_false": {
            name: {
                "mean_m": float(
                    value["absolute"][name]["mean_m"]
                    - reference["absolute"][name]["mean_m"]
                ),
                "p90_m": float(
                    value["absolute"][name]["p90_m"]
                    - reference["absolute"][name]["p90_m"]
                ),
            }
            for name in NAMES
        },
    }
    quality_nondegraded = all(
        relative["absolute_delta_hybrid_minus_cem_false"][name][summary]
        <= 1.0e-12
        for name in NAMES
        for summary in ("mean_m", "p90_m")
    )
    marginal = added_subset_report(
        added=added,
        selected=hybrid,
        absolute=absolute,
    )
    direction = {}
    for cohort, report in zip(cohort_names, reports):
        delta = report["hybrid_relative_to_cem_false"]
        consistent = bool(
            delta["coverage_delta"] > 0.0
            and delta["dual_delta"] >= 0.0
            and delta["harm_delta"] <= 0.0
            and all(
                delta["absolute_delta_hybrid_minus_cem_false"][name][summary]
                <= 1.0e-12
                for name in NAMES
                for summary in ("mean_m", "p90_m")
            )
        )
        direction[cohort] = {
            "consistent_improvement_direction": consistent,
            "relative": delta,
            "cem_marginal_added_windows": report["cem_marginal_added_windows"],
        }
    consistent_count = sum(
        value["consistent_improvement_direction"] for value in direction.values()
    )
    checks = {
        "pooled_coverage_at_least_0.40": value["promotion_fraction"] >= 0.40,
        "pooled_dual_at_least_0.85": value["dual_precision"] >= 0.85,
        "pooled_harm_at_most_0.15": value["harm_rate_per_promotion"] <= 0.15,
        "all_pooled_mean_and_p90_nondegraded": quality_nondegraded,
        "two_route_disjoint_cohorts_consistent_direction": consistent_count >= 2,
    }
    promising = all(checks.values())
    checks["promising_development_result"] = promising

    result = {
        "schema": "recon_cem_safe_addonly_v56_pooled_development_result",
        "status": (
            "promising_development_result_frozen"
            if promising
            else "development_result_failed_frozen_rule"
        ),
        "evidence_role": "spent_B_C_development_not_blind_not_production",
        "cem": True,
        "state_gate_used": False,
        "fixed_pool_action_replacement_allowed": False,
        "attribution": (
            "Hybrid retains all safe fixed-pool promotions. Only the marginal "
            "added-window report is attributed to CEM."
        ),
        "cohorts": cohort_names,
        "states": int(len(hybrid)),
        "routes": int(len(np.unique(route))),
        "route_overlap": 0,
        "methods": methods,
        "hybrid_relative_to_cem_false": relative,
        "cem_marginal_added_windows": marginal,
        "cohort_direction": direction,
        "promising_development_checks": checks,
        "iteration_closure_logs": closure_logs,
        "preregistration": str(args.preregistration),
        "preregistration_sha256": file_sha256(args.preregistration),
        "artifact_bindings": audit_bindings,
        "reserve_accessed": False,
        "fresh_confirmation_started": False,
        "production_safe_claim_allowed": False,
        "next_action": (
            "freeze v56 as promising development; decide separately whether to "
            "allocate a truly fresh confirmation"
            if promising
            else "stop v56 without post-hoc B/C tuning"
        ),
    }
    write_json(args.output, result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "methods": methods,
                "hybrid_relative_to_cem_false": relative,
                "cem_marginal_added_windows": marginal,
                "cohort_direction": direction,
                "promising_development_checks": checks,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
