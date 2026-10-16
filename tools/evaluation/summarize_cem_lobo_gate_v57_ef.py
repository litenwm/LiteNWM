#!/usr/bin/env python3
"""Audit and summarize the one-time E/F v57 CEM development evaluation."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from tools.evaluation.build_recon100_k17_source_v1 import metric_labels  # noqa: E402
from tools.evaluation.build_recon_inner_k17_source_v2 import (  # noqa: E402
    local_ground_truth,
)
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
    parser.add_argument("--preoutcome-freeze", type=Path, required=True)
    parser.add_argument("--trajectory-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261057)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_rows(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def q_summary(mean: np.ndarray, covariance: np.ndarray) -> dict[str, object]:
    sign, logdet = np.linalg.slogdet(covariance.astype(np.float64))
    if not np.all(sign > 0):
        raise RuntimeError("proposal covariance is not positive definite")
    dimension = mean.shape[1]
    entropy = 0.5 * (
        dimension * (1.0 + math.log(2.0 * math.pi)) + logdet
    )
    return {
        "mean_vector": mean.mean(axis=0).astype(float).tolist(),
        "mean_l2_across_windows": float(np.linalg.norm(mean, axis=1).mean()),
        "variance_diagonal_mean_vector": np.diagonal(
            covariance, axis1=1, axis2=2
        ).mean(axis=0).astype(float).tolist(),
        "variance_diagonal_mean": float(
            np.diagonal(covariance, axis1=1, axis2=2).mean()
        ),
        "gaussian_entropy_mean": float(entropy.mean()),
    }


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    pre = read_json(args.preoutcome_freeze)
    if (
        pre.get("status")
        != "E_F_candidates_policy_and_selected_index_frozen_before_outcomes"
        or pre.get("E_F_labels_opened")
        or pre.get("E_F_outcomes_accessed")
        or pre.get("fixed_pool_action_replacement_allowed")
        or pre.get("state_gate_used")
        or pre.get("reserve_labels_opened")
        or pre.get("reserve_outcomes_accessed")
    ):
        raise RuntimeError("invalid E/F pre-outcome freeze")
    reserve_path = Path(str(pre["reserve_freeze"]))
    reserve = read_json(reserve_path)
    reserve_unchanged = (
        file_sha256(reserve_path) == pre["reserve_freeze_sha256"]
        and not reserve.get("labels_opened")
        and not reserve.get("outcomes_accessed")
    )
    if not reserve_unchanged:
        raise RuntimeError("sealed reserve changed or was opened")

    route_sets: dict[str, set[str]] = {}
    cohort_reports: dict[str, object] = {}
    flags = []
    pooled_absolute = []
    pooled_trajectory = []
    pooled_route = []
    pooled_false = []
    pooled_hybrid = []
    pooled_added = []
    pooled_iteration = {index: [] for index in range(3)}
    audit = {
        "elite_only_from_safe_mask": True,
        "safe_below_4_has_no_elite_padding": True,
        "safe_below_4_proposal_unchanged_observable_transitions": True,
        "fixed_safe_actions_never_replaced": True,
        "selected_index_matches_preoutcome_SHA": True,
        "prelabel_and_contract_match_preoutcome_SHA": True,
        "E_F_routes_disjoint": True,
        "reserve_still_sealed_and_SHA_unchanged": reserve_unchanged,
    }

    for bank in ("bank_e", "bank_f"):
        frozen = pre["cohorts"][bank]
        contract_path = Path(str(frozen["preutility_contract"]))
        selector_freeze_path = Path(str(frozen["addonly_freeze"]))
        selected_path = Path(str(frozen["selected_index"]))
        cem_selected_path = Path(str(frozen["cem_selected_actions"]))
        contract = read_json(contract_path)
        selector_freeze = read_json(selector_freeze_path)
        if (
            file_sha256(contract_path) != frozen["preutility_contract_sha256"]
            or file_sha256(selector_freeze_path)
            != frozen["addonly_freeze_sha256"]
            or file_sha256(selected_path) != frozen["selected_index_sha256"]
            or file_sha256(cem_selected_path)
            != frozen["cem_selected_actions_sha256"]
            or file_sha256(Path(str(frozen["prelabel_freeze"])))
            != frozen["prelabel_freeze_sha256"]
        ):
            audit["selected_index_matches_preoutcome_SHA"] = False
            raise RuntimeError(f"{bank} pre-outcome artifact changed")
        if (
            selector_freeze.get("fixed_pool_action_replacement_allowed")
            or selector_freeze.get("state_gate_used")
            or contract.get("state_gate_used")
            or contract.get("reserve_accessed")
        ):
            raise RuntimeError(f"{bank} selection contract changed")

        result_path = selector_freeze_path.parent / "result.json"
        result = read_json(result_path)
        outcomes_path = Path(str(result["outcomes"]))
        if file_sha256(outcomes_path) != result["outcomes_sha256"]:
            raise RuntimeError(f"{bank} outcome artifact changed")
        with np.load(outcomes_path, allow_pickle=False) as payload:
            absolute = payload["combined_absolute_metrics"].astype(np.float32)
            trajectory = payload["trajectory"].astype(str)
            route = payload["route_group"].astype(str)
            cem_false = payload["safe_cem_false_selected_index"].astype(
                np.int64
            )
            hybrid = payload["hybrid_combined_selected_index"].astype(np.int64)
            added = payload["cem_added_window"].astype(bool)
        with np.load(cem_selected_path, allow_pickle=False) as payload:
            final_local = payload[
                "universal_d_cem_true_selected_index"
            ].astype(np.int64)
            final_safe = payload["final_cem_safe"].astype(bool)
        if not np.array_equal(hybrid[cem_false != 0], cem_false[cem_false != 0]):
            audit["fixed_safe_actions_never_replaced"] = False
            raise RuntimeError(f"{bank} fixed-safe action was replaced")
        if not np.array_equal(added, (cem_false == 0) & (final_local != 0)):
            raise RuntimeError(f"{bank} add-only attribution changed")
        row_index = np.arange(len(cem_false))
        if not final_safe[row_index[final_local != 0], final_local[final_local != 0]].all():
            raise RuntimeError(f"{bank} final action is not candidate-gate safe")

        manifest_path = Path(str(contract["paired_comparison"]["manifest"]))
        rows = read_rows(manifest_path)
        ground_truth = np.stack(
            [
                local_ground_truth(
                    args.trajectory_root
                    / str(row["trajectory_id"])
                    / "traj_data.pkl",
                    int(row["current_t"]),
                )
                for row in rows
            ]
        ).astype(np.float32)
        routes = {str(row["route_id"]) for row in rows}
        route_sets[bank] = routes
        abstain = cem_false == 0
        oracle_any = np.zeros(len(rows), dtype=bool)
        valid_oracle_any = np.zeros(len(rows), dtype=bool)
        safe_oracle_any = np.zeros(len(rows), dtype=bool)
        update_any = np.zeros(len(rows), dtype=bool)
        iteration_reports = []
        proposal_cache = []
        update_cache = []
        final_dual = np.zeros(len(rows), dtype=bool)
        final_harm = np.zeros(len(rows), dtype=bool)

        run_dir = contract_path.parent
        for iteration in range(3):
            directory = run_dir / f"iteration_{iteration:02d}"
            with np.load(directory / "population.npz", allow_pickle=False) as payload:
                candidates = payload[
                    "candidates_feature_waypoint_units"
                ].astype(np.float32)
                proposal_mean = payload["proposal_mean"].astype(np.float32)
                proposal_covariance = payload["proposal_covariance"].astype(
                    np.float32
                )
            with np.load(directory / "frozen_scores.npz", allow_pickle=False) as payload:
                valid = payload["valid"].astype(bool)
                safe = payload["safe"].astype(bool)
                energy = payload["energy"].astype(np.float32)
                elite = payload["elite_index_for_update"].astype(np.int64)
                updated = payload["proposal_updated"].astype(bool)
                safe_count = payload["safe_count"].astype(np.int64)
            if candidates.shape[1] != 17:
                raise RuntimeError(f"{bank} iteration {iteration} population mismatch")
            if not np.array_equal(updated, safe_count >= 4):
                raise RuntimeError(f"{bank} iteration {iteration} update mismatch")
            for state in range(len(updated)):
                if updated[state] and not safe[state, elite[state]].all():
                    audit["elite_only_from_safe_mask"] = False
                    raise RuntimeError(f"{bank} iteration {iteration} unsafe elite")
                if not updated[state] and not np.all(elite[state] == -1):
                    audit["safe_below_4_has_no_elite_padding"] = False
                    raise RuntimeError(f"{bank} iteration {iteration} elite padding")
            proposal_cache.append((proposal_mean, proposal_covariance))
            update_cache.append(updated)

            candidate_absolute = metric_labels(
                candidates, ground_truth, 0.25, 0.4, 0.6
            )["values"]
            delta = candidate_absolute - candidate_absolute[:, :1]
            dual = (delta[..., 0] < -1.0e-7) & (
                delta[..., 3] < -1.0e-7
            )
            harm = (delta[..., 0] > 1.0e-7) | (
                delta[..., 3] > 1.0e-7
            )
            helpful = dual & ~harm
            helpful[:, 0] = False
            oracle_any |= helpful.any(axis=1)
            valid_oracle_any |= (helpful & valid).any(axis=1)
            safe_oracle_any |= (helpful & valid & safe).any(axis=1)
            update_any |= updated

            elite_state = np.repeat(np.flatnonzero(updated), 4)
            elite_candidate = elite[updated].reshape(-1)
            elite_delta = delta[elite_state, elite_candidate]
            predicted = -energy[elite_state, elite_candidate]
            true_utility = -0.5 * (elite_delta[:, 0] + elite_delta[:, 3])
            elite_dual = (elite_delta[:, 0] < -1.0e-7) & (
                elite_delta[:, 3] < -1.0e-7
            )
            elite_harm = (elite_delta[:, 0] > 1.0e-7) | (
                elite_delta[:, 3] > 1.0e-7
            )
            correlation = (
                float(np.corrcoef(predicted, true_utility)[0, 1])
                if predicted.std() > 0 and true_utility.std() > 0
                else None
            )
            iteration_reports.append(
                {
                    "iteration": iteration,
                    "population": 16,
                    "valid_candidates": int(valid[:, 1:].sum()),
                    "safe_candidates": int(safe[:, 1:].sum()),
                    "elite_candidates": int(len(elite_state)),
                    "updated_windows": int(updated.sum()),
                    "fallback_windows": int((~updated).sum()),
                    "support_acceptance": float(valid[:, 1:].mean()),
                    "ood_or_invalid_candidates": int((~valid[:, 1:]).sum()),
                    "proposal": q_summary(proposal_mean, proposal_covariance),
                    "predicted_utility_mean": float(predicted.mean()),
                    "true_utility_mean": float(true_utility.mean()),
                    "predicted_true_correlation": correlation,
                    "elite_dual": float(elite_dual.mean()),
                    "elite_harm": float(elite_harm.mean()),
                }
            )
            pooled_iteration[iteration].append(
                {
                    "predicted": predicted,
                    "true": true_utility,
                    "dual": elite_dual,
                    "harm": elite_harm,
                }
            )
            if iteration == 2:
                chosen_delta = delta[row_index, final_local]
                final_dual = (chosen_delta[:, 0] < -1.0e-7) & (
                    chosen_delta[:, 3] < -1.0e-7
                )
                final_harm = (chosen_delta[:, 0] > 1.0e-7) | (
                    chosen_delta[:, 3] > 1.0e-7
                )

        transition_audit = []
        for iteration in range(2):
            fallback = ~update_cache[iteration]
            mean_equal = np.array_equal(
                proposal_cache[iteration][0][fallback],
                proposal_cache[iteration + 1][0][fallback],
            )
            covariance_equal = np.array_equal(
                proposal_cache[iteration][1][fallback],
                proposal_cache[iteration + 1][1][fallback],
            )
            passed = bool(mean_equal and covariance_equal)
            audit[
                "safe_below_4_proposal_unchanged_observable_transitions"
            ] &= passed
            transition_audit.append(
                {
                    "from_iteration": iteration,
                    "to_iteration": iteration + 1,
                    "fallback_windows": int(fallback.sum()),
                    "mean_exactly_unchanged": bool(mean_equal),
                    "covariance_exactly_unchanged": bool(covariance_equal),
                    "passed": passed,
                }
            )

        opportunity = abstain & oracle_any
        valid_stage = opportunity & valid_oracle_any
        safe_stage = valid_stage & safe_oracle_any
        update_stage = safe_stage & update_any
        final_stage = update_stage & (final_local != 0)
        add = added
        funnel = {
            "fixed_pool_abstain": int(abstain.sum()),
            "sampled_true_dual_harmfree_opportunity": int(opportunity.sum()),
            "opportunity_support_valid": int(valid_stage.sum()),
            "opportunity_candidate_gate_safe": int(safe_stage.sum()),
            "opportunity_safe4_q_updated": int(update_stage.sum()),
            "opportunity_final_safe_candidate": int(final_stage.sum()),
            "addonly_promotions_all": int(add.sum()),
            "addonly_true_dual": int((add & final_dual).sum()),
            "addonly_true_harm": int((add & final_harm).sum()),
            "oracle_opportunity_capture_rate_addonly": float(
                add.sum() / opportunity.sum()
            ),
            "oracle_opportunity_capture_rate_true_dual": float(
                (add & final_dual).sum() / opportunity.sum()
            ),
        }
        flags.append(
            {
                "abstain": abstain,
                "opportunity": opportunity,
                "valid": valid_stage,
                "safe": safe_stage,
                "update": update_stage,
                "final": final_stage,
                "add": add,
                "add_dual": add & final_dual,
                "add_harm": add & final_harm,
            }
        )
        cohort_reports[bank] = {
            "states": len(rows),
            "routes": len(routes),
            "funnel": funnel,
            "iteration": iteration_reports,
            "fallback_proposal_transition_audit": transition_audit,
            "methods": result["methods"],
            "relative_to_cem_false": result[
                "hybrid_relative_to_cem_false"
            ],
            "cem_marginal_added_windows": result[
                "cem_marginal_added_windows"
            ],
            "selected_index_sha256": file_sha256(selected_path),
            "outcomes_sha256": file_sha256(outcomes_path),
        }
        pooled_absolute.append(absolute)
        pooled_trajectory.append(trajectory)
        pooled_route.append(route)
        pooled_false.append(cem_false)
        pooled_hybrid.append(hybrid)
        pooled_added.append(added)

    route_overlap = sorted(route_sets["bank_e"] & route_sets["bank_f"])
    audit["E_F_routes_disjoint"] = not route_overlap
    if route_overlap:
        raise RuntimeError(f"E/F route overlap: {route_overlap}")

    absolute = np.concatenate(pooled_absolute)
    trajectory = np.concatenate(pooled_trajectory)
    route = np.concatenate(pooled_route)
    cem_false = np.concatenate(pooled_false)
    hybrid = np.concatenate(pooled_hybrid)
    added = np.concatenate(pooled_added)
    methods_index = {
        "incumbent": np.zeros(len(hybrid), dtype=np.int64),
        "safe_cem_false": cem_false,
        "safe_fixed_plus_cem_addonly": hybrid,
    }
    methods = {
        name: method_metrics(
            selected,
            absolute,
            trajectory,
            route,
            replicates=args.bootstrap_replicates,
            seed=args.bootstrap_seed + 100 * index,
        )
        for index, (name, selected) in enumerate(methods_index.items())
    }
    reference = methods["safe_cem_false"]
    hybrid_metrics = methods["safe_fixed_plus_cem_addonly"]
    relative = {
        "coverage_delta": float(
            hybrid_metrics["promotion_fraction"]
            - reference["promotion_fraction"]
        ),
        "dual_delta": float(
            hybrid_metrics["dual_precision"] - reference["dual_precision"]
        ),
        "harm_delta": float(
            hybrid_metrics["harm_rate_per_promotion"]
            - reference["harm_rate_per_promotion"]
        ),
        "absolute_delta_hybrid_minus_cem_false": {
            name: {
                summary: float(
                    hybrid_metrics["absolute"][name][summary]
                    - reference["absolute"][name][summary]
                )
                for summary in ("mean_m", "p90_m")
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
        added=added, selected=hybrid, absolute=absolute
    )

    pooled_funnel = {
        key: int(sum(item[key].sum() for item in flags))
        for key in (
            "abstain",
            "opportunity",
            "valid",
            "safe",
            "update",
            "final",
            "add",
            "add_dual",
            "add_harm",
        )
    }
    pooled_funnel["oracle_opportunity_capture_rate_addonly"] = float(
        pooled_funnel["add"] / pooled_funnel["opportunity"]
    )
    pooled_funnel["oracle_opportunity_capture_rate_true_dual"] = float(
        pooled_funnel["add_dual"] / pooled_funnel["opportunity"]
    )
    losses = {
        "proposal_no_oracle": pooled_funnel["abstain"]
        - pooled_funnel["opportunity"],
        "support_rejected_oracle": pooled_funnel["opportunity"]
        - pooled_funnel["valid"],
        "candidate_gate_rejected_valid_oracle": pooled_funnel["valid"]
        - pooled_funnel["safe"],
        "safe4_update_loss": pooled_funnel["safe"]
        - pooled_funnel["update"],
        "final_selector_loss_after_update": pooled_funnel["update"]
        - pooled_funnel["final"],
    }

    optimizer = []
    for iteration in range(3):
        predicted = np.concatenate(
            [item["predicted"] for item in pooled_iteration[iteration]]
        )
        true = np.concatenate(
            [item["true"] for item in pooled_iteration[iteration]]
        )
        dual = np.concatenate(
            [item["dual"] for item in pooled_iteration[iteration]]
        )
        harm = np.concatenate(
            [item["harm"] for item in pooled_iteration[iteration]]
        )
        optimizer.append(
            {
                "iteration": iteration,
                "elite_candidates": len(predicted),
                "predicted_utility_mean": float(predicted.mean()),
                "true_utility_mean": float(true.mean()),
                "predicted_true_correlation": float(
                    np.corrcoef(predicted, true)[0, 1]
                ),
                "elite_dual": float(dual.mean()),
                "elite_harm": float(harm.mean()),
            }
        )
    x = np.arange(3, dtype=np.float64)
    predicted_values = np.asarray(
        [item["predicted_utility_mean"] for item in optimizer]
    )
    true_values = np.asarray([item["true_utility_mean"] for item in optimizer])
    predicted_slope = float(np.polyfit(x, predicted_values, 1)[0])
    true_slope = float(np.polyfit(x, true_values, 1)[0])
    predicted_gain = float(predicted_values[-1] - predicted_values[0])
    true_gain = float(true_values[-1] - true_values[0])
    exploitation = {
        "iteration": optimizer,
        "predicted_utility_slope_per_iteration": predicted_slope,
        "true_utility_slope_per_iteration": true_slope,
        "slope_ratio_predicted_over_true": (
            float(predicted_slope / true_slope)
            if abs(true_slope) > 1.0e-12
            else None
        ),
        "predicted_utility_gain_iteration0_to_2": predicted_gain,
        "true_utility_gain_iteration0_to_2": true_gain,
        "gain_ratio_predicted_over_true": (
            float(predicted_gain / true_gain)
            if abs(true_gain) > 1.0e-12
            else None
        ),
        "optimizer_exploitation_confirmed_on_E_F": bool(
            predicted_gain > 0.20 and true_gain < 0.01
        ),
    }

    cohort_consistency = all(
        cohort_reports[bank]["relative_to_cem_false"]["coverage_delta"] > 0
        and cohort_reports[bank]["cem_marginal_added_windows"]["dual"] >= 0.85
        and cohort_reports[bank]["cem_marginal_added_windows"]["harm"] <= 0.15
        and cohort_reports[bank]["promising_development_checks"][
            "all_mean_and_p90_nondegraded"
        ]
        if "promising_development_checks" in cohort_reports[bank]
        else False
        for bank in ("bank_e", "bank_f")
    )
    # The source result keeps the quality check outside the copied cohort report.
    cohort_quality = {
        bank: read_json(
            Path(str(pre["cohorts"][bank]["addonly_freeze"])).parent
            / "result.json"
        )["promising_development_checks"]["all_mean_and_p90_nondegraded"]
        for bank in ("bank_e", "bank_f")
    }
    cohort_consistency = all(
        cohort_reports[bank]["relative_to_cem_false"]["coverage_delta"] > 0
        and cohort_reports[bank]["cem_marginal_added_windows"]["dual"] >= 0.85
        and cohort_reports[bank]["cem_marginal_added_windows"]["harm"] <= 0.15
        and cohort_quality[bank]
        for bank in ("bank_e", "bank_f")
    )

    rule = pre["substantial_gain_rule"]
    checks = {
        "added_windows_at_least_20": pooled_funnel["add"]
        >= int(rule["added_windows_min"]),
        "coverage_gain_at_least_0.10": relative["coverage_delta"]
        >= float(rule["coverage_absolute_gain_min"]),
        "pooled_dual_at_least_0.85": hybrid_metrics["dual_precision"]
        >= float(rule["pooled_dual_min"]),
        "pooled_harm_at_most_0.15": hybrid_metrics[
            "harm_rate_per_promotion"
        ]
        <= float(rule["pooled_harm_max"]),
        "all_four_mean_and_p90_nondegraded": quality_nondegraded,
        "two_cohorts_consistent_direction": cohort_consistency,
    }
    checks["substantial_gain_pass"] = all(checks.values())
    audit["all_checks_passed"] = all(audit.values())

    report = {
        "schema": "recon_cem_lobo_gate_v57_EF_one_time_result",
        "status": (
            "substantial_promising_development_result"
            if checks["substantial_gain_pass"]
            else "frozen_one_time_EF_development_failed_substantial_gain_rule"
        ),
        "evidence_role": "spent_development_not_blind_not_production_safe",
        "preoutcome_freeze": str(args.preoutcome_freeze),
        "preoutcome_freeze_sha256": file_sha256(args.preoutcome_freeze),
        "candidate_gate": pre["candidate_gate"],
        "formula": pre["formula"],
        "cohorts": cohort_reports,
        "cohort_all_mean_p90_nondegraded": cohort_quality,
        "pooled_methods": methods,
        "pooled_relative_to_cem_false": relative,
        "pooled_cem_marginal_added_windows": marginal,
        "pooled_funnel": pooled_funnel,
        "funnel_losses": losses,
        "dominant_observed_loss": max(losses, key=losses.get),
        "predicted_vs_true_utility": exploitation,
        "substantial_gain_checks": checks,
        "automatic_acceptance_audit": audit,
        "post_E_F_tuning_allowed": False,
        "reserve_accessed": False,
        "reserve_still_sealed": True,
        "production_safe_claim_allowed": False,
    }
    write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
