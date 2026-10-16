#!/usr/bin/env python3
"""Evaluate frozen v70 CEM depths 3/4/5 on spent pilot outcomes only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from tools.evaluation.build_recon100_k17_source_v1 import metric_labels  # noqa: E402
from tools.evaluation.evaluate_dualdomain_cem_tail_v66_replay import (  # noqa: E402
    METRICS,
    decision_metrics,
)
from tools.evaluation.export_recon_dense_dinov3_features_v1 import (  # noqa: E402
    load_npz,
    write_json_atomic,
    write_npz_atomic,
)
from tools.planning.run_cem_safe_elite_v55 import select_lowest_energy_safe  # noqa: E402
from tools.scoring.train_recon_dense_meta_dinov3_v1 import file_sha256  # noqa: E402


TOLERANCE = 1.0e-7


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--recon-continuation-dir", type=Path, required=True)
    parser.add_argument("--scand-continuation-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def average_ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
        start = stop
    return ranks


def spearman(left: np.ndarray, right: np.ndarray) -> float | None:
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    finite = np.isfinite(left) & np.isfinite(right)
    if finite.sum() < 3:
        return None
    left_rank = average_ranks(left[finite])
    right_rank = average_ranks(right[finite])
    if left_rank.std() <= 0.0 or right_rank.std() <= 0.0:
        return None
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def compare_depth(challenger: dict[str, Any], depth3: dict[str, Any]) -> dict[str, Any]:
    mean_delta = {
        name: challenger["absolute_mean_m"][name] - depth3["absolute_mean_m"][name]
        for name in METRICS
    }
    p90_delta = {
        name: challenger["absolute_p90_m"][name] - depth3["absolute_p90_m"][name]
        for name in METRICS
    }
    all_mean_nondegraded = all(value <= TOLERANCE for value in mean_delta.values())
    all_p90_nondegraded = all(value <= TOLERANCE for value in p90_delta.values())
    return {
        "coverage_delta": challenger["coverage"] - depth3["coverage"],
        "dual_precision_delta": (
            challenger["dual_precision"] - depth3["dual_precision"]
            if challenger["dual_precision"] is not None and depth3["dual_precision"] is not None
            else None
        ),
        "harm_rate_delta": (
            challenger["harm_rate"] - depth3["harm_rate"]
            if challenger["harm_rate"] is not None and depth3["harm_rate"] is not None
            else None
        ),
        "absolute_mean_delta_m": mean_delta,
        "absolute_p90_delta_m": p90_delta,
        "mean_delta_sum_m": float(sum(mean_delta.values())),
        "p90_delta_sum_m": float(sum(p90_delta.values())),
        "all_four_mean_nondegraded": all_mean_nondegraded,
        "all_four_p90_nondegraded": all_p90_nondegraded,
        "stable_four_metric_improvement": (
            all_mean_nondegraded
            and all_p90_nondegraded
            and sum(mean_delta.values()) < -TOLERANCE
            and sum(p90_delta.values()) < -TOLERANCE
        ),
    }


def round_diagnostic(
    population_path: Path,
    score_path: Path,
    ground_truth: np.ndarray,
) -> dict[str, Any]:
    population = load_npz(population_path)
    scores = load_npz(score_path)
    candidates = population["candidates_feature_waypoint_units"].astype(np.float32)
    energy = scores["energy"].astype(np.float32)
    safe = scores["safe"].astype(bool)
    valid = scores["valid"].astype(bool)
    labels = metric_labels(candidates, ground_truth, 0.25, 0.4, 0.6)
    delta = labels["targets"][:, :, :4]
    true_objective_gain = -delta[:, :, 3]
    scorer_preference = -energy
    challenger = np.ones_like(safe, dtype=bool)
    challenger[:, 0] = False
    valid_mask = valid & challenger & np.isfinite(energy)
    safe_mask = safe & challenger & np.isfinite(energy)
    dual = (delta[:, :, 0] < -TOLERANCE) & (delta[:, :, 3] < -TOLERANCE)
    harm = (delta[:, :, 0] > TOLERANCE) | (delta[:, :, 3] > TOLERANCE)
    selected = select_lowest_energy_safe(energy, safe)
    row = np.arange(len(selected))
    promoted = selected != 0
    selected_delta = delta[row, selected]
    selected_dual = promoted & (selected_delta[:, 0] < -TOLERANCE) & (
        selected_delta[:, 3] < -TOLERANCE
    )
    selected_harm = promoted & (
        (selected_delta[:, 0] > TOLERANCE) | (selected_delta[:, 3] > TOLERANCE)
    )
    promotions = int(promoted.sum())
    return {
        "population_sha256": file_sha256(population_path),
        "frozen_scores_sha256": file_sha256(score_path),
        "valid_challengers": int(valid_mask.sum()),
        "safe_challengers": int(safe_mask.sum()),
        "safe_candidate_dual_precision": (
            float(dual[safe_mask].mean()) if safe_mask.any() else None
        ),
        "safe_candidate_harm_rate": (
            float(harm[safe_mask].mean()) if safe_mask.any() else None
        ),
        "scorer_true_gain_spearman_valid": spearman(
            scorer_preference[valid_mask], true_objective_gain[valid_mask]
        ),
        "scorer_true_gain_spearman_safe": spearman(
            scorer_preference[safe_mask], true_objective_gain[safe_mask]
        ),
        "safe_candidate_true_objective_gain_mean_m": (
            float(true_objective_gain[safe_mask].mean()) if safe_mask.any() else None
        ),
        "round_local_promotions": promotions,
        "round_local_dual_precision": (
            float(selected_dual.sum() / promotions) if promotions else None
        ),
        "round_local_harm_rate": (
            float(selected_harm.sum() / promotions) if promotions else None
        ),
        "round_local_true_objective_gain_mean_m": (
            float((-selected_delta[promoted, 3]).mean()) if promotions else None
        ),
        "round_local_scorer_energy_mean": (
            float(energy[row[promoted], selected[promoted]].mean()) if promotions else None
        ),
    }


def evaluate_cohort(
    cohort: str,
    prereg: dict[str, Any],
    continuation_dir: Path,
) -> dict[str, Any]:
    binding = prereg["cohorts"][cohort]
    continuation_path = continuation_dir / "continuation_freeze.json"
    continuation = read_json(continuation_path)
    if continuation.get("status") != "depth4_and_depth5_actions_frozen_before_spent_outcome_diagnostics":
        raise RuntimeError(f"continuation is not frozen: {continuation_path}")
    if continuation.get("cohort") != cohort:
        raise RuntimeError("continuation cohort mismatch")
    if continuation.get("labels_opened") or continuation.get("outcomes_used_by_cem"):
        raise RuntimeError("continuation reports outcome use by CEM")
    if continuation.get("reserve_accessed") or continuation.get("go2_accessed"):
        raise RuntimeError("continuation reports forbidden reserve/Go2 access")
    if continuation.get("cached_ltwm_or_scorer_rounds_recomputed") != 0:
        raise RuntimeError("cached first three rounds were recomputed")
    if continuation.get("preregistration_sha256") != file_sha256(Path(prereg["path"])):
        raise RuntimeError("continuation is bound to a different preregistration")

    outcome_path = Path(binding["spent_ground_truth_source"])
    if file_sha256(outcome_path) != binding["spent_ground_truth_source_sha256"]:
        raise RuntimeError(f"spent ground-truth source changed: {outcome_path}")
    outcome = load_npz(outcome_path)
    ground_truth = outcome["ground_truth_local_m"].astype(np.float32)
    depth3_path = Path(binding["depth3_selected_actions"])
    depth_paths = {
        3: depth3_path,
        4: Path(continuation["depth_artifacts"]["4"]["selected_actions"]),
        5: Path(continuation["depth_artifacts"]["5"]["selected_actions"]),
    }
    record = None
    summaries: dict[str, Any] = {}
    selection_arrays: dict[str, np.ndarray] = {}
    archive_diagnostics: dict[str, Any] = {}
    for depth, path in depth_paths.items():
        arrays = load_npz(path)
        local_record = arrays["record_key"].astype(str)
        if record is None:
            record = local_record
        elif not np.array_equal(local_record, record):
            raise RuntimeError("depth record identity mismatch")
        if not np.array_equal(outcome["record_key"].astype(str), local_record):
            raise RuntimeError("spent outcome/selection identity mismatch")
        candidates = arrays["final_cem_feature_waypoint_units"].astype(np.float32)
        selected = arrays["final_cem_selected_index"].astype(np.int64)
        summary, local_arrays = decision_metrics(candidates, selected, ground_truth)
        summaries[str(depth)] = summary
        for key, value in local_arrays.items():
            selection_arrays[f"depth_{depth}__{key}"] = value
        row = np.arange(len(selected))
        promoted = selected != 0
        archive_energy = arrays["archive_energy"].astype(np.float32)
        archive_diagnostics[str(depth)] = {
            "selected_actions": str(path.resolve()),
            "selected_actions_sha256": file_sha256(path),
            "archive_energy_mean": (
                float(archive_energy[promoted].mean()) if promoted.any() else None
            ),
            "archive_source_counts": {
                "none": int((arrays["archive_source_round"] == -2).sum()),
                "fixed_pool": int((arrays["archive_source_round"] == -1).sum()),
                **{
                    f"iteration_{iteration:02d}": int(
                        (arrays["archive_source_round"] == iteration).sum()
                    )
                    for iteration in range(depth)
                },
            },
            "selected_true_objective_gain_mean_m": (
                float((-local_arrays["delta_metrics"][promoted, 3]).mean())
                if promoted.any()
                else None
            ),
        }

    comparisons = {
        str(depth): compare_depth(summaries[str(depth)], summaries["3"])
        for depth in (4, 5)
    }
    base_replay_dir = Path(continuation["depth3_freeze"]).parent
    per_iteration = {}
    for iteration in range(5):
        directory = (
            base_replay_dir / f"iteration_{iteration:02d}"
            if iteration < 3
            else continuation_dir / f"iteration_{iteration:02d}"
        )
        per_iteration[str(iteration)] = round_diagnostic(
            directory / "population.npz",
            directory / "frozen_scores.npz",
            ground_truth,
        )
        if iteration >= 3:
            per_iteration[str(iteration)]["outcome_free_cem_audit"] = next(
                value
                for value in continuation["added_iteration_audit"]
                if value["iteration"] == iteration
            )

    exploitation = {}
    saturation = {}
    for depth in (4, 5):
        previous = depth - 1
        energy_before = archive_diagnostics[str(previous)]["archive_energy_mean"]
        energy_after = archive_diagnostics[str(depth)]["archive_energy_mean"]
        objective_delta = (
            summaries[str(depth)]["absolute_mean_m"]["objective"]
            - summaries[str(previous)]["absolute_mean_m"]["objective"]
        )
        harm_before = summaries[str(previous)]["harm_rate"]
        harm_after = summaries[str(depth)]["harm_rate"]
        scorer_energy_improved = (
            energy_before is not None
            and energy_after is not None
            and energy_after < energy_before - TOLERANCE
        )
        exploitation[str(depth)] = {
            "archive_scorer_energy_improved": scorer_energy_improved,
            "true_objective_mean_delta_m_vs_previous_depth": objective_delta,
            "harm_rate_delta_vs_previous_depth": (
                harm_after - harm_before
                if harm_after is not None and harm_before is not None
                else None
            ),
            "scorer_exploitation_flag": bool(
                scorer_energy_improved
                and (
                    objective_delta > TOLERANCE
                    or (
                        harm_after is not None
                        and harm_before is not None
                        and harm_after - harm_before >= 0.02 - TOLERANCE
                    )
                )
            ),
        }
        comparison = comparisons[str(depth)]
        saturation[str(depth)] = {
            "coverage_gain_below_2pp": comparison["coverage_delta"] < 0.02 - TOLERANCE,
            "no_stable_four_metric_improvement": not comparison[
                "stable_four_metric_improvement"
            ],
            "benefit_saturation_flag": bool(
                comparison["coverage_delta"] < 0.02 - TOLERANCE
                and not comparison["stable_four_metric_improvement"]
            ),
        }
    return {
        "domain": continuation["domain"],
        "cohort": cohort,
        "states": continuation["states"],
        "depth_metrics": summaries,
        "comparisons_to_depth3": comparisons,
        "archive_diagnostics": archive_diagnostics,
        "per_iteration_candidate_diagnostics": per_iteration,
        "scorer_exploitation": exploitation,
        "benefit_saturation": saturation,
        "continuation_freeze": str(continuation_path.resolve()),
        "continuation_freeze_sha256": file_sha256(continuation_path),
        "spent_ground_truth_source": str(outcome_path.resolve()),
        "spent_ground_truth_source_sha256": file_sha256(outcome_path),
        "selection_arrays": selection_arrays,
    }


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(args.output_dir)
    prereg = read_json(args.preregistration)
    if prereg.get("status") != "frozen_before_depth4_or_depth5_candidate_evaluation":
        raise RuntimeError("depth pilot preregistration is not active")
    if prereg["source_bindings"]["evaluator"]["sha256"] != file_sha256(Path(__file__)):
        raise RuntimeError("depth pilot evaluator changed after preregistration")
    if prereg.get("reserve_may_be_used") or prereg.get("go2_may_be_used"):
        raise RuntimeError("preregistration permits a forbidden evidence source")
    prereg["path"] = str(args.preregistration.resolve())

    cohorts = {
        "recon_bank_g": evaluate_cohort(
            "recon_bank_g", prereg, args.recon_continuation_dir
        ),
        "scand_fresh_outer": evaluate_cohort(
            "scand_fresh_outer", prereg, args.scand_continuation_dir
        ),
    }
    variant_checks = {}
    for depth in (4, 5):
        recon_metrics = cohorts["recon_bank_g"]["depth_metrics"][str(depth)]
        scand_metrics = cohorts["scand_fresh_outer"]["depth_metrics"][str(depth)]
        recon_comparison = cohorts["recon_bank_g"]["comparisons_to_depth3"][str(depth)]
        scand_comparison = cohorts["scand_fresh_outer"]["comparisons_to_depth3"][str(depth)]
        checks = {
            "recon_dual_at_least_0_85": recon_metrics["dual_precision"] is not None
            and recon_metrics["dual_precision"] >= 0.85,
            "recon_harm_at_most_0_15": recon_metrics["harm_rate"] is not None
            and recon_metrics["harm_rate"] <= 0.15,
            "scand_dual_at_least_0_85": scand_metrics["dual_precision"] is not None
            and scand_metrics["dual_precision"] >= 0.85,
            "scand_harm_at_most_0_15": scand_metrics["harm_rate"] is not None
            and scand_metrics["harm_rate"] <= 0.15,
            "recon_all_four_mean_nondegraded": recon_comparison[
                "all_four_mean_nondegraded"
            ],
            "recon_all_four_p90_nondegraded": recon_comparison[
                "all_four_p90_nondegraded"
            ],
            "both_domains_coverage_gain_at_least_2pp": (
                recon_comparison["coverage_delta"] >= 0.02 - TOLERANCE
                and scand_comparison["coverage_delta"] >= 0.02 - TOLERANCE
            ),
            "both_domains_stable_four_metric_improvement": (
                recon_comparison["stable_four_metric_improvement"]
                and scand_comparison["stable_four_metric_improvement"]
            ),
        }
        checks["two_domain_improvement_trigger"] = bool(
            checks["both_domains_coverage_gain_at_least_2pp"]
            or checks["both_domains_stable_four_metric_improvement"]
        )
        checks["qualifies_for_A_to_H_expansion"] = all(
            value
            for key, value in checks.items()
            if key
            not in (
                "both_domains_coverage_gain_at_least_2pp",
                "both_domains_stable_four_metric_improvement",
            )
        )
        variant_checks[str(depth)] = checks

    qualifying = [
        depth
        for depth in (4, 5)
        if variant_checks[str(depth)]["qualifies_for_A_to_H_expansion"]
    ]
    decision = {
        "qualifying_depths": qualifying,
        "selected_depth_for_possible_A_to_H_expansion": min(qualifying) if qualifying else 3,
        "expand_to_A_to_H": bool(qualifying),
        "retain_formal_v70_depth3": not bool(qualifying),
        "selection_rule": "choose shallowest qualifying depth; otherwise retain frozen depth 3",
        "independent_generalization_claim_allowed": False,
    }

    args.output_dir.mkdir(parents=True, exist_ok=False)
    selected_arrays = {}
    for cohort, value in cohorts.items():
        for key, array in value.pop("selection_arrays").items():
            selected_arrays[f"{cohort}__{key}"] = array
    selected_metrics_path = args.output_dir / "selected_metrics.npz"
    write_npz_atomic(selected_metrics_path, **selected_arrays)
    result = {
        "schema": "v70_cem_depth_3_4_5_spent_pilot_result",
        "status": "spent_pilot_evaluated_without_model_or_threshold_changes",
        "cohorts": cohorts,
        "variant_checks": variant_checks,
        "pilot_decision": decision,
        "preregistration": str(args.preregistration.resolve()),
        "preregistration_sha256": file_sha256(args.preregistration),
        "selected_metrics": str(selected_metrics_path.resolve()),
        "selected_metrics_sha256": file_sha256(selected_metrics_path),
        "only_cem_iterations_varied": True,
        "cached_first_three_rounds_recomputed": False,
        "reserve_accessed": False,
        "reserve_used_for_iteration_selection": False,
        "go2_accessed": False,
        "model_or_thresholds_changed": False,
        "new_independent_cohort_required_for_confirmation": True,
        "production_safe_claim_allowed": False,
    }
    result_path = args.output_dir / "result.json"
    write_json_atomic(result_path, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
