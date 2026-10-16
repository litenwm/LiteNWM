#!/usr/bin/env python3
"""Correct NaN-sensitive representation evidence in the spent SCAND diagnostic.

This is a read-only supplemental audit. It does not retrain a selector, revisit a
cohort, or alter the preregistered usefulness decision.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.evaluation import run_scand_bottleneck_diagnostic_v1 as diagnostic  # noqa: E402
from tools.evaluation import run_v72_proposal_tail_reranker as v72  # noqa: E402
from tools.evaluation import run_v75_structural_selective_classifier as v75  # noqa: E402


OUTPUT = diagnostic.OUTPUT
RESULT = OUTPUT / "result.json"
DERIVED = OUTPUT / "derived_predictions.npz"
DESTINATION = OUTPUT / "audit_correction.json"


def finite_mean(values: np.ndarray) -> float | None:
    value = np.asarray(values, dtype=np.float64)
    value = value[np.isfinite(value)]
    return float(value.mean()) if len(value) else None


def representations(
    data: dict[str, Any], feature_names: list[str], fallback: str
) -> tuple[np.ndarray, int]:
    keep = [
        index
        for index, name in enumerate(feature_names)
        if name not in {"domain_recon", "domain_scand", "formal_selected"}
    ]
    feature = data["v75_relative"][..., keep].astype(np.float64)
    output = np.zeros((len(feature), 2 * len(keep)), dtype=np.float64)
    empty = 0
    for state in range(len(feature)):
        formal = int(data["formal_index"][state])
        eligible = data["valid"][state] & data["safe"][state]
        eligible[formal] = False
        if not eligible.any():
            empty += 1
            if fallback == "all_valid":
                eligible = data["valid"][state].copy()
                eligible[formal] = False
            if not eligible.any():
                eligible[formal] = True
        value = feature[state, eligible]
        output[state] = np.concatenate((value.mean(axis=0), value.std(axis=0)))
    if not np.isfinite(output).all():
        raise RuntimeError("corrected state representation remains non-finite")
    return output, empty


def standardized(
    train: np.ndarray, test: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    mean = train.mean(axis=0)
    scale = train.std(axis=0)
    active = scale >= 1.0e-6
    return (train[:, active] - mean[active]) / scale[active], (
        test[:, active] - mean[active]
    ) / scale[active]


def representation_audit(
    data: dict[str, Any], feature_names: list[str], prediction: np.ndarray, fallback: str
) -> dict[str, Any]:
    representation, empty = representations(data, feature_names, fallback)
    domain = data["domain"].astype(str)
    group = data["group"].astype(str)
    scand_states = np.flatnonzero(domain == "scand")

    correct = 0
    for state in scand_states:
        train = scand_states[scand_states != state]
        x_train, x_test = standardized(representation[train], representation[[state]])
        centroids = {
            route: x_train[group[train] == route].mean(axis=0)
            for route in np.unique(group[train])
        }
        selected = min(
            centroids,
            key=lambda route: float(np.linalg.norm(x_test[0] - centroids[route])),
        )
        correct += int(selected == group[state])

    domain_target = (domain == "scand").astype(np.int64)
    domain_score = np.full(len(domain), np.nan, dtype=np.float64)
    distance = np.full(len(domain), np.nan, dtype=np.float64)
    for outer in range(5):
        train = np.flatnonzero(data["fold"] != outer)
        test = np.flatnonzero(data["fold"] == outer)
        x_train, x_test = standardized(representation[train], representation[test])
        x_train = np.concatenate((x_train, np.ones((len(x_train), 1))), axis=1)
        x_test = np.concatenate((x_test, np.ones((len(x_test), 1))), axis=1)
        matrix = x_train.T @ x_train
        matrix.flat[:: len(matrix) + 1] += 20.0
        coefficient = np.linalg.solve(matrix, x_train.T @ domain_target[train])
        domain_score[test] = x_test @ coefficient

        scand_train = np.flatnonzero((domain == "scand") & (data["fold"] != outer))
        scand_test = np.flatnonzero((domain == "scand") & (data["fold"] == outer))
        x_train, x_test = standardized(
            representation[scand_train], representation[scand_test]
        )
        centroids = np.stack(
            [
                x_train[group[scand_train] == route].mean(axis=0)
                for route in np.unique(group[scand_train])
            ]
        )
        distance[scand_test] = np.min(
            np.sqrt(np.sum((x_test[:, None] - centroids[None]) ** 2, axis=-1)),
            axis=1,
        )

    state_error = np.full(len(domain), np.nan, dtype=np.float64)
    eligible_states = 0
    for state in scand_states:
        mask = diagnostic.eligible_mask(data, np.asarray([state]))[0]
        if mask.any():
            eligible_states += 1
            state_error[state] = float(
                np.mean(np.abs(prediction[state, mask] - data["gain"][state, mask, 3]))
            )

    route_distance = []
    route_error = []
    routes = []
    for route in np.unique(group[scand_states]):
        local = scand_states[group[scand_states] == route]
        error = finite_mean(state_error[local])
        if error is not None:
            routes.append(route)
            route_distance.append(float(np.mean(distance[local])))
            route_error.append(error)
    route_distance_value = np.asarray(route_distance, dtype=np.float64)
    route_error_value = np.asarray(route_error, dtype=np.float64)
    cutoff = float(np.quantile(route_distance_value, 0.75))
    high = route_distance_value >= cutoff
    return {
        "fallback_for_no_eligible_challenger": fallback,
        "states_without_eligible_challenger": int(empty),
        "scand_states_with_objective_error": int(eligible_states),
        "route_identification_accuracy_leave_one_window_out": float(
            correct / len(scand_states)
        ),
        "route_identification_chance": 1.0 / len(np.unique(group[scand_states])),
        "route_identification_multiple_of_chance": float(
            correct / len(scand_states) * len(np.unique(group[scand_states]))
        ),
        "domain_identification_route_oof_auroc": diagnostic.auc_score(
            domain_score, domain_target
        ),
        "support_distance": {
            "routes_with_finite_error": int(len(routes)),
            "route_distance_error_pearson": diagnostic.pearson(
                route_distance_value, route_error_value
            ),
            "top_quartile_distance_cutoff": cutoff,
            "top_quartile_mean_objective_mae_m": finite_mean(route_error_value[high]),
            "lower_three_quartiles_mean_objective_mae_m": finite_mean(
                route_error_value[~high]
            ),
            "routes": routes,
            "route_distances": route_distance,
            "route_objective_mae_m": route_error,
        },
    }


def duplicate_boundary_audit(data: dict[str, Any]) -> dict[str, Any]:
    disagreement_margins = []
    disagreement_objective_difference = []
    pairs = 0
    disagreement = 0
    scand_states = np.flatnonzero(data["domain"].astype(str) == "scand")
    for state in scand_states:
        index = np.flatnonzero(data["valid"][state])
        waypoint = data["candidates"][state, index].astype(np.float64) * 0.25
        distance = np.sqrt(
            np.mean(
                np.sum((waypoint[:, None] - waypoint[None]) ** 2, axis=-1),
                axis=-1,
            )
        )
        left, right = np.triu_indices(len(index), 1)
        for a, b in zip(left[distance[left, right] <= 0.005], right[distance[left, right] <= 0.005]):
            first, second = index[a], index[b]
            pairs += 1
            changed = (
                bool(data["dual_vs_formal"][state, first])
                != bool(data["dual_vs_formal"][state, second])
                or bool(data["harm_vs_formal"][state, first])
                != bool(data["harm_vs_formal"][state, second])
            )
            if changed:
                disagreement += 1
                gain = data["gain"][state, [first, second]][:, [0, 3]].astype(np.float64)
                disagreement_margins.append(float(np.min(np.abs(gain))))
                disagreement_objective_difference.append(
                    abs(
                        float(data["true_metrics"][state, first, 3])
                        - float(data["true_metrics"][state, second, 3])
                    )
                )
    margin = np.asarray(disagreement_margins, dtype=np.float64)
    difference = np.asarray(disagreement_objective_difference, dtype=np.float64)
    return {
        "candidate_distance_threshold_m": 0.005,
        "pairs": int(pairs),
        "pairs_with_any_dual_or_harm_label_disagreement": int(disagreement),
        "any_label_disagreement_rate": float(disagreement / pairs),
        "minimum_abs_ate_or_objective_gain_margin_median_m": float(np.median(margin)),
        "minimum_abs_ate_or_objective_gain_margin_p90_m": float(np.quantile(margin, 0.9)),
        "disagreements_within_1mm_of_zero_boundary_rate": float(np.mean(margin <= 0.001)),
        "disagreements_within_5mm_of_zero_boundary_rate": float(np.mean(margin <= 0.005)),
        "objective_difference_mean_m": float(difference.mean()),
        "objective_difference_p90_m": float(np.quantile(difference, 0.9)),
        "interpretation": "Label flips are assessed against the exact zero-gain dual/harm boundary; this distinguishes boundary sensitivity from record-key or metric-definition mismatch.",
    }


def main() -> None:
    if DESTINATION.exists():
        raise FileExistsError(DESTINATION)
    base = v72.read_json(RESULT)
    data, data_audit = v75.load_data()
    derived = v72.load_npz(DERIVED)
    names = data_audit["v75_feature_audit"]["relative_feature_names"]
    prediction = derived["strict_scand_only_objective_gain_prediction"]
    corrected = {
        "schema": "scand_bottleneck_diagnostic_audit_correction_v1",
        "scope": "read_only_supplemental_audit_no_model_training",
        "reason": "Original representation aggregation took mean/std over an empty eligible set for some states, propagating NaNs and collapsing support distances.",
        "base_result_sha256": v72.sha256_file(RESULT),
        "derived_predictions_sha256": v72.sha256_file(DERIVED),
        "preregistration_sha256": v72.sha256_file(diagnostic.PREREG),
        "representation_sensitivity": {
            fallback: representation_audit(data, names, prediction, fallback)
            for fallback in ("formal", "all_valid")
        },
        "near_duplicate_boundary_audit": duplicate_boundary_audit(data),
        "time_alignment_evidence": {
            "manifest_has_current_t_goal_t_history_timesteps": True,
            "manifest_rule_observed": "history=[current_t-3..current_t], goal_t=current_t+4",
            "cached_source_fields_include_explicit_timestamps": False,
            "end_to_end_action_ground_truth_alignment_independently_reverified_here": False,
            "interpretation": "The manifest contract is internally explicit, but the final source/ground-truth NPZ identity fields alone do not independently prove frame-to-action alignment.",
        },
        "verdict_impact": {
            "learning_curve_changed": False,
            "random_window_vs_route_changed": False,
            "domain_matrix_changed": False,
            "candidate_upper_bound_changed": False,
            "pre_registered_verdict": base["independent_route_value"]["verdict"],
            "verdict_changed": False,
        },
        "reserve_accessed": False,
        "new_data_ingested": False,
        "v76_trained": False,
        "v70_or_cem_mutated": False,
    }
    v72.write_json(DESTINATION, corrected)
    print(json.dumps(corrected, sort_keys=True))


if __name__ == "__main__":
    main()
