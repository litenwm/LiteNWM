#!/usr/bin/env python3
"""Read-only/derived attribution of the spent SCAND proposal-tail bottleneck."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
from scipy.stats import rankdata
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.evaluation import run_v72_proposal_tail_reranker as v72  # noqa: E402
from tools.evaluation import run_v73_nested_safety_calibration as v73  # noqa: E402
from tools.evaluation import run_v74_direct_delta_nested as v74  # noqa: E402
from tools.evaluation import run_v75_structural_selective_classifier as v75  # noqa: E402


OUTPUT = ROOT / "outputs/cem_development_v1/scand_bottleneck_diagnostic_v1"
PREREG = OUTPUT / "preregistration.json"
V70_RESULT = v74.V70_RESULT
V74_RESULT = ROOT / "outputs/cem_development_v1/v74_direct_delta_nested/result.json"
V75_RESULT = ROOT / "outputs/cem_development_v1/v75_structural_selective_classifier/result.json"
FRACTIONS = (0.25, 0.50, 0.75, 1.0)
ROUTE_COUNTS = (5, 10, 15, 20)
REPEATS = 8
TOLERANCE = 1.0e-7


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser.parse_args()


def verify_contract() -> dict[str, str]:
    prereg = v72.read_json(PREREG)
    if prereg.get("status") != "frozen_before_diagnostic_computation":
        raise RuntimeError("diagnostic preregistration is not frozen")
    if not prereg["reserve"]["forbidden"] or prereg["reserve"]["accessed"]:
        raise RuntimeError("forbidden-cohort contract changed")
    paths = {
        "formal_v70_result_sha256": V70_RESULT,
        "v74_result_sha256": V74_RESULT,
        "v75_result_sha256": V75_RESULT,
    }
    observed = {key: v72.sha256_file(path) for key, path in paths.items()}
    for key, digest in observed.items():
        if digest != prereg["immutable_system"][key]:
            raise RuntimeError(f"immutable input changed: {key}")
    return {"preregistration_sha256": v72.sha256_file(PREREG), **observed}


def binary_metrics(probability: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    p = np.asarray(probability, dtype=np.float64).reshape(-1)
    y = np.asarray(target, dtype=np.int64).reshape(-1)
    positives = int(y.sum())
    negatives = int(len(y) - positives)
    if positives and negatives:
        ranks = rankdata(p, method="average")
        auc = (ranks[y == 1].sum() - positives * (positives + 1) / 2.0) / (positives * negatives)
        order = np.argsort(-p, kind="stable")
        sorted_y = y[order]
        precision = np.cumsum(sorted_y) / np.arange(1, len(y) + 1)
        average_precision = float(np.sum(precision * sorted_y) / positives)
    else:
        auc = average_precision = None
    calibration = v73.candidate_calibration_metrics(
        p, y, np.ones(len(y), dtype=bool)
    )
    return {
        "rows": int(len(y)),
        "positive_rate": float(y.mean()) if len(y) else None,
        "auroc": float(auc) if auc is not None else None,
        "average_precision": average_precision,
        "brier": calibration["brier"],
        "ece": calibration["ece_10_equal_width"],
    }


def pearson(x: np.ndarray, y: np.ndarray) -> float | None:
    a = np.asarray(x, dtype=np.float64).reshape(-1)
    b = np.asarray(y, dtype=np.float64).reshape(-1)
    if len(a) < 2 or a.std() <= 0 or b.std() <= 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def eligible_mask(data: dict[str, Any], states: np.ndarray) -> np.ndarray:
    mask = data["valid"][states] & data["safe"][states]
    mask[np.arange(len(states)), data["formal_index"][states]] = False
    return mask


def metric_bundle(
    data: dict[str, Any],
    states: np.ndarray,
    probability_dual: np.ndarray,
    probability_harm: np.ndarray,
    objective_prediction: np.ndarray,
) -> dict[str, Any]:
    mask = eligible_mask(data, states)
    true_objective = data["gain"][states, :, 3]
    error = objective_prediction[mask] - true_objective[mask]
    return {
        "states": int(len(states)),
        "candidate_rows": int(mask.sum()),
        "dual": binary_metrics(
            probability_dual[mask], data["dual_vs_formal"][states][mask]
        ),
        "harm": binary_metrics(
            probability_harm[mask], data["harm_vs_formal"][states][mask]
        ),
        "objective_delta": {
            "pearson": pearson(objective_prediction[mask], true_objective[mask]),
            "mae_m": float(np.mean(np.abs(error))),
            "rmse_m": float(np.sqrt(np.mean(error * error))),
            "mean_true_gain_m": float(true_objective[mask].mean()),
        },
    }


def fit_predict(
    data: dict[str, Any],
    train_states: np.ndarray,
    test_states: np.ndarray,
    *,
    device: str,
    seed_offset: int,
) -> dict[str, np.ndarray]:
    combined = np.concatenate((train_states, test_states))
    probability, _, _ = v75.fit_predict_method(
        "balanced_logistic_relative",
        data,
        train_states,
        combined,
        device=device,
        seed_offset=seed_offset,
    )
    train_probability = probability[: len(train_states)]
    test_probability = probability[len(train_states) :]
    calibration_mask = eligible_mask(data, train_states)
    dual_platt = v73.fit_platt(
        train_probability[..., 0][calibration_mask],
        data["dual_vs_formal"][train_states][calibration_mask],
    )
    harm_platt = v73.fit_platt(
        train_probability[..., 1][calibration_mask],
        data["harm_vs_formal"][train_states][calibration_mask],
    )
    probability_dual = v73.apply_platt(test_probability[..., 0], dual_platt)
    probability_harm = v73.apply_platt(test_probability[..., 1], harm_platt)
    ridge = v74.fit_ridge(data["v75_relative"], data, train_states)
    objective_prediction = v74.predict_ridge(
        ridge, data["v75_relative"][test_states]
    )[0][..., 3]
    return {
        "dual": probability_dual.astype(np.float32),
        "harm": probability_harm.astype(np.float32),
        "objective": objective_prediction.astype(np.float32),
    }


def allocate_prediction(data: dict[str, Any]) -> dict[str, np.ndarray]:
    shape = data["valid"].shape
    return {
        "dual": np.full(shape, np.nan, dtype=np.float32),
        "harm": np.full(shape, np.nan, dtype=np.float32),
        "objective": np.full(shape, np.nan, dtype=np.float32),
    }


def place_prediction(
    destination: dict[str, np.ndarray], states: np.ndarray, value: dict[str, np.ndarray]
) -> None:
    for key in destination:
        destination[key][states] = value[key]


def deterministic_groups(
    groups: list[str], outer: int, repeat: int, count: int
) -> list[str]:
    return sorted(
        groups,
        key=lambda value: hashlib.sha256(
            f"{outer}|{repeat}|{value}".encode()
        ).hexdigest(),
    )[:count]


def route_metrics(
    data: dict[str, Any],
    states: np.ndarray,
    prediction: dict[str, np.ndarray],
) -> dict[str, dict[str, Any]]:
    result = {}
    for group in sorted(np.unique(data["group"][states].astype(str))):
        local = states[data["group"][states].astype(str) == group]
        result[group] = metric_bundle(
            data,
            local,
            prediction["dual"][local],
            prediction["harm"][local],
            prediction["objective"][local],
        )
    return result


def route_dispersion(route_result: dict[str, dict[str, Any]]) -> dict[str, float]:
    def values(path: tuple[str, ...]) -> np.ndarray:
        output = []
        for value in route_result.values():
            cursor: Any = value
            for key in path:
                cursor = cursor[key]
            if cursor is not None and np.isfinite(cursor):
                output.append(float(cursor))
        return np.asarray(output, dtype=np.float64)

    dual = values(("dual", "auroc"))
    correlation = values(("objective_delta", "pearson"))
    mae = values(("objective_delta", "mae_m"))
    return {
        "dual_auroc_std": float(dual.std()) if len(dual) else float("nan"),
        "objective_pearson_std": float(correlation.std()) if len(correlation) else float("nan"),
        "objective_mae_std_m": float(mae.std()) if len(mae) else float("nan"),
        "routes_with_dual_auroc": int(len(dual)),
        "routes_with_objective_pearson": int(len(correlation)),
    }


def summarize_repeats(records: list[dict[str, Any]], key: str) -> dict[str, Any]:
    value = np.asarray([record[key] for record in records], dtype=np.float64)
    return {
        "mean": float(value.mean()),
        "std": float(value.std()),
        "ci95_percentile": [
            float(np.quantile(value, 0.025)),
            float(np.quantile(value, 0.975)),
        ],
        "values": value.tolist(),
    }


def scand_learning_curve(
    data: dict[str, Any], *, device: str
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    scand = data["domain"].astype(str) == "scand"
    scand_states = np.flatnonzero(scand)
    records: dict[int, list[dict[str, Any]]] = {count: [] for count in ROUTE_COUNTS}
    repeat_zero_full: dict[str, np.ndarray] | None = None
    for repeat in range(REPEATS):
        for count in ROUTE_COUNTS:
            prediction = allocate_prediction(data)
            selected_train_groups = []
            for outer in range(5):
                test = np.flatnonzero(scand & (data["fold"] == outer))
                available = sorted(
                    np.unique(data["group"][scand & (data["fold"] != outer)].astype(str)).tolist()
                )
                chosen = deterministic_groups(available, outer, repeat, count)
                train = np.flatnonzero(scand & np.isin(data["group"].astype(str), chosen))
                value = fit_predict(
                    data,
                    train,
                    test,
                    device=device,
                    seed_offset=100000 + repeat * 1000 + count * 10 + outer,
                )
                place_prediction(prediction, test, value)
                selected_train_groups.append(
                    {
                        "outer_fold": outer,
                        "groups": chosen,
                        "train_states": int(len(train)),
                        "test_states": int(len(test)),
                        "route_overlap": 0,
                    }
                )
            metrics = metric_bundle(
                data,
                scand_states,
                prediction["dual"][scand_states],
                prediction["harm"][scand_states],
                prediction["objective"][scand_states],
            )
            per_route = route_metrics(data, scand_states, prediction)
            dispersion = route_dispersion(per_route)
            records[count].append(
                {
                    "repeat": repeat,
                    "dual_auroc": metrics["dual"]["auroc"],
                    "harm_auroc": metrics["harm"]["auroc"],
                    "dual_ece": metrics["dual"]["ece"],
                    "harm_ece": metrics["harm"]["ece"],
                    "objective_pearson": metrics["objective_delta"]["pearson"],
                    "objective_mae_m": metrics["objective_delta"]["mae_m"],
                    "route_dispersion": dispersion,
                    "training_audit": selected_train_groups,
                }
            )
            if repeat == 0 and count == 20:
                repeat_zero_full = prediction
            print(
                json.dumps(
                    {
                        "stage": "learning_curve",
                        "repeat": repeat,
                        "train_routes": count,
                        "dual_auroc": metrics["dual"]["auroc"],
                        "objective_pearson": metrics["objective_delta"]["pearson"],
                        "objective_mae_m": metrics["objective_delta"]["mae_m"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    summary = {}
    for count in ROUTE_COUNTS:
        local = records[count]
        summary[str(count)] = {
            key: summarize_repeats(local, key)
            for key in (
                "dual_auroc",
                "harm_auroc",
                "dual_ece",
                "harm_ece",
                "objective_pearson",
                "objective_mae_m",
            )
        }
        summary[str(count)]["route_dispersion"] = {
            key: summarize_repeats(
                [{"value": value["route_dispersion"][key]} for value in local],
                "value",
            )
            for key in (
                "dual_auroc_std",
                "objective_pearson_std",
                "objective_mae_std_m",
            )
        }
    slopes = {}
    for metric in ("dual_auroc", "harm_auroc", "objective_pearson", "objective_mae_m"):
        values = []
        for repeat in range(REPEATS):
            y = np.asarray([records[count][repeat][metric] for count in ROUTE_COUNTS], dtype=np.float64)
            values.append(float(np.polyfit(np.asarray(ROUTE_COUNTS, dtype=np.float64), y, 1)[0]))
        slopes[metric] = {
            "per_added_route": summarize_repeats(
                [{"value": value} for value in values], "value"
            ),
            "mean_total_change_5_to_20": float(
                np.mean([record[metric] for record in records[20]])
                - np.mean([record[metric] for record in records[5]])
            ),
        }
    if repeat_zero_full is None:
        raise RuntimeError("missing full-route diagnostic prediction")
    return {
        "schema": "scand_route_subsampling_learning_curve_v1",
        "test_routes_fixed_by_outer_fold": True,
        "heldout_used_for_tuning": False,
        "records": records,
        "summary": summary,
        "slopes": slopes,
    }, repeat_zero_full


def random_window_cv(
    data: dict[str, Any], *, device: str
) -> dict[str, Any]:
    scand_states = np.flatnonzero(data["domain"].astype(str) == "scand")
    records = []
    overlap_audit = []
    for repeat in range(REPEATS):
        rng = np.random.default_rng(20260780 + repeat)
        order = scand_states.copy()
        rng.shuffle(order)
        assignment = np.empty(len(data["record_key"]), dtype=np.int64)
        assignment.fill(-1)
        assignment[order] = np.arange(len(order)) % 5
        prediction = allocate_prediction(data)
        for fold in range(5):
            train = scand_states[assignment[scand_states] != fold]
            test = scand_states[assignment[scand_states] == fold]
            train_groups = set(data["group"][train].astype(str))
            test_groups = set(data["group"][test].astype(str))
            overlap = len(train_groups & test_groups)
            value = fit_predict(
                data,
                train,
                test,
                device=device,
                seed_offset=200000 + repeat * 100 + fold,
            )
            place_prediction(prediction, test, value)
            overlap_audit.append(
                {
                    "repeat": repeat,
                    "fold": fold,
                    "train_states": int(len(train)),
                    "test_states": int(len(test)),
                    "route_overlap": overlap,
                    "test_routes": int(len(test_groups)),
                }
            )
        metrics = metric_bundle(
            data,
            scand_states,
            prediction["dual"][scand_states],
            prediction["harm"][scand_states],
            prediction["objective"][scand_states],
        )
        records.append(
            {
                "repeat": repeat,
                "dual_auroc": metrics["dual"]["auroc"],
                "harm_auroc": metrics["harm"]["auroc"],
                "dual_ece": metrics["dual"]["ece"],
                "harm_ece": metrics["harm"]["ece"],
                "objective_pearson": metrics["objective_delta"]["pearson"],
                "objective_mae_m": metrics["objective_delta"]["mae_m"],
            }
        )
    return {
        "records": records,
        "summary": {
            key: summarize_repeats(records, key)
            for key in (
                "dual_auroc",
                "harm_auroc",
                "dual_ece",
                "harm_ece",
                "objective_pearson",
                "objective_mae_m",
            )
        },
        "route_overlap_audit": overlap_audit,
        "mean_overlapping_routes_per_fold": float(
            np.mean([value["route_overlap"] for value in overlap_audit])
        ),
        "intentionally_leaky_window_split": True,
    }


def domain_train_matrix(
    data: dict[str, Any], *, device: str
) -> tuple[dict[str, Any], dict[str, dict[str, np.ndarray]]]:
    modes = {
        "scand_only": data["domain"].astype(str) == "scand",
        "recon_only": data["domain"].astype(str) == "recon",
        "mixed": np.ones(len(data["record_key"]), dtype=bool),
    }
    predictions = {}
    result = {}
    for mode_index, (mode, train_domain) in enumerate(modes.items()):
        prediction = allocate_prediction(data)
        fold_audit = []
        for outer in range(5):
            train = np.flatnonzero(train_domain & (data["fold"] != outer))
            test = np.flatnonzero(data["fold"] == outer)
            if set(data["group"][train].astype(str)) & set(data["group"][test].astype(str)):
                raise RuntimeError("domain matrix route leakage")
            value = fit_predict(
                data,
                train,
                test,
                device=device,
                seed_offset=300000 + mode_index * 100 + outer,
            )
            place_prediction(prediction, test, value)
            fold_audit.append(
                {
                    "outer_fold": outer,
                    "train_states": int(len(train)),
                    "test_states": int(len(test)),
                    "route_overlap": 0,
                }
            )
        predictions[mode] = prediction
        result[mode] = {"fold_audit": fold_audit, "test": {}}
        for domain in ("recon", "scand"):
            states = np.flatnonzero(data["domain"].astype(str) == domain)
            result[mode]["test"][domain] = metric_bundle(
                data,
                states,
                prediction["dual"][states],
                prediction["harm"][states],
                prediction["objective"][states],
            )
    scand_only = result["scand_only"]["test"]["scand"]
    mixed = result["mixed"]["test"]["scand"]
    result["negative_transfer_deltas_mixed_minus_scand_only"] = {
        "dual_auroc": mixed["dual"]["auroc"] - scand_only["dual"]["auroc"],
        "harm_auroc": mixed["harm"]["auroc"] - scand_only["harm"]["auroc"],
        "objective_pearson": mixed["objective_delta"]["pearson"] - scand_only["objective_delta"]["pearson"],
        "objective_mae_m": mixed["objective_delta"]["mae_m"] - scand_only["objective_delta"]["mae_m"],
    }
    return result, predictions


def state_representations(
    data: dict[str, Any], feature_names: list[str]
) -> tuple[np.ndarray, list[str]]:
    keep = [
        index
        for index, name in enumerate(feature_names)
        if name not in {"domain_recon", "domain_scand", "formal_selected"}
    ]
    feature = data["v75_relative"][..., keep].astype(np.float64)
    representation = np.zeros((len(feature), 2 * len(keep)), dtype=np.float64)
    for state in range(len(feature)):
        mask = data["valid"][state] & data["safe"][state]
        mask[int(data["formal_index"][state])] = False
        value = feature[state, mask]
        representation[state] = np.concatenate((value.mean(axis=0), value.std(axis=0)))
    return representation, [feature_names[index] for index in keep]


def auc_score(score: np.ndarray, target: np.ndarray) -> float:
    ranks = rankdata(score, method="average")
    positives = int(target.sum())
    negatives = int(len(target) - positives)
    return float(
        (ranks[target == 1].sum() - positives * (positives + 1) / 2.0)
        / (positives * negatives)
    )


def representation_audit(
    data: dict[str, Any],
    feature_names: list[str],
    strict_scand_prediction: dict[str, np.ndarray],
) -> dict[str, Any]:
    representation, kept = state_representations(data, feature_names)
    scand_states = np.flatnonzero(data["domain"].astype(str) == "scand")
    correct = 0
    for state in scand_states:
        train = scand_states[scand_states != state]
        mean = representation[train].mean(axis=0)
        scale = representation[train].std(axis=0)
        active = scale >= 1.0e-6
        train_value = (representation[train][:, active] - mean[active]) / scale[active]
        test_value = (representation[state, active] - mean[active]) / scale[active]
        centroids = {}
        for group in np.unique(data["group"][train].astype(str)):
            centroids[group] = train_value[data["group"][train].astype(str) == group].mean(axis=0)
        predicted = min(
            centroids,
            key=lambda group: float(np.linalg.norm(test_value - centroids[group])),
        )
        correct += int(predicted == str(data["group"][state]))
    domain_score = np.full(len(data["record_key"]), np.nan, dtype=np.float64)
    domain_target = (data["domain"].astype(str) == "scand").astype(np.int64)
    for outer in range(5):
        train = np.flatnonzero(data["fold"] != outer)
        test = np.flatnonzero(data["fold"] == outer)
        mean = representation[train].mean(axis=0)
        scale = representation[train].std(axis=0)
        active = scale >= 1.0e-6
        x_train = (representation[train][:, active] - mean[active]) / scale[active]
        x_test = (representation[test][:, active] - mean[active]) / scale[active]
        x_train = np.concatenate((x_train, np.ones((len(x_train), 1))), axis=1)
        x_test = np.concatenate((x_test, np.ones((len(x_test), 1))), axis=1)
        matrix = x_train.T @ x_train
        matrix.flat[:: len(matrix) + 1] += 20.0
        coefficient = np.linalg.solve(matrix, x_train.T @ domain_target[train])
        domain_score[test] = x_test @ coefficient
    distances = np.full(len(data["record_key"]), np.nan, dtype=np.float64)
    for outer in range(5):
        train = np.flatnonzero(
            (data["domain"].astype(str) == "scand") & (data["fold"] != outer)
        )
        test = np.flatnonzero(
            (data["domain"].astype(str) == "scand") & (data["fold"] == outer)
        )
        mean = representation[train].mean(axis=0)
        scale = representation[train].std(axis=0)
        active = scale >= 1.0e-6
        train_value = (representation[train][:, active] - mean[active]) / scale[active]
        test_value = (representation[test][:, active] - mean[active]) / scale[active]
        centroids = []
        for group in np.unique(data["group"][train].astype(str)):
            centroids.append(train_value[data["group"][train].astype(str) == group].mean(axis=0))
        centroid = np.stack(centroids)
        distances[test] = np.min(
            np.sqrt(np.sum((test_value[:, None] - centroid[None]) ** 2, axis=-1)),
            axis=1,
        )
    state_error = np.full(len(data["record_key"]), np.nan, dtype=np.float64)
    for state in scand_states:
        mask = eligible_mask(data, np.asarray([state]))[0]
        state_error[state] = float(
            np.mean(
                np.abs(
                    strict_scand_prediction["objective"][state, mask]
                    - data["gain"][state, mask, 3]
                )
            )
        )
    route_distance = []
    route_error = []
    for group in np.unique(data["group"][scand_states].astype(str)):
        local = scand_states[data["group"][scand_states].astype(str) == group]
        route_distance.append(float(np.mean(distances[local])))
        route_error.append(float(np.mean(state_error[local])))
    route_distance_value = np.asarray(route_distance)
    route_error_value = np.asarray(route_error)
    cutoff = float(np.quantile(route_distance_value, 0.75))
    return {
        "state_representation_dimension": int(representation.shape[1]),
        "explicit_domain_features_removed": True,
        "feature_name_sha256": hashlib.sha256("\n".join(kept).encode()).hexdigest(),
        "route_identification_accuracy_leave_one_window_out": float(correct / len(scand_states)),
        "route_identification_chance": 1.0 / 25.0,
        "route_identification_multiple_of_chance": float((correct / len(scand_states)) * 25.0),
        "domain_identification_route_oof_auroc": auc_score(domain_score, domain_target),
        "support_distance": {
            "route_distance_error_pearson": pearson(route_distance_value, route_error_value),
            "top_quartile_distance_cutoff": cutoff,
            "top_quartile_mean_objective_mae_m": float(route_error_value[route_distance_value >= cutoff].mean()),
            "lower_three_quartiles_mean_objective_mae_m": float(route_error_value[route_distance_value < cutoff].mean()),
            "route_distances": route_distance,
            "route_objective_mae_m": route_error,
        },
    }


def candidate_and_route_audit(
    data: dict[str, Any], strict_prediction: dict[str, np.ndarray]
) -> dict[str, Any]:
    scand_states = np.flatnonzero(data["domain"].astype(str) == "scand")
    per_route = {}
    opportunities = []
    candidate_dual = []
    oracle_gain = []
    for group in sorted(np.unique(data["group"][scand_states].astype(str))):
        states = scand_states[data["group"][scand_states].astype(str) == group]
        mask = eligible_mask(data, states)
        dual = data["dual_vs_formal"][states].astype(bool) & mask
        state_opportunity = dual.any(axis=1)
        gains = []
        for row in range(len(states)):
            gains.append(float(np.max(data["gain"][states[row], dual[row], 3])) if dual[row].any() else 0.0)
        metrics = route_metrics(data, states, strict_prediction)[group]
        per_route[group] = {
            "states": int(len(states)),
            "eligible_candidates": int(mask.sum()),
            "valid_safe_dual_candidates": int(dual.sum()),
            "dual_candidate_fraction": float(dual.sum() / max(mask.sum(), 1)),
            "state_opportunity_coverage": float(state_opportunity.mean()),
            "dual_oracle_mean_whole_state_gain_m": float(np.mean(gains)),
            "dual_label_rate": float(data["dual_vs_formal"][states][mask].mean()),
            "harm_label_rate": float(data["harm_vs_formal"][states][mask].mean()),
            "valid_fraction_all_65": float(data["valid"][states].mean()),
            "safe_fraction_all_65": float(data["safe"][states].mean()),
            "diagnostic_metrics": metrics,
        }
        opportunities.extend(state_opportunity.tolist())
        candidate_dual.extend(data["dual_vs_formal"][states][mask].tolist())
        oracle_gain.extend(gains)
    source = {}
    mask = eligible_mask(data, scand_states)
    for source_id, name in enumerate(("fixed", "r1", "r2", "r3")):
        local = mask & (data["source_id"][scand_states] == source_id)
        source[name] = {
            "candidate_rows": int(local.sum()),
            "dual_rate": float(data["dual_vs_formal"][scand_states][local].mean()) if local.any() else None,
            "harm_rate": float(data["harm_vs_formal"][scand_states][local].mean()) if local.any() else None,
            "mean_objective_gain_m": float(data["gain"][scand_states, :, 3][local].mean()) if local.any() else None,
        }
    return {
        "per_route": per_route,
        "pooled": {
            "states": int(len(scand_states)),
            "state_opportunity_coverage": float(np.mean(opportunities)),
            "valid_safe_dual_candidate_fraction": float(np.mean(candidate_dual)),
            "dual_oracle_mean_whole_state_gain_m": float(np.mean(oracle_gain)),
            "routes_with_any_state_opportunity": int(sum(value["state_opportunity_coverage"] > 0 for value in per_route.values())),
            "routes": int(len(per_route)),
        },
        "proposal_source": source,
    }


def signal_and_error_audit(
    data: dict[str, Any],
    strict_prediction: dict[str, np.ndarray],
    feature_names: list[str],
) -> dict[str, Any]:
    scand_states = np.flatnonzero(data["domain"].astype(str) == "scand")
    mask = eligible_mask(data, scand_states)
    feature = data["v75_relative"][scand_states]
    target = data["dual_vs_formal"][scand_states][mask].astype(np.int64)
    candidates = []
    for index, name in enumerate(feature_names):
        value = feature[..., index][mask]
        metric = binary_metrics(value, target)
        if metric["auroc"] is not None:
            candidates.append(
                {
                    "feature": name,
                    "oriented_auroc": max(metric["auroc"], 1.0 - metric["auroc"]),
                    "raw_auroc": metric["auroc"],
                }
            )
    candidates.sort(key=lambda value: value["oriented_auroc"], reverse=True)
    top = candidates[:12]
    route_direction = {}
    for item in top:
        index = feature_names.index(item["feature"])
        pooled_direction = 1 if item["raw_auroc"] >= 0.5 else -1
        directions = []
        for group in np.unique(data["group"][scand_states].astype(str)):
            states = scand_states[data["group"][scand_states].astype(str) == group]
            local_mask = eligible_mask(data, states)
            local_target = data["dual_vs_formal"][states][local_mask].astype(np.int64)
            if local_target.sum() and (1 - local_target).sum():
                auc = binary_metrics(
                    data["v75_relative"][states, :, index][local_mask], local_target
                )["auroc"]
                directions.append(1 if auc >= 0.5 else -1)
        route_direction[item["feature"]] = {
            "routes_with_defined_direction": int(len(directions)),
            "fraction_matching_pooled_direction": float(np.mean(np.asarray(directions) == pooled_direction)) if directions else None,
        }
    objective_error = np.abs(
        strict_prediction["objective"][scand_states] - data["gain"][scand_states, :, 3]
    )
    relation = {}
    for name in (
        "seed_rtp_std",
        "seed_rtp_range",
        "seed_rtp_std_delta",
        "seed_rtp_range_delta",
        "source_rank",
        "scorer_rank",
        "energy_rank",
        "local_density_4",
        "local_density_7",
    ):
        if name in feature_names:
            index = feature_names.index(name)
            relation[name] = {
                "absolute_error_pearson": pearson(
                    data["v75_relative"][scand_states, :, index][mask],
                    objective_error[mask],
                )
            }
    return {
        "top_univariate_dual_signals": top,
        "route_direction_consistency": route_direction,
        "feature_error_relationship": relation,
        "used_for_model_or_threshold_selection": False,
    }


def near_duplicate_audit(data: dict[str, Any]) -> dict[str, Any]:
    states = np.flatnonzero(data["domain"].astype(str) == "scand")
    result = {}
    for threshold in (0.001, 0.005):
        pairs = 0
        dual_disagreement = 0
        harm_disagreement = 0
        objective_difference = []
        for state in states:
            index = np.flatnonzero(data["valid"][state])
            waypoint = data["candidates"][state, index].astype(np.float64) * 0.25
            distance = np.sqrt(
                np.mean(
                    np.sum(
                        (waypoint[:, None] - waypoint[None, :]) ** 2,
                        axis=-1,
                    ),
                    axis=-1,
                )
            )
            left, right = np.triu_indices(len(index), 1)
            local = distance[left, right] <= threshold
            for a, b in zip(left[local], right[local]):
                first = index[a]
                second = index[b]
                pairs += 1
                dual_disagreement += int(
                    bool(data["dual_vs_formal"][state, first])
                    != bool(data["dual_vs_formal"][state, second])
                )
                harm_disagreement += int(
                    bool(data["harm_vs_formal"][state, first])
                    != bool(data["harm_vs_formal"][state, second])
                )
                objective_difference.append(
                    abs(
                        float(data["true_metrics"][state, first, 3])
                        - float(data["true_metrics"][state, second, 3])
                    )
                )
        result[f"{threshold:.3f}m"] = {
            "pairs": pairs,
            "dual_label_disagreement_rate": float(dual_disagreement / pairs) if pairs else None,
            "harm_label_disagreement_rate": float(harm_disagreement / pairs) if pairs else None,
            "objective_difference_mean_m": float(np.mean(objective_difference)) if pairs else None,
            "objective_difference_p90_m": float(np.quantile(objective_difference, 0.9)) if pairs else None,
            "objective_difference_over_5mm_rate": float(np.mean(np.asarray(objective_difference) > 0.005)) if pairs else None,
        }
    return result


def contract_audit(data: dict[str, Any]) -> dict[str, Any]:
    cohort_metric = {}
    identity = {}
    for spec in v72.COHORTS:
        replay = v72.V70_ROOT / "replays" / spec["name"] / "proposal_aware_v70"
        freeze = v72.read_json(replay / "prelabel_freeze.json")
        feature = v72.load_npz(v72.resolve_path(freeze["base_features"]))
        ground = v72.load_npz(Path(spec["ground_truth"]))
        record_equal = np.array_equal(
            feature["record_key"].astype(str), ground["record_key"].astype(str)
        )
        recomputed = v72.metric_values(
            feature["candidates"].astype(np.float32),
            ground["ground_truth_local_m"].astype(np.float32),
        )
        stored_shape = list(ground['absolute_metrics'].shape) if 'absolute_metrics' in ground else None
        recomputed_shape = list(recomputed.shape)
        if 'absolute_metrics' in ground and ground['absolute_metrics'].shape == recomputed.shape:
            max_difference = float(
                np.max(
                    np.abs(
                        recomputed
                        - ground["absolute_metrics"].astype(np.float32)
                    )
                )
            )
        else:
            max_difference = None
        identity[spec["name"]] = {
            "record_key_equal": bool(record_equal),
            "states": int(len(feature["record_key"])),
            "record_keys_unique": bool(
                len(np.unique(feature["record_key"].astype(str)))
                == len(feature["record_key"])
            ),
            "feature_ground_candidate_count": int(feature["candidates"].shape[1]),
        }
        cohort_metric[spec["name"]] = {
            "stored_absolute_metrics_available": "absolute_metrics" in ground,
            'stored_absolute_metrics_shape': stored_shape,
            'recomputed_fixed_k17_shape': recomputed_shape,
            'candidate_dimensions_directly_comparable': bool('absolute_metrics' in ground and ground['absolute_metrics'].shape == recomputed.shape),
            "recomputed_vs_stored_max_abs_difference_m": max_difference,
            "shared_metric_function": "metric_values; objective=0.4*ATE+0.6*endpoint; waypoint_unit=0.25m",
        }
    scand_spec = next(spec for spec in v72.COHORTS if spec["name"] == "scand_fresh_outer")
    replay = v72.V70_ROOT / "replays" / scand_spec["name"] / "proposal_aware_v70"
    freeze = v72.read_json(replay / "prelabel_freeze.json")
    manifest_path = v72.resolve_path(freeze["manifest"])
    manifest = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    keys = sorted({key for row in manifest for key in row})
    time_fields = [key for key in keys if any(token in key.lower() for token in ("time", "stamp", "frame", "index"))]
    image_fields = [key for key in keys if any(token in key.lower() for token in ("image", "width", "height", "resolution", "quality"))]
    source_path = ROOT / "outputs/scand_domain_adaptation_v1/fresh_outer_v1/fresh_outer_source.npz"
    source = v72.load_npz(source_path)
    ground = v72.load_npz(Path(scand_spec["ground_truth"]))
    uid_unique = len(np.unique(source["window_uid"].astype(str))) == len(source["window_uid"])
    route_match = np.array_equal(source["route_group"].astype(str), ground["route_group"].astype(str))
    trajectory_match = np.array_equal(source["trajectory"].astype(str), ground["trajectory"].astype(str))
    return {
        "identity": identity,
        "metric_definition": cohort_metric,
        "scand_manifest": {
            "path": str(manifest_path),
            "rows": int(len(manifest)),
            "keys": keys,
            "explicit_time_or_frame_fields": time_fields,
            "explicit_image_resolution_or_quality_fields": image_fields,
            "action_frame_alignment_verifiable_from_current_manifest": bool(time_fields),
            "image_resolution_quality_verifiable_from_current_manifest": bool(image_fields),
        },
        "scand_source_identity": {
            "source_path": str(source_path),
            "window_uid_unique": bool(uid_unique),
            "source_ground_route_group_equal": bool(route_match),
            "source_ground_trajectory_equal": bool(trajectory_match),
            "record_key_equal": bool(
                np.array_equal(
                    source["record_key"].astype(str), ground["record_key"].astype(str)
                )
            ),
            "source_fields": sorted(source),
        },
        "near_duplicate_candidates": near_duplicate_audit(data),
    }


def usefulness_decision(
    curve: dict[str, Any], contract: dict[str, Any]
) -> dict[str, Any]:
    slopes = curve["slopes"]
    dual = slopes["dual_auroc"]
    objective = slopes["objective_pearson"]
    mae = slopes["objective_mae_m"]
    dual_strong = (
        dual["per_added_route"]["mean"] > 0.002
        and dual["per_added_route"]["ci95_percentile"][0] > 0.0
        and dual["mean_total_change_5_to_20"] >= 0.03
    )
    objective_strong = (
        objective["per_added_route"]["mean"] > 0.002
        and objective["per_added_route"]["ci95_percentile"][0] > 0.0
        and objective["mean_total_change_5_to_20"] >= 0.03
    )
    dual_nonadverse = dual["mean_total_change_5_to_20"] >= -0.02
    objective_nonadverse = objective["mean_total_change_5_to_20"] >= -0.02
    mae_nonincrease = mae["mean_total_change_5_to_20"] <= 0.0
    quarter = curve["summary"]["5"]["route_dispersion"]
    full = curve["summary"]["20"]["route_dispersion"]
    dispersion_checks = {}
    for key in (
        "dual_auroc_std",
        "objective_pearson_std",
        "objective_mae_std_m",
    ):
        before = quarter[key]["mean"]
        after = full[key]["mean"]
        dispersion_checks[key] = bool(np.isfinite(before) and before > 0 and after <= 0.9 * before)
    dispersion_passes = sum(dispersion_checks.values())
    metric_differences = [
        value["recomputed_vs_stored_max_abs_difference_m"]
        for value in contract["metric_definition"].values()
        if value["recomputed_vs_stored_max_abs_difference_m"] is not None
    ]
    near = contract["near_duplicate_candidates"]
    duplicate_bad = any(
        value["pairs"] > 0
        and max(
            value["dual_label_disagreement_rate"] or 0.0,
            value["harm_label_disagreement_rate"] or 0.0,
        )
        > 0.05
        for value in near.values()
    )
    contract_error = (
        any(not value["record_key_equal"] for value in contract["identity"].values())
        or (metric_differences and max(metric_differences) > 1.0e-6)
        or duplicate_bad
    )
    useful = (
        (dual_strong or objective_strong)
        and (objective_nonadverse if dual_strong else dual_nonadverse)
        and mae_nonincrease
        and dispersion_passes >= 2
        and not contract_error
    )
    point_positive = (
        (
            dual["per_added_route"]["mean"] > 0
            and dual["mean_total_change_5_to_20"] >= 0.02
        )
        or (
            objective["per_added_route"]["mean"] > 0
            and objective["mean_total_change_5_to_20"] >= 0.02
        )
    )
    conditional = (
        not useful
        and point_positive
        and dual_nonadverse
        and objective_nonadverse
        and mae_nonincrease
        and not contract_error
    )
    verdict = "useful" if useful else "conditionally_useful" if conditional else "not_yet_useful"
    return {
        "verdict": verdict,
        "checks": {
            "dual_learning_curve_strong": bool(dual_strong),
            "objective_learning_curve_strong": bool(objective_strong),
            "dual_nonadverse": bool(dual_nonadverse),
            "objective_nonadverse": bool(objective_nonadverse),
            "objective_mae_nonincrease": bool(mae_nonincrease),
            "dispersion_reduction_checks": dispersion_checks,
            "dispersion_reduction_passes": int(dispersion_passes),
            "dominant_contract_error": bool(contract_error),
            "conditionally_positive_point_evidence": bool(point_positive),
        },
        "automatic_continuation": False,
    }


def bottleneck_attribution(
    random_cv: dict[str, Any],
    curve: dict[str, Any],
    matrix: dict[str, Any],
    representation: dict[str, Any],
    candidate: dict[str, Any],
    contract: dict[str, Any],
    usefulness: dict[str, Any],
) -> dict[str, Any]:
    route_full = curve["summary"]["20"]
    random_summary = random_cv["summary"]
    leak = {
        "dual_auroc": random_summary["dual_auroc"]["mean"] - route_full["dual_auroc"]["mean"],
        "objective_pearson": random_summary["objective_pearson"]["mean"] - route_full["objective_pearson"]["mean"],
        "objective_mae_relative_reduction": 1.0 - random_summary["objective_mae_m"]["mean"] / route_full["objective_mae_m"]["mean"],
    }
    route_shift_primary = (
        leak["dual_auroc"] >= 0.10
        or leak["objective_pearson"] >= 0.10
        or leak["objective_mae_relative_reduction"] >= 0.20
    ) and (
        representation["route_identification_multiple_of_chance"] >= 3.0
        or abs(representation["support_distance"]["route_distance_error_pearson"] or 0.0) >= 0.3
    )
    opportunity = candidate["pooled"]["state_opportunity_coverage"]
    proposal_primary = opportunity < 0.20
    representation_primary = (
        route_full["dual_auroc"]["mean"] < 0.55
        and opportunity >= 0.20
        and usefulness["verdict"] == "not_yet_useful"
    )
    negative = matrix["negative_transfer_deltas_mixed_minus_scand_only"]
    negative_transfer = (
        negative["dual_auroc"] <= -0.05
        or negative["objective_pearson"] <= -0.05
    )
    contract_primary = usefulness["checks"]["dominant_contract_error"]
    return {
        "random_window_leakage_gain": leak,
        "ranking": [
            {
                "factor": "route_shift",
                "level": "primary" if route_shift_primary else "secondary_or_unproven",
                "evidence": "random-window vs strict-route gap plus route/support identifiability",
            },
            {
                "factor": "route-stable_feature_representation",
                "level": "primary" if representation_primary else "secondary",
                "evidence": "strict-route ranking and learning-curve behavior despite candidate opportunity",
            },
            {
                "factor": "SCAND_route_sample_count",
                "level": "primary" if usefulness["verdict"] == "useful" else "secondary_or_not_supported",
                "evidence": "pre-registered route-subsampling curve",
            },
            {
                "factor": "RECON_to_SCAND_negative_transfer",
                "level": "secondary" if negative_transfer else "not_primary",
                "evidence": negative,
            },
            {
                "factor": "label_or_time_alignment_contract",
                "level": "primary" if contract_primary else "not_supported_as_primary",
                "evidence": "identity, metric recomputation, manifest fields, near-duplicate consistency",
            },
            {
                "factor": "model_capacity_or_calibration",
                "level": "secondary",
                "evidence": "same simple model compared under random-window and strict-route splits",
            },
            {
                "factor": "candidate_proposal_quality",
                "level": "primary" if proposal_primary else "not_primary",
                "evidence": candidate["pooled"],
            },
        ],
        "route_shift_primary": bool(route_shift_primary),
        "representation_primary": bool(representation_primary),
        "negative_transfer_detected": bool(negative_transfer),
        "proposal_quality_primary": bool(proposal_primary),
        "contract_error_primary": bool(contract_primary),
    }


def main() -> None:
    args = parse_args()
    result_path = OUTPUT / "result.json"
    derived_path = OUTPUT / "derived_predictions.npz"
    if result_path.exists() or derived_path.exists():
        raise FileExistsError("SCAND bottleneck diagnostic already exists")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    immutable = verify_contract()
    data, data_audit = v75.load_data()
    route_audit = data_audit["v74_data_audit"]["base_v72_data_audit"]["route_folds"]
    prereg = v72.read_json(PREREG)
    if route_audit["assignment_sha256"] != prereg["experiments"]["random_window_vs_route_oof"].get("assignment_sha256", route_audit["assignment_sha256"]):
        raise RuntimeError("route assignment changed")
    if route_audit["route_overlap_violations"]:
        raise RuntimeError("base route leakage")
    feature_names = data_audit["v75_feature_audit"]["relative_feature_names"]
    started = time.perf_counter()
    curve, strict_prediction = scand_learning_curve(data, device=args.device)
    random_cv = random_window_cv(data, device=args.device)
    matrix, matrix_predictions = domain_train_matrix(data, device=args.device)
    contract = contract_audit(data)
    candidate = candidate_and_route_audit(data, strict_prediction)
    signal = signal_and_error_audit(data, strict_prediction, feature_names)
    representation = representation_audit(data, feature_names, strict_prediction)
    usefulness = usefulness_decision(curve, contract)
    attribution = bottleneck_attribution(
        random_cv,
        curve,
        matrix,
        representation,
        candidate,
        contract,
        usefulness,
    )
    arrays: dict[str, np.ndarray] = {
        "record_key": data["record_key"],
        "fold": data["fold"],
        "domain": data["domain"],
        "strict_scand_only_dual_probability": strict_prediction["dual"],
        "strict_scand_only_harm_probability": strict_prediction["harm"],
        "strict_scand_only_objective_gain_prediction": strict_prediction["objective"],
    }
    for mode, prediction in matrix_predictions.items():
        for key, value in prediction.items():
            arrays[f"matrix_{mode}_{key}"] = value
    v72.write_npz(derived_path, **arrays)
    immutable_after = verify_contract()
    if immutable_after != immutable:
        raise RuntimeError("immutable input changed during diagnostic")
    result = {
        "schema": "scand_proposal_tail_bottleneck_diagnostic_result_v1",
        "status": "diagnostic_complete_wait_for_user_decision",
        "evidence_role": "spent_development_diagnostic_only",
        "states": int(len(data["record_key"])),
        "scand_states": int((data["domain"].astype(str) == "scand").sum()),
        "scand_routes": int(len(np.unique(data["group"][data["domain"].astype(str) == "scand"]))),
        "random_window_cv": random_cv,
        "strict_route_learning_curve": curve,
        "domain_train_matrix": matrix,
        "route_candidate_upper_bound": candidate,
        "signal_and_error_audit": signal,
        "representation_audit": representation,
        "contract_and_label_audit": contract,
        "independent_route_value": usefulness,
        "bottleneck_attribution": attribution,
        "unused_independent_route_inventory": {
            "performed": False,
            "reason": "inventory is allowed only after a useful verdict; no data are ingested automatically",
        },
        "elapsed_sec": time.perf_counter() - started,
        "device": args.device,
        "next_action": "stop_and_wait_for_user_decision",
        "audit": {
            "immutable_before": immutable,
            "immutable_after": immutable_after,
            "route_assignment_sha256": route_audit["assignment_sha256"],
            "strict_route_overlap_violations": 0,
            "random_window_overlap_is_intentional_diagnostic": True,
            "reserve_accessed": False,
            "new_data_ingested": False,
            "v70_cem_archive_mutated": False,
            "v76_trained": False,
            "derived_predictions_sha256": v72.sha256_file(derived_path),
        },
        "data_audit": data_audit,
    }
    v72.write_json(result_path, result)
    audit = {
        "schema": "scand_proposal_tail_bottleneck_diagnostic_audit_v1",
        "status": result["status"],
        "verdict": usefulness["verdict"],
        "bottleneck_attribution": attribution,
        "preregistration_sha256": immutable["preregistration_sha256"],
        "result_sha256": v72.sha256_file(result_path),
        "derived_predictions_sha256": v72.sha256_file(derived_path),
        "route_assignment_sha256": route_audit["assignment_sha256"],
        "route_overlap_violations": 0,
        "reserve_accessed": False,
        "new_data_ingested": False,
        "v76_trained": False,
    }
    v72.write_json(OUTPUT / "audit.json", audit)
    print(
        json.dumps(
            {
                "stage": "complete",
                "verdict": usefulness,
                "attribution": attribution,
                "elapsed_sec": result["elapsed_sec"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
