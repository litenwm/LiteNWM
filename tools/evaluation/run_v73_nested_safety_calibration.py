#!/usr/bin/env python3
"""Nested route-OOF domain-conditional safety calibration for frozen v72 rankers."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.evaluation import run_v72_proposal_tail_reranker as v72  # noqa: E402


OUTPUT = ROOT / "outputs/cem_development_v1/v73_nested_safety_calibration"
PREREG = OUTPUT / "preregistration.json"
V72_RESULT = ROOT / "outputs/cem_development_v1/v72_proposal_tail_reranker/result.json"
V72_PREDICTIONS = ROOT / "outputs/cem_development_v1/v72_proposal_tail_reranker/oof_predictions.npz"
V70_RESULT = ROOT / "outputs/cem_development_v1/dualdomain_cem_tail_v70_broad_hashfold_archive/result.json"
RANKERS = ("pointwise_ridge_core", "multitask_mlp_extended")
METHODS = ("domain_platt_wilson", "domain_conformal_risk")
REPORTED = (*METHODS, "nested_selected")
TOP_K = 5
Z = 1.96
TOLERANCE = 1.0e-7
PLATT_LEVELS = (
    (0.50, 0.50),
    (0.60, 0.40),
    (0.70, 0.30),
    (0.80, 0.20),
    (0.90, 0.10),
    (0.95, 0.05),
    (0.975, 0.025),
)
CONFORMAL_QUANTILES = (0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser.parse_args()


def verify_contract() -> dict[str, Any]:
    prereg = v72.read_json(PREREG)
    if prereg.get("status") != "frozen_before_v73_computation":
        raise RuntimeError("v73 preregistration is not frozen")
    if not prereg.get("reserve_forbidden") or prereg.get("reserve_accessed"):
        raise RuntimeError("v73 forbidden-cohort contract changed")
    expected = prereg["immutable_inputs"]
    checks = {
        "formal_v70_result_sha256": v72.sha256_file(V70_RESULT),
        "v72_result_sha256": v72.sha256_file(V72_RESULT),
        "v72_oof_predictions_sha256": v72.sha256_file(V72_PREDICTIONS),
    }
    for key, observed in checks.items():
        if observed != expected[key]:
            raise RuntimeError(f"immutable input changed: {key}")
    return {"preregistration_sha256": v72.sha256_file(PREREG), **checks}


def sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -30.0, 30.0)))


def logit(probability: np.ndarray) -> np.ndarray:
    value = np.clip(np.asarray(probability, dtype=np.float64), 1.0e-5, 1.0 - 1.0e-5)
    return np.log(value / (1.0 - value))


def fit_platt(probability: np.ndarray, target: np.ndarray) -> np.ndarray:
    x = logit(probability).reshape(-1)
    y = np.asarray(target, dtype=np.float64).reshape(-1)
    design = np.stack((x, np.ones_like(x)), axis=1)
    coefficient = np.asarray((1.0, 0.0), dtype=np.float64)
    regularization = np.diag((1.0e-3, 1.0e-6))
    for _ in range(60):
        prediction = sigmoid(design @ coefficient)
        weight = np.maximum(prediction * (1.0 - prediction), 1.0e-6)
        gradient = design.T @ (prediction - y) + regularization @ coefficient
        hessian = (design.T * weight) @ design + regularization
        step = np.linalg.solve(hessian, gradient)
        coefficient -= step
        if float(np.linalg.norm(step)) < 1.0e-8:
            break
    return coefficient.astype(np.float32)


def apply_platt(probability: np.ndarray, coefficient: np.ndarray) -> np.ndarray:
    return sigmoid(logit(probability) * float(coefficient[0]) + float(coefficient[1])).astype(np.float32)


def wilson(successes: int, total: int, z: float = Z) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 1.0
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    radius = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def inner_oof_cache_path(outer_fold: int) -> Path:
    return OUTPUT / "inner_oof" / f"outer_{outer_fold:02d}.npz"


def build_inner_oof(data: dict[str, Any], outer_fold: int, *, device: str) -> dict[str, np.ndarray]:
    path = inner_oof_cache_path(outer_fold)
    outer_train = np.flatnonzero(data["fold"] != outer_fold)
    if path.is_file():
        cached = v72.load_npz(path)
        if not (
            np.array_equal(cached["state_index"].astype(np.int64), outer_train)
            and np.array_equal(
                cached["record_key"].astype(str), data["record_key"][outer_train].astype(str)
            )
        ):
            raise RuntimeError(f"inner cache identity changed: outer {outer_fold}")
        return cached
    shape = (len(data["record_key"]), data["valid"].shape[1])
    prediction = {
        name: {
            key: np.full(shape, np.nan, dtype=np.float32)
            for key in ("score", "p_dual", "p_harm")
        }
        for name in RANKERS
    }
    records = []
    for inner_fold in range(5):
        if inner_fold == outer_fold:
            continue
        inner_train = np.flatnonzero(
            (data["fold"] != outer_fold) & (data["fold"] != inner_fold)
        )
        inner_valid = np.flatnonzero(data["fold"] == inner_fold)
        train_groups = set(data["group"][inner_train].astype(str))
        valid_groups = set(data["group"][inner_valid].astype(str))
        if train_groups & valid_groups:
            raise RuntimeError("inner route leakage")
        started = time.perf_counter()
        point = v72.fit_pointwise_ridge(data["core"], data, inner_train)
        safety = v72.fit_linear_safety(
            data["core"],
            data,
            inner_train,
            device=device,
            seed=20260773 + outer_fold * 10 + inner_fold,
        )
        prediction["pointwise_ridge_core"]["score"][inner_valid] = v72.predict_ridge(
            point, data["core"][inner_valid]
        )
        dual, harm = v72.predict_linear_safety(safety, data["core"][inner_valid])
        prediction["pointwise_ridge_core"]["p_dual"][inner_valid] = dual
        prediction["pointwise_ridge_core"]["p_harm"][inner_valid] = harm
        mlp = v72.fit_mlp(
            data["extended"],
            data,
            inner_train,
            device=device,
            seed=20261773 + outer_fold * 10 + inner_fold,
        )
        rank, dual, harm = v72.predict_mlp(
            mlp, data["extended"][inner_valid], device=device
        )
        prediction["multitask_mlp_extended"]["score"][inner_valid] = rank
        prediction["multitask_mlp_extended"]["p_dual"][inner_valid] = dual
        prediction["multitask_mlp_extended"]["p_harm"][inner_valid] = harm
        record = {
            "outer_fold": outer_fold,
            "inner_fold": inner_fold,
            "train_states": int(len(inner_train)),
            "valid_states": int(len(inner_valid)),
            "train_route_groups": int(len(train_groups)),
            "valid_route_groups": int(len(valid_groups)),
            "route_overlap": 0,
            "mlp_final_loss": float(mlp["epoch_audit"][-1]["mean_batch_loss"]),
            "elapsed_sec": time.perf_counter() - started,
        }
        records.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
    arrays: dict[str, np.ndarray] = {
        "state_index": outer_train,
        "record_key": data["record_key"][outer_train],
        "fold": data["fold"][outer_train],
        "audit_json": np.asarray(json.dumps(records, sort_keys=True)),
    }
    for name in RANKERS:
        for key in prediction[name]:
            local = prediction[name][key][outer_train]
            if not np.isfinite(local).all():
                raise RuntimeError(f"incomplete inner OOF: outer={outer_fold} {name}/{key}")
            arrays[f"{name}__{key}"] = local
    v72.write_npz(path, **arrays)
    return arrays


def expand_inner_prediction(
    data: dict[str, Any], cached: dict[str, np.ndarray], name: str
) -> dict[str, np.ndarray]:
    shape = data["valid"].shape
    result = {
        key: np.full(shape, np.nan, dtype=np.float32)
        for key in ("score", "p_dual", "p_harm")
    }
    index = cached["state_index"].astype(np.int64)
    for key in result:
        result[key][index] = cached[f"{name}__{key}"].astype(np.float32)
    return result


def challenger_topk(score: np.ndarray, valid: np.ndarray, formal: np.ndarray) -> np.ndarray:
    result = np.full((len(score), TOP_K), -1, dtype=np.int64)
    for state in range(len(score)):
        mask = valid[state].copy()
        mask[int(formal[state])] = False
        order = np.argsort(score[state], kind="stable")
        local = order[mask[order]][:TOP_K]
        result[state, : len(local)] = local
    return result


def calibrated_probabilities(
    prediction: dict[str, np.ndarray], policy: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    if policy["calibration_method"] == "domain_platt_wilson":
        return (
            apply_platt(prediction["p_dual"], np.asarray(policy["dual_platt"])),
            apply_platt(prediction["p_harm"], np.asarray(policy["harm_platt"])),
        )
    return prediction["p_dual"].astype(np.float32), prediction["p_harm"].astype(np.float32)


def apply_policy(
    data: dict[str, Any],
    state_index: np.ndarray,
    prediction: dict[str, np.ndarray],
    policy: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    formal = data["formal_index"][state_index].astype(np.int64)
    if not policy.get("active"):
        probability_dual, probability_harm = calibrated_probabilities(
            {key: prediction[key][state_index] for key in prediction}, policy
        )
        return formal.copy(), probability_dual, probability_harm, challenger_topk(
            prediction["score"][state_index], data["valid"][state_index], formal
        )
    local_prediction = {key: prediction[key][state_index] for key in prediction}
    probability_dual, probability_harm = calibrated_probabilities(local_prediction, policy)
    top = challenger_topk(local_prediction["score"], data["valid"][state_index], formal)
    selected = formal.copy()
    for row in range(len(state_index)):
        for candidate in top[row]:
            if candidate < 0:
                continue
            if policy["calibration_method"] == "domain_platt_wilson":
                passed = (
                    probability_dual[row, candidate] >= policy["dual_threshold"]
                    and probability_harm[row, candidate] <= policy["harm_threshold"]
                )
            else:
                risk = max(
                    1.0 - float(probability_dual[row, candidate]),
                    float(probability_harm[row, candidate]),
                )
                passed = risk <= policy["risk_threshold"]
            if passed:
                selected[row] = candidate
                break
    return selected, probability_dual, probability_harm, top


def incremental_metrics(
    data: dict[str, Any], state_index: np.ndarray, selected: np.ndarray
) -> dict[str, Any]:
    formal = data["formal_index"][state_index]
    replacement = selected != formal
    row = np.arange(len(state_index))
    selected_metric = data["true_metrics"][state_index, selected]
    formal_metric = data["true_metrics"][state_index, formal]
    dual = replacement & (
        data["dual_vs_formal"][state_index, selected].astype(bool)
    )
    harm = replacement & (
        data["harm_vs_formal"][state_index, selected].astype(bool)
    )
    count = int(replacement.sum())
    dual_count = int(dual.sum())
    harm_count = int(harm.sum())
    dual_interval = wilson(dual_count, count)
    harm_interval = wilson(harm_count, count)
    return {
        "states": int(len(state_index)),
        "replacements": count,
        "acceptance_rate": float(replacement.mean()),
        "dual_count": dual_count,
        "dual_precision": float(dual_count / count) if count else None,
        "dual_wilson_95": {"lower": dual_interval[0], "upper": dual_interval[1]},
        "harm_count": harm_count,
        "harm_rate": float(harm_count / count) if count else None,
        "harm_wilson_95": {"lower": harm_interval[0], "upper": harm_interval[1]},
        "mean_objective_gain_m": float(
            formal_metric[:, 3].mean() - selected_metric[:, 3].mean()
        ),
        "mean_ate_gain_m": float(
            formal_metric[:, 0].mean() - selected_metric[:, 0].mean()
        ),
    }


def candidate_calibration_metrics(
    probability: np.ndarray, target: np.ndarray, mask: np.ndarray
) -> dict[str, Any]:
    p = np.asarray(probability[mask], dtype=np.float64)
    y = np.asarray(target[mask], dtype=np.float64)
    edges = np.linspace(0.0, 1.0, 11)
    curve = []
    ece = 0.0
    for index in range(10):
        local = (p >= edges[index]) & (
            p <= edges[index + 1] if index == 9 else p < edges[index + 1]
        )
        count = int(local.sum())
        if count:
            predicted = float(p[local].mean())
            observed = float(y[local].mean())
            ece += count / len(p) * abs(predicted - observed)
        else:
            predicted = observed = None
        curve.append({
            "lower": float(edges[index]),
            "upper": float(edges[index + 1]),
            "count": count,
            "mean_probability": predicted,
            "observed_rate": observed,
        })
    return {
        "rows": int(len(p)),
        "brier": float(np.mean((p - y) ** 2)),
        "ece_10_equal_width": float(ece),
        "curve": curve,
    }


def base_policy_record(
    data: dict[str, Any],
    state_index: np.ndarray,
    prediction: dict[str, np.ndarray],
    method: str,
    model: str,
) -> dict[str, Any]:
    mask = data["valid"][state_index].copy()
    row = np.arange(len(state_index))
    mask[row, data["formal_index"][state_index]] = False
    if method == "domain_platt_wilson":
        dual_platt = fit_platt(
            prediction["p_dual"][state_index][mask],
            data["dual_vs_formal"][state_index][mask],
        )
        harm_platt = fit_platt(
            prediction["p_harm"][state_index][mask],
            data["harm_vs_formal"][state_index][mask],
        )
        return {
            "calibration_method": method,
            "ranker": model,
            "dual_platt": dual_platt.tolist(),
            "harm_platt": harm_platt.tolist(),
            "active": False,
        }
    return {
        "calibration_method": method,
        "ranker": model,
        "active": False,
    }


def fit_policy_candidate(
    data: dict[str, Any],
    state_index: np.ndarray,
    prediction: dict[str, np.ndarray],
    method: str,
    model: str,
) -> dict[str, Any]:
    base = base_policy_record(data, state_index, prediction, method, model)
    local_prediction = {key: prediction[key][state_index] for key in prediction}
    probability_dual, probability_harm = calibrated_probabilities(local_prediction, base)
    mask = data["valid"][state_index].copy()
    row = np.arange(len(state_index))
    mask[row, data["formal_index"][state_index]] = False
    if method == "domain_platt_wilson":
        levels = [
            {"dual_threshold": dual, "harm_threshold": harm, "strictness": index}
            for index, (dual, harm) in enumerate(PLATT_LEVELS)
        ]
    else:
        risk = np.maximum(1.0 - probability_dual, probability_harm)
        levels = []
        for index, quantile in enumerate(CONFORMAL_QUANTILES):
            threshold = float(np.quantile(risk[mask], quantile, method="higher"))
            levels.append({
                "risk_threshold": threshold,
                "risk_quantile": quantile,
                "strictness": len(CONFORMAL_QUANTILES) - index,
            })
    evaluations = []
    for level in levels:
        policy = {**base, **level, "active": True}
        selected, _, _, _ = apply_policy(data, state_index, prediction, policy)
        metrics = incremental_metrics(data, state_index, selected)
        feasible = (
            metrics["replacements"] > 0
            and metrics["dual_wilson_95"]["lower"] >= 0.85
            and metrics["harm_wilson_95"]["upper"] <= 0.15
            and metrics["mean_objective_gain_m"] > 0.0
        )
        evaluations.append({"parameters": level, "metrics": metrics, "feasible": feasible})
    feasible = [value for value in evaluations if value["feasible"]]
    pool = feasible if feasible else evaluations
    best = max(
        pool,
        key=lambda value: (
            value["metrics"]["mean_objective_gain_m"],
            value["metrics"]["replacements"],
            value["parameters"]["strictness"],
        ),
    )
    policy = {
        **base,
        **best["parameters"],
        "active": bool(best["feasible"]),
        "inner_metrics": best["metrics"],
        "all_levels": evaluations,
    }
    before_dual = candidate_calibration_metrics(
        local_prediction["p_dual"],
        data["dual_vs_formal"][state_index],
        mask,
    )
    before_harm = candidate_calibration_metrics(
        local_prediction["p_harm"],
        data["harm_vs_formal"][state_index],
        mask,
    )
    policy["inner_calibration"] = {
        "dual_before": before_dual,
        "harm_before": before_harm,
        "dual_after": candidate_calibration_metrics(
            probability_dual,
            data["dual_vs_formal"][state_index],
            mask,
        ),
        "harm_after": candidate_calibration_metrics(
            probability_harm,
            data["harm_vs_formal"][state_index],
            mask,
        ),
    }
    return policy


def choose_policy(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    feasible = [policy for policy in candidates if policy.get("active")]
    if feasible:
        return max(
            feasible,
            key=lambda policy: (
                policy["inner_metrics"]["mean_objective_gain_m"],
                policy["inner_metrics"]["replacements"],
                policy["calibration_method"] == "domain_platt_wilson",
                policy["ranker"] == "pointwise_ridge_core",
            ),
        )
    diagnostic = max(
        candidates,
        key=lambda policy: (
            policy["inner_metrics"]["dual_wilson_95"]["lower"]
            - policy["inner_metrics"]["harm_wilson_95"]["upper"],
            policy["inner_metrics"]["mean_objective_gain_m"],
        ),
    )
    return {**diagnostic, "active": False, "fallback_reason": "no_inner_wilson_feasible_policy"}


def compact_policy(policy: dict[str, Any]) -> dict[str, Any]:
    keep = (
        "calibration_method",
        "ranker",
        "active",
        "dual_platt",
        "harm_platt",
        "dual_threshold",
        "harm_threshold",
        "risk_threshold",
        "risk_quantile",
        "inner_metrics",
        "fallback_reason",
    )
    return {key: policy[key] for key in keep if key in policy}


def method_success_gate(
    summaries: dict[str, Any], formal: dict[str, Any]
) -> dict[str, Any]:
    pooled = summaries["pooled"]
    recon = summaries["recon_a_to_h"]
    scand = summaries["fresh_scand"]
    gain = formal["pooled"]["absolute_mean_m"]["objective"] - pooled["absolute_mean_m"]["objective"]
    checks = {
        "recon_mean_ate_non_worse": recon["absolute_mean_m"]["ate"] <= formal["recon_a_to_h"]["absolute_mean_m"]["ate"] + TOLERANCE,
        "recon_mean_objective_non_worse": recon["absolute_mean_m"]["objective"] <= formal["recon_a_to_h"]["absolute_mean_m"]["objective"] + TOLERANCE,
        "scand_mean_ate_non_worse": scand["absolute_mean_m"]["ate"] <= formal["fresh_scand"]["absolute_mean_m"]["ate"] + TOLERANCE,
        "scand_mean_objective_non_worse": scand["absolute_mean_m"]["objective"] <= formal["fresh_scand"]["absolute_mean_m"]["objective"] + TOLERANCE,
        "scand_objective_p90_within_1mm": scand["absolute_p90_m"]["objective"] <= formal["fresh_scand"]["absolute_p90_m"]["objective"] + 0.001,
        "incremental_dual_at_least_0_85": summaries["pooled"]["dual_precision_vs_formal_v70"] is not None and summaries["pooled"]["dual_precision_vs_formal_v70"] >= 0.85,
        "incremental_harm_at_most_0_15": summaries["pooled"]["harm_rate_vs_formal_v70"] is not None and summaries["pooled"]["harm_rate_vs_formal_v70"] <= 0.15,
        "pooled_objective_gain_at_least_1mm": gain >= 0.001,
    }
    return {
        "checks": checks,
        "passed": bool(all(checks.values())),
        "pooled_mean_objective_gain_m": float(gain),
        "preferred_4mm_gain_met": bool(gain >= 0.004),
    }


def outer_calibration_report(
    data: dict[str, Any],
    probability_dual: np.ndarray,
    probability_harm: np.ndarray,
) -> dict[str, Any]:
    report = {}
    for domain, key in (("recon", "recon_a_to_h"), ("scand", "fresh_scand")):
        states = data["domain"].astype(str) == domain
        mask = states[:, None] & data["valid"] & np.isfinite(probability_dual) & np.isfinite(probability_harm)
        row = np.arange(len(data["record_key"]))
        mask[row, data["formal_index"]] = False
        report[key] = {
            "dual": candidate_calibration_metrics(
                probability_dual, data["dual_vs_formal"], mask
            ),
            "harm": candidate_calibration_metrics(
                probability_harm, data["harm_vs_formal"], mask
            ),
        }
    return report


def selection_errors(
    data: dict[str, Any],
    selected: np.ndarray,
    topk: np.ndarray,
) -> dict[str, Any]:
    replacement = selected != data["formal_index"]
    row = np.arange(len(selected))
    selected_dual = data["dual_vs_formal"][row, selected].astype(bool)
    selected_harm = data["harm_vs_formal"][row, selected].astype(bool)
    opportunity = np.zeros(len(selected), dtype=bool)
    for state in range(len(selected)):
        local = topk[state]
        local = local[local >= 0]
        opportunity[state] = bool(data["dual_vs_formal"][state, local].astype(bool).any())
    false_reject = (~replacement) & opportunity
    return {
        "accepted_replacements": int(replacement.sum()),
        "nondual_accepts": int((replacement & ~selected_dual).sum()),
        "harmful_false_accepts": int((replacement & selected_harm).sum()),
        "fallback_states": int((~replacement).sum()),
        "fallback_with_true_dual_top5_opportunity": int(false_reject.sum()),
        "false_reject_rate_given_top5_opportunity": float(false_reject.sum() / max(int(opportunity.sum()), 1)),
        "states_with_true_dual_top5_opportunity": int(opportunity.sum()),
    }


def calibration_latency_ms(
    data: dict[str, Any],
    outer_prediction: dict[str, dict[str, np.ndarray]],
    policies: dict[tuple[int, str], dict[str, Any]],
) -> dict[str, float]:
    measurements = []
    for outer_fold in range(5):
        for domain in ("recon", "scand"):
            states = np.flatnonzero(
                (data["fold"] == outer_fold) & (data["domain"].astype(str) == domain)
            )
            if not len(states):
                continue
            policy = policies[(outer_fold, domain)]
            prediction = outer_prediction[policy["ranker"]]
            for state in states:
                local_index = np.asarray((state,), dtype=np.int64)
                started = time.perf_counter_ns()
                apply_policy(data, local_index, prediction, policy)
                measurements.append((time.perf_counter_ns() - started) / 1.0e6)
    value = np.asarray(measurements, dtype=np.float64)
    return {
        "mean": float(value.mean()),
        "p50": float(np.quantile(value, 0.5)),
        "p90": float(np.quantile(value, 0.9)),
        "states": int(len(value)),
        "scope": "v73 probability calibration plus top-5 safety selection; frozen v70/v72 inference excluded",
    }


def main() -> None:
    args = parse_args()
    result_path = OUTPUT / "result.json"
    prediction_path = OUTPUT / "outer_oof_predictions.npz"
    if result_path.exists() or prediction_path.exists():
        raise FileExistsError("v73 result already exists")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    immutable = verify_contract()
    data, data_audit = v72.load_data()
    if data_audit["route_folds"]["assignment_sha256"] != v72.read_json(PREREG)["nested_route_oof"]["outer_assignment_sha256"]:
        raise RuntimeError("outer route assignment changed")
    frozen = v72.load_npz(V72_PREDICTIONS)
    if not (
        np.array_equal(frozen["record_key"].astype(str), data["record_key"].astype(str))
        and np.array_equal(frozen["fold"].astype(np.int64), data["fold"])
    ):
        raise RuntimeError("v72 outer OOF identity changed")
    outer_prediction = {
        name: {
            key: frozen[f"{name}__{key}"].astype(np.float32)
            for key in ("score", "p_dual", "p_harm")
        }
        for name in RANKERS
    }
    selected = {
        name: data["formal_index"].copy().astype(np.int64) for name in REPORTED
    }
    topk = {
        name: np.full((len(data["record_key"]), TOP_K), -1, dtype=np.int64)
        for name in REPORTED
    }
    probability_dual = {
        name: np.full(data["valid"].shape, np.nan, dtype=np.float32) for name in REPORTED
    }
    probability_harm = {
        name: np.full(data["valid"].shape, np.nan, dtype=np.float32) for name in REPORTED
    }
    chosen_policies: dict[str, dict[tuple[int, str], dict[str, Any]]] = {
        name: {} for name in REPORTED
    }
    policy_audit = []
    inner_hashes = {}
    started = time.perf_counter()
    for outer_fold in range(5):
        cached = build_inner_oof(data, outer_fold, device=args.device)
        cache_path = inner_oof_cache_path(outer_fold)
        inner_hashes[f"outer_{outer_fold}"] = v72.sha256_file(cache_path)
        inner_prediction = {
            name: expand_inner_prediction(data, cached, name) for name in RANKERS
        }
        for domain in ("recon", "scand"):
            inner_states = np.flatnonzero(
                (data["fold"] != outer_fold) & (data["domain"].astype(str) == domain)
            )
            outer_states = np.flatnonzero(
                (data["fold"] == outer_fold) & (data["domain"].astype(str) == domain)
            )
            all_candidates: dict[str, list[dict[str, Any]]] = {method: [] for method in METHODS}
            for method in METHODS:
                for model in RANKERS:
                    all_candidates[method].append(
                        fit_policy_candidate(
                            data,
                            inner_states,
                            inner_prediction[model],
                            method,
                            model,
                        )
                    )
            standalone = {
                method: choose_policy(all_candidates[method]) for method in METHODS
            }
            nested = choose_policy(
                [policy for method in METHODS for policy in all_candidates[method]]
            )
            local_policies = {**standalone, "nested_selected": nested}
            for reported, policy in local_policies.items():
                chosen_policies[reported][(outer_fold, domain)] = policy
                model_prediction = outer_prediction[policy["ranker"]]
                local_selected, local_dual, local_harm, local_topk = apply_policy(
                    data, outer_states, model_prediction, policy
                )
                selected[reported][outer_states] = local_selected
                topk[reported][outer_states] = local_topk
                probability_dual[reported][outer_states] = local_dual
                probability_harm[reported][outer_states] = local_harm
            policy_audit.append({
                "outer_fold": outer_fold,
                "domain": domain,
                "inner_states": int(len(inner_states)),
                "outer_states": int(len(outer_states)),
                "outer_truth_used_for_policy": False,
                "candidate_policies": {
                    method: [compact_policy(policy) for policy in values]
                    for method, values in all_candidates.items()
                },
                "selected_policies": {
                    name: compact_policy(policy) for name, policy in local_policies.items()
                },
            })
            print(json.dumps({
                "stage": "outer_policy",
                "outer_fold": outer_fold,
                "domain": domain,
                "nested": compact_policy(nested),
            }, sort_keys=True), flush=True)
    formal = v72.formal_indices_summary(data)
    noncem = v72.noncem_indices_summary(data)
    method_results = {}
    for name in REPORTED:
        summaries = v72.selection_summaries(data, selected[name])
        method_results[name] = {
            "selection": summaries,
            "success_gate": method_success_gate(summaries, formal),
            "comparison_vs_formal_v70": {
                key: v72.method_delta(summaries[key], formal[key]) for key in summaries
            },
            "comparison_vs_noncem": {
                key: v72.method_delta(summaries[key], noncem[key]) for key in summaries
            },
            "selection_errors": selection_errors(data, selected[name], topk[name]),
            "outer_calibration": outer_calibration_report(
                data, probability_dual[name], probability_harm[name]
            ),
            "active_policy_counts": {
                "active": int(sum(policy.get("active", False) for policy in chosen_policies[name].values())),
                "fallback_only": int(sum(not policy.get("active", False) for policy in chosen_policies[name].values())),
                "rankers": {
                    model: int(sum(policy["ranker"] == model for policy in chosen_policies[name].values()))
                    for model in RANKERS
                },
                "calibration_methods": {
                    method: int(sum(policy["calibration_method"] == method for policy in chosen_policies[name].values()))
                    for method in METHODS
                },
            },
        }
    latency = calibration_latency_ms(
        data, outer_prediction, chosen_policies["nested_selected"]
    )
    output_arrays: dict[str, np.ndarray] = {
        "record_key": data["record_key"],
        "fold": data["fold"],
        "formal_index": data["formal_index"],
    }
    for name in REPORTED:
        output_arrays[f"{name}__selected_index"] = selected[name]
        output_arrays[f"{name}__topk"] = topk[name]
        output_arrays[f"{name}__calibrated_dual"] = probability_dual[name]
        output_arrays[f"{name}__calibrated_harm"] = probability_harm[name]
    v72.write_npz(prediction_path, **output_arrays)
    nested_gate = method_results["nested_selected"]["success_gate"]
    result = {
        "schema": "v73_nested_route_oof_domain_safety_calibration_v1",
        "status": "promising_spent_development_result" if nested_gate["passed"] else "development_gate_not_met",
        "success_method": "nested_selected",
        "success_gate": nested_gate,
        "methods": method_results,
        "baselines": {
            "formal_v70": formal,
            "v70_fixed_noncem": noncem,
        },
        "policy_audit": policy_audit,
        "inner_oof_cache_hashes": inner_hashes,
        "outer_oof_predictions": str(prediction_path),
        "outer_oof_predictions_sha256": v72.sha256_file(prediction_path),
        "cpu_added_latency_ms_per_state": latency,
        "immutable_input_audit": immutable,
        "data_audit": data_audit,
        "elapsed_sec": time.perf_counter() - started,
        "reserve_accessed": False,
        "formal_v70_mutated": False,
        "v72_mutated": False,
        "cem_or_candidate_changed": False,
        "outer_truth_used_for_policy_or_threshold": False,
        "production_safe_claim_allowed": False,
        "independent_confirmation_required": True,
        "limitations": [
            "All cohorts are already-spent development cohorts.",
            "The outer route folds estimate development generalization but are not a new independent cohort.",
            "Any passing policy remains development-only until evaluated once on a future untouched cohort.",
        ],
    }
    v72.write_json(result_path, result)
    v72.write_json(OUTPUT / "audit.json", {
        "result_sha256": v72.sha256_file(result_path),
        "outer_oof_predictions_sha256": v72.sha256_file(prediction_path),
        "preregistration_sha256": v72.sha256_file(PREREG),
        "inner_oof_cache_hashes": inner_hashes,
        "reserve_accessed": False,
        "formal_v70_mutated": False,
        "v72_mutated": False,
        "cem_or_candidate_changed": False,
        "outer_truth_used_for_policy_or_threshold": False,
    })
    print(json.dumps({
        "status": result["status"],
        "success_gate": nested_gate,
        "cpu_added_latency_ms_per_state": latency,
        "result": str(result_path),
    }, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
