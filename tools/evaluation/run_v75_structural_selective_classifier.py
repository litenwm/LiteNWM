#!/usr/bin/env python3
"""Final spent-development structural dual/harm classifier diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.evaluation import run_v72_proposal_tail_reranker as v72  # noqa: E402
from tools.evaluation import run_v73_nested_safety_calibration as v73  # noqa: E402
from tools.evaluation import run_v74_direct_delta_nested as v74  # noqa: E402


OUTPUT = ROOT / "outputs/cem_development_v1/v75_structural_selective_classifier"
PREREG = OUTPUT / "preregistration.json"
V74_PREREG = ROOT / "outputs/cem_development_v1/v74_direct_delta_nested/preregistration.json"
V74_RESULT = ROOT / "outputs/cem_development_v1/v74_direct_delta_nested/result.json"
V74_PREDICTIONS = ROOT / "outputs/cem_development_v1/v74_direct_delta_nested/outer_oof_predictions.npz"
V74_LEVEL_AUDIT = ROOT / "outputs/cem_development_v1/v74_direct_delta_nested/inner_level_audit.json"
METHODS = ("balanced_logistic_relative", "focal_mlp_structural")
REPORTED = (*METHODS, "nested_selected")
LEVELS = (
    (0.50, 0.50),
    (0.60, 0.40),
    (0.70, 0.30),
    (0.80, 0.20),
    (0.90, 0.10),
    (0.95, 0.05),
    (0.975, 0.025),
)
TOLERANCE = 1.0e-7


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser.parse_args()


def verify_contract() -> dict[str, str]:
    prereg = v72.read_json(PREREG)
    if prereg.get("status") != "frozen_before_v75_computation":
        raise RuntimeError("v75 preregistration is not frozen")
    if not prereg["reserve"]["forbidden"] or prereg["reserve"]["accessed_by_v75"]:
        raise RuntimeError("v75 forbidden-cohort contract changed")
    paths = {
        "formal_v70_result_sha256": v74.V70_RESULT,
        "v74_preregistration_sha256": V74_PREREG,
        "v74_result_sha256": V74_RESULT,
        "v74_outer_oof_predictions_sha256": V74_PREDICTIONS,
        "v74_inner_level_audit_sha256": V74_LEVEL_AUDIT,
    }
    observed = {key: v72.sha256_file(path) for key, path in paths.items()}
    for key, digest in observed.items():
        if digest != prereg["immutable_inputs"][key]:
            raise RuntimeError(f"immutable input changed: {key}")
    return {"preregistration_sha256": v72.sha256_file(PREREG), **observed}


def feature_indices(names: list[str]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    disagreement = {
        "seed_ate_std",
        "seed_ate_range",
        "seed_rtp_std",
        "seed_rtp_range",
        "seed_support_std",
        "seed_support_range",
    }
    relative = []
    extended = []
    for index, name in enumerate(names):
        allowed_relative = (
            "delta" in name
            or name in disagreement
            or name in {
                "source_rank",
                "scorer_rank",
                "energy_rank",
                "domain_recon",
                "domain_scand",
                "valid",
                "safe",
                "v70_score_available",
                "formal_selected",
            }
            or name.startswith("source_")
            or name.startswith("local_density_")
        )
        if allowed_relative:
            relative.append(index)
        if allowed_relative or name.startswith("v70_extension_abs_"):
            extended.append(index)
    relative_value = np.asarray(sorted(set(relative)), dtype=np.int64)
    extended_value = np.asarray(sorted(set(extended)), dtype=np.int64)
    selected_names = [names[index] for index in extended_value]
    forbidden_tokens = ("route", "record", "true_", "predicted_gain")
    if any(any(token in name for token in forbidden_tokens) for name in selected_names):
        raise RuntimeError("forbidden identity/truth feature selected")
    return relative_value, extended_value, {
        "relative_dimension": int(len(relative_value)),
        "extended_dimension": int(len(extended_value)),
        "relative_feature_names": [names[index] for index in relative_value],
        "extended_feature_names": selected_names,
        "relative_names_sha256": hashlib.sha256("\n".join(names[index] for index in relative_value).encode()).hexdigest(),
        "extended_names_sha256": hashlib.sha256("\n".join(selected_names).encode()).hexdigest(),
        "route_or_record_identity_used": False,
        "truth_or_prior_prediction_used": False,
    }


def load_data() -> tuple[dict[str, Any], dict[str, Any]]:
    data, audit = v74.load_data()
    relative, extended, feature_audit = feature_indices(audit["compact_feature_names"])
    data["v75_relative"] = data["v74_compact"][..., relative].astype(np.float32)
    data["v75_extended"] = data["v74_compact"][..., extended].astype(np.float32)
    if not np.isfinite(data["v75_relative"]).all() or not np.isfinite(data["v75_extended"]).all():
        raise RuntimeError("non-finite v75 features")
    return data, {"v74_data_audit": audit, "v75_feature_audit": feature_audit}


def train_mask(data: dict[str, Any], states: np.ndarray) -> np.ndarray:
    state = np.zeros(len(data["record_key"]), dtype=bool)
    state[states] = True
    mask = state[:, None] & data["valid"] & data["safe"]
    mask[np.arange(len(state)), data["formal_index"]] = False
    return mask


class LinearClassifier(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.linear = nn.Linear(dimension, 2)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.linear(value)


class StructuralMLP(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(dimension, 64),
            nn.SiLU(),
            nn.Linear(64, 32),
            nn.SiLU(),
            nn.Linear(32, 2),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.network(value)


def fit_member(
    features: np.ndarray,
    data: dict[str, Any],
    states: np.ndarray,
    norm: dict[str, np.ndarray],
    *,
    kind: str,
    device: str,
    seed: int,
) -> tuple[nn.Module, list[float], dict[str, float]]:
    mask = train_mask(data, states)
    x = torch.from_numpy(v72.normalize(features, norm)[mask]).to(device)
    target_np = np.stack((data["dual_vs_formal"][mask], data["harm_vs_formal"][mask]), axis=-1).astype(np.float32)
    y = torch.from_numpy(target_np).to(device)
    positive = np.maximum(target_np.sum(axis=0), 1.0)
    negative = np.maximum(len(target_np) - positive, 1.0)
    pos_weight = torch.from_numpy((negative / positive).astype(np.float32)).to(device)
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
    if kind == "linear":
        model: nn.Module = LinearClassifier(x.shape[-1]).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3.0e-3, weight_decay=1.0e-2)
        epochs = 32
    else:
        model = StructuralMLP(x.shape[-1]).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3, weight_decay=1.0e-4)
        epochs = 24
    generator = torch.Generator(device=device).manual_seed(seed + 100)
    losses = []
    for _ in range(epochs):
        order = torch.randperm(len(x), generator=generator, device=device)
        total = 0.0
        seen = 0
        for start in range(0, len(order), 4096):
            index = order[start : start + 4096]
            logits = model(x[index])
            bce = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, y[index], pos_weight=pos_weight, reduction="none"
            )
            if kind == "focal":
                probability = torch.sigmoid(logits)
                pt = y[index] * probability + (1.0 - y[index]) * (1.0 - probability)
                bce = bce * (1.0 - pt).pow(2.0)
            loss = bce.mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(index)
            seen += len(index)
        losses.append(total / seen)
    return model.eval(), losses, {
        "dual_positive_fraction": float(target_np[:, 0].mean()),
        "harm_positive_fraction": float(target_np[:, 1].mean()),
        "dual_pos_weight": float(negative[0] / positive[0]),
        "harm_pos_weight": float(negative[1] / positive[1]),
    }


def predict_members(
    models: list[nn.Module], features: np.ndarray, norm: dict[str, np.ndarray], *, device: str
) -> tuple[np.ndarray, np.ndarray]:
    value = v72.normalize(features, norm)
    outputs = []
    for model in models:
        chunks = []
        with torch.inference_mode():
            for start in range(0, len(value), 128):
                tensor = torch.from_numpy(value[start : start + 128]).to(device)
                chunks.append(torch.sigmoid(model(tensor)).cpu().numpy().astype(np.float32))
        outputs.append(np.concatenate(chunks))
    stack = np.stack(outputs)
    return stack.mean(axis=0).astype(np.float32), stack.std(axis=0).astype(np.float32)


def fit_predict_method(
    method: str,
    data: dict[str, Any],
    train_states: np.ndarray,
    predict_states: np.ndarray,
    *,
    device: str,
    seed_offset: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    started = time.perf_counter()
    if method == "balanced_logistic_relative":
        features = data["v75_relative"]
        kind = "linear"
        seeds = (7501 + seed_offset,)
    else:
        features = data["v75_extended"]
        kind = "focal"
        seeds = tuple(value + seed_offset for value in (7511, 7512, 7513))
    mask = train_mask(data, train_states)
    norm = v72.fit_normalizer(features, mask)
    models = []
    losses = []
    balance = None
    for seed in seeds:
        model, history, local_balance = fit_member(
            features,
            data,
            train_states,
            norm,
            kind=kind,
            device=device,
            seed=seed,
        )
        models.append(model)
        losses.append(history[-1])
        balance = local_balance
    mean, std = predict_members(models, features[predict_states], norm, device=device)
    del models
    if device == "cuda":
        torch.cuda.empty_cache()
    elapsed = time.perf_counter() - started
    return mean, std, {
        "method": method,
        "train_states": int(len(train_states)),
        "predict_states": int(len(predict_states)),
        "training_rows": int(mask.sum()),
        "elapsed_sec": elapsed,
        "fit_plus_batched_predict_ms_per_predicted_state": 1000.0 * elapsed / max(len(predict_states), 1),
        "final_member_losses": losses,
        "class_balance": balance,
    }


def cache_path(outer: int) -> Path:
    return OUTPUT / "nested_cache" / f"outer_{outer:02d}.npz"


def build_cache(data: dict[str, Any], outer: int, *, device: str) -> dict[str, np.ndarray]:
    path = cache_path(outer)
    outer_train = np.flatnonzero(data["fold"] != outer)
    outer_test = np.flatnonzero(data["fold"] == outer)
    if path.is_file():
        cached = v72.load_npz(path)
        if not (
            np.array_equal(cached["inner_state_index"], outer_train)
            and np.array_equal(cached["outer_state_index"], outer_test)
            and np.array_equal(cached["inner_record_key"].astype(str), data["record_key"][outer_train].astype(str))
            and np.array_equal(cached["outer_record_key"].astype(str), data["record_key"][outer_test].astype(str))
        ):
            raise RuntimeError(f"v75 cache identity changed: outer={outer}")
        return cached
    inner_mean = {method: np.full((*data["valid"].shape, 2), np.nan, dtype=np.float32) for method in METHODS}
    inner_std = {method: np.full((*data["valid"].shape, 2), np.nan, dtype=np.float32) for method in METHODS}
    inner_audit = []
    for inner in range(5):
        if inner == outer:
            continue
        train = np.flatnonzero((data["fold"] != outer) & (data["fold"] != inner))
        valid = np.flatnonzero(data["fold"] == inner)
        if set(data["group"][train].astype(str)) & set(data["group"][valid].astype(str)):
            raise RuntimeError("inner route leakage")
        for method_index, method in enumerate(METHODS):
            mean, std, record = fit_predict_method(
                method,
                data,
                train,
                valid,
                device=device,
                seed_offset=outer * 100 + inner * 10 + method_index,
            )
            inner_mean[method][valid] = mean
            inner_std[method][valid] = std
            inner_audit.append({"outer_fold": outer, "inner_fold": inner, "route_overlap": 0, **record})
            print(json.dumps({"stage": "inner_fit", **inner_audit[-1]}, sort_keys=True), flush=True)
    outer_audit = []
    outer_mean = {}
    outer_std = {}
    for method_index, method in enumerate(METHODS):
        mean, std, record = fit_predict_method(
            method,
            data,
            outer_train,
            outer_test,
            device=device,
            seed_offset=outer * 1000 + method_index,
        )
        outer_mean[method] = mean
        outer_std[method] = std
        outer_audit.append({"outer_fold": outer, "route_overlap": 0, **record})
        print(json.dumps({"stage": "outer_fit", **outer_audit[-1]}, sort_keys=True), flush=True)
    arrays: dict[str, np.ndarray] = {
        "inner_state_index": outer_train,
        "outer_state_index": outer_test,
        "inner_record_key": data["record_key"][outer_train],
        "outer_record_key": data["record_key"][outer_test],
        "audit_json": np.asarray(json.dumps({"inner": inner_audit, "outer": outer_audit}, sort_keys=True)),
    }
    for method in METHODS:
        local_mean = inner_mean[method][outer_train]
        local_std = inner_std[method][outer_train]
        if not np.isfinite(local_mean).all() or not np.isfinite(local_std).all():
            raise RuntimeError(f"incomplete v75 inner OOF: outer={outer} method={method}")
        arrays[f"{method}__inner_probability"] = local_mean
        arrays[f"{method}__inner_std"] = local_std
        arrays[f"{method}__outer_probability"] = outer_mean[method]
        arrays[f"{method}__outer_std"] = outer_std[method]
    v72.write_npz(path, **arrays)
    return arrays


def expand_inner(data: dict[str, Any], cached: dict[str, np.ndarray], method: str) -> np.ndarray:
    result = np.full((*data["valid"].shape, 2), np.nan, dtype=np.float32)
    result[cached["inner_state_index"].astype(np.int64)] = cached[f"{method}__inner_probability"]
    return result


def calibrated(probability: np.ndarray, policy: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    return (
        v73.apply_platt(probability[..., 0], np.asarray(policy["dual_platt"], dtype=np.float32)),
        v73.apply_platt(probability[..., 1], np.asarray(policy["harm_platt"], dtype=np.float32)),
    )


def apply_policy(
    data: dict[str, Any], states: np.ndarray, probability: np.ndarray, policy: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    formal = data["formal_index"][states].astype(np.int64)
    selected = formal.copy()
    dual, harm = calibrated(probability, policy)
    eligible = data["valid"][states] & data["safe"][states]
    eligible[np.arange(len(states)), formal] = False
    passing = eligible & (dual >= float(policy["dual_threshold"])) & (harm <= float(policy["harm_threshold"]))
    score = dual - harm
    if policy.get("active"):
        for row in range(len(states)):
            index = np.flatnonzero(passing[row])
            if len(index):
                selected[row] = int(index[np.argmax(score[row, index])])
    return selected, dual, harm, passing


def fit_policy(
    data: dict[str, Any], states: np.ndarray, probability: np.ndarray, method: str
) -> dict[str, Any]:
    local = probability[states]
    formal = data["formal_index"][states]
    mask = data["valid"][states] & data["safe"][states]
    mask[np.arange(len(states)), formal] = False
    dual_platt = v73.fit_platt(local[..., 0][mask], data["dual_vs_formal"][states][mask])
    harm_platt = v73.fit_platt(local[..., 1][mask], data["harm_vs_formal"][states][mask])
    evaluations = []
    for strictness, (dual_threshold, harm_threshold) in enumerate(LEVELS):
        policy = {
            "method": method,
            "active": True,
            "dual_platt": dual_platt.tolist(),
            "harm_platt": harm_platt.tolist(),
            "dual_threshold": dual_threshold,
            "harm_threshold": harm_threshold,
            "strictness": strictness,
        }
        selected, _, _, _ = apply_policy(data, states, local, policy)
        metrics = v74.incremental_metrics(data, states, selected)
        feasible = (
            metrics["replacements"] > 0
            and metrics["dual_wilson_95"]["lower"] >= 0.85
            and metrics["harm_wilson_95"]["upper"] <= 0.15
            and metrics["mean_gain_m"]["objective"] > 0.0
        )
        evaluations.append({
            "parameters": {
                "dual_threshold": dual_threshold,
                "harm_threshold": harm_threshold,
                "strictness": strictness,
            },
            "metrics": metrics,
            "feasible": feasible,
        })
    feasible = [value for value in evaluations if value["feasible"]]
    pool = feasible if feasible else evaluations
    best = max(
        pool,
        key=lambda value: (
            value["metrics"]["mean_gain_m"]["objective"],
            value["metrics"]["replacements"],
            value["parameters"]["strictness"],
        ),
    )
    return {
        "method": method,
        "active": bool(best["feasible"]),
        "dual_platt": dual_platt.tolist(),
        "harm_platt": harm_platt.tolist(),
        **best["parameters"],
        "inner_metrics": best["metrics"],
        "fallback_reason": None if best["feasible"] else "no_inner_wilson_feasible_structural_classifier_policy",
        "all_levels": evaluations,
    }


def choose_nested(policies: list[dict[str, Any]]) -> dict[str, Any]:
    feasible = [policy for policy in policies if policy["active"]]
    if feasible:
        return max(
            feasible,
            key=lambda policy: (
                policy["inner_metrics"]["mean_gain_m"]["objective"],
                policy["inner_metrics"]["replacements"],
                policy["strictness"],
                policy["method"] == "balanced_logistic_relative",
            ),
        )
    diagnostic = max(
        policies,
        key=lambda policy: (
            policy["inner_metrics"]["dual_wilson_95"]["lower"] - policy["inner_metrics"]["harm_wilson_95"]["upper"],
            policy["inner_metrics"]["mean_gain_m"]["objective"],
        ),
    )
    return {**diagnostic, "active": False, "fallback_reason": "no_inner_wilson_feasible_method"}


def compact_policy(policy: dict[str, Any], *, include_levels: bool = False) -> dict[str, Any]:
    keep = (
        "method",
        "active",
        "dual_platt",
        "harm_platt",
        "dual_threshold",
        "harm_threshold",
        "strictness",
        "inner_metrics",
        "fallback_reason",
    )
    result = {key: policy[key] for key in keep if key in policy}
    if include_levels:
        result["all_levels"] = policy["all_levels"]
    return result


def success_gate(summaries: dict[str, Any], formal: dict[str, Any]) -> dict[str, Any]:
    pooled = summaries["pooled"]
    recon = summaries["recon_a_to_h"]
    scand = summaries["fresh_scand"]
    gain = formal["pooled"]["absolute_mean_m"]["objective"] - pooled["absolute_mean_m"]["objective"]
    checks = {
        "nonzero_replacements": pooled["replacements_vs_formal_v70"] > 0,
        "incremental_dual_at_least_0_85": pooled["dual_precision_vs_formal_v70"] is not None and pooled["dual_precision_vs_formal_v70"] >= 0.85,
        "incremental_harm_at_most_0_15": pooled["harm_rate_vs_formal_v70"] is not None and pooled["harm_rate_vs_formal_v70"] <= 0.15,
        "recon_mean_ate_non_worse": recon["absolute_mean_m"]["ate"] <= formal["recon_a_to_h"]["absolute_mean_m"]["ate"] + TOLERANCE,
        "recon_mean_objective_non_worse": recon["absolute_mean_m"]["objective"] <= formal["recon_a_to_h"]["absolute_mean_m"]["objective"] + TOLERANCE,
        "scand_mean_ate_non_worse": scand["absolute_mean_m"]["ate"] <= formal["fresh_scand"]["absolute_mean_m"]["ate"] + TOLERANCE,
        "scand_mean_objective_non_worse": scand["absolute_mean_m"]["objective"] <= formal["fresh_scand"]["absolute_mean_m"]["objective"] + TOLERANCE,
        "scand_objective_p90_within_1mm": scand["absolute_p90_m"]["objective"] <= formal["fresh_scand"]["absolute_p90_m"]["objective"] + 0.001,
        "pooled_objective_gain_at_least_1mm": gain >= 0.001,
    }
    return {"checks": checks, "passed": bool(all(checks.values())), "pooled_mean_objective_gain_m": float(gain)}


def classifier_report(
    data: dict[str, Any], dual: np.ndarray, harm: np.ndarray
) -> dict[str, Any]:
    from sklearn.metrics import average_precision_score, roc_auc_score  # type: ignore

    report = {}
    for domain in ("recon", "scand"):
        state = data["domain"].astype(str) == domain
        mask = state[:, None] & data["valid"] & data["safe"] & np.isfinite(dual) & np.isfinite(harm)
        mask[np.arange(len(state)), data["formal_index"]] = False
        targets = {
            "dual": data["dual_vs_formal"][mask].astype(np.int64),
            "harm": data["harm_vs_formal"][mask].astype(np.int64),
        }
        probabilities = {"dual": dual[mask], "harm": harm[mask]}
        local = {"rows": int(mask.sum())}
        for name in ("dual", "harm"):
            y = targets[name]
            p = probabilities[name]
            local[name] = {
                "auroc": float(roc_auc_score(y, p)),
                "average_precision": float(average_precision_score(y, p)),
                **v73.candidate_calibration_metrics(p, y, np.ones(len(y), dtype=bool)),
            }
        report[domain] = local
    return report


def classifier_report_no_sklearn(
    data: dict[str, Any], dual: np.ndarray, harm: np.ndarray
) -> dict[str, Any]:
    from scipy.stats import rankdata

    report = {}
    for domain in ("recon", "scand"):
        state = data["domain"].astype(str) == domain
        mask = state[:, None] & data["valid"] & data["safe"] & np.isfinite(dual) & np.isfinite(harm)
        mask[np.arange(len(state)), data["formal_index"]] = False
        local = {"rows": int(mask.sum())}
        for name, probability, target in (
            ("dual", dual[mask], data["dual_vs_formal"][mask].astype(np.int64)),
            ("harm", harm[mask], data["harm_vs_formal"][mask].astype(np.int64)),
        ):
            rank = rankdata(probability, method="average")
            positive = rank[target == 1]
            positives = len(positive)
            negatives = int((target == 0).sum())
            auc = (positive.sum() - positives * (positives + 1) / 2.0) / (positives * negatives)
            order = np.argsort(-probability, kind="stable")
            sorted_target = target[order]
            precision = np.cumsum(sorted_target) / np.arange(1, len(sorted_target) + 1)
            ap = float(np.sum(precision * sorted_target) / positives)
            local[name] = {
                "auroc": float(auc),
                "average_precision": ap,
                **v73.candidate_calibration_metrics(probability, target, np.ones(len(target), dtype=bool)),
            }
        report[domain] = local
    return report


def selection_errors(data: dict[str, Any], selected: np.ndarray) -> dict[str, int | float]:
    replacement = selected != data["formal_index"]
    row = np.arange(len(selected))
    dual = data["dual_vs_formal"][row, selected].astype(bool)
    harm = data["harm_vs_formal"][row, selected].astype(bool)
    opportunity = np.zeros(len(selected), dtype=bool)
    for state in range(len(selected)):
        mask = data["valid"][state] & data["safe"][state]
        mask[int(data["formal_index"][state])] = False
        opportunity[state] = bool(data["dual_vs_formal"][state, mask].astype(bool).any())
    return {
        "accepted_replacements": int(replacement.sum()),
        "dual_accepts": int((replacement & dual).sum()),
        "harmful_accepts": int((replacement & harm).sum()),
        "fallback_states": int((~replacement).sum()),
        "states_with_true_dual_opportunity": int(opportunity.sum()),
        "fallback_with_true_dual_opportunity": int(((~replacement) & opportunity).sum()),
        "false_reject_rate_given_opportunity": float((((~replacement) & opportunity).sum()) / max(opportunity.sum(), 1)),
    }


def policy_latency(
    data: dict[str, Any], probability: dict[str, np.ndarray], policies: dict[tuple[int, str], dict[str, Any]]
) -> dict[str, Any]:
    values = []
    for state in range(len(data["record_key"])):
        fold = int(data["fold"][state])
        domain = str(data["domain"][state])
        policy = policies[(fold, domain)]
        started = time.perf_counter_ns()
        apply_policy(data, np.asarray([state]), probability[policy["method"]][state : state + 1], policy)
        values.append((time.perf_counter_ns() - started) / 1.0e6)
    value = np.asarray(values)
    return {
        "mean": float(value.mean()),
        "p50": float(np.quantile(value, 0.5)),
        "p90": float(np.quantile(value, 0.9)),
        "states": int(len(value)),
        "scope": "cached Platt calibration, probability gate, and one-candidate selection only; model forward and formal-v70 planner excluded",
    }


def inner_level_summary(policy_audit: list[dict[str, Any]]) -> dict[str, Any]:
    levels = []
    for record in policy_audit:
        for policy in record["candidate_policies"].values():
            levels.extend(policy["all_levels"])
    nonzero = [value for value in levels if value["metrics"]["replacements"] > 0]
    return {
        "levels": int(len(levels)),
        "nonzero_levels": int(len(nonzero)),
        "dual_wilson_lcb_passes": int(sum(value["metrics"]["dual_wilson_95"]["lower"] >= 0.85 for value in nonzero)),
        "harm_wilson_ucb_passes": int(sum(value["metrics"]["harm_wilson_95"]["upper"] <= 0.15 for value in nonzero)),
        "joint_wilson_passes": int(sum(value["metrics"]["dual_wilson_95"]["lower"] >= 0.85 and value["metrics"]["harm_wilson_95"]["upper"] <= 0.15 for value in nonzero)),
        "fully_feasible_levels": int(sum(value["feasible"] for value in levels)),
    }


def main() -> None:
    args = parse_args()
    result_path = OUTPUT / "result.json"
    prediction_path = OUTPUT / "outer_oof_predictions.npz"
    if result_path.exists() or prediction_path.exists():
        raise FileExistsError("v75 result already exists")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    immutable = verify_contract()
    data, data_audit = load_data()
    route = data_audit["v74_data_audit"]["base_v72_data_audit"]["route_folds"]
    prereg = v72.read_json(PREREG)
    if route["assignment_sha256"] != prereg["nested_route_oof"]["outer_assignment_sha256"]:
        raise RuntimeError("outer assignment changed")
    probability = {method: np.full((*data["valid"].shape, 2), np.nan, dtype=np.float32) for method in METHODS}
    probability_std = {method: np.full_like(probability[method], np.nan) for method in METHODS}
    selected = {name: data["formal_index"].copy().astype(np.int64) for name in REPORTED}
    calibrated_dual = {name: np.full(data["valid"].shape, np.nan, dtype=np.float32) for name in REPORTED}
    calibrated_harm = {name: np.full(data["valid"].shape, np.nan, dtype=np.float32) for name in REPORTED}
    chosen: dict[str, dict[tuple[int, str], dict[str, Any]]] = {name: {} for name in REPORTED}
    policy_audit = []
    cache_hashes = {}
    started = time.perf_counter()
    for outer in range(5):
        cached = build_cache(data, outer, device=args.device)
        cache_hashes[f"outer_{outer}"] = v72.sha256_file(cache_path(outer))
        outer_all = cached["outer_state_index"].astype(np.int64)
        for method in METHODS:
            probability[method][outer_all] = cached[f"{method}__outer_probability"]
            probability_std[method][outer_all] = cached[f"{method}__outer_std"]
        inner_probability = {method: expand_inner(data, cached, method) for method in METHODS}
        for domain in ("recon", "scand"):
            inner_states = np.flatnonzero((data["fold"] != outer) & (data["domain"].astype(str) == domain))
            outer_states = np.flatnonzero((data["fold"] == outer) & (data["domain"].astype(str) == domain))
            if set(data["group"][inner_states].astype(str)) & set(data["group"][outer_states].astype(str)):
                raise RuntimeError("outer route leakage")
            policies = [fit_policy(data, inner_states, inner_probability[method], method) for method in METHODS]
            local = {policy["method"]: policy for policy in policies}
            local["nested_selected"] = choose_nested(policies)
            for name, policy in local.items():
                chosen[name][(outer, domain)] = policy
                local_selected, dual, harm, _ = apply_policy(
                    data, outer_states, probability[policy["method"]][outer_states], policy
                )
                selected[name][outer_states] = local_selected
                calibrated_dual[name][outer_states] = dual
                calibrated_harm[name][outer_states] = harm
            policy_audit.append({
                "outer_fold": outer,
                "domain": domain,
                "inner_states": int(len(inner_states)),
                "outer_states": int(len(outer_states)),
                "route_overlap": 0,
                "outer_truth_used_for_selection": False,
                "candidate_policies": {policy["method"]: compact_policy(policy, include_levels=True) for policy in policies},
                "selected_policies": {name: compact_policy(policy) for name, policy in local.items()},
            })
            print(json.dumps({"stage": "outer_policy", "outer_fold": outer, "domain": domain, "nested": compact_policy(local["nested_selected"])}, sort_keys=True), flush=True)
    if not all(np.isfinite(value).all() for value in probability.values()):
        raise RuntimeError("incomplete v75 outer OOF probabilities")
    formal = v72.formal_indices_summary(data)
    methods = {}
    for name in REPORTED:
        summaries = v72.selection_summaries(data, selected[name])
        methods[name] = {
            "selection": summaries,
            "success_gate": success_gate(summaries, formal),
            "comparison_vs_formal_v70": {key: v72.method_delta(summaries[key], formal[key]) for key in summaries},
            "selection_errors": selection_errors(data, selected[name]),
            "classifier_report": classifier_report_no_sklearn(data, calibrated_dual[name], calibrated_harm[name]),
            "active_policy_counts": {
                "active": int(sum(policy["active"] for policy in chosen[name].values())),
                "fallback_only": int(sum(not policy["active"] for policy in chosen[name].values())),
                "methods": {method: int(sum(policy["method"] == method for policy in chosen[name].values())) for method in METHODS},
            },
            "scand_tail": {
                "objective_mean_m": summaries["fresh_scand"]["absolute_mean_m"]["objective"],
                "objective_p90_m": summaries["fresh_scand"]["absolute_p90_m"]["objective"],
                "ate_mean_m": summaries["fresh_scand"]["absolute_mean_m"]["ate"],
                "ate_p90_m": summaries["fresh_scand"]["absolute_p90_m"]["ate"],
            },
        }
    level_summary = inner_level_summary(policy_audit)
    latency = policy_latency(data, probability, chosen["nested_selected"])
    arrays: dict[str, np.ndarray] = {
        "record_key": data["record_key"],
        "fold": data["fold"],
        "domain": data["domain"],
        "formal_index": data["formal_index"],
    }
    for method in METHODS:
        arrays[f"{method}__probability"] = probability[method]
        arrays[f"{method}__probability_std"] = probability_std[method]
    for name in REPORTED:
        arrays[f"{name}__selected_index"] = selected[name]
        arrays[f"{name}__calibrated_dual"] = calibrated_dual[name]
        arrays[f"{name}__calibrated_harm"] = calibrated_harm[name]
    v72.write_npz(prediction_path, **arrays)
    immutable_after = verify_contract()
    if immutable_after != immutable:
        raise RuntimeError("immutable input changed during v75")
    gate = methods["nested_selected"]["success_gate"]
    result = {
        "schema": "v75_structural_selective_classifier_result_v1",
        "status": "development_gate_met" if gate["passed"] else "development_gate_not_met_stop_spent_cohort_iteration",
        "evidence_role": "final_spent_development_iteration_not_independent_validation",
        "states": int(len(data["record_key"])),
        "routes": int(prereg["nested_route_oof"]["outer_route_groups"]),
        "candidates_per_state": int(data["valid"].shape[1]),
        "formal_v70": formal,
        "methods": methods,
        "inner_level_summary": level_summary,
        "policy_audit": policy_audit,
        "latency_ms": latency,
        "elapsed_sec": time.perf_counter() - started,
        "device": args.device,
        "stop_rule_triggered": bool(not gate["passed"]),
        "minimum_next_step_if_failed": "new fully independent SCAND proposal-tail routes/labels or training a route-stable representation; do not relax safety, add CEM depth, or reuse reserve",
        "audit": {
            "immutable_before": immutable,
            "immutable_after": immutable_after,
            "cache_sha256": cache_hashes,
            "outer_assignment_sha256": route["assignment_sha256"],
            "outer_route_overlap_violations": route["route_overlap_violations"],
            "inner_route_overlap_violations": 0,
            "reserve_accessed": False,
            "formal_v70_mutated": False,
            "cem_or_archive_mutated": False,
            "dwm_ltwm_rerun": False,
            "outer_truth_used_for_selection": False,
            "prediction_sha256": v72.sha256_file(prediction_path),
        },
        "data_audit": data_audit,
    }
    v72.write_json(result_path, result)
    audit_result = {
        "schema": "v75_structural_selective_classifier_audit_v1",
        "status": result["status"],
        "success_gate": gate,
        "inner_level_summary": level_summary,
        "stop_rule_triggered": result["stop_rule_triggered"],
        "preregistration_sha256": immutable["preregistration_sha256"],
        "result_sha256": v72.sha256_file(result_path),
        "predictions_sha256": v72.sha256_file(prediction_path),
        "cache_sha256": cache_hashes,
        "route_overlap_violations": 0,
        "reserve_accessed": False,
        "formal_v70_mutated": False,
    }
    v72.write_json(OUTPUT / "audit.json", audit_result)
    print(json.dumps({
        "stage": "complete",
        "status": result["status"],
        "success_gate": gate,
        "inner_level_summary": level_summary,
        "selection": methods["nested_selected"]["selection"]["pooled"],
        "elapsed_sec": result["elapsed_sec"],
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
