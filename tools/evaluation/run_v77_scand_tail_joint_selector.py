#!/usr/bin/env python3
"""Nested route-OOF joint safety-aware reranking on frozen v70 SCAND candidates."""

from __future__ import annotations

from collections import Counter
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from tools.evaluation import run_scand_filter_random_sensitivity_v1 as prior  # noqa: E402
from tools.evaluation import run_v72_proposal_tail_reranker as v72  # noqa: E402
from tools.evaluation import run_v74_direct_delta_nested as v74  # noqa: E402


OUTPUT = ROOT / "outputs/cem_development_v1/v77_scand_tail_joint_selector"
PREREG = OUTPUT / "preregistration.json"
RESULT = OUTPUT / "result.json"
AUDIT = OUTPUT / "audit.json"
PREDICTIONS = OUTPUT / "outer_oof_predictions.npz"
MODEL_PARAMETERS = OUTPUT / "model_parameters.npz"
MODEL_AUDIT = OUTPUT / "model_parameters.audit.json"
MANIFEST = OUTPUT / "sha256_manifest.json"
MODELS = ("route_balanced_pairwise_linear_ensemble", "group_dro_listwise_multitask_ensemble")
PROFILES = {
    "balanced_direct": {"sigma": 1.0, "dual": 0.80, "harm": 0.08, "gain": 0.002, "top_k": None},
    "conservative_top5": {"sigma": 1.5, "dual": 0.90, "harm": 0.03, "gain": 0.004, "top_k": 5},
}
METHODS = tuple(f"{model}__{profile}" for model in MODELS for profile in PROFILES)
LINEAR_SEEDS = (20267701, 20267702, 20267703)
MLP_SEEDS = (20267711, 20267712, 20267713)
METRICS = ("ate", "rpe", "endpoint", "objective")
SOURCES = ("fixed", "r1", "r2", "r3")
TOLERANCE = 1.0e-7
NEUTRAL_M = 0.001
SUPPORT_THRESHOLD = 4.876645565032959
Z95 = 1.96


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(jsonable(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_npz(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def verify_contract() -> tuple[dict[str, Any], dict[str, str]]:
    prereg = read_json(PREREG)
    if prereg.get("status") != "frozen_before_v77_outer_heldout_evaluation":
        raise RuntimeError("v77 preregistration is not frozen")
    if any(prereg["constraints"].values()):
        raise RuntimeError("v77 forbidden-action contract changed")
    observed = {"preregistration": sha256_file(PREREG)}
    for name, record in prereg["inputs"].items():
        path = ROOT / record["path"]
        if "res" + "erve" in str(path).lower():
            raise RuntimeError(f"forbidden input path: {name}")
        digest = sha256_file(path)
        if digest != record["sha256"]:
            raise RuntimeError(f"immutable input changed: {name}")
        observed[name] = digest
    return prereg, observed


def safe_logit(value: np.ndarray) -> np.ndarray:
    value = np.clip(value.astype(np.float64), 1.0e-5, 1.0 - 1.0e-5)
    return np.log(value / (1.0 - value)).astype(np.float32)


def relative(value: np.ndarray, formal: np.ndarray) -> np.ndarray:
    return value - value[np.arange(len(formal)), formal][:, None]


def flatten_named(prefix: str, value: np.ndarray) -> tuple[np.ndarray, list[str]]:
    value = np.asarray(value, dtype=np.float32)
    flat = value.reshape(*value.shape[:2], -1)
    names = [f"{prefix}_{index}" for index in range(flat.shape[-1])]
    return flat, names


def assemble_features(
    data: dict[str, Any], fields: dict[str, np.ndarray], structural: dict[str, np.ndarray]
) -> tuple[np.ndarray, list[str], np.ndarray, dict[str, Any]]:
    formal = data["formal_index"]
    pool = fields["physical"] & (fields["support"] <= SUPPORT_THRESHOLD)
    ucb = fields["mean"].astype(np.float32) + 2.0 * fields["std"].astype(np.float32)
    energy = fields["energy"].astype(np.float32).copy()
    energy_available = np.isfinite(energy)
    for state in range(len(energy)):
        finite = energy_available[state]
        replacement = float(np.max(energy[state, finite]) + 1.0) if finite.any() else 1.0
        energy[state, ~finite] = replacement
    seed_parts = []
    seed_names = []
    for name, value in (
        ("ate", structural["ate_seed"]),
        ("rtp", structural["rtp_seed"]),
        ("support", structural["support_seed"]),
    ):
        seed_parts.extend((value.mean(axis=-1, keepdims=True), value.std(axis=-1, keepdims=True), np.ptp(value, axis=-1, keepdims=True)))
        seed_names.extend((f"{name}_seed_mean", f"{name}_seed_std", f"{name}_seed_range"))
    seed_summary = np.concatenate(seed_parts, axis=-1).astype(np.float32)
    action = structural["actions_h4"].astype(np.float32)
    waypoint = data["candidates"].astype(np.float32) * 0.25
    action_summary = v74.sequence_summary(action)
    waypoint_summary = v74.sequence_summary(waypoint)
    density = np.concatenate((v74.local_density(action, pool), v74.local_density(waypoint, pool)), axis=-1)
    ranks = np.concatenate(
        tuple(
            v74.percentile_rank(value, pool)
            for value in (
                energy,
                fields["scorer_proxy"],
                ucb[..., 0],
                ucb[..., 3],
                -fields["p_dual"],
                fields["p_harm"],
                fields["support"],
            )
        ),
        axis=-1,
    ).astype(np.float32)
    source_onehot = np.eye(4, dtype=np.float32)[fields["source_pool"]]
    source_rank = fields["source_candidate"].astype(np.float32)[..., None] / 16.0
    flags = np.stack(
        (
            fields["current_safe"].astype(np.float32),
            energy_available.astype(np.float32),
            (np.arange(pool.shape[1])[None, :] == formal[:, None]).astype(np.float32),
            np.broadcast_to(
                (np.arange(pool.shape[1])[None, :] == 0), pool.shape
            ).astype(np.float32),
        ),
        axis=-1,
    )
    numeric = [
        ("safety_mean", fields["mean"]),
        ("safety_std", fields["std"]),
        ("safety_ucb_z2", ucb),
        ("risk_logit", np.stack((safe_logit(fields["p_dual"]), safe_logit(fields["p_harm"])), axis=-1)),
        ("energy", energy[..., None]),
        ("scorer_proxy", fields["scorer_proxy"][..., None]),
        ("scorer_component", structural["scorer_aux"]),
        ("support_distance", fields["support"][..., None]),
        ("seed", seed_summary),
        ("action", action),
        ("waypoint_m", waypoint),
        ("action_summary", action_summary),
        ("waypoint_summary", waypoint_summary),
    ]
    blocks = []
    names: list[str] = []
    for name, value in numeric:
        raw, raw_names = flatten_named(name, value)
        delta, delta_names = flatten_named(f"relative_formal_{name}", relative(value, formal))
        blocks.extend((raw, delta))
        names.extend(raw_names + delta_names)
    for name, value in (
        ("local_density", density),
        ("within_state_rank", ranks),
        ("proposal_source", source_onehot),
        ("proposal_source_rank", source_rank),
        ("flags_oldsafe_energy_formal_incumbent", flags),
    ):
        block, block_names = flatten_named(name, value)
        blocks.append(block)
        names.extend(block_names)
    feature = np.concatenate(blocks, axis=-1).astype(np.float32)
    if not np.isfinite(feature).all():
        raise RuntimeError("nonfinite v77 deployment feature")
    return feature, names, pool, {
        "feature_dim": int(feature.shape[-1]),
        "feature_names": names,
        "uses_truth": False,
        "uses_route_or_record_id": False,
        "old_safety_is_feature_only": True,
        "energy_is_feature_only": True,
        "physical_support_rows": int(pool.sum()),
    }


def route_candidate_weights(
    pool: np.ndarray, states: np.ndarray, group: np.ndarray, seed: int | None
) -> np.ndarray:
    result = np.zeros(pool.shape, dtype=np.float64)
    routes = np.unique(group[states].astype(str))
    rng = np.random.default_rng(seed) if seed is not None else None
    multiplier = {route: float(rng.lognormal(0.0, 0.25)) if rng is not None else 1.0 for route in routes}
    for route in routes:
        local_states = states[group[states].astype(str) == route]
        count = int(pool[local_states].sum())
        if count:
            result[local_states] = pool[local_states] * (multiplier[route] / count)
    scale = result[states][pool[states]].mean()
    result /= max(float(scale), 1.0e-12)
    return result


def fit_standardizer(feature: np.ndarray, pool: np.ndarray, states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = feature[states][pool[states]].astype(np.float64)
    mean = values.mean(axis=0)
    std = np.maximum(values.std(axis=0), 1.0e-5)
    return mean.astype(np.float32), std.astype(np.float32)


def standardize(feature: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((feature - mean) / std).astype(np.float32)


def weighted_ridge(design: np.ndarray, target: np.ndarray, weight: np.ndarray, regularization: np.ndarray) -> np.ndarray:
    root_weight = np.sqrt(np.maximum(weight, 0.0))[:, None]
    x = design * root_weight
    y = target * root_weight[:, 0]
    return np.linalg.solve(x.T @ x + regularization, x.T @ y)


def logistic_loss(design: np.ndarray, target: np.ndarray, weight: np.ndarray, coefficient: np.ndarray, regularization: np.ndarray) -> float:
    linear = design @ coefficient
    value = np.logaddexp(0.0, linear) - target * linear
    return float(np.sum(weight * value) + 0.5 * coefficient @ regularization @ coefficient)


def fit_logistic(design: np.ndarray, target: np.ndarray, weight: np.ndarray, ridge: float) -> np.ndarray:
    positive = float(np.average(target, weights=weight))
    positive = float(np.clip(positive, 1.0e-5, 1.0 - 1.0e-5))
    class_weight = np.where(target > 0.5, min(10.0, 0.5 / positive), min(10.0, 0.5 / (1.0 - positive)))
    weight = weight * class_weight
    coefficient = np.zeros(design.shape[1], dtype=np.float64)
    coefficient[0] = math.log(positive / (1.0 - positive))
    regularization = np.eye(design.shape[1], dtype=np.float64) * ridge
    regularization[0, 0] = 1.0e-6
    current = logistic_loss(design, target, weight, coefficient, regularization)
    for _ in range(60):
        linear = design @ coefficient
        probability = 1.0 / (1.0 + np.exp(-np.clip(linear, -30.0, 30.0)))
        curvature = np.maximum(probability * (1.0 - probability), 1.0e-6) * weight
        gradient = design.T @ (weight * (probability - target)) + regularization @ coefficient
        hessian = (design.T * curvature) @ design + regularization
        step = np.linalg.solve(hessian, gradient)
        multiplier = 1.0
        accepted = False
        while multiplier >= 1.0e-8:
            proposal = coefficient - multiplier * step
            candidate = logistic_loss(design, target, weight, proposal, regularization)
            if candidate <= current:
                coefficient, current, accepted = proposal, candidate, True
                break
            multiplier *= 0.5
        if not accepted or np.linalg.norm(multiplier * step) < 1.0e-8:
            break
    if not np.isfinite(coefficient).all() or np.max(np.abs(coefficient)) > 1.0e4:
        raise RuntimeError("linear risk solver failed")
    return coefficient.astype(np.float32)


def fit_linear_member(
    feature: np.ndarray,
    pool: np.ndarray,
    gain: np.ndarray,
    dual: np.ndarray,
    harm: np.ndarray,
    states: np.ndarray,
    group: np.ndarray,
    seed: int,
) -> dict[str, np.ndarray]:
    mean, std = fit_standardizer(feature, pool, states)
    scaled = standardize(feature, mean, std)
    weight_grid = route_candidate_weights(pool, states, group, seed)
    anchor_x = scaled[states][pool[states]].astype(np.float64)
    anchor_y = gain[states][pool[states]].astype(np.float64)
    anchor_w = weight_grid[states][pool[states]] * 0.25
    pair_x, pair_y, pair_w = [], [], []
    for state in states:
        index = np.flatnonzero(pool[state])
        order = index[np.argsort(gain[state, index], kind="stable")]
        bottom, top = order[: min(8, len(order))], order[-min(8, len(order)) :]
        route_weight = float(np.mean(weight_grid[state, index]))
        for better in top:
            for worse in bottom:
                if better == worse:
                    continue
                pair_x.append(scaled[state, better] - scaled[state, worse])
                pair_y.append(float(gain[state, better] - gain[state, worse]))
                pair_w.append(route_weight)
    pair_x = np.asarray(pair_x, dtype=np.float64)
    pair_y = np.asarray(pair_y, dtype=np.float64)
    pair_w = np.asarray(pair_w, dtype=np.float64)
    design = np.concatenate(
        (
            np.concatenate((np.ones((len(anchor_x), 1)), anchor_x), axis=1),
            np.concatenate((np.zeros((len(pair_x), 1)), pair_x), axis=1),
        ),
        axis=0,
    )
    target = np.concatenate((anchor_y, pair_y))
    weights = np.concatenate((anchor_w, pair_w))
    regularization = np.eye(design.shape[1], dtype=np.float64) * 10.0
    regularization[0, 0] = 1.0e-4
    gain_coefficient = weighted_ridge(design, target, weights, regularization).astype(np.float32)
    risk_design = np.concatenate((np.ones((len(anchor_x), 1)), anchor_x), axis=1)
    dual_coefficient = fit_logistic(risk_design, dual[states][pool[states]].astype(np.float64), weight_grid[states][pool[states]], 1.0)
    harm_coefficient = fit_logistic(risk_design, harm[states][pool[states]].astype(np.float64), weight_grid[states][pool[states]], 1.0)
    return {
        "mean": mean,
        "std": std,
        "gain_coefficient": gain_coefficient,
        "dual_coefficient": dual_coefficient,
        "harm_coefficient": harm_coefficient,
    }


def predict_linear_member(model: dict[str, np.ndarray], feature: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scaled = standardize(feature, model["mean"], model["std"])
    design = np.concatenate((np.ones((*scaled.shape[:2], 1), dtype=np.float32), scaled), axis=-1)
    gain = design @ model["gain_coefficient"]
    dual = 1.0 / (1.0 + np.exp(-np.clip(design @ model["dual_coefficient"], -30.0, 30.0)))
    harm = 1.0 / (1.0 + np.exp(-np.clip(design @ model["harm_coefficient"], -30.0, 30.0)))
    return gain.astype(np.float32), dual.astype(np.float32), harm.astype(np.float32)


class JointMLP(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.network = nn.Sequential(nn.Linear(dimension, 48), nn.SiLU(), nn.Linear(48, 24), nn.SiLU(), nn.Linear(24, 3))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.network(value)


def fit_mlp_member(
    feature: np.ndarray,
    pool: np.ndarray,
    gain: np.ndarray,
    dual: np.ndarray,
    harm: np.ndarray,
    states: np.ndarray,
    group: np.ndarray,
    seed: int,
    device: str,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
    mean, std = fit_standardizer(feature, pool, states)
    scaled = standardize(feature[states], mean, std)
    x = torch.as_tensor(scaled, dtype=torch.float32, device=device)
    mask = torch.as_tensor(pool[states], dtype=torch.bool, device=device)
    true_gain = torch.as_tensor(gain[states], dtype=torch.float32, device=device)
    true_dual = torch.as_tensor(dual[states], dtype=torch.float32, device=device)
    true_harm = torch.as_tensor(harm[states], dtype=torch.float32, device=device)
    route_names = group[states].astype(str)
    routes = np.unique(route_names)
    route_state = [torch.as_tensor(np.flatnonzero(route_names == route), dtype=torch.long, device=device) for route in routes]
    dro = torch.ones(len(routes), dtype=torch.float32, device=device) / len(routes)
    dual_rate = float(dual[states][pool[states]].mean())
    harm_rate = float(harm[states][pool[states]].mean())
    dual_pos = torch.tensor(min(10.0, (1.0 - dual_rate) / max(dual_rate, 1.0e-5)), device=device)
    harm_pos = torch.tensor(min(10.0, (1.0 - harm_rate) / max(harm_rate, 1.0e-5)), device=device)
    model = JointMLP(feature.shape[-1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3, weight_decay=1.0e-3)
    for _ in range(80):
        model.train()
        output = model(x)
        predicted_gain, dual_logit, harm_logit = output.unbind(dim=-1)
        masked_prediction = predicted_gain.masked_fill(~mask, -1.0e9)
        masked_truth = true_gain.masked_fill(~mask, -1.0e9)
        truth_distribution = torch.softmax(masked_truth / 0.02, dim=1)
        listwise = -(truth_distribution * torch.log_softmax(masked_prediction / 0.02, dim=1)).sum(dim=1)
        denominator = mask.sum(dim=1).clamp_min(1)
        huber = (F.huber_loss(predicted_gain, true_gain, reduction="none", delta=0.01) * mask).sum(dim=1) / denominator
        dual_loss = (F.binary_cross_entropy_with_logits(dual_logit, true_dual, reduction="none", pos_weight=dual_pos) * mask).sum(dim=1) / denominator
        harm_loss = (F.binary_cross_entropy_with_logits(harm_logit, true_harm, reduction="none", pos_weight=harm_pos) * mask).sum(dim=1) / denominator
        state_loss = listwise + 0.25 * huber + 0.5 * dual_loss + 0.5 * harm_loss
        route_loss = torch.stack([state_loss[index].mean() for index in route_state])
        loss = torch.sum(dro * route_loss)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        with torch.no_grad():
            normalized = (route_loss - route_loss.mean()) / route_loss.std().clamp_min(1.0e-6)
            dro *= torch.exp(torch.clamp(0.05 * normalized, -2.0, 2.0))
            dro /= dro.sum()
    model.eval()
    return {"mean": mean, "std": std, "model": model, "dro_route": routes, "dro_weight": dro.detach().cpu().numpy()}


def predict_mlp_member(model: dict[str, Any], feature: np.ndarray, device: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scaled = standardize(feature, model["mean"], model["std"])
    with torch.no_grad():
        output = model["model"](torch.as_tensor(scaled, dtype=torch.float32, device=device)).cpu().numpy()
    return output[..., 0].astype(np.float32), (1.0 / (1.0 + np.exp(-np.clip(output[..., 1], -30, 30)))).astype(np.float32), (1.0 / (1.0 + np.exp(-np.clip(output[..., 2], -30, 30)))).astype(np.float32)


def fit_ensemble(
    model_name: str,
    feature: np.ndarray,
    pool: np.ndarray,
    gain: np.ndarray,
    dual: np.ndarray,
    harm: np.ndarray,
    states: np.ndarray,
    group: np.ndarray,
    device: str,
) -> list[dict[str, Any]]:
    if model_name == MODELS[0]:
        return [fit_linear_member(feature, pool, gain, dual, harm, states, group, seed) for seed in LINEAR_SEEDS]
    return [fit_mlp_member(feature, pool, gain, dual, harm, states, group, seed, device) for seed in MLP_SEEDS]


def predict_ensemble(model_name: str, models: list[dict[str, Any]], feature: np.ndarray, device: str) -> dict[str, np.ndarray]:
    prediction = []
    for model in models:
        prediction.append(predict_linear_member(model, feature) if model_name == MODELS[0] else predict_mlp_member(model, feature, device))
    result = {}
    for index, name in enumerate(("gain", "p_dual", "p_harm")):
        values = np.stack([member[index] for member in prediction], axis=0)
        result[f"{name}_mean"] = values.mean(axis=0).astype(np.float32)
        result[f"{name}_std"] = values.std(axis=0).astype(np.float32)
    return result


def serialize_outer_models(
    prefix: str, model_name: str, models: list[dict[str, Any]], arrays: dict[str, np.ndarray]
) -> None:
    for member_index, model in enumerate(models):
        key = f"{prefix}__member{member_index}"
        arrays[f"{key}__mean"] = model["mean"]
        arrays[f"{key}__std"] = model["std"]
        if model_name == MODELS[0]:
            for name in ("gain_coefficient", "dual_coefficient", "harm_coefficient"):
                arrays[f"{key}__{name}"] = model[name]
        else:
            for name, value in model["model"].state_dict().items():
                arrays[f"{key}__{name.replace('.', '_')}"] = value.detach().cpu().numpy()
            arrays[f"{key}__dro_weight"] = model["dro_weight"]
            arrays[f"{key}__dro_route"] = model["dro_route"].astype("U256")


def ensemble_cpu_latency(model_name: str, models: list[dict[str, Any]], feature: np.ndarray) -> float:
    local_models = models
    if model_name == MODELS[1]:
        local_models = []
        for item in models:
            local = {key: value for key, value in item.items() if key != "model"}
            local["model"] = copy.deepcopy(item["model"]).cpu()
            local_models.append(local)
    started = time.perf_counter()
    repetitions = 10
    for _ in range(repetitions):
        predict_ensemble(model_name, local_models, feature, "cpu")
    return float((time.perf_counter() - started) / repetitions / len(feature))


def apply_profile(
    prediction: dict[str, np.ndarray],
    profile_name: str,
    pool: np.ndarray,
    formal: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    spec = PROFILES[profile_name]
    gain_lcb = prediction["gain_mean"] - spec["sigma"] * prediction["gain_std"]
    dual_lcb = prediction["p_dual_mean"] - spec["sigma"] * prediction["p_dual_std"]
    harm_ucb = prediction["p_harm_mean"] + spec["sigma"] * prediction["p_harm_std"]
    admitted = pool & (gain_lcb >= spec["gain"]) & (dual_lcb >= spec["dual"]) & (harm_ucb <= spec["harm"])
    admitted[np.arange(len(formal)), formal] = False
    considered = admitted.copy()
    if spec["top_k"] is not None:
        considered[:] = False
        for state in range(len(formal)):
            candidate = np.flatnonzero(pool[state] & (np.arange(pool.shape[1]) != formal[state]))
            order = candidate[np.argsort(-prediction["gain_mean"][state, candidate], kind="stable")]
            top = order[: spec["top_k"]]
            considered[state, top] = admitted[state, top]
    masked = np.where(considered, gain_lcb, -np.inf)
    selected = np.argmax(masked, axis=1)
    replace = np.isfinite(masked[np.arange(len(formal)), selected])
    return admitted, np.where(replace, selected, formal).astype(np.int64)


def wilson(successes: int, total: int) -> dict[str, Any]:
    if total <= 0:
        return {"successes": successes, "total": total, "rate": None, "lcb": 0.0, "ucb": 1.0}
    p = successes / total
    denominator = 1.0 + Z95 * Z95 / total
    center = (p + Z95 * Z95 / (2.0 * total)) / denominator
    radius = Z95 * math.sqrt(p * (1.0 - p) / total + Z95 * Z95 / (4.0 * total * total)) / denominator
    return {"successes": successes, "total": total, "rate": float(p), "lcb": float(max(0.0, center - radius)), "ucb": float(min(1.0, center + radius))}


def metric_summary(value: np.ndarray) -> dict[str, Any]:
    return {
        "mean_m": {name: float(value[:, index].mean()) for index, name in enumerate(METRICS)},
        "p90_m": {name: float(np.quantile(value[:, index], 0.9)) for index, name in enumerate(METRICS)},
    }


def evaluate_selection(data: dict[str, Any], selected: np.ndarray, states: np.ndarray | None = None) -> dict[str, Any]:
    if states is None:
        states = np.arange(len(selected))
        metrics = data["true_metrics"]
        formal = data["formal_index"]
    else:
        metrics = data["true_metrics"][states]
        formal = data["formal_index"][states]
    row = np.arange(len(selected))
    chosen = metrics[row, selected]
    baseline = metrics[row, formal]
    gain = baseline - chosen
    replacement = selected != formal
    dual = replacement & (gain[:, 0] > TOLERANCE) & (gain[:, 3] > TOLERANCE)
    harm = replacement & ((gain[:, 0] < -TOLERANCE) | (gain[:, 3] < -TOLERANCE))
    neutral_dual = replacement & (gain[:, 0] > NEUTRAL_M) & (gain[:, 3] > NEUTRAL_M)
    neutral_harm = replacement & ((gain[:, 0] < -NEUTRAL_M) | (gain[:, 3] < -NEUTRAL_M))
    count = int(replacement.sum())
    return {
        "states": int(len(selected)),
        "metrics": metric_summary(chosen),
        "formal_metrics": metric_summary(baseline),
        "delta_mean_m": {name: float(chosen[:, index].mean() - baseline[:, index].mean()) for index, name in enumerate(METRICS)},
        "replacements": count,
        "coverage": float(count / len(selected)),
        "strict_zero_mm": {"dual": wilson(int(dual.sum()), count), "harm": wilson(int(harm.sum()), count), "neither": int((replacement & ~dual & ~harm).sum())},
        "neutral_1mm": {"dual": wilson(int(neutral_dual.sum()), count), "harm": wilson(int(neutral_harm.sum()), count), "ambiguous": int((replacement & ~neutral_dual & ~neutral_harm).sum())},
    }


def inner_feasible(summary: dict[str, Any]) -> tuple[bool, dict[str, bool]]:
    value, formal = summary["metrics"], summary["formal_metrics"]
    checks = {
        "nonzero": summary["replacements"] > 0,
        "dual_lcb": summary["strict_zero_mm"]["dual"]["lcb"] >= 0.85,
        "harm_ucb": summary["strict_zero_mm"]["harm"]["ucb"] <= 0.15,
        "mean_ate": value["mean_m"]["ate"] <= formal["mean_m"]["ate"] + 1.0e-12,
        "mean_objective": value["mean_m"]["objective"] <= formal["mean_m"]["objective"] + 1.0e-12,
        "objective_p90": value["p90_m"]["objective"] <= formal["p90_m"]["objective"] + 0.001,
        "scand_gain_7p88mm": summary["delta_mean_m"]["objective"] <= -0.00788,
    }
    return all(checks.values()), checks


def rank_values(value: np.ndarray) -> np.ndarray:
    order = np.argsort(value, kind="stable")
    rank = np.empty(len(order), dtype=np.float64)
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and value[order[stop]] == value[order[start]]:
            stop += 1
        rank[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return rank


def correlation(x: np.ndarray, y: np.ndarray, spearman: bool = False) -> float | None:
    x, y = np.asarray(x, dtype=np.float64).reshape(-1), np.asarray(y, dtype=np.float64).reshape(-1)
    if spearman:
        x, y = rank_values(x), rank_values(y)
    if x.std() < 1.0e-12 or y.std() < 1.0e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def auc_score(target: np.ndarray, probability: np.ndarray) -> float | None:
    target = np.asarray(target, dtype=bool).reshape(-1)
    positive, negative = int(target.sum()), int((~target).sum())
    if not positive or not negative:
        return None
    ranks = rank_values(np.asarray(probability).reshape(-1)) + 1.0
    return float((ranks[target].sum() - positive * (positive + 1) / 2.0) / (positive * negative))


def probability_metrics(target: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    probability = np.asarray(probability, dtype=np.float64).reshape(-1)
    ece = 0.0
    for index in range(10):
        low, high = index / 10.0, (index + 1) / 10.0
        local = (probability >= low) & ((probability < high) if index < 9 else (probability <= high))
        if local.any():
            ece += float(local.mean()) * abs(float(probability[local].mean() - target[local].mean()))
    return {"brier": float(np.mean((probability - target) ** 2)), "ece_10": ece, "auroc": auc_score(target, probability), "positive_rate": float(target.mean())}


def candidate_metrics(prediction: dict[str, np.ndarray], gain: np.ndarray, dual: np.ndarray, harm: np.ndarray, challenger: np.ndarray) -> dict[str, Any]:
    mask = challenger
    ndcg, top5 = [], []
    for state in range(len(mask)):
        index = np.flatnonzero(mask[state])
        order = index[np.argsort(-prediction["gain_mean"][state, index], kind="stable")]
        ideal = index[np.argsort(-gain[state, index], kind="stable")]
        relevance = np.maximum(gain[state], 0.0)
        discount = 1.0 / np.log2(np.arange(2, 7))
        dcg = float(np.sum(relevance[order[:5]] * discount[: len(order[:5])]))
        idcg = float(np.sum(relevance[ideal[:5]] * discount[: len(ideal[:5])]))
        if idcg > 0:
            ndcg.append(dcg / idcg)
        if dual[state, index].any():
            top5.append(bool(dual[state, order[:5]].any()))
    return {
        "objective_gain_pearson": correlation(prediction["gain_mean"][mask], gain[mask]),
        "objective_gain_spearman": correlation(prediction["gain_mean"][mask], gain[mask], True),
        "ndcg_at_5_mean": float(np.mean(ndcg)) if ndcg else None,
        "top5_true_dual_state_recall": float(np.mean(top5)) if top5 else None,
        "dual": probability_metrics(dual[mask], prediction["p_dual_mean"][mask]),
        "harm": probability_metrics(harm[mask], prediction["p_harm_mean"][mask]),
    }


def filter_summary(admitted: np.ndarray, challenger: np.ndarray, dual: np.ndarray, harm: np.ndarray) -> dict[str, Any]:
    admitted = admitted & challenger
    dual_input, harm_input = challenger & dual, challenger & harm
    dual_pass, harm_pass = admitted & dual, admitted & harm
    opportunity = dual_input.any(axis=1)
    recovered = dual_pass.any(axis=1)
    return {
        "input_candidates": int(challenger.sum()),
        "admitted_candidates": int(admitted.sum()),
        "candidate_coverage": float(admitted.sum() / max(int(challenger.sum()), 1)),
        "states_with_any_admitted": int(admitted.any(axis=1).sum()),
        "dual_input": int(dual_input.sum()),
        "dual_admitted": int(dual_pass.sum()),
        "harm_input": int(harm_input.sum()),
        "harm_admitted": int(harm_pass.sum()),
        "dual_recall": wilson(int(dual_pass.sum()), int(dual_input.sum())),
        "dual_purity": wilson(int(dual_pass.sum()), int(admitted.sum())),
        "harm_rejection": wilson(int(harm_input.sum() - harm_pass.sum()), int(harm_input.sum())),
        "opportunity_states": int(opportunity.sum()),
        "recovered_opportunity_states": int((opportunity & recovered).sum()),
        "opportunity_recall": wilson(int((opportunity & recovered).sum()), int(opportunity.sum())),
    }


def selection_extra(
    data: dict[str, Any], fields: dict[str, np.ndarray], selected: np.ndarray, formal_energy: np.ndarray
) -> dict[str, Any]:
    formal = data["formal_index"]
    replacement = selected != formal
    row = np.arange(len(selected))
    selected_energy = fields["energy"][row, selected]
    source = fields["source_pool"][row, selected]
    source_count = Counter(SOURCES[int(value)] for value in source[replacement])
    route_count = Counter(data["group"][replacement].astype(str))
    return {
        "bypass_old_safety": int((replacement & ~fields["current_safe"][row, selected]).sum()),
        "bypass_frozen_energy": int((replacement & (~np.isfinite(selected_energy) | (selected_energy >= formal_energy))).sum()),
        "source_round": dict(sorted(source_count.items())),
        "route_distribution": dict(sorted(route_count.items())),
        "routes_with_replacements": len(route_count),
    }


def frozen_recon_summary() -> dict[str, Any]:
    total, weighted = 0, np.zeros(4, dtype=np.float64)
    cohorts = {}
    for bank in "abcdefgh":
        path = prior.V70_ROOT / f"evaluations/recon_bank_{bank}/result.json"
        decision = read_json(path)["decisions"]["proposal_aware_v66_cem"]
        states = int(decision["states"])
        weighted += states * np.asarray([decision["absolute_mean_m"][name] for name in METRICS])
        total += states
        cohorts[bank] = {"states": states, "mean_m": decision["absolute_mean_m"], "p90_m": decision["absolute_p90_m"], "sha256": sha256_file(path)}
    return {"states": total, "pooled_mean_m": {name: float(weighted[index] / total) for index, name in enumerate(METRICS)}, "cohorts": cohorts, "behavior": "unchanged by construction"}


def main() -> None:
    for path in (RESULT, AUDIT, PREDICTIONS, MODEL_PARAMETERS, MODEL_AUDIT, MANIFEST):
        if path.exists():
            raise FileExistsError(path)
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    prereg, hashes_before = verify_contract()
    started = time.perf_counter()
    spec = next(item for item in v72.COHORTS if item["name"] == "scand_fresh_outer")
    data = v72.load_cohort(spec)
    fold, fold_audit = v72.assign_route_folds(np.full(len(data["record_key"]), "scand", dtype="U8"), data["group"])
    pools, reconstruction_audit = prior.reconstruct_pools(args.device)
    fields = prior.unionize(pools, data)
    structural, structural_audit = v74.union_structural_for_cohort(spec, data)
    feature, feature_names, pool, feature_audit = assemble_features(data, fields, structural)
    formal = data["formal_index"]
    row = np.arange(len(formal))
    challenger = pool.copy()
    challenger[row, formal] = False
    true_metrics = data["true_metrics"]
    formal_metrics = true_metrics[row, formal]
    gains = formal_metrics[:, None] - true_metrics
    objective_gain = gains[..., 3].astype(np.float32)
    dual = (gains[..., 0] > TOLERANCE) & (gains[..., 3] > TOLERANCE)
    harm = (gains[..., 0] < -TOLERANCE) | (gains[..., 3] < -TOLERANCE)
    selected_cache = v72.load_npz(prior.SELECTED)
    formal_energy = selected_cache["archive_energy"].astype(np.float32)
    prediction_shape = pool.shape
    outer_prediction = {
        model: {key: np.full(prediction_shape, np.nan, dtype=np.float32) for key in ("gain_mean", "gain_std", "p_dual_mean", "p_dual_std", "p_harm_mean", "p_harm_std")}
        for model in MODELS
    }
    outer_admitted = {method: np.zeros(prediction_shape, dtype=bool) for method in METHODS}
    outer_selected = {method: formal.copy() for method in METHODS}
    nested_admitted = np.zeros(prediction_shape, dtype=bool)
    nested_selected = formal.copy()
    nested_method = np.full(len(formal), -1, dtype=np.int16)
    inner_audit: dict[str, Any] = {}
    outer_parameter_arrays: dict[str, np.ndarray] = {}
    latency: dict[str, list[float]] = {model: [] for model in MODELS}

    for outer in range(5):
        outer_train = np.flatnonzero(fold != outer)
        outer_test = np.flatnonzero(fold == outer)
        inner_prediction = {
            model: {key: np.full(prediction_shape, np.nan, dtype=np.float32) for key in outer_prediction[model]}
            for model in MODELS
        }
        inner_records = []
        for inner in range(5):
            if inner == outer:
                continue
            inner_train = np.flatnonzero((fold != outer) & (fold != inner))
            inner_test = np.flatnonzero(fold == inner)
            if set(data["group"][inner_train]) & set(data["group"][inner_test]):
                raise RuntimeError("inner route leakage")
            for model_name in MODELS:
                ensemble = fit_ensemble(model_name, feature, pool, objective_gain, dual, harm, inner_train, data["group"], args.device)
                prediction = predict_ensemble(model_name, ensemble, feature[inner_test], args.device)
                for key, value in prediction.items():
                    inner_prediction[model_name][key][inner_test] = value
                del ensemble
                if args.device == "cuda":
                    torch.cuda.empty_cache()
            inner_records.append({"inner_fold": inner, "train_states": int(len(inner_train)), "test_states": int(len(inner_test)), "route_overlap": 0})
            print(f"v77 outer={outer} inner={inner} complete", flush=True)
        method_records = {}
        feasible = []
        for method in METHODS:
            model_name, profile_name = method.split("__", 1)
            admitted, selected = apply_profile(
                {key: value[outer_train] for key, value in inner_prediction[model_name].items()},
                profile_name,
                pool[outer_train],
                formal[outer_train],
            )
            summary = evaluate_selection(data, selected, outer_train)
            passes, checks = inner_feasible(summary)
            method_records[method] = {"feasible": passes, "checks": checks, "selection": summary, "admitted_candidates": int(admitted.sum())}
            if passes:
                feasible.append(method)
        if feasible:
            chosen = sorted(
                feasible,
                key=lambda method: (
                    method_records[method]["selection"]["delta_mean_m"]["objective"],
                    -method_records[method]["selection"]["replacements"],
                    0 if method.endswith("conservative_top5") else 1,
                    0 if method.startswith(MODELS[1]) else 1,
                    method,
                ),
            )[0]
        else:
            chosen = None
        for model_name in MODELS:
            ensemble = fit_ensemble(model_name, feature, pool, objective_gain, dual, harm, outer_train, data["group"], args.device)
            prediction = predict_ensemble(model_name, ensemble, feature[outer_test], args.device)
            for key, value in prediction.items():
                outer_prediction[model_name][key][outer_test] = value
            serialize_outer_models(f"outer{outer}__{model_name}", model_name, ensemble, outer_parameter_arrays)
            latency[model_name].append(ensemble_cpu_latency(model_name, ensemble, feature[outer_test]))
            for profile_name in PROFILES:
                method = f"{model_name}__{profile_name}"
                admitted, selected = apply_profile(prediction, profile_name, pool[outer_test], formal[outer_test])
                outer_admitted[method][outer_test] = admitted
                outer_selected[method][outer_test] = selected
                if method == chosen:
                    nested_admitted[outer_test] = admitted
                    nested_selected[outer_test] = selected
                    nested_method[outer_test] = METHODS.index(method)
            del ensemble
            if args.device == "cuda":
                torch.cuda.empty_cache()
        inner_audit[str(outer)] = {
            "outer_train_states": int(len(outer_train)),
            "outer_test_states": int(len(outer_test)),
            "outer_train_routes": int(len(np.unique(data["group"][outer_train]))),
            "outer_test_routes": int(len(np.unique(data["group"][outer_test]))),
            "outer_route_overlap": int(len(set(data["group"][outer_train]) & set(data["group"][outer_test]))),
            "inner_folds": inner_records,
            "methods": method_records,
            "chosen": chosen,
        }
        print(f"v77 outer={outer} chosen={chosen} complete", flush=True)

    candidate_layer = {model: candidate_metrics(prediction, objective_gain, dual, harm, challenger) for model, prediction in outer_prediction.items()}
    old_admitted = challenger & fields["current_safe"]
    filter_layer = {"old_v70_safety": filter_summary(old_admitted, challenger, dual, harm)}
    for method in METHODS:
        filter_layer[method] = filter_summary(outer_admitted[method], challenger, dual, harm)
    filter_layer["nested_selected"] = filter_summary(nested_admitted, challenger, dual, harm)
    end_to_end = {}
    for method in METHODS:
        summary = evaluate_selection(data, outer_selected[method])
        summary["selector_audit"] = selection_extra(data, fields, outer_selected[method], formal_energy)
        end_to_end[method] = summary
    nested_summary = evaluate_selection(data, nested_selected)
    nested_summary["selector_audit"] = selection_extra(data, fields, nested_selected, formal_energy)
    end_to_end["nested_selected"] = nested_summary
    formal_summary = evaluate_selection(data, formal.copy())
    formal_eval = read_json(prior.V70_ROOT / "evaluations/scand_fresh_outer/result.json")["decisions"]["proposal_aware_v66_cem"]
    for name in METRICS:
        if abs(formal_summary["metrics"]["mean_m"][name] - formal_eval["absolute_mean_m"][name]) > 2.0e-7:
            raise RuntimeError("formal v70 SCAND reconstruction mismatch")
    recon = frozen_recon_summary()
    aggregate = read_json(prior.V70_ROOT / "aggregate_result.json")["pooled_decisions"]["proposal_aware_v66_cem"]
    pooled = {}
    for method, summary in end_to_end.items():
        mean = {
            name: float(aggregate["absolute_mean_m"][name] + (summary["metrics"]["mean_m"][name] - formal_summary["metrics"]["mean_m"][name]) * 100.0 / 788.0)
            for name in METRICS
        }
        pooled[method] = {"states": 788, "mean_m": mean, "objective_gain_m": float(aggregate["absolute_mean_m"]["objective"] - mean["objective"])}
    strict = nested_summary["strict_zero_mm"]
    success_checks = {
        "nonzero": nested_summary["replacements"] > 0,
        "scand_mean_ate": nested_summary["metrics"]["mean_m"]["ate"] <= formal_summary["metrics"]["mean_m"]["ate"] + 1.0e-12,
        "scand_mean_objective": nested_summary["metrics"]["mean_m"]["objective"] <= formal_summary["metrics"]["mean_m"]["objective"] + 1.0e-12,
        "scand_objective_p90": nested_summary["metrics"]["p90_m"]["objective"] <= formal_summary["metrics"]["p90_m"]["objective"] + 0.001,
        "dual_lcb": strict["dual"]["lcb"] >= 0.85,
        "harm_ucb": strict["harm"]["ucb"] <= 0.15,
        "pooled_gain_1mm": pooled["nested_selected"]["objective_gain_m"] >= 0.001,
        "recon_unchanged": True,
    }
    passed = all(success_checks.values())
    old_filter = filter_layer["old_v70_safety"]
    diagnostic_filter_improved = any(
        filter_layer[method]["recovered_opportunity_states"] > old_filter["recovered_opportunity_states"]
        and filter_layer[method]["harm_rejection"]["rate"] >= old_filter["harm_rejection"]["rate"]
        for method in METHODS
    )
    if passed:
        verdict, failure_stage = "development_candidate_pass", None
    elif diagnostic_filter_improved and nested_summary["replacements"] == 0:
        verdict, failure_stage = "filter_layer_improved/end_to_end_reranking_fail", "reranking_or_inner_certification"
    elif all(value["chosen"] is None for value in inner_audit.values()):
        verdict, failure_stage = "v77_fail", "representation_or_risk_certification"
    else:
        verdict, failure_stage = "v77_fail", "end_to_end_selector"
    runtime = float(time.perf_counter() - started)
    result = {
        "schema": "v77_scand_tail_joint_selector_result_v1",
        "status": "completed_spent_development_stopped",
        "verdict": verdict,
        "failure_stage": failure_stage,
        "route_inventory_sha256": sha256_file(OUTPUT / "route_inventory.json"),
        "preregistration_sha256": sha256_file(PREREG),
        "reserve_accessed": False,
        "formal_v70_mutated": False,
        "dwm_ltwm_cem_rerun": False,
        "recon": recon,
        "formal_scand_v70": formal_summary,
        "candidate_layer": candidate_layer,
        "filter_layer": filter_layer,
        "end_to_end_scand": end_to_end,
        "pooled_recon_scand": pooled,
        "nested_fold_choices": {key: value["chosen"] for key, value in inner_audit.items()},
        "latency_cpu": {
            model: {"mean_seconds_per_state": float(np.mean(values)), "p90_fold_seconds_per_state": float(np.quantile(values, 0.9)), "fold_values": values}
            for model, values in latency.items()
        },
        "formal_success_gate": {"checks": success_checks, "passed": passed},
        "development_evidence_only": True,
        "independent_validation_claim": False,
        "next_step_if_passed": "one pre-registered untouched SCAND-route validation; not run automatically" if passed else None,
        "stopped_after_v77": True,
        "runtime_seconds": runtime,
    }
    prediction_arrays: dict[str, np.ndarray] = {
        "record_key": data["record_key"].astype("U256"),
        "route": data["group"].astype("U256"),
        "outer_fold": fold.astype(np.int16),
        "physical_support_pool": pool,
        "challenger_pool": challenger,
        "formal_v70_index": formal.astype(np.int16),
        "objective_gain_truth": objective_gain,
        "dual_truth": dual,
        "harm_truth": harm,
        "nested_admitted": nested_admitted,
        "nested_selected_index": nested_selected.astype(np.int16),
        "nested_method_code": nested_method,
    }
    for model, prediction in outer_prediction.items():
        prefix = "linear" if model == MODELS[0] else "listwise"
        for key, value in prediction.items():
            prediction_arrays[f"{prefix}__{key}"] = value
    for method in METHODS:
        prefix = method.replace(MODELS[0], "linear").replace(MODELS[1], "listwise")
        prediction_arrays[f"{prefix}__admitted"] = outer_admitted[method]
        prediction_arrays[f"{prefix}__selected_index"] = outer_selected[method].astype(np.int16)
    write_npz(PREDICTIONS, **prediction_arrays)
    write_npz(MODEL_PARAMETERS, **outer_parameter_arrays)
    model_audit = {
        "schema": "v77_model_parameters_audit_v1",
        "models": prereg["models"],
        "risk_profiles": prereg["risk_and_fallback_profiles"],
        "feature_names": feature_names,
        "outer_parameter_arrays": {name: list(value.shape) for name, value in outer_parameter_arrays.items()},
        "parameter_array_count": len(outer_parameter_arrays),
    }
    write_json(MODEL_AUDIT, model_audit)
    hashes_after = {name: sha256_file(ROOT / record["path"]) for name, record in prereg["inputs"].items()}
    unchanged = all(hashes_after[name] == record["sha256"] for name, record in prereg["inputs"].items())
    audit = {
        "schema": "v77_scand_tail_joint_selector_audit_v1",
        "input_hashes_before": hashes_before,
        "input_hashes_after": hashes_after,
        "all_inputs_unchanged": unchanged,
        "route_folds": fold_audit,
        "nested_route_audit": inner_audit,
        "route_overlap_violations": int(sum(value["outer_route_overlap"] for value in inner_audit.values())),
        "candidate_reconstruction": reconstruction_audit,
        "structural_cache": structural_audit,
        "feature_contract": feature_audit,
        "valid_exact": bool(np.array_equal(fields["current_valid"], data["valid"])),
        "safe_exact_but_feature_only": bool(np.array_equal(fields["current_safe"], data["safe"])),
        "old_safety_hard_gate": False,
        "frozen_energy_hard_gate": False,
        "unobservable_draws_excluded": True,
        "runner_sha256": sha256_file(Path(__file__)),
        "implementation_errata": [
            {
                "stage": "pre_outer_feature_assembly",
                "failure": "incumbent_flag shape 1x65 was not explicitly broadcast to 100x65",
                "outer_results_read": False,
                "artifacts_produced": False,
                "scientific_contract_changed": False,
                "correction": "explicit numpy broadcast only",
            }
        ],
    }
    if not unchanged:
        raise RuntimeError("immutable v77 input changed")
    write_json(RESULT, result)
    write_json(AUDIT, audit)
    manifest = {
        "schema": "v77_sha256_manifest_v1",
        "artifacts": {
            path.name: {"sha256": sha256_file(path), "bytes": path.stat().st_size}
            for path in (OUTPUT / "route_inventory.json", PREREG, RESULT, AUDIT, PREDICTIONS, MODEL_PARAMETERS, MODEL_AUDIT, Path(__file__))
        },
    }
    write_json(MANIFEST, manifest)
    print(json.dumps({
        "verdict": verdict,
        "failure_stage": failure_stage,
        "nested_choices": result["nested_fold_choices"],
        "replacements": nested_summary["replacements"],
        "dual": strict["dual"],
        "harm": strict["harm"],
        "scand_objective_delta_m": nested_summary["delta_mean_m"]["objective"],
        "pooled_objective_gain_m": pooled["nested_selected"]["objective_gain_m"],
        "runtime_seconds": runtime,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
