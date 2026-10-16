#!/usr/bin/env python3
"""Strict nested route-OOF direct-delta diagnostic on spent v70 candidates.

This runner never generates candidates or calls DWM/LTWM.  It predicts the
continuous metric gain of each cached candidate relative to the formal-v70
fallback.  Every method, residual quantile, and admission margin is chosen on
inner route-OOF predictions; held-out outer labels are evaluation-only.
"""

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
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.evaluation import run_v72_proposal_tail_reranker as v72  # noqa: E402


OUTPUT = ROOT / "outputs/cem_development_v1/v74_direct_delta_nested"
PREREG = OUTPUT / "preregistration.json"
V70_RESULT = ROOT / "outputs/cem_development_v1/dualdomain_cem_tail_v70_broad_hashfold_archive/result.json"
V72_RESULT = ROOT / "outputs/cem_development_v1/v72_proposal_tail_reranker/result.json"
V72_PREDICTIONS = ROOT / "outputs/cem_development_v1/v72_proposal_tail_reranker/oof_predictions.npz"
V73_RESULT = ROOT / "outputs/cem_development_v1/v73_nested_safety_calibration/result.json"
V73_PREDICTIONS = ROOT / "outputs/cem_development_v1/v73_nested_safety_calibration/outer_oof_predictions.npz"
METHODS = ("ridge_delta_compact", "huber_mlp_delta_extended")
REPORTED = (*METHODS, "nested_selected")
METRICS = ("ate", "rpe", "endpoint", "objective")
QUANTILES = (0.90, 0.95, 0.975, 0.99)
OBJECTIVE_MARGINS = (0.0, 0.001, 0.002, 0.004)
Z = 1.96
TOLERANCE = 1.0e-7


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser.parse_args()


def verify_contract() -> dict[str, str]:
    prereg = v72.read_json(PREREG)
    if prereg.get("status") != "frozen_before_v74_computation":
        raise RuntimeError("v74 preregistration is not frozen")
    if not prereg["reserve"]["forbidden"] or prereg["reserve"]["accessed_by_v74"]:
        raise RuntimeError("v74 forbidden-cohort contract changed")
    paths = {
        "formal_v70_result_sha256": V70_RESULT,
        "v72_result_sha256": V72_RESULT,
        "v72_oof_predictions_sha256": V72_PREDICTIONS,
        "v73_result_sha256": V73_RESULT,
        "v73_outer_oof_predictions_sha256": V73_PREDICTIONS,
    }
    observed = {key: v72.sha256_file(path) for key, path in paths.items()}
    expected = prereg["immutable_inputs"]
    for key, digest in observed.items():
        if digest != expected[key]:
            raise RuntimeError(f"immutable input changed: {key}")
    return {"preregistration_sha256": v72.sha256_file(PREREG), **observed}


def union_structural_for_cohort(
    spec: dict[str, Any], cohort: dict[str, Any]
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    replay = v72.V70_ROOT / "replays" / spec["name"] / "proposal_aware_v70"
    freeze = v72.read_json(replay / "prelabel_freeze.json")
    paths = [v72.resolve_path(freeze["base_features"])] + [
        replay / f"iteration_{iteration:02d}" / "cem_features.npz"
        for iteration in range(v72.ITERATIONS)
    ]
    v72.reject_forbidden_paths(paths)
    pools = []
    hashes = {}
    for source, path in enumerate(paths):
        payload = v72.load_npz(path)
        required = {
            "candidates",
            "actions_h4",
            "raw_ltwm_features",
            "scorer_v2_aux_features",
            "ate_by_seed",
            "rtp_by_seed",
            "support_by_seed",
            "record_key",
        }
        missing = sorted(required - set(payload))
        if missing:
            raise RuntimeError(f"missing structural cache fields {missing}: {path}")
        if not np.array_equal(payload["record_key"].astype(str), cohort["record_key"]):
            raise RuntimeError(f"structural record mismatch: {spec['name']} source={source}")
        pools.append(payload)
        hashes[f"source_{source}"] = v72.sha256_file(path)
    states, candidates = cohort["candidates"].shape[:2]
    arrays = {
        "actions_h4": np.empty((states, candidates, 4, 3), dtype=np.float32),
        "raw_ltwm": np.empty((states, candidates, 112), dtype=np.float32),
        "scorer_aux": np.empty((states, candidates, 10), dtype=np.float32),
        "ate_seed": np.empty((states, candidates, 3), dtype=np.float32),
        "rtp_seed": np.empty((states, candidates, 3), dtype=np.float32),
        "support_seed": np.empty((states, candidates, 3), dtype=np.float32),
        "source_rank": np.empty((states, candidates), dtype=np.float32),
    }
    reconstructed = np.empty_like(cohort["candidates"])
    for state in range(states):
        seen: set[bytes] = set()
        cursor = 0
        for payload in pools:
            local_candidates = payload["candidates"].astype(np.float32)
            for index in range(local_candidates.shape[1]):
                key = v72.exact_action_key(local_candidates[state, index])
                if key in seen:
                    continue
                seen.add(key)
                reconstructed[state, cursor] = local_candidates[state, index]
                arrays["actions_h4"][state, cursor] = payload["actions_h4"][state, index]
                arrays["raw_ltwm"][state, cursor] = payload["raw_ltwm_features"][state, index]
                arrays["scorer_aux"][state, cursor] = payload["scorer_v2_aux_features"][state, index]
                arrays["ate_seed"][state, cursor] = payload["ate_by_seed"][:, state, index]
                arrays["rtp_seed"][state, cursor] = payload["rtp_by_seed"][:, state, index]
                arrays["support_seed"][state, cursor] = payload["support_by_seed"][:, state, index]
                arrays["source_rank"][state, cursor] = index / max(local_candidates.shape[1] - 1, 1)
                cursor += 1
        if cursor != candidates:
            raise RuntimeError(f"structural union cardinality changed: {spec['name']} state={state} got={cursor}")
    if not np.array_equal(reconstructed, cohort["candidates"]):
        raise RuntimeError(f"structural candidate order mismatch: {spec['name']}")
    return arrays, {
        "cohort": spec["name"],
        "available_fields": sorted(required),
        "source_hashes": hashes,
    }


def relative(value: np.ndarray, formal: np.ndarray) -> np.ndarray:
    row = np.arange(len(value))
    return value - value[row, formal][:, None]


def sequence_summary(value: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(value, axis=-1)
    first = np.linalg.norm(np.diff(value, axis=2), axis=-1)
    second = np.linalg.norm(np.diff(value, n=2, axis=2), axis=-1)
    endpoint = np.linalg.norm(value[..., -1, :], axis=-1)[..., None]
    path = np.sum(first, axis=-1, keepdims=True)
    smooth = np.sum(second, axis=-1, keepdims=True)
    return np.concatenate((norm, first, second, endpoint, path, smooth), axis=-1).astype(np.float32)


def percentile_rank(value: np.ndarray, mask: np.ndarray) -> np.ndarray:
    result = np.ones(value.shape, dtype=np.float32)
    for state in range(len(value)):
        index = np.flatnonzero(mask[state] & np.isfinite(value[state]))
        if not len(index):
            continue
        order = index[np.argsort(value[state, index], kind="stable")]
        result[state, order] = np.arange(len(order), dtype=np.float32) / max(len(order) - 1, 1)
    return result[..., None]


def local_density(value: np.ndarray, mask: np.ndarray) -> np.ndarray:
    states, candidates = value.shape[:2]
    flat = value.reshape(states, candidates, -1).astype(np.float64)
    result = np.zeros((states, candidates, 4), dtype=np.float32)
    for state in range(states):
        distance = np.sqrt(np.sum((flat[state, :, None] - flat[state, None, :]) ** 2, axis=-1))
        distance[~mask[state], :] = np.inf
        distance[:, ~mask[state]] = np.inf
        np.fill_diagonal(distance, np.inf)
        ordered = np.sort(distance, axis=1)
        count = int(mask[state].sum())
        if count <= 1:
            continue
        finite_mean = np.sum(np.where(np.isfinite(distance), distance, 0.0), axis=1) / max(count - 1, 1)
        result[state, :, 0] = np.where(np.isfinite(ordered[:, 0]), ordered[:, 0], 0.0)
        result[state, :, 1] = np.mean(np.where(np.isfinite(ordered[:, : min(3, count - 1)]), ordered[:, : min(3, count - 1)], 0.0), axis=1)
        result[state, :, 2] = np.mean(np.where(np.isfinite(ordered[:, : min(5, count - 1)]), ordered[:, : min(5, count - 1)], 0.0), axis=1)
        result[state, :, 3] = finite_mean
    return result


def assemble_features(
    data: dict[str, Any], structural: dict[str, np.ndarray]
) -> tuple[np.ndarray, np.ndarray, list[str], list[str], dict[str, Any]]:
    formal = data["formal_index"]
    eligible = data["valid"] & data["safe"]
    action = structural["actions_h4"].astype(np.float32)
    waypoint = data["candidates"].astype(np.float32)
    scorer = structural["scorer_aux"].astype(np.float32)
    seed = np.concatenate(
        (structural["ate_seed"], structural["rtp_seed"], structural["support_seed"]),
        axis=-1,
    ).astype(np.float32)
    seed_summary_parts = []
    seed_summary_names = []
    for name, value in (
        ("ate", structural["ate_seed"]),
        ("rtp", structural["rtp_seed"]),
        ("support", structural["support_seed"]),
    ):
        seed_summary_parts.extend(
            (value.mean(axis=-1, keepdims=True), value.std(axis=-1, keepdims=True), np.ptp(value, axis=-1, keepdims=True))
        )
        seed_summary_names.extend((f"seed_{name}_mean", f"seed_{name}_std", f"seed_{name}_range"))
    seed_summary = np.concatenate(seed_summary_parts, axis=-1).astype(np.float32)
    action_summary = sequence_summary(action)
    waypoint_summary = sequence_summary(waypoint)
    extension = data["extended"][..., -13:].astype(np.float32)
    source = np.eye(4, dtype=np.float32)[data["source_id"]]
    rank = np.concatenate(
        (
            percentile_rank(data["scorer_proxy"], eligible),
            percentile_rank(data["v70_energy"], eligible & data["v70_score_available"]),
        ),
        axis=-1,
    )
    density = np.concatenate((local_density(waypoint * 0.25, eligible), local_density(action, eligible)), axis=-1)
    domain = np.zeros((*eligible.shape, 2), dtype=np.float32)
    domain[..., 0] = (data["domain"].astype(str) == "recon")[:, None]
    domain[..., 1] = (data["domain"].astype(str) == "scand")[:, None]
    formal_flag = np.zeros((*eligible.shape, 1), dtype=np.float32)
    formal_flag[np.arange(len(formal)), formal, 0] = 1.0
    flags = np.concatenate(
        (data["valid"][..., None], data["safe"][..., None], data["v70_score_available"][..., None], formal_flag),
        axis=-1,
    ).astype(np.float32)
    blocks = [
        scorer,
        relative(scorer, formal),
        seed,
        relative(seed, formal),
        seed_summary,
        relative(seed_summary, formal),
        action.reshape(*action.shape[:2], -1),
        relative(action, formal).reshape(*action.shape[:2], -1),
        action_summary,
        relative(action_summary, formal),
        waypoint.reshape(*waypoint.shape[:2], -1),
        relative(waypoint, formal).reshape(*waypoint.shape[:2], -1),
        waypoint_summary,
        relative(waypoint_summary, formal),
        extension,
        relative(extension, formal),
        source,
        structural["source_rank"][..., None],
        rank,
        density,
        domain,
        flags,
    ]
    names = []
    names += [f"scorer_abs_{i}" for i in range(10)] + [f"scorer_delta_{i}" for i in range(10)]
    names += [f"seed_abs_{i}" for i in range(9)] + [f"seed_delta_{i}" for i in range(9)]
    names += seed_summary_names + [f"{name}_delta" for name in seed_summary_names]
    names += [f"action_abs_{i}" for i in range(12)] + [f"action_delta_{i}" for i in range(12)]
    names += [f"action_summary_{i}" for i in range(action_summary.shape[-1])] + [f"action_summary_delta_{i}" for i in range(action_summary.shape[-1])]
    names += [f"waypoint_abs_{i}" for i in range(8)] + [f"waypoint_delta_{i}" for i in range(8)]
    names += [f"waypoint_summary_{i}" for i in range(waypoint_summary.shape[-1])] + [f"waypoint_summary_delta_{i}" for i in range(waypoint_summary.shape[-1])]
    names += [f"v70_extension_abs_{i}" for i in range(13)] + [f"v70_extension_delta_{i}" for i in range(13)]
    names += [f"source_{i}" for i in range(4)] + ["source_rank", "scorer_rank", "energy_rank"]
    names += [f"local_density_{i}" for i in range(8)]
    names += ["domain_recon", "domain_scand", "valid", "safe", "v70_score_available", "formal_selected"]
    compact = np.concatenate(blocks, axis=-1).astype(np.float32)
    if compact.shape[-1] != len(names):
        raise RuntimeError(f"compact feature name mismatch: {compact.shape[-1]} != {len(names)}")
    raw = structural["raw_ltwm"].astype(np.float32)
    extended = np.concatenate((compact, raw, relative(raw, formal)), axis=-1).astype(np.float32)
    extended_names = names + [f"raw_ltwm_abs_{i}" for i in range(112)] + [f"raw_ltwm_delta_{i}" for i in range(112)]
    if not np.isfinite(compact).all() or not np.isfinite(extended).all():
        raise RuntimeError("non-finite v74 structural features")
    audit = {
        "compact_dimension": int(compact.shape[-1]),
        "extended_dimension": int(extended.shape[-1]),
        "states": int(len(compact)),
        "candidates_per_state": int(compact.shape[1]),
        "eligible_valid_safe_candidates": int(eligible.sum()),
        "new_cached_signals_present": {
            "actions_h4": True,
            "three_seed_ate_rtp_support": True,
            "candidate_local_density": True,
            "proposal_source_and_rank": True,
            "raw_ltwm_absolute_and_relative": True,
        },
        "v72_or_v73_probabilities_used": False,
    }
    return compact, extended, names, extended_names, audit


def load_data() -> tuple[dict[str, Any], dict[str, Any]]:
    data, base_audit = v72.load_data()
    structures = []
    structural_audit = []
    for spec in v72.COHORTS:
        mask = data["cohort"].astype(str) == spec["name"]
        local = v72.subset(data, mask)
        value, audit = union_structural_for_cohort(spec, local)
        structures.append(value)
        structural_audit.append(audit)
    keys = structures[0].keys()
    structural = {key: np.concatenate([value[key] for value in structures]) for key in keys}
    compact, extended, compact_names, extended_names, feature_audit = assemble_features(data, structural)
    data["v74_compact"] = compact
    data["v74_extended"] = extended
    data["gain"] = (
        data["true_metrics"][np.arange(len(data["record_key"])), data["formal_index"]][:, None]
        - data["true_metrics"]
    ).astype(np.float32)
    return data, {
        "base_v72_data_audit": base_audit,
        "structural_sources": structural_audit,
        "feature_audit": feature_audit,
        "compact_feature_names": compact_names,
        "extended_feature_names_sha256": hashlib.sha256("\n".join(extended_names).encode()).hexdigest(),
    }


def train_mask(data: dict[str, Any], states: np.ndarray) -> np.ndarray:
    state = np.zeros(len(data["record_key"]), dtype=bool)
    state[states] = True
    mask = state[:, None] & data["valid"] & data["safe"]
    mask[np.arange(len(state)), data["formal_index"]] = False
    return mask


def fit_ridge(features: np.ndarray, data: dict[str, Any], states: np.ndarray) -> dict[str, Any]:
    mask = train_mask(data, states)
    norm = v72.fit_normalizer(features, mask)
    x = v72.normalize(features, norm)[mask].astype(np.float32)
    y = data["gain"][mask].astype(np.float32)
    x = np.concatenate((x, np.ones((len(x), 1), dtype=np.float32)), axis=1)
    matrix = (x.T @ x).astype(np.float64)
    rhs = (x.T @ y).astype(np.float64)
    matrix.flat[:: len(matrix) + 1] += 50.0
    matrix[-1, -1] -= 50.0
    coefficient = np.linalg.solve(matrix, rhs).astype(np.float32)
    return {"normalization": norm, "coefficient": coefficient, "training_rows": int(mask.sum())}


def predict_ridge(model: dict[str, Any], features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = v72.normalize(features, model["normalization"])
    mean = np.einsum("nkd,dm->nkm", x, model["coefficient"][:-1]) + model["coefficient"][-1]
    return mean.astype(np.float32), np.zeros_like(mean, dtype=np.float32)


class DeltaMLP(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(dimension, 96),
            nn.SiLU(),
            nn.Linear(96, 48),
            nn.SiLU(),
            nn.Linear(48, 4),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.network(value)


def fit_mlp_member(
    features: np.ndarray,
    data: dict[str, Any],
    states: np.ndarray,
    norm: dict[str, np.ndarray],
    *,
    device: str,
    seed: int,
) -> tuple[DeltaMLP, list[float]]:
    mask = train_mask(data, states)
    x = torch.from_numpy(v72.normalize(features, norm)[mask]).to(device)
    y = torch.from_numpy(data["gain"][mask].astype(np.float32)).to(device)
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = DeltaMLP(x.shape[-1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3, weight_decay=1.0e-4)
    losses = []
    generator = torch.Generator(device=device).manual_seed(seed + 100)
    for _ in range(24):
        order = torch.randperm(len(x), generator=generator, device=device)
        total = 0.0
        seen = 0
        for start in range(0, len(order), 4096):
            index = order[start : start + 4096]
            prediction = model(x[index])
            loss = torch.nn.functional.smooth_l1_loss(prediction, y[index], beta=0.01)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(index)
            seen += len(index)
        losses.append(total / seen)
    return model.eval(), losses


def fit_mlp_ensemble(
    features: np.ndarray,
    data: dict[str, Any],
    states: np.ndarray,
    *,
    device: str,
    seed_offset: int,
) -> dict[str, Any]:
    mask = train_mask(data, states)
    norm = v72.fit_normalizer(features, mask)
    models = []
    loss = []
    for base in (7401, 7402, 7403):
        model, history = fit_mlp_member(
            features, data, states, norm, device=device, seed=base + seed_offset
        )
        models.append(model)
        loss.append(history)
    return {"normalization": norm, "models": models, "loss": loss, "training_rows": int(mask.sum())}


def predict_mlp(
    model: dict[str, Any], features: np.ndarray, *, device: str
) -> tuple[np.ndarray, np.ndarray]:
    normalized = v72.normalize(features, model["normalization"])
    outputs = []
    for member in model["models"]:
        chunks = []
        with torch.inference_mode():
            for start in range(0, len(normalized), 128):
                tensor = torch.from_numpy(normalized[start : start + 128]).to(device)
                chunks.append(member(tensor).cpu().numpy().astype(np.float32))
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
    if method == "ridge_delta_compact":
        model = fit_ridge(data["v74_compact"], data, train_states)
        mean, std = predict_ridge(model, data["v74_compact"][predict_states])
        final_loss = None
    else:
        model = fit_mlp_ensemble(
            data["v74_extended"], data, train_states, device=device, seed_offset=seed_offset
        )
        mean, std = predict_mlp(model, data["v74_extended"][predict_states], device=device)
        final_loss = [float(value[-1]) for value in model["loss"]]
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
    elapsed = time.perf_counter() - started
    return mean, std, {
        "method": method,
        "train_states": int(len(train_states)),
        "predict_states": int(len(predict_states)),
        "training_rows": int(train_mask(data, train_states).sum()),
        "elapsed_sec": elapsed,
        "amortized_fit_plus_batched_predict_ms_per_predicted_state": 1000.0 * elapsed / max(len(predict_states), 1),
        "final_member_losses": final_loss,
    }


def outer_cache_path(outer: int) -> Path:
    return OUTPUT / "nested_cache" / f"outer_{outer:02d}.npz"


def build_outer_cache(
    data: dict[str, Any], outer: int, *, device: str
) -> dict[str, np.ndarray]:
    path = outer_cache_path(outer)
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
            raise RuntimeError(f"v74 nested cache identity changed: outer={outer}")
        return cached
    inner_mean = {method: np.full((*data["valid"].shape, 4), np.nan, dtype=np.float32) for method in METHODS}
    inner_std = {method: np.full((*data["valid"].shape, 4), np.nan, dtype=np.float32) for method in METHODS}
    records = []
    for inner in range(5):
        if inner == outer:
            continue
        train = np.flatnonzero((data["fold"] != outer) & (data["fold"] != inner))
        valid = np.flatnonzero(data["fold"] == inner)
        train_groups = set(data["group"][train].astype(str))
        valid_groups = set(data["group"][valid].astype(str))
        if train_groups & valid_groups:
            raise RuntimeError("inner route leakage")
        for method_index, method in enumerate(METHODS):
            mean, std, audit = fit_predict_method(
                method,
                data,
                train,
                valid,
                device=device,
                seed_offset=outer * 100 + inner * 10 + method_index,
            )
            inner_mean[method][valid] = mean
            inner_std[method][valid] = std
            records.append({"outer_fold": outer, "inner_fold": inner, "route_overlap": 0, **audit})
            print(json.dumps({"stage": "inner_fit", **records[-1]}, sort_keys=True), flush=True)
    outer_records = []
    outer_mean = {}
    outer_std = {}
    for method_index, method in enumerate(METHODS):
        mean, std, audit = fit_predict_method(
            method,
            data,
            outer_train,
            outer_test,
            device=device,
            seed_offset=outer * 1000 + method_index,
        )
        outer_mean[method] = mean
        outer_std[method] = std
        outer_records.append({"outer_fold": outer, "route_overlap": 0, **audit})
        print(json.dumps({"stage": "outer_fit", **outer_records[-1]}, sort_keys=True), flush=True)
    arrays: dict[str, np.ndarray] = {
        "inner_state_index": outer_train,
        "outer_state_index": outer_test,
        "inner_record_key": data["record_key"][outer_train],
        "outer_record_key": data["record_key"][outer_test],
        "audit_json": np.asarray(json.dumps({"inner": records, "outer": outer_records}, sort_keys=True)),
    }
    for method in METHODS:
        local_mean = inner_mean[method][outer_train]
        local_std = inner_std[method][outer_train]
        if not np.isfinite(local_mean).all() or not np.isfinite(local_std).all():
            raise RuntimeError(f"incomplete inner OOF direct-delta prediction: outer={outer} method={method}")
        arrays[f"{method}__inner_mean"] = local_mean
        arrays[f"{method}__inner_std"] = local_std
        arrays[f"{method}__outer_mean"] = outer_mean[method]
        arrays[f"{method}__outer_std"] = outer_std[method]
    v72.write_npz(path, **arrays)
    return arrays


def expand_inner(data: dict[str, Any], cached: dict[str, np.ndarray], method: str) -> np.ndarray:
    result = np.full((*data["valid"].shape, 4), np.nan, dtype=np.float32)
    result[cached["inner_state_index"].astype(np.int64)] = cached[f"{method}__inner_mean"]
    return result


def wilson(successes: int, total: int) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 1.0
    p = successes / total
    denominator = 1.0 + Z * Z / total
    center = (p + Z * Z / (2.0 * total)) / denominator
    radius = Z * math.sqrt(p * (1.0 - p) / total + Z * Z / (4.0 * total * total)) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def apply_policy(
    data: dict[str, Any], states: np.ndarray, prediction: np.ndarray, policy: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    formal = data["formal_index"][states].astype(np.int64)
    selected = formal.copy()
    lcb = prediction - np.asarray(policy["residual_upper_m"], dtype=np.float32)
    eligible = data["valid"][states] & data["safe"][states]
    eligible[np.arange(len(states)), formal] = False
    passing = eligible & (lcb[..., 0] >= 0.0) & (lcb[..., 1] >= 0.0) & (lcb[..., 2] >= 0.0) & (lcb[..., 3] >= float(policy["objective_margin_m"]))
    if policy.get("active"):
        for row in range(len(states)):
            index = np.flatnonzero(passing[row])
            if len(index):
                selected[row] = int(index[np.argmax(lcb[row, index, 3])])
    return selected, lcb


def incremental_metrics(data: dict[str, Any], states: np.ndarray, selected: np.ndarray) -> dict[str, Any]:
    formal = data["formal_index"][states]
    replacement = selected != formal
    row = np.arange(len(states))
    selected_metric = data["true_metrics"][states, selected]
    formal_metric = data["true_metrics"][states, formal]
    gain = formal_metric - selected_metric
    dual = replacement & data["dual_vs_formal"][states, selected].astype(bool)
    harm = replacement & data["harm_vs_formal"][states, selected].astype(bool)
    count = int(replacement.sum())
    dual_count = int(dual.sum())
    harm_count = int(harm.sum())
    dual_interval = wilson(dual_count, count)
    harm_interval = wilson(harm_count, count)
    return {
        "states": int(len(states)),
        "replacements": count,
        "acceptance_rate": float(replacement.mean()),
        "dual_count": dual_count,
        "dual_precision": float(dual_count / count) if count else None,
        "dual_wilson_95": {"lower": dual_interval[0], "upper": dual_interval[1]},
        "harm_count": harm_count,
        "harm_rate": float(harm_count / count) if count else None,
        "harm_wilson_95": {"lower": harm_interval[0], "upper": harm_interval[1]},
        "mean_gain_m": {name: float(gain[:, index].mean()) for index, name in enumerate(METRICS)},
    }


def fit_policy(
    data: dict[str, Any], states: np.ndarray, prediction: np.ndarray, method: str
) -> dict[str, Any]:
    formal = data["formal_index"][states]
    local_prediction = prediction[states]
    eligible = data["valid"][states] & data["safe"][states]
    eligible[np.arange(len(states)), formal] = False
    true_gain = data["gain"][states]
    residual = local_prediction - true_gain
    state_worst = []
    for row in range(len(states)):
        if eligible[row].any():
            state_worst.append(np.max(residual[row, eligible[row]], axis=0))
    state_worst_value = np.asarray(state_worst, dtype=np.float32)
    evaluations = []
    for quantile in QUANTILES:
        residual_upper = np.quantile(state_worst_value, quantile, axis=0, method="higher").astype(np.float32)
        for margin in OBJECTIVE_MARGINS:
            policy = {
                "method": method,
                "active": True,
                "residual_quantile": quantile,
                "residual_upper_m": residual_upper.tolist(),
                "objective_margin_m": margin,
            }
            selected, _ = apply_policy(data, states, local_prediction, policy)
            metrics = incremental_metrics(data, states, selected)
            feasible = (
                metrics["replacements"] > 0
                and metrics["dual_wilson_95"]["lower"] >= 0.85
                and metrics["harm_wilson_95"]["upper"] <= 0.15
                and metrics["mean_gain_m"]["objective"] > 0.0
            )
            evaluations.append({"parameters": {key: policy[key] for key in ("residual_quantile", "residual_upper_m", "objective_margin_m")}, "metrics": metrics, "feasible": feasible})
    feasible = [value for value in evaluations if value["feasible"]]
    pool = feasible if feasible else evaluations
    best = max(
        pool,
        key=lambda value: (
            value["metrics"]["mean_gain_m"]["objective"],
            value["metrics"]["replacements"],
            value["parameters"]["residual_quantile"],
            value["parameters"]["objective_margin_m"],
        ),
    )
    return {
        "method": method,
        "active": bool(best["feasible"]),
        **best["parameters"],
        "inner_metrics": best["metrics"],
        "fallback_reason": None if best["feasible"] else "no_inner_wilson_feasible_direct_delta_policy",
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
                policy["method"] == "ridge_delta_compact",
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


def compact_policy(policy: dict[str, Any]) -> dict[str, Any]:
    keep = (
        "method", "active", "residual_quantile", "residual_upper_m",
        "objective_margin_m", "inner_metrics", "fallback_reason",
    )
    return {key: policy[key] for key in keep if key in policy}


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


def prediction_report(data: dict[str, Any], prediction: np.ndarray) -> dict[str, Any]:
    report = {}
    for domain in ("recon", "scand"):
        state = data["domain"].astype(str) == domain
        mask = state[:, None] & data["valid"] & data["safe"]
        mask[np.arange(len(state)), data["formal_index"]] = False
        error = prediction[mask].astype(np.float64) - data["gain"][mask].astype(np.float64)
        local = {"rows": int(mask.sum()), "mae_m": {}, "rmse_m": {}, "pearson": {}}
        for index, name in enumerate(METRICS):
            local["mae_m"][name] = float(np.mean(np.abs(error[:, index])))
            local["rmse_m"][name] = float(np.sqrt(np.mean(error[:, index] ** 2)))
            x = prediction[..., index][mask].astype(np.float64)
            y = data["gain"][..., index][mask].astype(np.float64)
            local["pearson"][name] = float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else None
        report[domain] = local
    return report


def feature_signal_report(
    data: dict[str, Any], names: list[str]
) -> dict[str, Any]:
    from scipy.stats import rankdata

    features = data["v74_compact"]
    report = {}
    prefixes = (
        "scorer", "seed", "action", "waypoint", "v70_extension",
        "source", "local_density", "domain", "safe",
    )
    for domain in ("recon", "scand"):
        state = data["domain"].astype(str) == domain
        eligible = state[:, None] & data["valid"] & data["safe"]
        eligible[np.arange(len(state)), data["formal_index"]] = False
        dual = data["dual_vs_formal"].astype(bool)
        harm = data["harm_vs_formal"].astype(bool)
        comparison = eligible & (dual | harm)
        y = dual[comparison].astype(np.int64)
        rows = []
        if y.sum() and (1 - y).sum():
            for index, name in enumerate(names):
                x = features[..., index][comparison].astype(np.float64)
                rank = rankdata(x, method="average")
                positive = rank[y == 1]
                auc = (positive.sum() - len(positive) * (len(positive) + 1) / 2.0) / (len(positive) * int((y == 0).sum()))
                rows.append({"feature": name, "auc_oriented": float(max(auc, 1.0 - auc)), "raw_auc": float(auc)})
        rows.sort(key=lambda value: (value["auc_oriented"], value["feature"]), reverse=True)
        group = {}
        for prefix in prefixes:
            local = [value for value in rows if value["feature"].startswith(prefix)]
            if local:
                group[prefix] = local[0]
        report[domain] = {
            "comparison_rows": int(comparison.sum()),
            "dual_rows": int(y.sum()),
            "harm_rows": int((1 - y).sum()),
            "top_univariate_signals_posthoc_outer_diagnostic_only": rows[:15],
            "best_by_semantic_group": group,
            "used_for_policy_or_threshold_selection": False,
        }
    return report


def policy_latency_ms(
    data: dict[str, Any], predictions: dict[str, np.ndarray], policies: dict[tuple[int, str], dict[str, Any]]
) -> dict[str, Any]:
    measurements = []
    for state in range(len(data["record_key"])):
        domain = str(data["domain"][state])
        fold = int(data["fold"][state])
        policy = policies[(fold, domain)]
        started = time.perf_counter_ns()
        apply_policy(data, np.asarray([state]), predictions[policy["method"]][state : state + 1], policy)
        measurements.append((time.perf_counter_ns() - started) / 1.0e6)
    value = np.asarray(measurements)
    return {
        "mean": float(value.mean()),
        "p50": float(np.quantile(value, 0.5)),
        "p90": float(np.quantile(value, 0.9)),
        "states": int(len(value)),
        "scope": "cached four-delta lower-bound filtering and selection only; model forward and formal v70 planner excluded",
    }


def scand_tail(selection: dict[str, Any]) -> dict[str, Any]:
    value = selection["fresh_scand"]
    return {
        "objective_mean_m": value["absolute_mean_m"]["objective"],
        "objective_p90_m": value["absolute_p90_m"]["objective"],
        "ate_mean_m": value["absolute_mean_m"]["ate"],
        "ate_p90_m": value["absolute_p90_m"]["ate"],
    }


def main() -> None:
    args = parse_args()
    result_path = OUTPUT / "result.json"
    prediction_path = OUTPUT / "outer_oof_predictions.npz"
    if result_path.exists() or prediction_path.exists():
        raise FileExistsError("v74 result already exists")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    immutable = verify_contract()
    data, data_audit = load_data()
    prereg = v72.read_json(PREREG)
    route = data_audit["base_v72_data_audit"]["route_folds"]
    if route["assignment_sha256"] != prereg["nested_route_oof"]["outer_assignment_sha256"]:
        raise RuntimeError("outer route assignment changed")
    if route["route_overlap_violations"]:
        raise RuntimeError("outer route leakage")
    predictions = {
        method: np.full((*data["valid"].shape, 4), np.nan, dtype=np.float32)
        for method in METHODS
    }
    prediction_std = {method: np.full_like(predictions[method], np.nan) for method in METHODS}
    selected = {name: data["formal_index"].copy().astype(np.int64) for name in REPORTED}
    lcb = {name: np.full((*data["valid"].shape, 4), np.nan, dtype=np.float32) for name in REPORTED}
    chosen: dict[str, dict[tuple[int, str], dict[str, Any]]] = {name: {} for name in REPORTED}
    policy_audit = []
    cache_hashes = {}
    started = time.perf_counter()
    for outer in range(5):
        cached = build_outer_cache(data, outer, device=args.device)
        cache_hashes[f"outer_{outer}"] = v72.sha256_file(outer_cache_path(outer))
        outer_states_all = cached["outer_state_index"].astype(np.int64)
        for method in METHODS:
            predictions[method][outer_states_all] = cached[f"{method}__outer_mean"]
            prediction_std[method][outer_states_all] = cached[f"{method}__outer_std"]
        inner_prediction = {method: expand_inner(data, cached, method) for method in METHODS}
        for domain in ("recon", "scand"):
            inner_states = np.flatnonzero((data["fold"] != outer) & (data["domain"].astype(str) == domain))
            outer_states = np.flatnonzero((data["fold"] == outer) & (data["domain"].astype(str) == domain))
            train_groups = set(data["group"][inner_states].astype(str))
            outer_groups = set(data["group"][outer_states].astype(str))
            if train_groups & outer_groups:
                raise RuntimeError("outer route leakage")
            policies = [fit_policy(data, inner_states, inner_prediction[method], method) for method in METHODS]
            local = {policy["method"]: policy for policy in policies}
            local["nested_selected"] = choose_nested(policies)
            for name, policy in local.items():
                chosen[name][(outer, domain)] = policy
                prediction = predictions[policy["method"]][outer_states]
                local_selected, local_lcb = apply_policy(data, outer_states, prediction, policy)
                selected[name][outer_states] = local_selected
                lcb[name][outer_states] = local_lcb
            policy_audit.append({
                "outer_fold": outer,
                "domain": domain,
                "inner_states": int(len(inner_states)),
                "outer_states": int(len(outer_states)),
                "inner_route_groups": int(len(train_groups)),
                "outer_route_groups": int(len(outer_groups)),
                "route_overlap": 0,
                "outer_truth_used_for_selection": False,
                "policies": {name: compact_policy(policy) for name, policy in local.items()},
            })
            print(json.dumps({"stage": "outer_policy", "outer_fold": outer, "domain": domain, "nested": compact_policy(local["nested_selected"])}, sort_keys=True), flush=True)
    if not all(np.isfinite(value).all() for value in predictions.values()):
        raise RuntimeError("incomplete outer OOF prediction")
    formal = v72.formal_indices_summary(data)
    noncem = v72.noncem_indices_summary(data)
    methods = {}
    for name in REPORTED:
        summaries = v72.selection_summaries(data, selected[name])
        methods[name] = {
            "selection": summaries,
            "success_gate": success_gate(summaries, formal),
            "comparison_vs_formal_v70": {key: v72.method_delta(summaries[key], formal[key]) for key in summaries},
            "comparison_vs_noncem": {key: v72.method_delta(summaries[key], noncem[key]) for key in summaries},
            "scand_tail": scand_tail(summaries),
            "active_policy_counts": {
                "active": int(sum(policy["active"] for policy in chosen[name].values())),
                "fallback_only": int(sum(not policy["active"] for policy in chosen[name].values())),
                "methods": {method: int(sum(policy["method"] == method for policy in chosen[name].values())) for method in METHODS},
            },
        }
    prediction_reports = {method: prediction_report(data, predictions[method]) for method in METHODS}
    latency = policy_latency_ms(data, predictions, chosen["nested_selected"])
    feature_signal = feature_signal_report(data, data_audit["compact_feature_names"])
    output_arrays: dict[str, np.ndarray] = {
        "record_key": data["record_key"],
        "fold": data["fold"],
        "formal_index": data["formal_index"],
        "domain": data["domain"],
    }
    for method in METHODS:
        output_arrays[f"{method}__predicted_gain_mean"] = predictions[method]
        output_arrays[f"{method}__predicted_gain_std"] = prediction_std[method]
    for name in REPORTED:
        output_arrays[f"{name}__selected_index"] = selected[name]
        output_arrays[f"{name}__gain_lcb"] = lcb[name]
    v72.write_npz(prediction_path, **output_arrays)
    immutable_after = verify_contract()
    if immutable_after != immutable:
        raise RuntimeError("immutable input changed during v74")
    nested_gate = methods["nested_selected"]["success_gate"]
    result = {
        "schema": "v74_direct_delta_nested_result_v1",
        "status": "development_gate_met" if nested_gate["passed"] else "development_gate_not_met",
        "evidence_role": "spent_development_only_not_independent_validation",
        "method": "nested route-OOF direct continuous gain prediction with domain-conditional one-sided residual bounds",
        "states": int(len(data["record_key"])),
        "routes": int(prereg["nested_route_oof"]["outer_route_groups"]),
        "candidates_per_state": int(data["valid"].shape[1]),
        "formal_v70": formal,
        "noncem": noncem,
        "methods": methods,
        "prediction_reports": prediction_reports,
        "feature_signal_diagnostic": feature_signal,
        "policy_audit": policy_audit,
        "latency_ms": latency,
        "elapsed_sec": time.perf_counter() - started,
        "device": args.device,
        "audit": {
            "immutable_inputs_before": immutable,
            "immutable_inputs_after": immutable_after,
            "nested_cache_sha256": cache_hashes,
            "outer_assignment_sha256": route["assignment_sha256"],
            "outer_route_overlap_violations": route["route_overlap_violations"],
            "inner_route_overlap_violations": 0,
            "reserve_accessed": False,
            "formal_v70_mutated": False,
            "v70_cem_or_candidate_mutated": False,
            "dwm_ltwm_rerun": False,
            "v72_v73_probabilities_used_as_features": False,
            "outer_truth_used_for_selection": False,
            "prediction_sha256": v72.sha256_file(prediction_path),
        },
        "data_audit": data_audit,
    }
    v72.write_json(result_path, result)
    audit = {
        "schema": "v74_direct_delta_nested_audit_v1",
        "status": result["status"],
        "success_gate": nested_gate,
        "preregistration_sha256": immutable["preregistration_sha256"],
        "result_sha256": v72.sha256_file(result_path),
        "predictions_sha256": v72.sha256_file(prediction_path),
        "nested_cache_sha256": cache_hashes,
        "outer_assignment_sha256": route["assignment_sha256"],
        "route_overlap_violations": 0,
        "reserve_accessed": False,
        "formal_v70_mutated": False,
        "dwm_ltwm_rerun": False,
    }
    v72.write_json(OUTPUT / "audit.json", audit)
    print(json.dumps({
        "stage": "complete",
        "status": result["status"],
        "success_gate": nested_gate,
        "selection": methods["nested_selected"]["selection"]["pooled"],
        "elapsed_sec": result["elapsed_sec"],
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
