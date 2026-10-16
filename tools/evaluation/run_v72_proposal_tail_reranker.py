#!/usr/bin/env python3
"""Route-OOF proposal-tail reranker for the spent v70 development caches.

The runner never generates candidates and never calls DWM/LTWM.  It consumes
only the frozen fixed-K17 plus three v70 proposal rounds, and it keeps the
formal-v70 selected action as the fallback.  Future-pose labels are visible
only to training folds and to post-hoc evaluation of the held-out route fold.
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
OUTPUT = ROOT / "outputs/cem_development_v1/v72_proposal_tail_reranker"
PREREG = OUTPUT / "preregistration.json"
V70_ROOT = (
    ROOT
    / "outputs/cem_development_v1/dualdomain_cem_tail_v70_broad_hashfold_archive"
)
ORACLE_RESULT = (
    ROOT / "outputs/cem_development_v1/v70_oracle_diagnostic/result.json"
)
ITERATIONS = 3
FOLDS = 5
TOLERANCE = 1.0e-7
METRICS = ("ate", "rpe", "endpoint", "objective")
MODEL_NAMES = (
    "pointwise_ridge_core",
    "pairwise_ridge_core",
    "multitask_mlp_extended",
)


COHORTS: tuple[dict[str, Any], ...] = (
    *tuple(
        {
            "name": f"recon_bank_{bank}",
            "domain": "recon",
            "ground_truth": (
                ROOT
                / "outputs/cem_development_v1/recon_target_calibration_v20"
                / f"bank_{bank}/source_labeled.npz"
            ),
        }
        for bank in "abcdef"
    ),
    *tuple(
        {
            "name": f"recon_bank_{bank}",
            "domain": "recon",
            "ground_truth": (
                ROOT
                / "outputs/cem_development_v1/recon_cem_boundary_v60/evaluation_v60"
                / f"bank_{bank}_v60_outcomes.npz"
            ),
        }
        for bank in "gh"
    ),
    {
        "name": "scand_fresh_outer",
        "domain": "scand",
        "ground_truth": (
            ROOT
            / "outputs/scand_domain_adaptation_v1/fresh_outer_v1/outcomes"
            / "fresh_outer_outcomes.npz"
        ),
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("probe", "full"), required=True)
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


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def write_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {key: payload[key].copy() for key in payload.files}


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def reject_forbidden_paths(paths: list[Path]) -> None:
    forbidden = "res" + "erve"
    for path in paths:
        if forbidden in str(path).lower():
            raise RuntimeError(f"forbidden final-cohort path: {path}")


def metric_values(candidates: np.ndarray, ground_truth: np.ndarray) -> np.ndarray:
    value = np.asarray(candidates, dtype=np.float64) * 0.25
    truth = np.asarray(ground_truth, dtype=np.float64)
    error = value - truth[:, None]
    squared = np.sum(error * error, axis=-1)
    ate = np.sqrt(np.mean(squared, axis=-1))
    endpoint = np.sqrt(squared[..., -1])
    relative = np.diff(value, axis=2) - np.diff(truth, axis=1)[:, None]
    rpe = np.sqrt(np.mean(np.sum(relative * relative, axis=-1), axis=-1))
    objective = 0.4 * ate + 0.6 * endpoint
    return np.stack((ate, rpe, endpoint, objective), axis=-1).astype(np.float32)


def exact_action_key(action: np.ndarray) -> bytes:
    return np.ascontiguousarray(action, dtype=np.float32).tobytes()


def feature_block(
    feature: dict[str, np.ndarray],
    candidates: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    action = np.asarray(feature["action_features"], dtype=np.float32)
    raw = np.asarray(feature["raw_ltwm_features"], dtype=np.float32)
    scorer = np.asarray(feature["scorer_v2_aux_features"], dtype=np.float32)
    support = np.asarray(feature["support_features"], dtype=np.float32)
    geometry = np.asarray(candidates, dtype=np.float32).reshape(
        len(candidates), candidates.shape[1], -1
    )
    continuous = np.concatenate((action, raw, scorer, support, geometry), axis=-1)
    scorer_proxy = 0.4 * scorer[..., 6] + 0.6 * scorer[..., 8]
    return continuous, scorer_proxy.astype(np.float32), scorer


def score_extension(score: dict[str, np.ndarray] | None, shape: tuple[int, int]) -> np.ndarray:
    states, candidates = shape
    if score is None:
        return np.zeros((states, candidates, 13), dtype=np.float32)
    energy = np.asarray(score["energy"], dtype=np.float32)
    mean = np.asarray(score["mean"], dtype=np.float32)
    std = np.asarray(score["std"], dtype=np.float32)
    dual = np.asarray(score["dual_probability"], dtype=np.float32)[..., None]
    harm = np.asarray(score["harm_probability"], dtype=np.float32)[..., None]
    rank = np.asarray(score["rank_logit"], dtype=np.float32)[..., None]
    available = np.ones((states, candidates, 1), dtype=np.float32)
    values = np.concatenate((available, energy[..., None], mean, std, dual, harm, rank), axis=-1)
    values[~np.isfinite(values)] = 0.0
    return values.astype(np.float32)


def source_for_formal(selected: dict[str, np.ndarray]) -> np.ndarray:
    result = np.full(len(selected["record_key"]), "incumbent", dtype="U16")
    source_round = selected["archive_source_round"].astype(np.int64)
    result[source_round == -1] = "fixed"
    for iteration in range(ITERATIONS):
        result[source_round == iteration] = f"r{iteration + 1}"
    return result


def load_cohort(spec: dict[str, Any]) -> dict[str, Any]:
    replay = V70_ROOT / "replays" / spec["name"] / "proposal_aware_v70"
    freeze_path = replay / "prelabel_freeze.json"
    selected_path = replay / "selected_actions.npz"
    ground_path = Path(spec["ground_truth"])
    reject_forbidden_paths([replay, freeze_path, selected_path, ground_path, PREREG])
    freeze = read_json(freeze_path)
    if freeze.get("outcomes_used_by_cem") or freeze.get("reserve_accessed"):
        raise RuntimeError(f"v70 replay contract is not label-free: {spec['name']}")
    selected = load_npz(selected_path)
    ground = load_npz(ground_path)
    record = selected["record_key"].astype(str)
    if not np.array_equal(record, ground["record_key"].astype(str)):
        raise RuntimeError(f"ground-truth record mismatch: {spec['name']}")
    base_feature_path = resolve_path(freeze["base_features"])
    base_feature = load_npz(base_feature_path)
    if not np.array_equal(record, base_feature["record_key"].astype(str)):
        raise RuntimeError(f"base feature record mismatch: {spec['name']}")
    base = selected["base_k17_feature_waypoint_units"].astype(np.float32)
    if not np.array_equal(base, base_feature["candidates"].astype(np.float32)):
        raise RuntimeError(f"base candidate mismatch: {spec['name']}")
    pools: list[dict[str, Any]] = []
    base_continuous, base_proxy, _ = feature_block(base_feature, base)
    pools.append({
        "source": "fixed",
        "candidate": base,
        "continuous": base_continuous,
        "scorer_proxy": base_proxy,
        "valid": selected["base_valid"].astype(bool),
        "safe": selected["base_safe"].astype(bool),
        "extension": score_extension(None, base.shape[:2]),
    })
    hashes: dict[str, str] = {
        "prelabel_freeze": sha256_file(freeze_path),
        "selected_actions": sha256_file(selected_path),
        "ground_truth": sha256_file(ground_path),
        "base_features": sha256_file(base_feature_path),
    }
    for iteration in range(ITERATIONS):
        directory = replay / f"iteration_{iteration:02d}"
        population_path = directory / "population.npz"
        feature_path = directory / "cem_features.npz"
        score_path = directory / "frozen_scores.npz"
        population = load_npz(population_path)
        feature = load_npz(feature_path)
        score = load_npz(score_path)
        for payload in (population, feature, score):
            if not np.array_equal(record, payload["record_key"].astype(str)):
                raise RuntimeError(
                    f"iteration {iteration} record mismatch: {spec['name']}"
                )
        candidate = population["candidates_feature_waypoint_units"].astype(np.float32)
        if not np.array_equal(candidate, feature["candidates"].astype(np.float32)):
            raise RuntimeError(
                f"iteration {iteration} candidate mismatch: {spec['name']}"
            )
        continuous, proxy, _ = feature_block(feature, candidate)
        pools.append({
            "source": f"r{iteration + 1}",
            "candidate": candidate,
            "continuous": continuous,
            "scorer_proxy": proxy,
            "valid": score["valid"].astype(bool),
            "safe": score["safe"].astype(bool),
            "extension": score_extension(score, candidate.shape[:2]),
        })
        hashes[f"round_{iteration + 1}_population"] = sha256_file(population_path)
        hashes[f"round_{iteration + 1}_features"] = sha256_file(feature_path)
        hashes[f"round_{iteration + 1}_scores"] = sha256_file(score_path)

    ground_truth = ground["ground_truth_local_m"].astype(np.float32)
    fixed_index = selected["fixed_pool_selected_index"].astype(np.int64)
    row = np.arange(len(record))
    noncem_action = base[row, fixed_index]
    final = selected["final_cem_feature_waypoint_units"].astype(np.float32)
    final_index = selected["final_cem_selected_index"].astype(np.int64)
    formal_action = final[row, final_index]
    formal_source = source_for_formal(selected)
    states = len(record)
    expected_unique = 65
    continuous_dim = pools[0]["continuous"].shape[-1]
    candidates = np.empty((states, expected_unique, 4, 2), dtype=np.float32)
    continuous = np.empty((states, expected_unique, continuous_dim), dtype=np.float32)
    extension = np.empty((states, expected_unique, 13), dtype=np.float32)
    proxy = np.empty((states, expected_unique), dtype=np.float32)
    valid = np.empty((states, expected_unique), dtype=bool)
    safe = np.empty((states, expected_unique), dtype=bool)
    source_id = np.empty((states, expected_unique), dtype=np.int64)
    formal_union_index = np.empty(states, dtype=np.int64)
    noncem_union_index = np.empty(states, dtype=np.int64)
    for state in range(states):
        seen: dict[bytes, int] = {}
        cursor = 0
        for pool_id, pool in enumerate(pools):
            for candidate_index in range(pool["candidate"].shape[1]):
                key = exact_action_key(pool["candidate"][state, candidate_index])
                if key in seen:
                    continue
                if cursor >= expected_unique:
                    raise RuntimeError(f"too many unique candidates: {spec['name']} {state}")
                seen[key] = cursor
                candidates[state, cursor] = pool["candidate"][state, candidate_index]
                continuous[state, cursor] = pool["continuous"][state, candidate_index]
                extension[state, cursor] = pool["extension"][state, candidate_index]
                proxy[state, cursor] = pool["scorer_proxy"][state, candidate_index]
                valid[state, cursor] = pool["valid"][state, candidate_index]
                safe[state, cursor] = pool["safe"][state, candidate_index]
                source_id[state, cursor] = pool_id
                cursor += 1
        if cursor != expected_unique:
            raise RuntimeError(
                f"unique candidate count changed: {spec['name']} {state}: {cursor}"
            )
        formal_key = exact_action_key(formal_action[state])
        noncem_key = exact_action_key(noncem_action[state])
        if formal_key not in seen or noncem_key not in seen:
            raise RuntimeError(f"frozen selection absent from union: {spec['name']} {state}")
        formal_union_index[state] = seen[formal_key]
        noncem_union_index[state] = seen[noncem_key]

    true_metrics = metric_values(candidates, ground_truth)
    formal_metrics = true_metrics[row, formal_union_index]
    delta = true_metrics - formal_metrics[:, None]
    dual_vs_formal = (
        (delta[..., 0] < -TOLERANCE) & (delta[..., 3] < -TOLERANCE) & valid
    )
    harm_vs_formal = (
        ((delta[..., 0] > TOLERANCE) | (delta[..., 3] > TOLERANCE)) & valid
    )
    objective = true_metrics[..., 3]
    target_rank = np.empty_like(objective, dtype=np.float32)
    hard_positive = np.zeros_like(valid)
    hard_negative = np.zeros_like(valid)
    for state in range(states):
        local = valid[state]
        mean = float(objective[state, local].mean())
        scale = max(float(objective[state, local].std()), 1.0e-4)
        target_rank[state] = (objective[state] - mean) / scale
        order = np.argsort(objective[state], kind="stable")
        top = order[local[order]][:4]
        hard_positive[state, top] = dual_vs_formal[state, top]
        hard_positive[state, formal_union_index[state]] = False
        available = extension[state, :, 0] > 0.5
        energy = extension[state, :, 1]
        finite = available & valid[state] & np.isfinite(energy)
        if finite.any():
            cutoff = float(np.median(energy[finite]))
            hard_negative[state] = (
                finite & safe[state] & harm_vs_formal[state] & (energy <= cutoff)
            )

    formal_continuous = continuous[row, formal_union_index]
    relative = continuous - formal_continuous[:, None]
    source_onehot = np.eye(4, dtype=np.float32)[source_id]
    domain_id = 0 if spec["domain"] == "recon" else 1
    domain_onehot = np.zeros((states, expected_unique, 2), dtype=np.float32)
    domain_onehot[..., domain_id] = 1.0
    formal_selected = np.zeros((states, expected_unique, 1), dtype=np.float32)
    formal_selected[row, formal_union_index, 0] = 1.0
    core = np.concatenate(
        (
            relative,
            source_onehot,
            domain_onehot,
            valid[..., None].astype(np.float32),
            safe[..., None].astype(np.float32),
            formal_selected,
        ),
        axis=-1,
    ).astype(np.float32)
    extended = np.concatenate((core, continuous, extension), axis=-1).astype(np.float32)
    family = base_feature["family"].astype(str)
    group = np.asarray(
        [f"{spec['domain']}::{value}" for value in family], dtype="U128"
    )
    return {
        "name": spec["name"],
        "domain": spec["domain"],
        "record_key": record,
        "group": group,
        "candidates": candidates,
        "ground_truth": ground_truth,
        "true_metrics": true_metrics,
        "core": core,
        "extended": extended,
        "valid": valid,
        "safe": safe,
        "source_id": source_id,
        "scorer_proxy": proxy,
        "v70_energy": extension[..., 1],
        "v70_score_available": extension[..., 0] > 0.5,
        "formal_index": formal_union_index,
        "noncem_index": noncem_union_index,
        "target_rank": target_rank,
        "dual_vs_formal": dual_vs_formal.astype(np.float32),
        "harm_vs_formal": harm_vs_formal.astype(np.float32),
        "hard_positive": hard_positive,
        "hard_negative": hard_negative,
        "formal_source": formal_source,
        "input_hashes": hashes,
    }


def concatenate_cohorts(cohorts: list[dict[str, Any]]) -> dict[str, Any]:
    array_keys = (
        "record_key",
        "group",
        "candidates",
        "ground_truth",
        "true_metrics",
        "core",
        "extended",
        "valid",
        "safe",
        "source_id",
        "scorer_proxy",
        "v70_energy",
        "v70_score_available",
        "formal_index",
        "noncem_index",
        "target_rank",
        "dual_vs_formal",
        "harm_vs_formal",
        "hard_positive",
        "hard_negative",
        "formal_source",
    )
    data = {key: np.concatenate([cohort[key] for cohort in cohorts]) for key in array_keys}
    data["domain"] = np.concatenate(
        [np.full(len(cohort["record_key"]), cohort["domain"], dtype="U8") for cohort in cohorts]
    )
    data["cohort"] = np.concatenate(
        [np.full(len(cohort["record_key"]), cohort["name"], dtype="U32") for cohort in cohorts]
    )
    data["input_hashes"] = {cohort["name"]: cohort["input_hashes"] for cohort in cohorts}
    return data


def assign_route_folds(domain: np.ndarray, group: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    fold = np.full(len(group), -1, dtype=np.int64)
    audit: dict[str, Any] = {"folds": FOLDS, "domains": {}}
    mapping: dict[str, int] = {}
    for domain_name in ("recon", "scand"):
        local = domain.astype(str) == domain_name
        unique, counts = np.unique(group[local].astype(str), return_counts=True)
        records = sorted(
            zip(unique.tolist(), counts.tolist()),
            key=lambda item: (-item[1], hashlib.sha256(item[0].encode()).hexdigest()),
        )
        load = [0] * FOLDS
        groups_per_fold = [0] * FOLDS
        for name, count in records:
            target = min(range(FOLDS), key=lambda value: (load[value], value))
            mapping[name] = target
            load[target] += int(count)
            groups_per_fold[target] += 1
        audit["domains"][domain_name] = {
            "states_per_fold": load,
            "groups_per_fold": groups_per_fold,
            "groups": len(records),
        }
    for index, name in enumerate(group.astype(str)):
        fold[index] = mapping[name]
    if (fold < 0).any():
        raise RuntimeError("unassigned route fold")
    leakage = []
    for name in np.unique(group.astype(str)):
        local_fold = np.unique(fold[group.astype(str) == name])
        if len(local_fold) != 1:
            leakage.append(name)
    if leakage:
        raise RuntimeError(f"route leakage across folds: {leakage[:3]}")
    audit["fold_states_pooled"] = [int((fold == value).sum()) for value in range(FOLDS)]
    audit["route_overlap_violations"] = 0
    audit["assignment_sha256"] = hashlib.sha256(
        json.dumps(mapping, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return fold, audit


def fit_normalizer(features: np.ndarray, mask: np.ndarray) -> dict[str, np.ndarray]:
    values = features[mask].astype(np.float64)
    mean = values.mean(axis=0)
    scale = values.std(axis=0)
    active = np.isfinite(mean) & np.isfinite(scale) & (scale >= 1.0e-6)
    return {
        "mean": mean[active].astype(np.float32),
        "scale": scale[active].astype(np.float32),
        "active": np.flatnonzero(active).astype(np.int64),
    }


def normalize(features: np.ndarray, norm: dict[str, np.ndarray]) -> np.ndarray:
    value = features[..., norm["active"]].astype(np.float32)
    value = (value - norm["mean"]) / norm["scale"]
    return np.clip(value, -8.0, 8.0).astype(np.float32)


def candidate_weights(data: dict[str, Any], states: np.ndarray) -> np.ndarray:
    weights = np.ones(data["valid"].shape, dtype=np.float32)
    weights[data["hard_positive"]] = 4.0
    weights[data["hard_negative"]] = 6.0
    row = np.arange(len(weights))
    weights[row, data["formal_index"]] = np.maximum(
        weights[row, data["formal_index"]], 2.0
    )
    result = np.zeros_like(weights)
    result[states] = weights[states]
    return result


def fit_pointwise_ridge(
    features: np.ndarray,
    data: dict[str, Any],
    train_states: np.ndarray,
    ridge_lambda: float = 10.0,
) -> dict[str, Any]:
    state_mask = np.zeros(len(features), dtype=bool)
    state_mask[train_states] = True
    mask = state_mask[:, None] & data["valid"]
    norm = fit_normalizer(features, mask)
    x = normalize(features, norm)[mask].astype(np.float64)
    y = data["target_rank"][mask].astype(np.float64)
    weight = candidate_weights(data, train_states)[mask].astype(np.float64)
    x = np.concatenate((x, np.ones((len(x), 1), dtype=np.float64)), axis=1)
    root_weight = np.sqrt(weight)[:, None]
    xw = x * root_weight
    yw = y * root_weight[:, 0]
    matrix = xw.T @ xw
    matrix.flat[:: len(matrix) + 1] += ridge_lambda
    matrix[-1, -1] -= ridge_lambda
    coefficient = np.linalg.solve(matrix, xw.T @ yw).astype(np.float32)
    return {"normalization": norm, "coefficient": coefficient}


def predict_ridge(model: dict[str, Any], features: np.ndarray) -> np.ndarray:
    x = normalize(features, model["normalization"])
    return (
        np.einsum("nkd,d->nk", x, model["coefficient"][:-1])
        + model["coefficient"][-1]
    ).astype(np.float32)


def pair_indices(data: dict[str, Any], states: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    left: list[tuple[int, int]] = []
    right: list[tuple[int, int]] = []
    weights: list[float] = []
    objective = data["true_metrics"][..., 3]
    for state in states.tolist():
        valid = data["valid"][state]
        order = np.argsort(objective[state], kind="stable")
        order = order[valid[order]]
        top = order[:4]
        bottom = order[-4:]
        formal = int(data["formal_index"][state])
        pairs: set[tuple[int, int, int]] = set()
        for good in top:
            for bad in bottom:
                if objective[state, good] + TOLERANCE < objective[state, bad]:
                    pairs.add((int(good), int(bad), 3))
            if objective[state, good] + TOLERANCE < objective[state, formal]:
                pairs.add((int(good), formal, 4))
        hard = np.flatnonzero(data["hard_negative"][state])[:4]
        for good in top:
            for bad in hard:
                if objective[state, good] + TOLERANCE < objective[state, bad]:
                    pairs.add((int(good), int(bad), 6))
        quantile = order[np.linspace(0, len(order) - 1, 6).round().astype(int)]
        for good, bad in zip(quantile[:-1], quantile[1:]):
            if objective[state, good] + TOLERANCE < objective[state, bad]:
                pairs.add((int(good), int(bad), 1))
        for good, bad, weight in sorted(pairs):
            left.append((state, good))
            right.append((state, bad))
            weights.append(float(weight))
    return (
        np.asarray(left, dtype=np.int64),
        np.asarray(right, dtype=np.int64),
        np.asarray(weights, dtype=np.float32),
    )


def fit_pairwise_ridge(
    features: np.ndarray,
    data: dict[str, Any],
    train_states: np.ndarray,
    ridge_lambda: float = 5.0,
) -> dict[str, Any]:
    state_mask = np.zeros(len(features), dtype=bool)
    state_mask[train_states] = True
    norm = fit_normalizer(features, state_mask[:, None] & data["valid"])
    normalized = normalize(features, norm)
    left, right, weight = pair_indices(data, train_states)
    x = normalized[left[:, 0], left[:, 1]].astype(np.float64) - normalized[
        right[:, 0], right[:, 1]
    ].astype(np.float64)
    y = (
        data["target_rank"][left[:, 0], left[:, 1]]
        - data["target_rank"][right[:, 0], right[:, 1]]
    ).astype(np.float64)
    root_weight = np.sqrt(weight.astype(np.float64))[:, None]
    xw = x * root_weight
    yw = y * root_weight[:, 0]
    matrix = xw.T @ xw
    matrix.flat[:: len(matrix) + 1] += ridge_lambda
    coefficient = np.linalg.solve(matrix, xw.T @ yw).astype(np.float32)
    return {"normalization": norm, "coefficient": coefficient, "pairs": len(left)}


def predict_pairwise_ridge(model: dict[str, Any], features: np.ndarray) -> np.ndarray:
    x = normalize(features, model["normalization"])
    return np.einsum("nkd,d->nk", x, model["coefficient"]).astype(np.float32)


class LinearSafety(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.linear = nn.Linear(dimension, 2)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.linear(value)


def fit_linear_safety(
    features: np.ndarray,
    data: dict[str, Any],
    train_states: np.ndarray,
    *,
    device: str,
    seed: int,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    state_mask = np.zeros(len(features), dtype=bool)
    state_mask[train_states] = True
    mask = state_mask[:, None] & data["valid"]
    norm = fit_normalizer(features, mask)
    x = normalize(features, norm)[mask]
    target = np.stack(
        (data["dual_vs_formal"][mask], data["harm_vs_formal"][mask]), axis=1
    ).astype(np.float32)
    weight = candidate_weights(data, train_states)[mask]
    model = LinearSafety(x.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3.0e-3, weight_decay=1.0e-4)
    positive = target.sum(axis=0)
    negative = len(target) - positive
    pos_weight = torch.from_numpy(
        np.clip(negative / np.maximum(positive, 1.0), 0.5, 6.0).astype(np.float32)
    ).to(device)
    rng = np.random.default_rng(seed)
    for _ in range(24):
        order = rng.permutation(len(x))
        for start in range(0, len(order), 4096):
            index = order[start : start + 4096]
            xb = torch.from_numpy(x[index]).to(device)
            yb = torch.from_numpy(target[index]).to(device)
            wb = torch.from_numpy(weight[index]).to(device)
            logits = model(xb)
            loss = nn.functional.binary_cross_entropy_with_logits(
                logits, yb, pos_weight=pos_weight, reduction="none"
            ).mean(dim=1)
            loss = (loss * wb).sum() / wb.sum().clamp_min(1.0)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return {
        "normalization": norm,
        "weight": model.linear.weight.detach().cpu().numpy().astype(np.float32),
        "bias": model.linear.bias.detach().cpu().numpy().astype(np.float32),
    }


def predict_linear_safety(model: dict[str, Any], features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = normalize(features, model["normalization"])
    logits = np.einsum("nkd,hd->nkh", x, model["weight"]) + model["bias"]
    probability = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
    return probability[..., 0].astype(np.float32), probability[..., 1].astype(np.float32)


class MultiTaskMLP(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(dimension, 128),
            nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(128, 32),
            nn.SiLU(),
        )
        self.head = nn.Linear(32, 3)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.head(self.trunk(value))


def fit_mlp(
    features: np.ndarray,
    data: dict[str, Any],
    train_states: np.ndarray,
    *,
    device: str,
    seed: int,
    epochs: int = 18,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
    state_mask = np.zeros(len(features), dtype=bool)
    state_mask[train_states] = True
    mask = state_mask[:, None] & data["valid"]
    norm = fit_normalizer(features, mask)
    normalized = normalize(features, norm)
    state_index, candidate_index = np.nonzero(mask)
    x = normalized[state_index, candidate_index]
    target_rank = data["target_rank"][state_index, candidate_index]
    target_dual = data["dual_vs_formal"][state_index, candidate_index]
    target_harm = data["harm_vs_formal"][state_index, candidate_index]
    row_weight = candidate_weights(data, train_states)[state_index, candidate_index]
    left, right, pair_weight = pair_indices(data, train_states)
    pair_left = normalized[left[:, 0], left[:, 1]]
    pair_right = normalized[right[:, 0], right[:, 1]]
    model = MultiTaskMLP(x.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=2.0e-4)
    labels = np.stack((target_dual, target_harm), axis=1).astype(np.float32)
    positive = labels.sum(axis=0)
    negative = len(labels) - positive
    pos_weight = torch.from_numpy(
        np.clip(negative / np.maximum(positive, 1.0), 0.5, 6.0).astype(np.float32)
    ).to(device)
    rng = np.random.default_rng(seed)
    started = time.perf_counter()
    epoch_audit = []
    for epoch in range(epochs):
        order = rng.permutation(len(x))
        pair_order = rng.permutation(len(pair_left))
        pair_cursor = 0
        loss_total = 0.0
        batches = 0
        model.train()
        for start in range(0, len(order), 2048):
            index = order[start : start + 2048]
            xb = torch.from_numpy(x[index]).to(device)
            output = model(xb)
            rank_loss = nn.functional.smooth_l1_loss(
                output[:, 0],
                torch.from_numpy(target_rank[index]).to(device),
                reduction="none",
            )
            safe_loss = nn.functional.binary_cross_entropy_with_logits(
                output[:, 1:],
                torch.from_numpy(labels[index]).to(device),
                pos_weight=pos_weight,
                reduction="none",
            ).mean(dim=1)
            weight = torch.from_numpy(row_weight[index]).to(device)
            point_loss = ((rank_loss + 0.8 * safe_loss) * weight).sum() / weight.sum().clamp_min(1.0)
            pair_count = min(len(index), len(pair_order) - pair_cursor)
            if pair_count == 0:
                pair_order = rng.permutation(len(pair_left))
                pair_cursor = 0
                pair_count = min(len(index), len(pair_order))
            pindex = pair_order[pair_cursor : pair_cursor + pair_count]
            pair_cursor += pair_count
            left_score = model(torch.from_numpy(pair_left[pindex]).to(device))[:, 0]
            right_score = model(torch.from_numpy(pair_right[pindex]).to(device))[:, 0]
            pweight = torch.from_numpy(pair_weight[pindex]).to(device)
            pair_loss = (nn.functional.softplus(left_score - right_score) * pweight).sum() / pweight.sum().clamp_min(1.0)
            loss = point_loss + 0.5 * pair_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            loss_total += float(loss.detach().cpu())
            batches += 1
        epoch_audit.append({
            "epoch": epoch + 1,
            "mean_batch_loss": loss_total / max(batches, 1),
            "elapsed_sec": time.perf_counter() - started,
        })
    return {
        "normalization": norm,
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "dimension": x.shape[1],
        "epochs": epochs,
        "pairs": len(left),
        "epoch_audit": epoch_audit,
    }


def predict_mlp(
    model_record: dict[str, Any], features: np.ndarray, *, device: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = normalize(features, model_record["normalization"])
    model = MultiTaskMLP(int(model_record["dimension"])).to(device)
    model.load_state_dict(model_record["state_dict"], strict=True)
    model.eval()
    flat = x.reshape(-1, x.shape[-1])
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(flat), 8192):
            outputs.append(model(torch.from_numpy(flat[start : start + 8192]).to(device)).cpu().numpy())
    output = np.concatenate(outputs).reshape(x.shape[0], x.shape[1], 3)
    probability = 1.0 / (1.0 + np.exp(-np.clip(output[..., 1:], -30.0, 30.0)))
    return (
        output[..., 0].astype(np.float32),
        probability[..., 0].astype(np.float32),
        probability[..., 1].astype(np.float32),
    )


def select_unconstrained(score: np.ndarray, valid: np.ndarray) -> np.ndarray:
    masked = np.where(valid, score, np.inf)
    return np.argmin(masked, axis=1).astype(np.int64)


def select_safe(
    score: np.ndarray,
    p_dual: np.ndarray,
    p_harm: np.ndarray,
    data: dict[str, Any],
) -> np.ndarray:
    row = np.arange(len(score))
    formal = data["formal_index"]
    eligible = (
        data["valid"]
        & (p_dual >= 0.5)
        & (p_harm <= 0.5)
        & (score < score[row, formal, None])
    )
    eligible[row, formal] = False
    masked = np.where(eligible, score, np.inf)
    selected = np.argmin(masked, axis=1).astype(np.int64)
    none = ~eligible.any(axis=1)
    selected[none] = formal[none]
    return selected


def summarize_selection(data: dict[str, Any], selected: np.ndarray) -> dict[str, Any]:
    row = np.arange(len(selected))
    metrics = data["true_metrics"][row, selected]
    incumbent = data["true_metrics"][:, 0]
    formal = data["true_metrics"][row, data["formal_index"]]
    promoted = selected != 0
    replacement = selected != data["formal_index"]
    delta_incumbent = metrics - incumbent
    delta_formal = metrics - formal
    dual = promoted & (delta_incumbent[:, 0] < -TOLERANCE) & (
        delta_incumbent[:, 3] < -TOLERANCE
    )
    harm = promoted & ((delta_incumbent[:, 0] > TOLERANCE) | (
        delta_incumbent[:, 3] > TOLERANCE
    ))
    dual_formal = replacement & (delta_formal[:, 0] < -TOLERANCE) & (
        delta_formal[:, 3] < -TOLERANCE
    )
    harm_formal = replacement & ((delta_formal[:, 0] > TOLERANCE) | (
        delta_formal[:, 3] > TOLERANCE
    ))
    promotions = int(promoted.sum())
    replacements = int(replacement.sum())
    source_names = np.asarray(("fixed", "r1", "r2", "r3"))
    source = source_names[data["source_id"][row, selected]]
    return {
        "states": int(len(selected)),
        "absolute_mean_m": {
            name: float(metrics[:, index].mean()) for index, name in enumerate(METRICS)
        },
        "absolute_p90_m": {
            name: float(np.quantile(metrics[:, index], 0.9)) for index, name in enumerate(METRICS)
        },
        "coverage": float(promoted.mean()),
        "promotions": promotions,
        "dual_successes": int(dual.sum()),
        "dual_precision": float(dual.sum() / promotions) if promotions else None,
        "harm_count": int(harm.sum()),
        "harm_rate": float(harm.sum() / promotions) if promotions else None,
        "replacements_vs_formal_v70": replacements,
        "replacement_fraction_vs_formal_v70": float(replacement.mean()),
        "dual_replacements_vs_formal_v70": int(dual_formal.sum()),
        "dual_precision_vs_formal_v70": (
            float(dual_formal.sum() / replacements) if replacements else None
        ),
        "harm_replacements_vs_formal_v70": int(harm_formal.sum()),
        "harm_rate_vs_formal_v70": (
            float(harm_formal.sum() / replacements) if replacements else None
        ),
        "selected_source_counts": {
            str(name): int((source == name).sum()) for name in np.unique(source)
        },
    }


def subset(data: dict[str, Any], mask: np.ndarray) -> dict[str, Any]:
    result = {}
    for key, value in data.items():
        if isinstance(value, np.ndarray) and len(value) == len(mask):
            result[key] = value[mask]
        else:
            result[key] = value
    return result


def selection_summaries(data: dict[str, Any], selected: np.ndarray) -> dict[str, Any]:
    result = {"pooled": summarize_selection(data, selected)}
    for domain_name, key in (("recon", "recon_a_to_h"), ("scand", "fresh_scand")):
        mask = data["domain"].astype(str) == domain_name
        result[key] = summarize_selection(subset(data, mask), selected[mask])
    return result


def rank_positions(value: np.ndarray) -> np.ndarray:
    order = np.argsort(value, kind="stable")
    position = np.empty(len(value), dtype=np.float64)
    position[order] = np.arange(len(value), dtype=np.float64)
    return position


def ranking_metrics(
    score: np.ndarray,
    objective: np.ndarray,
    valid: np.ndarray,
) -> dict[str, float]:
    spearman = []
    ndcg = []
    recall = []
    regret = []
    for state in range(len(score)):
        index = np.flatnonzero(valid[state] & np.isfinite(score[state]))
        if len(index) < 2:
            continue
        predicted = score[state, index]
        true = objective[state, index]
        pr = rank_positions(predicted)
        tr = rank_positions(true)
        if pr.std() > 0 and tr.std() > 0:
            spearman.append(float(np.corrcoef(pr, tr)[0, 1]))
        true_best = int(index[np.argmin(true)])
        predicted_order = index[np.argsort(predicted, kind="stable")]
        recall.append(float(true_best in predicted_order[:5]))
        m = len(index)
        true_rank_by_candidate = {int(candidate): rank for rank, candidate in enumerate(index[np.argsort(true, kind="stable")])}
        relevance = np.asarray(
            [(m - true_rank_by_candidate[int(candidate)]) / m for candidate in predicted_order[:5]],
            dtype=np.float64,
        )
        ideal = np.sort(relevance)[::-1]
        discount = np.log2(np.arange(2, len(relevance) + 2))
        dcg = np.sum((np.power(2.0, relevance) - 1.0) / discount)
        idcg = np.sum((np.power(2.0, ideal) - 1.0) / discount)
        ndcg.append(float(dcg / idcg) if idcg > 0 else 1.0)
        regret.append(float(objective[state, predicted_order[0]] - objective[state, true_best]))
    return {
        "states": int(len(regret)),
        "spearman_mean": float(np.mean(spearman)),
        "ndcg_at_5_mean": float(np.mean(ndcg)),
        "true_top1_recall_at_5": float(np.mean(recall)),
        "selection_regret_mean_m": float(np.mean(regret)),
    }


def formal_indices_summary(data: dict[str, Any]) -> dict[str, Any]:
    return selection_summaries(data, data["formal_index"])


def noncem_indices_summary(data: dict[str, Any]) -> dict[str, Any]:
    return selection_summaries(data, data["noncem_index"])


def oracle_indices(data: dict[str, Any]) -> np.ndarray:
    return select_unconstrained(data["true_metrics"][..., 3], data["valid"])


def method_delta(challenger: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    return {
        "absolute_mean_delta_m": {
            name: challenger["absolute_mean_m"][name] - baseline["absolute_mean_m"][name]
            for name in METRICS
        },
        "absolute_p90_delta_m": {
            name: challenger["absolute_p90_m"][name] - baseline["absolute_p90_m"][name]
            for name in METRICS
        },
        "coverage_delta": challenger["coverage"] - baseline["coverage"],
    }


def gate_result(
    summaries: dict[str, Any],
    formal: dict[str, Any],
    oracle_gap_m: float,
) -> dict[str, Any]:
    pooled = summaries["pooled"]
    recon = summaries["recon_a_to_h"]
    scand = summaries["fresh_scand"]
    gain = formal["pooled"]["absolute_mean_m"]["objective"] - pooled["absolute_mean_m"]["objective"]
    checks = {
        "recon_mean_objective_non_worse": recon["absolute_mean_m"]["objective"] <= formal["recon_a_to_h"]["absolute_mean_m"]["objective"] + TOLERANCE,
        "recon_mean_ate_non_worse": recon["absolute_mean_m"]["ate"] <= formal["recon_a_to_h"]["absolute_mean_m"]["ate"] + TOLERANCE,
        "scand_mean_objective_non_worse": scand["absolute_mean_m"]["objective"] <= formal["fresh_scand"]["absolute_mean_m"]["objective"] + TOLERANCE,
        "scand_mean_ate_non_worse": scand["absolute_mean_m"]["ate"] <= formal["fresh_scand"]["absolute_mean_m"]["ate"] + TOLERANCE,
        "pooled_dual_at_least_0_85": pooled["dual_precision"] is not None and pooled["dual_precision"] >= 0.85,
        "pooled_harm_at_most_0_15": pooled["harm_rate"] is not None and pooled["harm_rate"] <= 0.15,
        "scand_harm_at_most_0_15": scand["harm_rate"] is not None and scand["harm_rate"] <= 0.15,
    }
    return {
        "eligible_for_model_selection": bool(all(checks.values())),
        "checks": checks,
        "mean_objective_gain_vs_formal_v70_m": float(gain),
        "selection_regret_recovery_fraction": float(gain / oracle_gap_m),
        "meets_20_percent_regret_recovery": bool(gain >= 0.0079),
    }


def load_data() -> tuple[dict[str, Any], dict[str, Any]]:
    prereg = read_json(PREREG)
    if prereg.get("status") != "frozen_before_v72_training":
        raise RuntimeError("v72 preregistration is not frozen")
    if prereg.get("reserve_accessed_by_v72") or not prereg.get("reserve_forbidden"):
        raise RuntimeError("v72 forbidden-cohort contract changed")
    if sha256_file(V70_ROOT / "result.json") != prereg["formal_v70"]["result_sha256"]:
        raise RuntimeError("formal v70 result hash changed")
    cohorts = []
    for spec in COHORTS:
        print(json.dumps({"stage": "load", "cohort": spec["name"]}), flush=True)
        cohorts.append(load_cohort(spec))
    data = concatenate_cohorts(cohorts)
    if len(data["record_key"]) != int(prereg["allowed_cohorts"]["pooled"]):
        raise RuntimeError("pooled state count changed")
    if data["candidates"].shape[1] != int(prereg["candidate_union"]["expected_unique_per_state"]):
        raise RuntimeError("candidate cardinality changed")
    fold, fold_audit = assign_route_folds(data["domain"], data["group"])
    data["fold"] = fold
    audit = {
        "preregistration_sha256": sha256_file(PREREG),
        "formal_v70_result_sha256": sha256_file(V70_ROOT / "result.json"),
        "states": int(len(data["record_key"])),
        "unique_candidates_per_state": int(data["candidates"].shape[1]),
        "core_feature_dimension": int(data["core"].shape[-1]),
        "extended_feature_dimension": int(data["extended"].shape[-1]),
        "route_folds": fold_audit,
        "input_hashes": data["input_hashes"],
        "reserve_accessed": False,
        "formal_v70_mutated": False,
        "dwm_ltwm_rerun": False,
    }
    return data, audit


def run_probe(data: dict[str, Any], audit: dict[str, Any]) -> None:
    result_path = OUTPUT / "probe_result.json"
    prediction_path = OUTPUT / "probe_predictions.npz"
    if result_path.exists() or prediction_path.exists():
        raise FileExistsError("v72 probe output already exists")
    predictions = np.full(data["valid"].shape, np.nan, dtype=np.float32)
    fold_records = []
    started = time.perf_counter()
    for heldout in range(FOLDS):
        train_states = np.flatnonzero(data["fold"] != heldout)
        test_states = np.flatnonzero(data["fold"] == heldout)
        local_started = time.perf_counter()
        model = fit_pointwise_ridge(data["core"], data, train_states)
        predictions[test_states] = predict_ridge(model, data["core"][test_states])
        fold_records.append({
            "heldout_fold": heldout,
            "train_states": int(len(train_states)),
            "test_states": int(len(test_states)),
            "active_features": int(len(model["normalization"]["active"])),
            "elapsed_sec": time.perf_counter() - local_started,
        })
        print(json.dumps(fold_records[-1]), flush=True)
    selected = select_unconstrained(predictions, data["valid"])
    summaries = selection_summaries(data, selected)
    formal = formal_indices_summary(data)
    ranking = ranking_metrics(
        predictions, data["true_metrics"][..., 3], data["valid"]
    )
    gain = formal["pooled"]["absolute_mean_m"]["objective"] - summaries["pooled"]["absolute_mean_m"]["objective"]
    passed = (
        gain >= 0.004
        and summaries["recon_a_to_h"]["absolute_mean_m"]["objective"] <= formal["recon_a_to_h"]["absolute_mean_m"]["objective"] + TOLERANCE
        and summaries["fresh_scand"]["absolute_mean_m"]["objective"] <= formal["fresh_scand"]["absolute_mean_m"]["objective"] + TOLERANCE
    )
    write_npz(
        prediction_path,
        record_key=data["record_key"],
        fold=data["fold"],
        score=predictions,
        selected_index=selected,
    )
    result = {
        "schema": "v72_route_oof_feature_probe_v1",
        "status": "complete",
        "model": "pointwise_ridge_core",
        "route_oof": True,
        "probe_passed": bool(passed),
        "probe_pass_rule": read_json(PREREG)["probe"]["pass_rule"],
        "mean_objective_gain_vs_formal_v70_m": float(gain),
        "selection": summaries,
        "formal_v70": formal,
        "comparison_vs_formal": {
            key: method_delta(summaries[key], formal[key]) for key in summaries
        },
        "ranking": ranking,
        "fold_records": fold_records,
        "elapsed_sec": time.perf_counter() - started,
        "audit": audit,
        "predictions": str(prediction_path),
        "predictions_sha256": sha256_file(prediction_path),
        "reserve_accessed": False,
    }
    write_json(result_path, result)
    write_json(OUTPUT / "probe_audit.json", {
        "result_sha256": sha256_file(result_path),
        "predictions_sha256": sha256_file(prediction_path),
        "preregistration_sha256": sha256_file(PREREG),
        "reserve_accessed": False,
    })
    print(json.dumps({
        "probe_passed": bool(passed),
        "mean_objective_gain_vs_formal_v70_m": float(gain),
        "ranking": ranking,
        "result": str(result_path),
    }, indent=2, sort_keys=True), flush=True)


def oof_full_models(
    data: dict[str, Any], *, device: str
) -> tuple[dict[str, dict[str, np.ndarray]], list[dict[str, Any]]]:
    predictions = {
        name: {
            "score": np.full(data["valid"].shape, np.nan, dtype=np.float32),
            "p_dual": np.full(data["valid"].shape, np.nan, dtype=np.float32),
            "p_harm": np.full(data["valid"].shape, np.nan, dtype=np.float32),
        }
        for name in MODEL_NAMES
    }
    records = []
    probe = load_npz(OUTPUT / "probe_predictions.npz")
    if not (
        np.array_equal(probe["record_key"].astype(str), data["record_key"].astype(str))
        and np.array_equal(probe["fold"].astype(np.int64), data["fold"])
    ):
        raise RuntimeError("probe prediction identity changed")
    predictions["pointwise_ridge_core"]["score"] = probe["score"].astype(np.float32)
    for heldout in range(FOLDS):
        train_states = np.flatnonzero(data["fold"] != heldout)
        test_states = np.flatnonzero(data["fold"] == heldout)
        started = time.perf_counter()
        safety = fit_linear_safety(
            data["core"], data, train_states, device=device, seed=20260722 + heldout
        )
        p_dual, p_harm = predict_linear_safety(safety, data["core"][test_states])
        for name in ("pointwise_ridge_core", "pairwise_ridge_core"):
            predictions[name]["p_dual"][test_states] = p_dual
            predictions[name]["p_harm"][test_states] = p_harm
        pairwise = fit_pairwise_ridge(data["core"], data, train_states)
        predictions["pairwise_ridge_core"]["score"][test_states] = predict_pairwise_ridge(
            pairwise, data["core"][test_states]
        )
        mlp = fit_mlp(
            data["extended"],
            data,
            train_states,
            device=device,
            seed=20260772 + heldout,
        )
        rank, dual, harm = predict_mlp(mlp, data["extended"][test_states], device=device)
        predictions["multitask_mlp_extended"]["score"][test_states] = rank
        predictions["multitask_mlp_extended"]["p_dual"][test_states] = dual
        predictions["multitask_mlp_extended"]["p_harm"][test_states] = harm
        record = {
            "heldout_fold": heldout,
            "train_states": int(len(train_states)),
            "test_states": int(len(test_states)),
            "linear_safety_active_features": int(len(safety["normalization"]["active"])),
            "pairwise_pairs": int(pairwise["pairs"]),
            "pairwise_active_features": int(len(pairwise["normalization"]["active"])),
            "mlp_pairs": int(mlp["pairs"]),
            "mlp_active_features": int(len(mlp["normalization"]["active"])),
            "mlp_final_loss": float(mlp["epoch_audit"][-1]["mean_batch_loss"]),
            "elapsed_sec": time.perf_counter() - started,
        }
        records.append(record)
        print(json.dumps(record), flush=True)
    for name in MODEL_NAMES:
        for key in predictions[name]:
            if not np.isfinite(predictions[name][key]).all():
                raise RuntimeError(f"incomplete OOF prediction: {name}/{key}")
    return predictions, records


def fit_full_selected_model(
    name: str, data: dict[str, Any], *, device: str
) -> tuple[dict[str, Any], Path]:
    states = np.arange(len(data["record_key"]), dtype=np.int64)
    if name in ("pointwise_ridge_core", "pairwise_ridge_core"):
        rank = (
            fit_pointwise_ridge(data["core"], data, states)
            if name == "pointwise_ridge_core"
            else fit_pairwise_ridge(data["core"], data, states)
        )
        safety = fit_linear_safety(
            data["core"], data, states, device=device, seed=20260872
        )
        checkpoint = OUTPUT / "v72_full_checkpoint.npz"
        write_npz(
            checkpoint,
            rank_active=rank["normalization"]["active"],
            rank_mean=rank["normalization"]["mean"],
            rank_scale=rank["normalization"]["scale"],
            rank_coefficient=rank["coefficient"],
            safety_active=safety["normalization"]["active"],
            safety_mean=safety["normalization"]["mean"],
            safety_scale=safety["normalization"]["scale"],
            safety_weight=safety["weight"],
            safety_bias=safety["bias"],
            model_name=np.asarray(name),
        )
        record = {"model_name": name, "rank": rank, "safety": safety}
        return record, checkpoint
    mlp = fit_mlp(
        data["extended"], data, states, device=device, seed=20260872
    )
    checkpoint = OUTPUT / "v72_full_checkpoint.pt"
    torch.save({
        "schema": "v72_proposal_tail_multitask_mlp_v1",
        "model_name": name,
        "normalization": mlp["normalization"],
        "dimension": mlp["dimension"],
        "state_dict": mlp["state_dict"],
        "epochs": mlp["epochs"],
        "selection_policy": read_json(PREREG)["fixed_selection_policy"],
        "development_only": True,
        "independent_confirmation_required": True,
        "reserve_accessed": False,
    }, checkpoint)
    return {"model_name": name, "mlp": mlp}, checkpoint


def latency_cpu_ms(name: str, model: dict[str, Any], data: dict[str, Any]) -> dict[str, float]:
    state = 0
    values = []
    if name in ("pointwise_ridge_core", "pairwise_ridge_core"):
        rank = model["rank"]
        safety = model["safety"]
        feature = data["core"][state : state + 1]
        predict_rank = predict_ridge if name == "pointwise_ridge_core" else predict_pairwise_ridge
        for iteration in range(230):
            started = time.perf_counter_ns()
            score = predict_rank(rank, feature)
            p_dual, p_harm = predict_linear_safety(safety, feature)
            _ = score + p_dual + p_harm
            elapsed = (time.perf_counter_ns() - started) / 1.0e6
            if iteration >= 30:
                values.append(elapsed)
    else:
        mlp = model["mlp"]
        feature = data["extended"][state : state + 1]
        torch.set_num_threads(1)
        for iteration in range(230):
            started = time.perf_counter_ns()
            score, p_dual, p_harm = predict_mlp(mlp, feature, device="cpu")
            _ = score + p_dual + p_harm
            elapsed = (time.perf_counter_ns() - started) / 1.0e6
            if iteration >= 30:
                values.append(elapsed)
    value = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(value.mean()),
        "p50": float(np.quantile(value, 0.5)),
        "p90": float(np.quantile(value, 0.9)),
        "runs": int(len(value)),
    }


def run_full(data: dict[str, Any], audit: dict[str, Any], *, device: str) -> None:
    result_path = OUTPUT / "result.json"
    prediction_path = OUTPUT / "oof_predictions.npz"
    if result_path.exists() or prediction_path.exists():
        raise FileExistsError("v72 full output already exists")
    probe = read_json(OUTPUT / "probe_result.json")
    if not probe.get("probe_passed"):
        raise RuntimeError("pre-registered feature probe did not pass")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    started = time.perf_counter()
    predictions, fold_records = oof_full_models(data, device=device)
    formal = formal_indices_summary(data)
    noncem = noncem_indices_summary(data)
    oracle_selected = oracle_indices(data)
    oracle = selection_summaries(data, oracle_selected)
    oracle_result = read_json(ORACLE_RESULT)
    cited_gap = float(
        oracle_result["decomposition"]["unconstrained_objective"]
        ["positive_gap_m_objective_mean"]["remaining_selection_regret"]
    )
    model_results = {}
    output_arrays: dict[str, np.ndarray] = {
        "record_key": data["record_key"],
        "fold": data["fold"],
        "formal_index": data["formal_index"],
        "noncem_index": data["noncem_index"],
        "oracle_support_valid_index": oracle_selected,
    }
    for name in MODEL_NAMES:
        local = predictions[name]
        selected = select_safe(local["score"], local["p_dual"], local["p_harm"], data)
        summaries = selection_summaries(data, selected)
        ranking = ranking_metrics(local["score"], data["true_metrics"][..., 3], data["valid"])
        gate = gate_result(summaries, formal, cited_gap)
        model_results[name] = {
            "selection": summaries,
            "ranking": ranking,
            "gate": gate,
            "comparison_vs_formal_v70": {
                key: method_delta(summaries[key], formal[key]) for key in summaries
            },
            "comparison_vs_noncem": {
                key: method_delta(summaries[key], noncem[key]) for key in summaries
            },
            "comparison_vs_support_valid_oracle": {
                key: method_delta(summaries[key], oracle[key]) for key in summaries
            },
        }
        for key, value in local.items():
            output_arrays[f"{name}__{key}"] = value
        output_arrays[f"{name}__selected_index"] = selected
    eligible = [
        name for name in MODEL_NAMES
        if model_results[name]["gate"]["eligible_for_model_selection"]
    ]
    if eligible:
        selected_model = min(
            eligible,
            key=lambda name: model_results[name]["selection"]["pooled"]
            ["absolute_mean_m"]["objective"],
        )
    else:
        selected_model = min(
            MODEL_NAMES,
            key=lambda name: model_results[name]["selection"]["pooled"]
            ["absolute_mean_m"]["objective"],
        )
    full_model, checkpoint = fit_full_selected_model(selected_model, data, device=device)
    latency = latency_cpu_ms(selected_model, full_model, data)
    selected_gate = dict(model_results[selected_model]["gate"])
    selected_gate["cpu_latency_target_met"] = latency["p90"] < 10.0
    selected_gate["all_success_checks_met"] = bool(
        selected_gate["eligible_for_model_selection"]
        and selected_gate["meets_20_percent_regret_recovery"]
        and selected_gate["cpu_latency_target_met"]
    )
    write_npz(prediction_path, **output_arrays)
    scorer_proxy_ranking = ranking_metrics(
        data["scorer_proxy"], data["true_metrics"][..., 3], data["valid"]
    )
    v70_tail_mask = data["valid"] & data["v70_score_available"]
    v70_tail_ranking = ranking_metrics(
        data["v70_energy"], data["true_metrics"][..., 3], v70_tail_mask
    )
    result = {
        "schema": "v72_proposal_tail_route_oof_reranker_v1",
        "status": (
            "promising_development_result"
            if selected_gate["all_success_checks_met"]
            else "development_gate_not_met"
        ),
        "selected_model": selected_model,
        "selected_model_gate": selected_gate,
        "model_results": model_results,
        "baselines": {
            "v70_fixed_noncem": noncem,
            "formal_v70_cem": formal,
            "support_valid_existing_union_oracle": oracle,
        },
        "baseline_ranking": {
            "scorer_v2_three_seed_mean_proxy_full_65": scorer_proxy_ranking,
            "formal_v70_energy_proposal_rounds_only": v70_tail_ranking,
            "formal_v70_energy_limitation": "fixed-pool v70 energy was not cached, so this diagnostic covers only r1-r3 score-available candidates",
        },
        "oracle_gap_reference_m": cited_gap,
        "route_oof": True,
        "fold_records": fold_records,
        "cpu_added_latency_ms_per_state": latency,
        "full_checkpoint": str(checkpoint),
        "full_checkpoint_sha256": sha256_file(checkpoint),
        "oof_predictions": str(prediction_path),
        "oof_predictions_sha256": sha256_file(prediction_path),
        "preregistration": str(PREREG),
        "preregistration_sha256": sha256_file(PREREG),
        "audit": audit,
        "elapsed_sec": time.perf_counter() - started,
        "reserve_accessed": False,
        "formal_v70_mutated": False,
        "dwm_ltwm_rerun": False,
        "production_safe_claim_allowed": False,
        "independent_confirmation_required": True,
    }
    write_json(result_path, result)
    write_json(OUTPUT / "audit.json", {
        "result_sha256": sha256_file(result_path),
        "oof_predictions_sha256": sha256_file(prediction_path),
        "full_checkpoint_sha256": sha256_file(checkpoint),
        "preregistration_sha256": sha256_file(PREREG),
        "reserve_accessed": False,
        "formal_v70_mutated": False,
        "dwm_ltwm_rerun": False,
    })
    print(json.dumps({
        "status": result["status"],
        "selected_model": selected_model,
        "selected_model_gate": selected_gate,
        "cpu_added_latency_ms_per_state": latency,
        "result": str(result_path),
    }, indent=2, sort_keys=True), flush=True)


def main() -> None:
    args = parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    data, audit = load_data()
    if args.phase == "probe":
        run_probe(data, audit)
    else:
        run_full(data, audit, device=args.device)


if __name__ == "__main__":
    main()
