#!/usr/bin/env python3
"""Strict nested route-OOF incremental outcome/risk student for spent SCAND."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
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

from planning.proposal_tail_sidecar_v1 import canonical_json, sha256_file  # noqa: E402


OUTPUT = ROOT / "outputs/cem_development_v1/incremental_outcome_student_v1/pairwise_student_v1"
PREREG = OUTPUT / "preregistration.json"
RESULT = OUTPUT / "result.json"
AUDIT = OUTPUT / "audit.json"
PREDICTIONS = OUTPUT / "outer_oof_predictions.npz"
PARAMETERS = OUTPUT / "model_parameters.npz"
TABLE = ROOT / "outputs/cem_development_v1/incremental_outcome_student_v1/spent_runtime_pair_table.npz"
TABLE_AUDIT = ROOT / "outputs/cem_development_v1/incremental_outcome_student_v1/spent_runtime_pair_table.audit.json"
FOLDS = 5
SEED = 20267810
PROFILES = {
    "strict_090_010": {"dual": 0.90, "harm": 0.10, "helpful": 0.90},
    "balanced_085_015": {"dual": 0.85, "harm": 0.15, "helpful": 0.85},
    "harm_tight_080_010": {"dual": 0.80, "harm": 0.10, "helpful": 0.85},
}
METRICS = ("ate", "rpe", "endpoint", "objective")
TOL = 1.0e-7


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("preregister", "run"), required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    require(path.is_file(), f"missing NPZ: {path}")
    with np.load(path, allow_pickle=False) as payload:
        return {key: payload[key].copy() for key in payload.files}


def read_json(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"missing JSON: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    require(not path.exists(), f"refusing to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(canonical_json(payload) + "\n", encoding="utf-8")
    tmp.replace(path)


def script_sha() -> str:
    return sha256_file(Path(__file__).resolve())


def preregister() -> None:
    require(TABLE.is_file() and TABLE_AUDIT.is_file(), "spent runtime table is missing")
    audit = read_json(TABLE_AUDIT)
    require(audit.get("status") == "complete_spent_development_only", "table audit not complete")
    require(audit.get("sealed_validation_included") is False, "validation entered spent table")
    require(int(audit["routes_after_shared_route_deduplication"]) >= 10, "insufficient routes")
    require(not PREREG.exists(), f"preregistration already exists: {PREREG}")
    payload = {
        "schema": "scand_pairwise_incremental_student_preregistration_v1",
        "status": "frozen_before_nested_outer_oof_predictions",
        "evidence_role": "adaptive_spent_development_nested_route_oof_only",
        "target": {
            "reference": "challenger relative to frozen formal-v70 selected/fallback action",
            "heads": ["objective_gain_m", "ate_gain_m", "helpful", "dual", "harm"],
            "objective": "0.4*ATE+0.6*endpoint",
            "strict_dual": "ATE gain>0 and objective gain>0",
            "strict_harm": "ATE gain<0 or objective gain<0",
        },
        "features": {
            "source": "351-dimensional spent_runtime_pair_table runtime-safe contract",
            "includes": ["candidate-minus-formal deltas", "formal-v70 context", "H1-H4 temporal LTWM", "action geometry/curvature", "three-seed disagreement", "old safety as feature only"],
            "forbidden": ["route ID", "record ID", "trajectory ID", "ground truth", "validation metadata"],
        },
        "model": {
            "family": "formal-conditioned pairwise residual multitask student",
            "architecture": [351, 64, 32, 5],
            "activation": "GELU",
            "normalization": "train-fold feature standardization plus LayerNorm",
            "loss": "route-balanced SmoothL1 objective/ATE gain plus BCE helpful/dual/harm",
            "optimizer": "AdamW",
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "epochs": 8,
            "batch_size": 4096,
            "seeds": "deterministic function of outer/inner fold; no model-family selection",
        },
        "nested_route_oof": {
            "outer_folds": 5,
            "assignment": "route groups sorted by descending state count then sha256; greedily assigned to least-loaded fold",
            "inner": "four held route-fold predictions inside each outer-train set",
            "outer_labels_used_for_model_feature_gate_selection": False,
            "shared_route_names_across spent sources": "same group and same fold",
        },
        "candidate_policy": {
            "pool": "frozen physical+support valid candidates; formal-v70 candidate excluded from replacement",
            "old_safety": "feature only, not a hard gate",
            "profiles": PROFILES,
            "candidate_requirements": ["predicted objective gain>0", "predicted ATE gain>0", "profile helpful/dual/harm probability"],
            "choice": "maximum predicted objective gain; stable candidate-index tie break",
            "fallback": "formal-v70 action",
        },
        "inner_profile_selection": {
            "minimum_replacements": 22,
            "dual_wilson_lcb_min": 0.85,
            "harm_wilson_ucb_max": 0.15,
            "all-state_mean_objective_gain_min_m": 0.001,
            "minimum_routes_with_replacement": 4,
            "nonnegative_route_fraction_min": 0.70,
            "choice": "largest all-state objective gain, then more replacements, then lexical profile; none feasible means outer fallback",
        },
        "final_success_gate": {
            "minimum_replacements": 22,
            "dual_wilson_lcb_min": 0.85,
            "harm_wilson_ucb_max": 0.15,
            "all_state_mean_objective_gain_min_m": 0.001,
            "route_cluster_bootstrap_gain_lcb_gt_m": 0.0,
            "minimum_routes_with_replacement": 5,
            "nonnegative_route_fraction_min": 0.70,
            "mean_ate_and_objective_nonworse": True,
            "objective_p90_max_regression_m": 0.001,
            "robot_and_camera_strata_objective_nonworse": True,
            "zero_replacement_cannot_pass": True,
        },
        "validation_rule": "Only if every nested development success gate passes may the already sealed validation policy be frozen and opened once. Otherwise validation remains unopened.",
        "inputs": {
            "table": {"path": str(TABLE.relative_to(ROOT)), "sha256": sha256_file(TABLE)},
            "table_audit": {"path": str(TABLE_AUDIT.relative_to(ROOT)), "sha256": sha256_file(TABLE_AUDIT)},
            "runner": {"path": str(Path(__file__).resolve().relative_to(ROOT)), "sha256": script_sha()},
        },
        "formal_v70_mutated": False,
        "reserve_accessed": False,
        "sealed_validation_opened": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(PREREG, payload)
    print(canonical_json({"status": payload["status"], "preregistration": str(PREREG), "sha256": sha256_file(PREREG)}))


def verify_prereg() -> dict[str, Any]:
    value = read_json(PREREG)
    require(value.get("status") == "frozen_before_nested_outer_oof_predictions", "preregistration not frozen")
    for name, entry in value["inputs"].items():
        path = ROOT / entry["path"]
        require("reserve" not in str(path).lower(), f"forbidden input path: {path}")
        require(sha256_file(path) == entry["sha256"], f"input changed: {name}")
    return value


def assign_folds(route: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    unique, counts = np.unique(route.astype(str), return_counts=True)
    records = sorted(zip(unique.tolist(), counts.tolist()), key=lambda item: (-item[1], hashlib.sha256(item[0].encode()).hexdigest()))
    loads = [0] * FOLDS
    mapping = {}
    for name, count in records:
        fold = min(range(FOLDS), key=lambda f: (loads[f], f))
        mapping[name] = fold
        loads[fold] += count
    fold = np.asarray([mapping[name] for name in route.astype(str)], dtype=np.int8)
    return fold, {"fold_state_loads": loads, "route_to_fold": dict(sorted(mapping.items())), "route_overlap": 0}


class Student(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, 64), nn.LayerNorm(64), nn.GELU(), nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 5))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


def fit_model(data: dict[str, np.ndarray], states: np.ndarray, device: str, seed: int) -> tuple[Student, np.ndarray, np.ndarray, dict[str, float]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
    feature = data["runtime_pair_feature"].astype(np.float32)
    pool = data["physical_support_pool"].astype(bool).copy()
    formal = data["formal_v70_index"].astype(np.int64)
    pool[np.arange(len(pool)), formal] = False
    local = pool[states]
    flat_state, flat_candidate = np.where(local)
    global_state = states[flat_state]
    x = feature[global_state, flat_candidate]
    route = data["route_group"].astype(str)
    unique, route_count = np.unique(route[global_state], return_counts=True)
    count = dict(zip(unique.tolist(), route_count.tolist()))
    weight = np.asarray([1.0 / count[name] for name in route[global_state]], dtype=np.float32)
    weight /= weight.mean()
    mean = np.average(x.astype(np.float64), axis=0, weights=weight).astype(np.float32)
    variance = np.average((x.astype(np.float64) - mean) ** 2, axis=0, weights=weight).astype(np.float32)
    std = np.maximum(np.sqrt(variance), 1.0e-5).astype(np.float32)
    x = ((x - mean) / std).astype(np.float32)
    y = np.stack((
        data["objective_gain_vs_formal_v70_m"][global_state, flat_candidate] * 1000.0,
        data["ate_gain_vs_formal_v70_m"][global_state, flat_candidate] * 1000.0,
        data["helpful_vs_formal_v70"][global_state, flat_candidate].astype(np.float32),
        data["dual_vs_formal_v70"][global_state, flat_candidate].astype(np.float32),
        data["harm_vs_formal_v70"][global_state, flat_candidate].astype(np.float32),
    ), axis=-1).astype(np.float32)
    positives = np.maximum(y[:, 2:].sum(axis=0), 1.0)
    negatives = np.maximum(len(y) - positives, 1.0)
    pos_weight = torch.as_tensor(np.clip(negatives / positives, 0.25, 8.0), device=device, dtype=torch.float32)
    model = Student(x.shape[-1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3, weight_decay=1.0e-4)
    rng = np.random.default_rng(seed)
    batch = 4096
    model.train()
    last_loss = 0.0
    xt = torch.from_numpy(x)
    yt = torch.from_numpy(y)
    wt = torch.from_numpy(weight)
    for _ in range(8):
        order = rng.permutation(len(x))
        total_loss = 0.0
        total_weight = 0.0
        for start in range(0, len(order), batch):
            index = torch.from_numpy(order[start : start + batch])
            xb = xt[index].to(device)
            yb = yt[index].to(device)
            wb = wt[index].to(device)
            pred = model(xb)
            reg = F.smooth_l1_loss(pred[:, :2], yb[:, :2], beta=5.0, reduction="none").mean(dim=1)
            cls = F.binary_cross_entropy_with_logits(pred[:, 2:], yb[:, 2:], pos_weight=pos_weight, reduction="none").mean(dim=1)
            loss = ((reg + cls) * wb).sum() / wb.sum()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * float(wb.sum().cpu())
            total_weight += float(wb.sum().cpu())
        last_loss = total_loss / max(total_weight, 1.0)
    return model, mean, std, {"rows": len(x), "routes": len(unique), "last_epoch_weighted_loss": last_loss}


def predict(model: Student, mean: np.ndarray, std: np.ndarray, feature: np.ndarray, device: str) -> np.ndarray:
    shape = feature.shape[:2]
    x = ((feature.reshape(-1, feature.shape[-1]).astype(np.float32) - mean) / std).astype(np.float32)
    output = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(x), 8192):
            raw = model(torch.from_numpy(x[start : start + 8192]).to(device)).cpu().numpy()
            raw[:, :2] /= 1000.0
            raw[:, 2:] = 1.0 / (1.0 + np.exp(-np.clip(raw[:, 2:], -30.0, 30.0)))
            output.append(raw.astype(np.float32))
    return np.concatenate(output).reshape(*shape, 5)


def wilson(success: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total == 0:
        return 0.0, 1.0
    p = success / total
    den = 1.0 + z * z / total
    center = (p + z * z / (2 * total)) / den
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / den
    return center - radius, center + radius


def select(pred: np.ndarray, data: dict[str, np.ndarray], states: np.ndarray, profile: dict[str, float]) -> np.ndarray:
    pool = data["physical_support_pool"][states].astype(bool).copy()
    formal = data["formal_v70_index"][states].astype(np.int64)
    pool[np.arange(len(states)), formal] = False
    eligible = pool & (pred[..., 0] > 0) & (pred[..., 1] > 0) & (pred[..., 2] >= profile["helpful"]) & (pred[..., 3] >= profile["dual"]) & (pred[..., 4] <= profile["harm"])
    selected = formal.copy()
    for local in range(len(states)):
        index = np.flatnonzero(eligible[local])
        if len(index):
            selected[local] = int(index[np.argmax(pred[local, index, 0])])
    return selected


def selection_stats(data: dict[str, np.ndarray], states: np.ndarray, selected: np.ndarray) -> dict[str, Any]:
    formal = data["formal_v70_index"][states].astype(np.int64)
    replacement = selected != formal
    n = int(replacement.sum())
    row = states
    cand = selected
    dual = data["dual_vs_formal_v70"][row, cand] & replacement
    harm = data["harm_vs_formal_v70"][row, cand] & replacement
    dcount = int(dual.sum())
    hcount = int(harm.sum())
    dual_lcb, _ = wilson(dcount, n)
    _, harm_ucb = wilson(hcount, n)
    objective_gain = data["objective_gain_vs_formal_v70_m"][row, cand] * replacement
    route = data["route_group"][states].astype(str)
    route_gain = {}
    for name in np.unique(route[replacement]):
        local = route == name
        route_gain[name] = float(objective_gain[local].mean())
    nonnegative = float(np.mean(np.asarray(list(route_gain.values())) >= 0.0)) if route_gain else 0.0
    return {
        "replacements": n,
        "coverage": n / len(states),
        "dual_count": dcount,
        "harm_count": hcount,
        "dual_empirical": dcount / n if n else 0.0,
        "harm_empirical": hcount / n if n else 0.0,
        "dual_wilson_lcb": dual_lcb,
        "harm_wilson_ucb": harm_ucb,
        "all_state_mean_objective_gain_m": float(objective_gain.mean()),
        "routes_with_replacement": len(route_gain),
        "nonnegative_route_fraction": nonnegative,
        "route_objective_gain_m": dict(sorted(route_gain.items())),
    }


def inner_feasible(stats: dict[str, Any]) -> bool:
    return bool(stats["replacements"] >= 22 and stats["dual_wilson_lcb"] >= 0.85 and stats["harm_wilson_ucb"] <= 0.15 and stats["all_state_mean_objective_gain_m"] >= 0.001 and stats["routes_with_replacement"] >= 4 and stats["nonnegative_route_fraction"] >= 0.70)


def metric_summary(value: np.ndarray) -> dict[str, dict[str, float]]:
    return {
        "mean_m": {name: float(value[:, i].mean()) for i, name in enumerate(METRICS)},
        "p90_m": {name: float(np.quantile(value[:, i], 0.9)) for i, name in enumerate(METRICS)},
    }


def cluster_bootstrap_gain(gain: np.ndarray, route: np.ndarray, draws: int = 5000) -> dict[str, float]:
    names = np.unique(route.astype(str))
    rng = np.random.default_rng(20267899)
    values = np.empty(draws, dtype=np.float64)
    indices = {name: np.flatnonzero(route.astype(str) == name) for name in names}
    for draw in range(draws):
        sample = rng.choice(names, size=len(names), replace=True)
        joined = np.concatenate([indices[name] for name in sample])
        values[draw] = gain[joined].mean()
    return {"mean_m": float(gain.mean()), "lcb95_m": float(np.quantile(values, 0.025)), "ucb95_m": float(np.quantile(values, 0.975)), "draws": draws, "clusters": len(names)}


def run(device: str) -> None:
    for path in (RESULT, AUDIT, PREDICTIONS, PARAMETERS):
        require(not path.exists(), f"refusing to overwrite: {path}")
    prereg = verify_prereg()
    require(device != "cuda" or torch.cuda.is_available(), "CUDA requested but unavailable")
    data = load_npz(TABLE)
    route = data["route_group"].astype(str)
    fold, fold_audit = assign_folds(route)
    feature = data["runtime_pair_feature"].astype(np.float32)
    states_all = np.arange(len(route))
    predictions = np.full((*feature.shape[:2], 5), np.nan, dtype=np.float32)
    selected_oof = data["formal_v70_index"].astype(np.int64).copy()
    chosen_profile = np.full(len(route), "fallback", dtype="U32")
    inner_audit = {}
    parameters = {}
    started = time.perf_counter()
    for outer in range(FOLDS):
        outer_test = states_all[fold == outer]
        outer_train = states_all[fold != outer]
        inner_pred = np.full((*feature.shape[:2], 5), np.nan, dtype=np.float32)
        fit_records = []
        for inner in range(FOLDS):
            if inner == outer:
                continue
            inner_test = states_all[fold == inner]
            inner_train = states_all[(fold != outer) & (fold != inner)]
            require(not (set(route[inner_train]) & set(route[inner_test])), "inner route leakage")
            model, mean, std, fit = fit_model(data, inner_train, device, SEED + outer * 10 + inner)
            inner_pred[inner_test] = predict(model, mean, std, feature[inner_test], device)
            fit_records.append({"inner": inner, "train_states": len(inner_train), "test_states": len(inner_test), "route_overlap": 0, **fit})
            print(f"pairwise student outer={outer} inner={inner} complete", flush=True)
        profile_records = {}
        feasible = []
        for name, profile in PROFILES.items():
            selected = select(inner_pred[outer_train], data, outer_train, profile)
            stats = selection_stats(data, outer_train, selected)
            stats["feasible"] = inner_feasible(stats)
            profile_records[name] = stats
            if stats["feasible"]:
                feasible.append(name)
        selected_name = None
        if feasible:
            selected_name = sorted(feasible, key=lambda name: (-profile_records[name]["all_state_mean_objective_gain_m"], -profile_records[name]["replacements"], name))[0]
        model, mean, std, final_fit = fit_model(data, outer_train, device, SEED + outer * 10 + 9)
        outer_pred = predict(model, mean, std, feature[outer_test], device)
        predictions[outer_test] = outer_pred
        if selected_name is not None:
            selected_oof[outer_test] = select(outer_pred, data, outer_test, PROFILES[selected_name])
            chosen_profile[outer_test] = selected_name
        for name, tensor in model.state_dict().items():
            parameters[f"outer_{outer}__{name}"] = tensor.detach().cpu().numpy()
        parameters[f"outer_{outer}__mean"] = mean
        parameters[f"outer_{outer}__std"] = std
        inner_audit[str(outer)] = {"outer_train_states": len(outer_train), "outer_test_states": len(outer_test), "outer_route_overlap": 0, "inner_fits": fit_records, "profiles": profile_records, "selected_profile": selected_name, "final_fit": final_fit}
        print(f"pairwise student outer={outer} profile={selected_name or 'fallback'} complete", flush=True)
    require(np.isfinite(predictions).all(), "outer predictions incomplete")
    formal = data["formal_v70_index"].astype(np.int64)
    row = np.arange(len(formal))
    formal_metrics = data["true_metrics_m"][row, formal]
    selected_metrics = data["true_metrics_m"][row, selected_oof]
    stats = selection_stats(data, states_all, selected_oof)
    objective_gain = formal_metrics[:, 3] - selected_metrics[:, 3]
    bootstrap = cluster_bootstrap_gain(objective_gain, route)
    route_replacement = selected_oof != formal
    strata = {}
    for field in ("robot", "camera_contract", "cohort"):
        strata[field] = {}
        for name in np.unique(data[field].astype(str)):
            local = data[field].astype(str) == name
            local_stats = selection_stats(data, np.flatnonzero(local), selected_oof[local])
            local_stats["formal"] = metric_summary(formal_metrics[local])
            local_stats["student"] = metric_summary(selected_metrics[local])
            strata[field][name] = local_stats
    formal_summary = metric_summary(formal_metrics)
    student_summary = metric_summary(selected_metrics)
    mean_ate_nonworse = student_summary["mean_m"]["ate"] <= formal_summary["mean_m"]["ate"] + 1.0e-12
    mean_objective_nonworse = student_summary["mean_m"]["objective"] <= formal_summary["mean_m"]["objective"] + 1.0e-12
    p90_ok = student_summary["p90_m"]["objective"] <= formal_summary["p90_m"]["objective"] + 0.001
    stratum_ok = all(record["student"]["mean_m"]["objective"] <= record["formal"]["mean_m"]["objective"] + 1.0e-12 for field in ("robot", "camera_contract") for record in strata[field].values())
    checks = {
        "minimum_replacements": stats["replacements"] >= 22,
        "dual_wilson_lcb": stats["dual_wilson_lcb"] >= 0.85,
        "harm_wilson_ucb": stats["harm_wilson_ucb"] <= 0.15,
        "objective_gain_at_least_1mm": stats["all_state_mean_objective_gain_m"] >= 0.001,
        "route_cluster_bootstrap_lcb_positive": bootstrap["lcb95_m"] > 0.0,
        "minimum_routes_with_replacement": stats["routes_with_replacement"] >= 5,
        "nonnegative_route_fraction": stats["nonnegative_route_fraction"] >= 0.70,
        "mean_ate_nonworse": mean_ate_nonworse,
        "mean_objective_nonworse": mean_objective_nonworse,
        "objective_p90_within_1mm": p90_ok,
        "robot_and_camera_strata_objective_nonworse": stratum_ok,
    }
    passed = all(checks.values())
    runtime = time.perf_counter() - started
    result = {
        "schema": "scand_pairwise_incremental_student_nested_oof_result_v1",
        "status": "completed_spent_development_nested_route_oof",
        "verdict": "development_pass_policy_may_be_frozen_before_one_time_validation" if passed else "development_fail_validation_remains_sealed",
        "scope": "adaptive spent development only; not independent evidence",
        "states": len(route),
        "routes": len(np.unique(route)),
        "trajectories": len(np.unique(data["trajectory_id"].astype(str))),
        "candidate_rows": int(np.prod(feature.shape[:2])),
        "physical_support_rows": int(data["physical_support_pool"].sum()),
        "nested_selection": stats,
        "formal_v70": formal_summary,
        "student": student_summary,
        "absolute_delta_student_minus_v70": {
            "mean_m": {name: student_summary["mean_m"][name] - formal_summary["mean_m"][name] for name in METRICS},
            "p90_m": {name: student_summary["p90_m"][name] - formal_summary["p90_m"][name] for name in METRICS},
        },
        "route_cluster_bootstrap_objective_gain": bootstrap,
        "strata": strata,
        "inner_outer_audit": inner_audit,
        "success_checks": checks,
        "success_gate_passed": passed,
        "sealed_validation_opened": False,
        "next_action": "freeze exact policy and open sealed validation once" if passed else "stop this model family; keep validation sealed",
        "runtime_seconds": runtime,
        "device": device,
        "formal_v70_mutated": False,
        "reserve_accessed": False,
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with PREDICTIONS.open("wb") as handle:
        np.savez_compressed(handle, record_key=data["record_key"], route_group=route, outer_fold=fold, predictions=predictions, selected_index=selected_oof, formal_v70_index=formal, chosen_profile=chosen_profile)
    with PARAMETERS.open("wb") as handle:
        np.savez_compressed(handle, **parameters)
    write_json(RESULT, result)
    audit = {
        "schema": "scand_pairwise_incremental_student_audit_v1",
        "status": "complete",
        "preregistration_sha256": sha256_file(PREREG),
        "table_sha256": sha256_file(TABLE),
        "runner_sha256": script_sha(),
        "result_sha256": sha256_file(RESULT),
        "predictions_sha256": sha256_file(PREDICTIONS),
        "parameters_sha256": sha256_file(PARAMETERS),
        "fold_audit": fold_audit,
        "outer_labels_used_for_selection": False,
        "route_overlap": 0,
        "sealed_validation_opened": False,
        "reserve_accessed": False,
        "formal_v70_mutated": False,
    }
    write_json(AUDIT, audit)
    print(canonical_json({"status": result["status"], "verdict": result["verdict"], "success": passed, "replacements": stats["replacements"], "dual_lcb": stats["dual_wilson_lcb"], "harm_ucb": stats["harm_wilson_ucb"], "objective_gain_m": stats["all_state_mean_objective_gain_m"], "runtime_seconds": runtime, "result": str(RESULT), "result_sha256": sha256_file(RESULT)}))


def main() -> None:
    args = parse_args()
    if args.phase == "preregister":
        preregister()
    else:
        run(args.device)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(canonical_json({"status": "fail_closed", "error_type": type(error).__name__, "error": str(error), "sealed_validation_opened": False, "reserve_accessed": False}), file=sys.stderr)
        raise SystemExit(2) from error
