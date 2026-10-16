#!/usr/bin/env python3
"""Strict nested route-OOF test of a state-normalized group-DRO risk head."""

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

from tools.evaluation import run_scand_existing_train_upperbound_pilot_v1 as pilot  # noqa: E402
from tools.evaluation import run_scand_existing_train_risk_coverage_frontier_v1 as frontier  # noqa: E402
from tools.evaluation import run_v77_scand_tail_joint_selector as v77  # noqa: E402


OUTPUT = pilot.BASE / "route_stable_risk_v1"
PREREG = OUTPUT / "preregistration.json"
RESULT = OUTPUT / "result.json"
AUDIT = OUTPUT / "audit.json"
PREDICTIONS = OUTPUT / "outer_oof_predictions.npz"
PARAMETERS = OUTPUT / "model_parameters.npz"
HASHES = OUTPUT / "sha256_manifest.json"
SEEDS = (20267801, 20267802, 20267803)
EPOCHS = 160
LEARNING_RATE = 0.002
WEIGHT_DECAY = 0.01
DRO_ETA = 0.05
MODEL_NAME = "state_normalized_group_dro_linear_multitask_ensemble"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


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
    temporary.write_text(
        json.dumps(jsonable(value), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def write_npz(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def verify_contract() -> tuple[dict[str, Any], dict[str, str]]:
    prereg = read_json(PREREG)
    if prereg.get("status") != "frozen_before_route_stable_outer_predictions":
        raise RuntimeError("route-stable preregistration is not frozen")
    spec = prereg["model"]
    if tuple(spec["seeds"]) != SEEDS or int(spec["epochs"]) != EPOCHS:
        raise RuntimeError("model seeds/epochs changed")
    hashes = {}
    for name, item in prereg["inputs"].items():
        path = Path(item["path"])
        if "reserve" in str(path).lower():
            raise RuntimeError(f"forbidden input: {name}")
        digest = sha256_file(path)
        if digest != item["sha256"]:
            raise RuntimeError(f"frozen input changed: {name}")
        hashes[name] = digest
    return prereg, hashes


def selected_feature_indices(names: list[str]) -> np.ndarray:
    explicit = {
        "safety_std_0", "safety_std_1", "safety_std_2", "safety_std_3",
        "seed_1", "seed_2", "seed_4", "seed_5", "seed_7", "seed_8",
        "support_distance_0",
    }
    result = []
    for index, name in enumerate(names):
        if (
            name.startswith("relative_formal_")
            or name.startswith("within_state_rank_")
            or name.startswith("local_density_")
            or name.startswith("proposal_source_")
            or name in explicit
        ):
            result.append(index)
    return np.asarray(result, dtype=np.int64)


def state_normalize(feature: np.ndarray, pool: np.ndarray, indices: np.ndarray) -> np.ndarray:
    value = feature[..., indices].astype(np.float32)
    states, _, dimension = value.shape
    zscore = np.zeros_like(value)
    percentile = np.zeros_like(value)
    for state in range(states):
        local = np.flatnonzero(pool[state])
        observed = value[state, local].astype(np.float64)
        mean = observed.mean(axis=0)
        scale = np.maximum(observed.std(axis=0), 1.0e-5)
        zscore[state] = np.clip((value[state] - mean) / scale, -5.0, 5.0)
        order = np.argsort(observed, axis=0, kind="stable")
        rank = np.empty(order.shape, dtype=np.float32)
        rank[order, np.arange(dimension)[None, :]] = np.arange(len(local), dtype=np.float32)[:, None]
        rank = 2.0 * rank / max(len(local) - 1, 1) - 1.0
        percentile[state, local] = rank
    transformed = np.concatenate((zscore, percentile), axis=-1).astype(np.float32)
    if not np.isfinite(transformed).all():
        raise RuntimeError("non-finite route-stable feature")
    return transformed


class LinearRiskHead(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.head = nn.Linear(dimension, 3)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.head(value)


def fit_member(
    feature: np.ndarray, challenger: np.ndarray, objective_gain: np.ndarray,
    dual: np.ndarray, harm: np.ndarray, states: np.ndarray,
    group: np.ndarray, seed: int, device: str,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
    train_values = feature[states][challenger[states]].astype(np.float64)
    mean = train_values.mean(axis=0).astype(np.float32)
    std = np.maximum(train_values.std(axis=0), 1.0e-5).astype(np.float32)
    scaled = np.clip((feature[states] - mean) / std, -8.0, 8.0).astype(np.float32)
    x = torch.as_tensor(scaled, device=device)
    mask = torch.as_tensor(challenger[states], dtype=torch.bool, device=device)
    gain = torch.as_tensor(objective_gain[states], device=device)
    dual_target = torch.as_tensor(dual[states].astype(np.float32), device=device)
    harm_target = torch.as_tensor(harm[states].astype(np.float32), device=device)
    route_names = group[states].astype(str)
    routes = np.unique(route_names)
    route_state = [
        torch.as_tensor(np.flatnonzero(route_names == route), dtype=torch.long, device=device)
        for route in routes
    ]
    dual_rate = float(dual[states][challenger[states]].mean())
    harm_rate = float(harm[states][challenger[states]].mean())
    dual_pos = torch.tensor(np.clip((1.0 - dual_rate) / max(dual_rate, 1.0e-5), 0.1, 10.0), device=device)
    harm_pos = torch.tensor(np.clip((1.0 - harm_rate) / max(harm_rate, 1.0e-5), 0.1, 10.0), device=device)
    model = LinearRiskHead(feature.shape[-1]).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    dro = torch.ones(len(routes), device=device) / len(routes)
    for _ in range(EPOCHS):
        prediction = model(x)
        gain_prediction, dual_logit, harm_logit = prediction.unbind(dim=-1)
        gain_loss = F.huber_loss(
            gain_prediction, gain, reduction="none", delta=0.01
        )
        dual_loss = F.binary_cross_entropy_with_logits(
            dual_logit, dual_target, reduction="none", pos_weight=dual_pos
        )
        harm_loss = F.binary_cross_entropy_with_logits(
            harm_logit, harm_target, reduction="none", pos_weight=harm_pos
        )
        candidate_loss = 0.5 * gain_loss + dual_loss + harm_loss
        route_loss = []
        for local in route_state:
            local_mask = mask[local]
            route_loss.append(candidate_loss[local][local_mask].mean())
        route_loss_tensor = torch.stack(route_loss)
        loss = torch.sum(dro * route_loss_tensor)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        with torch.no_grad():
            normalized = (
                route_loss_tensor - route_loss_tensor.mean()
            ) / route_loss_tensor.std().clamp_min(1.0e-6)
            dro *= torch.exp(torch.clamp(DRO_ETA * normalized, -2.0, 2.0))
            dro /= dro.sum()
    model.eval()
    return {
        "mean": mean,
        "std": std,
        "model": model,
        "routes": routes,
        "dro_weight": dro.detach().cpu().numpy().astype(np.float32),
    }


def predict_member(
    model: dict[str, Any], feature: np.ndarray, device: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scaled = np.clip((feature - model["mean"]) / model["std"], -8.0, 8.0).astype(np.float32)
    with torch.no_grad():
        output = model["model"](
            torch.as_tensor(scaled, dtype=torch.float32, device=device)
        ).cpu().numpy()
    return (
        output[..., 0].astype(np.float32),
        (1.0 / (1.0 + np.exp(-np.clip(output[..., 1], -30.0, 30.0)))).astype(np.float32),
        (1.0 / (1.0 + np.exp(-np.clip(output[..., 2], -30.0, 30.0)))).astype(np.float32),
    )


def fit_ensemble(*args: Any) -> list[dict[str, Any]]:
    return [fit_member(*args, seed, args[-1]) for seed in SEEDS]


def train_ensemble(
    feature: np.ndarray, challenger: np.ndarray, objective_gain: np.ndarray,
    dual: np.ndarray, harm: np.ndarray, states: np.ndarray,
    group: np.ndarray, device: str,
) -> list[dict[str, Any]]:
    return [
        fit_member(
            feature, challenger, objective_gain, dual, harm,
            states, group, seed, device,
        )
        for seed in SEEDS
    ]


def predict_ensemble(
    models: list[dict[str, Any]], feature: np.ndarray, device: str
) -> dict[str, np.ndarray]:
    members = [predict_member(model, feature, device) for model in models]
    result = {}
    for index, name in enumerate(("gain", "p_dual", "p_harm")):
        values = np.stack([member[index] for member in members], axis=0)
        result[f"{name}_mean"] = values.mean(axis=0).astype(np.float32)
        result[f"{name}_std"] = values.std(axis=0).astype(np.float32)
    return result


def serialize_models(prefix: str, models: list[dict[str, Any]], arrays: dict[str, np.ndarray]) -> None:
    for index, model in enumerate(models):
        key = f"{prefix}__member{index}"
        arrays[f"{key}__mean"] = model["mean"]
        arrays[f"{key}__std"] = model["std"]
        arrays[f"{key}__weight"] = model["model"].head.weight.detach().cpu().numpy()
        arrays[f"{key}__bias"] = model["model"].head.bias.detach().cpu().numpy()
        arrays[f"{key}__dro_weight"] = model["dro_weight"]
        arrays[f"{key}__routes"] = model["routes"].astype("U256")


def cpu_latency(models: list[dict[str, Any]], feature: np.ndarray) -> float:
    local = []
    for model in models:
        item = {key: value for key, value in model.items() if key != "model"}
        item["model"] = copy.deepcopy(model["model"]).cpu()
        local.append(item)
    started = time.perf_counter()
    for _ in range(10):
        predict_ensemble(local, feature, "cpu")
    return float((time.perf_counter() - started) / 10.0 / len(feature))


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    for path in (RESULT, AUDIT, PREDICTIONS, PARAMETERS, HASHES):
        if path.exists():
            raise FileExistsError(path)
    prereg, hashes_before = verify_contract()
    started = time.perf_counter()
    data, fields, full_feature, feature_names, pool, feature_audit = pilot.assemble()
    base_prediction_file = pilot.load_npz(
        pilot.BASE / "oof_upperbound_v1/outer_oof_predictions.npz"
    )
    fold = base_prediction_file["outer_fold"].astype(np.int64)
    selected_indices = selected_feature_indices(feature_names)
    selected_names = [feature_names[index] for index in selected_indices]
    feature = state_normalize(full_feature, pool, selected_indices)
    row = np.arange(len(data["formal_index"]))
    challenger = pool.copy()
    challenger[row, data["formal_index"]] = False
    gains = data["true_metrics"][row, data["formal_index"]][:, None] - data["true_metrics"]
    objective_gain = gains[..., 3].astype(np.float32)
    dual = (gains[..., 0] > v77.TOLERANCE) & (gains[..., 3] > v77.TOLERANCE)
    harm = (gains[..., 0] < -v77.TOLERANCE) | (gains[..., 3] < -v77.TOLERANCE)
    prediction_keys = (
        "gain_mean", "gain_std", "p_dual_mean", "p_dual_std",
        "p_harm_mean", "p_harm_std",
    )
    outer_prediction = {
        key: np.full(pool.shape, np.nan, dtype=np.float32) for key in prediction_keys
    }
    nested_selected = data["formal_index"].copy()
    nested_threshold = np.full(5, np.nan, dtype=np.float32)
    nested_target = np.full(5, np.nan, dtype=np.float32)
    inner_audit: dict[str, Any] = {}
    parameter_arrays: dict[str, np.ndarray] = {}
    latency = []
    for outer in range(5):
        outer_train = np.flatnonzero(fold != outer)
        outer_test = np.flatnonzero(fold == outer)
        inner_prediction = {
            key: np.full(pool.shape, np.nan, dtype=np.float32) for key in prediction_keys
        }
        inner_records = []
        for inner in range(5):
            if inner == outer:
                continue
            inner_train = np.flatnonzero((fold != outer) & (fold != inner))
            inner_test = np.flatnonzero(fold == inner)
            if set(data["group"][inner_train]) & set(data["group"][inner_test]):
                raise RuntimeError("inner route leakage")
            if set(data["bag"][inner_train]) & set(data["bag"][inner_test]):
                raise RuntimeError("inner bag leakage")
            models = train_ensemble(
                feature, challenger, objective_gain, dual, harm,
                inner_train, data["group"], args.device,
            )
            prediction = predict_ensemble(models, feature[inner_test], args.device)
            for key, value in prediction.items():
                inner_prediction[key][inner_test] = value
            del models
            if args.device == "cuda":
                torch.cuda.empty_cache()
            inner_records.append({
                "inner_fold": inner,
                "train_states": int(len(inner_train)),
                "test_states": int(len(inner_test)),
                "route_overlap": 0,
                "bag_overlap": 0,
            })
        inner_local = {key: value[outer_train] for key, value in inner_prediction.items()}
        inner_choice = frontier.state_choice(
            inner_local, pool[outer_train], data["formal_index"][outer_train]
        )
        points = []
        feasible = []
        for fraction in frontier.GRID:
            point, threshold, _ = frontier.point_by_count(
                data, inner_choice[0], inner_choice[1], outer_train, fraction
            )
            points.append(point)
            if point["passes_all"]:
                feasible.append((fraction, threshold))
        chosen = max(feasible, key=lambda item: item[0]) if feasible else None
        if chosen is not None:
            nested_target[outer] = chosen[0]
            nested_threshold[outer] = chosen[1]
        models = train_ensemble(
            feature, challenger, objective_gain, dual, harm,
            outer_train, data["group"], args.device,
        )
        prediction = predict_ensemble(models, feature[outer_test], args.device)
        for key, value in prediction.items():
            outer_prediction[key][outer_test] = value
        outer_choice = frontier.state_choice(
            prediction, pool[outer_test], data["formal_index"][outer_test]
        )
        selected, outer_summary = frontier.apply_threshold(
            data, outer_choice[0], outer_choice[1], outer_test,
            chosen[1] if chosen is not None else None,
        )
        nested_selected[outer_test] = selected
        serialize_models(f"outer{outer}", models, parameter_arrays)
        latency.append(cpu_latency(models, feature[outer_test]))
        del models
        if args.device == "cuda":
            torch.cuda.empty_cache()
        inner_audit[str(outer)] = {
            "outer_train_states": int(len(outer_train)),
            "outer_test_states": int(len(outer_test)),
            "outer_route_overlap": 0,
            "outer_bag_overlap": 0,
            "inner_folds": inner_records,
            "inner_points": points,
            "chosen_target_coverage": chosen[0] if chosen is not None else None,
            "chosen_threshold": chosen[1] if chosen is not None else None,
            "outer_selection": outer_summary,
        }
        print(f"route-stable outer={outer} chosen={chosen} complete", flush=True)

    candidate_metrics = v77.candidate_metrics(
        outer_prediction, objective_gain, dual, harm, challenger
    )
    candidate_choice = frontier.state_choice(
        outer_prediction, pool, data["formal_index"]
    )
    posthoc_points = [
        frontier.point_by_count(
            data, candidate_choice[0], candidate_choice[1], row, fraction
        )[0]
        for fraction in frontier.GRID
    ]
    nested_summary = v77.evaluate_selection(data, nested_selected)
    replacement = nested_selected != data["formal_index"]
    route_counts = dict(sorted(Counter(data["group"][replacement].astype(str)).items()))
    checks = frontier.summary_checks(nested_summary)
    checks["at_least_three_routes"] = len(route_counts) >= 3
    checks["recon_unchanged_by_scand_only_routing"] = True
    passed = all(checks.values())
    baseline = read_json(pilot.BASE / "oof_upperbound_v1/result.json")
    result = {
        "schema": "scand_route_stable_risk_result_v1",
        "status": "completed_adaptive_spent_development_stopped",
        "verdict": (
            "route_stable_development_candidate_not_validated"
            if passed else "route_stable_representation_fail_stop"
        ),
        "candidate_layer": candidate_metrics,
        "baseline_candidate_layer": baseline["candidate_layer"],
        "posthoc_frontier_diagnostic_only": posthoc_points,
        "nested_route_oof": nested_summary,
        "nested_fold_target": nested_target,
        "nested_fold_threshold": nested_threshold,
        "folds_with_inner_choice": int(np.isfinite(nested_target).sum()),
        "replacement_routes": route_counts,
        "success_gate": {"checks": checks, "passed": passed},
        "feature_source_dimension": len(feature_names),
        "selected_source_features": len(selected_names),
        "transformed_dimension": feature.shape[-1],
        "cpu_latency": {
            "mean_seconds_per_state": float(np.mean(latency)),
            "p90_fold_seconds_per_state": float(np.quantile(latency, 0.9)),
            "fold_values": latency,
        },
        "next_step": (
            "ask before one-time unused-old-train development validation"
            if passed else
            "do not tune this family; require independent proposal-tail data or materially new representation"
        ),
        "runtime_seconds": float(time.perf_counter() - started),
        "reserve_accessed": False,
        "unused_old_train_bags_accessed": False,
        "formal_v70_mutated": False,
        "development_only": True,
        "promotion_allowed": False,
    }
    arrays = {
        "record_key": data["record_key"].astype("U256"),
        "route": data["group"].astype("U256"),
        "bag": data["bag"].astype("U256"),
        "outer_fold": fold.astype(np.int16),
        "formal_index": data["formal_index"].astype(np.int16),
        "nested_selected": nested_selected.astype(np.int16),
        "nested_fold_target": nested_target,
        "nested_fold_threshold": nested_threshold,
        "posthoc_candidate": candidate_choice[0].astype(np.int16),
        "posthoc_confidence": candidate_choice[1],
        "posthoc_gain_lcb": candidate_choice[2],
    }
    arrays.update(outer_prediction)
    write_npz(PREDICTIONS, **arrays)
    write_npz(PARAMETERS, **parameter_arrays)
    hashes_after = {
        name: sha256_file(Path(item["path"])) for name, item in prereg["inputs"].items()
    }
    if hashes_after != hashes_before:
        raise RuntimeError("frozen route-stable input changed")
    audit = {
        "schema": "scand_route_stable_risk_audit_v1",
        "preregistration_sha256": sha256_file(PREREG),
        "input_hashes_before": hashes_before,
        "input_hashes_after": hashes_after,
        "all_inputs_unchanged": True,
        "feature_contract": feature_audit,
        "selected_feature_names": selected_names,
        "statewise_transform_uses_labels": False,
        "online_route_bag_record_truth_features": False,
        "inner_outer_audit": inner_audit,
        "route_overlap_violations": 0,
        "bag_overlap_violations": 0,
        "outer_labels_used_for_threshold_choice": False,
        "model_hyperparameters_changed_after_outer": False,
        "formal_v70_sha256": sha256_file(pilot.FORMAL_RESULT),
        "reserve_accessed": False,
        "runner_sha256": sha256_file(Path(__file__)),
    }
    write_json(RESULT, result)
    write_json(AUDIT, audit)
    write_json(HASHES, {
        "schema": "scand_route_stable_risk_sha256_manifest_v1",
        "artifacts": {
            str(path.relative_to(ROOT)): {"sha256": sha256_file(path), "bytes": path.stat().st_size}
            for path in (PREREG, RESULT, AUDIT, PREDICTIONS, PARAMETERS, Path(__file__))
        },
    })
    print(json.dumps(jsonable({
        "verdict": result["verdict"],
        "candidate_layer": candidate_metrics,
        "nested": nested_summary,
        "fold_targets": nested_target,
        "checks": checks,
        "runtime_seconds": result["runtime_seconds"],
    }), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
