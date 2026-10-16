#!/usr/bin/env python3
"""Single preregistered SCAND same-state hard-negative LTWM experiment.

The model remains a true-future latent predictor.  Training sees only history
latents, the executed H4 action, the actual t+4 latent, and label-free union65
actions.  ATE/objective outcomes are loaded only after held-route latent
predictions have been produced, for post-hoc planning evaluation.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from planning.nomad_action_adapter import nomad_waypoints_to_local_actions  # noqa: E402
from recon_latents.models_v2 import (  # noqa: E402
    LiteTokenFormerConfig,
    LiteTokenFormerWMV2,
    count_trainable_parameters,
)


DEFAULT_CONTRACT = ROOT / "configs/recon/ltwm_scand_same_state_hardneg_v1.json"
DEFAULT_OUTPUT = ROOT / "outputs/cem_development_v1/ltwm_scand_same_state_hardneg_v1"
EPS = 1.0e-8
TOL = 1.0e-7


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("baseline", "oof", "final"), required=True)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    return parser.parse_args()


def canonical_json(value: Any) -> str:
    return json.dumps(
        jsonable(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(canonical_json(value) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {key: payload[key].copy() for key in payload.files}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def verify_entry(entry: dict[str, Any], path_key: str, hash_key: str) -> Path:
    path = resolve(entry[path_key])
    require(path.is_file(), f"missing input: {path}")
    actual = sha256_file(path)
    require(actual == entry[hash_key], f"input hash mismatch: {path}")
    return path


def load_contract(path: Path) -> dict[str, Any]:
    contract = json.loads(path.read_text(encoding="utf-8"))
    require(
        contract.get("status") == "frozen_before_baseline_and_single_route_oof_run",
        "experiment contract is not frozen",
    )
    verify_entry(contract["inputs"]["frozen_best_manifest"], "path", "sha256")
    verify_entry(contract["inputs"]["formal_v70_result"], "path", "sha256")
    verify_entry(contract["inputs"]["warm_checkpoint"], "path", "sha256")
    verify_entry(contract["inputs"]["frozen_common_table"], "path", "sha256")
    for cohort in contract["inputs"]["cohorts"]:
        verify_entry(cohort, "feature_path", "feature_sha256")
        verify_entry(cohort, "latent_path", "latent_sha256")
        verify_entry(cohort, "truth_path", "truth_sha256")
    require(contract["data_contract"]["outcome_labels_in_training"] is False, "outcome training forbidden")
    require(contract["data_contract"]["ate_or_objective_gradient_into_ltwm"] is False, "outcome gradients forbidden")
    require(contract["evaluation"]["sealed_validation_opened"] is False, "sealed validation forbidden")
    return contract


def assign_folds(route: np.ndarray, folds: int) -> tuple[np.ndarray, dict[str, Any]]:
    unique, counts = np.unique(route.astype(str), return_counts=True)
    records = sorted(
        zip(unique.tolist(), counts.tolist()),
        key=lambda item: (-item[1], hashlib.sha256(item[0].encode()).hexdigest()),
    )
    loads = [0] * folds
    mapping: dict[str, int] = {}
    for name, count in records:
        fold = min(range(folds), key=lambda value: (loads[value], value))
        mapping[name] = fold
        loads[fold] += count
    assignment = np.asarray([mapping[name] for name in route.astype(str)], dtype=np.int8)
    return assignment, {
        "fold_state_loads": loads,
        "route_to_fold": dict(sorted(mapping.items())),
        "route_overlap": 0,
    }


def load_data(contract: dict[str, Any]) -> dict[str, np.ndarray]:
    table = load_npz(resolve(contract["inputs"]["frozen_common_table"]["path"]))
    record_parts: list[np.ndarray] = []
    history_parts: list[np.ndarray] = []
    goal_parts: list[np.ndarray] = []
    ground_parts: list[np.ndarray] = []
    cohort_parts: list[np.ndarray] = []
    for cohort in contract["inputs"]["cohorts"]:
        feature = load_npz(resolve(cohort["feature_path"]))
        latent = load_npz(resolve(cohort["latent_path"]))
        truth = load_npz(resolve(cohort["truth_path"]))
        states = int(cohort["states"])
        require(len(feature["record_key"]) == states, f"feature state mismatch: {cohort['name']}")
        require(latent["history"].shape == (states, 4, 256, 384), f"history shape mismatch: {cohort['name']}")
        require(latent["goal"].shape == (states, 256, 384), f"goal shape mismatch: {cohort['name']}")
        require(truth["ground_truth_local_m"].shape == (states, 4, 2), f"truth shape mismatch: {cohort['name']}")
        require(
            np.array_equal(feature["record_key"].astype(str), truth["record_key"].astype(str)),
            f"feature/truth record mismatch: {cohort['name']}",
        )
        record_parts.append(feature["record_key"].astype(str))
        history_parts.append(latent["history"].astype(np.float32))
        goal_parts.append(latent["goal"].astype(np.float32))
        ground_parts.append(truth["ground_truth_local_m"].astype(np.float32))
        cohort_parts.append(np.full(states, cohort["name"]))
    record = np.concatenate(record_parts)
    require(
        np.array_equal(record, table["record_key"].astype(str)),
        "cohort order does not exactly match frozen common table",
    )
    require(len(record) == 800, "expected exactly 800 states")
    candidates = table["candidates_feature_waypoint_units"].astype(np.float32)
    ground_m = np.concatenate(ground_parts)
    candidate_actions = nomad_waypoints_to_local_actions(candidates, dataset="recon")
    executed_actions = nomad_waypoints_to_local_actions(
        ground_m / np.float32(0.25),
        dataset="recon",
    )
    candidate_m = candidates * np.float32(0.25)
    path_rms = np.sqrt(
        np.mean(
            np.sum((candidate_m - ground_m[:, None]) ** 2, axis=-1),
            axis=-1,
        )
    )
    endpoint = np.linalg.norm(
        candidate_m[:, :, -1] - ground_m[:, None, -1],
        axis=-1,
    )
    threshold = 0.05
    pool = table["physical_support_pool"].astype(bool)
    distinct = pool & ((path_rms >= threshold) | (endpoint >= threshold))
    minimum = int(contract["data_contract"]["minimum_hard_negatives"])
    require(int((distinct.sum(axis=1) >= minimum).sum()) == 799, "geometry probe contract changed")
    formal = table["formal_v70_index"].astype(np.int64)
    planning_pool = pool.copy()
    planning_pool[np.arange(len(formal)), formal] = True
    require(planning_pool.any(axis=1).all(), "state without planning candidate")
    return {
        "record_key": record,
        "route": table["route_group"].astype(str),
        "cohort": np.concatenate(cohort_parts),
        "history": np.concatenate(history_parts),
        "goal": np.concatenate(goal_parts),
        "ground_m": ground_m,
        "candidate_actions": candidate_actions.astype(np.float32),
        "executed_actions": executed_actions.astype(np.float32),
        "candidates": candidates,
        "distinct": distinct,
        "planning_pool": planning_pool,
        "formal": formal,
        "true_metrics": table["true_metrics_m"].astype(np.float32),
    }


def load_model(contract: dict[str, Any], device: torch.device) -> tuple[LiteTokenFormerWMV2, dict[str, Any]]:
    checkpoint_path = resolve(contract["inputs"]["warm_checkpoint"]["path"])
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    require(
        payload.get("temporal_protocol") == {"history_frames": 4, "future_frames": 4},
        "warm checkpoint temporal protocol changed",
    )
    model = LiteTokenFormerWMV2(LiteTokenFormerConfig(**payload["config"])).to(device)
    model.load_state_dict(payload["model"], strict=True)
    return model, payload


def trainable_action_path(model: LiteTokenFormerWMV2) -> list[torch.nn.Parameter]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    prefixes = ("action_encoder.", "warp.theta_mlp.")
    infixes = (".action_norm.", ".action_attn.", ".modulation.")
    selected = []
    for name, parameter in model.named_parameters():
        if name.startswith(prefixes) or any(token in name for token in infixes):
            parameter.requires_grad_(True)
            selected.append(parameter)
    require(selected, "action-path scope selected no parameters")
    return selected


def amp_context() -> torch.autocast:
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


def prediction_errors(
    prediction: torch.Tensor,
    goal: torch.Tensor,
    current: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    h4 = prediction[:, :, 3].float()
    target = goal[:, None].float()
    mse = (h4 - target).square().mean(dim=(-1, -2))
    absolute_energy = goal.float().square().mean(dim=(-1, -2)).clamp_min(EPS)
    residual_energy = (
        (goal.float() - current.float()).square().mean(dim=(-1, -2)).clamp_min(1.0e-4)
    )
    return mse / absolute_energy[:, None], mse / residual_energy[:, None]


def infer_indices(
    model: LiteTokenFormerWMV2,
    data: dict[str, np.ndarray],
    indices: np.ndarray,
    *,
    device: torch.device,
    candidate_chunk: int,
    label: str,
) -> dict[str, np.ndarray]:
    model.eval()
    states = len(indices)
    candidate_absolute = np.full((states, 65), np.nan, dtype=np.float32)
    candidate_residual = np.full((states, 65), np.nan, dtype=np.float32)
    executed_absolute = np.full(states, np.nan, dtype=np.float32)
    executed_residual = np.full(states, np.nan, dtype=np.float32)
    started = time.perf_counter()
    with torch.inference_mode():
        for local, state in enumerate(indices.tolist()):
            history = torch.from_numpy(data["history"][state : state + 1]).to(device)
            goal = torch.from_numpy(data["goal"][state : state + 1]).to(device)
            current = history[:, -1]
            executed = torch.from_numpy(data["executed_actions"][state : state + 1]).to(device)
            with amp_context():
                output = model(history, executed)
            absolute, residual = prediction_errors(
                output["predicted_tokens"],
                goal,
                current,
            )
            executed_absolute[local] = float(absolute[0, 0])
            executed_residual[local] = float(residual[0, 0])
            for start in range(0, 65, candidate_chunk):
                end = min(start + candidate_chunk, 65)
                actions = torch.from_numpy(
                    data["candidate_actions"][state : state + 1, start:end]
                ).to(device)
                with amp_context():
                    output = model(history, actions)
                absolute, residual = prediction_errors(
                    output["predicted_tokens"],
                    goal,
                    current,
                )
                candidate_absolute[local, start:end] = absolute[0].cpu().numpy()
                candidate_residual[local, start:end] = residual[0].cpu().numpy()
            if (local + 1) % 20 == 0 or local + 1 == states:
                print(
                    canonical_json(
                        {
                            "stage": label,
                            "completed": local + 1,
                            "total": states,
                            "elapsed_seconds": time.perf_counter() - started,
                        }
                    ),
                    flush=True,
                )
    require(np.isfinite(candidate_residual).all(), "nonfinite candidate prediction")
    require(np.isfinite(executed_residual).all(), "nonfinite executed prediction")
    return {
        "state_index": indices.astype(np.int16),
        "candidate_absolute_nmse": candidate_absolute,
        "candidate_residual_nmse": candidate_residual,
        "executed_absolute_nmse": executed_absolute,
        "executed_residual_nmse": executed_residual,
    }


def latent_summary(
    prediction: dict[str, np.ndarray],
    data: dict[str, np.ndarray],
    indices: np.ndarray,
) -> dict[str, Any]:
    candidate = prediction["candidate_residual_nmse"]
    executed = prediction["executed_residual_nmse"]
    distinct = data["distinct"][indices]
    wins = (executed[:, None] < candidate) & distinct
    denominators = distinct.sum(axis=1)
    valid_state = denominators > 0
    rank = 1 + ((candidate < executed[:, None]) & distinct).sum(axis=1)
    routes = data["route"][indices]
    route_values: dict[str, Any] = {}
    for route in sorted(np.unique(routes)):
        local = routes == route
        route_values[route] = {
            "states": int(local.sum()),
            "executed_absolute_nmse": float(prediction["executed_absolute_nmse"][local].mean()),
            "executed_residual_nmse": float(executed[local].mean()),
            "pairwise_win_rate": float(wins[local].sum() / distinct[local].sum()),
            "retrieval_top1": float(np.mean(rank[local] == 1)),
            "mean_reciprocal_rank": float(np.mean(1.0 / rank[local])),
        }
    return {
        "states": int(len(indices)),
        "states_with_distinct_candidates": int(valid_state.sum()),
        "distinct_candidate_comparisons": int(distinct.sum()),
        "executed_absolute_nmse": float(prediction["executed_absolute_nmse"].mean()),
        "executed_residual_nmse": float(executed.mean()),
        "pairwise_win_rate": float(wins.sum() / distinct.sum()),
        "retrieval_top1": float(np.mean(rank[valid_state] == 1)),
        "mean_reciprocal_rank": float(np.mean(1.0 / rank[valid_state])),
        "mean_rank": float(np.mean(rank[valid_state])),
        "route": route_values,
    }


def baseline_phase(
    contract: dict[str, Any],
    contract_path: Path,
    output: Path,
    device: torch.device,
) -> None:
    prediction_path = output / "baseline_predictions.npz"
    result_path = output / "baseline_result.json"
    require(not prediction_path.exists(), f"refusing to overwrite {prediction_path}")
    require(not result_path.exists(), f"refusing to overwrite {result_path}")
    data = load_data(contract)
    model, payload = load_model(contract, device)
    started = time.perf_counter()
    prediction = infer_indices(
        model,
        data,
        np.arange(800, dtype=np.int64),
        device=device,
        candidate_chunk=int(contract["training"]["candidate_inference_chunk"]),
        label="frozen_warm_baseline",
    )
    write_npz(
        prediction_path,
        record_key=data["record_key"],
        route_group=data["route"],
        **prediction,
    )
    summary = latent_summary(prediction, data, np.arange(800))
    result = {
        "schema": "ltwm_scand_same_state_hard_negative_baseline_v1",
        "status": "complete",
        "contract": {
            "path": str(contract_path.resolve()),
            "sha256": sha256_file(contract_path),
        },
        "warm_checkpoint": {
            "path": str(resolve(contract["inputs"]["warm_checkpoint"]["path"])),
            "sha256": contract["inputs"]["warm_checkpoint"]["sha256"],
            "config": payload["config"],
        },
        "latent_retrieval": summary,
        "geometry": {
            "distinct_threshold_m": 0.05,
            "states_with_four_hard_negatives": int((data["distinct"].sum(axis=1) >= 4).sum()),
            "distinct_candidate_rows": int(data["distinct"].sum()),
        },
        "predictions": {
            "path": str(prediction_path.resolve()),
            "sha256": sha256_file(prediction_path),
        },
        "runtime_seconds": time.perf_counter() - started,
        "outcome_labels_used": False,
        "ate_or_objective_gradient_into_ltwm": False,
        "formal_v70_mutated": False,
        "sealed_validation_opened": False,
    }
    write_json(result_path, result)
    print(canonical_json({"status": "baseline_complete", "result": str(result_path), **summary}), flush=True)


def load_baseline(output: Path, data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    path = output / "baseline_predictions.npz"
    require(path.is_file(), "run --phase baseline first")
    value = load_npz(path)
    require(np.array_equal(value["record_key"].astype(str), data["record_key"]), "baseline record mismatch")
    require(np.array_equal(value["state_index"], np.arange(800, dtype=np.int16)), "baseline state order mismatch")
    return value


def hard_negative_index(
    baseline: dict[str, np.ndarray],
    data: dict[str, np.ndarray],
    state: int,
    count: int,
) -> np.ndarray:
    allowed = data["distinct"][state]
    index = np.flatnonzero(allowed)
    require(len(index) >= count, f"state {state} has fewer than {count} hard negatives")
    order = np.argsort(
        baseline["candidate_residual_nmse"][state, index],
        kind="stable",
    )
    return index[order[:count]]


def route_uniform_schedule(
    indices: np.ndarray,
    route: np.ndarray,
    steps: int,
    seed: int,
) -> np.ndarray:
    by_route = {
        name: indices[route[indices] == name]
        for name in sorted(np.unique(route[indices]))
    }
    rng = np.random.default_rng(seed)
    names = np.asarray(list(by_route), dtype=object)
    schedule = []
    while len(schedule) < steps:
        local_names = names.copy()
        rng.shuffle(local_names)
        for name in local_names:
            choices = by_route[str(name)]
            schedule.append(int(choices[rng.integers(0, len(choices))]))
            if len(schedule) == steps:
                break
    return np.asarray(schedule, dtype=np.int64)


def train_fixed(
    model: LiteTokenFormerWMV2,
    data: dict[str, np.ndarray],
    baseline: dict[str, np.ndarray],
    train_indices: np.ndarray,
    *,
    contract: dict[str, Any],
    device: torch.device,
    seed: int,
    label: str,
) -> dict[str, Any]:
    training = contract["training"]
    loss_contract = contract["loss"]
    parameters = trainable_action_path(model)
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    steps = int(training["steps_per_fold"])
    eligible = train_indices[data["distinct"][train_indices].sum(axis=1) >= int(contract["data_contract"]["minimum_hard_negatives"])]
    require(len(eligible) > 0, "no eligible training states")
    schedule = route_uniform_schedule(eligible, data["route"], steps, seed)
    model.train()
    rows = []
    started = time.perf_counter()
    for step, state in enumerate(schedule.tolist(), start=1):
        negative_index = hard_negative_index(
            baseline,
            data,
            state,
            int(contract["data_contract"]["hard_negatives_per_train_state"]),
        )
        history = torch.from_numpy(data["history"][state : state + 1]).to(device)
        goal = torch.from_numpy(data["goal"][state : state + 1]).to(device)
        executed = torch.from_numpy(data["executed_actions"][state : state + 1, None]).to(device)
        negative = torch.from_numpy(data["candidate_actions"][state : state + 1, negative_index]).to(device)
        actions = torch.cat((executed, negative), dim=1)
        optimizer.zero_grad(set_to_none=True)
        with amp_context():
            output = model(history, actions)
            absolute, residual = prediction_errors(
                output["predicted_tokens"],
                goal,
                history[:, -1],
            )
            logits = -torch.log1p(residual) / float(loss_contract["ranking_temperature"])
            ranking = F.cross_entropy(logits, torch.zeros(1, dtype=torch.long, device=device))
            true_absolute = torch.log1p(absolute[:, 0]).mean()
            true_residual = torch.log1p(residual[:, 0]).mean()
            loss = (
                float(loss_contract["true_h4_absolute_log_nmse_weight"]) * true_absolute
                + float(loss_contract["true_h4_residual_log_nmse_weight"]) * true_residual
                + float(loss_contract["same_state_hard_negative_cross_entropy_weight"]) * ranking
            )
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            parameters,
            float(training["gradient_clip"]),
        )
        optimizer.step()
        rows.append(
            (
                float(loss.detach()),
                float(true_absolute.detach()),
                float(true_residual.detach()),
                float(ranking.detach()),
                float(gradient_norm),
            )
        )
        if step % 40 == 0 or step == steps:
            recent = np.asarray(rows[-40:], dtype=np.float64)
            print(
                canonical_json(
                    {
                        "stage": label,
                        "step": step,
                        "steps": steps,
                        "loss": float(recent[:, 0].mean()),
                        "true_abs": float(recent[:, 1].mean()),
                        "true_residual": float(recent[:, 2].mean()),
                        "ranking": float(recent[:, 3].mean()),
                        "elapsed_seconds": time.perf_counter() - started,
                    }
                ),
                flush=True,
            )
    values = np.asarray(rows, dtype=np.float64)
    return {
        "steps": steps,
        "eligible_states": int(len(eligible)),
        "train_routes": int(len(np.unique(data["route"][eligible]))),
        "route_sample_counts": dict(sorted(Counter(data["route"][schedule].tolist()).items())),
        "mean_loss": float(values[:, 0].mean()),
        "last_40_loss": float(values[-40:, 0].mean()),
        "mean_true_absolute_log_nmse": float(values[:, 1].mean()),
        "mean_true_residual_log_nmse": float(values[:, 2].mean()),
        "mean_ranking_cross_entropy": float(values[:, 3].mean()),
        "mean_gradient_norm": float(values[:, 4].mean()),
        "trainable_parameters": int(sum(parameter.numel() for parameter in parameters)),
        "total_parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "runtime_seconds": time.perf_counter() - started,
    }


def save_checkpoint(
    path: Path,
    model: LiteTokenFormerWMV2,
    warm_payload: dict[str, Any],
    *,
    contract_path: Path,
    training_summary: dict[str, Any],
    fold: int | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    payload = {
        "model": model.state_dict(),
        "config": asdict(model.config),
        "temporal_protocol": {"history_frames": 4, "future_frames": 4},
        "parameter_count": count_trainable_parameters(model),
        "total_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "parent_checkpoint_sha256": sha256_file(resolve(json.loads(contract_path.read_text())["inputs"]["warm_checkpoint"]["path"])),
        "parent_split_lock_sha256": warm_payload.get("split_lock_sha256"),
        "split_lock_sha256": None,
        "adaptation_contract": str(contract_path.resolve()),
        "adaptation_contract_sha256": sha256_file(contract_path),
        "outer_fold": fold,
        "training_summary": training_summary,
        "outcome_labels_used": False,
        "ate_or_objective_gradient_into_ltwm": False,
    }
    torch.save(payload, temporary)
    temporary.replace(path)


def stack_fold_predictions(
    fold_values: list[tuple[np.ndarray, dict[str, np.ndarray]]],
) -> dict[str, np.ndarray]:
    result = {
        "candidate_absolute_nmse": np.full((800, 65), np.nan, dtype=np.float32),
        "candidate_residual_nmse": np.full((800, 65), np.nan, dtype=np.float32),
        "executed_absolute_nmse": np.full(800, np.nan, dtype=np.float32),
        "executed_residual_nmse": np.full(800, np.nan, dtype=np.float32),
    }
    for indices, prediction in fold_values:
        for key in result:
            result[key][indices] = prediction[key]
    require(all(np.isfinite(value).all() for value in result.values()), "OOF prediction incomplete")
    return result


def selection_from_error(
    error: np.ndarray,
    pool: np.ndarray,
) -> np.ndarray:
    masked = np.where(pool, error, np.inf)
    require(np.isfinite(masked).any(axis=1).all(), "planning state without finite candidate")
    return np.argmin(masked, axis=1).astype(np.int64)


def selection_summary(
    data: dict[str, np.ndarray],
    selected: np.ndarray,
    oracle_gain: float | None = None,
) -> dict[str, Any]:
    row = np.arange(len(selected))
    formal = data["formal"]
    truth = data["true_metrics"]
    formal_metric = truth[row, formal].astype(np.float64)
    selected_metric = truth[row, selected].astype(np.float64)
    gain = formal_metric - selected_metric
    replacement = selected != formal
    dual = (gain[:, 0] > TOL) & (gain[:, 3] > TOL)
    harm = (gain[:, 0] < -TOL) | (gain[:, 3] < -TOL)
    route_gain = {}
    for route in sorted(np.unique(data["route"])):
        mask = data["route"] == route
        route_gain[route] = float(gain[mask, 3].mean())
    replacements = int(replacement.sum())
    objective_gain = float(gain[:, 3].mean())
    return {
        "replacements": replacements,
        "coverage": float(replacement.mean()),
        "ate_gain_m": float(gain[:, 0].mean()),
        "objective_gain_m": objective_gain,
        "dual_count": int((dual & replacement).sum()),
        "dual_rate": float((dual & replacement).sum() / replacements) if replacements else 0.0,
        "harm_count": int((harm & replacement).sum()),
        "harm_rate": float((harm & replacement).sum() / replacements) if replacements else 0.0,
        "formal_mean_m": {
            "ate": float(formal_metric[:, 0].mean()),
            "rpe": float(formal_metric[:, 1].mean()),
            "endpoint": float(formal_metric[:, 2].mean()),
            "objective": float(formal_metric[:, 3].mean()),
        },
        "selected_mean_m": {
            "ate": float(selected_metric[:, 0].mean()),
            "rpe": float(selected_metric[:, 1].mean()),
            "endpoint": float(selected_metric[:, 2].mean()),
            "objective": float(selected_metric[:, 3].mean()),
        },
        "positive_route_fraction": float(np.mean(np.asarray(list(route_gain.values())) > 0.0)),
        "route_objective_gain_m": route_gain,
        "oracle_gain_ratio": (
            float(objective_gain / oracle_gain)
            if oracle_gain is not None and oracle_gain > 0
            else None
        ),
    }


def oracle_and_ranking(
    data: dict[str, np.ndarray],
    error: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    row = np.arange(800)
    objective = data["true_metrics"][:, :, 3]
    oracle = np.argmin(np.where(data["planning_pool"], objective, np.inf), axis=1)
    opportunity = objective[row, oracle] < objective[row, data["formal"]] - TOL
    order = np.argsort(np.where(data["planning_pool"], error, np.inf), axis=1, kind="stable")
    recall = {}
    for topk in (1, 3, 5, 10):
        hit = (order[:, :topk] == oracle[:, None]).any(axis=1)
        recall[str(topk)] = {
            "opportunity_states": int(opportunity.sum()),
            "exact_oracle_candidate_recall": float(hit[opportunity].mean()),
        }
    return oracle, recall


def compare_route_latent(
    baseline: dict[str, Any],
    adapted: dict[str, Any],
) -> dict[str, Any]:
    routes = sorted(baseline["route"])
    absolute_better = []
    residual_better = []
    retrieval_better = []
    values = {}
    for route in routes:
        before = baseline["route"][route]
        after = adapted["route"][route]
        absolute = after["executed_absolute_nmse"] < before["executed_absolute_nmse"]
        residual = after["executed_residual_nmse"] < before["executed_residual_nmse"]
        retrieval = after["pairwise_win_rate"] > before["pairwise_win_rate"]
        absolute_better.append(absolute)
        residual_better.append(residual)
        retrieval_better.append(retrieval)
        values[route] = {
            "executed_absolute_nmse_delta": after["executed_absolute_nmse"] - before["executed_absolute_nmse"],
            "executed_residual_nmse_delta": after["executed_residual_nmse"] - before["executed_residual_nmse"],
            "pairwise_win_rate_delta": after["pairwise_win_rate"] - before["pairwise_win_rate"],
        }
    return {
        "routes": len(routes),
        "absolute_nmse_improved_route_fraction": float(np.mean(absolute_better)),
        "residual_nmse_improved_route_fraction": float(np.mean(residual_better)),
        "pairwise_win_improved_route_fraction": float(np.mean(retrieval_better)),
        "route_delta": values,
    }


def oof_phase(
    contract: dict[str, Any],
    contract_path: Path,
    output: Path,
    device: torch.device,
) -> None:
    result_path = output / "result.json"
    prediction_path = output / "outer_route_oof_predictions.npz"
    require(not result_path.exists(), f"refusing to overwrite {result_path}")
    require(not prediction_path.exists(), f"refusing to overwrite {prediction_path}")
    data = load_data(contract)
    baseline = load_baseline(output, data)
    folds = int(contract["training"]["outer_route_folds"])
    assignment, fold_audit = assign_folds(data["route"], folds)
    fold_values = []
    fold_records: dict[str, Any] = {}
    started = time.perf_counter()
    for fold in range(folds):
        fold_dir = output / f"fold_{fold}"
        checkpoint_path = fold_dir / "checkpoint.pt"
        prediction_fold_path = fold_dir / "held_route_predictions.npz"
        train_index = np.flatnonzero(assignment != fold)
        test_index = np.flatnonzero(assignment == fold)
        require(
            not set(data["route"][train_index]).intersection(data["route"][test_index]),
            f"route leakage in fold {fold}",
        )
        if checkpoint_path.is_file() and prediction_fold_path.is_file():
            prediction = load_npz(prediction_fold_path)
            require(np.array_equal(prediction["state_index"], test_index.astype(np.int16)), f"resume state mismatch fold {fold}")
            fold_values.append((test_index, prediction))
            fold_records[str(fold)] = {
                "status": "resumed",
                "checkpoint": str(checkpoint_path.resolve()),
                "checkpoint_sha256": sha256_file(checkpoint_path),
                "prediction_sha256": sha256_file(prediction_fold_path),
                "outer_train_states": int(len(train_index)),
                "outer_test_states": int(len(test_index)),
                "outer_route_overlap": 0,
            }
            print(canonical_json({"stage": "resume_fold", "fold": fold}), flush=True)
            continue
        require(not checkpoint_path.exists() and not prediction_fold_path.exists(), f"partial fold artifact requires manual inspection: {fold_dir}")
        set_seed(int(contract["training"]["seed"]) + fold)
        model, warm_payload = load_model(contract, device)
        training_summary = train_fixed(
            model,
            data,
            baseline,
            train_index,
            contract=contract,
            device=device,
            seed=int(contract["training"]["seed"]) + fold,
            label=f"outer_fold_{fold}_train",
        )
        prediction = infer_indices(
            model,
            data,
            test_index,
            device=device,
            candidate_chunk=int(contract["training"]["candidate_inference_chunk"]),
            label=f"outer_fold_{fold}_held_route",
        )
        fold_dir.mkdir(parents=True, exist_ok=True)
        write_npz(prediction_fold_path, **prediction)
        save_checkpoint(
            checkpoint_path,
            model,
            warm_payload,
            contract_path=contract_path,
            training_summary=training_summary,
            fold=fold,
        )
        fold_values.append((test_index, prediction))
        fold_records[str(fold)] = {
            "status": "trained",
            "checkpoint": str(checkpoint_path.resolve()),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "prediction_sha256": sha256_file(prediction_fold_path),
            "outer_train_states": int(len(train_index)),
            "outer_test_states": int(len(test_index)),
            "outer_train_routes": int(len(np.unique(data["route"][train_index]))),
            "outer_test_routes": int(len(np.unique(data["route"][test_index]))),
            "outer_route_overlap": 0,
            "training": training_summary,
        }
        del model
        torch.cuda.empty_cache()
    oof = stack_fold_predictions(fold_values)
    write_npz(
        prediction_path,
        record_key=data["record_key"],
        route_group=data["route"],
        outer_fold=assignment,
        **oof,
    )
    all_index = np.arange(800, dtype=np.int64)
    baseline_latent = latent_summary(baseline, data, all_index)
    oof_latent = latent_summary(oof, data, all_index)
    route_latent = compare_route_latent(baseline_latent, oof_latent)
    oracle_selected, baseline_recall = oracle_and_ranking(
        data,
        baseline["candidate_residual_nmse"],
    )
    _, oof_recall = oracle_and_ranking(data, oof["candidate_residual_nmse"])
    oracle_summary = selection_summary(data, oracle_selected)
    oracle_gain = oracle_summary["objective_gain_m"]
    warm_selected = selection_from_error(
        baseline["candidate_residual_nmse"],
        data["planning_pool"],
    )
    oof_selected = selection_from_error(
        oof["candidate_residual_nmse"],
        data["planning_pool"],
    )
    warm_summary = selection_summary(data, warm_selected, oracle_gain)
    oof_summary = selection_summary(data, oof_selected, oracle_gain)
    gate_contract = contract["oof_go_gate"]
    checks = {
        "executed_h4_absolute_nmse_guard": (
            oof_latent["executed_absolute_nmse"]
            <= baseline_latent["executed_absolute_nmse"]
            * (1.0 + float(gate_contract["executed_h4_absolute_nmse_relative_degradation_max"]))
        ),
        "executed_h4_residual_nmse_improves": (
            oof_latent["executed_residual_nmse"]
            < baseline_latent["executed_residual_nmse"]
        ),
        "pairwise_win_improvement": (
            oof_latent["pairwise_win_rate"] - baseline_latent["pairwise_win_rate"]
            >= float(gate_contract["pairwise_retrieval_win_rate_absolute_improvement_min"])
        ),
        "pairwise_win_minimum": (
            oof_latent["pairwise_win_rate"]
            >= float(gate_contract["pairwise_retrieval_win_rate_min"])
        ),
        "retrieval_top1_improves": (
            oof_latent["retrieval_top1"] > baseline_latent["retrieval_top1"]
        ),
        "objective_gain_exceeds_existing_best": (
            oof_summary["objective_gain_m"]
            > float(gate_contract["objective_gain_m_min_exclusive"])
        ),
        "ate_gain_positive": oof_summary["ate_gain_m"] > 0.0,
        "harm_not_uncontrolled": oof_summary["harm_rate"] <= float(gate_contract["harm_rate_max"]),
        "positive_route_fraction": (
            oof_summary["positive_route_fraction"]
            >= float(gate_contract["positive_route_fraction_min"])
        ),
        "oracle_gain_ratio": (
            oof_summary["oracle_gain_ratio"]
            >= float(gate_contract["oracle_objective_gain_ratio_min"])
        ),
    }
    go = all(checks.values())
    formal_path = resolve(contract["inputs"]["formal_v70_result"]["path"])
    table_path = resolve(contract["inputs"]["frozen_common_table"]["path"])
    result = {
        "schema": "ltwm_scand_same_state_hard_negative_route_oof_result_v1",
        "status": "complete_spent_development_performance_prototype",
        "decision": (
            "go_train_identical_full_development_checkpoint_then_recon_guard"
            if go
            else "no_go_withdraw_same_state_hard_negative_action_path_hypothesis"
        ),
        "contract": {
            "path": str(contract_path.resolve()),
            "sha256": sha256_file(contract_path),
        },
        "states": 800,
        "route_groups": 42,
        "fold_assignment": fold_audit,
        "folds": fold_records,
        "frozen_warm_latent": baseline_latent,
        "outer_route_oof_latent": oof_latent,
        "route_latent_consistency": route_latent,
        "fixed_minimum_goal_latent_error_planner": {
            "frozen_warm": warm_summary,
            "outer_route_oof_adapted": oof_summary,
            "physical_support_objective_oracle": oracle_summary,
            "ranking": {
                "frozen_warm": baseline_recall,
                "outer_route_oof_adapted": oof_recall,
            },
        },
        "go_gate": {"checks": checks, "passed": go},
        "artifacts": {
            "baseline_predictions_sha256": sha256_file(output / "baseline_predictions.npz"),
            "oof_predictions": str(prediction_path.resolve()),
            "oof_predictions_sha256": sha256_file(prediction_path),
        },
        "protection": {
            "formal_v70_sha256_expected": contract["inputs"]["formal_v70_result"]["sha256"],
            "formal_v70_sha256_after": sha256_file(formal_path),
            "formal_v70_unchanged": sha256_file(formal_path) == contract["inputs"]["formal_v70_result"]["sha256"],
            "frozen_table_sha256_expected": contract["inputs"]["frozen_common_table"]["sha256"],
            "frozen_table_sha256_after": sha256_file(table_path),
            "frozen_table_unchanged": sha256_file(table_path) == contract["inputs"]["frozen_common_table"]["sha256"],
            "outcome_labels_used_for_training_or_checkpoint_selection": False,
            "ate_or_objective_gradient_into_ltwm": False,
            "sealed_validation_opened": False,
            "cem_mutated": False,
        },
        "runtime_seconds": time.perf_counter() - started,
    }
    require(result["protection"]["formal_v70_unchanged"], "formal-v70 mutated")
    require(result["protection"]["frozen_table_unchanged"], "frozen common table mutated")
    write_json(result_path, result)
    print(
        canonical_json(
            {
                "status": "oof_complete",
                "decision": result["decision"],
                "go": go,
                "latent": oof_latent,
                "planning": oof_summary,
                "result": str(result_path),
            }
        ),
        flush=True,
    )


def final_phase(
    contract: dict[str, Any],
    contract_path: Path,
    output: Path,
    device: torch.device,
) -> None:
    result_path = output / "result.json"
    require(result_path.is_file(), "run --phase oof first")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    require(result["go_gate"]["passed"] is True, "OOF gate did not authorize final fit")
    checkpoint_path = output / "final" / "checkpoint.pt"
    summary_path = output / "final" / "training_summary.json"
    require(not checkpoint_path.exists(), f"refusing to overwrite {checkpoint_path}")
    require(not summary_path.exists(), f"refusing to overwrite {summary_path}")
    data = load_data(contract)
    baseline = load_baseline(output, data)
    set_seed(int(contract["training"]["seed"]) + 100)
    model, warm_payload = load_model(contract, device)
    training_summary = train_fixed(
        model,
        data,
        baseline,
        np.arange(800, dtype=np.int64),
        contract=contract,
        device=device,
        seed=int(contract["training"]["seed"]) + 100,
        label="full_800_train",
    )
    save_checkpoint(
        checkpoint_path,
        model,
        warm_payload,
        contract_path=contract_path,
        training_summary=training_summary,
        fold=None,
    )
    final = {
        "schema": "ltwm_scand_same_state_hard_negative_final_fit_v1",
        "status": "trained_pending_recon_paired_guard",
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "training": training_summary,
        "contract_sha256": sha256_file(contract_path),
        "formal_v70_mutated": False,
        "sealed_validation_opened": False,
        "outcome_labels_used_for_training": False,
        "ate_or_objective_gradient_into_ltwm": False,
    }
    write_json(summary_path, final)
    print(canonical_json(final), flush=True)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    contract_path = args.contract.resolve()
    contract = load_contract(contract_path)
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if args.phase == "baseline":
        baseline_phase(contract, contract_path, args.output, device)
    elif args.phase == "oof":
        oof_phase(contract, contract_path, args.output, device)
    else:
        final_phase(contract, contract_path, args.output, device)


if __name__ == "__main__":
    main()
