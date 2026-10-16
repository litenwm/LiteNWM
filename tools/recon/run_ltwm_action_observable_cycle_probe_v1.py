#!/usr/bin/env python3
"""Route-OOF action-observability and frozen-LTWM cycle probe."""

from __future__ import annotations

import argparse
from dataclasses import asdict
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
from torch import nn


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(Path(__file__).resolve().parent)]

from planning.nomad_action_adapter import integrate_local_actions  # noqa: E402
from recon_latents.models_v2 import integrate_se2_prefix  # noqa: E402
import run_ltwm_scand_same_state_hardneg_v1 as previous  # noqa: E402


DEFAULT_CONTRACT = ROOT / "configs/recon/ltwm_action_observable_cycle_probe_v1.json"
DEFAULT_OUTPUT = ROOT / "outputs/cem_development_v1/ltwm_action_observable_cycle_probe_v1"
TOL = 1.0e-7


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    return previous.sha256_file(path)


def resolve(path: str | Path) -> Path:
    return previous.resolve(path)


def write_json(path: Path, value: Any) -> None:
    previous.write_json(path, value)


def write_npz(path: Path, **arrays: np.ndarray) -> None:
    previous.write_npz(path, **arrays)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    return previous.load_npz(path)


def canonical_json(value: Any) -> str:
    return previous.canonical_json(value)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_contract(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    contract = json.loads(path.read_text(encoding="utf-8"))
    require(
        contract.get("status") == "frozen_before_single_route_oof_probe",
        "probe contract is not frozen",
    )
    for section, path_key, hash_key in (
        (contract["lineage"]["withdrawal_evidence"], "contract_path", "contract_sha256"),
        (contract["lineage"]["withdrawal_evidence"], "runner_path", "runner_sha256"),
        (contract["lineage"]["withdrawal_evidence"], "result_path", "result_sha256"),
        (contract["inputs"]["data_contract"], "path", "sha256"),
        (contract["inputs"]["warm_checkpoint"], "path", "sha256"),
        (contract["inputs"]["frozen_common_table"], "path", "sha256"),
    ):
        input_path = resolve(section[path_key])
        require(input_path.is_file(), f"missing input: {input_path}")
        require(sha256_file(input_path) == section[hash_key], f"input hash mismatch: {input_path}")
    data_contract_path = resolve(contract["inputs"]["data_contract"]["path"])
    data_contract = previous.load_contract(data_contract_path)
    require(
        contract["inverse_probe"]["outcome_labels_in_training"] is False,
        "outcome labels are forbidden in inverse training",
    )
    require(
        contract["evaluation"]["sealed_validation_opened"] is False,
        "sealed validation is forbidden",
    )
    return contract, data_contract


class ActionObservableInverse(nn.Module):
    """Decode H1-H4 behavior action from actual current/future latent motion."""

    def __init__(self) -> None:
        super().__init__()
        self.channel = nn.Sequential(nn.Linear(384, 24), nn.GELU())
        self.spatial = nn.Sequential(
            nn.Conv2d(48, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 48, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(48, 32, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4 * 4 * 32, 128),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(128, 12),
        )

    def forward(
        self,
        previous_latent: torch.Tensor,
        current_latent: torch.Tensor,
        future_latent: torch.Tensor,
    ) -> torch.Tensor:
        future_delta = self.channel(future_latent - current_latent)
        past_delta = self.channel(current_latent - previous_latent)
        value = torch.cat((future_delta, past_delta), dim=-1)
        value = value.reshape(-1, 16, 16, 48).permute(0, 3, 1, 2)
        return self.head(self.spatial(value))


def route_uniform_batches(
    indices: np.ndarray,
    routes: np.ndarray,
    *,
    steps: int,
    batch_size: int,
    seed: int,
) -> np.ndarray:
    unique = np.asarray(sorted(np.unique(routes[indices])), dtype=object)
    by_route = {name: indices[routes[indices] == name] for name in unique}
    rng = np.random.default_rng(seed)
    result = np.empty((steps, batch_size), dtype=np.int64)
    for step in range(steps):
        names = rng.choice(unique, size=batch_size, replace=True)
        for position, name in enumerate(names):
            choices = by_route[str(name)]
            result[step, position] = choices[rng.integers(0, len(choices))]
    return result


def target_statistics(actions: np.ndarray, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    flat = actions[indices].reshape(len(indices), 12).astype(np.float32)
    mean = flat.mean(axis=0)
    scale = flat.std(axis=0)
    scale = np.maximum(scale, 0.01).astype(np.float32)
    return mean.astype(np.float32), scale


def train_probe(
    model: ActionObservableInverse,
    data: dict[str, np.ndarray],
    train_indices: np.ndarray,
    *,
    contract: dict[str, Any],
    device: torch.device,
    seed: int,
    label: str,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    training = contract["training"]
    mean, scale = target_statistics(data["executed_actions"], train_indices)
    mean_tensor = torch.from_numpy(mean).to(device)
    scale_tensor = torch.from_numpy(scale).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    schedule = route_uniform_batches(
        train_indices,
        data["route"],
        steps=int(training["fixed_steps_per_fold"]),
        batch_size=int(training["batch_size"]),
        seed=seed,
    )
    model.train()
    rows = []
    started = time.perf_counter()
    for step, index in enumerate(schedule, start=1):
        history = torch.from_numpy(data["history"][index]).to(device)
        target_action = torch.from_numpy(
            data["executed_actions"][index].reshape(len(index), 12)
        ).to(device)
        normalized_target = (target_action - mean_tensor) / scale_tensor
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            normalized_prediction = model(history[:, -2], history[:, -1], torch.from_numpy(data["goal"][index]).to(device))
            action_loss = F.smooth_l1_loss(normalized_prediction, normalized_target)
            predicted_action = normalized_prediction * scale_tensor + mean_tensor
            predicted_path = integrate_se2_prefix(predicted_action.reshape(-1, 4, 3))[..., :2]
            target_path = integrate_se2_prefix(target_action.reshape(-1, 4, 3))[..., :2]
            path_loss = F.smooth_l1_loss(predicted_path, target_path)
            loss = action_loss + 0.25 * path_loss
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            float(training["gradient_clip"]),
        )
        optimizer.step()
        rows.append(
            (
                float(loss.detach()),
                float(action_loss.detach()),
                float(path_loss.detach()),
                float(gradient_norm),
            )
        )
        if step % 100 == 0 or step == len(schedule):
            recent = np.asarray(rows[-100:], dtype=np.float64)
            print(
                canonical_json(
                    {
                        "stage": label,
                        "step": step,
                        "steps": len(schedule),
                        "loss": float(recent[:, 0].mean()),
                        "action_loss": float(recent[:, 1].mean()),
                        "path_loss": float(recent[:, 2].mean()),
                        "elapsed_seconds": time.perf_counter() - started,
                    }
                ),
                flush=True,
            )
    value = np.asarray(rows, dtype=np.float64)
    summary = {
        "steps": int(len(schedule)),
        "batch_size": int(schedule.shape[1]),
        "train_states": int(len(train_indices)),
        "train_routes": int(len(np.unique(data["route"][train_indices]))),
        "mean_loss": float(value[:, 0].mean()),
        "last_100_loss": float(value[-100:, 0].mean()),
        "mean_action_loss": float(value[:, 1].mean()),
        "mean_path_loss": float(value[:, 2].mean()),
        "mean_gradient_norm": float(value[:, 3].mean()),
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "runtime_seconds": time.perf_counter() - started,
    }
    return summary, mean, scale


def predict_real_future(
    model: ActionObservableInverse,
    data: dict[str, np.ndarray],
    indices: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    predictions = []
    batch_size = 32
    mean_tensor = torch.from_numpy(mean).to(device)
    scale_tensor = torch.from_numpy(scale).to(device)
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            index = indices[start : start + batch_size]
            history = torch.from_numpy(data["history"][index]).to(device)
            goal = torch.from_numpy(data["goal"][index]).to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                normalized = model(history[:, -2], history[:, -1], goal)
            prediction = normalized.float() * scale_tensor + mean_tensor
            predictions.append(prediction.cpu().numpy())
    return np.concatenate(predictions).reshape(len(indices), 4, 3).astype(np.float32)


def score_candidates(
    candidate_action: np.ndarray,
    target_action: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    scale_value = np.asarray(scale, dtype=np.float32)
    if scale_value.ndim == 1:
        scale_value = scale_value.reshape(1, 1, 4, 3)
    elif scale_value.ndim == 2:
        require(
            scale_value.shape == (len(candidate_action), 12),
            "per-state action scale shape mismatch",
        )
        scale_value = scale_value.reshape(len(candidate_action), 1, 4, 3)
    else:
        raise ValueError(f"unexpected action scale shape: {scale_value.shape}")
    difference = (
        candidate_action - target_action[:, None]
    ) / scale_value
    return np.mean(difference * difference, axis=(-1, -2)).astype(np.float32)


def forward_cycle_predictions(
    inverse: ActionObservableInverse,
    data: dict[str, np.ndarray],
    indices: np.ndarray,
    goal_action: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    *,
    data_contract: dict[str, Any],
    device: torch.device,
    label: str,
) -> dict[str, np.ndarray]:
    world_model, _ = previous.load_model(data_contract, device)
    world_model.eval().requires_grad_(False)
    inverse.eval().requires_grad_(False)
    direct_score = score_candidates(
        data["candidate_actions"][indices],
        goal_action,
        scale,
    )
    forward_score = np.full((len(indices), 65), np.nan, dtype=np.float32)
    cycle_mse = np.full((len(indices), 65), np.nan, dtype=np.float32)
    mean_tensor = torch.from_numpy(mean).to(device)
    scale_tensor = torch.from_numpy(scale).to(device)
    chunk = 16
    started = time.perf_counter()
    with torch.inference_mode():
        for local, state in enumerate(indices.tolist()):
            history = torch.from_numpy(data["history"][state : state + 1]).to(device)
            goal_normalized = torch.from_numpy(
                ((goal_action[local].reshape(12) - mean) / scale)[None]
            ).to(device)
            for start in range(0, 65, chunk):
                end = min(start + chunk, 65)
                actions = torch.from_numpy(
                    data["candidate_actions"][state : state + 1, start:end]
                ).to(device)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    prediction = world_model(history, actions)["predicted_tokens"][:, :, 3]
                    count = end - start
                    previous_latent = history[:, -2, None].expand(1, count, 256, 384).reshape(count, 256, 384)
                    current_latent = history[:, -1, None].expand(1, count, 256, 384).reshape(count, 256, 384)
                    decoded_normalized = inverse(
                        previous_latent,
                        current_latent,
                        prediction.reshape(count, 256, 384),
                    )
                decoded_normalized = decoded_normalized.float()
                candidate_normalized = (
                    actions.reshape(count, 12).float() - mean_tensor
                ) / scale_tensor
                cycle_mse[local, start:end] = (
                    (decoded_normalized - candidate_normalized)
                    .square()
                    .mean(dim=1)
                    .cpu()
                    .numpy()
                )
                forward_score[local, start:end] = (
                    (decoded_normalized - goal_normalized)
                    .square()
                    .mean(dim=1)
                    .cpu()
                    .numpy()
                )
            if (local + 1) % 20 == 0 or local + 1 == len(indices):
                print(
                    canonical_json(
                        {
                            "stage": label,
                            "completed": local + 1,
                            "total": int(len(indices)),
                            "elapsed_seconds": time.perf_counter() - started,
                        }
                    ),
                    flush=True,
                )
    del world_model
    torch.cuda.empty_cache()
    require(np.isfinite(forward_score).all(), "nonfinite forward-inverse score")
    require(np.isfinite(cycle_mse).all(), "nonfinite cycle error")
    return {
        "state_index": indices.astype(np.int16),
        "goal_action_prediction": goal_action,
        "action_mean": np.broadcast_to(mean, (len(indices), 12)).copy(),
        "action_scale": np.broadcast_to(scale, (len(indices), 12)).copy(),
        "direct_inverse_score": direct_score,
        "forward_inverse_score": forward_score,
        "candidate_cycle_mse": cycle_mse,
    }


def physical_error_summary(
    predicted: np.ndarray,
    target: np.ndarray,
    naive: np.ndarray,
    scale: np.ndarray,
    routes: np.ndarray,
) -> dict[str, Any]:
    def errors(value: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        standardized = np.sqrt(
            np.mean(((value - target) / scale.reshape(-1, 4, 3)) ** 2, axis=(-1, -2))
        )
        path = integrate_local_actions(value)
        truth_path = integrate_local_actions(target)
        path_error = np.sqrt(np.mean(np.sum((path - truth_path) ** 2, axis=-1), axis=-1))
        endpoint_error = np.linalg.norm(path[:, -1] - truth_path[:, -1], axis=-1)
        return standardized, path_error, endpoint_error

    predicted_error = errors(predicted)
    naive_error = errors(naive)
    route = {}
    for name in sorted(np.unique(routes)):
        local = routes == name
        route[name] = {
            "states": int(local.sum()),
            "predicted_path_rmse_m": float(predicted_error[1][local].mean()),
            "mean_action_path_rmse_m": float(naive_error[1][local].mean()),
            "path_improved": bool(predicted_error[1][local].mean() < naive_error[1][local].mean()),
        }
    return {
        "standardized_action_rmse": float(predicted_error[0].mean()),
        "mean_action_standardized_rmse": float(naive_error[0].mean()),
        "standardized_action_relative": float(predicted_error[0].mean() / naive_error[0].mean()),
        "integrated_path_rmse_m": float(predicted_error[1].mean()),
        "mean_action_integrated_path_rmse_m": float(naive_error[1].mean()),
        "integrated_path_relative": float(predicted_error[1].mean() / naive_error[1].mean()),
        "endpoint_error_m": float(predicted_error[2].mean()),
        "mean_action_endpoint_error_m": float(naive_error[2].mean()),
        "endpoint_error_relative": float(predicted_error[2].mean() / naive_error[2].mean()),
        "integrated_path_improved_route_fraction": float(
            np.mean([value["path_improved"] for value in route.values()])
        ),
        "route": route,
    }


def performance_checks(
    summary: dict[str, Any],
    contract: dict[str, Any],
) -> dict[str, bool]:
    gate = contract["performance_relevance_gate"]
    return {
        "objective_gain": summary["objective_gain_m"]
        > float(gate["direct_inverse_objective_gain_m_min_exclusive"]),
        "ate_gain": summary["ate_gain_m"] > 0.0,
        "harm_rate": summary["harm_rate"] <= float(gate["direct_inverse_harm_rate_max"]),
        "positive_route_fraction": summary["positive_route_fraction"]
        >= float(gate["direct_inverse_positive_route_fraction_min"]),
        "oracle_gain_ratio": summary["oracle_gain_ratio"]
        >= float(gate["direct_inverse_oracle_gain_ratio_min"]),
    }


def main() -> None:
    args = parse_args()
    require(torch.cuda.is_available(), "CUDA is required")
    contract_path = args.contract.resolve()
    contract, data_contract = load_contract(contract_path)
    output = args.output.resolve()
    result_path = output / "result.json"
    prediction_path = output / "outer_route_oof_predictions.npz"
    require(not result_path.exists(), f"refusing to overwrite {result_path}")
    require(not prediction_path.exists(), f"refusing to overwrite {prediction_path}")
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    data = previous.load_data(data_contract)
    folds = int(contract["training"]["outer_route_folds"])
    assignment, fold_audit = previous.assign_folds(data["route"], folds)
    target = data["executed_actions"].astype(np.float32)
    predicted = np.full_like(target, np.nan)
    naive = np.full_like(target, np.nan)
    scale_by_state = np.full_like(target, np.nan)
    direct_score = np.full((800, 65), np.nan, dtype=np.float32)
    forward_score = np.full((800, 65), np.nan, dtype=np.float32)
    cycle_mse = np.full((800, 65), np.nan, dtype=np.float32)
    fold_records: dict[str, Any] = {}
    started = time.perf_counter()
    for fold in range(folds):
        fold_dir = output / f"fold_{fold}"
        checkpoint_path = fold_dir / "inverse_probe.pt"
        fold_prediction_path = fold_dir / "held_route_predictions.npz"
        train_index = np.flatnonzero(assignment != fold)
        test_index = np.flatnonzero(assignment == fold)
        require(
            not set(data["route"][train_index]).intersection(data["route"][test_index]),
            f"route leakage in fold {fold}",
        )
        if checkpoint_path.is_file() and fold_prediction_path.is_file():
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            fold_prediction = load_npz(fold_prediction_path)
            require(np.array_equal(fold_prediction["state_index"], test_index.astype(np.int16)), f"resume mismatch fold {fold}")
            model = ActionObservableInverse().to(device)
            model.load_state_dict(checkpoint["model"], strict=True)
            training_summary = checkpoint["training"]
            mean = checkpoint["action_mean"].numpy()
            scale = checkpoint["action_scale"].numpy()
            status = "resumed"
        else:
            require(
                not checkpoint_path.exists() and not fold_prediction_path.exists(),
                f"partial fold artifact requires inspection: {fold_dir}",
            )
            set_seed(int(contract["training"]["seed"]) + fold)
            model = ActionObservableInverse().to(device)
            training_summary, mean, scale = train_probe(
                model,
                data,
                train_index,
                contract=contract,
                device=device,
                seed=int(contract["training"]["seed"]) + fold,
                label=f"outer_fold_{fold}_inverse_train",
            )
            goal_prediction = predict_real_future(
                model,
                data,
                test_index,
                mean,
                scale,
                device,
            )
            fold_prediction = forward_cycle_predictions(
                model,
                data,
                test_index,
                goal_prediction,
                mean,
                scale,
                data_contract=data_contract,
                device=device,
                label=f"outer_fold_{fold}_warm_cycle",
            )
            fold_dir.mkdir(parents=True, exist_ok=True)
            write_npz(fold_prediction_path, **fold_prediction)
            torch.save(
                {
                    "model": model.state_dict(),
                    "architecture": asdict_dummy(),
                    "action_mean": torch.from_numpy(mean),
                    "action_scale": torch.from_numpy(scale),
                    "training": training_summary,
                    "outer_fold": fold,
                    "contract_sha256": sha256_file(contract_path),
                    "outcome_labels_used": False,
                },
                checkpoint_path,
            )
            status = "trained"
        predicted[test_index] = fold_prediction["goal_action_prediction"]
        naive[test_index] = mean.reshape(4, 3)
        scale_by_state[test_index] = scale.reshape(4, 3)
        direct_score[test_index] = fold_prediction["direct_inverse_score"]
        forward_score[test_index] = fold_prediction["forward_inverse_score"]
        cycle_mse[test_index] = fold_prediction["candidate_cycle_mse"]
        fold_records[str(fold)] = {
            "status": status,
            "outer_train_states": int(len(train_index)),
            "outer_test_states": int(len(test_index)),
            "outer_train_routes": int(len(np.unique(data["route"][train_index]))),
            "outer_test_routes": int(len(np.unique(data["route"][test_index]))),
            "outer_route_overlap": 0,
            "training": training_summary,
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "predictions_sha256": sha256_file(fold_prediction_path),
        }
        del model
        torch.cuda.empty_cache()
    require(np.isfinite(predicted).all(), "OOF real-future prediction incomplete")
    require(np.isfinite(forward_score).all(), "OOF forward-cycle prediction incomplete")
    physical = physical_error_summary(
        predicted,
        target,
        naive,
        scale_by_state,
        data["route"],
    )
    planning_pool = data["planning_pool"]
    objective_oracle = np.argmin(
        np.where(planning_pool, data["true_metrics"][:, :, 3], np.inf),
        axis=1,
    )
    oracle_summary = previous.selection_summary(data, objective_oracle)
    oracle_gain = oracle_summary["objective_gain_m"]
    direct_selected = previous.selection_from_error(direct_score, planning_pool)
    forward_selected = previous.selection_from_error(forward_score, planning_pool)
    true_action_score = score_candidates(data["candidate_actions"], target, scale_by_state.reshape(800, 12))
    true_action_selected = previous.selection_from_error(true_action_score, planning_pool)
    direct_summary = previous.selection_summary(data, direct_selected, oracle_gain)
    forward_summary = previous.selection_summary(data, forward_selected, oracle_gain)
    true_action_summary = previous.selection_summary(data, true_action_selected, oracle_gain)
    observability_gate = contract["observability_gate"]
    observability_checks = {
        "standardized_action_rmse": physical["standardized_action_relative"]
        <= float(observability_gate["standardized_action_rmse_relative_to_mean_action_max"]),
        "integrated_path_rmse": physical["integrated_path_relative"]
        <= float(observability_gate["integrated_path_rmse_relative_to_mean_action_max"]),
        "endpoint_error": physical["endpoint_error_relative"]
        <= float(observability_gate["endpoint_error_relative_to_mean_action_max"]),
        "route_direction": physical["integrated_path_improved_route_fraction"]
        >= float(observability_gate["integrated_path_improved_route_fraction_min"]),
    }
    direct_checks = performance_checks(direct_summary, contract)
    forward_checks = performance_checks(forward_summary, contract)
    observability_pass = all(observability_checks.values())
    direct_pass = all(direct_checks.values())
    forward_pass = all(forward_checks.values())
    if not observability_pass:
        decision = "stop_real_future_dino_latent_not_action_observable_enough"
    elif not direct_pass:
        decision = "stop_action_observable_signal_not_held_route_ate_relevant"
    elif forward_pass:
        decision = "go_freeze_existing_forward_inverse_metric_no_ltwm_retraining"
    else:
        decision = "go_single_frozen_inverse_action_cycle_ltwm_training"
    write_npz(
        prediction_path,
        record_key=data["record_key"],
        route_group=data["route"],
        outer_fold=assignment,
        goal_action_prediction=predicted,
        mean_action_prediction=naive,
        action_scale=scale_by_state,
        direct_inverse_score=direct_score,
        forward_inverse_score=forward_score,
        candidate_cycle_mse=cycle_mse,
        direct_selected_index=direct_selected.astype(np.int16),
        forward_selected_index=forward_selected.astype(np.int16),
        true_action_nearest_index=true_action_selected.astype(np.int16),
    )
    cycle_mask = planning_pool
    result = {
        "schema": "ltwm_action_observable_cycle_probe_route_oof_result_v1",
        "status": "complete_spent_development_physical_probe",
        "decision": decision,
        "contract": {
            "path": str(contract_path),
            "sha256": sha256_file(contract_path),
        },
        "states": 800,
        "route_groups": 42,
        "fold_assignment": fold_audit,
        "folds": fold_records,
        "real_future_action_observability": physical,
        "warm_ltwm_candidate_action_cycle": {
            "standardized_rmse": float(np.sqrt(cycle_mse[cycle_mask].mean())),
            "candidate_rows": int(cycle_mask.sum()),
        },
        "fixed_planners": {
            "direct_inverse_real_goal_to_candidate_action": direct_summary,
            "forward_inverse_predicted_future_to_real_goal": forward_summary,
            "true_executed_action_nearest_candidate_ceiling": true_action_summary,
            "physical_support_objective_oracle": oracle_summary,
        },
        "gates": {
            "observability": {
                "checks": observability_checks,
                "passed": observability_pass,
            },
            "direct_inverse_performance_relevance": {
                "checks": direct_checks,
                "passed": direct_pass,
            },
            "forward_inverse_performance": {
                "checks": forward_checks,
                "passed": forward_pass,
            },
        },
        "artifacts": {
            "predictions": str(prediction_path),
            "predictions_sha256": sha256_file(prediction_path),
        },
        "protection": {
            "formal_v70_sha256_after": sha256_file(resolve(data_contract["inputs"]["formal_v70_result"]["path"])),
            "formal_v70_unchanged": sha256_file(resolve(data_contract["inputs"]["formal_v70_result"]["path"]))
            == data_contract["inputs"]["formal_v70_result"]["sha256"],
            "frozen_table_sha256_after": sha256_file(resolve(contract["inputs"]["frozen_common_table"]["path"])),
            "frozen_table_unchanged": sha256_file(resolve(contract["inputs"]["frozen_common_table"]["path"]))
            == contract["inputs"]["frozen_common_table"]["sha256"],
            "outcome_labels_used_for_training_or_checkpoint_selection": False,
            "ate_or_objective_gradient_into_ltwm": False,
            "sealed_validation_opened": False,
            "cem_mutated": False,
        },
        "runtime_seconds": time.perf_counter() - started,
    }
    require(result["protection"]["formal_v70_unchanged"], "formal-v70 mutated")
    require(result["protection"]["frozen_table_unchanged"], "frozen table mutated")
    write_json(result_path, result)
    concise = {
        "status": result["status"],
        "decision": decision,
        "physical": {key: value for key, value in physical.items() if key != "route"},
        "cycle": result["warm_ltwm_candidate_action_cycle"],
        "direct_inverse": without_routes(direct_summary),
        "forward_inverse": without_routes(forward_summary),
        "gates": result["gates"],
        "result": str(result_path),
    }
    print(canonical_json(concise), flush=True)


def asdict_dummy() -> dict[str, Any]:
    return {
        "channel_projection": [384, 24],
        "spatial_channels": [48, 64, 48, 32],
        "head": [512, 128, 12],
        "dropout": 0.10,
    }


def without_routes(summary: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in summary.items() if key != "route_objective_gain_m"}


if __name__ == "__main__":
    main()
