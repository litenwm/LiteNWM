#!/usr/bin/env python3
"""Route-OOF soft-correspondence motion probe for adapted-DINO future latents."""

from __future__ import annotations

import argparse
from collections import Counter
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

from recon_latents.models_v2 import integrate_se2_prefix  # noqa: E402
import run_ltwm_action_observable_cycle_probe_v1 as sparse  # noqa: E402
import run_ltwm_full_stream_action_observability_v1 as full  # noqa: E402
import run_ltwm_scand_same_state_hardneg_v1 as previous  # noqa: E402


DEFAULT_CONTRACT = ROOT / "configs/recon/ltwm_soft_correspondence_motion_probe_v1.json"
DEFAULT_OUTPUT = ROOT / "outputs/cem_development_v1/ltwm_soft_correspondence_motion_probe_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def resolve(path: str | Path) -> Path:
    return previous.resolve(path)


def sha256_file(path: Path) -> str:
    return previous.sha256_file(path)


def verify_file(path: Path, expected: str) -> None:
    require(path.is_file(), f"missing input: {path}")
    require(sha256_file(path) == expected, f"input hash mismatch: {path}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_contract(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    contract = json.loads(path.read_text(encoding="utf-8"))
    require(
        contract.get("status")
        == "frozen_before_single_soft_correspondence_route_oof_probe",
        "soft-correspondence contract is not frozen",
    )
    for section, path_key, hash_key in (
        (
            contract["lineage"]["withdrawn_full_stream_probe"],
            "contract_path",
            "contract_sha256",
        ),
        (
            contract["lineage"]["withdrawn_full_stream_probe"],
            "runner_path",
            "runner_sha256",
        ),
        (
            contract["lineage"]["withdrawn_full_stream_probe"],
            "result_path",
            "result_sha256",
        ),
        (contract["inputs"]["data_contract"], "path", "sha256"),
        (contract["inputs"]["frozen_common_table"], "path", "sha256"),
        (contract["inputs"]["formal_v70_result"], "path", "sha256"),
        (contract["inputs"]["warm_checkpoint"], "path", "sha256"),
    ):
        verify_file(resolve(section[path_key]), section[hash_key])
    stream = contract["inputs"]["full_stream_latents"]
    stream_root = resolve(stream["root"])
    verify_file(stream_root / "metadata.json", stream["metadata_sha256"])
    verify_file(stream_root / "splits.json", stream["splits_sha256"])
    verify_file(stream_root / "trajectories.jsonl", stream["trajectories_sha256"])
    require(
        full.ordered_shard_aggregate(stream_root)
        == stream["ordered_shard_sha256_aggregate"],
        "full-stream shard aggregate mismatch",
    )
    require(contract["representation"]["outcome_labels_in_training"] is False, "outcomes forbidden")
    require(contract["representation"]["ate_or_objective_gradient"] is False, "outcome gradients forbidden")
    require(contract["sealed_validation_opened"] is False, "sealed validation forbidden")
    return contract, previous.load_contract(
        resolve(contract["inputs"]["data_contract"]["path"])
    )


class SoftCorrespondenceInverse(nn.Module):
    """Decode motion using explicit spatial correspondence plus token residuals."""

    def __init__(self, *, temperature: float, spatial_sigma: float) -> None:
        super().__init__()
        self.temperature = float(temperature)
        self.spatial_sigma = float(spatial_sigma)
        self.channel = nn.Sequential(nn.Linear(384, 16), nn.GELU())
        self.spatial = nn.Sequential(
            nn.Conv2d(42, 64, kernel_size=3, padding=1),
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
        axis = torch.arange(16, dtype=torch.float32)
        y, x = torch.meshgrid(axis, axis, indexing="ij")
        coordinates = torch.stack((x.reshape(-1), y.reshape(-1)), dim=-1)
        distance_squared = (
            coordinates[:, None] - coordinates[None]
        ).square().sum(dim=-1)
        self.register_buffer("coordinates", coordinates, persistent=True)
        self.register_buffer(
            "spatial_log_prior",
            -distance_squared / (2.0 * self.spatial_sigma**2),
            persistent=True,
        )
        normalized = coordinates / 7.5 - 1.0
        self.register_buffer(
            "normalized_coordinates",
            normalized.reshape(1, 16, 16, 2),
            persistent=True,
        )

    def correspondence(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            source_normalized = F.normalize(source.float(), dim=-1)
            target_normalized = F.normalize(target.float(), dim=-1)
            similarity = torch.bmm(
                source_normalized, target_normalized.transpose(1, 2)
            )
            probability = torch.softmax(
                similarity / self.temperature + self.spatial_log_prior,
                dim=-1,
            )
            expected = torch.matmul(probability, self.coordinates)
            flow = (expected - self.coordinates) / 8.0
            peak = similarity.max(dim=-1).values[..., None]
            entropy = -(
                probability * probability.clamp_min(1.0e-8).log()
            ).sum(dim=-1, keepdim=True) / math.log(256.0)
            field = torch.cat((flow, peak, entropy), dim=-1)
        return field

    def forward(
        self,
        previous_latent: torch.Tensor,
        current_latent: torch.Tensor,
        future_latent: torch.Tensor,
    ) -> torch.Tensor:
        future_residual = self.channel(future_latent - current_latent)
        past_residual = self.channel(current_latent - previous_latent)
        future_motion = self.correspondence(current_latent, future_latent)
        past_motion = self.correspondence(previous_latent, current_latent)
        coordinate = self.normalized_coordinates.expand(len(current_latent), -1, -1, -1)
        value = torch.cat(
            (
                future_residual.reshape(-1, 16, 16, 16),
                past_residual.reshape(-1, 16, 16, 16),
                future_motion.reshape(-1, 16, 16, 4),
                past_motion.reshape(-1, 16, 16, 4),
                coordinate,
            ),
            dim=-1,
        )
        value = value.permute(0, 3, 1, 2)
        return self.head(self.spatial(value))


def build_model(contract: dict[str, Any], device: torch.device) -> SoftCorrespondenceInverse:
    settings = contract["representation"]["soft_correspondence"]
    return SoftCorrespondenceInverse(
        temperature=float(settings["temperature"]),
        spatial_sigma=float(settings["spatial_gaussian_sigma_patches"]),
    ).to(device)


def target_statistics(
    evaluation_data: dict[str, np.ndarray],
    train_index: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    flat = evaluation_data["executed_actions"][train_index].reshape(
        len(train_index), 12
    )
    mean = flat.mean(axis=0).astype(np.float32)
    scale = np.maximum(flat.std(axis=0), 0.01).astype(np.float32)
    return mean, scale


def train_fold(
    model: SoftCorrespondenceInverse,
    stream: dict[str, Any],
    stream_eligible: np.ndarray,
    evaluation_data: dict[str, np.ndarray],
    target_train: np.ndarray,
    *,
    contract: dict[str, Any],
    device: torch.device,
    seed: int,
    label: str,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    training = contract["training"]
    composition = training["batch_composition"]
    stream_batch_size = int(composition["route-excluded full-stream windows"])
    target_batch_size = int(composition["outer-train frozen-table states"])
    require(
        stream_batch_size + target_batch_size == int(training["batch_size"]),
        "batch composition mismatch",
    )
    steps = int(training["fixed_steps_per_fold"])
    stream_schedule = sparse.route_uniform_batches(
        stream_eligible,
        stream["window_route"],
        steps=steps,
        batch_size=stream_batch_size,
        seed=seed,
    )
    target_schedule = sparse.route_uniform_batches(
        target_train,
        evaluation_data["route"],
        steps=steps,
        batch_size=target_batch_size,
        seed=seed + 1000,
    )
    mean, scale = target_statistics(evaluation_data, target_train)
    mean_tensor = torch.from_numpy(mean).to(device)
    scale_tensor = torch.from_numpy(scale).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    model.train()
    rows = []
    started = time.perf_counter()
    for step, (stream_index, target_index) in enumerate(
        zip(stream_schedule, target_schedule), start=1
    ):
        stream_previous, stream_current, stream_future = full.gather_latents(
            stream, stream_index
        )
        target_history = evaluation_data["history"][target_index]
        previous_latent = torch.cat(
            (
                torch.from_numpy(stream_previous).to(device),
                torch.from_numpy(target_history[:, -2]).to(device),
            ),
            dim=0,
        )
        current_latent = torch.cat(
            (
                torch.from_numpy(stream_current).to(device),
                torch.from_numpy(target_history[:, -1]).to(device),
            ),
            dim=0,
        )
        future_latent = torch.cat(
            (
                torch.from_numpy(stream_future).to(device),
                torch.from_numpy(evaluation_data["goal"][target_index]).to(device),
            ),
            dim=0,
        )
        target_action = torch.from_numpy(
            np.concatenate(
                (
                    stream["window_actions"][stream_index],
                    evaluation_data["executed_actions"][target_index],
                )
            ).reshape(-1, 12)
        ).to(device)
        normalized_target = (target_action - mean_tensor) / scale_tensor
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            prediction = model(previous_latent, current_latent, future_latent)
            action_loss = F.smooth_l1_loss(prediction, normalized_target)
            predicted_action = prediction * scale_tensor + mean_tensor
            predicted_path = integrate_se2_prefix(
                predicted_action.reshape(-1, 4, 3)
            )[..., :2]
            target_path = integrate_se2_prefix(
                target_action.reshape(-1, 4, 3)
            )[..., :2]
            path_loss = F.smooth_l1_loss(predicted_path, target_path)
            loss = action_loss + 0.25 * path_loss
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(training["gradient_clip"])
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
        if step % 100 == 0 or step == steps:
            recent = np.asarray(rows[-100:], dtype=np.float64)
            print(
                previous.canonical_json(
                    {
                        "stage": label,
                        "step": step,
                        "steps": steps,
                        "loss": float(recent[:, 0].mean()),
                        "action_loss": float(recent[:, 1].mean()),
                        "path_loss": float(recent[:, 2].mean()),
                        "elapsed_seconds": time.perf_counter() - started,
                    }
                ),
                flush=True,
            )
    values = np.asarray(rows, dtype=np.float64)
    return (
        {
            "steps": steps,
            "batch_size": int(training["batch_size"]),
            "stream_examples_per_batch": stream_batch_size,
            "target_examples_per_batch": target_batch_size,
            "eligible_stream_windows": int(len(stream_eligible)),
            "eligible_stream_routes": int(
                len(np.unique(stream["window_route"][stream_eligible]))
            ),
            "outer_train_states": int(len(target_train)),
            "outer_train_routes": int(
                len(np.unique(evaluation_data["route"][target_train]))
            ),
            "stream_route_sample_counts": dict(
                sorted(Counter(stream["window_route"][stream_schedule.reshape(-1)].tolist()).items())
            ),
            "target_route_sample_counts": dict(
                sorted(Counter(evaluation_data["route"][target_schedule.reshape(-1)].tolist()).items())
            ),
            "mean_loss": float(values[:, 0].mean()),
            "last_100_loss": float(values[-100:, 0].mean()),
            "mean_action_loss": float(values[:, 1].mean()),
            "mean_path_loss": float(values[:, 2].mean()),
            "mean_gradient_norm": float(values[:, 3].mean()),
            "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
            "runtime_seconds": time.perf_counter() - started,
        },
        mean,
        scale,
    )


def predict(
    model: SoftCorrespondenceInverse,
    data: dict[str, np.ndarray],
    indices: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    mean_tensor = torch.from_numpy(mean).to(device)
    scale_tensor = torch.from_numpy(scale).to(device)
    values = []
    with torch.inference_mode():
        for start in range(0, len(indices), 16):
            index = indices[start : start + 16]
            history = torch.from_numpy(data["history"][index]).to(device)
            goal = torch.from_numpy(data["goal"][index]).to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                normalized = model(history[:, -2], history[:, -1], goal)
            values.append((normalized.float() * scale_tensor + mean_tensor).cpu().numpy())
    return np.concatenate(values).reshape(len(indices), 4, 3).astype(np.float32)


def performance_checks(
    summary: dict[str, Any],
    ate_oracle_gain: float,
    contract: dict[str, Any],
) -> tuple[dict[str, bool], float]:
    gate = contract["performance_gate"]
    ratio = float(summary["ate_gain_m"] / ate_oracle_gain)
    checks = {
        "ate_gain": summary["ate_gain_m"] > float(gate["ate_gain_m_min_exclusive"]),
        "objective_gain": summary["objective_gain_m"]
        > float(gate["objective_gain_m_min_exclusive"]),
        "harm_rate": summary["harm_rate"] <= float(gate["harm_rate_max"]),
        "positive_route_fraction": summary["positive_route_fraction"]
        >= float(gate["positive_route_fraction_min"]),
        "ate_oracle_gain_ratio": ratio >= float(gate["ate_oracle_gain_ratio_min"]),
    }
    return checks, ratio


def without_routes(value: dict[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if key != "route_objective_gain_m"}


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
    evaluation_data = previous.load_data(data_contract)
    frozen_table = previous.load_npz(
        resolve(contract["inputs"]["frozen_common_table"]["path"])
    )
    require(
        np.array_equal(
            frozen_table["record_key"].astype(str),
            evaluation_data["record_key"].astype(str),
        ),
        "frozen-table/evaluation order mismatch",
    )
    evaluation_data["trajectory_id"] = frozen_table["trajectory_id"].astype(str)
    stream = full.load_stream(contract, evaluation_data)
    folds = int(contract["training"]["outer_route_folds"])
    assignment, fold_audit = previous.assign_folds(evaluation_data["route"], folds)
    predicted = np.full_like(evaluation_data["executed_actions"], np.nan)
    naive = np.full_like(predicted, np.nan)
    scale_by_state = np.full_like(predicted, np.nan)
    fold_records: dict[str, Any] = {}
    started = time.perf_counter()
    for fold in range(folds):
        fold_dir = output / f"fold_{fold}"
        checkpoint_path = fold_dir / "inverse_probe.pt"
        held_prediction_path = fold_dir / "held_route_goal_prediction.npz"
        target_train = np.flatnonzero(assignment != fold)
        test_index = np.flatnonzero(assignment == fold)
        held_routes = np.unique(evaluation_data["route"][test_index]).astype(str)
        stream_eligible = np.flatnonzero(
            ~np.isin(stream["window_route"], held_routes)
        )
        eligible_trajectory = np.unique(stream["window_trajectory"][stream_eligible])
        held_exact = set(evaluation_data["trajectory_id"][test_index].astype(str))
        eligible_ids = {
            str(stream["trajectory_ids"][index]) for index in eligible_trajectory
        }
        require(
            not set(held_routes).intersection(
                evaluation_data["route"][target_train].astype(str)
            ),
            f"target route leakage fold {fold}",
        )
        require(
            not set(held_routes).intersection(
                stream["window_route"][stream_eligible].astype(str)
            ),
            f"stream route leakage fold {fold}",
        )
        require(
            not held_exact.intersection(eligible_ids),
            f"exact trajectory leakage fold {fold}",
        )
        if checkpoint_path.is_file() and held_prediction_path.is_file():
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            held_prediction = previous.load_npz(held_prediction_path)
            require(
                np.array_equal(held_prediction["state_index"], test_index.astype(np.int16)),
                f"resume mismatch fold {fold}",
            )
            model = build_model(contract, device)
            model.load_state_dict(checkpoint["model"], strict=True)
            training_summary = checkpoint["training"]
            mean = checkpoint["action_mean"].numpy()
            scale = checkpoint["action_scale"].numpy()
            status = "resumed"
        else:
            require(
                not checkpoint_path.exists() and not held_prediction_path.exists(),
                f"partial fold artifact requires inspection: {fold_dir}",
            )
            set_seed(int(contract["training"]["seed"]) + fold)
            model = build_model(contract, device)
            training_summary, mean, scale = train_fold(
                model,
                stream,
                stream_eligible,
                evaluation_data,
                target_train,
                contract=contract,
                device=device,
                seed=int(contract["training"]["seed"]) + fold,
                label=f"outer_fold_{fold}_soft_correspondence",
            )
            goal_prediction = predict(
                model, evaluation_data, test_index, mean, scale, device
            )
            fold_dir.mkdir(parents=True, exist_ok=True)
            previous.write_npz(
                held_prediction_path,
                state_index=test_index.astype(np.int16),
                goal_action_prediction=goal_prediction,
            )
            temporary = checkpoint_path.with_suffix(".pt.tmp")
            torch.save(
                {
                    "model": model.state_dict(),
                    "architecture": {
                        "name": "SoftCorrespondenceInverse",
                        "parameter_count": int(
                            sum(parameter.numel() for parameter in model.parameters())
                        ),
                        "temperature": model.temperature,
                        "spatial_sigma": model.spatial_sigma,
                    },
                    "action_mean": torch.from_numpy(mean),
                    "action_scale": torch.from_numpy(scale),
                    "training": training_summary,
                    "outer_fold": fold,
                    "held_routes": held_routes.tolist(),
                    "route_overlap": 0,
                    "exact_trajectory_overlap": 0,
                    "contract_sha256": sha256_file(contract_path),
                    "outcome_labels_used": False,
                },
                temporary,
            )
            temporary.replace(checkpoint_path)
            held_prediction = previous.load_npz(held_prediction_path)
            status = "trained"
        predicted[test_index] = held_prediction["goal_action_prediction"]
        naive[test_index] = mean.reshape(4, 3)
        scale_by_state[test_index] = scale.reshape(4, 3)
        fold_records[str(fold)] = {
            "status": status,
            "outer_train_states": int(len(target_train)),
            "outer_test_states": int(len(test_index)),
            "held_routes": held_routes.tolist(),
            "eligible_full_stream_windows": int(len(stream_eligible)),
            "eligible_full_stream_trajectories": int(len(eligible_trajectory)),
            "route_overlap": 0,
            "exact_trajectory_overlap": 0,
            "training": training_summary,
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "predictions_sha256": sha256_file(held_prediction_path),
        }
        del model
        torch.cuda.empty_cache()
    require(np.isfinite(predicted).all(), "OOF prediction incomplete")
    physical = sparse.physical_error_summary(
        predicted,
        evaluation_data["executed_actions"],
        naive,
        scale_by_state,
        evaluation_data["route"],
    )
    primary_error = full.path_score(evaluation_data["candidate_actions"], predicted)
    primary_selected = previous.selection_from_error(
        primary_error, evaluation_data["planning_pool"]
    )
    true_path_error = full.path_score(
        evaluation_data["candidate_actions"], evaluation_data["executed_actions"]
    )
    true_path_selected = previous.selection_from_error(
        true_path_error, evaluation_data["planning_pool"]
    )
    true_path_summary = previous.selection_summary(evaluation_data, true_path_selected)
    primary_summary = previous.selection_summary(evaluation_data, primary_selected)
    checks, ratio = performance_checks(
        primary_summary, float(true_path_summary["ate_gain_m"]), contract
    )
    passed = all(checks.values())
    decision = (
        "go_decode_frozen_warm_ltwm_once"
        if passed
        else "no_go_freeze_current_adapted_dino_motion_representation"
    )
    previous.write_npz(
        prediction_path,
        record_key=evaluation_data["record_key"],
        route_group=evaluation_data["route"],
        outer_fold=assignment,
        goal_action_prediction=predicted,
        mean_action_prediction=naive,
        action_scale=scale_by_state,
        primary_path_score=primary_error,
        primary_selected_index=primary_selected.astype(np.int16),
        true_path_selected_index=true_path_selected.astype(np.int16),
    )
    result = {
        "schema": "ltwm_soft_correspondence_motion_route_oof_result_v1",
        "status": "complete_spent_development_representation_probe",
        "decision": decision,
        "contract": {
            "path": str(contract_path),
            "sha256": sha256_file(contract_path),
        },
        "runner": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "representation": {
            "name": "SoftCorrespondenceInverse",
            "parameter_count": int(
                sum(parameter.numel() for parameter in build_model(contract, device).parameters())
            ),
            "outcome_labels_used": False,
        },
        "fold_assignment": fold_audit,
        "folds": fold_records,
        "real_future_action_observability": physical,
        "fixed_planners": {
            "primary_predicted_path": {
                **primary_summary,
                "ate_oracle_gain_ratio": ratio,
            },
            "true_path_ceiling": true_path_summary,
        },
        "performance_gate": {
            "checks": checks,
            "passed": passed,
        },
        "artifacts": {
            "predictions": str(prediction_path),
            "predictions_sha256": sha256_file(prediction_path),
        },
        "protection": {
            "formal_v70_sha256_after": sha256_file(
                resolve(contract["inputs"]["formal_v70_result"]["path"])
            ),
            "formal_v70_unchanged": sha256_file(
                resolve(contract["inputs"]["formal_v70_result"]["path"])
            )
            == contract["inputs"]["formal_v70_result"]["sha256"],
            "frozen_table_sha256_after": sha256_file(
                resolve(contract["inputs"]["frozen_common_table"]["path"])
            ),
            "frozen_table_unchanged": sha256_file(
                resolve(contract["inputs"]["frozen_common_table"]["path"])
            )
            == contract["inputs"]["frozen_common_table"]["sha256"],
            "outer_route_overlap": 0,
            "outer_exact_trajectory_overlap": 0,
            "outcome_labels_used_for_training_or_checkpoint_selection": False,
            "ate_or_objective_gradient_into_ltwm": False,
            "sealed_validation_opened": False,
            "cem_mutated": False,
        },
        "runtime_seconds": time.perf_counter() - started,
    }
    require(result["protection"]["formal_v70_unchanged"], "formal-v70 mutated")
    require(result["protection"]["frozen_table_unchanged"], "frozen table mutated")
    previous.write_json(result_path, result)
    print(
        previous.canonical_json(
            {
                "status": result["status"],
                "decision": decision,
                "physical": {
                    key: value for key, value in physical.items() if key != "route"
                },
                "primary": without_routes(result["fixed_planners"]["primary_predicted_path"]),
                "true_path_ceiling": without_routes(true_path_summary),
                "performance_gate": result["performance_gate"],
                "result": str(result_path),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
