#!/usr/bin/env python3
"""Strict route-OOF raw optical-flow future-motion latent probe."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import sys
import time
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(Path(__file__).resolve().parent)]

from recon_latents.models_v2 import integrate_se2_prefix  # noqa: E402
import run_ltwm_action_aware_adapter_probe_v1 as adapter_probe  # noqa: E402
import run_ltwm_action_observable_cycle_probe_v1 as sparse  # noqa: E402
import run_ltwm_full_stream_action_observability_v1 as full  # noqa: E402
import run_ltwm_scand_same_state_hardneg_v1 as previous  # noqa: E402
import run_ltwm_soft_correspondence_motion_probe_v1 as soft  # noqa: E402


DEFAULT_CONTRACT = ROOT / "configs/recon/ltwm_raw_flow_motion_probe_v1.json"
DEFAULT_OUTPUT = ROOT / "outputs/cem_development_v1/ltwm_raw_flow_motion_probe_v1"


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
        contract.get("status") == "frozen_before_single_raw_flow_route_oof_probe",
        "raw-flow contract is not frozen",
    )
    for section, path_key, hash_key in (
        (contract["lineage"]["frozen_dino_ceiling"], "contract_path", "contract_sha256"),
        (contract["lineage"]["frozen_dino_ceiling"], "runner_path", "runner_sha256"),
        (contract["lineage"]["frozen_dino_ceiling"], "result_path", "result_sha256"),
        (contract["inputs"]["data_contract"], "path", "sha256"),
        (contract["inputs"]["frozen_common_table"], "path", "sha256"),
        (contract["inputs"]["formal_v70_result"], "path", "sha256"),
    ):
        verify_file(resolve(section[path_key]), section[hash_key])
    stream = contract["inputs"]["full_stream_latents"]
    root = resolve(stream["root"])
    verify_file(root / "metadata.json", stream["metadata_sha256"])
    verify_file(root / "splits.json", stream["splits_sha256"])
    verify_file(root / "trajectories.jsonl", stream["trajectories_sha256"])
    require(
        full.ordered_shard_aggregate(root) == stream["ordered_shard_sha256_aggregate"],
        "full-stream aggregate mismatch",
    )
    require(contract["flow_latent"]["persistent_new_table"] is False, "flow cache must remain ephemeral")
    require(contract["flow_latent"]["outcome_labels_used"] is False, "outcomes forbidden")
    require(contract["probe"]["ate_or_objective_gradient"] is False, "outcome gradients forbidden")
    require(contract["sealed_validation_opened"] is False, "sealed validation forbidden")
    return contract, previous.load_contract(resolve(contract["inputs"]["data_contract"]["path"]))


def load_gray(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    require(image is not None, f"cannot read image: {path}")
    return cv2.resize(image, (224, 224), interpolation=cv2.INTER_CUBIC)


def flow_feature(
    estimator: Any,
    source: np.ndarray,
    target: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
) -> np.ndarray:
    flow = estimator.calc(source, target, None)
    require(flow.shape == (224, 224, 2) and np.isfinite(flow).all(), "invalid DIS flow")
    warped = cv2.remap(
        target,
        (grid_x + flow[..., 0]).astype(np.float32),
        (grid_y + flow[..., 1]).astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    residual = np.abs(source.astype(np.float32) - warped.astype(np.float32)) / 255.0
    small_flow = cv2.resize(flow, (16, 16), interpolation=cv2.INTER_AREA)
    magnitude = np.linalg.norm(small_flow, axis=-1, keepdims=True)
    small_residual = cv2.resize(
        residual, (16, 16), interpolation=cv2.INTER_AREA
    )[..., None]
    feature = np.concatenate(
        (small_flow / 32.0, magnitude / 32.0, small_residual), axis=-1
    )
    return feature.astype(np.float16)


def build_flow_data(
    contract: dict[str, Any],
    data_contract: dict[str, Any],
    data: dict[str, np.ndarray],
) -> dict[str, Any]:
    cv2.setNumThreads(max(1, min(8, cv2.getNumberOfCPUs())))
    estimator = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    axis = np.arange(224, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(axis, axis)
    stream_root = resolve(contract["inputs"]["full_stream_latents"]["root"])
    metadata = json.loads((stream_root / "metadata.json").read_text(encoding="utf-8"))
    raw_root = Path(contract["inputs"]["full_stream_latents"]["raw_root"]).resolve()
    stream_flow = []
    stream_action = []
    stream_route = []
    stream_trajectory = []
    started = time.perf_counter()
    total_pairs = 0
    for trajectory_index, row in enumerate(metadata["trajectories"]):
        trajectory_id = str(row["trajectory_id"])
        paths = sorted(
            (raw_root / trajectory_id).glob("*.jpg"),
            key=lambda value: int(value.stem),
        )
        require(len(paths) == int(row["num_frames"]), f"image count: {trajectory_id}")
        images = [load_gray(path) for path in paths]
        adjacent = [
            flow_feature(estimator, images[index], images[index + 1], grid_x, grid_y)
            for index in range(len(images) - 1)
        ]
        currents = range(3, len(images) - 4)
        future = [
            flow_feature(estimator, images[t], images[t + 4], grid_x, grid_y)
            for t in currents
        ]
        with np.load(stream_root / "shards" / str(row["shard_file"]), allow_pickle=False) as shard:
            action = shard["actions_continuous"].astype(np.float32)
        route = full.route_group(trajectory_id)
        for future_index, t in enumerate(currents):
            stream_flow.append(
                np.concatenate(
                    (
                        adjacent[t - 3],
                        adjacent[t - 2],
                        adjacent[t - 1],
                        future[future_index],
                    ),
                    axis=-1,
                )
            )
            stream_action.append(action[t : t + 4])
            stream_route.append(route)
            stream_trajectory.append(trajectory_id)
        total_pairs += len(adjacent) + len(future)
        if (trajectory_index + 1) % 10 == 0 or trajectory_index + 1 == len(metadata["trajectories"]):
            print(
                previous.canonical_json(
                    {
                        "stage": "ephemeral_full_stream_dis_flow",
                        "trajectories": trajectory_index + 1,
                        "total": len(metadata["trajectories"]),
                        "flow_pairs": total_pairs,
                        "windows": len(stream_flow),
                        "elapsed_seconds": time.perf_counter() - started,
                    }
                ),
                flush=True,
            )
    require(len(stream_flow) == 40763, "unexpected stream flow windows")
    roots = adapter_probe.cohort_roots(data_contract)
    target_flow = np.empty((800, 16, 16, 16), dtype=np.float16)
    for index in range(800):
        key = str(data["record_key"][index])
        trajectory_id, current_text = key.rsplit(":", 1)
        current_t = int(current_text)
        root = roots[str(data["cohort"][index])]
        frames = [current_t - 3, current_t - 2, current_t - 1, current_t, current_t + 4]
        images = [load_gray(root / trajectory_id / f"{frame}.jpg") for frame in frames]
        pairs = [
            flow_feature(estimator, images[left], images[right], grid_x, grid_y)
            for left, right in ((0, 1), (1, 2), (2, 3), (3, 4))
        ]
        target_flow[index] = np.concatenate(pairs, axis=-1)
        if (index + 1) % 100 == 0:
            print(
                previous.canonical_json(
                    {
                        "stage": "ephemeral_target_dis_flow",
                        "states": index + 1,
                        "total": 800,
                    }
                ),
                flush=True,
            )
    return {
        "stream_flow": np.stack(stream_flow).astype(np.float16),
        "stream_action": np.stack(stream_action).astype(np.float32),
        "stream_route": np.asarray(stream_route),
        "stream_trajectory": np.asarray(stream_trajectory),
        "target_flow": target_flow,
        "flow_pairs": int(total_pairs + 4 * 800),
        "runtime_seconds": time.perf_counter() - started,
    }


class FlowMotionProbe(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.spatial = nn.Sequential(
            nn.Conv2d(18, 64, kernel_size=3, padding=1),
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
        axis = torch.linspace(-1.0, 1.0, 16)
        y, x = torch.meshgrid(axis, axis, indexing="ij")
        self.register_buffer(
            "coordinate",
            torch.stack((x, y), dim=0)[None],
        )

    def forward(self, flow: torch.Tensor) -> torch.Tensor:
        value = torch.cat(
            (
                flow.permute(0, 3, 1, 2),
                self.coordinate.expand(len(flow), -1, -1, -1),
            ),
            dim=1,
        )
        return self.head(self.spatial(value))


def target_statistics(
    data: dict[str, np.ndarray], indices: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    flat = data["executed_actions"][indices].reshape(len(indices), 12)
    return (
        flat.mean(axis=0).astype(np.float32),
        np.maximum(flat.std(axis=0), 0.01).astype(np.float32),
    )


def train_fold(
    model: FlowMotionProbe,
    flow_data: dict[str, Any],
    stream_eligible: np.ndarray,
    data: dict[str, np.ndarray],
    target_train: np.ndarray,
    *,
    contract: dict[str, Any],
    device: torch.device,
    seed: int,
    label: str,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    training = contract["training"]
    composition = training["batch_composition"]
    stream_batch = int(composition["route-excluded full-stream windows"])
    target_batch = int(composition["outer-train frozen-table states"])
    steps = int(training["fixed_steps_per_fold"])
    stream_schedule = sparse.route_uniform_batches(
        stream_eligible,
        flow_data["stream_route"],
        steps=steps,
        batch_size=stream_batch,
        seed=seed,
    )
    target_schedule = sparse.route_uniform_batches(
        target_train,
        data["route"],
        steps=steps,
        batch_size=target_batch,
        seed=seed + 1000,
    )
    mean, scale = target_statistics(data, target_train)
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
        flow = torch.from_numpy(
            np.concatenate(
                (
                    flow_data["stream_flow"][stream_index],
                    flow_data["target_flow"][target_index],
                )
            )
        ).to(device)
        target_action = torch.from_numpy(
            np.concatenate(
                (
                    flow_data["stream_action"][stream_index],
                    data["executed_actions"][target_index],
                )
            ).reshape(-1, 12)
        ).to(device)
        normalized_target = (target_action - mean_tensor) / scale_tensor
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            prediction = model(flow)
            action_loss = F.smooth_l1_loss(prediction, normalized_target)
            predicted_action = prediction * scale_tensor + mean_tensor
            path_loss = F.smooth_l1_loss(
                integrate_se2_prefix(predicted_action.reshape(-1, 4, 3))[..., :2],
                integrate_se2_prefix(target_action.reshape(-1, 4, 3))[..., :2],
            )
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
    values = np.asarray(rows)
    return (
        {
            "steps": steps,
            "batch_size": int(training["batch_size"]),
            "eligible_stream_windows": int(len(stream_eligible)),
            "outer_train_states": int(len(target_train)),
            "mean_loss": float(values[:, 0].mean()),
            "last_100_loss": float(values[-100:, 0].mean()),
            "mean_action_loss": float(values[:, 1].mean()),
            "mean_path_loss": float(values[:, 2].mean()),
            "mean_gradient_norm": float(values[:, 3].mean()),
            "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
            "stream_route_sample_counts": dict(
                sorted(Counter(flow_data["stream_route"][stream_schedule.reshape(-1)].tolist()).items())
            ),
            "runtime_seconds": time.perf_counter() - started,
        },
        mean,
        scale,
    )


def predict(
    model: FlowMotionProbe,
    target_flow: np.ndarray,
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
        for start in range(0, len(indices), 32):
            index = indices[start : start + 32]
            flow = torch.from_numpy(target_flow[index]).to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                normalized = model(flow)
            values.append((normalized.float() * scale_tensor + mean_tensor).cpu().numpy())
    return np.concatenate(values).reshape(len(indices), 4, 3).astype(np.float32)


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
    table = previous.load_npz(resolve(contract["inputs"]["frozen_common_table"]["path"]))
    require(
        np.array_equal(table["record_key"].astype(str), data["record_key"].astype(str)),
        "table/evaluation order mismatch",
    )
    data["trajectory_id"] = table["trajectory_id"].astype(str)
    flow_data = build_flow_data(contract, data_contract, data)
    folds = int(contract["training"]["outer_route_folds"])
    assignment, fold_audit = previous.assign_folds(data["route"], folds)
    predicted = np.full_like(data["executed_actions"], np.nan)
    naive = np.full_like(predicted, np.nan)
    scale_by_state = np.full_like(predicted, np.nan)
    fold_records: dict[str, Any] = {}
    started = time.perf_counter()
    for fold in range(folds):
        fold_dir = output / f"fold_{fold}"
        checkpoint_path = fold_dir / "flow_motion_probe.pt"
        held_path = fold_dir / "held_route_goal_prediction.npz"
        target_train = np.flatnonzero(assignment != fold)
        test_index = np.flatnonzero(assignment == fold)
        held_routes = np.unique(data["route"][test_index]).astype(str)
        stream_eligible = np.flatnonzero(
            ~np.isin(flow_data["stream_route"], held_routes)
        )
        eligible_ids = set(flow_data["stream_trajectory"][stream_eligible].astype(str))
        require(
            not set(held_routes).intersection(data["route"][target_train].astype(str)),
            f"target route leakage fold {fold}",
        )
        require(
            not set(held_routes).intersection(flow_data["stream_route"][stream_eligible].astype(str)),
            f"stream route leakage fold {fold}",
        )
        require(
            not set(data["trajectory_id"][test_index].astype(str)).intersection(eligible_ids),
            f"exact trajectory leakage fold {fold}",
        )
        set_seed(int(contract["training"]["seed"]) + fold)
        model = FlowMotionProbe().to(device)
        training_summary, mean, scale = train_fold(
            model,
            flow_data,
            stream_eligible,
            data,
            target_train,
            contract=contract,
            device=device,
            seed=int(contract["training"]["seed"]) + fold,
            label=f"outer_fold_{fold}_raw_flow",
        )
        goal_prediction = predict(
            model, flow_data["target_flow"], test_index, mean, scale, device
        )
        fold_dir.mkdir(parents=True, exist_ok=True)
        previous.write_npz(
            held_path,
            state_index=test_index.astype(np.int16),
            goal_action_prediction=goal_prediction,
        )
        temporary = checkpoint_path.with_suffix(".pt.tmp")
        torch.save(
            {
                "model": model.state_dict(),
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
        predicted[test_index] = goal_prediction
        naive[test_index] = mean.reshape(4, 3)
        scale_by_state[test_index] = scale.reshape(4, 3)
        fold_records[str(fold)] = {
            "outer_train_states": int(len(target_train)),
            "outer_test_states": int(len(test_index)),
            "held_routes": held_routes.tolist(),
            "eligible_full_stream_windows": int(len(stream_eligible)),
            "route_overlap": 0,
            "exact_trajectory_overlap": 0,
            "training": training_summary,
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "predictions_sha256": sha256_file(held_path),
        }
        del model
        torch.cuda.empty_cache()
    require(np.isfinite(predicted).all(), "OOF prediction incomplete")
    physical = sparse.physical_error_summary(
        predicted, data["executed_actions"], naive, scale_by_state, data["route"]
    )
    primary_error = full.path_score(data["candidate_actions"], predicted)
    primary_selected = previous.selection_from_error(primary_error, data["planning_pool"])
    true_error = full.path_score(data["candidate_actions"], data["executed_actions"])
    true_selected = previous.selection_from_error(true_error, data["planning_pool"])
    true_summary = previous.selection_summary(data, true_selected)
    primary_summary = previous.selection_summary(data, primary_selected)
    checks, ratio = soft.performance_checks(
        primary_summary, float(true_summary["ate_gain_m"]), contract
    )
    passed = all(checks.values())
    decision = (
        "go_freeze_raw_flow_motion_latent"
        if passed
        else "no_go_existing_raw_rgb_pose_supervision_insufficient"
    )
    previous.write_npz(
        prediction_path,
        record_key=data["record_key"],
        route_group=data["route"],
        outer_fold=assignment,
        goal_action_prediction=predicted,
        mean_action_prediction=naive,
        action_scale=scale_by_state,
        primary_path_score=primary_error,
        primary_selected_index=primary_selected.astype(np.int16),
        true_path_selected_index=true_selected.astype(np.int16),
    )
    result = {
        "schema": "ltwm_raw_flow_motion_route_oof_result_v1",
        "status": "complete_spent_development_raw_flow_probe",
        "decision": decision,
        "contract": {"path": str(contract_path), "sha256": sha256_file(contract_path)},
        "runner": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "ephemeral_flow": {
            "algorithm": "OpenCV DISOpticalFlow preset MEDIUM",
            "flow_pairs": flow_data["flow_pairs"],
            "runtime_seconds": flow_data["runtime_seconds"],
            "persistent_new_table": False,
        },
        "fold_assignment": fold_audit,
        "folds": fold_records,
        "real_future_action_observability": physical,
        "fixed_planners": {
            "primary_predicted_path": {
                **primary_summary,
                "ate_oracle_gain_ratio": ratio,
            },
            "true_path_ceiling": true_summary,
        },
        "performance_gate": {"checks": checks, "passed": passed},
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
            "outcome_labels_used_for_training_or_checkpoint_selection": False,
            "ate_or_objective_gradient": False,
            "sealed_validation_opened": False,
            "cem_mutated": False,
        },
        "runtime_seconds_excluding_flow": time.perf_counter() - started,
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
                "primary": soft.without_routes(
                    result["fixed_planners"]["primary_predicted_path"]
                ),
                "performance_gate": result["performance_gate"],
                "result": str(result_path),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
