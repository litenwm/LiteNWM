#!/usr/bin/env python3
"""Strict route-OOF platform-conditioned raw-flow motion probe."""

from __future__ import annotations

import argparse
import json
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
import run_ltwm_full_stream_action_observability_v1 as full  # noqa: E402
import run_ltwm_raw_flow_motion_probe_v1 as raw  # noqa: E402
import run_ltwm_scand_same_state_hardneg_v1 as previous  # noqa: E402
import run_ltwm_soft_correspondence_motion_probe_v1 as soft  # noqa: E402


DEFAULT_CONTRACT = ROOT / "configs/recon/ltwm_platform_conditioned_flow_probe_v1.json"
DEFAULT_OUTPUT = ROOT / "outputs/cem_development_v1/ltwm_platform_conditioned_flow_probe_v1"


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
        == "frozen_before_single_platform_conditioned_flow_route_oof_probe",
        "conditioned-flow contract is not frozen",
    )
    for section, path_key, hash_key in (
        (contract["lineage"]["unconditioned_flow"], "contract_path", "contract_sha256"),
        (contract["lineage"]["unconditioned_flow"], "runner_path", "runner_sha256"),
        (contract["lineage"]["unconditioned_flow"], "result_path", "result_sha256"),
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
    require(contract["flow_latent"]["persistent_new_table"] is False, "flow cache must be ephemeral")
    require(contract["probe"]["outcome_labels_in_training"] is False, "outcomes forbidden")
    require(contract["probe"]["ate_or_objective_gradient"] is False, "outcome gradients forbidden")
    require(contract["sealed_validation_opened"] is False, "sealed validation forbidden")
    return contract, previous.load_contract(resolve(contract["inputs"]["data_contract"]["path"]))


class ConditionedFlowProbe(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.spatial = nn.Sequential(
            nn.Conv2d(18, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 48, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(48, 32, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Flatten(),
        )
        self.head = nn.Sequential(
            nn.Linear(512 + 5, 128),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(128, 12),
        )
        axis = torch.linspace(-1.0, 1.0, 16)
        y, x = torch.meshgrid(axis, axis, indexing="ij")
        self.register_buffer("coordinate", torch.stack((x, y), dim=0)[None])

    def forward(self, flow: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        value = torch.cat(
            (
                flow.permute(0, 3, 1, 2),
                self.coordinate.expand(len(flow), -1, -1, -1),
            ),
            dim=1,
        )
        return self.head(torch.cat((self.spatial(value), condition), dim=-1))


def encode_categories(
    contract: dict[str, Any],
    flow_data: dict[str, Any],
    table: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    robots = contract["conditioning"]["robot_order"]
    cameras = contract["conditioning"]["camera_order"]
    robot_lookup = {name: index for index, name in enumerate(robots)}
    camera_lookup = {name: index for index, name in enumerate(cameras)}
    stream_robot_name = np.asarray(
        [trajectory.split("_")[3] for trajectory in flow_data["stream_trajectory"]]
    )
    target_robot_name = table["robot"].astype(str)
    target_camera_name = table["camera_contract"].astype(str)
    require(set(stream_robot_name).issubset(robot_lookup), "unknown stream robot")
    require(set(target_robot_name).issubset(robot_lookup), "unknown target robot")
    require(set(target_camera_name).issubset(camera_lookup), "unknown target camera")
    return {
        "stream_robot": np.asarray([robot_lookup[value] for value in stream_robot_name], dtype=np.int8),
        "target_robot": np.asarray([robot_lookup[value] for value in target_robot_name], dtype=np.int8),
        "stream_camera": np.zeros(len(stream_robot_name), dtype=np.int8),
        "target_camera": np.asarray([camera_lookup[value] for value in target_camera_name], dtype=np.int8),
    }


def action_statistics(
    data: dict[str, np.ndarray],
    target_robot: np.ndarray,
    train_index: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    means = np.empty((2, 12), dtype=np.float32)
    scales = np.empty((2, 12), dtype=np.float32)
    for robot in range(2):
        index = train_index[target_robot[train_index] == robot]
        require(len(index) >= 20, f"insufficient target robot states: {robot}")
        flat = data["executed_actions"][index].reshape(len(index), 12)
        means[robot] = flat.mean(axis=0)
        scales[robot] = np.maximum(flat.std(axis=0), 0.01)
    return means, scales


def condition_tensor(
    robot: np.ndarray,
    camera: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    value = np.zeros((len(robot), 5), dtype=np.float32)
    value[np.arange(len(robot)), robot] = 1.0
    value[np.arange(len(robot)), 2 + camera] = 1.0
    return torch.from_numpy(value).to(device)


def train_fold(
    model: ConditionedFlowProbe,
    flow_data: dict[str, Any],
    categories: dict[str, np.ndarray],
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
    from run_ltwm_action_observable_cycle_probe_v1 import route_uniform_batches

    stream_schedule = route_uniform_batches(
        stream_eligible,
        flow_data["stream_route"],
        steps=steps,
        batch_size=stream_batch,
        seed=seed,
    )
    target_schedule = route_uniform_batches(
        target_train,
        data["route"],
        steps=steps,
        batch_size=target_batch,
        seed=seed + 1000,
    )
    means, scales = action_statistics(data, categories["target_robot"], target_train)
    means_tensor = torch.from_numpy(means).to(device)
    scales_tensor = torch.from_numpy(scales).to(device)
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
        robot = np.concatenate(
            (
                categories["stream_robot"][stream_index],
                categories["target_robot"][target_index],
            )
        )
        camera = np.concatenate(
            (
                categories["stream_camera"][stream_index],
                categories["target_camera"][target_index],
            )
        )
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
        robot_tensor = torch.from_numpy(robot.astype(np.int64)).to(device)
        normalized_target = (
            target_action - means_tensor[robot_tensor]
        ) / scales_tensor[robot_tensor]
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            normalized = model(flow, condition_tensor(robot, camera, device))
            action_loss = F.smooth_l1_loss(normalized, normalized_target)
            predicted_action = (
                normalized * scales_tensor[robot_tensor] + means_tensor[robot_tensor]
            )
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
            "runtime_seconds": time.perf_counter() - started,
        },
        means,
        scales,
    )


def predict(
    model: ConditionedFlowProbe,
    flow_data: dict[str, Any],
    categories: dict[str, np.ndarray],
    indices: np.ndarray,
    means: np.ndarray,
    scales: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    means_tensor = torch.from_numpy(means).to(device)
    scales_tensor = torch.from_numpy(scales).to(device)
    values = []
    with torch.inference_mode():
        for start in range(0, len(indices), 32):
            index = indices[start : start + 32]
            robot = categories["target_robot"][index]
            camera = categories["target_camera"][index]
            robot_tensor = torch.from_numpy(robot.astype(np.int64)).to(device)
            flow = torch.from_numpy(flow_data["target_flow"][index]).to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                normalized = model(flow, condition_tensor(robot, camera, device))
            values.append(
                (
                    normalized.float() * scales_tensor[robot_tensor]
                    + means_tensor[robot_tensor]
                )
                .cpu()
                .numpy()
            )
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
    flow_data = raw.build_flow_data(contract, data_contract, data)
    categories = encode_categories(contract, flow_data, table)
    folds = int(contract["training"]["outer_route_folds"])
    assignment, fold_audit = previous.assign_folds(data["route"], folds)
    predicted = np.full_like(data["executed_actions"], np.nan)
    naive = np.full_like(predicted, np.nan)
    scale_by_state = np.full_like(predicted, np.nan)
    fold_records: dict[str, Any] = {}
    started = time.perf_counter()
    for fold in range(folds):
        fold_dir = output / f"fold_{fold}"
        checkpoint_path = fold_dir / "conditioned_flow_probe.pt"
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
        model = ConditionedFlowProbe().to(device)
        training_summary, means, scales = train_fold(
            model,
            flow_data,
            categories,
            stream_eligible,
            data,
            target_train,
            contract=contract,
            device=device,
            seed=int(contract["training"]["seed"]) + fold,
            label=f"outer_fold_{fold}_conditioned_flow",
        )
        goal_prediction = predict(
            model, flow_data, categories, test_index, means, scales, device
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
                "action_mean_by_robot": torch.from_numpy(means),
                "action_scale_by_robot": torch.from_numpy(scales),
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
        for state in test_index:
            robot = categories["target_robot"][state]
            naive[state] = means[robot].reshape(4, 3)
            scale_by_state[state] = scales[robot].reshape(4, 3)
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
    from run_ltwm_action_observable_cycle_probe_v1 import physical_error_summary

    physical = physical_error_summary(
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
        "go_freeze_platform_conditioned_flow_latent"
        if passed
        else "no_go_calibrated_geometry_supervision_required"
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
        "schema": "ltwm_platform_conditioned_flow_route_oof_result_v1",
        "status": "complete_spent_development_conditioned_flow_probe",
        "decision": decision,
        "contract": {"path": str(contract_path), "sha256": sha256_file(contract_path)},
        "runner": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "ephemeral_flow": {
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
