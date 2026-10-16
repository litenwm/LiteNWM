#!/usr/bin/env python3
"""Strict route-OOF action observability from the complete SCAND behavior stream."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import sys
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(Path(__file__).resolve().parent)]

from planning.nomad_action_adapter import integrate_local_actions  # noqa: E402
from recon_latents.models_v2 import integrate_se2_prefix  # noqa: E402
import run_ltwm_action_observable_cycle_probe_v1 as sparse  # noqa: E402
import run_ltwm_scand_same_state_hardneg_v1 as previous  # noqa: E402


DEFAULT_CONTRACT = ROOT / "configs/recon/ltwm_full_stream_action_observability_v1.json"
DEFAULT_OUTPUT = ROOT / "outputs/cem_development_v1/ltwm_full_stream_action_observability_v1"


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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def route_group(trajectory_id: str) -> str:
    fields = trajectory_id.split("_")
    require(
        len(fields) >= 7 and fields[:2] == ["random", "mdps"],
        f"unexpected SCAND trajectory id: {trajectory_id}",
    )
    return "__".join(sorted((fields[4].lower(), fields[5].lower())))


def verify_file(path: Path, expected: str) -> None:
    require(path.is_file(), f"missing input: {path}")
    require(sha256_file(path) == expected, f"input hash mismatch: {path}")


def ordered_shard_aggregate(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted((root / "shards").glob("*.npz")):
        relative = path.relative_to(ROOT)
        digest.update(f"{sha256_file(path)}  {relative.as_posix()}\n".encode())
    return digest.hexdigest()


def load_contract(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    contract = json.loads(path.read_text(encoding="utf-8"))
    require(
        contract.get("status") == "frozen_before_single_full_stream_route_oof_probe",
        "full-stream contract is not frozen",
    )
    for section, path_key, hash_key in (
        (contract["lineage"]["sparse_probe"], "contract_path", "contract_sha256"),
        (contract["lineage"]["sparse_probe"], "runner_path", "runner_sha256"),
        (contract["lineage"]["sparse_probe"], "result_path", "result_sha256"),
        (contract["inputs"]["data_contract"], "path", "sha256"),
        (contract["inputs"]["frozen_common_table"], "path", "sha256"),
        (contract["inputs"]["formal_v70_result"], "path", "sha256"),
        (contract["inputs"]["warm_checkpoint"], "path", "sha256"),
        (contract["inputs"]["encoder"], "path", "sha256"),
    ):
        verify_file(resolve(section[path_key]), section[hash_key])
    stream = contract["inputs"]["full_stream_latents"]
    stream_root = resolve(stream["root"])
    verify_file(stream_root / "metadata.json", stream["metadata_sha256"])
    verify_file(stream_root / "splits.json", stream["splits_sha256"])
    verify_file(stream_root / "trajectories.jsonl", stream["trajectories_sha256"])
    require(
        ordered_shard_aggregate(stream_root)
        == stream["ordered_shard_sha256_aggregate"],
        "full-stream shard aggregate hash mismatch",
    )
    require(contract["inverse_probe"]["outcome_labels_in_training"] is False, "outcomes forbidden")
    require(contract["inverse_probe"]["ate_or_objective_gradient"] is False, "outcome gradients forbidden")
    require(contract["sealed_validation_opened"] is False, "sealed validation forbidden")
    data_contract = previous.load_contract(resolve(contract["inputs"]["data_contract"]["path"]))
    return contract, data_contract


def load_stream(
    contract: dict[str, Any],
    evaluation_data: dict[str, np.ndarray],
) -> dict[str, Any]:
    stream_root = resolve(contract["inputs"]["full_stream_latents"]["root"])
    metadata = json.loads((stream_root / "metadata.json").read_text(encoding="utf-8"))
    require(metadata["num_trajectories"] == 234, "unexpected full-stream trajectory count")
    require(metadata["num_frames"] == 42401, "unexpected full-stream frame count")
    require(metadata["included_splits"] == ["train", "val"], "unexpected included splits")
    latent_by_trajectory: list[np.ndarray] = []
    trajectory_ids: list[str] = []
    trajectory_routes: list[str] = []
    window_trajectory: list[np.ndarray] = []
    window_current_t: list[np.ndarray] = []
    window_actions: list[np.ndarray] = []
    window_routes: list[np.ndarray] = []
    for trajectory_index, row in enumerate(metadata["trajectories"]):
        trajectory_id = str(row["trajectory_id"])
        path = stream_root / "shards" / str(row["shard_file"])
        with np.load(path, allow_pickle=False) as shard:
            require(str(shard["trajectory_id"].item()) == trajectory_id, f"trajectory mismatch: {path}")
            latent = shard["patch_latents"].astype(np.float16, copy=True)
            action = shard["actions_continuous"].astype(np.float32, copy=True)
        require(latent.shape == (int(row["num_frames"]), 16, 16, 384), f"latent shape: {path}")
        require(action.shape == (len(latent) - 1, 3), f"action shape: {path}")
        latent_by_trajectory.append(latent)
        trajectory_ids.append(trajectory_id)
        group = route_group(trajectory_id)
        trajectory_routes.append(group)
        current_t = np.arange(1, len(latent) - 4, dtype=np.int32)
        if len(current_t) == 0:
            continue
        actions = np.stack([action[t : t + 4] for t in current_t]).astype(np.float32)
        window_trajectory.append(
            np.full(len(current_t), trajectory_index, dtype=np.int16)
        )
        window_current_t.append(current_t)
        window_actions.append(actions)
        window_routes.append(np.full(len(current_t), group, dtype=f"<U{max(1, len(group))}"))
    result = {
        "latents": latent_by_trajectory,
        "trajectory_ids": np.asarray(trajectory_ids),
        "trajectory_routes": np.asarray(trajectory_routes),
        "window_trajectory": np.concatenate(window_trajectory),
        "window_current_t": np.concatenate(window_current_t),
        "window_actions": np.concatenate(window_actions),
        "window_route": np.concatenate(window_routes).astype(str),
    }
    require(len(result["window_actions"]) == 41231, "unexpected valid-window count")
    known_pairs = {
        (str(route), str(trajectory))
        for route, trajectory in zip(
            evaluation_data["route"], evaluation_data["trajectory_id"]
        )
    }
    require(
        all(route_group(trajectory) == route for route, trajectory in known_pairs),
        "route parser disagrees with frozen evaluation mapping",
    )
    return result


def target_statistics(
    actions: np.ndarray,
    indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    flat = actions[indices].reshape(len(indices), 12).astype(np.float32)
    mean = flat.mean(axis=0).astype(np.float32)
    scale = np.maximum(flat.std(axis=0), 0.01).astype(np.float32)
    return mean, scale


def gather_latents(
    stream: dict[str, Any],
    indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count = len(indices)
    previous_latent = np.empty((count, 256, 384), dtype=np.float16)
    current_latent = np.empty_like(previous_latent)
    future_latent = np.empty_like(previous_latent)
    trajectory = stream["window_trajectory"][indices]
    current_t = stream["window_current_t"][indices]
    for position, (trajectory_index, t) in enumerate(zip(trajectory, current_t)):
        latent = stream["latents"][int(trajectory_index)]
        previous_latent[position] = latent[int(t) - 1].reshape(256, 384)
        current_latent[position] = latent[int(t)].reshape(256, 384)
        future_latent[position] = latent[int(t) + 4].reshape(256, 384)
    return previous_latent, current_latent, future_latent


def train_fold(
    model: sparse.ActionObservableInverse,
    stream: dict[str, Any],
    eligible: np.ndarray,
    *,
    contract: dict[str, Any],
    device: torch.device,
    seed: int,
    label: str,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    training = contract["training"]
    mean, scale = target_statistics(stream["window_actions"], eligible)
    mean_tensor = torch.from_numpy(mean).to(device)
    scale_tensor = torch.from_numpy(scale).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    schedule = sparse.route_uniform_batches(
        eligible,
        stream["window_route"],
        steps=int(training["fixed_steps_per_fold"]),
        batch_size=int(training["batch_size"]),
        seed=seed,
    )
    model.train()
    rows = []
    started = time.perf_counter()
    for step, index in enumerate(schedule, start=1):
        previous_latent, current_latent, future_latent = gather_latents(stream, index)
        target_action = torch.from_numpy(
            stream["window_actions"][index].reshape(len(index), 12)
        ).to(device)
        normalized_target = (target_action - mean_tensor) / scale_tensor
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            prediction = model(
                torch.from_numpy(previous_latent).to(device),
                torch.from_numpy(current_latent).to(device),
                torch.from_numpy(future_latent).to(device),
            )
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
        if step % 100 == 0 or step == len(schedule):
            recent = np.asarray(rows[-100:], dtype=np.float64)
            print(
                previous.canonical_json(
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
    values = np.asarray(rows, dtype=np.float64)
    route_counts = Counter(stream["window_route"][schedule.reshape(-1)].tolist())
    return (
        {
            "steps": int(len(schedule)),
            "batch_size": int(schedule.shape[1]),
            "eligible_windows": int(len(eligible)),
            "eligible_routes": int(len(np.unique(stream["window_route"][eligible]))),
            "route_sample_counts": dict(sorted(route_counts.items())),
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


def path_score(
    candidate_actions: np.ndarray,
    predicted_actions: np.ndarray,
) -> np.ndarray:
    candidate_path = integrate_local_actions(
        candidate_actions.reshape(-1, 4, 3)
    ).reshape(len(candidate_actions), candidate_actions.shape[1], 4, 2)
    predicted_path = integrate_local_actions(predicted_actions)
    return np.mean(
        np.sum((candidate_path - predicted_path[:, None]) ** 2, axis=-1),
        axis=-1,
    ).astype(np.float32)


def performance_checks(
    summary: dict[str, Any],
    ate_oracle_gain: float,
    contract: dict[str, Any],
) -> tuple[dict[str, bool], float]:
    gate = contract["performance_gate"]
    ate_ratio = float(summary["ate_gain_m"] / ate_oracle_gain)
    checks = {
        "ate_gain": summary["ate_gain_m"] > float(gate["ate_gain_m_min_exclusive"]),
        "objective_gain": summary["objective_gain_m"]
        > float(gate["objective_gain_m_min_exclusive"]),
        "harm_rate": summary["harm_rate"] <= float(gate["harm_rate_max"]),
        "positive_route_fraction": summary["positive_route_fraction"]
        >= float(gate["positive_route_fraction_min"]),
        "ate_oracle_gain_ratio": ate_ratio >= float(gate["ate_oracle_gain_ratio_min"]),
    }
    return checks, ate_ratio


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
        "frozen-table/evaluation row order mismatch",
    )
    evaluation_data["trajectory_id"] = frozen_table["trajectory_id"].astype(str)
    stream = load_stream(contract, evaluation_data)
    folds = int(contract["training"]["outer_route_folds"])
    assignment, fold_audit = previous.assign_folds(evaluation_data["route"], folds)
    target = evaluation_data["executed_actions"].astype(np.float32)
    predicted = np.full_like(target, np.nan)
    naive = np.full_like(target, np.nan)
    scale_by_state = np.full_like(target, np.nan)
    fold_records: dict[str, Any] = {}
    started = time.perf_counter()
    for fold in range(folds):
        fold_dir = output / f"fold_{fold}"
        checkpoint_path = fold_dir / "inverse_probe.pt"
        prediction_fold_path = fold_dir / "held_route_goal_prediction.npz"
        test_index = np.flatnonzero(assignment == fold)
        held_routes = np.unique(evaluation_data["route"][test_index]).astype(str)
        eligible = np.flatnonzero(
            ~np.isin(stream["window_route"], held_routes)
        )
        eligible_trajectory = np.unique(stream["window_trajectory"][eligible])
        held_exact = set(evaluation_data["trajectory_id"][test_index].astype(str))
        eligible_ids = {
            str(stream["trajectory_ids"][index]) for index in eligible_trajectory
        }
        require(
            not set(held_routes).intersection(stream["window_route"][eligible]),
            f"route leakage in fold {fold}",
        )
        require(not held_exact.intersection(eligible_ids), f"exact trajectory leakage in fold {fold}")
        if checkpoint_path.is_file() and prediction_fold_path.is_file():
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            fold_prediction = previous.load_npz(prediction_fold_path)
            require(
                np.array_equal(fold_prediction["state_index"], test_index.astype(np.int16)),
                f"resume mismatch fold {fold}",
            )
            model = sparse.ActionObservableInverse().to(device)
            model.load_state_dict(checkpoint["model"], strict=True)
            training_summary = checkpoint["training"]
            mean = checkpoint["action_mean"].numpy()
            scale = checkpoint["action_scale"].numpy()
            status = "resumed"
        else:
            require(
                not checkpoint_path.exists() and not prediction_fold_path.exists(),
                f"partial fold artifact requires inspection: {fold_dir}",
            )
            set_seed(int(contract["training"]["seed"]) + fold)
            model = sparse.ActionObservableInverse().to(device)
            training_summary, mean, scale = train_fold(
                model,
                stream,
                eligible,
                contract=contract,
                device=device,
                seed=int(contract["training"]["seed"]) + fold,
                label=f"outer_fold_{fold}_full_stream_inverse",
            )
            goal_prediction = sparse.predict_real_future(
                model,
                evaluation_data,
                test_index,
                mean,
                scale,
                device,
            )
            fold_dir.mkdir(parents=True, exist_ok=True)
            previous.write_npz(
                prediction_fold_path,
                state_index=test_index.astype(np.int16),
                goal_action_prediction=goal_prediction,
            )
            temporary = checkpoint_path.with_suffix(".pt.tmp")
            torch.save(
                {
                    "model": model.state_dict(),
                    "architecture": sparse.asdict_dummy(),
                    "action_mean": torch.from_numpy(mean),
                    "action_scale": torch.from_numpy(scale),
                    "training": training_summary,
                    "outer_fold": fold,
                    "held_routes": held_routes.tolist(),
                    "eligible_windows": int(len(eligible)),
                    "eligible_trajectories": int(len(eligible_trajectory)),
                    "route_overlap": 0,
                    "exact_trajectory_overlap": 0,
                    "contract_sha256": sha256_file(contract_path),
                    "outcome_labels_used": False,
                },
                temporary,
            )
            temporary.replace(checkpoint_path)
            fold_prediction = previous.load_npz(prediction_fold_path)
            status = "trained"
        predicted[test_index] = fold_prediction["goal_action_prediction"]
        naive[test_index] = mean.reshape(4, 3)
        scale_by_state[test_index] = scale.reshape(4, 3)
        fold_records[str(fold)] = {
            "status": status,
            "outer_test_states": int(len(test_index)),
            "outer_test_routes": int(len(held_routes)),
            "held_routes": held_routes.tolist(),
            "eligible_full_stream_windows": int(len(eligible)),
            "eligible_full_stream_routes": int(
                len(np.unique(stream["window_route"][eligible]))
            ),
            "eligible_full_stream_trajectories": int(len(eligible_trajectory)),
            "route_overlap": 0,
            "exact_trajectory_overlap": 0,
            "training": training_summary,
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "predictions_sha256": sha256_file(prediction_fold_path),
        }
        del model
        torch.cuda.empty_cache()
    require(np.isfinite(predicted).all(), "OOF prediction incomplete")
    physical = sparse.physical_error_summary(
        predicted,
        target,
        naive,
        scale_by_state,
        evaluation_data["route"],
    )
    primary_error = path_score(evaluation_data["candidate_actions"], predicted)
    primary_selected = previous.selection_from_error(
        primary_error, evaluation_data["planning_pool"]
    )
    true_path_error = path_score(evaluation_data["candidate_actions"], target)
    true_path_selected = previous.selection_from_error(
        true_path_error, evaluation_data["planning_pool"]
    )
    true_path_summary = previous.selection_summary(evaluation_data, true_path_selected)
    ate_oracle_gain = float(true_path_summary["ate_gain_m"])
    primary_summary = previous.selection_summary(evaluation_data, primary_selected)
    action_error = sparse.score_candidates(
        evaluation_data["candidate_actions"], predicted, scale_by_state.reshape(800, 12)
    )
    action_selected = previous.selection_from_error(
        action_error, evaluation_data["planning_pool"]
    )
    action_summary = previous.selection_summary(evaluation_data, action_selected)
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
    performance_check, ate_ratio = performance_checks(
        primary_summary, ate_oracle_gain, contract
    )
    observability_pass = all(observability_checks.values())
    performance_pass = all(performance_check.values())
    if observability_pass and performance_pass:
        decision = "go_decode_frozen_warm_ltwm_once"
    else:
        decision = "no_go_withdraw_full_stream_sample_shortage_hypothesis"
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
        secondary_action_score=action_error,
        secondary_action_selected_index=action_selected.astype(np.int16),
        true_path_selected_index=true_path_selected.astype(np.int16),
    )
    result = {
        "schema": "ltwm_full_stream_action_observability_route_oof_result_v1",
        "status": "complete_spent_development_full_stream_probe",
        "decision": decision,
        "contract": {
            "path": str(contract_path),
            "sha256": sha256_file(contract_path),
        },
        "runner": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "stream": {
            "trajectories": int(len(stream["trajectory_ids"])),
            "valid_windows": int(len(stream["window_actions"])),
            "route_groups": int(len(np.unique(stream["window_route"]))),
            "route_group_overlap_with_evaluation": int(
                len(set(stream["window_route"]).intersection(evaluation_data["route"]))
            ),
        },
        "fold_assignment": fold_audit,
        "folds": fold_records,
        "real_future_action_observability": physical,
        "fixed_planners": {
            "primary_predicted_path": {
                **primary_summary,
                "ate_oracle_gain_ratio": ate_ratio,
            },
            "secondary_standardized_action": action_summary,
            "true_path_ceiling": true_path_summary,
        },
        "gates": {
            "observability": {
                "checks": observability_checks,
                "passed": observability_pass,
            },
            "primary_performance": {
                "checks": performance_check,
                "passed": performance_pass,
            },
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
                "gates": result["gates"],
                "result": str(result_path),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
