#!/usr/bin/env python3
"""Route-OOF action-aware residual adapter probe over native DINOv3 tokens."""

from __future__ import annotations

import argparse
from collections import Counter
import gc
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

from recon_latents.dinov3_adapter import (  # noqa: E402
    DinoV3AdapterConfig,
    DinoV3TokenAdapter,
    extract_dinov3_patch_tokens,
)
from recon_latents.models_v2 import integrate_se2_prefix  # noqa: E402
import encode_dinov3_adapted_resumable as encoder  # noqa: E402
import run_ltwm_action_observable_cycle_probe_v1 as sparse  # noqa: E402
import run_ltwm_full_stream_action_observability_v1 as full  # noqa: E402
import run_ltwm_scand_same_state_hardneg_v1 as previous  # noqa: E402
import run_ltwm_soft_correspondence_motion_probe_v1 as soft  # noqa: E402


DEFAULT_CONTRACT = ROOT / "configs/recon/ltwm_action_aware_adapter_probe_v1.json"
DEFAULT_OUTPUT = ROOT / "outputs/cem_development_v1/ltwm_action_aware_adapter_probe_v1"


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
        == "frozen_before_single_action_aware_adapter_route_oof_probe",
        "action-aware adapter contract is not frozen",
    )
    for section, path_key, hash_key in (
        (
            contract["lineage"]["withdrawn_decoder_hypothesis"],
            "contract_path",
            "contract_sha256",
        ),
        (
            contract["lineage"]["withdrawn_decoder_hypothesis"],
            "runner_path",
            "runner_sha256",
        ),
        (
            contract["lineage"]["withdrawn_decoder_hypothesis"],
            "result_path",
            "result_sha256",
        ),
        (contract["inputs"]["data_contract"], "path", "sha256"),
        (contract["inputs"]["frozen_common_table"], "path", "sha256"),
        (contract["inputs"]["formal_v70_result"], "path", "sha256"),
        (contract["inputs"]["frozen_adapter_checkpoint"], "path", "sha256"),
    ):
        verify_file(resolve(section[path_key]), section[hash_key])
    model_weight = (
        Path(contract["inputs"]["dinov3_model_dir"]).resolve() / "model.safetensors"
    )
    verify_file(model_weight, contract["inputs"]["dinov3_weight_sha256"])
    stream = contract["inputs"]["full_stream_latents"]
    stream_root = resolve(stream["root"])
    verify_file(stream_root / "metadata.json", stream["metadata_sha256"])
    verify_file(stream_root / "splits.json", stream["splits_sha256"])
    verify_file(stream_root / "trajectories.jsonl", stream["trajectories_sha256"])
    require(
        full.ordered_shard_aggregate(stream_root)
        == stream["ordered_shard_sha256_aggregate"],
        "full-stream aggregate mismatch",
    )
    require(contract["ephemeral_native_encoding"]["persistent_new_table"] is False, "native cache must remain ephemeral")
    require(contract["action_aware_adapter"]["outcome_labels_in_training"] is False, "outcomes forbidden")
    require(
        contract["action_aware_adapter"]["ate_or_objective_gradient_into_adapter_or_ltwm"]
        is False,
        "outcome gradients forbidden",
    )
    require(contract["sealed_validation_opened"] is False, "sealed validation forbidden")
    return contract, previous.load_contract(
        resolve(contract["inputs"]["data_contract"]["path"])
    )


def load_frozen_adapter(contract: dict[str, Any]) -> DinoV3TokenAdapter:
    payload = torch.load(
        resolve(contract["inputs"]["frozen_adapter_checkpoint"]["path"]),
        map_location="cpu",
        weights_only=False,
    )
    adapter = DinoV3TokenAdapter(DinoV3AdapterConfig(**payload["config"]))
    adapter.load_state_dict(payload["model"], strict=True)
    return adapter.eval().requires_grad_(False)


def load_backbone(contract: dict[str, Any], device: torch.device) -> nn.Module:
    from transformers import AutoModel

    return (
        AutoModel.from_pretrained(
            Path(contract["inputs"]["dinov3_model_dir"]),
            local_files_only=True,
            dtype=torch.float32,
        )
        .eval()
        .to(device)
        .requires_grad_(False)
    )


def encode_paths(
    backbone: nn.Module,
    paths: list[Path],
    device: torch.device,
    *,
    batch_size: int = 8,
) -> np.ndarray:
    result = []
    for start in range(0, len(paths), batch_size):
        image = torch.stack(
            [encoder.preprocess(path) for path in paths[start : start + batch_size]]
        ).to(device)
        with torch.inference_mode():
            hidden = backbone(pixel_values=image).last_hidden_state
            patch = extract_dinov3_patch_tokens(hidden)
        result.append(patch.cpu().numpy().astype(np.float16))
    return np.concatenate(result)


def cohort_roots(
    data_contract: dict[str, Any],
) -> dict[str, Path]:
    roots: dict[str, Path] = {}
    for cohort in data_contract["inputs"]["cohorts"]:
        with np.load(resolve(cohort["latent_path"]), allow_pickle=False) as payload:
            metadata = json.loads(str(payload["metadata"]))
        root = Path(metadata["trajectory_root"]).resolve()
        require(root.is_dir(), f"missing cohort root: {root}")
        roots[str(cohort["name"])] = root
    return roots


def encode_native_inputs(
    contract: dict[str, Any],
    data_contract: dict[str, Any],
    evaluation_data: dict[str, np.ndarray],
    device: torch.device,
) -> dict[str, Any]:
    backbone = load_backbone(contract, device)
    stream_root = resolve(contract["inputs"]["full_stream_latents"]["root"])
    metadata = json.loads((stream_root / "metadata.json").read_text(encoding="utf-8"))
    raw_root = Path(contract["inputs"]["full_stream_latents"]["raw_root"]).resolve()
    native_by_trajectory: list[np.ndarray] = []
    trajectory_ids: list[str] = []
    trajectory_routes: list[str] = []
    window_trajectory: list[np.ndarray] = []
    window_current_t: list[np.ndarray] = []
    window_actions: list[np.ndarray] = []
    window_routes: list[np.ndarray] = []
    started = time.perf_counter()
    for trajectory_index, row in enumerate(metadata["trajectories"]):
        trajectory_id = str(row["trajectory_id"])
        directory = raw_root / trajectory_id
        paths = sorted(directory.glob("*.jpg"), key=lambda value: int(value.stem))
        require(len(paths) == int(row["num_frames"]), f"stream image count: {trajectory_id}")
        native = encode_paths(backbone, paths, device)
        require(native.shape == (len(paths), 196, 1024), f"native shape: {trajectory_id}")
        native_by_trajectory.append(native)
        trajectory_ids.append(trajectory_id)
        route = full.route_group(trajectory_id)
        trajectory_routes.append(route)
        with np.load(stream_root / "shards" / str(row["shard_file"]), allow_pickle=False) as shard:
            action = shard["actions_continuous"].astype(np.float32)
        current_t = np.arange(1, len(paths) - 4, dtype=np.int32)
        if len(current_t):
            window_trajectory.append(
                np.full(len(current_t), trajectory_index, dtype=np.int16)
            )
            window_current_t.append(current_t)
            window_actions.append(
                np.stack([action[t : t + 4] for t in current_t]).astype(np.float32)
            )
            window_routes.append(np.full(len(current_t), route))
        if (trajectory_index + 1) % 10 == 0 or trajectory_index + 1 == len(metadata["trajectories"]):
            print(
                previous.canonical_json(
                    {
                        "stage": "ephemeral_native_stream_encoding",
                        "trajectories": trajectory_index + 1,
                        "total": len(metadata["trajectories"]),
                        "frames": int(sum(len(value) for value in native_by_trajectory)),
                        "elapsed_seconds": time.perf_counter() - started,
                    }
                ),
                flush=True,
            )
    roots = cohort_roots(data_contract)
    target_paths: list[Path] = []
    target_path_rows: list[list[Path]] = []
    for index in range(len(evaluation_data["record_key"])):
        key = str(evaluation_data["record_key"][index])
        trajectory_id, current_text = key.rsplit(":", 1)
        current_t = int(current_text)
        root = roots[str(evaluation_data["cohort"][index])]
        row = [
            root / trajectory_id / f"{frame}.jpg"
            for frame in (current_t - 1, current_t, current_t + 4)
        ]
        require(all(path.is_file() for path in row), f"missing target image: {key}")
        target_path_rows.append(row)
        target_paths.extend(row)
    unique_target_paths = sorted(set(target_paths), key=lambda value: str(value))
    print(
        previous.canonical_json(
            {
                "stage": "ephemeral_native_target_encoding",
                "unique_images": len(unique_target_paths),
                "state_frame_references": len(target_paths),
            }
        ),
        flush=True,
    )
    target_tokens = encode_paths(backbone, unique_target_paths, device)
    target_lookup = {path: index for index, path in enumerate(unique_target_paths)}
    target_native_index = np.asarray(
        [[target_lookup[path] for path in row] for row in target_path_rows],
        dtype=np.int32,
    )
    del backbone
    torch.cuda.empty_cache()
    gc.collect()
    stream = {
        "native": native_by_trajectory,
        "trajectory_ids": np.asarray(trajectory_ids),
        "trajectory_routes": np.asarray(trajectory_routes),
        "window_trajectory": np.concatenate(window_trajectory),
        "window_current_t": np.concatenate(window_current_t),
        "window_actions": np.concatenate(window_actions),
        "window_route": np.concatenate(window_routes).astype(str),
    }
    require(len(stream["window_actions"]) == 41231, "unexpected stream windows")
    return {
        "stream": stream,
        "target_tokens": target_tokens,
        "target_native_index": target_native_index,
        "unique_target_images": len(unique_target_paths),
    }


def gather_stream_native(
    stream: dict[str, Any],
    indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count = len(indices)
    previous_native = np.empty((count, 196, 1024), dtype=np.float16)
    current_native = np.empty_like(previous_native)
    future_native = np.empty_like(previous_native)
    trajectory = stream["window_trajectory"][indices]
    current_t = stream["window_current_t"][indices]
    for position, (trajectory_index, t) in enumerate(zip(trajectory, current_t)):
        native = stream["native"][int(trajectory_index)]
        previous_native[position] = native[int(t) - 1]
        current_native[position] = native[int(t)]
        future_native[position] = native[int(t) + 4]
    return previous_native, current_native, future_native


class DifferentiableCorrespondenceInverse(soft.SoftCorrespondenceInverse):
    def correspondence(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
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
        return torch.cat((flow, peak, entropy), dim=-1)


class ActionAwareAdapterProbe(nn.Module):
    def __init__(
        self,
        reference_adapter: DinoV3TokenAdapter,
        *,
        temperature: float = 0.07,
        spatial_sigma: float = 4.0,
    ) -> None:
        super().__init__()
        self.reference_adapter = reference_adapter.eval().requires_grad_(False)
        self.motion_norm = nn.LayerNorm(1024)
        self.motion_down = nn.Linear(1024, 32)
        self.motion_up = nn.Linear(32, 384)
        nn.init.zeros_(self.motion_up.weight)
        nn.init.zeros_(self.motion_up.bias)
        self.inverse = DifferentiableCorrespondenceInverse(
            temperature=temperature,
            spatial_sigma=spatial_sigma,
        )

    def adapt(self, native: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            reference = self.reference_adapter(native)
        delta = self.motion_up(F.gelu(self.motion_down(self.motion_norm(native))))
        delta = F.interpolate(
            delta.permute(0, 3, 1, 2).reshape(-1, 384, 14, 14),
            size=(16, 16),
            mode="bicubic",
            align_corners=False,
        ).permute(0, 2, 3, 1)
        return reference + delta, reference

    def forward(
        self,
        previous_native: torch.Tensor,
        current_native: torch.Tensor,
        future_native: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        count = len(previous_native)
        native = torch.cat(
            (previous_native, current_native, future_native), dim=0
        ).reshape(-1, 14, 14, 1024)
        adapted, reference = self.adapt(native)
        previous, current, future = adapted.split(count, dim=0)
        prediction = self.inverse(
            previous.reshape(count, 256, 384),
            current.reshape(count, 256, 384),
            future.reshape(count, 256, 384),
        )
        cosine = F.cosine_similarity(adapted, reference, dim=-1).mean(dim=(1, 2))
        relative_mse = (adapted - reference).square().mean(dim=(1, 2, 3)) / (
            reference.square().mean(dim=(1, 2, 3)) + 1.0e-8
        )
        cosine = cosine.reshape(3, count).mean(dim=0)
        relative_mse = relative_mse.reshape(3, count).mean(dim=0)
        return prediction, cosine, relative_mse


def build_model(
    contract: dict[str, Any],
    device: torch.device,
) -> ActionAwareAdapterProbe:
    model = ActionAwareAdapterProbe(load_frozen_adapter(contract))
    return model.to(device)


def target_statistics(
    evaluation_data: dict[str, np.ndarray],
    indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    flat = evaluation_data["executed_actions"][indices].reshape(len(indices), 12)
    return (
        flat.mean(axis=0).astype(np.float32),
        np.maximum(flat.std(axis=0), 0.01).astype(np.float32),
    )


def train_fold(
    model: ActionAwareAdapterProbe,
    native_data: dict[str, Any],
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
    loss_contract = contract["loss"]
    composition = training["batch_composition"]
    stream_batch = int(composition["route-excluded full-stream windows"])
    target_batch = int(composition["outer-train frozen-table states"])
    steps = int(training["fixed_steps_per_fold"])
    stream = native_data["stream"]
    stream_schedule = sparse.route_uniform_batches(
        stream_eligible,
        stream["window_route"],
        steps=steps,
        batch_size=stream_batch,
        seed=seed,
    )
    target_schedule = sparse.route_uniform_batches(
        target_train,
        evaluation_data["route"],
        steps=steps,
        batch_size=target_batch,
        seed=seed + 1000,
    )
    mean, scale = target_statistics(evaluation_data, target_train)
    mean_tensor = torch.from_numpy(mean).to(device)
    scale_tensor = torch.from_numpy(scale).to(device)
    parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    model.train()
    model.reference_adapter.eval()
    rows = []
    started = time.perf_counter()
    for step, (stream_index, target_index) in enumerate(
        zip(stream_schedule, target_schedule), start=1
    ):
        stream_native = gather_stream_native(stream, stream_index)
        target_native = native_data["target_tokens"][
            native_data["target_native_index"][target_index]
        ]
        inputs = []
        for field in range(3):
            inputs.append(
                torch.from_numpy(
                    np.concatenate((stream_native[field], target_native[:, field]))
                ).to(device)
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
            prediction, cosine, relative_mse = model(*inputs)
            action_loss = F.smooth_l1_loss(prediction, normalized_target)
            predicted_action = prediction * scale_tensor + mean_tensor
            predicted_path = integrate_se2_prefix(
                predicted_action.reshape(-1, 4, 3)
            )[..., :2]
            target_path = integrate_se2_prefix(
                target_action.reshape(-1, 4, 3)
            )[..., :2]
            path_loss = F.smooth_l1_loss(predicted_path, target_path)
            semantic_cosine_loss = 1.0 - cosine.mean()
            semantic_relative_mse = relative_mse.mean()
            loss = (
                float(loss_contract["standardized_action_smooth_l1_weight"])
                * action_loss
                + float(loss_contract["integrated_xy_path_smooth_l1_weight"])
                * path_loss
                + float(loss_contract["semantic_cosine_distance_weight"])
                * semantic_cosine_loss
                + float(loss_contract["semantic_relative_mse_weight"])
                * semantic_relative_mse
            )
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            parameters, float(training["gradient_clip"])
        )
        optimizer.step()
        rows.append(
            (
                float(loss.detach()),
                float(action_loss.detach()),
                float(path_loss.detach()),
                float(cosine.mean().detach()),
                float(relative_mse.mean().detach()),
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
                        "semantic_cosine": float(recent[:, 3].mean()),
                        "semantic_relative_mse": float(recent[:, 4].mean()),
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
            "stream_examples_per_batch": stream_batch,
            "target_examples_per_batch": target_batch,
            "eligible_stream_windows": int(len(stream_eligible)),
            "outer_train_states": int(len(target_train)),
            "mean_loss": float(values[:, 0].mean()),
            "last_100_loss": float(values[-100:, 0].mean()),
            "mean_action_loss": float(values[:, 1].mean()),
            "mean_path_loss": float(values[:, 2].mean()),
            "mean_semantic_cosine": float(values[:, 3].mean()),
            "mean_semantic_relative_mse": float(values[:, 4].mean()),
            "mean_gradient_norm": float(values[:, 5].mean()),
            "trainable_parameters": int(
                sum(parameter.numel() for parameter in parameters)
            ),
            "runtime_seconds": time.perf_counter() - started,
        },
        mean,
        scale,
    )


def predict(
    model: ActionAwareAdapterProbe,
    native_data: dict[str, Any],
    indices: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    mean_tensor = torch.from_numpy(mean).to(device)
    scale_tensor = torch.from_numpy(scale).to(device)
    predictions = []
    cosines = []
    relative_mses = []
    with torch.inference_mode():
        for start in range(0, len(indices), 8):
            index = indices[start : start + 8]
            native = native_data["target_tokens"][
                native_data["target_native_index"][index]
            ]
            inputs = [
                torch.from_numpy(native[:, field]).to(device) for field in range(3)
            ]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                normalized, cosine, relative_mse = model(*inputs)
            predictions.append(
                (normalized.float() * scale_tensor + mean_tensor).cpu().numpy()
            )
            cosines.append(cosine.float().cpu().numpy())
            relative_mses.append(relative_mse.float().cpu().numpy())
    return (
        np.concatenate(predictions).reshape(len(indices), 4, 3).astype(np.float32),
        np.concatenate(cosines).astype(np.float32),
        np.concatenate(relative_mses).astype(np.float32),
    )


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
    native_data = encode_native_inputs(
        contract, data_contract, evaluation_data, device
    )
    stream = native_data["stream"]
    folds = int(contract["training"]["outer_route_folds"])
    assignment, fold_audit = previous.assign_folds(evaluation_data["route"], folds)
    predicted = np.full_like(evaluation_data["executed_actions"], np.nan)
    naive = np.full_like(predicted, np.nan)
    scale_by_state = np.full_like(predicted, np.nan)
    semantic_cosine = np.full(800, np.nan, dtype=np.float32)
    semantic_relative_mse = np.full(800, np.nan, dtype=np.float32)
    fold_records: dict[str, Any] = {}
    started = time.perf_counter()
    for fold in range(folds):
        fold_dir = output / f"fold_{fold}"
        checkpoint_path = fold_dir / "action_aware_adapter_probe.pt"
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
                native_data,
                stream_eligible,
                evaluation_data,
                target_train,
                contract=contract,
                device=device,
                seed=int(contract["training"]["seed"]) + fold,
                label=f"outer_fold_{fold}_action_aware_adapter",
            )
            goal_prediction, held_cosine, held_relative_mse = predict(
                model, native_data, test_index, mean, scale, device
            )
            fold_dir.mkdir(parents=True, exist_ok=True)
            previous.write_npz(
                held_prediction_path,
                state_index=test_index.astype(np.int16),
                goal_action_prediction=goal_prediction,
                semantic_cosine=held_cosine,
                semantic_relative_mse=held_relative_mse,
            )
            temporary = checkpoint_path.with_suffix(".pt.tmp")
            torch.save(
                {
                    "model": model.state_dict(),
                    "architecture": {
                        "name": "ActionAwareAdapterProbe",
                        "trainable_parameters": training_summary["trainable_parameters"],
                        "motion_rank": 32,
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
        require(
            np.array_equal(held_prediction["state_index"], test_index.astype(np.int16)),
            f"held prediction mismatch fold {fold}",
        )
        predicted[test_index] = held_prediction["goal_action_prediction"]
        naive[test_index] = mean.reshape(4, 3)
        scale_by_state[test_index] = scale.reshape(4, 3)
        semantic_cosine[test_index] = held_prediction["semantic_cosine"]
        semantic_relative_mse[test_index] = held_prediction["semantic_relative_mse"]
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
    require(np.isfinite(semantic_cosine).all(), "semantic cosine incomplete")
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
    performance_check, ratio = soft.performance_checks(
        primary_summary, float(true_path_summary["ate_gain_m"]), contract
    )
    semantic_guard = contract["held_route_semantic_guard"]
    semantic_check = {
        "mean_cosine_similarity": float(semantic_cosine.mean())
        >= float(semantic_guard["mean_cosine_similarity_min"]),
        "relative_mse": float(semantic_relative_mse.mean())
        <= float(semantic_guard["relative_mse_max"]),
    }
    passed = all(performance_check.values()) and all(semantic_check.values())
    decision = (
        "go_single_matching_ltwm_path_cycle_training"
        if passed
        else "no_go_withdraw_existing_data_action_aware_adapter_hypothesis"
    )
    previous.write_npz(
        prediction_path,
        record_key=evaluation_data["record_key"],
        route_group=evaluation_data["route"],
        outer_fold=assignment,
        goal_action_prediction=predicted,
        mean_action_prediction=naive,
        action_scale=scale_by_state,
        semantic_cosine=semantic_cosine,
        semantic_relative_mse=semantic_relative_mse,
        primary_path_score=primary_error,
        primary_selected_index=primary_selected.astype(np.int16),
        true_path_selected_index=true_path_selected.astype(np.int16),
    )
    result = {
        "schema": "ltwm_action_aware_adapter_route_oof_result_v1",
        "status": "complete_spent_development_adapter_probe",
        "decision": decision,
        "contract": {
            "path": str(contract_path),
            "sha256": sha256_file(contract_path),
        },
        "runner": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "ephemeral_native_encoding": {
            "stream_frames": int(sum(len(value) for value in stream["native"])),
            "stream_trajectories": int(len(stream["native"])),
            "target_unique_images": int(native_data["unique_target_images"]),
            "persistent_new_table": False,
        },
        "fold_assignment": fold_audit,
        "folds": fold_records,
        "held_route_semantic_preservation": {
            "mean_cosine_similarity": float(semantic_cosine.mean()),
            "mean_relative_mse": float(semantic_relative_mse.mean()),
            "checks": semantic_check,
            "passed": all(semantic_check.values()),
        },
        "real_future_action_observability": physical,
        "fixed_planners": {
            "primary_predicted_path": {
                **primary_summary,
                "ate_oracle_gain_ratio": ratio,
            },
            "true_path_ceiling": true_path_summary,
        },
        "performance_gate": {
            "checks": performance_check,
            "passed": all(performance_check.values()),
        },
        "combined_gate_passed": passed,
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
            "ate_or_objective_gradient_into_adapter_or_ltwm": False,
            "sealed_validation_opened": False,
            "cem_mutated": False,
        },
        "runtime_seconds_excluding_native_encoding": time.perf_counter() - started,
    }
    require(result["protection"]["formal_v70_unchanged"], "formal-v70 mutated")
    require(result["protection"]["frozen_table_unchanged"], "frozen table mutated")
    previous.write_json(result_path, result)
    print(
        previous.canonical_json(
            {
                "status": result["status"],
                "decision": decision,
                "semantic": result["held_route_semantic_preservation"],
                "physical": {
                    key: value for key, value in physical.items() if key != "route"
                },
                "primary": soft.without_routes(
                    result["fixed_planners"]["primary_predicted_path"]
                ),
                "performance_gate": result["performance_gate"],
                "combined_gate_passed": passed,
                "result": str(result_path),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
