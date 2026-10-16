#!/usr/bin/env python3
"""Observed-hard-negative listwise fine-tuning for the dual-latent LTWM.

This is a deliberately narrow offline experiment.  It fine-tunes the existing
conditional predictor against real future DINO latents.  The true trajectory
and nearby observed trajectories share one listwise prediction set; no ATE,
navigation objective, or sealed validation data is loaded.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
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
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [
    str(ROOT),
    str(ROOT / "src"),
    str(Path(__file__).resolve().parent),
]

from latent_window_utils import ShardAwareBatchSampler  # noqa: E402
from recon_latents.dataset_highres_dual_v1 import (  # noqa: E402
    HighResDualLatentWindowDataset,
)
from recon_latents.losses_highres_dual_action_residual_v1 import (  # noqa: E402
    per_candidate_dual_nmse,
)
from recon_latents.models_highres_dual_v1 import (  # noqa: E402
    HighResDualLatentWMConfig,
    HighResMultiLayerDualLatentWM,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/recon/highres_dual_observed_listwise_v1.json",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(
        jsonable(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(canonical_json(value) + "\n", encoding="utf-8")
    temporary.replace(path)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def amp_context(name: str):
    if name == "off":
        return nullcontext()
    dtype = torch.bfloat16 if name == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def load_model(
    path: Path,
    device: torch.device,
) -> tuple[HighResMultiLayerDualLatentWM, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    require(payload.get("outcome_gradients") is False, "base used outcome gradients")
    require(payload.get("sealed_test_opened") is False, "base opened sealed test")
    model = HighResMultiLayerDualLatentWM(
        HighResDualLatentWMConfig(**payload["config"])
    )
    model.load_state_dict(payload["model"], strict=True)
    return model.to(device), payload


def freeze_nonconditional_state(model: HighResMultiLayerDualLatentWM) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    for module in (
        model.semantic_history,
        model.geometry_history,
        model.action_decode,
        model.semantic_logvar,
        model.geometry_logvar,
    ):
        for parameter in module.parameters():
            parameter.requires_grad_(False)


def make_loader(
    dataset: HighResDualLatentWindowDataset,
    *,
    batch_size: int,
    seed: int,
    training: bool,
) -> DataLoader:
    if training:
        return DataLoader(
            dataset,
            batch_sampler=ShardAwareBatchSampler(
                dataset,
                batch_size,
                drop_last=True,
                seed=seed,
            ),
            num_workers=0,
            pin_memory=True,
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=True,
    )


def collect_action_bank(
    dataset: HighResDualLatentWindowDataset,
    stride: int,
) -> np.ndarray:
    rows: list[np.ndarray] = []
    for trajectory in dataset.trajectories:
        with np.load(trajectory.shard_path, allow_pickle=False) as shard:
            actions = shard["actions_continuous"].astype(np.float32, copy=False)
        maximum = len(actions) - dataset.horizon + 1
        for start in range(0, max(0, maximum), stride):
            rows.append(
                np.ascontiguousarray(
                    actions[start : start + dataset.horizon],
                    dtype=np.float32,
                )
            )
    require(rows, "empty action bank")
    return np.stack(rows)


def action_features(actions: np.ndarray) -> np.ndarray:
    """Represent local action shape and its cumulative SE(2) prefix."""

    actions = np.asarray(actions, dtype=np.float32)
    flat = actions.reshape(-1, actions.shape[-2], 3)
    cumulative = np.zeros_like(flat)
    x = np.zeros(len(flat), dtype=np.float32)
    y = np.zeros(len(flat), dtype=np.float32)
    yaw = np.zeros(len(flat), dtype=np.float32)
    for step in range(flat.shape[1]):
        dx = flat[:, step, 0]
        dy = flat[:, step, 1]
        cosine = np.cos(yaw)
        sine = np.sin(yaw)
        x = x + cosine * dx - sine * dy
        y = y + sine * dx + cosine * dy
        yaw = yaw + flat[:, step, 2]
        cumulative[:, step, 0] = x
        cumulative[:, step, 1] = y
        cumulative[:, step, 2] = yaw
    features = np.concatenate((flat, cumulative), axis=-1).reshape(len(flat), -1)
    return features.reshape(*actions.shape[:-2], features.shape[-1])


class HardActionBank:
    def __init__(self, actions: np.ndarray) -> None:
        self.actions = np.asarray(actions, dtype=np.float32)
        raw = action_features(self.actions)
        self.mean = raw.mean(axis=0, keepdims=True)
        self.scale = np.maximum(raw.std(axis=0, keepdims=True), 0.02)
        self.features = (raw - self.mean) / self.scale

    def candidates(
        self,
        true_actions: np.ndarray,
        count: int,
    ) -> np.ndarray:
        require(count >= 2, "candidate count must be >=2")
        true_features = (action_features(true_actions) - self.mean) / self.scale
        distance = np.square(
            (true_features[:, None] - self.features[None]).astype(np.float32)
        ).mean(axis=-1)
        result = np.empty(
            (len(true_actions), count, *true_actions.shape[1:]),
            dtype=np.float32,
        )
        result[:, 0] = true_actions
        for row in range(len(true_actions)):
            order = np.argsort(distance[row], kind="stable")
            chosen: list[int] = []
            for index in order:
                if distance[row, index] < 1.0e-5:
                    continue
                if any(
                    np.mean(
                        (
                            self.features[index]
                            - self.features[previous]
                        )
                        ** 2
                    )
                    < 1.0e-5
                    for previous in chosen
                ):
                    continue
                chosen.append(int(index))
                if len(chosen) == count - 1:
                    break
            require(len(chosen) == count - 1, "insufficient distinct hard actions")
            result[row, 1:] = self.actions[np.asarray(chosen)]
        return result


def batch_to_device(
    batch: dict[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    return (
        batch["semantic_history"].to(device, non_blocking=True),
        batch["geometry_history"].to(device, non_blocking=True),
        batch["semantic_future"].to(device, non_blocking=True),
        batch["geometry_future"].to(device, non_blocking=True),
        batch["actions"].to(device, non_blocking=True),
    )


def candidate_error(
    output: dict[str, torch.Tensor],
    semantic_future: torch.Tensor,
    geometry_future: torch.Tensor,
) -> torch.Tensor:
    return per_candidate_dual_nmse(
        output,
        semantic_future,
        geometry_future,
    ).mean(dim=-1)


def h4_nmse(
    output: dict[str, torch.Tensor],
    semantic_future: torch.Tensor,
    geometry_future: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    semantic = output["predicted_semantic_tokens"][:, 0, -1].float()
    geometry = output["predicted_geometry_tokens"][:, 0, -1].float()
    semantic_target = semantic_future[:, -1].float()
    geometry_target = geometry_future[:, -1].float()
    semantic_error = (
        (semantic - semantic_target).square().mean(dim=(-1, -2))
        / semantic_target.square().mean(dim=(-1, -2)).clamp_min(1.0e-8)
    )
    geometry_error = (
        (geometry - geometry_target).square().mean(dim=(-1, -2))
        / geometry_target.square().mean(dim=(-1, -2)).clamp_min(1.0e-8)
    )
    dual = 0.5 * (semantic_error + geometry_error)
    return (
        semantic_error.cpu().numpy(),
        geometry_error.cpu().numpy(),
        dual.cpu().numpy(),
    )


@torch.no_grad()
def evaluate(
    model: HighResMultiLayerDualLatentWM,
    loader: DataLoader,
    bank: HardActionBank,
    *,
    candidates: int,
    maximum_windows: int,
    device: torch.device,
    amp: str,
) -> dict[str, Any]:
    model.eval()
    true_errors: list[np.ndarray] = []
    hard_errors: list[np.ndarray] = []
    semantic_h4: list[np.ndarray] = []
    geometry_h4: list[np.ndarray] = []
    dual_h4: list[np.ndarray] = []
    route_rows: dict[str, list[tuple[float, float]]] = {}
    windows = 0
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    for batch in loader:
        semantic_history, geometry_history, semantic_future, geometry_future, actions = (
            batch_to_device(batch, device)
        )
        trajectories = [str(value) for value in batch["trajectory_id"]]
        if maximum_windows > 0 and windows + len(actions) > maximum_windows:
            keep = maximum_windows - windows
            if keep <= 0:
                break
            semantic_history = semantic_history[:keep]
            geometry_history = geometry_history[:keep]
            semantic_future = semantic_future[:keep]
            geometry_future = geometry_future[:keep]
            actions = actions[:keep]
            trajectories = trajectories[:keep]
        candidate_numpy = bank.candidates(
            actions.cpu().numpy(),
            candidates,
        )
        candidate_tensor = torch.from_numpy(candidate_numpy).to(
            device,
            non_blocking=True,
        )
        with amp_context(amp):
            output = model(
                semantic_history,
                geometry_history,
                candidate_tensor,
            )
        errors = candidate_error(
            output,
            semantic_future,
            geometry_future,
        )
        sem, geo, dual = h4_nmse(output, semantic_future, geometry_future)
        true_errors.append(errors[:, 0].cpu().numpy())
        hard_errors.append(errors[:, 1:].cpu().numpy())
        semantic_h4.append(sem)
        geometry_h4.append(geo)
        dual_h4.append(dual)
        true_numpy = errors[:, 0].cpu().numpy()
        hard_numpy = errors[:, 1:].mean(dim=1).cpu().numpy()
        for trajectory, true_value, hard_value in zip(
            trajectories,
            true_numpy.tolist(),
            hard_numpy.tolist(),
        ):
            route_rows.setdefault(trajectory, []).append(
                (float(true_value), float(hard_value))
            )
        windows += len(actions)
        if maximum_windows > 0 and windows >= maximum_windows:
            break
    require(windows > 0, "evaluation produced no windows")
    true_array = np.concatenate(true_errors)
    hard_array = np.concatenate(hard_errors)
    rank = 1 + (hard_array < true_array[:, None]).sum(axis=1)
    route_gap = {
        route: float(np.mean([hard - true for true, hard in rows]))
        for route, rows in route_rows.items()
    }
    return {
        "windows": windows,
        "candidates_including_true": candidates,
        "semantic_h4_nmse": float(np.concatenate(semantic_h4).mean()),
        "geometry_h4_nmse": float(np.concatenate(geometry_h4).mean()),
        "dual_h4_nmse": float(np.concatenate(dual_h4).mean()),
        "true_action_dual_nmse": float(true_array.mean()),
        "hard_action_dual_nmse": float(hard_array.mean()),
        "hard_gap_fraction": float(
            (hard_array.mean() - true_array.mean())
            / max(float(true_array.mean()), 1.0e-8)
        ),
        "retrieval_top1": float(np.mean(rank == 1)),
        "pairwise_action_win_rate": float(
            np.mean(true_array[:, None] < hard_array)
        ),
        "mean_rank": float(rank.mean()),
        "mean_reciprocal_rank": float(np.mean(1.0 / rank)),
        "positive_gap_trajectory_fraction": float(
            np.mean(np.asarray(list(route_gap.values())) > 0.0)
        ),
        "trajectory_count": len(route_gap),
        "runtime_seconds": time.perf_counter() - started,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
    }


def train(
    model: HighResMultiLayerDualLatentWM,
    loader: DataLoader,
    bank: HardActionBank,
    config: dict[str, Any],
    *,
    device: torch.device,
) -> dict[str, Any]:
    training = config["training"]
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    steps = int(training["steps"])
    amp = str(training["amp"])
    temperature = float(training["temperature"])
    listwise_weight = float(training["listwise_weight"])
    rows: list[tuple[float, float, float, float]] = []
    completed = 0
    epochs = 0
    started = time.perf_counter()
    model.train()
    while completed < steps:
        epochs += 1
        sampler = getattr(loader, "batch_sampler", None)
        if hasattr(sampler, "epoch"):
            sampler.epoch = epochs - 1
        for batch in loader:
            semantic_history, geometry_history, semantic_future, geometry_future, actions = (
                batch_to_device(batch, device)
            )
            candidate_numpy = bank.candidates(
                actions.cpu().numpy(),
                int(training["candidates"]),
            )
            candidates = torch.from_numpy(candidate_numpy).to(
                device,
                non_blocking=True,
            )
            optimizer.zero_grad(set_to_none=True)
            with amp_context(amp):
                output = model(
                    semantic_history,
                    geometry_history,
                    candidates,
                )
                errors = candidate_error(
                    output,
                    semantic_future,
                    geometry_future,
                )
                reconstruction = errors[:, 0].mean()
                listwise = F.cross_entropy(
                    -errors / temperature,
                    torch.zeros(
                        len(errors),
                        dtype=torch.long,
                        device=device,
                    ),
                )
                loss = reconstruction + listwise_weight * listwise
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                parameters,
                float(training["gradient_clip"]),
            )
            optimizer.step()
            rows.append(
                (
                    float(loss.detach()),
                    float(reconstruction.detach()),
                    float(listwise.detach()),
                    float(gradient_norm.detach()),
                )
            )
            completed += 1
            if completed % 100 == 0 or completed == steps:
                recent = np.asarray(rows[-100:], dtype=np.float64)
                print(
                    canonical_json(
                        {
                            "stage": "training",
                            "step": completed,
                            "steps": steps,
                            "loss": float(recent[:, 0].mean()),
                            "reconstruction": float(recent[:, 1].mean()),
                            "listwise": float(recent[:, 2].mean()),
                            "elapsed_seconds": time.perf_counter() - started,
                        }
                    ),
                    flush=True,
                )
            if completed >= steps:
                break
    values = np.asarray(rows, dtype=np.float64)
    return {
        "steps": completed,
        "epochs_touched": epochs,
        "mean_loss": float(values[:, 0].mean()),
        "mean_reconstruction": float(values[:, 1].mean()),
        "mean_listwise": float(values[:, 2].mean()),
        "mean_gradient_norm": float(values[:, 3].mean()),
        "last_100_loss": float(values[-100:, 0].mean()),
        "trainable_parameters": int(sum(parameter.numel() for parameter in parameters)),
        "runtime_seconds": time.perf_counter() - started,
    }


def main() -> None:
    args = parse_args()
    require(torch.cuda.is_available(), "CUDA unavailable")
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    require(config.get("schema") == "highres_dual_observed_listwise_v1", "bad config")
    protection = config["protection"]
    require(protection["ate_or_objective_loaded"] is False, "outcome leakage")
    require(protection["sealed_validation_opened"] is False, "sealed leakage")
    if args.smoke:
        config = json.loads(json.dumps(config))
        config["training"]["steps"] = 2
        config["evaluation"]["maximum_windows"] = 16
    output = args.output_dir.resolve()
    require(not output.exists(), f"refusing to overwrite: {output}")
    output.mkdir(parents=True, exist_ok=False)
    latent_root = resolve(config["inputs"]["latent_root"])
    checkpoint_path = resolve(config["inputs"]["base_checkpoint"])
    require(latent_root.is_dir(), f"missing latent root: {latent_root}")
    require(checkpoint_path.is_file(), f"missing checkpoint: {checkpoint_path}")
    seed = int(config["training"]["seed"])
    set_seed(seed)
    device = torch.device("cuda")
    model, base_payload = load_model(checkpoint_path, device)
    freeze_nonconditional_state(model)
    train_dataset = HighResDualLatentWindowDataset(
        latent_root,
        "train",
        history_len=4,
        horizon=4,
        stream_cache_shards=2,
    )
    val_dataset = HighResDualLatentWindowDataset(
        latent_root,
        "val",
        history_len=4,
        horizon=4,
        stream_cache_shards=2,
    )
    train_loader = make_loader(
        train_dataset,
        batch_size=int(config["training"]["batch_size"]),
        seed=seed,
        training=True,
    )
    val_loader = make_loader(
        val_dataset,
        batch_size=int(config["evaluation"]["batch_size"]),
        seed=seed + 10000,
        training=False,
    )
    train_bank = HardActionBank(
        collect_action_bank(
            train_dataset,
            int(config["training"]["action_bank_stride"]),
        )
    )
    val_bank = HardActionBank(collect_action_bank(val_dataset, 1))
    started = time.perf_counter()
    baseline = evaluate(
        model,
        val_loader,
        val_bank,
        candidates=int(config["evaluation"]["candidates"]),
        maximum_windows=int(config["evaluation"]["maximum_windows"]),
        device=device,
        amp=str(config["training"]["amp"]),
    )
    training = train(
        model,
        train_loader,
        train_bank,
        config,
        device=device,
    )
    final = evaluate(
        model,
        val_loader,
        val_bank,
        candidates=int(config["evaluation"]["candidates"]),
        maximum_windows=int(config["evaluation"]["maximum_windows"]),
        device=device,
        amp=str(config["training"]["amp"]),
    )
    degradation = (
        final["dual_h4_nmse"] - baseline["dual_h4_nmse"]
    ) / baseline["dual_h4_nmse"]
    top1_gain = final["retrieval_top1"] - baseline["retrieval_top1"]
    pairwise_gain = (
        final["pairwise_action_win_rate"]
        - baseline["pairwise_action_win_rate"]
    )
    checks = {
        "reconstruction_preserved": degradation <= 0.05,
        "hard_top1_improved": top1_gain >= 0.03,
        "hard_pairwise_improved": pairwise_gain >= 0.03,
        "hard_pairwise_minimum": final["pairwise_action_win_rate"] >= 0.65,
        "route_consistency": final["positive_gap_trajectory_fraction"] >= 0.75,
    }
    go = all(checks.values())
    checkpoint = output / "observed_listwise_final.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "config": base_payload["config"],
            "experiment": config,
            "baseline_validation": baseline,
            "final_validation": final,
            "outcome_gradients": False,
            "ate_or_objective_loaded": False,
            "sealed_test_opened": False,
        },
        checkpoint,
    )
    result = {
        "schema": "highres_dual_observed_listwise_result_v1",
        "status": "smoke_complete" if args.smoke else "complete",
        "decision": (
            "go_observed_listwise_action_identifiability"
            if go
            else "no_go_observed_listwise_action_identifiability"
        ),
        "go": go,
        "checks": checks,
        "derived_deltas": {
            "relative_reconstruction_degradation": float(degradation),
            "retrieval_top1_gain": float(top1_gain),
            "pairwise_action_win_rate_gain": float(pairwise_gain),
        },
        "baseline_validation": baseline,
        "final_validation": final,
        "training": training,
        "data": {
            "train_windows": len(train_dataset),
            "validation_windows": len(val_dataset),
            "train_trajectories": len(train_dataset.trajectories),
            "validation_trajectories": len(val_dataset.trajectories),
            "train_action_bank": len(train_bank.actions),
            "validation_action_bank": len(val_bank.actions),
        },
        "inputs": {
            "config": {
                "path": str(config_path),
                "sha256": sha256_file(config_path),
            },
            "base_checkpoint": {
                "path": str(checkpoint_path),
                "sha256": sha256_file(checkpoint_path),
            },
            "latent_metadata": {
                "path": str(latent_root / "metadata.json"),
                "sha256": sha256_file(latent_root / "metadata.json"),
            },
            "runner": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
        },
        "artifact": {
            "checkpoint": str(checkpoint),
            "sha256": sha256_file(checkpoint),
        },
        "protection": {
            "dino_encoder_frozen": True,
            "history_encoders_frozen": True,
            "action_decoder_unused": True,
            "ate_or_objective_loaded": False,
            "sealed_validation_opened": False,
            "formal_v70_modified": False,
        },
        "runtime_seconds": time.perf_counter() - started,
    }
    result_path = output / "result.json"
    write_json(result_path, result)
    print(
        canonical_json(
            {
                "status": result["status"],
                "decision": result["decision"],
                "go": go,
                "baseline": {
                    "dual_h4_nmse": baseline["dual_h4_nmse"],
                    "retrieval_top1": baseline["retrieval_top1"],
                    "pairwise": baseline["pairwise_action_win_rate"],
                },
                "final": {
                    "dual_h4_nmse": final["dual_h4_nmse"],
                    "retrieval_top1": final["retrieval_top1"],
                    "pairwise": final["pairwise_action_win_rate"],
                },
                "result": str(result_path),
                "result_sha256": sha256_file(result_path),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(
            canonical_json(
                {
                    "status": "fail_closed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "ate_or_objective_loaded": False,
                    "sealed_test_opened": False,
                }
            ),
            file=sys.stderr,
        )
        raise SystemExit(2) from error
