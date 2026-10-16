#!/usr/bin/env python3
"""Route-disjoint inverse-dynamics probe on real high-resolution dual latents."""

from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import random
import sys
import time
from typing import Any

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(Path(__file__).resolve().parent)]

from latent_window_utils import ShardAwareBatchSampler  # noqa: E402
from recon_latents.dataset_highres_dual_v1 import HighResDualLatentWindowDataset  # noqa: E402
from recon_latents.models_v2 import integrate_se2_prefix  # noqa: E402


LATENT_ROOT = ROOT / "data/highres_dual_latents/scand_v1"
FROZEN_LTWM = (
    ROOT
    / "outputs/highres_dual_ltwm_v1/frozen/highres_multilayer_dual_ltwm_scand_v1.pt"
)
FORMAL_V70 = (
    ROOT
    / "outputs/cem_development_v1/dualdomain_cem_tail_v70_broad_hashfold_archive/result.json"
)
DEFAULT_OUTPUT = (
    ROOT / "outputs/highres_dual_ltwm_v1/real_future_inverse_probe_v1"
)
SEED = 2026072407
STEPS = 2400
BATCH_SIZE = 16
LEARNING_RATE = 3.0e-4
WEIGHT_DECAY = 1.0e-4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--steps", type=int, default=STEPS)
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def recording_group(trajectory_id: str) -> str:
    prefix, fragment = trajectory_id.rsplit("_", 1)
    require(fragment.isdigit(), f"unexpected trajectory id: {trajectory_id}")
    return prefix


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class DualRealFutureInverse(nn.Module):
    """Decode four-step motion without receiving the action as an input."""

    def __init__(self) -> None:
        super().__init__()
        self.semantic = nn.Sequential(nn.Linear(384, 16), nn.GELU())
        self.geometry = nn.Sequential(nn.Linear(192, 16), nn.GELU())
        self.spatial = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 48, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(48, 48, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(48 * 4 * 4, 128),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(128, 12),
        )

    def forward(
        self,
        semantic_previous: torch.Tensor,
        semantic_current: torch.Tensor,
        semantic_future: torch.Tensor,
        geometry_previous: torch.Tensor,
        geometry_current: torch.Tensor,
        geometry_future: torch.Tensor,
    ) -> torch.Tensor:
        value = torch.cat(
            (
                self.semantic(semantic_future - semantic_current),
                self.semantic(semantic_current - semantic_previous),
                self.geometry(geometry_future - geometry_current),
                self.geometry(geometry_current - geometry_previous),
            ),
            dim=-1,
        )
        value = value.reshape(-1, 21, 28, 64).permute(0, 3, 1, 2)
        return self.head(self.spatial(value))


def amp_context():
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


def action_statistics(dataset: HighResDualLatentWindowDataset) -> tuple[np.ndarray, np.ndarray]:
    rows = []
    for trajectory in dataset.trajectories:
        with np.load(trajectory.shard_path, allow_pickle=False) as shard:
            actions = shard["actions_continuous"].astype(np.float32, copy=False)
        rows.append(
            np.stack(
                [actions[t : t + 4].reshape(12) for t in range(3, len(actions) - 3)]
            )
        )
    value = np.concatenate(rows)
    return value.mean(axis=0).astype(np.float32), np.maximum(
        value.std(axis=0), 0.01
    ).astype(np.float32)


def build_uniform_schedule(
    dataset: HighResDualLatentWindowDataset,
    steps: int,
) -> tuple[np.ndarray, dict[str, int]]:
    by_group: dict[str, list[int]] = defaultdict(list)
    for trajectory_index, trajectory in enumerate(dataset.trajectories):
        by_group[recording_group(trajectory.trajectory_id)].append(trajectory_index)
    groups = sorted(by_group)
    rng = np.random.default_rng(SEED)
    schedule = np.empty((steps, BATCH_SIZE), dtype=np.int64)
    counts: dict[str, int] = defaultdict(int)
    for step in range(steps):
        group = groups[int(rng.integers(0, len(groups)))]
        trajectory_index = int(rng.choice(by_group[group]))
        choices = np.asarray(
            dataset.window_indices_by_shard()[trajectory_index], dtype=np.int64
        )
        schedule[step] = rng.choice(
            choices, size=BATCH_SIZE, replace=len(choices) < BATCH_SIZE
        )
        counts[group] += BATCH_SIZE
    return schedule, dict(sorted(counts.items()))


def collate_indices(
    dataset: HighResDualLatentWindowDataset,
    indices: np.ndarray,
) -> dict[str, Any]:
    rows = [dataset[int(index)] for index in indices]
    result: dict[str, Any] = {}
    for key in (
        "semantic_history",
        "geometry_history",
        "semantic_future",
        "geometry_future",
        "actions",
    ):
        result[key] = torch.stack([row[key] for row in rows])
    result["trajectory_id"] = [str(row["trajectory_id"]) for row in rows]
    return result


def predict_normalized(
    model: DualRealFutureInverse,
    batch: dict[str, Any],
    device: torch.device,
) -> torch.Tensor:
    semantic_history = batch["semantic_history"].to(device, non_blocking=True)
    geometry_history = batch["geometry_history"].to(device, non_blocking=True)
    semantic_future = batch["semantic_future"].to(device, non_blocking=True)
    geometry_future = batch["geometry_future"].to(device, non_blocking=True)
    return model(
        semantic_history[:, -2],
        semantic_history[:, -1],
        semantic_future[:, -1],
        geometry_history[:, -2],
        geometry_history[:, -1],
        geometry_future[:, -1],
    )


def train(
    model: DualRealFutureInverse,
    dataset: HighResDualLatentWindowDataset,
    mean: np.ndarray,
    scale: np.ndarray,
    device: torch.device,
    steps: int,
) -> dict[str, Any]:
    schedule, group_counts = build_uniform_schedule(dataset, steps)
    mean_tensor = torch.from_numpy(mean).to(device)
    scale_tensor = torch.from_numpy(scale).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    rows = []
    started = time.perf_counter()
    model.train()
    for step, indices in enumerate(schedule, start=1):
        batch = collate_indices(dataset, indices)
        target = batch["actions"].reshape(BATCH_SIZE, 12).to(device)
        normalized_target = (target - mean_tensor) / scale_tensor
        optimizer.zero_grad(set_to_none=True)
        with amp_context():
            normalized_prediction = predict_normalized(model, batch, device)
            action_loss = F.smooth_l1_loss(
                normalized_prediction, normalized_target
            )
            prediction = normalized_prediction * scale_tensor + mean_tensor
            prediction_path = integrate_se2_prefix(
                prediction.reshape(-1, 4, 3)
            )[..., :2]
            target_path = integrate_se2_prefix(target.reshape(-1, 4, 3))[..., :2]
            path_loss = F.smooth_l1_loss(prediction_path, target_path)
            loss = action_loss + 0.25 * path_loss
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        rows.append(
            {
                "loss": float(loss.detach()),
                "action_loss": float(action_loss.detach()),
                "path_loss": float(path_loss.detach()),
                "gradient_norm": float(gradient_norm.detach()),
            }
        )
        if step % 100 == 0 or step == steps:
            recent = rows[-100:]
            print(
                canonical_json(
                    {
                        "stage": "train_real_future_inverse",
                        "step": step,
                        "steps": steps,
                        "loss": float(np.mean([row["loss"] for row in recent])),
                        "elapsed_seconds": time.perf_counter() - started,
                    }
                ),
                flush=True,
            )
    return {
        **{
            key: float(np.mean([row[key] for row in rows]))
            for key in rows[0]
        },
        "steps": steps,
        "batch_size": BATCH_SIZE,
        "recording_group_sample_counts": group_counts,
        "runtime_seconds": time.perf_counter() - started,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }


def structured_negatives(actions: torch.Tensor) -> torch.Tensor:
    result = []
    reversed_action = actions.flip(1)
    result.append(reversed_action)
    yaw_flip = actions.clone()
    yaw_flip[..., 2].neg_()
    result.append(yaw_flip)
    lateral_flip = actions.clone()
    lateral_flip[..., 1].neg_()
    result.append(lateral_flip)
    magnitude = actions.clone()
    magnitude[..., :2] *= 0.5
    magnitude[..., 2] *= 1.5
    result.append(magnitude)
    return torch.stack(result, dim=1)


def fixed_negative_actions(actions: torch.Tensor) -> torch.Tensor:
    observed = [
        torch.roll(actions, shifts=offset, dims=0) for offset in range(1, 8)
    ]
    return torch.cat((torch.stack(observed, dim=1), structured_negatives(actions)), dim=1)


def path_distance(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left_path = integrate_se2_prefix(left.reshape(-1, 4, 3))[..., :2]
    right_path = integrate_se2_prefix(right.reshape(-1, 4, 3))[..., :2]
    return (left_path - right_path).square().sum(dim=-1).mean(dim=-1)


@torch.no_grad()
def evaluate(
    model: DualRealFutureInverse,
    dataset: HighResDualLatentWindowDataset,
    mean: np.ndarray,
    scale: np.ndarray,
    device: torch.device,
) -> dict[str, Any]:
    loader = DataLoader(
        dataset,
        batch_sampler=ShardAwareBatchSampler(
            dataset,
            8,
            drop_last=False,
            seed=SEED + 10000,
        ),
        num_workers=0,
        pin_memory=True,
    )
    sampler = getattr(loader, "batch_sampler", None)
    if hasattr(sampler, "epoch"):
        sampler.epoch = 0
    mean_tensor = torch.from_numpy(mean).to(device)
    scale_tensor = torch.from_numpy(scale).to(device)
    action_sq = []
    mean_action_sq = []
    path_sq = []
    mean_path_sq = []
    endpoint_error = []
    mean_endpoint_error = []
    ranks = []
    pairwise = []
    group_rows: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
    started = time.perf_counter()
    model.eval()
    windows = 0
    for batch in loader:
        actions = batch["actions"].to(device, non_blocking=True)
        target = actions.reshape(len(actions), 12)
        with amp_context():
            normalized_prediction = predict_normalized(model, batch, device)
        prediction = normalized_prediction.float() * scale_tensor + mean_tensor
        mean_prediction = mean_tensor.expand_as(prediction)
        prediction_action = prediction.reshape(-1, 4, 3)
        mean_action = mean_prediction.reshape(-1, 4, 3)
        target_path = integrate_se2_prefix(actions)[..., :2]
        prediction_path = integrate_se2_prefix(prediction_action)[..., :2]
        mean_path = integrate_se2_prefix(mean_action)[..., :2]
        action_sq.append(
            (((prediction - target) / scale_tensor).square().mean(dim=-1)).cpu()
        )
        mean_action_sq.append(
            (((mean_prediction - target) / scale_tensor).square().mean(dim=-1)).cpu()
        )
        path_sq.append((prediction_path - target_path).square().mean(dim=(-1, -2)).cpu())
        mean_path_sq.append((mean_path - target_path).square().mean(dim=(-1, -2)).cpu())
        endpoint_error.append(
            torch.linalg.vector_norm(prediction_path[:, -1] - target_path[:, -1], dim=-1).cpu()
        )
        mean_endpoint_error.append(
            torch.linalg.vector_norm(mean_path[:, -1] - target_path[:, -1], dim=-1).cpu()
        )
        negatives = fixed_negative_actions(actions)
        true_distance = path_distance(prediction_action, actions)
        negative_distance = path_distance(
            prediction_action[:, None].expand_as(negatives).reshape(-1, 4, 3),
            negatives.reshape(-1, 4, 3),
        ).reshape(len(actions), -1)
        rank = 1 + (negative_distance < true_distance[:, None]).sum(dim=-1)
        ranks.append(rank.cpu())
        pairwise.append((true_distance[:, None] < negative_distance).float().cpu())
        per_item_path = (prediction_path - target_path).square().mean(dim=(-1, -2))
        per_item_mean = (mean_path - target_path).square().mean(dim=(-1, -2))
        for trajectory, item_path, item_mean, item_rank in zip(
            batch["trajectory_id"], per_item_path, per_item_mean, rank
        ):
            group_rows[recording_group(str(trajectory))].append(
                (float(item_path), float(item_mean), float(item_rank))
            )
        windows += len(actions)
        if windows % 512 < len(actions):
            print(
                canonical_json(
                    {
                        "stage": "evaluate_real_future_inverse",
                        "windows": windows,
                        "elapsed_seconds": time.perf_counter() - started,
                    }
                ),
                flush=True,
            )
    action_rmse = float(torch.cat(action_sq).mean().sqrt())
    mean_action_rmse = float(torch.cat(mean_action_sq).mean().sqrt())
    path_rmse = float(torch.cat(path_sq).mean().sqrt())
    mean_path_rmse = float(torch.cat(mean_path_sq).mean().sqrt())
    endpoint = float(torch.cat(endpoint_error).mean())
    mean_endpoint = float(torch.cat(mean_endpoint_error).mean())
    rank_array = torch.cat(ranks).numpy()
    pairwise_array = torch.cat(pairwise).numpy()
    groups = {
        group: {
            "windows": len(rows),
            "path_rmse_m": float(np.sqrt(np.mean([row[0] for row in rows]))),
            "mean_action_path_rmse_m": float(
                np.sqrt(np.mean([row[1] for row in rows]))
            ),
            "path_improved": bool(
                np.mean([row[0] for row in rows])
                < np.mean([row[1] for row in rows])
            ),
            "retrieval_top1": float(np.mean([row[2] == 1 for row in rows])),
        }
        for group, rows in sorted(group_rows.items())
    }
    return {
        "windows": windows,
        "recording_groups": len(groups),
        "standardized_action_rmse": action_rmse,
        "mean_action_standardized_rmse": mean_action_rmse,
        "standardized_action_relative": action_rmse / mean_action_rmse,
        "integrated_path_rmse_m": path_rmse,
        "mean_action_integrated_path_rmse_m": mean_path_rmse,
        "integrated_path_relative": path_rmse / mean_path_rmse,
        "endpoint_error_m": endpoint,
        "mean_action_endpoint_error_m": mean_endpoint,
        "endpoint_error_relative": endpoint / mean_endpoint,
        "pairwise_action_win_rate": float(pairwise_array.mean()),
        "retrieval_top1": float(np.mean(rank_array == 1)),
        "mean_reciprocal_rank": float(np.mean(1.0 / rank_array)),
        "mean_rank": float(rank_array.mean()),
        "path_improved_recording_fraction": float(
            np.mean([row["path_improved"] for row in groups.values()])
        ),
        "recording_group": groups,
        "runtime_seconds": time.perf_counter() - started,
    }


def main() -> None:
    args = parse_args()
    require(torch.cuda.is_available(), "CUDA is required")
    require(args.steps == STEPS, "this frozen probe only permits the preregistered step count")
    output = args.output.resolve()
    result_path = output / "result.json"
    checkpoint_path = output / "real_future_inverse_final.pt"
    require(not result_path.exists(), f"refusing to overwrite {result_path}")
    require(not checkpoint_path.exists(), f"refusing to overwrite {checkpoint_path}")
    require(LATENT_ROOT.is_dir(), f"missing latent root: {LATENT_ROOT}")
    require(FROZEN_LTWM.is_file(), f"missing frozen LTWM: {FROZEN_LTWM}")
    require(FORMAL_V70.is_file(), f"missing formal-v70: {FORMAL_V70}")
    formal_before = sha256_file(FORMAL_V70)
    frozen_before = sha256_file(FROZEN_LTWM)
    require(
        frozen_before == "67b6278805cb54401823b3ca15ff59685dbb2ef6a20961d8ae6ff63935b6c823",
        "frozen highres LTWM changed",
    )
    splits = json.loads((LATENT_ROOT / "splits.json").read_text(encoding="utf-8"))
    require(not splits.get("test"), "sealed test split must remain empty")
    train_groups = {recording_group(str(value)) for value in splits["train"]}
    val_groups = {recording_group(str(value)) for value in splits["val"]}
    require(not train_groups.intersection(val_groups), "recording-group leakage")
    set_seed(SEED)
    device = torch.device(args.device)
    train_set = HighResDualLatentWindowDataset(
        LATENT_ROOT, "train", 4, 4, stream_cache_shards=2
    )
    val_set = HighResDualLatentWindowDataset(
        LATENT_ROOT, "val", 4, 4, stream_cache_shards=2
    )
    mean, scale = action_statistics(train_set)
    model = DualRealFutureInverse().to(device)
    started = time.perf_counter()
    training = train(model, train_set, mean, scale, device, args.steps)
    validation = evaluate(model, val_set, mean, scale, device)
    checks = {
        "standardized_action_relative": validation["standardized_action_relative"] <= 0.80,
        "integrated_path_relative": validation["integrated_path_relative"] <= 0.80,
        "endpoint_error_relative": validation["endpoint_error_relative"] <= 0.80,
        "pairwise_action_win_rate": validation["pairwise_action_win_rate"] >= 0.80,
        "retrieval_top1": validation["retrieval_top1"] >= 0.20,
        "recording_consistency": validation["path_improved_recording_fraction"] >= 0.80,
    }
    go = all(checks.values())
    output.mkdir(parents=True, exist_ok=False)
    torch.save(
        {
            "model": model.state_dict(),
            "architecture": "DualRealFutureInverse_v1",
            "action_mean": torch.from_numpy(mean),
            "action_scale": torch.from_numpy(scale),
            "seed": SEED,
            "training": training,
            "train_recording_groups": sorted(train_groups),
            "validation_recording_groups": sorted(val_groups),
            "outcome_gradients": False,
            "sealed_test_opened": False,
        },
        checkpoint_path,
    )
    result = {
        "schema": "highres_dual_real_future_inverse_probe_result_v1",
        "status": "complete_single_route_disjoint_development_probe",
        "decision": (
            "GO_FREEZE_INVERSE_TEACHER_FOR_ONE_LTWM_CONSTRAINT"
            if go
            else "NO_GO_REAL_FUTURE_DUAL_LATENT_NOT_ACTION_OBSERVABLE_ENOUGH"
        ),
        "go": go,
        "checks": checks,
        "data": {
            "latent_root": str(LATENT_ROOT),
            "metadata_sha256": sha256_file(LATENT_ROOT / "metadata.json"),
            "splits_sha256": sha256_file(LATENT_ROOT / "splits.json"),
            "train_windows": len(train_set),
            "validation_windows": len(val_set),
            "train_trajectories": len(train_set.trajectories),
            "validation_trajectories": len(val_set.trajectories),
            "train_recording_groups": len(train_groups),
            "validation_recording_groups": len(val_groups),
            "recording_group_overlap": 0,
        },
        "training": training,
        "validation": validation,
        "artifacts": {
            "checkpoint": {
                "path": str(checkpoint_path),
                "sha256": sha256_file(checkpoint_path),
            },
            "runner": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
        },
        "protection": {
            "outcome_labels_loaded": False,
            "ate_or_objective_gradients": False,
            "sealed_validation_opened": False,
            "union65_rescored": False,
            "formal_v70_unchanged": sha256_file(FORMAL_V70) == formal_before,
            "formal_v70_sha256": sha256_file(FORMAL_V70),
            "frozen_highres_ltwm_unchanged": sha256_file(FROZEN_LTWM) == frozen_before,
            "frozen_highres_ltwm_sha256": sha256_file(FROZEN_LTWM),
        },
        "runtime_seconds": time.perf_counter() - started,
    }
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(canonical_json({"result": str(result_path), "decision": result["decision"]}))


if __name__ == "__main__":
    main()
