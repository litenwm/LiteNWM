#!/usr/bin/env python3
"""Learn one outcome-free perceptual metric for predicted-future/goal matching."""

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


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(Path(__file__).resolve().parent)]

from recon_latents.dataset_highres_dual_v1 import HighResDualLatentWindowDataset  # noqa: E402
from recon_latents.models_highres_dual_v1 import (  # noqa: E402
    HighResDualLatentWMConfig,
    HighResMultiLayerDualLatentWM,
)


LATENT_ROOT = ROOT / "data/highres_dual_latents/scand_v1"
FROZEN_LTWM = (
    ROOT
    / "outputs/highres_dual_ltwm_v1/frozen/highres_multilayer_dual_ltwm_scand_v1.pt"
)
FORMAL_V70 = (
    ROOT
    / "outputs/cem_development_v1/dualdomain_cem_tail_v70_broad_hashfold_archive/result.json"
)
DEFAULT_OUTPUT = ROOT / "outputs/highres_dual_ltwm_v1/future_goal_matcher_v1"
SEED = 2026072408
STEPS = 2400
BATCH_SIZE = 8
LEARNING_RATE = 3.0e-4
WEIGHT_DECAY = 1.0e-4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
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


def amp_context():
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


class FutureGoalMatcher(nn.Module):
    """A learned latent distance; it never receives actions or outcomes."""

    def __init__(self) -> None:
        super().__init__()
        self.semantic = nn.Sequential(nn.Linear(384, 12), nn.GELU())
        self.geometry = nn.Sequential(nn.Linear(192, 12), nn.GELU())
        self.spatial = nn.Sequential(
            nn.Conv2d(72, 48, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(48, 40, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(40, 32, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((3, 4)),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 3 * 4, 64),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(64, 1),
        )

    @staticmethod
    def branch_features(
        projection: nn.Module,
        prediction: torch.Tensor,
        goal: torch.Tensor,
        current: torch.Tensor,
    ) -> torch.Tensor:
        predicted = projection(prediction)
        goal_value = projection(goal)[:, None]
        current_value = projection(current)[:, None]
        error = predicted - goal_value
        direction_agreement = (
            (predicted - current_value) * (goal_value - current_value)
        )
        return torch.cat((error.abs(), error.square(), direction_agreement), dim=-1)

    def forward(
        self,
        predicted_semantic: torch.Tensor,
        predicted_geometry: torch.Tensor,
        semantic_goal: torch.Tensor,
        geometry_goal: torch.Tensor,
        semantic_current: torch.Tensor,
        geometry_current: torch.Tensor,
    ) -> torch.Tensor:
        value = torch.cat(
            (
                self.branch_features(
                    self.semantic,
                    predicted_semantic,
                    semantic_goal,
                    semantic_current,
                ),
                self.branch_features(
                    self.geometry,
                    predicted_geometry,
                    geometry_goal,
                    geometry_current,
                ),
            ),
            dim=-1,
        )
        batch, candidates = value.shape[:2]
        value = value.reshape(batch * candidates, 21, 28, 72).permute(0, 3, 1, 2)
        return self.head(self.spatial(value)).reshape(batch, candidates)


def fixed_negative_actions(actions: torch.Tensor) -> torch.Tensor:
    observed = [
        torch.roll(actions, shifts=offset, dims=0) for offset in range(1, 8)
    ]
    reversed_action = actions.flip(1)
    yaw_flip = actions.clone()
    yaw_flip[..., 2].neg_()
    lateral_flip = actions.clone()
    lateral_flip[..., 1].neg_()
    magnitude = actions.clone()
    magnitude[..., :2] *= 0.5
    magnitude[..., 2] *= 1.5
    structured = torch.stack(
        (reversed_action, yaw_flip, lateral_flip, magnitude), dim=1
    )
    return torch.cat((torch.stack(observed, dim=1), structured), dim=1)


def build_uniform_schedule(
    dataset: HighResDualLatentWindowDataset,
) -> tuple[np.ndarray, dict[str, int]]:
    by_group: dict[str, list[int]] = defaultdict(list)
    for trajectory_index, trajectory in enumerate(dataset.trajectories):
        by_group[recording_group(trajectory.trajectory_id)].append(trajectory_index)
    groups = sorted(by_group)
    rng = np.random.default_rng(SEED)
    schedule = np.empty((STEPS, BATCH_SIZE), dtype=np.int64)
    counts: dict[str, int] = defaultdict(int)
    for step in range(STEPS):
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


@torch.no_grad()
def predict_candidates(
    world_model: HighResMultiLayerDualLatentWM,
    semantic_history: torch.Tensor,
    geometry_history: torch.Tensor,
    candidate_actions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    with amp_context():
        output = world_model(
            semantic_history,
            geometry_history,
            candidate_actions,
        )
    return (
        output["predicted_semantic_tokens"][:, :, -1].float(),
        output["predicted_geometry_tokens"][:, :, -1].float(),
    )


def raw_dual_distance(
    predicted_semantic: torch.Tensor,
    predicted_geometry: torch.Tensor,
    semantic_goal: torch.Tensor,
    geometry_goal: torch.Tensor,
) -> torch.Tensor:
    semantic_energy = semantic_goal.square().mean(dim=(-1, -2)).clamp_min(1.0e-8)
    geometry_energy = geometry_goal.square().mean(dim=(-1, -2)).clamp_min(1.0e-8)
    semantic = (
        (predicted_semantic - semantic_goal[:, None])
        .square()
        .mean(dim=(-1, -2))
        / semantic_energy[:, None]
    )
    geometry = (
        (predicted_geometry - geometry_goal[:, None])
        .square()
        .mean(dim=(-1, -2))
        / geometry_energy[:, None]
    )
    return 0.5 * (semantic + geometry)


def train(
    matcher: FutureGoalMatcher,
    world_model: HighResMultiLayerDualLatentWM,
    dataset: HighResDualLatentWindowDataset,
    device: torch.device,
) -> dict[str, Any]:
    schedule, group_counts = build_uniform_schedule(dataset)
    optimizer = torch.optim.AdamW(
        matcher.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    rows = []
    started = time.perf_counter()
    matcher.train()
    world_model.eval()
    for step, indices in enumerate(schedule, start=1):
        batch = collate_indices(dataset, indices)
        (
            semantic_history,
            geometry_history,
            semantic_future,
            geometry_future,
            actions,
        ) = batch_to_device(batch, device)
        negative_actions = fixed_negative_actions(actions)
        candidate_actions = torch.cat((actions[:, None], negative_actions), dim=1)
        predicted_semantic, predicted_geometry = predict_candidates(
            world_model,
            semantic_history,
            geometry_history,
            candidate_actions,
        )
        optimizer.zero_grad(set_to_none=True)
        with amp_context():
            distance = matcher(
                predicted_semantic,
                predicted_geometry,
                semantic_future[:, -1],
                geometry_future[:, -1],
                semantic_history[:, -1],
                geometry_history[:, -1],
            )
            target = torch.zeros(BATCH_SIZE, dtype=torch.long, device=device)
            loss = F.cross_entropy(-distance, target)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(matcher.parameters(), 1.0)
        optimizer.step()
        with torch.no_grad():
            rank = 1 + (distance[:, 1:] < distance[:, :1]).sum(dim=1)
        rows.append(
            {
                "loss": float(loss.detach()),
                "gradient_norm": float(gradient_norm.detach()),
                "top1": float((rank == 1).float().mean()),
                "mean_rank": float(rank.float().mean()),
            }
        )
        if step % 100 == 0 or step == STEPS:
            recent = rows[-100:]
            print(
                canonical_json(
                    {
                        "stage": "train_future_goal_matcher",
                        "step": step,
                        "steps": STEPS,
                        "loss": float(np.mean([row["loss"] for row in recent])),
                        "top1": float(np.mean([row["top1"] for row in recent])),
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
        "steps": STEPS,
        "batch_size": BATCH_SIZE,
        "recording_group_sample_counts": group_counts,
        "runtime_seconds": time.perf_counter() - started,
        "parameter_count": sum(parameter.numel() for parameter in matcher.parameters()),
        "world_model_trainable_parameters": sum(
            parameter.numel()
            for parameter in world_model.parameters()
            if parameter.requires_grad
        ),
    }


@torch.no_grad()
def evaluate(
    matcher: FutureGoalMatcher,
    world_model: HighResMultiLayerDualLatentWM,
    dataset: HighResDualLatentWindowDataset,
    device: torch.device,
) -> dict[str, Any]:
    matcher.eval()
    world_model.eval()
    learned_ranks = []
    raw_ranks = []
    learned_pairs = []
    raw_pairs = []
    group_rows: dict[str, list[tuple[int, int]]] = defaultdict(list)
    windows = 0
    started = time.perf_counter()
    for trajectory_index, trajectory in enumerate(dataset.trajectories):
        indices = dataset.window_indices_by_shard()[trajectory_index]
        for start in range(0, len(indices), BATCH_SIZE):
            local = indices[start : start + BATCH_SIZE]
            if len(local) < BATCH_SIZE:
                local = local + local[: BATCH_SIZE - len(local)]
            batch = collate_indices(dataset, np.asarray(local))
            (
                semantic_history,
                geometry_history,
                semantic_future,
                geometry_future,
                actions,
            ) = batch_to_device(batch, device)
            negative_actions = fixed_negative_actions(actions)
            candidate_actions = torch.cat((actions[:, None], negative_actions), dim=1)
            predicted_semantic, predicted_geometry = predict_candidates(
                world_model,
                semantic_history,
                geometry_history,
                candidate_actions,
            )
            with amp_context():
                learned = matcher(
                    predicted_semantic,
                    predicted_geometry,
                    semantic_future[:, -1],
                    geometry_future[:, -1],
                    semantic_history[:, -1],
                    geometry_history[:, -1],
                ).float()
            raw = raw_dual_distance(
                predicted_semantic,
                predicted_geometry,
                semantic_future[:, -1],
                geometry_future[:, -1],
            )
            keep = min(BATCH_SIZE, len(indices) - start)
            learned_rank = 1 + (
                learned[:keep, 1:] < learned[:keep, :1]
            ).sum(dim=1)
            raw_rank = 1 + (raw[:keep, 1:] < raw[:keep, :1]).sum(dim=1)
            learned_ranks.append(learned_rank.cpu())
            raw_ranks.append(raw_rank.cpu())
            learned_pairs.append(
                (learned[:keep, :1] < learned[:keep, 1:]).float().cpu()
            )
            raw_pairs.append((raw[:keep, :1] < raw[:keep, 1:]).float().cpu())
            group = recording_group(trajectory.trajectory_id)
            group_rows[group].extend(
                [
                    (int(left), int(right))
                    for left, right in zip(learned_rank.cpu(), raw_rank.cpu())
                ]
            )
            windows += keep
            if windows % 512 < keep:
                print(
                    canonical_json(
                        {
                            "stage": "evaluate_future_goal_matcher",
                            "windows": windows,
                            "elapsed_seconds": time.perf_counter() - started,
                        }
                    ),
                    flush=True,
                )
    learned_rank_array = torch.cat(learned_ranks).numpy()
    raw_rank_array = torch.cat(raw_ranks).numpy()
    learned_pairwise = float(torch.cat(learned_pairs).mean())
    raw_pairwise = float(torch.cat(raw_pairs).mean())
    groups = {
        group: {
            "windows": len(rows),
            "learned_top1": float(np.mean([row[0] == 1 for row in rows])),
            "raw_top1": float(np.mean([row[1] == 1 for row in rows])),
            "top1_improved": bool(
                np.mean([row[0] == 1 for row in rows])
                > np.mean([row[1] == 1 for row in rows])
            ),
        }
        for group, rows in sorted(group_rows.items())
    }
    return {
        "windows": windows,
        "candidates_including_true": 12,
        "learned_pairwise_action_win_rate": learned_pairwise,
        "raw_dual_nmse_pairwise_action_win_rate": raw_pairwise,
        "pairwise_improvement": learned_pairwise - raw_pairwise,
        "learned_retrieval_top1": float(np.mean(learned_rank_array == 1)),
        "raw_dual_nmse_retrieval_top1": float(np.mean(raw_rank_array == 1)),
        "top1_improvement": float(
            np.mean(learned_rank_array == 1) - np.mean(raw_rank_array == 1)
        ),
        "learned_mean_rank": float(learned_rank_array.mean()),
        "raw_dual_nmse_mean_rank": float(raw_rank_array.mean()),
        "recording_top1_improved_fraction": float(
            np.mean([row["top1_improved"] for row in groups.values()])
        ),
        "recording_group": groups,
        "runtime_seconds": time.perf_counter() - started,
    }


def load_world_model(device: torch.device) -> tuple[HighResMultiLayerDualLatentWM, dict[str, Any]]:
    payload = torch.load(FROZEN_LTWM, map_location="cpu", weights_only=False)
    require(payload.get("outcome_gradients") is False, "frozen LTWM outcome leakage")
    require(payload.get("sealed_test_opened") is False, "frozen LTWM test leakage")
    model = HighResMultiLayerDualLatentWM(
        HighResDualLatentWMConfig(**payload["config"])
    )
    model.load_state_dict(payload["model"], strict=True)
    model = model.eval().to(device).requires_grad_(False)
    return model, payload


def main() -> None:
    args = parse_args()
    require(torch.cuda.is_available(), "CUDA is required")
    output = args.output.resolve()
    result_path = output / "result.json"
    checkpoint_path = output / "future_goal_matcher_final.pt"
    require(not output.exists(), f"refusing to overwrite {output}")
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
    world_model, _ = load_world_model(device)
    matcher = FutureGoalMatcher().to(device)
    started = time.perf_counter()
    training = train(matcher, world_model, train_set, device)
    validation = evaluate(matcher, world_model, val_set, device)
    checks = {
        "absolute_pairwise": validation["learned_pairwise_action_win_rate"] >= 0.80,
        "pairwise_improvement": validation["pairwise_improvement"] >= 0.02,
        "absolute_top1": validation["learned_retrieval_top1"] >= 0.20,
        "top1_improvement": validation["top1_improvement"] >= 0.03,
        "recording_consistency": validation["recording_top1_improved_fraction"] >= 0.50,
    }
    go = all(checks.values())
    output.mkdir(parents=True, exist_ok=False)
    torch.save(
        {
            "model": matcher.state_dict(),
            "architecture": "FutureGoalMatcher_v1",
            "seed": SEED,
            "training": training,
            "validation": validation,
            "base_checkpoint_sha256": frozen_before,
            "train_recording_groups": sorted(train_groups),
            "validation_recording_groups": sorted(val_groups),
            "outcome_gradients": False,
            "sealed_test_opened": False,
        },
        checkpoint_path,
    )
    result = {
        "schema": "highres_dual_future_goal_matcher_result_v1",
        "status": "complete_single_outcome_free_route_disjoint_metric_run",
        "decision": (
            "GO_EXPORT_ONE_UNION65_MATCH_SCORE"
            if go
            else "NO_GO_LEARNED_FUTURE_GOAL_METRIC"
        ),
        "go": go,
        "checks": checks,
        "data": {
            "latent_root": str(LATENT_ROOT),
            "metadata_sha256": sha256_file(LATENT_ROOT / "metadata.json"),
            "splits_sha256": sha256_file(LATENT_ROOT / "splits.json"),
            "train_windows": len(train_set),
            "validation_windows": len(val_set),
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
            "world_model_trainable_parameters": training[
                "world_model_trainable_parameters"
            ],
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
