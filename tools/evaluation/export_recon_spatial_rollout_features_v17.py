#!/usr/bin/env python3
"""Export outcome-free candidate-conditioned spatial LTWM features."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from recon_latents.models_v2 import (  # noqa: E402
    LiteTokenFormerConfig,
    LiteTokenFormerWMV2,
)
from tools.evaluation.export_recon_dense_dinov3_features_v1 import (  # noqa: E402
    checkpoint_pairs,
)
from tools.scoring.train_recon_dense_meta_dinov3_v1 import (  # noqa: E402
    file_sha256,
    write_json,
    write_npz,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-features", type=Path, required=True)
    parser.add_argument("--latent-cache", type=Path, required=True)
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=REPO_ROOT / "checkpoints/project",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--candidate-chunk", type=int, default=4)
    return parser.parse_args()


def spatial_rollout_feature(
    future: torch.Tensor,
    goal: torch.Tensor,
) -> torch.Tensor:
    """Return [B,K,129] spatial future-to-goal alignment features."""
    if future.ndim != 5 or future.shape[-2:] != (256, 384):
        raise RuntimeError("future token contract changed")
    future_norm = F.normalize(future, dim=-1)
    goal_norm = F.normalize(goal, dim=-1)
    goal_expanded = goal_norm[:, None, None]
    same = (future_norm * goal_expanded).sum(-1).reshape(
        *future.shape[:3], 16, 16
    )
    same_pool = same.reshape(
        *future.shape[:3], 4, 4, 4, 4
    ).mean(dim=(-3, -1)).flatten(2)

    source_y, source_x = torch.div(
        torch.arange(256, device=future.device),
        16,
        rounding_mode="floor",
    ), torch.arange(256, device=future.device) % 16
    match_values = []
    for horizon in range(4):
        similarity = torch.einsum(
            "bkid,bjd->bkij",
            future_norm[:, :, horizon],
            goal_norm,
        )
        top = similarity.topk(k=2, dim=-1).values
        maximum = top[..., 0]
        target = similarity.argmax(dim=-1)
        target_y = torch.div(target, 16, rounding_mode="floor")
        target_x = target % 16
        dx = target_x.float() - source_x[None, None].float()
        dy = target_y.float() - source_y[None, None].float()
        match_values.append(
            torch.stack(
                (
                    maximum.mean(-1),
                    maximum.std(-1),
                    torch.quantile(maximum, 0.10, dim=-1),
                    torch.quantile(maximum, 0.50, dim=-1),
                    torch.quantile(maximum, 0.90, dim=-1),
                    dx.mean(-1),
                    dx.std(-1),
                    dy.mean(-1),
                    dy.std(-1),
                    (top[..., 0] - top[..., 1]).mean(-1),
                ),
                dim=-1,
            )
        )
    match = torch.cat(match_values, dim=-1)

    terminal = future_norm[:, :, -1]
    offsets = []
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            source_indices = []
            target_indices = []
            for y in range(16):
                target_y = y + dy
                if not 0 <= target_y < 16:
                    continue
                for x in range(16):
                    target_x = x + dx
                    if 0 <= target_x < 16:
                        source_indices.append(y * 16 + x)
                        target_indices.append(target_y * 16 + target_x)
            source_index = torch.as_tensor(
                source_indices, device=future.device
            )
            target_index = torch.as_tensor(
                target_indices, device=future.device
            )
            value = (
                terminal[:, :, source_index]
                * goal_norm[:, None, target_index]
            ).sum(-1).mean(-1)
            offsets.append(value)
    offset = torch.stack(offsets, dim=-1)
    result = torch.cat((same_pool, match, offset), dim=-1)
    if result.shape[-1] != 129 or not torch.isfinite(result).all():
        raise RuntimeError(
            f"spatial rollout feature contract failed: {result.shape}"
        )
    return result


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    with np.load(args.base_features, allow_pickle=False) as payload:
        actions = payload["actions_h4"].astype(np.float32)
        record_key = payload["record_key"].astype(str)
    with np.load(args.latent_cache, allow_pickle=False) as payload:
        history = payload["history"].astype(np.float32)
        goal = payload["goal"].astype(np.float32)
    if (
        actions.shape[0] != len(history)
        or history.shape[1:] != (4, 256, 384)
        or goal.shape[1:] != (256, 384)
    ):
        raise RuntimeError("spatial rollout input identity changed")

    seed_features = []
    checkpoint_records = []
    started = time.perf_counter()
    for pair in checkpoint_pairs(args.checkpoint_root):
        checkpoint = torch.load(
            pair["wm"], map_location="cpu", weights_only=False
        )
        model = LiteTokenFormerWMV2(
            LiteTokenFormerConfig(**checkpoint["config"])
        ).to(device)
        model.load_state_dict(checkpoint["model"], strict=True)
        model.eval().requires_grad_(False)
        state_values = []
        for start in range(0, len(actions), args.batch_size):
            end = min(start + args.batch_size, len(actions))
            history_batch = torch.from_numpy(history[start:end]).to(device)
            goal_batch = torch.from_numpy(goal[start:end]).to(device)
            action_batch = torch.from_numpy(actions[start:end]).to(device)
            candidate_values = []
            for candidate_start in range(
                0, actions.shape[1], args.candidate_chunk
            ):
                candidate_end = min(
                    candidate_start + args.candidate_chunk,
                    actions.shape[1],
                )
                chunk = action_batch[:, candidate_start:candidate_end]
                autocast = (
                    torch.autocast("cuda", dtype=torch.bfloat16)
                    if device.type == "cuda"
                    else torch.autocast("cpu", enabled=False)
                )
                with torch.inference_mode(), autocast:
                    future = model(history_batch, chunk)[
                        "predicted_tokens"
                    ].float()
                    feature = spatial_rollout_feature(
                        future, goal_batch
                    )
                candidate_values.append(feature.cpu())
            state_values.append(torch.cat(candidate_values, dim=1).numpy())
            print(
                json.dumps(
                    {
                        "stage": "spatial_rollout_v17",
                        "seed": pair["seed"],
                        "completed_states": end,
                        "total_states": len(actions),
                        "elapsed_sec": time.perf_counter() - started,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        seed_features.append(
            np.concatenate(state_values).astype(np.float32)
        )
        checkpoint_records.append(
            {
                "seed": pair["seed"],
                "wm": str(pair["wm"]),
                "wm_sha256": pair["wm_sha256"],
            }
        )
        del model, checkpoint
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    by_seed = np.stack(seed_features)
    feature = np.concatenate(
        (by_seed.mean(0), by_seed.std(0)), axis=-1
    ).astype(np.float32)
    metadata = {
        "schema": "recon_spatial_rollout_features_v17",
        "states": int(len(actions)),
        "candidates": int(actions.shape[1]),
        "per_seed_dim": 129,
        "feature_dim": 258,
        "feature_contract": (
            "candidate-conditioned LTWM future-to-goal same-patch maps, "
            "token-match displacement/gap statistics, and terminal 5x5 "
            "offset correlations; aggregated as three-seed mean/std"
        ),
        "outcomes_accessed": False,
        "labels_in_output": False,
        "online_deployable": True,
        "base_features": str(args.base_features),
        "base_features_sha256": file_sha256(args.base_features),
        "latent_cache": str(args.latent_cache),
        "latent_cache_sha256": file_sha256(args.latent_cache),
        "checkpoints": checkpoint_records,
    }
    write_npz(
        args.output,
        record_key=record_key,
        spatial_rollout_features=feature,
        spatial_rollout_by_seed=by_seed.astype(np.float32),
        metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    audit_path = args.output.with_suffix(".audit.json")
    metadata["output"] = str(args.output)
    metadata["output_sha256"] = file_sha256(args.output)
    write_json(audit_path, metadata)
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
