#!/usr/bin/env python3
"""Export exact per-seed H1-H4 features for the frozen SCAND union65 pool.

The existing cache stores mean/std/min/max across the three frozen LTWM seeds.
This label-free replay restores the seed identity over time and additionally
records direct candidate-to-formal rollout latent distances.  It uses the
already-frozen state latents and candidate actions, and cannot influence CEM.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from planning.costs import latent_trajectory_features  # noqa: E402
from recon_latents.models_v2 import (  # noqa: E402
    LiteTokenFormerConfig,
    LiteTokenFormerWMV2,
)


SEEDS = (1701, 1702, 1703)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--temporal-features", type=Path, required=True)
    parser.add_argument("--latents", type=Path, required=True)
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=ROOT / "checkpoints/project",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--candidate-chunk", type=int, default=17)
    return parser.parse_args()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {name: np.asarray(payload[name]) for name in payload.files}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def write_npz_atomic(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    temporal_path = args.temporal_features.resolve()
    latent_path = args.latents.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    temporal = load_npz(temporal_path)
    latents = load_npz(latent_path)
    actions = temporal["actions_h4_m_rad"].astype(np.float32)
    formal_index = temporal["formal_v70_union_index"].astype(np.int64)
    states, candidates = actions.shape[:2]
    if actions.shape != (states, candidates, 4, 3) or candidates != 65:
        raise RuntimeError(f"unexpected union action shape: {actions.shape}")
    if latents["history"].shape[:2] != (states, 4):
        raise RuntimeError("history/state binding changed")
    if latents["goal"].shape[0] != states:
        raise RuntimeError("goal/state binding changed")
    if formal_index.shape != (states,):
        raise RuntimeError("formal index binding changed")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    scalar_by_seed = np.empty(
        (states, candidates, len(SEEDS), 7, 4),
        dtype=np.float32,
    )
    rollout_pair_to_formal = np.empty(
        (states, candidates, len(SEEDS), 2, 4),
        dtype=np.float32,
    )
    checkpoint_hashes = []
    started = time.perf_counter()
    action_cpu = torch.from_numpy(actions)
    for seed_position, seed in enumerate(SEEDS):
        checkpoint_path = (
            args.checkpoint_root.resolve()
            / "ltwm_expansion"
            / f"seed{seed}"
            / "best.pt"
        )
        checkpoint_hash = sha256_file(checkpoint_path)
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        model = LiteTokenFormerWMV2(
            LiteTokenFormerConfig(**checkpoint["config"])
        ).to(device)
        model.load_state_dict(checkpoint["model"], strict=True)
        model.eval().requires_grad_(False)
        checkpoint_hashes.append(
            {
                "seed": seed,
                "path": str(checkpoint_path),
                "sha256": checkpoint_hash,
            }
        )
        for start in range(0, states, args.batch_size):
            end = min(start + args.batch_size, states)
            history = torch.from_numpy(
                latents["history"][start:end].astype(np.float32)
            ).to(device)
            goal = torch.from_numpy(
                latents["goal"][start:end].astype(np.float32)
            ).to(device)
            local_actions = action_cpu[start:end].to(device)
            local_formal_index = formal_index[start:end]
            formal_actions = local_actions[
                torch.arange(end - start, device=device),
                torch.as_tensor(local_formal_index, device=device),
            ][:, None]
            autocast = (
                torch.autocast("cuda", dtype=torch.bfloat16)
                if device.type == "cuda"
                else torch.autocast("cpu", enabled=False)
            )
            with torch.inference_mode(), autocast:
                formal_future = model(history, formal_actions)["predicted_tokens"]
            formal_future = formal_future.float()
            for candidate_start in range(0, candidates, args.candidate_chunk):
                candidate_end = min(
                    candidate_start + args.candidate_chunk,
                    candidates,
                )
                chunk = local_actions[:, candidate_start:candidate_end]
                with torch.inference_mode(), autocast:
                    future = model(history, chunk)["predicted_tokens"]
                future = future.float()
                with torch.inference_mode():
                    scalar, _ = latent_trajectory_features(
                        future,
                        goal,
                        history[:, -1],
                    )
                    formal_expanded = formal_future.expand_as(future)
                    formal_energy = formal_expanded.square().mean(
                        dim=(-1, -2)
                    ).clamp_min(1.0e-8)
                    pair_nmse = (
                        (future - formal_expanded)
                        .square()
                        .mean(dim=(-1, -2))
                        / formal_energy
                    )
                    pair_cosine = 1.0 - F.cosine_similarity(
                        future.flatten(-2),
                        formal_expanded.flatten(-2),
                        dim=-1,
                    )
                scalar_by_seed[
                    start:end,
                    candidate_start:candidate_end,
                    seed_position,
                ] = scalar.cpu().numpy().reshape(
                    end - start,
                    candidate_end - candidate_start,
                    7,
                    4,
                )
                rollout_pair_to_formal[
                    start:end,
                    candidate_start:candidate_end,
                    seed_position,
                    0,
                ] = pair_nmse.cpu().numpy()
                rollout_pair_to_formal[
                    start:end,
                    candidate_start:candidate_end,
                    seed_position,
                    1,
                ] = pair_cosine.cpu().numpy()
            print(
                canonical_json(
                    {
                        "seed": seed,
                        "completed_states": end,
                        "states": states,
                        "elapsed_seconds": time.perf_counter() - started,
                    }
                ),
                flush=True,
            )
        del model, checkpoint
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    cached = temporal["ltwm_temporal_summary"].astype(np.float32)
    reconstructed = np.stack(
        (
            scalar_by_seed.mean(axis=2),
            scalar_by_seed.std(axis=2),
            scalar_by_seed.min(axis=2),
            scalar_by_seed.max(axis=2),
        ),
        axis=2,
    )
    maximum_absolute_error = float(np.max(np.abs(reconstructed - cached)))
    if maximum_absolute_error > 5.0e-4:
        raise RuntimeError(
            "exact-seed replay does not reproduce frozen summaries: "
            f"max_abs_error={maximum_absolute_error}"
        )
    metadata = {
        "schema": "scand_union65_exact_seed_temporal_v1",
        "status": "label_free_complete",
        "states": states,
        "candidates": candidates,
        "seeds": list(SEEDS),
        "latent_metrics": [
            "goal_nmse",
            "goal_cosine",
            "current_nmse",
            "goal_progress",
            "step_energy",
            "step_goal_direction_cosine",
            "goal_norm_ratio",
        ],
        "horizons": ["h1", "h2", "h3", "h4"],
        "rollout_pair_channels": [
            "candidate_vs_formal_nmse",
            "candidate_vs_formal_cosine_distance",
        ],
        "summary_reproduction_max_abs_error": maximum_absolute_error,
        "temporal_features": {
            "path": str(temporal_path),
            "sha256": sha256_file(temporal_path),
        },
        "latents": {
            "path": str(latent_path),
            "sha256": sha256_file(latent_path),
        },
        "checkpoints": checkpoint_hashes,
        "labels_opened": False,
        "formal_v70_mutated": False,
        "sealed_validation_opened": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    write_npz_atomic(
        output,
        record_key_audit_only=temporal["record_key_audit_only"].astype(str),
        candidate_hash=temporal["candidate_hash"].astype(str),
        formal_v70_union_index=formal_index,
        scalar_by_seed=scalar_by_seed,
        rollout_pair_to_formal=rollout_pair_to_formal,
        metadata=np.asarray(canonical_json(metadata)),
    )
    print(
        canonical_json(
            {
                "status": metadata["status"],
                "output": str(output),
                "sha256": sha256_file(output),
                "summary_reproduction_max_abs_error": maximum_absolute_error,
            }
        )
    )


if __name__ == "__main__":
    main()
