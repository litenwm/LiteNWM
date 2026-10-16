#!/usr/bin/env python3
"""Evaluate frozen H4 LTWM checkpoints on one strictly locked latent split."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from latent_window_utils import ShardAwareBatchSampler  # noqa: E402
from recon_latents.models_v2 import LiteTokenFormerConfig, LiteTokenFormerWMV2  # noqa: E402
from recon_latents.split_lock import (  # noqa: E402
    load_split_lock,
    validate_parent_split_reuse,
)
from train_litetokenformer_wm_v2_4ctx import (  # noqa: E402
    TokenSequenceStreamingDataset,
    amp_context,
    step_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--latent-root", type=Path, required=True)
    parser.add_argument("--split-lock", type=Path, required=True)
    parser.add_argument("--dataset", choices=("recon", "scand"), required=True)
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="Repeat for paired evaluation of multiple frozen checkpoints.",
    )
    parser.add_argument("--checkpoint-parent-split-lock", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--stream-cache-shards", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--amp", choices=("bf16", "fp16", "off"), default="bf16")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_checkpoints(values: list[str]) -> list[tuple[str, Path]]:
    result: list[tuple[str, Path]] = []
    labels: set[str] = set()
    for value in values:
        if "=" not in value:
            raise ValueError(f"checkpoint must be LABEL=PATH, got: {value}")
        label, raw_path = value.split("=", 1)
        if not label or label in labels:
            raise ValueError(f"checkpoint label must be non-empty and unique: {label!r}")
        path = Path(raw_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        labels.add(label)
        result.append((label, path))
    return result


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def validate_inputs(args: argparse.Namespace) -> tuple[str, dict[str, object]]:
    split_lock, split_lock_sha256 = load_split_lock(args.split_lock)
    metadata = json.loads((args.latent_root / "metadata.json").read_text(encoding="utf-8"))
    root_splits = json.loads((args.latent_root / "splits.json").read_text(encoding="utf-8"))
    if metadata.get("split_lock_sha256") != split_lock_sha256:
        raise ValueError("latent root is not bound to the active split lock")
    if metadata.get("split_lock_dataset") != args.dataset:
        raise ValueError("latent root dataset provenance does not match --dataset")
    expected_ids = split_lock[args.dataset]["splits"][args.split]
    if root_splits.get(args.split) != expected_ids:
        raise ValueError(f"latent root does not expose the exact locked {args.split} split")
    if args.split == "val" and root_splits.get("test"):
        raise ValueError("validation latent root exposes locked test trajectories")
    if args.split == "test" and (root_splits.get("train") or root_splits.get("val")):
        raise ValueError("test latent root must be a test-only view")
    return split_lock_sha256, metadata


def weighted_merge(
    rows: list[tuple[int, dict[str, list[float]]]],
) -> dict[str, list[float]]:
    total = sum(weight for weight, _ in rows)
    return {
        key: (
            sum(weight * np.asarray(row[key], dtype=np.float64) for weight, row in rows)
            / total
        ).tolist()
        for key in rows[0][1]
    }


def evaluate_checkpoint(
    label: str,
    checkpoint_path: Path,
    loader: DataLoader,
    args: argparse.Namespace,
    device: torch.device,
    split_lock_sha256: str,
    split_lock: dict[str, object],
) -> dict[str, object]:
    set_seed(args.seed + 9000)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_lock = payload.get("split_lock_sha256")
    checkpoint_lineage = None
    if checkpoint_lock != split_lock_sha256:
        if args.checkpoint_parent_split_lock is None:
            raise ValueError(
                f"{label}: checkpoint is not bound to the active split lock"
            )
        checkpoint_lineage = validate_parent_split_reuse(
            split_lock,
            parent_split_lock=args.checkpoint_parent_split_lock,
            artifact_split_lock_sha256=checkpoint_lock,
            unchanged_datasets=("recon",),
            preserved_datasets=("scand",),
        )
    if payload.get("temporal_protocol") != {"history_frames": 4, "future_frames": 4}:
        raise ValueError(f"{label}: checkpoint is not bound to the 4-history -> 4-future protocol")
    config = LiteTokenFormerConfig(**payload["config"])
    if config.horizon != 4:
        raise ValueError(f"{label}: checkpoint horizon is not 4")
    model = LiteTokenFormerWMV2(config).to(device)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()

    true_rows: list[tuple[int, dict[str, list[float]]]] = []
    reverse_rows: list[tuple[int, dict[str, list[float]]]] = []
    noisy_rows: list[tuple[int, dict[str, list[float]]]] = []
    finite = 0
    samples = 0
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if args.max_batches and batch_index >= args.max_batches:
                break
            history = batch["history_tokens"].to(device, non_blocking=True)
            target = batch["future_tokens"].to(device, non_blocking=True)
            actions = batch["actions"].to(device, non_blocking=True)
            noise_scale = actions.std(dim=(0, 1), keepdim=True).clamp_min(1e-4)
            with amp_context(args.amp):
                output = model(history, actions)
                reverse = model(history, actions.flip(1))
                noisy = model(
                    history,
                    actions + 0.5 * noise_scale * torch.randn_like(actions),
                )
            batch_size = int(history.shape[0])
            true_rows.append((batch_size, step_metrics(output["predicted_tokens"], target, history)))
            reverse_rows.append((batch_size, step_metrics(reverse["predicted_tokens"], target, history)))
            noisy_rows.append((batch_size, step_metrics(noisy["predicted_tokens"], target, history)))
            finite += int(torch.isfinite(output["predicted_tokens"]).all()) * batch_size
            samples += batch_size
    if not true_rows:
        raise RuntimeError("evaluation produced zero batches")
    true = weighted_merge(true_rows)
    reverse = weighted_merge(reverse_rows)
    noisy = weighted_merge(noisy_rows)
    return {
        "label": label,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_seed": payload.get("args", {}).get("seed"),
        "checkpoint_best_epoch": payload["history"][-1]["epoch"],
        "checkpoint_split_lineage": checkpoint_lineage or {
            "status": "PASS",
            "mode": "active_split_lock",
            "split_lock_sha256": split_lock_sha256,
        },
        "true": true,
        "reversed": reverse,
        "noisy": noisy,
        "reverse_nmse_gap_h4": reverse["nmse"][3] - true["nmse"][3],
        "noisy_nmse_gap_h4": noisy["nmse"][3] - true["nmse"][3],
        "finite_fraction": finite / samples,
        "batches": len(true_rows),
        "samples": samples,
        "runtime_seconds": time.perf_counter() - started,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for H4 checkpoint evaluation")
    if args.batch_size < 1 or args.max_batches < 0:
        raise ValueError("batch-size must be positive and max-batches must be non-negative")
    checkpoints = parse_checkpoints(args.checkpoint)
    split_lock, split_lock_sha256 = load_split_lock(args.split_lock)
    validated_sha256, metadata = validate_inputs(args)
    if validated_sha256 != split_lock_sha256:
        raise RuntimeError("split-lock changed during input validation")
    dataset = TokenSequenceStreamingDataset(
        args.latent_root,
        args.split,
        4,
        4,
        args.stream_cache_shards,
    )
    sampler = ShardAwareBatchSampler(
        dataset,
        args.batch_size,
        drop_last=False,
        seed=args.seed,
    )
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    device = torch.device("cuda")
    results = [
        evaluate_checkpoint(
            label,
            path,
            loader,
            args,
            device,
            split_lock_sha256,
            split_lock,
        )
        for label, path in checkpoints
    ]
    report = {
        "status": "PASS",
        "scientific_namespace": "behavior_trajectory_dino_token_prediction",
        "temporal_protocol": {"history_frames": 4, "future_frames": 4},
        "dataset": args.dataset,
        "split": args.split,
        "latent_root": str(args.latent_root),
        "latent_metadata_sha256": sha256_file(args.latent_root / "metadata.json"),
        "split_lock_sha256": split_lock_sha256,
        "num_windows": len(dataset),
        "batch_size": args.batch_size,
        "max_batches": args.max_batches,
        "seed": args.seed,
        "latent_provenance": {
            "model_sha256": metadata.get("model_sha256"),
            "adapter_checkpoint_sha256": metadata.get("adapter_checkpoint_sha256"),
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
