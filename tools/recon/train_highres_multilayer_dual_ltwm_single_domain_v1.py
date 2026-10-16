#!/usr/bin/env python3
"""Train the dual-latent LTWM on one explicitly named strict dataset."""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import recon_latents.losses_highres_dual_stable_v1  # noqa: E402,F401
import train_highres_multilayer_dual_ltwm_v1 as shared  # noqa: E402
from recon_latents.losses_highres_dual_v1 import (  # noqa: E402
    HighResDualLatentLossWeights,
)
from recon_latents.models_highres_dual_v1 import (  # noqa: E402
    HighResDualLatentWMConfig,
    HighResMultiLayerDualLatentWM,
)
from recon_latents.split_lock import file_sha256, load_split_lock  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latent-root", type=Path, required=True)
    parser.add_argument("--dataset", choices=("recon", "scand"), required=True)
    parser.add_argument("--split-lock", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batches-per-epoch", type=int, default=1000)
    parser.add_argument("--eval-batches-per-epoch", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--stream-cache-shards", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--amp", choices=("bf16", "fp16", "off"), default="bf16")
    parser.add_argument("--contrastive-every", type=int, default=4)
    parser.add_argument("--model-dim", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--action-layers", type=int, default=2)
    parser.add_argument("--rollout-blocks", type=int, default=3)
    parser.add_argument("--history-frames", type=int, choices=(4,), default=4)
    parser.add_argument("--future-frames", type=int, choices=(4,), default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    shared.set_seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    split_lock, split_lock_sha = load_split_lock(args.split_lock)
    root_report = shared.validate_dual_latent_root(
        args.latent_root,
        dataset=args.dataset,
        split_lock=split_lock,
        split_lock_sha256=split_lock_sha,
    )
    contract = root_report["representation"]
    train_dataset, val_dataset, train_loader_raw, val_loader = (
        shared.make_domain_loaders(args.latent_root, args, 0)
    )
    train_loader = shared.TaggedLoader(train_loader_raw, args.dataset)
    config = HighResDualLatentWMConfig(
        semantic_dim=int(contract["semantic_dim"]),
        geometry_dim=int(contract["geometry_dim"]),
        grid_height=int(contract["grid_height"]),
        grid_width=int(contract["grid_width"]),
        history_frames=args.history_frames,
        horizon=args.future_frames,
        model_dim=args.model_dim,
        heads=args.heads,
        action_layers=args.action_layers,
        rollout_blocks=args.rollout_blocks,
    )
    device = torch.device("cuda")
    model = HighResMultiLayerDualLatentWM(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda") if args.amp == "fp16" else None
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, object]] = []
    best = None
    best_score = float("inf")
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        train_metrics = shared.train_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            args,
            device,
            epoch,
        )
        val_metrics = shared.evaluate(
            model,
            val_loader,
            args,
            device,
        )
        score = 0.5 * (
            val_metrics["semantic"]["nmse"][-1]
            + val_metrics["geometry"]["nmse"][-1]
        )
        row = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
            "selection_score": score,
            "elapsed_seconds": time.perf_counter() - started,
        }
        history.append(row)
        payload = {
            "model": model.state_dict(),
            "config": asdict(config),
            "args": vars(args),
            "loss_weights": asdict(HighResDualLatentLossWeights()),
            "history": history,
            "parameter_count": sum(
                parameter.numel()
                for parameter in model.parameters()
                if parameter.requires_grad
            ),
            "split_lock_sha256": split_lock_sha,
            "latent_root": root_report,
            "representation_contract": contract,
            "dataset": args.dataset,
            "scientific_namespace": (
                "real_future_action_conditioned_dual_dinov3_latent_prediction"
            ),
            "stable_zero_motion_cosine": True,
            "outcome_gradients": False,
            "camera_conditioning": False,
            "sealed_test_opened": False,
        }
        torch.save(payload, args.checkpoint_dir / f"epoch_{epoch:03d}.pt")
        if np.isfinite(score) and score < best_score:
            best_score = float(score)
            best = copy.deepcopy(payload)
            torch.save(best, args.checkpoint_dir / "best.pt")
        print(
            json.dumps(
                {
                    "epoch_complete": epoch,
                    "selection_score": score,
                    "semantic_h4_nmse": val_metrics["semantic"]["nmse"][-1],
                    "geometry_h4_nmse": val_metrics["geometry"]["nmse"][-1],
                    "corrupted_nmse_gap_h4": val_metrics[
                        "corrupted_nmse_gap_h4"
                    ],
                }
            ),
            flush=True,
        )
    if best is None:
        raise RuntimeError("no finite checkpoint was produced")
    torch.save(payload, args.checkpoint_dir / "final.pt")
    summary = {
        "status": "complete",
        "model": "HighRes-MultiLayer-DualLatent-LTWM-v1",
        "dataset": args.dataset,
        "config": asdict(config),
        "loss_weights": asdict(HighResDualLatentLossWeights()),
        "parameter_count": payload["parameter_count"],
        "latent_root": root_report,
        "train_dataset": train_dataset.summary(),
        "val_dataset": val_dataset.summary(),
        "history": history,
        "best_score": best_score,
        "best_checkpoint": str(args.checkpoint_dir / "best.pt"),
        "best_checkpoint_sha256": file_sha256(args.checkpoint_dir / "best.pt"),
        "final_checkpoint": str(args.checkpoint_dir / "final.pt"),
        "final_checkpoint_sha256": file_sha256(args.checkpoint_dir / "final.pt"),
        "split_lock_sha256": split_lock_sha,
        "stable_zero_motion_cosine": True,
        "sealed_test_opened": False,
        "outcome_gradients": False,
        "camera_conditioning": False,
        "runtime_seconds": time.perf_counter() - started,
    }
    summary_path = args.report_dir / "train_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, default=str),
        encoding="utf-8",
    )
    (args.report_dir / "train_summary.md").write_text(
        "# High-Resolution Multi-Layer Dual-Latent LTWM v1\n\n"
        f"Dataset: **{args.dataset} train/val only**\n\n"
        f"Parameters: {summary['parameter_count']:,}\n\n"
        f"Best H4 dual-NMSE score: {best_score:.6f}\n\n"
        f"Best checkpoint SHA-256: `{summary['best_checkpoint_sha256']}`\n\n"
        "No ATE, navigation objective, camera calibration, or sealed-test signal "
        "enters this checkpoint.\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "best_checkpoint": summary["best_checkpoint"],
                "best_checkpoint_sha256": summary["best_checkpoint_sha256"],
                "best_score": best_score,
                "summary_sha256": file_sha256(summary_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
