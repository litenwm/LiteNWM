#!/usr/bin/env python3
"""Train one high-resolution multi-layer dual-latent LTWM on strict routes."""

from __future__ import annotations

import argparse
import copy
from contextlib import nullcontext
from dataclasses import asdict
import json
from pathlib import Path
import random
import sys
import time
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from latent_window_utils import ShardAwareBatchSampler  # noqa: E402
from recon_latents.dataset_highres_dual_v1 import (  # noqa: E402
    HighResDualLatentWindowDataset,
)
from recon_latents.losses_highres_dual_v1 import (  # noqa: E402
    HighResDualLatentLossWeights,
    compute_highres_dual_latent_loss,
)
from recon_latents.models_highres_dual_v1 import (  # noqa: E402
    HighResDualLatentWMConfig,
    HighResMultiLayerDualLatentWM,
)
from recon_latents.split_lock import file_sha256, load_split_lock  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latent-root", type=Path, required=True)
    parser.add_argument("--scand-latent-root", type=Path, default=None)
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
    parser.add_argument("--recon-batches-per-scand", type=int, default=1)
    parser.add_argument("--model-dim", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--action-layers", type=int, default=2)
    parser.add_argument("--rollout-blocks", type=int, default=3)
    parser.add_argument("--history-frames", type=int, choices=(4,), default=4)
    parser.add_argument("--future-frames", type=int, choices=(4,), default=4)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def amp_context(amp: str):
    if amp == "off":
        return nullcontext()
    dtype = torch.bfloat16 if amp == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def corrupt_actions(actions: torch.Tensor, index: int) -> torch.Tensor:
    mode = index % 4
    if mode == 0:
        return actions.flip(1)
    if mode == 1:
        result = actions.clone()
        result[..., 2].neg_()
        return result
    if mode == 2:
        return actions[:, :1].expand_as(actions)
    scale = actions.std(dim=(0, 1), keepdim=True).clamp_min(1.0e-4)
    return actions + 0.5 * scale * torch.randn_like(actions)


def validate_dual_latent_root(
    root: Path,
    *,
    dataset: str,
    split_lock: dict[str, object],
    split_lock_sha256: str,
) -> dict[str, object]:
    metadata_path = root / "metadata.json"
    splits_path = root / "splits.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    splits = json.loads(splits_path.read_text(encoding="utf-8"))
    if metadata.get("split_lock_sha256") != split_lock_sha256:
        raise ValueError(f"latent root is not bound to active split lock: {root}")
    if splits.get("test"):
        raise ValueError(f"training latent root exposes sealed test routes: {root}")
    expected = split_lock[dataset]["splits"]
    for split in ("train", "val"):
        if set(map(str, splits.get(split, ()))) != set(
            map(str, expected.get(split, ()))
        ):
            raise ValueError(f"{root} strict {split} routes do not match split lock")
    representation = metadata.get("representation", {})
    required = {
        "grid_height",
        "grid_width",
        "semantic_dim",
        "geometry_dim",
        "geometry_projection_sha256",
    }
    if missing := required.difference(representation):
        raise ValueError(f"{root} missing representation fields: {sorted(missing)}")
    expected_ids = {
        str(value)
        for split in ("train", "val")
        for value in splits.get(split, ())
    }
    rows = {
        str(row["trajectory_id"]): row for row in metadata.get("trajectories", ())
    }
    if set(rows) != expected_ids:
        raise ValueError(f"{root} metadata routes do not match train/val split")
    for row in rows.values():
        if not (root / "shards" / str(row["shard_file"])).is_file():
            raise FileNotFoundError(f"missing latent shard: {row['shard_file']}")
    return {
        "latent_root": str(root),
        "metadata_sha256": file_sha256(metadata_path),
        "splits_sha256": file_sha256(splits_path),
        "trajectory_count": len(rows),
        "frame_count": sum(int(row["num_frames"]) for row in rows.values()),
        "representation": representation,
        "status": "PASS",
    }


class TaggedLoader:
    def __init__(self, loader: DataLoader, domain: str) -> None:
        self.loader = loader
        self.domain = domain

    def __iter__(self) -> Iterator[dict[str, object]]:
        for batch in self.loader:
            yield {**batch, "domain": self.domain}


class AlternatingDomainLoader:
    def __init__(
        self,
        recon: DataLoader,
        scand: DataLoader,
        recon_batches_per_scand: int,
    ) -> None:
        if recon_batches_per_scand < 1:
            raise ValueError("recon-batches-per-scand must be positive")
        self.loaders = {"recon": recon, "scand": scand}
        self.schedule = ["recon"] * recon_batches_per_scand + ["scand"]

    def __iter__(self) -> Iterator[dict[str, object]]:
        iterators = {name: iter(loader) for name, loader in self.loaders.items()}
        while True:
            for domain in self.schedule:
                try:
                    batch = next(iterators[domain])
                except StopIteration:
                    iterators[domain] = iter(self.loaders[domain])
                    batch = next(iterators[domain])
                yield {**batch, "domain": domain}


def make_domain_loaders(
    root: Path,
    args: argparse.Namespace,
    seed_offset: int,
) -> tuple[
    HighResDualLatentWindowDataset,
    HighResDualLatentWindowDataset,
    DataLoader,
    DataLoader,
]:
    train = HighResDualLatentWindowDataset(
        root,
        "train",
        args.history_frames,
        args.future_frames,
        args.stream_cache_shards,
    )
    val = HighResDualLatentWindowDataset(
        root,
        "val",
        args.history_frames,
        args.future_frames,
        args.stream_cache_shards,
    )
    if not len(train) or not len(val):
        raise ValueError(f"empty train or validation windows in {root}")
    train_loader = DataLoader(
        train,
        batch_sampler=ShardAwareBatchSampler(
            train,
            args.batch_size,
            drop_last=True,
            seed=args.seed + seed_offset,
        ),
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val,
        batch_sampler=ShardAwareBatchSampler(
            val,
            args.eval_batch_size,
            drop_last=False,
            seed=10000 + args.seed + seed_offset,
        ),
        num_workers=args.num_workers,
        pin_memory=True,
    )
    return train, val, train_loader, val_loader


def branch_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    current: torch.Tensor,
) -> dict[str, list[float]]:
    prediction = prediction[:, 0].float()
    target = target.float()
    current = current[:, -1].float()
    mse = (prediction - target).square().mean(dim=(-1, -2))
    denominator = target.square().mean(dim=(-1, -2)).clamp_min(1.0e-8)
    nmse = (mse / denominator).mean(dim=0)
    cosine = (
        1.0
        - torch.nn.functional.cosine_similarity(
            prediction.flatten(-2),
            target.flatten(-2),
            dim=-1,
        )
    ).mean(dim=0)
    pred_delta = prediction - current[:, None]
    target_delta = target - current[:, None]
    norm_ratio = (
        torch.linalg.vector_norm(pred_delta.flatten(-2), dim=-1)
        / torch.linalg.vector_norm(
            target_delta.flatten(-2),
            dim=-1,
        ).clamp_min(1.0e-8)
    ).mean(dim=0)
    return {
        "nmse": nmse.tolist(),
        "cosine": cosine.tolist(),
        "delta_norm_ratio": norm_ratio.tolist(),
    }


def average_metric_rows(
    rows: list[tuple[int, dict[str, list[float]]]],
) -> dict[str, list[float]]:
    total = sum(weight for weight, _ in rows)
    return {
        key: (
            sum(
                weight * np.asarray(metrics[key], dtype=np.float64)
                for weight, metrics in rows
            )
            / total
        ).tolist()
        for key in rows[0][1]
    }


def evaluate(
    model: HighResMultiLayerDualLatentWM,
    loader: DataLoader,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, object]:
    model.eval()
    semantic_rows: list[tuple[int, dict[str, list[float]]]] = []
    geometry_rows: list[tuple[int, dict[str, list[float]]]] = []
    corrupt_semantic_rows: list[tuple[int, dict[str, list[float]]]] = []
    corrupt_geometry_rows: list[tuple[int, dict[str, list[float]]]] = []
    losses: list[float] = []
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if batch_index >= args.eval_batches_per_epoch:
                break
            semantic_history = batch["semantic_history"].to(device, non_blocking=True)
            geometry_history = batch["geometry_history"].to(device, non_blocking=True)
            semantic_future = batch["semantic_future"].to(device, non_blocking=True)
            geometry_future = batch["geometry_future"].to(device, non_blocking=True)
            actions = batch["actions"].to(device, non_blocking=True)
            with amp_context(args.amp):
                output = model(semantic_history, geometry_history, actions)
                corrupted = model(
                    semantic_history,
                    geometry_history,
                    corrupt_actions(actions, batch_index),
                )
                loss, _ = compute_highres_dual_latent_loss(
                    output,
                    semantic_future,
                    geometry_future,
                    semantic_history,
                    geometry_history,
                    actions,
                )
            weight = int(actions.shape[0])
            losses.append(float(loss))
            semantic_rows.append(
                (
                    weight,
                    branch_metrics(
                        output["predicted_semantic_tokens"],
                        semantic_future,
                        semantic_history,
                    ),
                )
            )
            geometry_rows.append(
                (
                    weight,
                    branch_metrics(
                        output["predicted_geometry_tokens"],
                        geometry_future,
                        geometry_history,
                    ),
                )
            )
            corrupt_semantic_rows.append(
                (
                    weight,
                    branch_metrics(
                        corrupted["predicted_semantic_tokens"],
                        semantic_future,
                        semantic_history,
                    ),
                )
            )
            corrupt_geometry_rows.append(
                (
                    weight,
                    branch_metrics(
                        corrupted["predicted_geometry_tokens"],
                        geometry_future,
                        geometry_history,
                    ),
                )
            )
    if not losses:
        raise RuntimeError("evaluation produced zero batches")
    semantic = average_metric_rows(semantic_rows)
    geometry = average_metric_rows(geometry_rows)
    corrupt_semantic = average_metric_rows(corrupt_semantic_rows)
    corrupt_geometry = average_metric_rows(corrupt_geometry_rows)
    return {
        "loss": float(np.mean(losses)),
        "semantic": semantic,
        "geometry": geometry,
        "corrupted_semantic": corrupt_semantic,
        "corrupted_geometry": corrupt_geometry,
        "corrupted_nmse_gap_h4": 0.5
        * (
            corrupt_semantic["nmse"][-1]
            - semantic["nmse"][-1]
            + corrupt_geometry["nmse"][-1]
            - geometry["nmse"][-1]
        ),
        "batches": len(losses),
        "samples": sum(weight for weight, _ in semantic_rows),
        "runtime_seconds": time.perf_counter() - started,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
    }


def train_epoch(
    model: HighResMultiLayerDualLatentWM,
    loader,
    optimizer: torch.optim.Optimizer,
    scaler,
    args: argparse.Namespace,
    device: torch.device,
    epoch: int,
) -> dict[str, object]:
    model.train()
    rows: list[dict[str, float]] = []
    domain_batches: dict[str, int] = {}
    started = time.perf_counter()
    for batch_index, batch in enumerate(loader):
        if batch_index >= args.batches_per_epoch:
            break
        domain = str(batch.get("domain", "recon"))
        domain_batches[domain] = domain_batches.get(domain, 0) + 1
        semantic_history = batch["semantic_history"].to(device, non_blocking=True)
        geometry_history = batch["geometry_history"].to(device, non_blocking=True)
        semantic_future = batch["semantic_future"].to(device, non_blocking=True)
        geometry_future = batch["geometry_future"].to(device, non_blocking=True)
        actions = batch["actions"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with amp_context(args.amp):
            output = model(semantic_history, geometry_history, actions)
            corrupted_output = None
            if args.contrastive_every and batch_index % args.contrastive_every == 0:
                corrupted_output = model(
                    semantic_history,
                    geometry_history,
                    corrupt_actions(actions, batch_index // args.contrastive_every),
                )
            loss, components = compute_highres_dual_latent_loss(
                output,
                semantic_future,
                geometry_future,
                semantic_history,
                geometry_history,
                actions,
                weights=HighResDualLatentLossWeights(),
                corrupted_output=corrupted_output,
            )
        if scaler is None:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
        else:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        rows.append(
            {
                "loss": float(loss.detach()),
                **{
                    name: float(value.detach())
                    for name, value in components.items()
                },
            }
        )
        if (batch_index + 1) % 100 == 0:
            print(
                json.dumps(
                    {
                        "epoch": epoch,
                        "batch": batch_index + 1,
                        "loss": float(
                            np.mean([row["loss"] for row in rows[-100:]])
                        ),
                    }
                ),
                flush=True,
            )
    if not rows:
        raise RuntimeError("training produced zero batches")
    return {
        **{
            key: float(np.mean([row[key] for row in rows]))
            for key in rows[0]
        },
        "batches": len(rows),
        "domain_batches": domain_batches,
        "runtime_seconds": time.perf_counter() - started,
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.batch_size < 1 or args.eval_batch_size < 1:
        raise ValueError("batch sizes must be positive")
    split_lock, split_lock_sha = load_split_lock(args.split_lock)
    roots = [("recon", args.latent_root)]
    if args.scand_latent_root is not None:
        roots.append(("scand", args.scand_latent_root))
    root_reports = {
        dataset: validate_dual_latent_root(
            root,
            dataset=dataset,
            split_lock=split_lock,
            split_lock_sha256=split_lock_sha,
        )
        for dataset, root in roots
    }
    contracts = [report["representation"] for report in root_reports.values()]
    contract_keys = (
        "grid_height",
        "grid_width",
        "semantic_dim",
        "geometry_dim",
        "geometry_projection_sha256",
    )
    if any(
        any(contract[key] != contracts[0][key] for key in contract_keys)
        for contract in contracts[1:]
    ):
        raise ValueError("RECON and SCAND dual-latent contracts differ")

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    train_datasets: dict[str, HighResDualLatentWindowDataset] = {}
    val_datasets: dict[str, HighResDualLatentWindowDataset] = {}
    val_loaders: dict[str, DataLoader] = {}
    recon_train, recon_val, recon_train_loader, recon_val_loader = (
        make_domain_loaders(args.latent_root, args, 0)
    )
    train_datasets["recon"] = recon_train
    val_datasets["recon"] = recon_val
    val_loaders["recon"] = recon_val_loader
    if args.scand_latent_root is None:
        train_loader = TaggedLoader(recon_train_loader, "recon")
    else:
        scand_train, scand_val, scand_train_loader, scand_val_loader = (
            make_domain_loaders(args.scand_latent_root, args, 100000)
        )
        train_datasets["scand"] = scand_train
        val_datasets["scand"] = scand_val
        val_loaders["scand"] = scand_val_loader
        train_loader = AlternatingDomainLoader(
            recon_train_loader,
            scand_train_loader,
            args.recon_batches_per_scand,
        )

    contract = contracts[0]
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
    history: list[dict[str, object]] = []
    best = None
    best_score = float("inf")
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            args,
            device,
            epoch,
        )
        val_metrics = {
            domain: evaluate(model, loader, args, device)
            for domain, loader in val_loaders.items()
        }
        score = float(
            np.mean(
                [
                    0.5
                    * (
                        metrics["semantic"]["nmse"][-1]
                        + metrics["geometry"]["nmse"][-1]
                    )
                    for metrics in val_metrics.values()
                ]
            )
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
            "latent_roots": root_reports,
            "representation_contract": contract,
            "scientific_namespace": (
                "real_future_action_conditioned_dual_dinov3_latent_prediction"
            ),
            "outcome_gradients": False,
            "camera_conditioning": False,
        }
        epoch_path = args.checkpoint_dir / f"epoch_{epoch:03d}.pt"
        torch.save(payload, epoch_path)
        if score < best_score:
            best_score = score
            best = copy.deepcopy(payload)
            torch.save(best, args.checkpoint_dir / "best.pt")
        print(
            json.dumps(
                {
                    "epoch_complete": epoch,
                    "score": score,
                    "h4_nmse": {
                        domain: {
                            "semantic": metrics["semantic"]["nmse"][-1],
                            "geometry": metrics["geometry"]["nmse"][-1],
                            "corrupt_gap": metrics["corrupted_nmse_gap_h4"],
                        }
                        for domain, metrics in val_metrics.items()
                    },
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
        "config": asdict(config),
        "loss_weights": asdict(HighResDualLatentLossWeights()),
        "parameter_count": payload["parameter_count"],
        "latent_roots": root_reports,
        "train_dataset": {
            domain: dataset.summary()
            for domain, dataset in train_datasets.items()
        },
        "val_dataset": {
            domain: dataset.summary() for domain, dataset in val_datasets.items()
        },
        "history": history,
        "best_score": best_score,
        "best_checkpoint": str(args.checkpoint_dir / "best.pt"),
        "best_checkpoint_sha256": file_sha256(args.checkpoint_dir / "best.pt"),
        "final_checkpoint": str(args.checkpoint_dir / "final.pt"),
        "final_checkpoint_sha256": file_sha256(args.checkpoint_dir / "final.pt"),
        "split_lock_sha256": split_lock_sha,
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
        f"Status: **{summary['status']}**\n\n"
        f"Parameters: {summary['parameter_count']:,}\n\n"
        f"Best latent validation score: {summary['best_score']:.6f}\n\n"
        f"Best checkpoint SHA-256: `{summary['best_checkpoint_sha256']}`\n\n"
        "The objective is real future DINOv3 semantic/geometry latent prediction "
        "only. No ATE, navigation objective, camera calibration, or sealed-test "
        "signal enters training.\n",
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
