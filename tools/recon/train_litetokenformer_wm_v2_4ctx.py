#!/usr/bin/env python3
"""Train LiteTokenFormer-WM-v2-4ctx on behavior H4 DINO token windows."""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from latent_window_utils import ShardAwareBatchSampler, StreamingLatentWindowDataset  # noqa: E402
from recon_latents.losses_v2 import V2LossWeights, compute_world_model_v2_loss  # noqa: E402
from recon_latents.models_v2 import LiteTokenFormerConfig, LiteTokenFormerWMV2, count_trainable_parameters  # noqa: E402
from recon_latents.split_lock import (  # noqa: E402
    file_sha256,
    load_split_lock,
    validate_latent_root,
    validate_parent_split_reuse,
)


class TokenSequenceStreamingDataset(StreamingLatentWindowDataset):
    """Return four historical and four future dense tokens without channel transpose."""

    def __getitem__(self, index: int) -> dict[str, object]:
        traj_index, history_start = self.windows[index]
        traj = self.trajectories[traj_index]
        z, actions_all = self._load_shard_cached(traj_index)
        t = history_start + self.history_len - 1
        history = np.ascontiguousarray(z[history_start : t + 1].reshape(self.history_len, 256, 384), dtype=np.float32)
        future = np.ascontiguousarray(z[t + 1 : t + self.horizon + 1].reshape(self.horizon, 256, 384), dtype=np.float32)
        actions = np.ascontiguousarray(actions_all[t : t + self.horizon], dtype=np.float32)
        return {
            "history_tokens": torch.from_numpy(history),
            "future_tokens": torch.from_numpy(future),
            "actions": torch.from_numpy(actions),
            "trajectory_id": traj.trajectory_id,
            "timestep": int(t),
        }


class TaggedLoader:
    def __init__(self, loader: DataLoader, domain: str) -> None:
        self.loader = loader
        self.domain = domain

    def __iter__(self):
        for batch in self.loader:
            yield {**batch, "domain": self.domain}


class AlternatingDomainLoader:
    """Cycle deterministic shard-aware loaders without mixing domains in a batch."""

    def __init__(
        self,
        recon_loader: DataLoader,
        scand_loader: DataLoader,
        recon_batches_per_scand: int,
    ) -> None:
        if recon_batches_per_scand < 1:
            raise ValueError("recon_batches_per_scand must be >= 1")
        self.loaders = {"recon": recon_loader, "scand": scand_loader}
        self.schedule = ["recon"] * recon_batches_per_scand + ["scand"]

    def __iter__(self):
        iterators = {name: iter(loader) for name, loader in self.loaders.items()}
        while True:
            for domain in self.schedule:
                try:
                    batch = next(iterators[domain])
                except StopIteration:
                    iterators[domain] = iter(self.loaders[domain])
                    batch = next(iterators[domain])
                yield {**batch, "domain": domain}


class ScheduledDomainLoader:
    """Cycle deterministic shard-aware loaders using an explicit domain schedule."""

    def __init__(self, loaders: dict[str, DataLoader], schedule: list[str]) -> None:
        if not loaders:
            raise ValueError("at least one domain loader is required")
        if not schedule or any(domain not in loaders for domain in schedule):
            raise ValueError("domain schedule must reference every active loader")
        self.loaders = loaders
        self.schedule = list(schedule)

    def __iter__(self):
        iterators = {name: iter(loader) for name, loader in self.loaders.items()}
        while True:
            for domain in self.schedule:
                try:
                    batch = next(iterators[domain])
                except StopIteration:
                    iterators[domain] = iter(self.loaders[domain])
                    batch = next(iterators[domain])
                yield {**batch, "domain": domain}


def parse_domain_schedule(value: str, active_domains: set[str]) -> list[str]:
    if value.strip().lower() == "auto":
        if active_domains == {"recon", "scand", "sacson"}:
            value = "recon:4,scand:1,sacson:2"
        elif active_domains == {"recon", "scand"}:
            value = "recon:3,scand:1"
        elif active_domains == {"recon"}:
            value = "recon:1"
        else:
            raise ValueError(f"unsupported active domain set: {sorted(active_domains)}")
    schedule: list[str] = []
    for item in value.split(","):
        name, sep, count_text = item.partition(":")
        name = name.strip().lower()
        if not sep:
            count_text = "1"
        if name not in active_domains:
            raise ValueError(f"domain schedule references inactive domain: {name}")
        count = int(count_text)
        if count < 1:
            raise ValueError("domain schedule counts must be positive")
        schedule.extend([name] * count)
    if set(schedule) != active_domains:
        missing = sorted(active_domains - set(schedule))
        raise ValueError(f"domain schedule must include all active domains; missing={missing}")
    return schedule


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--latent-root", type=Path, required=True)
    parser.add_argument("--scand-latent-root", type=Path, default=None)
    parser.add_argument("--sacson-latent-root", type=Path, default=None)
    parser.add_argument("--split-lock", type=Path, default=None)
    parser.add_argument("--recon-batches-per-scand", type=int, default=3)
    parser.add_argument("--domain-schedule", type=str, default="auto")
    parser.add_argument("--history-frames", type=int, choices=(4,), default=4)
    parser.add_argument("--future-frames", type=int, choices=(4,), default=4)
    parser.add_argument("--recon-val-h4-nmse-limit", type=float, default=None)
    parser.add_argument("--allow-unverified-warm-start", action="store_true")
    parser.add_argument(
        "--allow-parent-test-target-adaptation",
        action="store_true",
        help=(
            "Allow a warm start on a derived target-development lock only when "
            "every active trajectory belonged to the warm checkpoint parent test "
            "split. The latent training view must still omit active test IDs."
        ),
    )
    parser.add_argument("--warm-start-parent-split-lock", type=Path, default=None)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batches-per-epoch", type=int, default=1000)
    parser.add_argument("--eval-batches-per-epoch", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument(
        "--train-drop-last",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--stream-cache-shards", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--amp", choices=("bf16", "fp16", "off"), default="bf16")
    parser.add_argument("--contrastive-every", type=int, default=4)
    parser.add_argument("--consistency-every", type=int, default=8)
    parser.add_argument("--model-dim", type=int, default=192)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--history-layers", type=int, default=2)
    parser.add_argument("--action-layers", type=int, default=2)
    parser.add_argument("--rollout-blocks", type=int, default=6)
    parser.add_argument("--spatial-window", type=int, default=8)
    parser.add_argument("--action-output-residual", action="store_true")
    parser.add_argument("--action-residual-alpha-init", type=float, default=0.01)
    parser.add_argument("--action-cross-attention-floor", type=float, default=0.0)
    parser.add_argument("--action-cross-attention-enforce-floor", action="store_true")
    parser.add_argument("--strong-action-cross-attention", action="store_true")
    parser.add_argument("--strong-action-cross-attention-start-block", type=int, default=4)
    parser.add_argument("--strong-action-beta-init", type=float, default=0.01)
    parser.add_argument("--strong-action-beta-max", type=float, default=0.10)
    parser.add_argument("--block-causal-action-integration", action="store_true")
    parser.add_argument("--block-causal-action-start-block", type=int, default=4)
    parser.add_argument("--block-causal-beta-init", type=float, default=0.01)
    parser.add_argument("--block-causal-beta-max", type=float, default=0.10)
    parser.add_argument("--causal-token-action-gru", action="store_true")
    parser.add_argument("--causal-token-action-gru-after-block", type=int, default=3)
    parser.add_argument("--causal-token-action-gamma-init", type=float, default=0.01)
    parser.add_argument("--causal-token-action-gamma-max", type=float, default=0.10)
    parser.add_argument(
        "--train-scope",
        choices=("all", "action_cross_attention", "strong_action_cross_attention", "strong_action_plus_late_blocks", "block_causal_action_integration", "causal_token_action_gru", "dino_wm_action_concat", "output_calibration"),
        default="all",
    )
    parser.add_argument("--unfreeze-last-rollout-blocks", type=int, choices=(1, 2), default=2)
    parser.add_argument("--ordered-action-gru", action="store_true")
    parser.add_argument("--ordered-action-scale-init", type=float, default=0.05)
    parser.add_argument("--train-ordered-action-path", action="store_true")
    parser.add_argument("--dino-wm-action-concat", action="store_true")
    parser.add_argument("--objective-revision", choices=("base", "action_sensitive", "norm_calibration", "action_norm_balanced"), default="base")
    parser.add_argument("--warm-start-checkpoint", type=Path, default=None)
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    parser.add_argument("--checkpoint-every-steps", type=int, default=250)
    parser.add_argument("--lr-schedule", choices=("constant", "warmup_cosine"), default="constant")
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def amp_context(mode: str):
    if mode == "off":
        return torch.autocast(device_type="cuda", enabled=False)
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16 if mode == "bf16" else torch.float16)


def corrupt_actions(actions: torch.Tensor, batch_index: int) -> torch.Tensor:
    mode = batch_index % 6
    if mode == 0:
        return actions.flip(1)
    if mode == 1:
        order = torch.randperm(actions.shape[1], device=actions.device)
        return actions[:, order]
    if mode == 2:
        return actions[:, :1].expand_as(actions)
    if mode == 3:
        result = actions.clone(); result[..., 2].neg_(); return result
    if mode == 4:
        return 1.5 * actions
    scale = actions.std(dim=(0, 1), keepdim=True).clamp_min(1e-4)
    return actions + torch.randn_like(actions) * scale


def make_loaders(args: argparse.Namespace):
    def build(root: Path, domain: str, seed_offset: int):
        train = TokenSequenceStreamingDataset(
            root,
            "train",
            args.history_frames,
            args.future_frames,
            args.stream_cache_shards,
        )
        val = TokenSequenceStreamingDataset(
            root,
            "val",
            args.history_frames,
            args.future_frames,
            args.stream_cache_shards,
        )
        train_sampler = ShardAwareBatchSampler(
            train,
            args.batch_size,
            drop_last=args.train_drop_last,
            seed=args.seed + seed_offset,
        )
        val_sampler = ShardAwareBatchSampler(
            val,
            args.eval_batch_size,
            drop_last=False,
            seed=10000 + args.seed + seed_offset,
        )
        loader_kwargs = {
            "num_workers": args.num_workers,
            "pin_memory": True,
            "persistent_workers": args.num_workers > 0,
        }
        if args.num_workers > 0:
            loader_kwargs["prefetch_factor"] = args.prefetch_factor
        train_generator = torch.Generator().manual_seed(args.seed + seed_offset + 300000)
        val_generator = torch.Generator().manual_seed(args.seed + seed_offset + 400000)
        train_loader = DataLoader(
            train, batch_sampler=train_sampler, generator=train_generator, **loader_kwargs
        )
        val_loader = DataLoader(
            val, batch_sampler=val_sampler, generator=val_generator, **loader_kwargs
        )
        return train, val, train_loader, val_loader

    recon_train, recon_val, recon_train_loader, recon_val_loader = build(
        args.latent_root, "recon", 0
    )
    train_datasets = {"recon": recon_train}
    val_datasets = {"recon": recon_val}
    train_loaders = {"recon": recon_train_loader}
    val_loaders = {"recon": recon_val_loader}
    if args.scand_latent_root is not None:
        scand_train, scand_val, scand_train_loader, scand_val_loader = build(
            args.scand_latent_root, "scand", 100000
        )
        train_datasets["scand"] = scand_train
        val_datasets["scand"] = scand_val
        train_loaders["scand"] = scand_train_loader
        val_loaders["scand"] = scand_val_loader
    if args.sacson_latent_root is not None:
        sacson_train, sacson_val, sacson_train_loader, sacson_val_loader = build(
            args.sacson_latent_root, "sacson", 200000
        )
        train_datasets["sacson"] = sacson_train
        val_datasets["sacson"] = sacson_val
        train_loaders["sacson"] = sacson_train_loader
        val_loaders["sacson"] = sacson_val_loader
    schedule = parse_domain_schedule(args.domain_schedule, set(train_loaders))
    train_loader = ScheduledDomainLoader(train_loaders, schedule)
    return train_datasets, val_datasets, train_loader, val_loaders, train_loaders, schedule


def step_metrics(prediction: torch.Tensor, target: torch.Tensor, history: torch.Tensor) -> dict[str, list[float]]:
    pred = prediction[:, 0].float(); target = target.float(); current = history[:, -1].float()
    error = (pred - target).square().mean(dim=(-1, -2))
    denom = target.square().mean(dim=(-1, -2)).clamp_min(1e-8)
    nmse = (error / denom).mean(0)
    cosine = (1.0 - torch.nn.functional.cosine_similarity(pred.flatten(-2), target.flatten(-2), dim=-1)).mean(0)
    pred_delta = pred - current[:, None]; target_delta = target - current[:, None]
    ratio = (torch.linalg.vector_norm(pred_delta.flatten(-2), dim=-1) / torch.linalg.vector_norm(target_delta.flatten(-2), dim=-1).clamp_min(1e-8)).mean(0)
    return {"nmse": nmse.tolist(), "cosine": cosine.tolist(), "norm_ratio": ratio.tolist()}


def merge_metric_rows(
    rows: list[tuple[int, dict[str, list[float]]]],
) -> dict[str, list[float]]:
    total = sum(weight for weight, _ in rows)
    return {
        key: (
            sum(
                weight * np.asarray(row[key], dtype=np.float64)
                for weight, row in rows
            )
            / total
        ).tolist()
        for key in rows[0][1]
    }


def evaluate(model, loader, args, device) -> dict[str, object]:
    model.eval(); true_rows=[]; reverse_rows=[]; noisy_rows=[]; losses=[]; finite=0; total=0
    started=time.perf_counter(); torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if batch_index >= args.eval_batches_per_epoch: break
            history=batch["history_tokens"].to(device,non_blocking=True)
            target=batch["future_tokens"].to(device,non_blocking=True)
            actions=batch["actions"].to(device,non_blocking=True)
            noise_scale=actions.std(dim=(0,1),keepdim=True).clamp_min(1e-4)
            with amp_context(args.amp):
                output=model(history,actions)
                reverse=model(history,actions.flip(1))
                noisy=model(history,actions+0.5*noise_scale*torch.randn_like(actions))
                loss,_=compute_world_model_v2_loss(output,target,history,actions,weights=V2LossWeights())
            batch_size=int(history.shape[0])
            losses.append(float(loss)); true_rows.append((batch_size,step_metrics(output["predicted_tokens"],target,history)))
            reverse_rows.append((batch_size,step_metrics(reverse["predicted_tokens"],target,history)))
            noisy_rows.append((batch_size,step_metrics(noisy["predicted_tokens"],target,history)))
            finite += int(torch.isfinite(output["predicted_tokens"]).all()) * history.shape[0]; total += history.shape[0]
    true=merge_metric_rows(true_rows); reverse=merge_metric_rows(reverse_rows); noisy=merge_metric_rows(noisy_rows)
    return {
        "loss":float(np.mean(losses)),"true":true,"reversed":reverse,"noisy":noisy,
        "reverse_nmse_gap_h4":reverse["nmse"][3]-true["nmse"][3],
        "noisy_nmse_gap_h4":noisy["nmse"][3]-true["nmse"][3],
        "finite_fraction":finite/max(total,1),"batches":len(losses),"samples":total,
        "runtime_seconds":time.perf_counter()-started,
        "peak_gpu_memory_bytes":int(torch.cuda.max_memory_allocated(device)),
    }


def train_epoch(
    model,
    loader,
    optimizer,
    scaler,
    args,
    scheduler,
    device,
    weights,
    epoch,
    start_batch=0,
    global_step=0,
    checkpoint_callback=None,
):
    model.train()
    rows = []
    domains = Counter()
    started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    for batch_index, batch in enumerate(loader):
        if batch_index >= args.batches_per_epoch:
            break
        if batch_index < start_batch:
            continue
        step_index = global_step + batch_index - start_batch
        domain = str(batch.get("domain", "recon"))
        domains[domain] += 1
        history = batch["history_tokens"].to(device, non_blocking=True)
        target = batch["future_tokens"].to(device, non_blocking=True)
        actions = batch["actions"].to(device, non_blocking=True)
        with amp_context(args.amp):
            output = model(history, actions)
            corrupted = None
            prefix = None
            if args.contrastive_every > 0 and step_index % args.contrastive_every == 0:
                corrupted = model(history, corrupt_actions(actions, step_index // args.contrastive_every))
            if args.consistency_every > 0 and step_index % args.consistency_every == 0:
                prefix_actions = actions.clone()
                prefix_actions[:, 3:] = 0
                prefix = model(history, prefix_actions)
            loss, components = compute_world_model_v2_loss(
                output,
                target,
                history,
                actions,
                weights=weights,
                corrupted_output=corrupted,
                prefix_output=prefix,
            )
        if scaler is None:
            loss.backward()
            strong_grad_sq = sum(
                float(parameter.grad.detach().float().square().sum())
                for name, parameter in model.named_parameters()
                if (
                    ".strong_action_cross." in name
                    or ".block_causal_action." in name
                    or name.startswith("causal_token_action_gru.")
                ) and parameter.grad is not None
            )
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
        else:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            strong_grad_sq = sum(
                float(parameter.grad.detach().float().square().sum())
                for name, parameter in model.named_parameters()
                if (
                    ".strong_action_cross." in name
                    or ".block_causal_action." in name
                    or name.startswith("causal_token_action_gru.")
                ) and parameter.grad is not None
            )
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        if scheduler is not None:
            scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        rows.append({
            "loss": float(loss.detach()),
            "strong_action_attention_grad_norm": strong_grad_sq ** 0.5,
            "action_integration_grad_norm": strong_grad_sq ** 0.5,
            **{k: float(v.detach()) for k, v in components.items()},
        })
        if (batch_index + 1) % 100 == 0:
            print(json.dumps({
                "epoch": epoch,
                "batch": batch_index + 1,
                "global_step": step_index + 1,
                "loss": float(np.mean([r["loss"] for r in rows[-100:]])),
            }), flush=True)
        if checkpoint_callback is not None and args.checkpoint_every_steps > 0:
            if (step_index + 1) % args.checkpoint_every_steps == 0:
                checkpoint_callback(epoch, batch_index + 1, step_index + 1)
    if not rows:
        return {
            "loss": float("nan"),
            "batches": 0,
            "domain_batches": dict(domains),
            "runtime_seconds": time.perf_counter() - started,
        }
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]} | {
        "batches": len(rows),
        "domain_batches": dict(domains),
        "runtime_seconds": time.perf_counter() - started,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
    }


def main() -> None:
    args=parse_args(); set_seed(args.seed)
    if not torch.cuda.is_available(): raise RuntimeError("CUDA is required for v2-4ctx training")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    if (args.scand_latent_root is not None or args.sacson_latent_root is not None) and args.split_lock is None:
        raise ValueError("--split-lock is mandatory for multi-domain training")
    split_lock_sha256=None
    if args.split_lock is not None:
        split_lock,split_lock_sha256=load_split_lock(args.split_lock)
        roots = [
            ("recon", args.latent_root),
            ("scand", args.scand_latent_root),
            ("sacson", args.sacson_latent_root),
        ]
        for dataset,root in roots:
            if root is None: continue
            metadata=json.loads((root/"metadata.json").read_text(encoding="utf-8"))
            if metadata.get("split_lock_sha256") != split_lock_sha256:
                raise ValueError(f"latent root is not bound to the active split lock: {root}")
            root_splits=json.loads((root/"splits.json").read_text(encoding="utf-8"))
            if root_splits.get("test"):
                raise ValueError(f"training latent root exposes test trajectories: {root}")
            expected_training_view={**split_lock[dataset]["splits"],"test":[]}
            validate_latent_root(root,expected_training_view)
    args.checkpoint_dir.mkdir(parents=True,exist_ok=True); args.report_dir.mkdir(parents=True,exist_ok=True)
    device=torch.device("cuda")
    cfg=LiteTokenFormerConfig(horizon=args.future_frames,model_dim=args.model_dim,heads=args.heads,history_layers=args.history_layers,
                              action_layers=args.action_layers,rollout_blocks=args.rollout_blocks,spatial_window=args.spatial_window,
                              action_output_residual=args.action_output_residual,action_residual_alpha_init=args.action_residual_alpha_init,
                              action_cross_attention_floor=args.action_cross_attention_floor,
                              action_cross_attention_enforce_floor=args.action_cross_attention_enforce_floor,
                              ordered_action_gru=args.ordered_action_gru,
                              ordered_action_scale_init=args.ordered_action_scale_init,
                              dino_wm_action_concat=args.dino_wm_action_concat,
                              strong_action_cross_attention=args.strong_action_cross_attention,
                              strong_action_cross_attention_start_block=args.strong_action_cross_attention_start_block,
                              strong_action_beta_init=args.strong_action_beta_init,
                              strong_action_beta_max=args.strong_action_beta_max,
                              block_causal_action_integration=args.block_causal_action_integration,
                              block_causal_action_start_block=args.block_causal_action_start_block,
                              block_causal_beta_init=args.block_causal_beta_init,
                              block_causal_beta_max=args.block_causal_beta_max,
                              causal_token_action_gru=args.causal_token_action_gru,
                              causal_token_action_gru_after_block=args.causal_token_action_gru_after_block,
                              causal_token_action_gamma_init=args.causal_token_action_gamma_init,
                              causal_token_action_gamma_max=args.causal_token_action_gamma_max)
    model=LiteTokenFormerWMV2(cfg).to(device)
    warm_start_report=None
    if args.warm_start_checkpoint is not None:
        payload=torch.load(args.warm_start_checkpoint,map_location="cpu",weights_only=False)
        checkpoint_lock=payload.get("split_lock_sha256")
        verified_parent_reuse = None
        if (
            split_lock_sha256 is not None
            and checkpoint_lock != split_lock_sha256
        ):
            if args.warm_start_parent_split_lock is not None:
                if args.allow_parent_test_target_adaptation:
                    parent_lock,parent_lock_sha=load_split_lock(
                        args.warm_start_parent_split_lock
                    )
                    declared_parent=split_lock.get("lineage",{}).get(
                        "warm_start_parent_split_lock_sha256"
                    )
                    if declared_parent != parent_lock_sha:
                        raise RuntimeError(
                            "target-adaptation lock does not declare the supplied "
                            "warm-start parent lock"
                        )
                    if checkpoint_lock != parent_lock_sha:
                        raise RuntimeError(
                            "warm checkpoint is not bound to the supplied parent lock"
                        )
                    active_ids={
                        str(item)
                        for split_name in ("train","val","test")
                        for item in split_lock["recon"]["splits"].get(split_name,())
                    }
                    parent_test=set(map(
                        str,parent_lock["recon"]["splits"].get("test",())
                    ))
                    outside=sorted(active_ids-parent_test)
                    if outside:
                        raise RuntimeError(
                            "target adaptation may use only warm-parent test "
                            f"trajectories; outside={outside[:10]}"
                        )
                    verified_parent_reuse={
                        "status":"PASS",
                        "mode":"warm_parent_test_target_adaptation",
                        "parent_split_lock":str(args.warm_start_parent_split_lock),
                        "parent_split_lock_sha256":parent_lock_sha,
                        "active_trajectory_count":len(active_ids),
                        "active_ids_all_parent_test":True,
                        "outer_test_absent_from_latent_training_view":True,
                    }
                else:
                    verified_parent_reuse = validate_parent_split_reuse(
                        split_lock,
                        parent_split_lock=args.warm_start_parent_split_lock,
                        artifact_split_lock_sha256=checkpoint_lock,
                        unchanged_datasets=("recon",),
                        preserved_datasets=("scand",),
                    )
            elif not args.allow_unverified_warm_start:
                raise RuntimeError(
                    "warm-start checkpoint is not bound to the active strict split "
                    "lock; paper continuation requires "
                    "--warm-start-parent-split-lock"
                )
        state=payload.get("model",payload)
        additive_revision = args.ordered_action_gru or args.dino_wm_action_concat or args.strong_action_cross_attention or args.block_causal_action_integration or args.causal_token_action_gru
        result=model.load_state_dict(state,strict=not additive_revision)
        if additive_revision:
            allowed=("action_encoder.ordered_","action_concat_")
            unexpected_missing=[
                key for key in result.missing_keys
                if not key.startswith(allowed)
                and ".strong_action_cross." not in key
                and ".block_causal_action." not in key
                and not key.startswith("causal_token_action_gru.")
            ]
            if unexpected_missing or result.unexpected_keys:
                raise RuntimeError(f"unsafe residual warm start: missing={unexpected_missing}, unexpected={result.unexpected_keys}")
        warm_start_report={
            "checkpoint":str(args.warm_start_checkpoint),
            "checkpoint_sha256":file_sha256(args.warm_start_checkpoint),
            "checkpoint_split_lock_sha256":checkpoint_lock,
            "lineage":verified_parent_reuse or {
                "status":"PASS" if checkpoint_lock == split_lock_sha256 else "UNVERIFIED",
                "mode":"active_split_lock" if checkpoint_lock == split_lock_sha256 else "allow_unverified_warm_start",
            },
            "missing_keys":list(result.missing_keys),
            "unexpected_keys":list(result.unexpected_keys),
        }
    train_ds, val_ds, train_loader, val_loaders, train_loaders, domain_schedule = make_loaders(args)
    weights=V2LossWeights()
    if args.objective_revision=="action_sensitive":
        weights=V2LossWeights(norm=0.05,action_order=0.12,action_decode=0.10,uncertainty=0.02)
    elif args.objective_revision=="norm_calibration":
        weights=V2LossWeights(norm=0.15,action_order=0.12,action_decode=0.0,uncertainty=0.01)
    elif args.objective_revision=="action_norm_balanced":
        weights=V2LossWeights(norm=0.10,action_order=0.12,action_decode=0.10,uncertainty=0.02)
    if args.train_scope == "action_cross_attention":
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for block in model.blocks:
            for parameter in block.action_attn.parameters():
                parameter.requires_grad_(True)
        if args.train_ordered_action_path:
            if not args.ordered_action_gru:
                raise ValueError("--train-ordered-action-path requires --ordered-action-gru")
            for name,parameter in model.action_encoder.named_parameters():
                if name.startswith("ordered_"):
                    parameter.requires_grad_(True)
    elif args.train_scope in {"strong_action_cross_attention", "strong_action_plus_late_blocks"}:
        if not args.strong_action_cross_attention:
            raise ValueError("strong action train scope requires --strong-action-cross-attention")
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for block in model.blocks:
            if block.strong_action_cross is not None:
                for parameter in block.strong_action_cross.parameters():
                    parameter.requires_grad_(True)
        if args.train_scope == "strong_action_plus_late_blocks":
            for block in model.blocks[-args.unfreeze_last_rollout_blocks:]:
                for parameter in block.parameters():
                    parameter.requires_grad_(True)
    elif args.train_scope == "block_causal_action_integration":
        if not args.block_causal_action_integration:
            raise ValueError("block-causal action scope requires --block-causal-action-integration")
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for block in model.blocks:
            if block.block_causal_action is not None:
                for parameter in block.block_causal_action.parameters():
                    parameter.requires_grad_(True)
    elif args.train_scope == "causal_token_action_gru":
        if not args.causal_token_action_gru:
            raise ValueError("causal token-action GRU scope requires --causal-token-action-gru")
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for parameter in model.causal_token_action_gru.parameters():
            parameter.requires_grad_(True)
    elif args.train_scope == "dino_wm_action_concat":
        if not args.dino_wm_action_concat:
            raise ValueError("dino_wm_action_concat scope requires --dino-wm-action-concat")
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for name,parameter in model.named_parameters():
            if name.startswith("action_concat_"):
                parameter.requires_grad_(True)
        for block in model.blocks:
            for parameter in block.temporal_attn.parameters():
                parameter.requires_grad_(True)
            for parameter in block.temporal_norm.parameters():
                parameter.requires_grad_(True)
    elif args.train_scope == "output_calibration":
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for parameter in model.delta_head.parameters():
            parameter.requires_grad_(True)
    trainable_parameters=[parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable_parameters:
        raise RuntimeError("selected train scope has no trainable parameters")
    optimizer=torch.optim.AdamW(trainable_parameters,lr=args.lr,weight_decay=args.weight_decay)
    total_scheduled_steps = max(1, args.epochs * args.batches_per_epoch)
    if not 0.0 <= args.min_lr_ratio <= 1.0:
        raise ValueError("--min-lr-ratio must be in [0, 1]")
    if args.warmup_steps < 0 or args.warmup_steps >= total_scheduled_steps:
        raise ValueError("--warmup-steps must be >= 0 and smaller than total scheduled steps")
    def lr_multiplier(step: int) -> float:
        if args.lr_schedule == "constant":
            return 1.0
        if args.warmup_steps > 0 and step < args.warmup_steps:
            return float(step + 1) / float(args.warmup_steps)
        decay_steps = max(1, total_scheduled_steps - args.warmup_steps)
        progress = min(max((step - args.warmup_steps) / decay_steps, 0.0), 1.0)
        cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
        return args.min_lr_ratio + (1.0 - args.min_lr_ratio) * cosine
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_multiplier)
    scaler=torch.amp.GradScaler("cuda") if args.amp=="fp16" else None
    history=[]; best=None; best_score=float("inf"); total_started=time.perf_counter()
    resume_state = None
    resume_payload = None
    if args.resume_checkpoint is not None:
        resume_payload = torch.load(args.resume_checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(resume_payload["model"], strict=True)
        optimizer.load_state_dict(resume_payload["optimizer"])
        if resume_payload.get("scheduler") is not None:
            scheduler.load_state_dict(resume_payload["scheduler"])
        if scaler is not None and resume_payload.get("scaler") is not None:
            scaler.load_state_dict(resume_payload["scaler"])
        if resume_payload.get("rng") is not None:
            random.setstate(resume_payload["rng"]["python"])
            np.random.set_state(resume_payload["rng"]["numpy"])
            torch.set_rng_state(resume_payload["rng"]["torch"])
            if torch.cuda.is_available() and resume_payload["rng"].get("cuda") is not None:
                torch.cuda.set_rng_state_all(resume_payload["rng"]["cuda"])
        resume_state = resume_payload.get("training_state", {})
        history = list(resume_payload.get("history", []))
        best_score = float(resume_payload.get("best_score", float("inf")))
        if resume_payload.get("best_payload") is not None:
            best = resume_payload["best_payload"]
        print(json.dumps({"resumed_from": str(args.resume_checkpoint), "training_state": resume_state}), flush=True)
        for domain, sampler_epoch in resume_state.get("sampler_epochs", {}).items():
            sampler = getattr(train_loaders.get(domain), "batch_sampler", None)
            if sampler is not None:
                sampler.epoch = int(sampler_epoch)
    if resume_payload is not None and resume_payload.get("selection_baselines") is not None:
        selection_baselines = resume_payload["selection_baselines"]
    else:
        selection_baselines = {domain: evaluate(model, loader, args, device) for domain, loader in val_loaders.items()}
    baseline_scores = {
        domain: float(np.mean(metrics["true"]["nmse"])) + float(metrics["true"]["nmse"][3])
        for domain, metrics in selection_baselines.items()
    }
    start_epoch = int(resume_state.get("epoch", 1)) if resume_state else 1
    resume_batch = int(resume_state.get("batch_in_epoch", 0)) if resume_state else 0
    global_step = int(resume_state.get("global_step", 0)) if resume_state else 0
    schedule_metadata = {
        "domain_schedule": domain_schedule,
        "domain_schedule_string": args.domain_schedule,
        "batch_size": args.batch_size,
        "batches_per_epoch": args.batches_per_epoch,
        "epochs": args.epochs,
        "scheduled_updates": args.epochs * args.batches_per_epoch,
        "scheduled_samples": {
            domain: int(args.epochs * args.batches_per_epoch * args.batch_size * domain_schedule.count(domain) / len(domain_schedule))
            for domain in domain_schedule
        },
    }

    def current_payload(training_state=None):
        rng = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
        return {
            "model": model.state_dict(),
            "config": asdict(cfg),
            "args": vars(args),
            "loss_weights": asdict(weights),
            "history": history,
            "best_score": best_score,
            "best_payload": best,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "rng": rng,
            "training_state": training_state or {},
            "parameter_count": count_trainable_parameters(model),
            "split_lock_sha256": split_lock_sha256,
            "temporal_protocol":{"history_frames":4,"future_frames":4},
            "selection_baselines": selection_baselines,
            "selection_protocol":"equal-weight normalized macro over every active validation domain with RECON guard",
            "schedule_metadata": schedule_metadata,
        }

    def sampler_epochs_now():
        result = {}
        for domain, loader in train_loaders.items():
            sampler = getattr(loader, "batch_sampler", None)
            if sampler is not None:
                result[domain] = int(getattr(sampler, "epoch", 0))
        return result

    epoch_start_sampler_epochs = sampler_epochs_now()

    def save_latest(epoch, batch_in_epoch, step, sampler_epochs=None):
        if sampler_epochs is None:
            sampler_epochs = sampler_epochs_now()
        payload = current_payload({
            "epoch": int(epoch),
            "batch_in_epoch": int(batch_in_epoch),
            "global_step": int(step),
            "sampler_epochs": sampler_epochs,
        })
        tmp = args.checkpoint_dir / "latest.pt.tmp"
        torch.save(payload, tmp)
        tmp.replace(args.checkpoint_dir / "latest.pt")

    def step_checkpoint(epoch, batch_in_epoch, step):
        save_latest(
            epoch,
            batch_in_epoch,
            step,
            sampler_epochs=epoch_start_sampler_epochs,
        )

    for epoch in range(start_epoch,args.epochs+1):
        epoch_start_sampler_epochs = sampler_epochs_now()
        train_metrics=train_epoch(
            model, train_loader, optimizer, scaler, args, scheduler=scheduler,
            device=device, weights=weights, epoch=epoch,
            start_batch=resume_batch if epoch == start_epoch else 0,
            global_step=global_step,
            checkpoint_callback=step_checkpoint,
        )
        global_step += int(train_metrics.get("batches", 0))
        resume_batch = 0
        val_metrics={domain:evaluate(model,loader,args,device) for domain,loader in val_loaders.items()}
        domain_scores = {
            domain: float(np.mean(metrics["true"]["nmse"])) + float(metrics["true"]["nmse"][3])
            for domain, metrics in val_metrics.items()
        }
        normalized_domain_scores = {
            domain: domain_scores[domain] / max(float(baseline_scores[domain]), 1e-8)
            for domain in val_metrics
        }
        selection_domain="tridomain_macro"
        score=float(np.mean(list(normalized_domain_scores.values())))
        recon_guard_pass=True
        if args.recon_val_h4_nmse_limit is not None:
            recon_guard_pass=val_metrics["recon"]["true"]["nmse"][3] <= args.recon_val_h4_nmse_limit
            if not recon_guard_pass:
                score=float("inf")
        row={"epoch":epoch,"train":train_metrics,"val":val_metrics,"domain_scores":domain_scores,
             "normalized_domain_scores":normalized_domain_scores,"selection_score":score,
             "elapsed_seconds":time.perf_counter()-total_started}
        row["selection_domain"]=selection_domain
        row["recon_guard_pass"]=recon_guard_pass
        history.append(row); print(json.dumps({
            "epoch_complete":epoch,
            "score":score,
            "selection_domain":selection_domain,
            "recon_guard_pass":recon_guard_pass,
            "val_h4_nmse":{domain:metrics["true"]["nmse"][3] for domain,metrics in val_metrics.items()},
        }),flush=True)
        payload=current_payload({
            "epoch": int(epoch + 1),
            "batch_in_epoch": 0,
            "global_step": int(global_step),
        })
        payload["history"] = history
        torch.save(payload,args.checkpoint_dir/f"epoch_{epoch:03d}.pt")
        if score<best_score:
            best_score=score
            best=copy.deepcopy(payload)
            best["best_payload"] = None
            torch.save(best,args.checkpoint_dir/"best.pt")
        save_latest(epoch + 1, 0, global_step)
    payload=current_payload({
        "epoch": int(args.epochs + 1),
        "batch_in_epoch": 0,
        "global_step": int(global_step),
    })
    torch.save(payload,args.checkpoint_dir/"final.pt")
    selection_status="complete" if best is not None else "guard_failed"
    best_checkpoint=str(args.checkpoint_dir/"best.pt") if best is not None else None
    summary={
        "status":selection_status,"model":"LiteTokenFormer-WM-v2-4ctx-base","seed":args.seed,
        "parameter_count":count_trainable_parameters(model),"total_parameter_count":sum(p.numel() for p in model.parameters()),"config":asdict(cfg),"loss_weights":asdict(weights),
        "train_dataset":{domain:dataset.summary() for domain,dataset in train_ds.items()},
        "val_dataset":{domain:dataset.summary() for domain,dataset in val_ds.items()},
        "history":history,"best_score":best_score,
        "best_checkpoint":best_checkpoint,"final_checkpoint":str(args.checkpoint_dir/"final.pt"),
        "warm_start":warm_start_report,
        "split_lock_sha256":split_lock_sha256,
        "temporal_protocol":{"history_frames":4,"future_frames":4},
        "train_cache":{domain:dataset.cache_stats() for domain,dataset in train_ds.items()},
        "val_cache":{domain:dataset.cache_stats() for domain,dataset in val_ds.items()},
        "runtime_seconds":time.perf_counter()-total_started,
        "scientific_namespace":"behavior_trajectory_dino_token_prediction",
    }
    (args.report_dir/"train_summary.json").write_text(json.dumps(summary,indent=2,default=str),encoding="utf-8")
    (args.report_dir/"train_summary.md").write_text(
        "# LiteTokenFormer-WM-v2-4ctx Training\n\n"
        f"Status: **{summary['status']}**\n\nParameters: {summary['parameter_count']:,}\n\n"
        f"Best checkpoint: `{summary['best_checkpoint']}`\n\n"
        "This training uses behavior-trajectory visual futures only; it does not claim candidate-specific counterfactual visual accuracy.\n",
        encoding="utf-8",
    )
    print(json.dumps({"status":selection_status,"best_checkpoint":best_checkpoint,"best_score":best_score},indent=2))


if __name__=="__main__":
    main()
