#!/usr/bin/env python3
"""Train a v70-compatible scorer-v2 selector from three-domain H4 latents.

The original v70 selector consumes candidate-level physical labels.  The new
three-domain run does not have those labels for every SACSoN window, so this
stage uses a deterministic latent counterfactual target: each candidate is
rolled through the newly trained LTWM, then scored by prediction error against
the observed future and latent goal progress.  The resulting checkpoint keeps
the scorer-v2 architecture and candidate-independent interface while recording
the proxy-target provenance explicitly.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "tools/recon")]

from recon_latents.models_v2 import LiteTokenFormerConfig, LiteTokenFormerWMV2  # noqa: E402
from recon_latents.losses_v2 import (  # noqa: E402
    TrajectoryQualityLossConfig,
    trajectory_quality_loss,
)
from latent_window_utils import ShardAwareBatchSampler, StreamingLatentWindowDataset  # noqa: E402
from scoring.trajectory_quality_scorer_v2 import (  # noqa: E402
    TrajectoryQualityScorerConfig,
    TrajectoryQualityScorerV2,
)


DOMAINS = ("recon", "scand", "sacson")
HISTORY = 4
HORIZON = 4
K = 16


class WindowDataset(StreamingLatentWindowDataset):
    def __getitem__(self, index: int) -> dict[str, object]:
        traj_index, history_start = self.windows[index]
        traj = self.trajectories[traj_index]
        z, actions_all = self._load_shard_cached(traj_index)
        t = history_start + self.history_len - 1
        history = np.ascontiguousarray(
            z[history_start : t + 1].reshape(HISTORY, 256, 384),
            dtype=np.float32,
        )
        future = np.ascontiguousarray(
            z[t + 1 : t + HORIZON + 1].reshape(HORIZON, 256, 384),
            dtype=np.float32,
        )
        actions = np.ascontiguousarray(
            actions_all[t : t + HORIZON],
            dtype=np.float32,
        )
        return {
            "history_tokens": torch.from_numpy(history),
            "future_tokens": torch.from_numpy(future),
            "actions": torch.from_numpy(actions),
            "trajectory_id": traj.trajectory_id,
            "timestep": int(t),
        }


class ScheduledLoader:
    def __init__(self, loaders: dict[str, DataLoader], schedule: tuple[str, ...]):
        self.loaders = loaders
        self.schedule = schedule

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recon-latent-root", type=Path, required=True)
    parser.add_argument("--scand-latent-root", type=Path, required=True)
    parser.add_argument("--sacson-latent-root", type=Path, required=True)
    parser.add_argument("--split-lock", type=Path, required=True)
    parser.add_argument("--world-model-checkpoint", type=Path, required=True)
    parser.add_argument("--warm-start-scorer", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--steps-per-epoch", type=int, default=1200)
    parser.add_argument("--eval-steps", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--stream-cache-shards", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loaders(args: argparse.Namespace):
    roots = {
        "recon": args.recon_latent_root,
        "scand": args.scand_latent_root,
        "sacson": args.sacson_latent_root,
    }
    train_loaders: dict[str, DataLoader] = {}
    val_loaders: dict[str, DataLoader] = {}
    datasets: dict[str, tuple[WindowDataset, WindowDataset]] = {}
    for offset, (domain, root) in enumerate(roots.items()):
        train = WindowDataset(root, "train", HISTORY, HORIZON, args.stream_cache_shards)
        val = WindowDataset(root, "val", HISTORY, HORIZON, args.stream_cache_shards)
        train_sampler = ShardAwareBatchSampler(
            train,
            args.batch_size,
            drop_last=True,
            seed=args.seed + 100000 * offset,
        )
        val_sampler = ShardAwareBatchSampler(
            val,
            args.eval_batch_size,
            drop_last=False,
            seed=args.seed + 100000 * offset + 50000,
        )
        train_loaders[domain] = DataLoader(
            train,
            batch_sampler=train_sampler,
            num_workers=0,
            pin_memory=True,
        )
        val_loaders[domain] = DataLoader(
            val,
            batch_sampler=val_sampler,
            num_workers=0,
            pin_memory=True,
        )
        datasets[domain] = (train, val)
    return datasets, ScheduledLoader(
        train_loaders,
        ("recon", "recon", "scand", "sacson"),
    ), val_loaders


def load_world_model(path: Path, device: torch.device) -> LiteTokenFormerWMV2:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = dict(payload.get("config", {}))
    allowed = set(LiteTokenFormerConfig.__dataclass_fields__)
    config = {key: value for key, value in config.items() if key in allowed}
    model = LiteTokenFormerWMV2(LiteTokenFormerConfig(**config)).to(device)
    state = payload.get("model", payload)
    model.load_state_dict(state, strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def candidate_actions(actions: torch.Tensor) -> torch.Tensor:
    """Deterministic K=16 action proposals anchored at candidate zero."""
    base = actions[:, None].expand(-1, K, -1, -1).clone()
    base[:, 0] = actions
    base[:, 1] = actions.flip(1)
    base[:, 2, :, 2] = -actions[:, :, 2]
    base[:, 3] = 0.5 * actions
    base[:, 4] = 1.5 * actions
    base[:, 5, :, 0] = 0.0
    base[:, 6, :, 1] = 0.0
    base[:, 7, :, 2] = 0.0
    base[:, 8] = actions.roll(1, dims=1)
    base[:, 9] = actions.roll(-1, dims=1)
    base[:, 10, :, 0] = -actions[:, :, 0]
    base[:, 11, :, 1] = -actions[:, :, 1]
    base[:, 12, :, :2] = 0.0
    base[:, 13] = actions + 0.25 * actions.std(dim=1, keepdim=True).clamp_min(1e-4)
    base[:, 14] = actions - 0.25 * actions.std(dim=1, keepdim=True).clamp_min(1e-4)
    base[:, 15] = actions.mean(dim=1, keepdim=True).expand(-1, HORIZON, -1)
    return base


@torch.no_grad()
def counterfactual_targets(
    world_model: LiteTokenFormerWMV2,
    history: torch.Tensor,
    future: torch.Tensor,
    actions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch = history.shape[0]
    candidates = candidate_actions(actions)
    history_rep = history[:, None].expand(-1, K, -1, -1, -1).reshape(
        batch * K, HISTORY, 256, 384
    )
    action_rep = candidates.reshape(batch * K, HORIZON, 3)
    predicted = world_model(history_rep, action_rep)["predicted_tokens"].reshape(
        batch, K, HORIZON, 256, 384
    )
    target_rep = future[:, None].expand_as(predicted)
    ate = (predicted - target_rep).float().square().mean(dim=(-1, -2, -3)).sqrt()
    current = history[:, -1]
    goal = future[:, -1]
    goal_distance = (current - goal).float().square().mean(dim=(-1, -2)).sqrt().clamp_min(1e-4)
    pred_distance = (predicted[:, :, -1] - goal[:, None]).float().square().mean(dim=(-1, -2)).sqrt()
    rtp = 1.0 - pred_distance / goal_distance[:, None]
    return candidates, torch.stack((ate, rtp.clamp(-1.0, 1.0)), dim=-1)


def fit_normalization(
    scorer: TrajectoryQualityScorerV2,
    world_model: LiteTokenFormerWMV2,
    loader: ScheduledLoader,
    device: torch.device,
    steps: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    values = []
    with torch.no_grad():
        for step, batch in enumerate(loader):
            if step >= steps:
                break
            history = batch["history_tokens"].to(device)
            actions = batch["actions"].to(device)
            candidates = candidate_actions(actions)
            current = history[:, -1]
            goal = batch["future_tokens"][:, -1].to(device)
            history_rep = history[:, None].expand(-1, K, -1, -1, -1).reshape(
                history.shape[0] * K, HISTORY, 256, 384
            )
            predicted = world_model(
                history_rep,
                candidates.reshape(history.shape[0] * K, HORIZON, 3),
            )["predicted_tokens"].reshape(history.shape[0], K, HORIZON, 256, 384)
            values.append(
                scorer.extract_features(
                    current_latent=current,
                    goal_latent=goal,
                    candidate_actions=candidates,
                    predicted_future=predicted,
                ).flatten(0, 1).float().cpu()
            )
    features = torch.cat(values, dim=0)
    return features.mean(0), features.std(0).clamp_min(1e-5)


def run_epoch(
    scorer: TrajectoryQualityScorerV2,
    world_model: LiteTokenFormerWMV2,
    loader,
    optimizer,
    device: torch.device,
    steps: int,
    train: bool,
) -> float:
    scorer.train(train)
    losses = []
    iterator = iter(loader)
    for _ in range(steps):
        batch = next(iterator)
        history = batch["history_tokens"].to(device, non_blocking=True)
        future = batch["future_tokens"].to(device, non_blocking=True)
        actions = batch["actions"].to(device, non_blocking=True)
        candidates, target = counterfactual_targets(world_model, history, future, actions)
        current = history[:, -1]
        goal = future[:, -1]
        with torch.no_grad():
            history_rep = history[:, None].expand(-1, K, -1, -1, -1).reshape(
                history.shape[0] * K, HISTORY, 256, 384
            )
            predicted = world_model(
                history_rep,
                candidates.reshape(history.shape[0] * K, HORIZON, 3),
            )["predicted_tokens"].reshape(history.shape[0], K, HORIZON, 256, 384)
        prediction = scorer(
            current_latent=current,
            goal_latent=goal,
            candidate_actions=candidates,
            predicted_future=predicted,
        )
        predicted_target = prediction.stacked()
        valid = torch.ones(target.shape[:2], dtype=torch.bool, device=device)
        loss = torch.nn.functional.smooth_l1_loss(predicted_target, target, reduction="none")
        loss = loss.mean(dim=-1)
        loss = loss.masked_fill(~valid, 0.0).mean()
        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(scorer.parameters(), 1.0)
            optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.output_dir.exists() and (args.output_dir / "summary.json").exists():
        print(json.dumps({"status": "already_complete", "output": str(args.output_dir)}))
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = torch.device(args.device)
    datasets, train_loader, val_loaders = make_loaders(args)
    world_model = load_world_model(args.world_model_checkpoint, device)
    scorer = TrajectoryQualityScorerV2(
        TrajectoryQualityScorerConfig(
            hidden_dim=640,
            hidden_layers=3,
            dropout=0.10,
            use_predicted_future=True,
        )
    ).to(device)
    if args.warm_start_scorer is not None:
        payload = torch.load(args.warm_start_scorer, map_location="cpu", weights_only=False)
        state = payload.get("model", payload)
        scorer.load_state_dict(state, strict=False)
    mean, scale = fit_normalization(scorer, world_model, train_loader, device)
    scorer.set_normalization(mean, scale)
    optimizer = torch.optim.AdamW(
        scorer.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    history: list[dict[str, Any]] = []
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(
            scorer,
            world_model,
            train_loader,
            optimizer,
            device,
            args.steps_per_epoch,
            True,
        )
        val_loss = {
            domain: run_epoch(
                scorer,
                world_model,
                loader,
                optimizer,
                device,
                args.eval_steps,
                False,
            )
            for domain, loader in val_loaders.items()
        }
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_macro_loss": float(np.mean(list(val_loss.values()))),
        }
        history.append(row)
        (args.output_dir / "metrics.jsonl").open("a", encoding="utf-8").write(
            json.dumps(row, sort_keys=True) + "\n"
        )
        torch.save(
            {
                "schema": "trajectory_quality_scorer_v2_checkpoint",
                "target_definition": "latent_counterfactual_proxy_v1",
                "model": scorer.state_dict(),
                "model_config": asdict(scorer.config),
                "split_lock": str(args.split_lock),
                "world_model_checkpoint": str(args.world_model_checkpoint),
                "world_model_checkpoint_sha256": __import__("hashlib").sha256(
                    args.world_model_checkpoint.read_bytes()
                ).hexdigest(),
                "seed": args.seed,
                "epoch": epoch,
                "domains": list(DOMAINS),
                "train_window_counts": {
                    domain: len(datasets[domain][0].windows) for domain in DOMAINS
                },
            },
            args.output_dir / "best.pt",
        )
    summary = {
        "status": "complete",
        "schema": "v70_tridomain_selector_training_summary",
        "seed": args.seed,
        "domains": list(DOMAINS),
        "target_definition": "latent_counterfactual_proxy_v1",
        "history": history,
        "best_checkpoint": str(args.output_dir / "best.pt"),
        "world_model_checkpoint": str(args.world_model_checkpoint),
        "split_lock": str(args.split_lock),
        "train_window_counts": {
            domain: len(datasets[domain][0].windows) for domain in DOMAINS
        },
        "val_window_counts": {
            domain: len(datasets[domain][1].windows) for domain in DOMAINS
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
