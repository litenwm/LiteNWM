#!/usr/bin/env python3
"""Train the v70 LiteTokenFormer successor on domain-pure tri-domain batches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO_ROOT), str(REPO_ROOT / "src"), str(REPO_ROOT / "tools/recon")]

from tools.recon import train_litetokenformer_wm_v2_4ctx as base  # noqa: E402
from tools.tridomain_v71.encode_dinov3_adapted_tridomain import validate_successor_parent_reuse  # noqa: E402


class ScheduledDomainLoader:
    """Cycle shard-aware loaders under a fixed, domain-pure schedule."""

    def __init__(self, loaders: dict[str, DataLoader], schedule: list[str]) -> None:
        if not schedule or any(domain not in loaders for domain in schedule):
            raise ValueError(f"invalid domain schedule {schedule} for {sorted(loaders)}")
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


def preflight(args: argparse.Namespace) -> None:
    if args.split_lock is None:
        raise ValueError("tri-domain training requires --split-lock")
    lock, lock_sha = base.load_split_lock(args.split_lock)
    roots = {
        "recon": args.latent_root,
        "scand": args.scand_latent_root,
        "sacson": args.sacson_latent_root,
    }
    if any(root is None for root in roots.values()):
        raise ValueError("RECON, SCAND, and SACSoN latent roots are all mandatory")
    for dataset, root in roots.items():
        metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
        if metadata.get("split_lock_sha256") != lock_sha:
            raise ValueError(f"{dataset} latent root is not bound to active split lock")
        splits = json.loads((root / "splits.json").read_text(encoding="utf-8"))
        if splits.get("test"):
            raise ValueError(f"{dataset} training root exposes outer test identities")
        base.validate_latent_root(root, {**lock[dataset]["splits"], "test": []})


def make_loaders(args: argparse.Namespace):
    def build(root: Path, seed_offset: int):
        train = base.TokenSequenceStreamingDataset(
            root,
            "train",
            args.history_frames,
            args.future_frames,
            args.stream_cache_shards,
        )
        val = base.TokenSequenceStreamingDataset(
            root,
            "val",
            args.history_frames,
            args.future_frames,
            args.stream_cache_shards,
        )
        train_sampler = base.ShardAwareBatchSampler(
            train,
            args.batch_size,
            drop_last=args.train_drop_last,
            seed=args.seed + seed_offset,
        )
        val_sampler = base.ShardAwareBatchSampler(
            val,
            args.eval_batch_size,
            drop_last=False,
            seed=10000 + args.seed + seed_offset,
        )
        return (
            train,
            val,
            DataLoader(
                train,
                batch_sampler=train_sampler,
                num_workers=args.num_workers,
                pin_memory=True,
            ),
            DataLoader(
                val,
                batch_sampler=val_sampler,
                num_workers=args.num_workers,
                pin_memory=True,
            ),
        )

    roots = {
        "recon": (args.latent_root, 0),
        "scand": (args.scand_latent_root, 100000),
        "sacson": (args.sacson_latent_root, 200000),
    }
    train_datasets = {}
    val_datasets = {}
    train_loaders = {}
    val_loaders = {}
    for domain, (root, offset) in roots.items():
        train, val, train_loader, val_loader = build(root, offset)
        train_datasets[domain] = train
        val_datasets[domain] = val
        train_loaders[domain] = train_loader
        val_loaders[domain] = val_loader
    return (
        train_datasets,
        val_datasets,
        ScheduledDomainLoader(train_loaders, args.domain_schedule),
        val_loaders,
    )


def main() -> None:
    original_parse_args = base.parse_args
    original_argv = list(sys.argv)
    wrapper = argparse.ArgumentParser(add_help=False)
    wrapper.add_argument("--sacson-latent-root", type=Path, required=True)
    wrapper.add_argument(
        "--domain-schedule",
        default="recon,recon,scand,sacson",
        help="comma-separated deterministic batch schedule",
    )
    wrapper_args, remaining = wrapper.parse_known_args()

    def parse_args():
        sys.argv[:] = [original_argv[0], *remaining]
        try:
            args = original_parse_args()
        finally:
            sys.argv[:] = original_argv
        args.sacson_latent_root = wrapper_args.sacson_latent_root
        args.domain_schedule = [
            value.strip() for value in wrapper_args.domain_schedule.split(",") if value.strip()
        ]
        preflight(args)
        return args

    base.parse_args = parse_args
    base.make_loaders = make_loaders
    base.validate_parent_split_reuse = validate_successor_parent_reuse
    base.main()


if __name__ == "__main__":
    main()
