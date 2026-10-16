#!/usr/bin/env python3
"""Train one unchanged LTWM with natural cross-recording branch supervision."""

from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(Path(__file__).resolve().parent)]

from recon_latents.losses_pseudobranch_v1 import (  # noqa: E402
    PseudoBranchLossWeights,
    compute_pseudobranch_world_model_loss,
)
import diagnose_natural_latent_branching_v1 as branching  # noqa: E402
import train_litetokenformer_wm_action_residual_v1 as residual_entry  # noqa: E402


trainer = residual_entry.trainer
_ORIGINAL_PARSE_ARGS = residual_entry._ORIGINAL_PARSE_ARGS
_BRANCH_CONTRACT: Path | None = None


def parse_args():
    global _BRANCH_CONTRACT
    if "--branch-contract" not in sys.argv:
        raise ValueError("--branch-contract is required")
    position = sys.argv.index("--branch-contract")
    _BRANCH_CONTRACT = Path(sys.argv[position + 1]).resolve()
    del sys.argv[position : position + 2]
    args = _ORIGINAL_PARSE_ARGS()
    if args.objective_revision != "base":
        raise ValueError("pseudo-branch entry requires --objective-revision base")
    args.objective_revision = "pure_future_latent_natural_pseudobranch_v1"
    return args


def file_sha256(path: Path) -> str:
    return branching.sha256_file(path)


def load_branch_contract() -> tuple[dict, dict]:
    assert _BRANCH_CONTRACT is not None
    contract = branching.read_json(_BRANCH_CONTRACT)
    if contract.get("status") != "frozen_before_single_pseudobranch_training":
        raise RuntimeError("pseudo-branch training contract not frozen")
    diagnostic_entry = contract["inputs"]["branching_diagnostic"]
    diagnostic_path = (ROOT / diagnostic_entry["path"]).resolve()
    if file_sha256(diagnostic_path) != diagnostic_entry["sha256"]:
        raise RuntimeError("branching diagnostic changed")
    diagnostic = branching.read_json(diagnostic_path)
    if not diagnostic.get("go"):
        raise RuntimeError("branching diagnostic did not authorize training")
    return contract, diagnostic


def build_directed_pairs(
    latent_root: Path,
    split: str,
    contract: dict,
    diagnostic: dict,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, float, dict]:
    windows = branching.load_windows(latent_root, split)
    signature = branching.build_signature(windows)
    _, recording_index = np.unique(
        windows["recording"].astype(str),
        return_inverse=True,
    )
    neighbor, similarity = branching.nearest_cross_recording(
        signature,
        recording_index,
        int(contract["pair_construction"]["top_k"]),
        int(contract["pair_construction"]["query_chunk_size"]),
        device,
    )
    state_threshold = float(
        diagnostic["thresholds_derived_from_random_cross_recording_pairs"][
            "state_similarity"
        ]
    )
    pairs, pair_similarity = branching.mutual_pairs(
        neighbor,
        similarity,
        state_threshold,
    )
    action_scale = np.std(windows["actions"], axis=0)
    action_scale = np.maximum(action_scale, 1.0e-4)
    action_difference = (
        windows["actions"][pairs[:, 0]] - windows["actions"][pairs[:, 1]]
    ) / action_scale
    action_distance = np.sqrt(np.mean(action_difference * action_difference, axis=1))
    diverse_threshold = float(
        diagnostic["thresholds_derived_from_random_cross_recording_pairs"][
            "diverse_action_distance"
        ]
    )
    diverse = action_distance >= diverse_threshold
    pairs = pairs[diverse]
    pair_similarity = pair_similarity[diverse]
    oriented = np.concatenate((pairs, pairs[:, ::-1]), axis=0)
    oriented_similarity = np.concatenate((pair_similarity, pair_similarity), axis=0)
    max_per_recording = int(
        contract["pair_construction"]["maximum_directed_pairs_per_recording"]
    )
    selected = []
    for recording in np.unique(recording_index):
        local = np.flatnonzero(recording_index[oriented[:, 0]] == recording)
        order = local[np.argsort(-oriented_similarity[local], kind="stable")]
        used_anchor = set()
        for index in order:
            anchor = int(oriented[index, 0])
            if anchor in used_anchor:
                continue
            selected.append(int(index))
            used_anchor.add(anchor)
            if len(used_anchor) >= max_per_recording:
                break
    selected = np.asarray(selected, dtype=np.int64)
    oriented = oriented[selected]
    oriented_similarity = oriented_similarity[selected]
    order = np.lexsort(
        (
            oriented[:, 0],
            recording_index[oriented[:, 0]],
        )
    )
    oriented = oriented[order]
    oriented_similarity = oriented_similarity[order]
    residual_scale = float(
        np.sqrt(np.mean(windows["future_residual"] ** 2))
    )
    audit = {
        "mutual_high_similarity_pairs": int(len(pair_similarity) + int((~diverse).sum())),
        "diverse_undirected_pairs": int(len(pairs)),
        "selected_directed_pairs": int(len(oriented)),
        "selected_anchor_windows": int(len(np.unique(oriented[:, 0]))),
        "selected_recordings": int(
            len(np.unique(recording_index[oriented[:, 0]]))
        ),
        "state_similarity_minimum": state_threshold,
        "diverse_action_distance_minimum": diverse_threshold,
        "maximum_directed_pairs_per_recording": max_per_recording,
        "residual_scale": residual_scale,
    }
    return windows, oriented, oriented_similarity, residual_scale, audit


class NaturalBranchDataset(Dataset):
    def __init__(
        self,
        latent_root: Path,
        split: str,
        contract: dict,
        diagnostic: dict,
        device: torch.device,
        stream_cache_shards: int,
    ) -> None:
        self.base = trainer.TokenSequenceStreamingDataset(
            latent_root,
            split,
            4,
            4,
            stream_cache_shards,
        )
        (
            self.windows,
            self.pairs,
            self.pair_similarity,
            self.residual_scale,
            self.pair_audit,
        ) = build_directed_pairs(
            latent_root,
            split,
            contract,
            diagnostic,
            device,
        )
        base_trajectory = np.asarray(
            [
                self.base.trajectories[trajectory_index].trajectory_id
                for trajectory_index, _ in self.base.windows
            ]
        )
        base_timestep = np.asarray(
            [history_start + 3 for _, history_start in self.base.windows]
        )
        if not np.array_equal(base_trajectory, self.windows["trajectory"]):
            raise RuntimeError("branch/base trajectory ordering changed")
        if not np.array_equal(base_timestep, self.windows["timestep"]):
            raise RuntimeError("branch/base timestep ordering changed")
        threshold = float(
            diagnostic["thresholds_derived_from_random_cross_recording_pairs"][
                "state_similarity"
            ]
        )
        self.similarity_weight = np.clip(
            (self.pair_similarity - threshold) / max(1.0 - threshold, 1.0e-6),
            0.10,
            1.0,
        ).astype(np.float32)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int):
        anchor, partner = self.pairs[index]
        item = self.base[int(anchor)]
        item["branch_actions"] = torch.from_numpy(
            self.windows["actions"][int(partner)].reshape(4, 3).copy()
        )
        item["observed_branch_delta"] = torch.from_numpy(
            (
                self.windows["future_residual"][int(partner)]
                - self.windows["future_residual"][int(anchor)]
            ).copy()
        )
        item["branch_similarity_weight"] = torch.tensor(
            self.similarity_weight[index],
            dtype=torch.float32,
        )
        return item

    def summary(self):
        return {
            "dataset_class": type(self).__name__,
            "split": self.base.split,
            "num_shards": len(self.base.trajectories),
            "num_windows": len(self.base.windows),
            "natural_branch_pair_audit": self.pair_audit,
            "outcome_labels_used": False,
        }

    def cache_stats(self):
        return self.base.cache_stats()


class AnchorShardBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        dataset: NaturalBranchDataset,
        batch_size: int,
        seed: int,
    ) -> None:
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.epoch = 0
        grouped = defaultdict(list)
        for pair_index, (anchor, _) in enumerate(dataset.pairs):
            trajectory_index = dataset.base.windows[int(anchor)][0]
            grouped[int(trajectory_index)].append(pair_index)
        self.grouped = dict(grouped)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        groups = np.asarray(list(self.grouped), dtype=np.int64)
        rng.shuffle(groups)
        for group in groups:
            local = np.asarray(self.grouped[int(group)], dtype=np.int64)
            rng.shuffle(local)
            for start in range(0, len(local), self.batch_size):
                batch = local[start : start + self.batch_size].tolist()
                if len(batch) == self.batch_size:
                    yield batch

    def __len__(self):
        return sum(len(value) // self.batch_size for value in self.grouped.values())


def make_loaders(args):
    contract, diagnostic = load_branch_contract()
    device = torch.device("cuda")
    train = NaturalBranchDataset(
        args.latent_root,
        "train",
        contract,
        diagnostic,
        device,
        args.stream_cache_shards,
    )
    train_sampler = AnchorShardBatchSampler(train, args.batch_size, args.seed)
    train_loader = DataLoader(
        train,
        batch_sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val = trainer.TokenSequenceStreamingDataset(
        args.latent_root,
        "val",
        4,
        4,
        args.stream_cache_shards,
    )
    val_sampler = trainer.ShardAwareBatchSampler(
        val,
        args.eval_batch_size,
        drop_last=False,
        seed=10000 + args.seed,
    )
    val_loader = DataLoader(
        val,
        batch_sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    return (
        {"recon": train},
        {"recon": val},
        trainer.TaggedLoader(train_loader, "recon"),
        {"recon": val_loader},
    )


def train_epoch(model, loader, optimizer, scaler, args, device, weights, epoch):
    model.train()
    rows = []
    domains = Counter()
    started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    residual_scale = loader.loader.dataset.residual_scale
    for batch_index, batch in enumerate(loader):
        if batch_index >= args.batches_per_epoch:
            break
        domains[str(batch.get("domain", "recon"))] += 1
        history = batch["history_tokens"].to(device, non_blocking=True)
        target = batch["future_tokens"].to(device, non_blocking=True)
        actions = batch["actions"].to(device, non_blocking=True)
        branch_actions = batch["branch_actions"].to(device, non_blocking=True)
        branch_delta = batch["observed_branch_delta"].to(device, non_blocking=True)
        similarity_weight = batch["branch_similarity_weight"].to(
            device,
            non_blocking=True,
        )
        with trainer.amp_context(args.amp):
            output = model(history, actions)
            branch_output = model(history, branch_actions)
            prefix = None
            if args.consistency_every > 0 and batch_index % args.consistency_every == 0:
                prefix_actions = actions.clone()
                prefix_actions[:, 3:] = 0
                prefix = model(history, prefix_actions)
            loss, components = compute_pseudobranch_world_model_loss(
                output,
                target,
                history,
                actions,
                weights=weights,
                corrupted_output=branch_output,
                prefix_output=prefix,
                observed_branch_delta=branch_delta,
                branch_similarity_weight=similarity_weight,
                residual_scale=residual_scale,
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
        optimizer.zero_grad(set_to_none=True)
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
        raise RuntimeError("pseudo-branch loader yielded no batches")
    return {
        key: float(np.mean([row[key] for row in rows]))
        for key in rows[0]
    } | {
        "batches": len(rows),
        "domain_batches": dict(domains),
        "runtime_seconds": time.perf_counter() - started,
    }


trainer.parse_args = parse_args
trainer.make_loaders = make_loaders
trainer.train_epoch = train_epoch
trainer.V2LossWeights = PseudoBranchLossWeights
trainer.compute_world_model_v2_loss = compute_pseudobranch_world_model_loss
trainer.step_metrics = residual_entry.step_metrics


if __name__ == "__main__":
    random.seed(0)
    trainer.main()
