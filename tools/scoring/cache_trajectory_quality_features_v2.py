#!/usr/bin/env python3
"""Cache clean scorer-v2 features from one frozen DINOv3-LTWM checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from recon_latents.models_v2 import (  # noqa: E402
    LiteTokenFormerConfig,
    LiteTokenFormerWMV2,
)
from scoring.trajectory_quality_scorer_v2 import (  # noqa: E402
    TrajectoryQualityScorerConfig,
    TrajectoryQualityScorerV2,
)


from scoring.trajectory_artifact_contract_v2 import (  # noqa: E402
    INNER_PARTITIONS,
    load_inner_manifest,
    validate_label_artifact,
    validate_outer_train_membership,
    validate_proposal_artifact,
)

HISTORY = 4
HORIZON = 4
K = 16
EPS = 1.0e-8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("recon", "scand"), required=True)
    parser.add_argument(
        "--partition",
        choices=INNER_PARTITIONS,
        required=True,
    )
    parser.add_argument("--latent-root", type=Path, required=True)
    parser.add_argument("--proposals", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--wm-checkpoint", type=Path, required=True)
    parser.add_argument("--inner-manifest", type=Path, required=True)
    parser.add_argument("--split-lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--candidate-chunk", type=int, default=4)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def validate_inputs(
    args: argparse.Namespace,
    proposal: dict[str, np.ndarray],
    labels: dict[str, np.ndarray],
    rows: list[dict[str, Any]],
) -> tuple[dict[str, str], dict[str, Any]]:
    split_lock_sha = file_sha256(args.split_lock)
    validate_outer_train_membership(args.split_lock, rows, dataset=args.dataset)
    proposal_metadata = validate_proposal_artifact(
        proposal,
        rows,
        dataset=args.dataset,
        partition=args.partition,
        split_lock_sha256=split_lock_sha,
    )
    label_metadata = validate_label_artifact(
        labels,
        proposal,
        dataset=args.dataset,
        partition=args.partition,
        split_lock_sha256=split_lock_sha,
    )
    manifest_sha = file_sha256(args.inner_manifest)
    if proposal_metadata.get("inner_manifest_sha256") != manifest_sha:
        raise RuntimeError("proposal is not bound to this inner manifest")
    if label_metadata.get("inner_manifest_sha256") != manifest_sha:
        raise RuntimeError("labels are not bound to this inner manifest")
    if label_metadata.get("proposal_sha256") != file_sha256(args.proposals):
        raise RuntimeError("labels are not bound to this proposal artifact")
    required_proposal = {
        "window_uid",
        "partition",
        "trajectory_id",
        "current_t",
        "goal_t",
        "history_timesteps",
        "actions_h4",
    }
    required_labels = {"window_uid", "ate", "rtp"}
    if missing := required_proposal.difference(proposal):
        raise RuntimeError(f"proposal is missing arrays: {sorted(missing)}")
    if missing := required_labels.difference(labels):
        raise RuntimeError(f"labels are missing arrays: {sorted(missing)}")
    if str(proposal["partition"].item()) != args.partition:
        raise RuntimeError("proposal partition mismatch")
    if "dataset" in proposal and str(proposal["dataset"].item()) != args.dataset:
        raise RuntimeError("proposal dataset mismatch")
    if not np.array_equal(proposal["window_uid"], labels["window_uid"]):
        raise RuntimeError("proposal/label window UID mismatch")
    count = len(proposal["window_uid"])
    if proposal["actions_h4"].shape != (count, K, HORIZON, 3):
        raise RuntimeError("expected proposal actions [N,16,4,3]")
    if labels["ate"].shape != (count, K) or labels["rtp"].shape != (count, K):
        raise RuntimeError("expected ATE/RTP labels [N,16]")
    if np.any(proposal["goal_t"] != proposal["current_t"] + HORIZON):
        raise RuntimeError("goal offset is not exactly t+4")
    expected_history = np.stack(
        [
            proposal["current_t"] - 3,
            proposal["current_t"] - 2,
            proposal["current_t"] - 1,
            proposal["current_t"],
        ],
        axis=1,
    )
    if not np.array_equal(proposal["history_timesteps"], expected_history):
        raise RuntimeError("history is not [t-3,t-2,t-1,t]")
    metadata = json.loads((args.latent_root / "metadata.json").read_text())
    split_lock_sha = file_sha256(args.split_lock)
    if metadata.get("split_lock_sha256") != split_lock_sha:
        raise RuntimeError("latent root/split-lock mismatch")
    if metadata.get("split_lock_dataset") != args.dataset:
        raise RuntimeError("latent root dataset mismatch")
    splits = json.loads((args.latent_root / "splits.json").read_text())
    if splits.get("test"):
        raise RuntimeError("feature cache refuses a latent root exposing test")
    allowed_ids = set(map(str, splits["train"]))
    proposal_ids = set(proposal["trajectory_id"].astype(str))
    if not proposal_ids.issubset(allowed_ids):
        raise RuntimeError("feature cache contains a non-outer-train trajectory")
    shard_map = {
        str(row["trajectory_id"]): str(row["shard_file"])
        for row in metadata["trajectories"]
    }
    return shard_map, metadata


def load_window_batch(
    latent_root: Path,
    shard_map: dict[str, str],
    proposal: dict[str, np.ndarray],
    shard_cache: dict[str, np.ndarray],
    start: int,
    end: int,
) -> tuple[np.ndarray, np.ndarray]:
    history = []
    goal = []
    for index in range(start, end):
        trajectory = str(proposal["trajectory_id"][index])
        if trajectory not in shard_cache:
            shard_cache.clear()
            with np.load(
                latent_root / "shards" / shard_map[trajectory],
                allow_pickle=False,
            ) as shard:
                shard_cache[trajectory] = shard["patch_latents"]
        tokens = shard_cache[trajectory]
        history.append(
            tokens[proposal["history_timesteps"][index]]
            .astype(np.float32)
            .reshape(HISTORY, 256, 384)
        )
        goal.append(
            tokens[int(proposal["goal_t"][index])]
            .astype(np.float32)
            .reshape(256, 384)
        )
    return np.stack(history), np.stack(goal)


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.batch_size <= 0 or args.candidate_chunk <= 0:
        raise ValueError("batch/chunk sizes must be positive")
    proposal = load_npz(args.proposals)
    labels = load_npz(args.labels)
    rows = load_inner_manifest(
        args.inner_manifest,
        dataset=args.dataset,
        partition=args.partition,
    )
    shard_map, latent_metadata = validate_inputs(args, proposal, labels, rows)
    checkpoint_sha = file_sha256(args.wm_checkpoint)
    checkpoint = torch.load(
        args.wm_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    split_lock_sha = file_sha256(args.split_lock)
    if checkpoint.get("split_lock_sha256") != split_lock_sha:
        raise RuntimeError("world-model checkpoint/split-lock mismatch")
    if checkpoint.get("temporal_protocol") != {
        "history_frames": HISTORY,
        "future_frames": HORIZON,
    }:
        raise RuntimeError("world-model checkpoint temporal protocol mismatch")
    model = LiteTokenFormerWMV2(LiteTokenFormerConfig(**checkpoint["config"])).to(
        args.device
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval().requires_grad_(False)
    extractor = (
        TrajectoryQualityScorerV2(
            TrajectoryQualityScorerConfig(
                hidden_dim=640,
                hidden_layers=3,
                dropout=0.10,
                use_predicted_future=True,
            )
        )
        .to(args.device)
        .eval()
    )

    feature_batches = []
    terminal_batches = []
    shard_cache: dict[str, np.ndarray] = {}
    started = time.perf_counter()
    actions_all = proposal["actions_h4"].astype(np.float32)
    with torch.inference_mode():
        for start in range(0, len(actions_all), args.batch_size):
            end = min(start + args.batch_size, len(actions_all))
            history_np, goal_np = load_window_batch(
                args.latent_root,
                shard_map,
                proposal,
                shard_cache,
                start,
                end,
            )
            history = torch.from_numpy(history_np).to(args.device)
            goal = torch.from_numpy(goal_np).to(args.device)
            actions = torch.from_numpy(actions_all[start:end]).to(args.device)
            chunks = []
            terminal_chunks = []
            for candidate_start in range(0, K, args.candidate_chunk):
                candidate_end = min(candidate_start + args.candidate_chunk, K)
                chunk_actions = actions[:, candidate_start:candidate_end]
                context = (
                    torch.autocast("cuda", dtype=torch.bfloat16)
                    if args.device == "cuda"
                    else torch.autocast("cpu", enabled=False)
                )
                with context:
                    future = model(history, chunk_actions)["predicted_tokens"]
                future = future.float()
                chunks.append(
                    extractor.extract_features(
                        current_latent=history[:, -1],
                        goal_latent=goal,
                        candidate_actions=chunk_actions,
                        predicted_future=future,
                    ).cpu()
                )
                terminal_nmse = (future[:, :, -1] - goal[:, None]).square().mean(
                    dim=(-1, -2)
                ) / goal[:, None].square().mean(dim=(-1, -2)).clamp_min(EPS)
                terminal_chunks.append(terminal_nmse.cpu())
            feature_batches.append(torch.cat(chunks, dim=1).numpy().astype(np.float16))
            terminal_batches.append(
                torch.cat(terminal_chunks, dim=1).numpy().astype(np.float32)
            )
            print(
                json.dumps(
                    {
                        "stage": "cache_scorer_v2_features",
                        "dataset": args.dataset,
                        "partition": args.partition,
                        "completed": end,
                        "total": len(actions_all),
                        "elapsed_sec": time.perf_counter() - started,
                    }
                ),
                flush=True,
            )
    features = np.concatenate(feature_batches)
    terminal_nmse = np.concatenate(terminal_batches)
    target = np.stack(
        (
            labels["ate"].astype(np.float32),
            labels["rtp"].astype(np.float32),
        ),
        axis=-1,
    )
    valid = (
        np.isfinite(features).all(axis=-1)
        & np.isfinite(target).all(axis=-1)
        & np.isfinite(actions_all).all(axis=(-1, -2))
    )
    if features.shape != (
        len(actions_all),
        K,
        extractor.input_dim,
    ):
        raise RuntimeError(f"unexpected feature shape: {features.shape}")
    if not valid.any(axis=1).all():
        raise RuntimeError("at least one window has no valid candidate")
    metadata = {
        "created_at": utc_now(),
        "decision": "TRAJECTORY_QUALITY_FEATURE_CACHE_V2_READY",
        "schema": "trajectory_quality_feature_cache_v2",
        "dataset": args.dataset,
        "partition": args.partition,
        "history": HISTORY,
        "horizon": HORIZON,
        "goal_offset": HORIZON,
        "candidates": K,
        "feature_dim": extractor.input_dim,
        "allowed_inputs": [
            "current_dinov3_latent",
            "goal_dinov3_latent",
            "candidate_actions_h4",
            "frozen_ltwm_predicted_future_h1_h4",
        ],
        "forbidden_inputs": [
            "ATE",
            "RTP",
            "logged_pose",
            "trajectory_id",
            "route_id",
            "candidate_index",
            "candidate_family",
            "dataset_id",
            "optimizer_iteration",
        ],
        "labels_stored_as_training_sidecar_not_features": True,
        "world_model_checkpoint": str(args.wm_checkpoint),
        "world_model_checkpoint_sha256": checkpoint_sha,
        "world_model_frozen": True,
        "world_model_trainable_parameters": 0,
        "split_lock_sha256": split_lock_sha,
        "inner_manifest_sha256": file_sha256(args.inner_manifest),
        "latent_metadata_sha256": file_sha256(args.latent_root / "metadata.json"),
        "proposal_sha256": file_sha256(args.proposals),
        "labels_sha256": file_sha256(args.labels),
        "outer_val_accessed": False,
        "test_accessed": False,
        "elapsed_sec": time.perf_counter() - started,
    }
    write_npz(
        args.output,
        window_uid=proposal["window_uid"].astype(str),
        dataset=np.asarray(args.dataset),
        partition=np.asarray(args.partition),
        trajectory_id=proposal["trajectory_id"].astype(str),
        route_id=proposal["route_id"].astype(str),
        current_t=proposal["current_t"].astype(np.int64),
        goal_t=proposal["goal_t"].astype(np.int64),
        features=features,
        target=target,
        actions_h4=actions_all,
        terminal_nmse=terminal_nmse,
        valid=valid,
        metadata=np.asarray(canonical_json(metadata)),
    )
    metadata["output_sha256"] = file_sha256(args.output)
    write_json(Path(str(args.output) + ".schema.json"), metadata)
    print(json.dumps(metadata, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
