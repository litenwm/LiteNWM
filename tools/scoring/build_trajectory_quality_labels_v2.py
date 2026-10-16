#!/usr/bin/env python3
"""Build strict H4 ATE/RTP labels for one frozen proposal artifact.

This is deliberately a separate, label-only process.  Proposal generation
must finish before this tool opens any ``traj_data.pkl`` file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from scoring.trajectory_artifact_contract_v2 import (  # noqa: E402
    INNER_PARTITIONS,
    LABEL_SCHEMA,
    file_sha256,
    label_arrays,
    load_inner_manifest,
    load_npz,
    local_future_positions,
    trajectory_quality_targets,
    validate_outer_train_membership,
    validate_label_artifact,
    validate_proposal_artifact,
    write_json_atomic,
    write_npz_atomic,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("recon", "scand"), required=True)
    parser.add_argument("--partition", choices=INNER_PARTITIONS, required=True)
    parser.add_argument("--trajectory-root", type=Path, required=True)
    parser.add_argument("--inner-manifest", type=Path, required=True)
    parser.add_argument("--proposals", type=Path, required=True)
    parser.add_argument("--split-lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_pose(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load pose arrays from a trusted, local NWM trajectory pickle."""

    with path.open("rb") as handle:
        try:
            payload = pickle.load(handle)
        except ModuleNotFoundError as error:
            if not str(error.name).startswith("numpy._core"):
                raise

            class NumpyCompatUnpickler(pickle.Unpickler):
                def find_class(self, module: str, name: str) -> Any:
                    return super().find_class(
                        module.replace("numpy._core", "numpy.core"), name
                    )

            handle.seek(0)
            payload = NumpyCompatUnpickler(handle).load()
    if not isinstance(payload, dict) or not {"position", "yaw"}.issubset(payload):
        raise ValueError(f"invalid pose pickle schema: {path}")
    positions = np.asarray(payload["position"], dtype=np.float64)
    yaw = np.asarray(payload["yaw"], dtype=np.float64).reshape(-1)
    if positions.ndim != 2 or positions.shape[1] < 2 or len(positions) != len(yaw):
        raise ValueError(f"invalid position/yaw shapes: {path}")
    if not np.isfinite(positions).all() or not np.isfinite(yaw).all():
        raise ValueError(f"non-finite pose values: {path}")
    return positions[:, :2], yaw


def resolve_pose_paths(root: Path, trajectory_ids: list[str]) -> dict[str, Path]:
    resolved_root = root.resolve(strict=True)
    result: dict[str, Path] = {}
    for trajectory_id in sorted(set(trajectory_ids)):
        if not trajectory_id or trajectory_id in {".", ".."}:
            raise ValueError(f"unsafe trajectory id: {trajectory_id!r}")
        candidate = (resolved_root / trajectory_id / "traj_data.pkl").resolve(
            strict=True
        )
        try:
            candidate.relative_to(resolved_root)
        except ValueError as error:
            raise ValueError(
                f"pose path escapes trajectory root: {candidate}"
            ) from error
        if not candidate.is_file():
            raise FileNotFoundError(candidate)
        result[trajectory_id] = candidate
    return result


def pose_bundle_sha256(paths: dict[str, Path]) -> str:
    digest = hashlib.sha256()
    for trajectory_id, path in sorted(paths.items()):
        digest.update(trajectory_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_sha256(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    rows = load_inner_manifest(
        args.inner_manifest,
        dataset=args.dataset,
        partition=args.partition,
    )
    validate_outer_train_membership(args.split_lock, rows, dataset=args.dataset)
    proposal = load_npz(args.proposals)
    split_lock_sha = file_sha256(args.split_lock)
    proposal_metadata = validate_proposal_artifact(
        proposal,
        rows,
        dataset=args.dataset,
        partition=args.partition,
        split_lock_sha256=split_lock_sha,
    )
    manifest_sha = file_sha256(args.inner_manifest)
    if proposal_metadata.get("inner_manifest_sha256") != manifest_sha:
        raise RuntimeError("proposal is not bound to this inner manifest")

    trajectory_ids = [str(row["trajectory_id"]) for row in rows]
    pose_paths = resolve_pose_paths(args.trajectory_root, trajectory_ids)
    pose_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    ground_truth = []
    for row in rows:
        trajectory_id = str(row["trajectory_id"])
        if trajectory_id not in pose_cache:
            pose_cache[trajectory_id] = load_pose(pose_paths[trajectory_id])
        positions, yaw = pose_cache[trajectory_id]
        ground_truth.append(
            local_future_positions(
                positions,
                yaw,
                current_t=int(row["current_t"]),
            )
        )
    ground_truth_array = np.stack(ground_truth).astype(np.float32)
    ate, rtp = trajectory_quality_targets(proposal["actions_h4"], ground_truth_array)
    metadata = {
        "schema": LABEL_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset,
        "partition": args.partition,
        "windows": len(rows),
        "candidates": int(ate.shape[1]),
        "horizon": 4,
        "split_lock_sha256": split_lock_sha,
        "inner_manifest_sha256": manifest_sha,
        "proposal_sha256": file_sha256(args.proposals),
        "pose_bundle_sha256": pose_bundle_sha256(pose_paths),
        "pose_trajectories": len(pose_paths),
        "target_contract": {
            "ate": "mean_h1_h4_euclidean_position_error_meters",
            "rtp": "(start_goal_distance-final_goal_distance)/max(start_goal_distance,1e-8)",
            "rtp_clipped": False,
        },
        "label_only_inputs": ["logged_position", "logged_yaw"],
        "planner_or_scorer_inputs": False,
        "outer_validation_accessed": False,
        "test_accessed": False,
    }
    arrays = label_arrays(
        proposal,
        ate,
        rtp,
        dataset=args.dataset,
        partition=args.partition,
        metadata=metadata,
    )
    validate_label_artifact(
        arrays,
        proposal,
        dataset=args.dataset,
        partition=args.partition,
        split_lock_sha256=split_lock_sha,
    )
    write_npz_atomic(args.output, **arrays)
    metadata["output_sha256"] = file_sha256(args.output)
    write_json_atomic(Path(str(args.output) + ".schema.json"), metadata)
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
