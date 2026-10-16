"""Small IO helpers for the local RECON dataset layout."""

from __future__ import annotations

import json
import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ReconTrajectory:
    trajectory_id: str
    path: Path
    image_paths: list[Path]
    positions: np.ndarray
    yaw: np.ndarray
    action_labels: list[str]
    atomic_labels_source: str = "provided"


def load_recon_trajectories(
    recon_root: str | Path,
    max_trajectories: int | None = None,
    allow_missing_atomic_labels: bool = False,
) -> tuple[list[ReconTrajectory], list[dict[str, Any]]]:
    """Load complete trajectory records and report skipped folders."""
    root = Path(recon_root)
    if not root.exists():
        raise FileNotFoundError(f"RECON root does not exist: {root}")

    records: list[ReconTrajectory] = []
    skipped: list[dict[str, Any]] = []
    for traj_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        if max_trajectories is not None and len(records) >= max_trajectories:
            break
        image_paths = sorted(traj_dir.glob("*.jpg"), key=lambda path: int(path.stem))
        pkl_path = traj_dir / "traj_data.pkl"
        if not pkl_path.exists():
            skipped.append(
                {"trajectory_id": traj_dir.name, "reason": "missing traj_data.pkl", "num_images": len(image_paths)}
            )
            continue
        with pkl_path.open("rb") as file:
            data = pickle.load(file)
        positions = np.asarray(data["position"], dtype=np.float32)
        yaw = np.asarray(data["yaw"], dtype=np.float32)
        if "atomic_labels" in data:
            labels = list(data["atomic_labels"])
            atomic_labels_source = "provided"
        elif allow_missing_atomic_labels:
            labels = ["unknown"] * min(len(image_paths), len(positions), len(yaw))
            atomic_labels_source = "missing_filled_unknown"
        else:
            skipped.append(
                {
                    "trajectory_id": traj_dir.name,
                    "reason": "missing atomic_labels",
                    "num_images": len(image_paths),
                    "num_positions": len(positions),
                    "num_yaw": len(yaw),
                }
            )
            continue
        lengths = {len(image_paths), len(positions), len(yaw), len(labels)}
        if len(lengths) != 1:
            skipped.append(
                {
                    "trajectory_id": traj_dir.name,
                    "reason": "length mismatch",
                    "num_images": len(image_paths),
                    "num_positions": len(positions),
                    "num_yaw": len(yaw),
                    "num_labels": len(labels),
                }
            )
            continue
        if len(image_paths) < 2:
            skipped.append({"trajectory_id": traj_dir.name, "reason": "fewer than 2 frames"})
            continue
        records.append(ReconTrajectory(traj_dir.name, traj_dir, image_paths, positions, yaw, labels, atomic_labels_source))
    return records, skipped


def make_splits(
    trajectory_ids: list[str],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 17,
) -> dict[str, list[str]]:
    """Create trajectory-level splits using a seeded shuffle over IDs."""
    total = train_ratio + val_ratio + test_ratio
    if total <= 0:
        raise ValueError("split ratios must sum to a positive value")
    train_ratio = train_ratio / total
    val_ratio = val_ratio / total

    ids = sorted(trajectory_ids)
    random.Random(seed).shuffle(ids)
    n = len(ids)
    train_end = int(round(n * train_ratio))
    val_end = train_end + int(round(n * val_ratio))
    return {"train": ids[:train_end], "val": ids[train_end:val_end], "test": ids[val_end:]}


def split_for_id(splits: dict[str, list[str]]) -> dict[str, str]:
    return {trajectory_id: split for split, ids in splits.items() for trajectory_id in ids}


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))
