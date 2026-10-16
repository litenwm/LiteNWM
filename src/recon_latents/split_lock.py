"""Immutable split-lock helpers for leakage-resistant LTWM experiments."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence


SPLITS = ("train", "val", "test")


def canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def recon_route_id(trajectory_id: str) -> str:
    """Collapse terminal RECON repeat suffixes into one route group."""

    return re.sub(r"_r\d+$", "", str(trajectory_id))


def normalize_scand_route(left: str, right: str) -> str:
    """Return an undirected route-pair identifier."""

    return "__".join(sorted((left.strip().lower(), right.strip().lower())))


def assert_disjoint_splits(
    splits: Mapping[str, Sequence[str]],
    *,
    group_for_id: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    sets = {split: set(map(str, splits.get(split, ()))) for split in SPLITS}
    exact_overlap: dict[str, list[str]] = {}
    group_overlap: dict[str, list[str]] = {}
    for index, left in enumerate(SPLITS):
        for right in SPLITS[index + 1 :]:
            key = f"{left}__{right}"
            exact_overlap[key] = sorted(sets[left] & sets[right])
            if group_for_id is not None:
                left_groups = {group_for_id[item] for item in sets[left]}
                right_groups = {group_for_id[item] for item in sets[right]}
                group_overlap[key] = sorted(left_groups & right_groups)
    if any(exact_overlap.values()):
        raise ValueError(f"exact split leakage: {exact_overlap}")
    if any(group_overlap.values()):
        raise ValueError(f"group split leakage: {group_overlap}")
    return {
        "split_counts": {split: len(sets[split]) for split in SPLITS},
        "exact_overlap": {key: 0 for key in exact_overlap},
        "group_overlap": {key: 0 for key in group_overlap},
    }


def load_split_lock(path: str | Path) -> tuple[dict[str, Any], str]:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    digest = file_sha256(path)
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if sidecar.exists():
        expected = sidecar.read_text(encoding="utf-8").strip().split()[0]
        if digest != expected:
            raise ValueError(
                f"split-lock SHA256 mismatch: expected={expected} actual={digest}"
            )
    return payload, digest


def validate_parent_split_reuse(
    active_lock: Mapping[str, Any],
    *,
    parent_split_lock: str | Path,
    artifact_split_lock_sha256: str | None,
    unchanged_datasets: Sequence[str] = (),
    preserved_datasets: Sequence[str] = (),
) -> dict[str, Any]:
    """Verify that an artifact from a declared parent lock is safe to reuse.

    ``unchanged_datasets`` must have byte-for-byte-equivalent split memberships.
    ``preserved_datasets`` may grow, but every parent trajectory must retain its
    original split. This deliberately fails closed when lineage is absent.
    """

    parent_lock, parent_sha = load_split_lock(parent_split_lock)
    declared_parent = (
        active_lock.get("lineage", {}).get("parent_split_lock_sha256")
    )
    if declared_parent != parent_sha:
        raise ValueError(
            "active split lock does not declare the supplied parent lock: "
            f"declared={declared_parent} supplied={parent_sha}"
        )
    if artifact_split_lock_sha256 != parent_sha:
        raise ValueError(
            "artifact is not bound to the declared parent split lock: "
            f"artifact={artifact_split_lock_sha256} parent={parent_sha}"
        )

    unchanged_report: dict[str, dict[str, int]] = {}
    for dataset in unchanged_datasets:
        active_splits = active_lock[dataset]["splits"]
        parent_splits = parent_lock[dataset]["splits"]
        for split in SPLITS:
            active_ids = set(map(str, active_splits.get(split, ())))
            parent_ids = set(map(str, parent_splits.get(split, ())))
            if active_ids != parent_ids:
                raise ValueError(
                    f"{dataset} {split} changed across parent reuse: "
                    f"added={sorted(active_ids-parent_ids)[:10]} "
                    f"removed={sorted(parent_ids-active_ids)[:10]}"
                )
        unchanged_report[dataset] = {
            split: len(set(map(str, active_splits.get(split, ()))))
            for split in SPLITS
        }

    preserved_report: dict[str, dict[str, int]] = {}
    for dataset in preserved_datasets:
        active_splits = active_lock[dataset]["splits"]
        parent_splits = parent_lock[dataset]["splits"]
        active_assignment = {
            str(item): split
            for split in SPLITS
            for item in active_splits.get(split, ())
        }
        for split in SPLITS:
            moved = sorted(
                str(item)
                for item in parent_splits.get(split, ())
                if active_assignment.get(str(item)) != split
            )
            if moved:
                raise ValueError(
                    f"{dataset} parent trajectories moved or disappeared from "
                    f"{split}: {moved[:10]}"
                )
        preserved_report[dataset] = {
            split: len(set(map(str, parent_splits.get(split, ()))))
            for split in SPLITS
        }

    return {
        "status": "PASS",
        "mode": "verified_parent_split_reuse",
        "parent_split_lock": str(Path(parent_split_lock)),
        "parent_split_lock_sha256": parent_sha,
        "unchanged_datasets": unchanged_report,
        "preserved_datasets": preserved_report,
    }


def validate_latent_root(
    latent_root: str | Path,
    expected_splits: Mapping[str, Sequence[str]],
    *,
    expected_grid: tuple[int, int] = (16, 16),
    expected_dim: int = 384,
) -> dict[str, Any]:
    root = Path(latent_root)
    metadata_path = root / "metadata.json"
    splits_path = root / "splits.json"
    if not metadata_path.is_file() or not splits_path.is_file():
        raise FileNotFoundError(f"incomplete latent root: {root}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    actual_splits = json.loads(splits_path.read_text(encoding="utf-8"))
    for split in SPLITS:
        expected = set(map(str, expected_splits.get(split, ())))
        actual = set(map(str, actual_splits.get(split, ())))
        if expected != actual:
            raise ValueError(
                f"{root} {split} split mismatch: "
                f"missing={sorted(expected - actual)[:10]} "
                f"unexpected={sorted(actual - expected)[:10]}"
            )
    assert_disjoint_splits(actual_splits)
    grid = tuple(int(value) for value in metadata.get("patch_grid_shape", ()))
    dim = int(metadata.get("latent_dim", -1))
    if grid != expected_grid or dim != expected_dim:
        raise ValueError(
            f"latent shape mismatch for {root}: grid={grid}, dim={dim}, "
            f"expected={expected_grid}x{expected_dim}"
        )
    by_id = {
        str(item["trajectory_id"]): item for item in metadata.get("trajectories", ())
    }
    expected_ids = {
        item for split in SPLITS for item in map(str, expected_splits.get(split, ()))
    }
    if expected_ids != set(by_id):
        raise ValueError(
            f"metadata/split trajectory mismatch for {root}: "
            f"metadata_only={sorted(set(by_id) - expected_ids)[:10]} "
            f"split_only={sorted(expected_ids - set(by_id))[:10]}"
        )
    missing_shards = [
        item["shard_file"]
        for item in by_id.values()
        if not (root / "shards" / str(item["shard_file"])).is_file()
    ]
    if missing_shards:
        raise FileNotFoundError(f"missing latent shards: {missing_shards[:10]}")
    return {
        "latent_root": str(root),
        "metadata_sha256": file_sha256(metadata_path),
        "splits_sha256": file_sha256(splits_path),
        "trajectory_count": len(by_id),
        "frame_count": int(sum(int(item["num_frames"]) for item in by_id.values())),
        "patch_grid_shape": list(grid),
        "latent_dim": dim,
        "status": "PASS",
    }
