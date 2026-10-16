#!/usr/bin/env python3
"""Lineage-aware wrapper for resume-safe RECON/SCAND/SACSoN encoding."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO_ROOT), str(REPO_ROOT / "src"), str(REPO_ROOT / "tools/recon")]

from tools.recon import encode_dinov3_adapted_resumable as base  # noqa: E402


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def validate_successor_parent_reuse(
    active_lock: Mapping[str, Any],
    *,
    parent_split_lock: str | Path,
    artifact_split_lock_sha256: str | None,
    unchanged_datasets: Sequence[str] = (),
    preserved_datasets: Sequence[str] = (),
) -> dict[str, Any]:
    parent_path = Path(parent_split_lock)
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    parent_sha = file_sha256(parent_path)
    lineage = active_lock.get("lineage", {})
    declared = {
        lineage.get("adapter_parent_split_lock_sha256"),
        lineage.get("warm_start_parent_split_lock_sha256"),
        lineage.get("parent_split_lock_sha256"),
    }
    if parent_sha not in declared:
        raise ValueError(f"active successor does not declare parent lock {parent_sha}")
    if artifact_split_lock_sha256 != parent_sha:
        raise ValueError(
            f"artifact lock mismatch: artifact={artifact_split_lock_sha256} parent={parent_sha}"
        )
    reports: dict[str, Any] = {}
    for dataset in unchanged_datasets:
        if active_lock[dataset]["splits"] != parent[dataset]["splits"]:
            raise ValueError(f"{dataset} membership changed across successor reuse")
        reports[dataset] = {
            split: len(active_lock[dataset]["splits"][split])
            for split in ("train", "val", "test")
        }
    for dataset in preserved_datasets:
        active_assignment = {
            str(value): split
            for split in ("train", "val", "test")
            for value in active_lock[dataset]["splits"][split]
        }
        moved = [
            str(value)
            for split in ("train", "val", "test")
            for value in parent[dataset]["splits"][split]
            if active_assignment.get(str(value)) != split
        ]
        if moved:
            raise ValueError(f"{dataset} parent identities moved: {moved[:10]}")
        reports[dataset] = {
            split: len(parent[dataset]["splits"][split])
            for split in ("train", "val", "test")
        }
    return {
        "status": "PASS",
        "mode": "verified_multigeneration_successor_reuse",
        "parent_split_lock": str(parent_path),
        "parent_split_lock_sha256": parent_sha,
        "datasets": reports,
    }


def main() -> None:
    original_parse_args = base.parse_args

    def parse_args():
        original_argv = list(sys.argv)
        try:
            dataset_index = original_argv.index("--dataset") + 1
        except ValueError as error:
            raise SystemExit("--dataset is required") from error
        dataset = original_argv[dataset_index]
        if dataset not in {"recon", "scand", "sacson"}:
            raise SystemExit(f"unsupported dataset: {dataset}")
        if dataset == "sacson":
            sys.argv[dataset_index] = "scand"
        try:
            args = original_parse_args()
        finally:
            sys.argv[:] = original_argv
        args.dataset = dataset
        return args

    base.parse_args = parse_args
    base.validate_parent_split_reuse = validate_successor_parent_reuse
    base.main()


if __name__ == "__main__":
    main()
