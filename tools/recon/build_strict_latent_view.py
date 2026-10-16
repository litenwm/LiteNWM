#!/usr/bin/env python3
"""Create a zero-copy latent-root view containing only explicitly allowed splits."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from recon_latents.split_lock import file_sha256, load_split_lock  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--strict-splits", type=Path, required=True)
    parser.add_argument("--split-lock", type=Path, required=True)
    parser.add_argument("--dataset", choices=("recon", "scand"), default="recon")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--include-splits", default="train,val")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    include = {
        item.strip() for item in args.include_splits.split(",") if item.strip()
    }
    if not include or not include <= {"train", "val", "test"}:
        raise ValueError("invalid --include-splits")
    lock, lock_sha = load_split_lock(args.split_lock)
    strict = json.loads(args.strict_splits.read_text(encoding="utf-8"))
    locked_splits = lock[args.dataset]["splits"]
    if any(
        set(map(str, strict[split])) != set(map(str, locked_splits[split]))
        for split in ("train", "val", "test")
    ):
        raise ValueError(
            f"{args.dataset} strict split file does not match the active split lock"
        )
    selected_splits = {
        split: sorted(map(str, strict[split])) if split in include else []
        for split in ("train", "val", "test")
    }
    selected_ids = {
        trajectory_id for values in selected_splits.values() for trajectory_id in values
    }
    source_metadata = json.loads(
        (args.source_root / "metadata.json").read_text(encoding="utf-8")
    )
    trajectories = [
        item
        for item in source_metadata["trajectories"]
        if str(item["trajectory_id"]) in selected_ids
    ]
    found = {str(item["trajectory_id"]) for item in trajectories}
    if found != selected_ids:
        raise ValueError(
            f"source latent root is missing strict IDs: {sorted(selected_ids-found)[:10]}"
        )
    args.output_root.mkdir(parents=True, exist_ok=False)
    source_shards = (args.source_root / "shards").resolve(strict=True)
    os.symlink(source_shards, args.output_root / "shards")
    metadata = {
        **source_metadata,
        "format": f"{source_metadata.get('format', 'latent')}_strict_view_v1",
        "source_latent_root": str(args.source_root),
        "split_lock_dataset": args.dataset,
        "source_metadata_sha256": file_sha256(args.source_root / "metadata.json"),
        "split_lock_sha256": lock_sha,
        "included_splits": sorted(include),
        "num_trajectories": len(trajectories),
        "num_frames": int(sum(int(item["num_frames"]) for item in trajectories)),
        "trajectories": trajectories,
    }
    (args.output_root / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    (args.output_root / "splits.json").write_text(
        json.dumps(selected_splits, indent=2), encoding="utf-8"
    )
    with (args.output_root / "trajectories.jsonl").open("w", encoding="utf-8") as handle:
        for item in trajectories:
            handle.write(json.dumps(item) + "\n")
    print(
        json.dumps(
            {
                "status": "PASS",
                "output_root": str(args.output_root),
                "included_splits": sorted(include),
                "trajectory_count": len(trajectories),
                "frame_count": metadata["num_frames"],
                "split_lock_sha256": lock_sha,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
