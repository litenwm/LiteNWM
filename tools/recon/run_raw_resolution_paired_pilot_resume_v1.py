#!/usr/bin/env python3
"""Validated-resume wrapper for the paired raw-resolution pilot."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(Path(__file__).resolve().parent)]

import run_raw_resolution_paired_pilot_v1 as base  # noqa: E402


def reuse_paired_latents(
    output: Path,
    specs,
    **_: object,
):
    paths = {"legacy320": {}, "raw448": {}}
    rows = []
    for spec in specs:
        pair = {
            mode: output / "latents" / mode / f"{spec.trajectory_id}.npz"
            for mode in ("legacy320", "raw448")
        }
        values = {}
        for mode, shard_path in pair.items():
            base.require(shard_path.is_file(), f"missing resume shard: {shard_path}")
            with np.load(shard_path, allow_pickle=False) as shard:
                base.require(
                    str(shard["trajectory_id"].item()) == spec.trajectory_id,
                    f"trajectory mismatch: {shard_path}",
                )
                base.require(
                    str(shard["split"].item()) == spec.split,
                    f"split mismatch: {shard_path}",
                )
                values[mode] = {
                    "semantic_shape": shard["semantic_latents"].shape,
                    "geometry_shape": shard["geometry_latents"].shape,
                    "actions": shard["actions_continuous"].copy(),
                    "timestamps": shard["timestamps_ns"].copy(),
                }
            paths[mode][spec.trajectory_id] = shard_path
        base.require(
            values["legacy320"]["semantic_shape"]
            == values["raw448"]["semantic_shape"],
            f"semantic pair mismatch: {spec.trajectory_id}",
        )
        base.require(
            values["legacy320"]["geometry_shape"]
            == values["raw448"]["geometry_shape"],
            f"geometry pair mismatch: {spec.trajectory_id}",
        )
        base.require(
            np.array_equal(
                values["legacy320"]["actions"],
                values["raw448"]["actions"],
            ),
            f"action pair mismatch: {spec.trajectory_id}",
        )
        base.require(
            np.array_equal(
                values["legacy320"]["timestamps"],
                values["raw448"]["timestamps"],
            ),
            f"timestamp pair mismatch: {spec.trajectory_id}",
        )
        frames = int(values["legacy320"]["semantic_shape"][0])
        rows.append(
            {
                "trajectory_id": spec.trajectory_id,
                "split": spec.split,
                "bag": str(spec.path),
                "bag_sha256": base.sha256_file(spec.path),
                "frames": frames,
                "windows": frames - 7,
                "paired_latents_reused_after_validation": True,
            }
        )
        print(
            base.canonical_json(
                {
                    "stage": "reused_paired_bag",
                    "trajectory": spec.trajectory_id,
                    "split": spec.split,
                    "frames": frames,
                    "windows": frames - 7,
                }
            ),
            flush=True,
        )
    return paths, rows


def main() -> None:
    base.prepare_latents = reuse_paired_latents
    base.__file__ = __file__
    base.main()


if __name__ == "__main__":
    main()
