#!/usr/bin/env python3
"""Bind shadow artifacts to one image snapshot and one goal image."""

from __future__ import annotations

import argparse
import hashlib
import os
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HISTORY = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact",
        action="append",
        required=True,
        type=Path,
        help="NPZ artifact to stamp; repeat for multiple artifacts",
    )
    parser.add_argument("--frame-dir", required=True, type=Path)
    parser.add_argument("--goal-image", required=True, type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--max-frame-span-s", type=float, default=2.0)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stamp_artifact(
    path: Path,
    *,
    run_id: str,
    goal_sha256: str,
    capture_ns: int,
    frame_names: np.ndarray,
    frame_hashes: np.ndarray,
) -> None:
    if path.suffix != ".npz" or not path.is_file():
        raise FileNotFoundError(f"NPZ artifact not found: {path}")

    with np.load(path, allow_pickle=False) as payload:
        values = {name: payload[name].copy() for name in payload.files}

    values.update(
        snapshot_run_id=np.asarray(run_id),
        snapshot_goal_sha256=np.asarray(goal_sha256),
        snapshot_capture_unix_ns=np.asarray(capture_ns, dtype=np.int64),
        snapshot_frame_names=frame_names,
        snapshot_frame_sha256=frame_hashes,
    )

    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary_name = handle.name
            np.savez_compressed(handle, **values)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if temporary_name is not None and os.path.exists(temporary_name):
            os.unlink(temporary_name)


def main() -> None:
    args = parse_args()

    frame_dir = args.frame_dir.expanduser().resolve()
    goal_path = args.goal_image.expanduser().resolve()
    artifacts = [
        path.expanduser().resolve()
        for path in args.artifact
    ]

    if not goal_path.is_file():
        raise FileNotFoundError(goal_path)

    frame_paths = [
        frame_dir / f"history_{index}.jpg"
        for index in range(HISTORY)
    ]
    missing = [str(path) for path in frame_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"snapshot frames missing: {missing}")

    mtimes_ns = np.asarray(
        [path.stat().st_mtime_ns for path in frame_paths],
        dtype=np.int64,
    )
    frame_span_s = float(mtimes_ns.max() - mtimes_ns.min()) / 1.0e9
    if frame_span_s > args.max_frame_span_s:
        raise RuntimeError(
            f"frame capture span {frame_span_s:.6f}s exceeds "
            f"{args.max_frame_span_s:.6f}s"
        )

    capture_ns = int(mtimes_ns.max())
    goal_sha256 = file_sha256(goal_path)
    run_id = args.run_id or (
        "v70-shadow-"
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "-"
        + uuid.uuid4().hex[:8]
    )
    frame_names = np.asarray([path.name for path in frame_paths])
    frame_hashes = np.asarray(
        [file_sha256(path) for path in frame_paths]
    )

    for artifact in artifacts:
        stamp_artifact(
            artifact,
            run_id=run_id,
            goal_sha256=goal_sha256,
            capture_ns=capture_ns,
            frame_names=frame_names,
            frame_hashes=frame_hashes,
        )
        print("stamped:", artifact)

    print("snapshot_run_id:", run_id)
    print("snapshot_goal_sha256:", goal_sha256)
    print("snapshot_capture_unix_ns:", capture_ns)
    print("frame_capture_span_s:", round(frame_span_s, 6))


if __name__ == "__main__":
    main()
