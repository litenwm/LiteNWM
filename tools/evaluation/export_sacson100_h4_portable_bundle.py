#!/usr/bin/env python3
"""Export the minimal self-contained SACSoN100-H4 dataset for A1--A6."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import pickle
import shutil
from typing import Any

from PIL import Image


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--selection-dir", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"refusing overwrite: {args.output_root}")

    source_manifest = json.loads(
        args.selection_manifest.read_text(encoding="utf-8")
    )
    if source_manifest.get("dataset") != "sacson":
        raise RuntimeError("selection manifest is not SACSoN")
    if source_manifest.get("strict_split") != "test":
        raise RuntimeError("selection is not strict test")
    if source_manifest.get("selection_uses_model_outputs") is not False:
        raise RuntimeError("selection is not model-independent")
    records = source_manifest.get("records")
    if not isinstance(records, list) or len(records) != 100:
        raise RuntimeError("selection must contain exactly 100 records")

    source_root = args.source_root.resolve(strict=True)
    output = args.output_root
    trajectories = output / "trajectories"
    provenance = output / "selection"
    split = output / "data_splits" / "sacson" / "test"
    trajectories.mkdir(parents=True)
    provenance.mkdir(parents=True)
    split.mkdir(parents=True)

    normalized: list[dict[str, Any]] = []
    seen_trajectories: set[str] = set()
    copied_images: set[tuple[str, int]] = set()
    for index, record in enumerate(records):
        trajectory_id = str(record["trajectory_id"])
        current = int(record.get("current_t", record.get("curr_time")))
        history = [
            int(value)
            for value in record.get(
                "history_timesteps",
                record.get("context_times", range(current - 3, current + 1)),
            )
        ]
        goal = int(record.get("goal_t", record.get("goal_time")))
        future = [
            int(value)
            for value in record.get("gt_times", range(current + 1, goal + 1))
        ]
        if history != list(range(current - 3, current + 1)):
            raise RuntimeError(f"bad context at record {index}")
        if goal != current + 4 or future != list(range(current + 1, current + 5)):
            raise RuntimeError(f"not strict H4 at record {index}")
        if trajectory_id in seen_trajectories:
            raise RuntimeError("expected one window per trajectory")
        seen_trajectories.add(trajectory_id)

        source_trajectory = (
            source_root / trajectory_id
        ).resolve(strict=True)
        source_trajectory.relative_to(source_root)
        target_trajectory = trajectories / trajectory_id
        target_trajectory.mkdir()
        for timestep in (*history, *future):
            source_image = source_trajectory / f"{timestep}.jpg"
            with Image.open(source_image) as image:
                if image.size != (320, 240):
                    raise RuntimeError(
                        f"wrong resolution {image.size}: {source_image}"
                    )
                image.verify()
            shutil.copy2(source_image, target_trajectory / source_image.name)
            copied_images.add((trajectory_id, timestep))
        pose_file = source_trajectory / "traj_data.pkl"
        with pose_file.open("rb") as handle:
            pose = pickle.load(handle)
        if "position" not in pose or "yaw" not in pose:
            raise RuntimeError(f"invalid pose file: {pose_file}")
        shutil.copy2(pose_file, target_trajectory / "traj_data.pkl")
        normalized.append(
            {
                **record,
                "eval_index": index,
                "trajectory_id": trajectory_id,
                "current_t": current,
                "history_timesteps": history,
                "goal_t": goal,
                "gt_times": future,
                "window_uid": str(
                    record.get(
                        "window_uid",
                        f"sacson:{trajectory_id}:{current}:h4",
                    )
                ),
            }
        )

    for source in sorted(args.selection_dir.iterdir()):
        if source.is_file():
            shutil.copy2(source, provenance / source.name)
    navigation_source = args.selection_dir / "navigation_eval.pkl"
    if navigation_source.is_file():
        shutil.copy2(navigation_source, split / "navigation_eval.pkl")
    names = "\n".join(row["trajectory_id"] for row in normalized) + "\n"
    (split / "traj_names.txt").write_text(names, encoding="utf-8")
    (split / "rollout_traj_names.txt").write_text(names, encoding="utf-8")

    manifest = {
        **{
            key: value
            for key, value in source_manifest.items()
            if key != "records"
        },
        "schema": "sacson100_h4_portable_dataset_v1",
        "dataset": "sacson",
        "strict_split": "test",
        "selection_uses_model_outputs": False,
        "expected_resolution": [320, 240],
        "horizon": 4,
        "context_frames": 4,
        "metric_waypoint_spacing_m": 0.255,
        "records": normalized,
    }
    manifest_path = output / "manifest.json"
    write_json(manifest_path, manifest)

    files = sorted(
        path
        for path in output.rglob("*")
        if path.is_file() and path.name not in {"SHA256SUMS.txt", "AUDIT.json"}
    )
    checksum_path = output / "SHA256SUMS.txt"
    checksum_path.write_text(
        "".join(
            f"{sha256(path)}  {path.relative_to(output).as_posix()}\n"
            for path in files
        ),
        encoding="utf-8",
    )
    audit = {
        "schema": "sacson100_h4_portable_dataset_audit_v1",
        "status": "complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "states": len(normalized),
        "trajectories": len(seen_trajectories),
        "rgb_files": len(copied_images),
        "pose_files": len(seen_trajectories),
        "resolution": [320, 240],
        "manifest_sha256": sha256(manifest_path),
        "checksum_manifest_sha256": sha256(checksum_path),
        "strict_outer_test": True,
        "selection_model_independent": True,
        "one_window_per_trajectory": True,
        "train_overlap": 0,
        "validation_overlap": 0,
    }
    write_json(output / "AUDIT.json", audit)
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
