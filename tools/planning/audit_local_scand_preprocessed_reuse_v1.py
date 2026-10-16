#!/usr/bin/env python3
"""Audit whether legacy preprocessed SCAND routes can replace raw bag ingest."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import pickle
from typing import Any

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PLAN = (
    ROOT
    / "data/proposal_tail_collection_v1/incoming/quarantine/official_scand_selective_v1/download_plan.json"
)
DEFAULT_OUTPUT = ROOT / "outputs/proposal_tail_collection_v1/local_preprocessed_reuse_audit_v1.json"
DEFAULT_SOURCE_ROOTS = (
    Path("/home/rainbow/nwm/data/scand"),
    Path("/home/rainbow/nwm/data/scand_320(no_use)"),
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class NumpyOnlyUnpickler(pickle.Unpickler):
    """Load legacy numpy trajectory dictionaries without arbitrary globals."""

    ALLOWED = {
        ("numpy.core.multiarray", "_reconstruct"): np.core.multiarray._reconstruct,
        ("numpy._core.multiarray", "_reconstruct"): np.core.multiarray._reconstruct,
        ("numpy", "ndarray"): np.ndarray,
        ("numpy", "dtype"): np.dtype,
    }

    def find_class(self, module: str, name: str) -> Any:
        value = self.ALLOWED.get((module, name))
        if value is None:
            raise pickle.UnpicklingError(f"forbidden pickle global: {module}.{name}")
        return value


def load_trajectory(path: Path) -> dict[str, Any]:
    value = NumpyOnlyUnpickler(io.BytesIO(path.read_bytes())).load()
    require(isinstance(value, dict), f"trajectory pickle is not a dictionary: {path}")
    return value


def inspect_route(path: Path) -> dict[str, Any]:
    images = list(path.glob("*.jpg"))
    try:
        images.sort(key=lambda item: int(item.stem))
        numeric_contiguous = [int(item.stem) for item in images] == list(range(len(images)))
    except ValueError:
        numeric_contiguous = False
    trajectory_path = path / "traj_data.pkl"
    trajectory = load_trajectory(trajectory_path) if trajectory_path.is_file() else {}
    position = trajectory.get("position")
    yaw = trajectory.get("yaw")
    timestamps = trajectory.get("timestamps_ns")
    action_keys = sorted(
        key
        for key in trajectory
        if key.lower()
        in {
            "action",
            "actions",
            "cmd_vel",
            "velocity",
            "velocities",
            "linear_velocity",
            "angular_velocity",
        }
    )
    resolutions: dict[str, int] = {}
    formats: dict[str, int] = {}
    modes: dict[str, int] = {}
    exif_datetime_count = 0
    image_error = None
    try:
        for image_path in images:
            with Image.open(image_path) as image:
                resolutions[f"{image.width}x{image.height}"] = (
                    resolutions.get(f"{image.width}x{image.height}", 0) + 1
                )
                formats[str(image.format)] = formats.get(str(image.format), 0) + 1
                modes[image.mode] = modes.get(image.mode, 0) + 1
                exif = image.getexif()
                if any(tag in exif for tag in (306, 36867, 36868)):
                    exif_datetime_count += 1
    except Exception as error:
        image_error = f"{type(error).__name__}: {error}"
    position_shape = list(position.shape) if isinstance(position, np.ndarray) else None
    yaw_shape = list(yaw.shape) if isinstance(yaw, np.ndarray) else None
    timestamp_shape = list(timestamps.shape) if isinstance(timestamps, np.ndarray) else None
    internal_alignment = (
        bool(images)
        and numeric_contiguous
        and position_shape == [len(images), 2]
        and yaw_shape == [len(images)]
    )
    route_metadata = path / "route_metadata.json"
    parent_manifest_candidates = [
        path.parent / "conversion_manifest.json",
        path.parent / "pipeline.jsonl",
    ]
    source_manifest_present = any(candidate.is_file() for candidate in parent_manifest_candidates)
    checks = {
        "jpeg_sequence_present_and_contiguous": bool(images) and numeric_contiguous and image_error is None,
        "processed_resolution_320x240_rgb": (
            resolutions == {"320x240": len(images)} and modes == {"RGB": len(images)}
        ),
        "pose_arrays_index_aligned": internal_alignment,
        "exact_frame_timestamps_present": timestamp_shape == [len(images)],
        "raw_action_or_velocity_present": bool(action_keys),
        "raw_camera_resolution_encoding_quality_recorded": route_metadata.is_file(),
        "source_bag_hash_and_topic_provenance_recorded": route_metadata.is_file() or source_manifest_present,
        "source_level_trajectory_completeness_provable": route_metadata.is_file() or source_manifest_present,
    }
    reuse_eligible = all(checks.values())
    first_failure = next((key for key, passed in checks.items() if not passed), None)
    return {
        "path": str(path),
        "files": {
            "jpeg_count": len(images),
            "traj_data_present": trajectory_path.is_file(),
            "traj_data_sha256": sha256_file(trajectory_path) if trajectory_path.is_file() else None,
            "route_metadata_present": route_metadata.is_file(),
            "parent_conversion_or_pipeline_manifest_present": source_manifest_present,
        },
        "images": {
            "resolutions": resolutions,
            "formats": formats,
            "modes": modes,
            "exif_datetime_count": exif_datetime_count,
            "numeric_contiguous": numeric_contiguous,
            "error": image_error,
        },
        "trajectory": {
            "keys": sorted(trajectory),
            "position_shape": position_shape,
            "yaw_shape": yaw_shape,
            "timestamps_ns_shape": timestamp_shape,
            "action_or_velocity_keys": action_keys,
            "index_alignment_only": internal_alignment,
        },
        "checks": checks,
        "reuse_eligible": reuse_eligible,
        "first_failure": first_failure,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--source-root", type=Path, action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    roots = tuple(args.source_root) if args.source_root else DEFAULT_SOURCE_ROOTS
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    selected = plan["selected"]
    rows = []
    for item in selected:
        expected = f"random_mdps_{item['official_file_name']}_0"
        matches = [root / expected for root in roots if (root / expected).is_dir()]
        inspections = [inspect_route(path) for path in matches]
        reusable = [inspection for inspection in inspections if inspection["reuse_eligible"]]
        rows.append(
            {
                "official_file_name": item["official_file_name"],
                "bag_id": item["bag_id"],
                "cohort_role": item["cohort_role"],
                "expected_trajectory_id": expected,
                "preprocessed_matches": inspections,
                "local_preprocessed_found": bool(matches),
                "reuse_eligible": bool(reusable),
                "decision": "reuse" if reusable else "download_required",
                "decision_reason": (
                    "at least one local copy satisfies the complete proposal-tail source contract"
                    if reusable
                    else (
                        "no local preprocessed route found"
                        if not matches
                        else "legacy preprocessing lacks mandatory source/timing/action/provenance fields"
                    )
                ),
            }
        )
    result = {
        "schema": "local_scand_preprocessed_reuse_audit_v1",
        "status": "complete_fail_closed",
        "plan": str(args.plan.resolve()),
        "plan_sha256": sha256_file(args.plan),
        "source_roots": [str(root) for root in roots],
        "contract": {
            "processed_images": "contiguous RGB JPEG at 320x240",
            "timing": "exact per-frame timestamps",
            "motion": "index-aligned pose plus raw action/velocity evidence",
            "camera": "original resolution, encoding, and JPEG/source metadata",
            "provenance": "source bag hash/topics and source-level trajectory completeness",
            "partial_or_inferred_fields_fail_closed": True,
        },
        "summary": {
            "planned_bags": len(rows),
            "raw_bags_found_locally": 0,
            "preprocessed_routes_found": sum(row["local_preprocessed_found"] for row in rows),
            "reuse_eligible": sum(row["reuse_eligible"] for row in rows),
            "download_required": sum(row["decision"] == "download_required" for row in rows),
            "completely_missing_preprocessed": sum(not row["local_preprocessed_found"] for row in rows),
        },
        "download_required": [
            row["official_file_name"] for row in rows if row["decision"] == "download_required"
        ],
        "rows": rows,
        "reserve_content_accessed": False,
        "labels_opened": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
