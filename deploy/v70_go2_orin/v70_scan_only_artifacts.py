"""Write fail-closed active-scan commissioning artifacts.

This module is deliberately free of ROS and Unitree dependencies so the
artifact contract can be tested without initializing any motion interface.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image


SCAN_ONLY_SCHEMA = "v70_active_history_scan_only_v1"
PARTIAL_SCAN_SCHEMA = "v70_active_history_scan_partial_v1"
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def _finite_float(name: str, value: object) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _wrap_angle(value: float) -> float:
    return float((value + math.pi) % (2.0 * math.pi) - math.pi)


def _pose_payload(name: str, pose: Mapping[str, object]) -> dict[str, float]:
    return {
        "receive_time_monotonic": _finite_float(
            f"{name}.receive_time_monotonic",
            pose["receive_time_monotonic"],
        ),
        "x_m": _finite_float(f"{name}.x_m", pose["x_m"]),
        "y_m": _finite_float(f"{name}.y_m", pose["y_m"]),
        "yaw_rad": _finite_float(f"{name}.yaw_rad", pose["yaw_rad"]),
    }


def write_scan_only_artifacts(
    *,
    output_dir: Path,
    pattern: str,
    step_yaw_rad: float,
    scan_targets: Sequence[float],
    capture_target_yaw_offsets: Sequence[float],
    images: Sequence[Image.Image],
    source_sha256: Sequence[str],
    poses: Sequence[Mapping[str, object]],
    origin_pose: Mapping[str, object],
    stop_asserted: bool,
) -> dict[str, Any]:
    """Save four ordered PNGs and their exact scan/odometry contract."""

    if not stop_asserted:
        raise RuntimeError("scan-only artifacts require STOP asserted")
    if not pattern:
        raise ValueError("scan pattern must be non-empty")
    if not (
        len(images)
        == len(source_sha256)
        == len(poses)
        == len(capture_target_yaw_offsets)
        == 4
    ):
        raise ValueError("scan-only artifacts require exactly four frames")
    if not scan_targets:
        raise ValueError("scan-only targets must be non-empty")

    step = _finite_float("step_yaw_rad", step_yaw_rad)
    if step <= 0.0:
        raise ValueError("step_yaw_rad must be positive")
    targets = [
        _finite_float(f"scan_targets[{index}]", value)
        for index, value in enumerate(scan_targets)
    ]
    capture_targets = [
        _finite_float(f"capture_targets[{index}]", value)
        for index, value in enumerate(capture_target_yaw_offsets)
    ]
    origin = _pose_payload("origin_pose", origin_pose)
    pose_payloads = [
        _pose_payload(f"poses[{index}]", pose)
        for index, pose in enumerate(poses)
    ]

    for index, (image, digest) in enumerate(zip(images, source_sha256)):
        if not isinstance(image, Image.Image):
            raise TypeError(f"images[{index}] is not a PIL image")
        if not SHA256_PATTERN.fullmatch(str(digest)):
            raise ValueError(f"source_sha256[{index}] is invalid")

    actual_offsets = [
        _wrap_angle(pose["yaw_rad"] - origin["yaw_rad"])
        for pose in pose_payloads
    ]
    final_pose = pose_payloads[-1]
    final_yaw_error = actual_offsets[-1]
    translation_drift = math.hypot(
        final_pose["x_m"] - origin["x_m"],
        final_pose["y_m"] - origin["y_m"],
    )

    destination = output_dir.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=False)
    frames: list[dict[str, Any]] = []
    for index, (image, digest, pose, target, actual) in enumerate(
        zip(
            images,
            source_sha256,
            pose_payloads,
            capture_targets,
            actual_offsets,
        )
    ):
        filename = f"history_{index}.png"
        image.save(destination / filename, format="PNG", compress_level=3)
        frames.append(
            {
                "history_index": index,
                "filename": filename,
                "source_jpeg_sha256": str(digest),
                "width": int(image.width),
                "height": int(image.height),
                "target_yaw_offset_rad": target,
                "actual_yaw_offset_rad": actual,
                "pose": pose,
            }
        )

    manifest: dict[str, Any] = {
        "schema": SCAN_ONLY_SCHEMA,
        "stop_asserted_before_write": True,
        "pattern": str(pattern),
        "step_yaw_rad": step,
        "scan_targets_yaw_offset_rad": targets,
        "capture_target_yaw_offsets_rad": capture_targets,
        "origin_pose": origin,
        "frames": frames,
        "final_yaw_error_rad": final_yaw_error,
        "translation_drift_m": translation_drift,
    }
    temporary = destination / "manifest.json.tmp"
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(destination / "manifest.json")
    return manifest


def write_partial_scan_artifacts(
    *,
    output_dir: Path,
    pattern: str,
    step_yaw_rad: float,
    scan_targets: Sequence[float],
    capture_target_yaw_offsets: Sequence[float],
    images: Sequence[Image.Image],
    source_sha256: Sequence[str],
    poses: Sequence[Mapping[str, object]],
    origin_pose: Mapping[str, object],
    failure_reason: str,
    failed_step_index: int,
    stop_asserted: bool,
) -> dict[str, Any]:
    """Save incomplete scan evidence without making it model-eligible."""

    if not stop_asserted:
        raise RuntimeError("partial scan artifacts require STOP asserted")
    if not pattern:
        raise ValueError("scan pattern must be non-empty")
    if not failure_reason:
        raise ValueError("failure_reason must be non-empty")
    if not isinstance(failed_step_index, int) or failed_step_index < 0:
        raise ValueError("failed_step_index must be a nonnegative integer")
    frame_count = len(images)
    if not 1 <= frame_count <= 4:
        raise ValueError("partial scan artifacts require one to four frames")
    if not (
        frame_count
        == len(source_sha256)
        == len(poses)
        == len(capture_target_yaw_offsets)
    ):
        raise ValueError("partial scan artifact frame metadata mismatch")
    if not scan_targets:
        raise ValueError("scan targets must be non-empty")

    step = _finite_float("step_yaw_rad", step_yaw_rad)
    if step <= 0.0:
        raise ValueError("step_yaw_rad must be positive")
    targets = [
        _finite_float(f"scan_targets[{index}]", value)
        for index, value in enumerate(scan_targets)
    ]
    capture_targets = [
        _finite_float(f"capture_targets[{index}]", value)
        for index, value in enumerate(capture_target_yaw_offsets)
    ]
    origin = _pose_payload("origin_pose", origin_pose)
    pose_payloads = [
        _pose_payload(f"poses[{index}]", pose)
        for index, pose in enumerate(poses)
    ]
    for index, (image, digest) in enumerate(zip(images, source_sha256)):
        if not isinstance(image, Image.Image):
            raise TypeError(f"images[{index}] is not a PIL image")
        if not SHA256_PATTERN.fullmatch(str(digest)):
            raise ValueError(f"source_sha256[{index}] is invalid")

    actual_offsets = [
        _wrap_angle(pose["yaw_rad"] - origin["yaw_rad"])
        for pose in pose_payloads
    ]
    last_pose = pose_payloads[-1]
    translation_drift = math.hypot(
        last_pose["x_m"] - origin["x_m"],
        last_pose["y_m"] - origin["y_m"],
    )
    destination = output_dir.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=False)
    frames: list[dict[str, Any]] = []
    for index, (image, digest, pose, target, actual) in enumerate(
        zip(
            images,
            source_sha256,
            pose_payloads,
            capture_targets,
            actual_offsets,
        )
    ):
        filename = f"partial_history_{index}.png"
        image.save(destination / filename, format="PNG", compress_level=3)
        frames.append(
            {
                "history_index": index,
                "filename": filename,
                "source_jpeg_sha256": str(digest),
                "width": int(image.width),
                "height": int(image.height),
                "target_yaw_offset_rad": target,
                "actual_yaw_offset_rad": actual,
                "pose": pose,
            }
        )

    manifest: dict[str, Any] = {
        "schema": PARTIAL_SCAN_SCHEMA,
        "stop_asserted_before_write": True,
        "eligible_for_world_model": False,
        "failure_reason": str(failure_reason),
        "failed_step_index": failed_step_index,
        "captured_frame_count": frame_count,
        "pattern": str(pattern),
        "step_yaw_rad": step,
        "scan_targets_yaw_offset_rad": targets,
        "capture_target_yaw_offsets_rad": capture_targets,
        "origin_pose": origin,
        "frames": frames,
        "last_yaw_offset_rad": actual_offsets[-1],
        "translation_drift_m": translation_drift,
    }
    temporary = destination / "manifest.json.tmp"
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(destination / "manifest.json")
    return manifest
