#!/usr/bin/env python3
"""Build or verify a fail-closed manifest for one incoming SCAND route.

The tool reads only ``route_metadata.json`` and the image files explicitly
listed there.  It never reads labels, truth, outcomes, or forbidden cohort
content.  Identity and timestamp fields are supplied by the data custodian;
image size, SHA-256, encoding, width, and height are computed from the files.
JPEG quality is metadata, not inferred from quantization tables: if it is not
provided, the manifest records ``null`` plus an explicit reason.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from planning.proposal_tail_sidecar_v1 import (  # noqa: E402
    canonical_json,
    identity_hash,
    reject_forbidden_path,
    sha256_file,
)


SOURCE_SCHEMA = "incoming_scand_route_source_v1"
AUDIT_SCHEMA = "incoming_scand_route_manifest_audit_v1"
ROLES = ("development", "one_time_validation")
IDENTITY_FIELDS = (
    "route_group",
    "route_id",
    "collection_day",
    "bag_id",
    "trajectory_id",
)
LABEL_TOKENS = {"label", "labels", "truth", "outcome", "outcomes", "groundtruth"}
JPEG_SOF_MARKERS = {
    0xC0,
    0xC1,
    0xC2,
    0xC3,
    0xC5,
    0xC6,
    0xC7,
    0xC9,
    0xCA,
    0xCB,
    0xCD,
    0xCE,
    0xCF,
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def tokens(value: str) -> set[str]:
    return {item for item in re.split(r"[^a-z0-9]+", value.lower()) if item}


def require_public_path(path: Path) -> None:
    reject_forbidden_path(path)
    require(not (tokens(path.name) & LABEL_TOKENS), f"refusing label/truth path: {path}")


def resolve_inside(root: Path, relative: str) -> Path:
    require(relative and not Path(relative).is_absolute(), "image path must be relative")
    candidate = (root / relative).resolve()
    resolved_root = root.resolve()
    require(candidate == resolved_root or resolved_root in candidate.parents, "image escapes route root")
    require_public_path(candidate)
    require(candidate.is_file(), f"missing image file: {relative}")
    return candidate


def _jpeg_dimensions(data: bytes) -> tuple[int, int]:
    require(data.startswith(b"\xff\xd8"), "JPEG magic mismatch")
    offset = 2
    while offset + 4 <= len(data):
        if data[offset] != 0xFF:
            offset += 1
            continue
        while offset < len(data) and data[offset] == 0xFF:
            offset += 1
        require(offset < len(data), "truncated JPEG marker")
        marker = data[offset]
        offset += 1
        if marker in (0xD8, 0xD9) or 0xD0 <= marker <= 0xD7:
            continue
        require(offset + 2 <= len(data), "truncated JPEG segment")
        length = int.from_bytes(data[offset : offset + 2], "big")
        require(length >= 2 and offset + length <= len(data), "bad JPEG segment length")
        if marker in JPEG_SOF_MARKERS:
            require(length >= 7, "truncated JPEG SOF")
            height = int.from_bytes(data[offset + 3 : offset + 5], "big")
            width = int.from_bytes(data[offset + 5 : offset + 7], "big")
            require(width > 0 and height > 0, "invalid JPEG dimensions")
            return width, height
        offset += length
    raise RuntimeError("JPEG has no supported SOF dimensions")


def inspect_image(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        require(len(data) >= 24 and data[12:16] == b"IHDR", "truncated PNG header")
        width = int.from_bytes(data[16:20], "big")
        height = int.from_bytes(data[20:24], "big")
        require(width > 0 and height > 0, "invalid PNG dimensions")
        encoding = "png"
    elif data.startswith(b"\xff\xd8"):
        width, height = _jpeg_dimensions(data)
        encoding = "jpeg"
    else:
        raise RuntimeError(f"unsupported image encoding: {path}")
    return {
        "width": width,
        "height": height,
        "encoding": encoding,
        "file_size": len(data),
        "file_sha256": sha256_file(path),
    }


def _identity(value: Any, field: str) -> str:
    require(isinstance(value, str) and value.strip(), f"missing identity: {field}")
    result = value.strip()
    require("<" not in result and ">" not in result, f"placeholder identity is forbidden: {field}")
    return result


def _number(value: Any, field: str) -> float:
    require(isinstance(value, (int, float)) and not isinstance(value, bool), f"missing numeric {field}")
    result = float(value)
    require(math.isfinite(result), f"non-finite {field}")
    return result


def load_source_metadata(route_dir: Path) -> tuple[Path, dict[str, Any]]:
    require_public_path(route_dir)
    require(route_dir.is_dir(), f"route directory does not exist: {route_dir}")
    for path in route_dir.rglob("*"):
        if path.is_file():
            require_public_path(path)
    metadata_path = route_dir / "route_metadata.json"
    require(metadata_path.is_file(), "missing route_metadata.json; example files are not inputs")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    require(metadata.get("schema") == SOURCE_SCHEMA, "incoming source schema mismatch")
    require(metadata.get("example_only") is not True, "example metadata cannot be built")
    require(metadata.get("public_nonreserve") is True, "source is not explicitly public/nonreserve")
    require(metadata.get("labels_opened") is False, "labels must remain unopened")
    require(metadata.get("cohort_role") in ROLES, "invalid cohort role")
    return metadata_path, metadata


def build_rows(route_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    metadata_path, metadata = load_source_metadata(route_dir)
    identity = {field: _identity(metadata.get(field), field) for field in IDENTITY_FIELDS}
    frames = metadata.get("frames")
    require(isinstance(frames, list) and frames, "frames must be a non-empty list")
    frame_map: dict[int, dict[str, Any]] = {}
    for item in frames:
        require(isinstance(item, Mapping), "frame entry must be an object")
        index = item.get("frame_index")
        require(isinstance(index, int) and not isinstance(index, bool), "frame_index must be int")
        require(index not in frame_map, f"duplicate frame_index: {index}")
        timestamp = _number(item.get("timestamp"), f"timestamp[{index}]")
        relative = item.get("image")
        require(isinstance(relative, str), f"missing image path for frame {index}")
        image_path = resolve_inside(route_dir, relative)
        image = inspect_image(image_path)
        supplied_encoding = item.get("encoding")
        if supplied_encoding is not None:
            normalized = str(supplied_encoding).lower().replace("jpg", "jpeg")
            require(normalized == image["encoding"], f"encoding mismatch for frame {index}")
        quality = item.get("jpeg_quality", metadata.get("jpeg_quality"))
        if quality is not None:
            require(image["encoding"] == "jpeg", "jpeg_quality supplied for non-JPEG frame")
            require(isinstance(quality, int) and 1 <= quality <= 100, "jpeg_quality must be 1..100")
            quality_reason = None
        else:
            quality_reason = (
                "not_provided_by_camera_or_encoder"
                if image["encoding"] == "jpeg"
                else "not_applicable_non_jpeg"
            )
        frame_map[index] = {
            "frame_index": index,
            "timestamp": timestamp,
            "path": str(image_path),
            **image,
            "jpeg_quality": quality,
            "jpeg_quality_unavailable_reason": quality_reason,
        }
    ordered_frames = sorted(frame_map.values(), key=lambda item: item["frame_index"])
    require(
        all(a["timestamp"] <= b["timestamp"] for a, b in zip(ordered_frames, ordered_frames[1:])),
        "frame timestamps are nonmonotonic",
    )

    windows = metadata.get("windows")
    require(isinstance(windows, list) and windows, "windows must be a non-empty list")
    seen_windows: set[str] = set()
    rows: list[dict[str, Any]] = []
    for item in windows:
        require(isinstance(item, Mapping), "window entry must be an object")
        window_uid = _identity(item.get("window_uid"), "window_uid")
        require(window_uid not in seen_windows, f"duplicate window_uid: {window_uid}")
        seen_windows.add(window_uid)
        history = item.get("history_timesteps")
        require(isinstance(history, list) and history, "history_timesteps must be non-empty")
        require(all(isinstance(value, int) and not isinstance(value, bool) for value in history), "history indices must be int")
        current = item.get("current_t")
        goal = item.get("goal_t")
        require(isinstance(current, int) and isinstance(goal, int), "current_t/goal_t must be int")
        sequence = [*history, current, goal]
        require(all(value in frame_map for value in sequence), "window references missing frame")
        timestamps = [frame_map[value]["timestamp"] for value in sequence]
        require(all(a <= b for a, b in zip(timestamps, timestamps[1:])), "window timestamps are nonmonotonic")
        selected = [frame_map[value] for value in sequence]
        camera = {
            "frame_indices": sequence,
            "width": [value["width"] for value in selected],
            "height": [value["height"] for value in selected],
            "encoding": [value["encoding"] for value in selected],
            "jpeg_quality": [value["jpeg_quality"] for value in selected],
            "jpeg_quality_unavailable_reason": [
                value["jpeg_quality_unavailable_reason"] for value in selected
            ],
            "file_paths": [value["path"] for value in selected],
            "file_sizes": [value["file_size"] for value in selected],
            "file_sha256": [value["file_sha256"] for value in selected],
        }
        rows.append(
            {
                "schema": "incoming_scand_window_manifest_row_v1",
                "dataset": "scand",
                "cohort_role": metadata["cohort_role"],
                **identity,
                "window_uid": window_uid,
                "history_timesteps": history,
                "current_t": current,
                "goal_t": goal,
                "history_timestamps": [frame_map[value]["timestamp"] for value in history],
                "current_timestamp": frame_map[current]["timestamp"],
                "goal_timestamp": frame_map[goal]["timestamp"],
                "context_image_paths": [frame_map[value]["path"] for value in history],
                "current_image_path": frame_map[current]["path"],
                "goal_image_path": frame_map[goal]["path"],
                "camera_metadata": camera,
                "timestamp_unit": metadata.get("timestamp_unit"),
                "source_route_dir": str(route_dir.resolve()),
                "source_metadata_sha256": sha256_file(metadata_path),
                "labels_opened": False,
            }
        )
    require(metadata.get("timestamp_unit") in ("seconds", "nanoseconds"), "timestamp_unit must be seconds or nanoseconds")
    audit = {
        "schema": AUDIT_SCHEMA,
        "status": "complete_prelabel_manifest",
        "cohort_role": metadata["cohort_role"],
        "route_dir": str(route_dir.resolve()),
        "source_metadata": str(metadata_path.resolve()),
        "source_metadata_sha256": sha256_file(metadata_path),
        "frames": len(frame_map),
        "windows": len(rows),
        "labels_opened": False,
        "reserve_content_accessed": False,
        "identity_hashes": {
            "route": identity_hash("route", identity["route_group"]),
            "day": identity_hash("day", identity["collection_day"]),
            "bag": identity_hash("bag", identity["bag_id"]),
            "trajectory": identity_hash("trajectory", identity["trajectory_id"]),
            "windows": [identity_hash("window", row["window_uid"]) for row in rows],
        },
    }
    return rows, audit


def validate_manifest_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    require(bool(rows), "manifest has no rows")
    records: set[str] = set()
    windows: set[str] = set()
    for row in rows:
        require(row.get("schema") == "incoming_scand_window_manifest_row_v1", "manifest row schema mismatch")
        require(row.get("dataset") == "scand", "manifest row is not SCAND")
        require(row.get("cohort_role") in ROLES, "manifest row role invalid")
        require(row.get("labels_opened") is False, "manifest reports opened labels")
        for field in IDENTITY_FIELDS:
            _identity(row.get(field), field)
        window_uid = _identity(row.get("window_uid"), "window_uid")
        require(window_uid not in windows, "duplicate manifest window")
        windows.add(window_uid)
        record = f"{row['trajectory_id']}:{int(row['current_t'])}"
        require(record not in records, "duplicate manifest record")
        records.add(record)
        history = row.get("history_timesteps")
        history_ts = row.get("history_timestamps")
        require(isinstance(history, list) and isinstance(history_ts, list), "missing history/timestamps")
        require(len(history) == len(history_ts) and len(history) > 0, "history timestamp length mismatch")
        timestamps = [*history_ts, row.get("current_timestamp"), row.get("goal_timestamp")]
        numeric = [_number(value, "manifest timestamp") for value in timestamps]
        require(all(a <= b for a, b in zip(numeric, numeric[1:])), "manifest timestamps nonmonotonic")
        camera = row.get("camera_metadata")
        require(isinstance(camera, Mapping), "missing camera_metadata")
        count = len(history) + 2
        for field in ("width", "height", "encoding", "jpeg_quality", "jpeg_quality_unavailable_reason", "file_paths", "file_sizes", "file_sha256"):
            require(isinstance(camera.get(field), list) and len(camera[field]) == count, f"camera {field} length mismatch")
        for index, raw_path in enumerate(camera["file_paths"]):
            path = Path(str(raw_path))
            require_public_path(path)
            require(path.is_file(), f"manifest image missing: {path}")
            require(path.stat().st_size == int(camera["file_sizes"][index]), "manifest image size changed")
            require(sha256_file(path) == str(camera["file_sha256"][index]), "manifest image SHA changed")
            inspected = inspect_image(path)
            require(inspected["width"] == int(camera["width"][index]), "manifest width mismatch")
            require(inspected["height"] == int(camera["height"][index]), "manifest height mismatch")
            require(inspected["encoding"] == str(camera["encoding"][index]), "manifest encoding mismatch")
            quality = camera["jpeg_quality"][index]
            reason = camera["jpeg_quality_unavailable_reason"][index]
            require(quality is not None or isinstance(reason, str), "missing jpeg null reason")


def read_manifest(path: Path) -> list[dict[str, Any]]:
    require_public_path(path)
    require(path.is_file(), f"missing manifest: {path}")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    validate_manifest_rows(rows)
    return rows


def default_audit_path(manifest: Path) -> Path:
    name = manifest.name[:-6] if manifest.name.endswith(".jsonl") else manifest.name
    return manifest.with_name(name + ".audit.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("build", "validate"), default="build")
    parser.add_argument("--route-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require_public_path(args.manifest)
    rows, audit = build_rows(args.route_dir)
    content = "".join(canonical_json(row) + "\n" for row in rows)
    if args.mode == "build":
        require(not args.manifest.exists(), f"refusing to overwrite manifest: {args.manifest}")
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(content, encoding="utf-8")
    else:
        require(args.manifest.is_file(), "manifest to validate is missing")
        require(args.manifest.read_text(encoding="utf-8") == content, "manifest differs from source files")
    validate_manifest_rows(rows)
    audit = {
        **audit,
        "mode": args.mode,
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256_file(args.manifest),
    }
    audit_path = args.audit_output or default_audit_path(args.manifest)
    require_public_path(audit_path)
    audit_path.write_text(canonical_json(audit) + "\n", encoding="utf-8")
    print(canonical_json({"status": audit["status"], "manifest": str(args.manifest), "audit": str(audit_path)}))


if __name__ == "__main__":
    main()
