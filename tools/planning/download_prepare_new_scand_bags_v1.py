#!/usr/bin/env python3
"""Selectively download and prelabel-process genuinely unused SCAND bags.

Safety boundary:
* a complete custodian-produced hash-only forbidden registry is mandatory;
* only the camera topic is decoded -- odometry/truth is never opened;
* the deterministic development/one-time-validation split is frozen before
  network access;
* downloaded bags and processed frames remain under incoming/quarantine;
* final cohort manifests are published only after the existing readiness
  checker passes all identity and route-disjoint checks.

The default invocation is plan-only.  Add ``--execute`` to perform network and
rosbag work after reviewing the frozen plan.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Mapping, Sequence
import urllib.error
import urllib.parse
import urllib.request

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from planning.proposal_tail_sidecar_v1 import (  # noqa: E402
    canonical_json,
    identity_hash,
    reject_forbidden_path,
    sha256_file,
)
from scoring.strict_inner_partitions import parse_scand_trajectory_id  # noqa: E402
from tools.planning.build_incoming_scand_route_manifest_v1 import (  # noqa: E402
    build_rows,
    validate_manifest_rows,
)
from tools.planning.check_proposal_tail_readiness_v1 import (  # noqa: E402
    evaluate_readiness,
    load_registry,
)


SCHEMA = "download_prepare_new_scand_bags_v1"
IMAGE_TOPICS = (
    "/image_raw/compressed",
    "/camera/rgb/image_raw/compressed",
)
SAMPLE_RATE_HZ = 4.0
PROCESSED_WIDTH = 320
PROCESSED_HEIGHT = 240
JPEG_QUALITY = 95
HISTORY_FRAMES = 4
GOAL_OFFSET_FRAMES = 4
DEFAULT_WINDOWS_PER_ROUTE = 40
MIN_WINDOWS_PER_ROUTE = 30
MAX_WINDOWS_PER_ROUTE = 50
DEFAULT_DEVELOPMENT_ROUTES = 12
DEFAULT_VALIDATION_ROUTES = 6
FINAL_TARGET_DEVELOPMENT_ROUTES = 20
FINAL_TARGET_VALIDATION_ROUTES = 10
ALLOWED_DOWNLOAD_HOST = "utexas.box.com"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--official-index",
        type=Path,
        default=ROOT
        / "data/proposal_tail_collection_v1/incoming/quarantine/official_scand/all_days_index.csv",
    )
    parser.add_argument(
        "--identity-inventory",
        type=Path,
        default=ROOT
        / "outputs/proposal_tail_collection_v1/official_scand_identity_inventory_20260723.json",
    )
    parser.add_argument(
        "--used-registry",
        type=Path,
        default=ROOT / "outputs/proposal_tail_collection_v1/used_v70_v77_identities.json",
    )
    parser.add_argument("--forbidden-registry", type=Path, required=True)
    parser.add_argument(
        "--incoming-root",
        type=Path,
        default=ROOT / "data/proposal_tail_collection_v1/incoming",
    )
    parser.add_argument("--development-routes", type=int, default=DEFAULT_DEVELOPMENT_ROUTES)
    parser.add_argument("--validation-routes", type=int, default=DEFAULT_VALIDATION_ROUTES)
    parser.add_argument("--windows-per-route", type=int, default=DEFAULT_WINDOWS_PER_ROUTE)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--max-total-download-gb", type=float, default=250.0)
    parser.add_argument("--min-free-after-download-gb", type=float, default=30.0)
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Perform downloads and camera-only processing. Default is plan-only.",
    )
    return parser.parse_args()


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def read_json(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"missing JSON: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON object required: {path}")
    return value


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(canonical_json(value) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_text_exact(path: Path, content: str) -> None:
    if path.exists():
        require(path.read_text(encoding="utf-8") == content, f"existing file differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def safe_float(value: Any, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"invalid numeric public metadata {field}: {value}") from error
    require(np.isfinite(result), f"non-finite public metadata {field}")
    return result


def stable_key(seed: int, *parts: Any) -> str:
    return hashlib.sha256("|".join((str(seed), *(str(part) for part in parts))).encode()).hexdigest()


def canonical_bag_id(file_name: str) -> str:
    value = str(file_name).strip()
    require(value and value.replace("_", "").isalnum(), f"unsafe official FileName: {value}")
    return f"random_mdps_{value}"


def validate_url(value: str) -> str:
    parsed = urllib.parse.urlparse(str(value).strip())
    require(parsed.scheme == "https", "SCAND bag URL must use HTTPS")
    require(parsed.hostname == ALLOWED_DOWNLOAD_HOST, f"unexpected SCAND download host: {parsed.hostname}")
    require(parsed.path.startswith("/shared/static/") and parsed.path.endswith(".bag"), "URL is not an official static bag link")
    require(not parsed.username and not parsed.password, "credentials in URL are forbidden")
    return parsed.geturl()


def public_rows(path: Path) -> dict[str, dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    result = {}
    for row in rows:
        file_name = str(row.get("FileName", "")).strip()
        require(file_name not in result, f"duplicate official FileName: {file_name}")
        bag = canonical_bag_id(file_name)
        parsed = parse_scand_trajectory_id(bag + "_0")
        result[file_name] = {
            "official_file_name": file_name,
            "bag_id": bag,
            "trajectory_id": bag + "_0",
            "route_group": parsed["route_group"],
            "collection_day": str(row.get("Date", "")).strip(),
            "person": str(row.get("Person", "")).strip(),
            "robot": str(row.get("Robot", "")).strip(),
            "start_location_display": str(row.get("Start Location", "")).strip(),
            "end_location_display": str(row.get("End Location", "")).strip(),
            "tags": str(row.get("Tags", "")).strip(),
            "distance_m": safe_float(row.get("Distance Travelled (meters)"), "distance"),
            "duration_seconds": safe_float(row.get("Time (secs)"), "duration"),
            "public_split": str(row.get("Train/Val", "")).strip(),
            "download_url": validate_url(str(row.get("Link to rosbag", ""))),
        }
    require(result, "official SCAND index is empty")
    return result


def filtered_candidates(
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, set[str]]]:
    inventory = read_json(args.identity_inventory)
    require(
        inventory.get("status") == "quarantine_fail_closed_missing_forbidden_registry",
        "identity inventory is not the expected pre-registry quarantine inventory",
    )
    require(
        inventory["inputs"]["official_index"]["sha256"] == sha256_file(args.official_index),
        "official index changed after identity inventory",
    )
    require(
        inventory["inputs"]["used_registry"]["sha256"] == sha256_file(args.used_registry),
        "used identity registry changed after inventory",
    )
    used, used_raw = load_registry(args.used_registry, forbidden=False)
    forbidden, forbidden_raw = load_registry(args.forbidden_registry, forbidden=True)
    require(str(forbidden_raw.get("custodian_id", "")).strip(), "forbidden registry lacks custodian_id")
    require(str(forbidden_raw.get("generated_at", "")).strip(), "forbidden registry lacks generated_at")
    index = public_rows(args.official_index)
    candidates = []
    for identity in inventory["rows"]:
        if not identity.get("identity_layer_clear"):
            continue
        row = index[str(identity["public_file_name"])]
        require(row["bag_id"] == identity["canonical_bag_id"], "inventory/index bag mismatch")
        require(row["route_group"] == identity["route_group"], "inventory/index route mismatch")
        hashes = {
            "route": identity_hash("route", row["route_group"]),
            "day": identity_hash("day", row["collection_day"]),
            "bag": identity_hash("bag", row["bag_id"]),
            "trajectory": identity_hash("trajectory", row["trajectory_id"]),
        }
        if any(hashes[kind] in used[kind] or hashes[kind] in forbidden[kind] for kind in hashes):
            continue
        candidates.append({**row, "identity_hashes": hashes})
    audit = {
        "official_index_sha256": sha256_file(args.official_index),
        "identity_inventory_sha256": sha256_file(args.identity_inventory),
        "used_registry_sha256": sha256_file(args.used_registry),
        "used_registry_status": used_raw.get("status"),
        "forbidden_registry_sha256": sha256_file(args.forbidden_registry),
        "forbidden_registry_status": forbidden_raw.get("status"),
        "forbidden_registry_custodian_id": forbidden_raw.get("custodian_id"),
        "input_identity_clear_bags": sum(bool(row.get("identity_layer_clear")) for row in inventory["rows"]),
        "registry_clear_bags": len(candidates),
        "registry_clear_routes": len({row["route_group"] for row in candidates}),
    }
    return candidates, audit, forbidden


def select_plan(args: argparse.Namespace, candidates: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    require(args.development_routes > 0 and args.validation_routes > 0, "both cohort route counts must be positive")
    require(MIN_WINDOWS_PER_ROUTE <= args.windows_per_route <= MAX_WINDOWS_PER_ROUTE, "windows-per-route must be 30..50")
    by_route: dict[str, list[Mapping[str, Any]]] = {}
    for row in candidates:
        by_route.setdefault(str(row["route_group"]), []).append(row)
    chosen = []
    for route, rows in by_route.items():
        # Duration is public, label-free metadata and is used only to avoid bags
        # that cannot provide the frozen number of image windows.
        selected = sorted(
            rows,
            key=lambda row: (
                -float(row["duration_seconds"]),
                stable_key(args.seed, "bag", row["bag_id"]),
            ),
        )[0]
        chosen.append(dict(selected))
    chosen.sort(key=lambda row: stable_key(args.seed, "route", row["route_group"]))
    total = args.development_routes + args.validation_routes
    require(len(chosen) >= total, f"only {len(chosen)} registry-clear routes; requested {total}")
    chosen = chosen[:total]
    # Allocate validation first and guarantee robot-platform representation in
    # both cohorts whenever a platform has at least two selected routes.
    by_robot: dict[str, list[int]] = {}
    for index, row in enumerate(chosen):
        by_robot.setdefault(str(row["robot"]), []).append(index)
    validation_indices: set[int] = set()
    for robot in sorted(by_robot):
        indices = by_robot[robot]
        if len(indices) >= 2 and len(validation_indices) < args.validation_routes:
            validation_indices.add(
                min(
                    indices,
                    key=lambda index: stable_key(
                        args.seed,
                        "validation-platform",
                        robot,
                        chosen[index]["route_group"],
                    ),
                )
            )
    remaining = sorted(
        (index for index in range(len(chosen)) if index not in validation_indices),
        key=lambda index: stable_key(
            args.seed, "validation-fill", chosen[index]["route_group"]
        ),
    )
    validation_indices.update(
        remaining[: args.validation_routes - len(validation_indices)]
    )
    for index, row in enumerate(chosen):
        row["cohort_role"] = (
            "one_time_validation" if index in validation_indices else "development"
        )
        row["windows_per_route"] = int(args.windows_per_route)
    require(
        sum(row["cohort_role"] == "development" for row in chosen)
        == args.development_routes,
        "development allocation cardinality mismatch",
    )
    require(
        not ({row["route_group"] for row in chosen if row["cohort_role"] == "development"}
             & {row["route_group"] for row in chosen if row["cohort_role"] == "one_time_validation"}),
        "development/validation route overlap",
    )
    return chosen


def freeze_plan(
    args: argparse.Namespace,
    selected: Sequence[Mapping[str, Any]],
    identity_audit: Mapping[str, Any],
    path: Path,
) -> dict[str, Any]:
    plan = {
        "schema": SCHEMA + "_plan",
        "status": "frozen_before_network_and_labels",
        "selection_rule": "one longest-duration public bag per route; seeded route subset; validation platform-stratified then seeded fill; route-disjoint",
        "seed": args.seed,
        "identity_audit": dict(identity_audit),
        "contract": {
            "sample_rate_hz": SAMPLE_RATE_HZ,
            "processed_resolution": [PROCESSED_WIDTH, PROCESSED_HEIGHT],
            "aspect_processing": "center crop to 4:3 then bicubic resize",
            "processed_encoding": "jpeg",
            "jpeg_quality": JPEG_QUALITY,
            "history_frames": HISTORY_FRAMES,
            "goal_offset_frames": GOAL_OFFSET_FRAMES,
            "windows_per_route": args.windows_per_route,
            "odom_or_truth_opened": False,
            "labels_opened": False,
        },
        "requested": {
            "development_routes": args.development_routes,
            "one_time_validation_routes": args.validation_routes,
        },
        "final_scientific_target": {
            "development_routes": FINAL_TARGET_DEVELOPMENT_ROUTES,
            "one_time_validation_routes": FINAL_TARGET_VALIDATION_ROUTES,
            "this_public_download_alone_satisfies_target": (
                args.development_routes >= FINAL_TARGET_DEVELOPMENT_ROUTES
                and args.validation_routes >= FINAL_TARGET_VALIDATION_ROUTES
            ),
        },
        "selected": list(selected),
        "network_started": False,
        "reserve_content_accessed": False,
    }
    content = canonical_json(plan) + "\n"
    write_text_exact(path, content)
    return plan


def request(url: str, *, timeout: float, headers: Mapping[str, str] | None = None, method: str | None = None):
    base = {"User-Agent": "SCAND-proposal-tail-ingest-v1/1.0"}
    if headers:
        base.update(headers)
    return urllib.request.urlopen(urllib.request.Request(url, headers=base, method=method), timeout=timeout)


def remote_size(url: str, timeout: float) -> int:
    try:
        with request(url, timeout=timeout, method="HEAD") as response:
            length = response.headers.get("Content-Length")
            if length is not None:
                return int(length)
    except urllib.error.HTTPError as error:
        if error.code not in (400, 403, 405):
            raise
    with request(url, timeout=timeout, headers={"Range": "bytes=0-0"}) as response:
        content_range = response.headers.get("Content-Range")
        require(content_range and "/" in content_range, "server did not report remote bag size")
        return int(content_range.rsplit("/", 1)[1])


def download_one(url: str, destination: Path, expected_size: int, timeout: float, retries: int) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        require(destination.stat().st_size == expected_size, f"existing bag size mismatch: {destination}")
        return {"resumed": False, "already_complete": True}
    partial = destination.with_suffix(destination.suffix + ".part")
    for attempt in range(1, retries + 1):
        try:
            existing = partial.stat().st_size if partial.exists() else 0
            headers = {"Range": f"bytes={existing}-"} if existing else {}
            with request(url, timeout=timeout, headers=headers) as response:
                status = int(getattr(response, "status", response.getcode()))
                mode = "ab" if existing and status == 206 else "wb"
                if existing and status != 206:
                    existing = 0
                with partial.open(mode) as handle:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        handle.write(chunk)
                    handle.flush()
                    os.fsync(handle.fileno())
            require(partial.stat().st_size == expected_size, f"downloaded size mismatch: {partial}")
            partial.replace(destination)
            return {"resumed": bool(existing), "already_complete": False}
        except Exception:
            if attempt >= retries:
                raise
            time.sleep(min(2**attempt, 10))
    raise AssertionError("unreachable")


def crop_resize(image: Image.Image) -> Image.Image:
    rgb = image.convert("RGB")
    width, height = rgb.size
    target_ratio = PROCESSED_WIDTH / PROCESSED_HEIGHT
    ratio = width / height
    if ratio > target_ratio:
        target_width = int(round(height * target_ratio))
        left = (width - target_width) // 2
        rgb = rgb.crop((left, 0, left + target_width, height))
    elif ratio < target_ratio:
        target_height = int(round(width / target_ratio))
        top = (height - target_height) // 2
        rgb = rgb.crop((0, top, width, top + target_height))
    return rgb.resize((PROCESSED_WIDTH, PROCESSED_HEIGHT), Image.Resampling.BICUBIC)


def choose_image_topic(connections: Sequence[Any]) -> str:
    available = {connection.topic for connection in connections}
    for topic in IMAGE_TOPICS:
        if topic in available:
            return topic
    raise RuntimeError(f"bag lacks supported compressed camera topic: {sorted(available)}")


def select_currents(frame_count: int, count: int) -> list[int]:
    eligible = list(range(HISTORY_FRAMES - 1, frame_count - GOAL_OFFSET_FRAMES))
    require(len(eligible) >= count, f"only {len(eligible)} eligible H4 windows; need {count}")
    indices = np.linspace(0, len(eligible) - 1, count)
    selected = sorted({eligible[int(round(index))] for index in indices})
    if len(selected) < count:
        selected_set = set(selected)
        selected.extend(value for value in eligible if value not in selected_set)
        selected = sorted(selected[:count])
    require(len(selected) == count, "window selection cardinality mismatch")
    return selected


def process_camera_only(
    bag_path: Path,
    destination: Path,
    row: Mapping[str, Any],
    forbidden: Mapping[str, set[str]],
) -> dict[str, Any]:
    try:
        from rosbags.highlevel import AnyReader
        from rosbags.typesys import Stores, get_typestore
    except ImportError as error:
        raise RuntimeError("nwm environment is missing rosbags") from error

    metadata_path = destination / "route_metadata.json"
    if metadata_path.exists():
        metadata = read_json(metadata_path)
        require(metadata.get("source_bag_sha256") == sha256_file(bag_path), "processed route source bag changed")
        rows, audit = build_rows(destination)
        return {"metadata": metadata, "manifest_rows": rows, "manifest_audit": audit, "reused": True}
    require(not destination.exists(), f"partial processed route directory exists: {destination}")
    destination.mkdir(parents=True)
    image_dir = destination / "images"
    image_dir.mkdir()
    typestore = get_typestore(Stores.ROS1_NOETIC)
    frames = []
    source_resolution_counts: dict[str, int] = {}
    source_formats: dict[str, int] = {}
    with AnyReader([bag_path], default_typestore=typestore) as reader:
        topic = choose_image_topic(reader.connections)
        connections = [connection for connection in reader.connections if connection.topic == topic]
        next_sample_ns = None
        previous_timestamp = None
        for connection, timestamp, rawdata in reader.messages(connections=connections):
            timestamp = int(timestamp)
            require(previous_timestamp is None or timestamp >= previous_timestamp, "camera timestamps are nonmonotonic")
            previous_timestamp = timestamp
            if next_sample_ns is not None and timestamp < next_sample_ns:
                continue
            message = reader.deserialize(rawdata, connection.msgtype)
            compressed = bytes(message.data)
            with Image.open(io.BytesIO(compressed)) as source:
                original_width, original_height = source.size
                processed = crop_resize(source)
            frame_index = len(frames)
            relative = f"images/{frame_index:06d}.jpg"
            processed.save(destination / relative, format="JPEG", quality=JPEG_QUALITY)
            source_format = str(getattr(message, "format", "") or "unknown")
            source_resolution_counts[f"{original_width}x{original_height}"] = (
                source_resolution_counts.get(f"{original_width}x{original_height}", 0) + 1
            )
            source_formats[source_format] = source_formats.get(source_format, 0) + 1
            frames.append(
                {
                    "frame_index": frame_index,
                    "timestamp": timestamp,
                    "image": relative,
                    "encoding": "jpeg",
                    "jpeg_quality": JPEG_QUALITY,
                    "source_width": original_width,
                    "source_height": original_height,
                    "source_compressed_format": source_format,
                    "source_message_timestamp_ns": timestamp,
                }
            )
            next_sample_ns = timestamp + int(round(1e9 / SAMPLE_RATE_HZ))

    current_values = select_currents(len(frames), int(row["windows_per_route"]))
    windows = []
    for current in current_values:
        uid = hashlib.sha256(
            f"proposal-tail-incoming-v1|{row['trajectory_id']}|{current}".encode()
        ).hexdigest()[:24]
        require(identity_hash("window", uid) not in forbidden["window"], "generated window overlaps forbidden registry")
        windows.append(
            {
                "window_uid": uid,
                "history_timesteps": list(range(current - HISTORY_FRAMES + 1, current + 1)),
                "current_t": current,
                "goal_t": current + GOAL_OFFSET_FRAMES,
            }
        )
    metadata = {
        "schema": "incoming_scand_route_source_v1",
        "public_nonreserve": True,
        "labels_opened": False,
        "cohort_role": row["cohort_role"],
        "route_group": row["route_group"],
        "route_id": row["route_group"],
        "collection_day": row["collection_day"],
        "bag_id": row["bag_id"],
        "trajectory_id": row["trajectory_id"],
        "timestamp_unit": "nanoseconds",
        "jpeg_quality": JPEG_QUALITY,
        "frames": frames,
        "windows": windows,
        "source_official_index_sha256": row["source_official_index_sha256"],
        "source_official_file_name": row["official_file_name"],
        "source_public_split": row["public_split"],
        "source_download_url": row["download_url"],
        "source_bag_path": str(bag_path.resolve()),
        "source_bag_size": bag_path.stat().st_size,
        "source_bag_sha256": sha256_file(bag_path),
        "camera_preprocessing": {
            "camera_topic": topic,
            "sample_rate_hz": SAMPLE_RATE_HZ,
            "source_resolution_counts": dict(sorted(source_resolution_counts.items())),
            "source_compressed_format_counts": dict(sorted(source_formats.items())),
            "processed_width": PROCESSED_WIDTH,
            "processed_height": PROCESSED_HEIGHT,
            "aspect_processing": "center crop to 4:3 then bicubic resize",
            "processed_encoding": "jpeg",
            "jpeg_quality": JPEG_QUALITY,
            "odom_topic_opened": False,
            "truth_or_labels_opened": False,
        },
        "environment_public_metadata": {
            "person": row["person"],
            "robot": row["robot"],
            "start_location_display": row["start_location_display"],
            "end_location_display": row["end_location_display"],
            "tags": row["tags"],
            "distance_m": row["distance_m"],
            "duration_seconds": row["duration_seconds"],
        },
    }
    write_json_atomic(metadata_path, metadata)
    manifest_rows, manifest_audit = build_rows(destination)
    return {"metadata": metadata, "manifest_rows": manifest_rows, "manifest_audit": manifest_audit, "reused": False}


def manifest_content(rows: Sequence[Mapping[str, Any]]) -> str:
    validate_manifest_rows(rows)
    return "".join(canonical_json(row) + "\n" for row in rows)


def execute(args: argparse.Namespace, plan: Mapping[str, Any], forbidden: Mapping[str, set[str]], plan_path: Path) -> dict[str, Any]:
    quarantine = args.incoming_root / "quarantine/official_scand_selective_v1"
    bag_root = quarantine / "bags"
    processed_root = quarantine / "processed"
    staging_root = quarantine / "staging_manifests"
    for path in (quarantine, bag_root, processed_root, staging_root):
        reject_forbidden_path(path)
    sizes = {}
    for row in plan["selected"]:
        sizes[row["bag_id"]] = remote_size(row["download_url"], args.timeout_seconds)
    total_bytes = sum(sizes.values())
    require(total_bytes <= args.max_total_download_gb * (1024**3), "planned downloads exceed max-total-download-gb")
    args.incoming_root.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(args.incoming_root).free
    require(
        free - total_bytes >= args.min_free_after_download_gb * (1024**3),
        "insufficient disk to preserve required free space after download",
    )

    cohort_rows: dict[str, list[dict[str, Any]]] = {
        "development": [],
        "one_time_validation": [],
    }
    bag_audit = []
    for original in plan["selected"]:
        row = {**original, "source_official_index_sha256": plan["identity_audit"]["official_index_sha256"]}
        bag_path = bag_root / f"{row['official_file_name']}.bag"
        download = download_one(
            row["download_url"],
            bag_path,
            sizes[row["bag_id"]],
            args.timeout_seconds,
            args.retries,
        )
        destination = processed_root / row["cohort_role"] / row["bag_id"]
        processed = process_camera_only(bag_path, destination, row, forbidden)
        cohort_rows[row["cohort_role"]].extend(processed["manifest_rows"])
        bag_audit.append(
            {
                "bag_id": row["bag_id"],
                "route_group": row["route_group"],
                "cohort_role": row["cohort_role"],
                "bag_path": str(bag_path),
                "bag_size": bag_path.stat().st_size,
                "bag_sha256": sha256_file(bag_path),
                "download": download,
                "processed_route": str(destination),
                "processed_frames": len(processed["metadata"]["frames"]),
                "processed_windows": len(processed["manifest_rows"]),
                "source_resolution_counts": processed["metadata"]["camera_preprocessing"]["source_resolution_counts"],
                "source_formats": processed["metadata"]["camera_preprocessing"]["source_compressed_format_counts"],
                "reused_processing": processed["reused"],
            }
        )

    development_staging = staging_root / "development.manifest.windows.jsonl"
    validation_staging = staging_root / "one_time_validation.manifest.windows.jsonl"
    write_text_exact(development_staging, manifest_content(cohort_rows["development"]))
    write_text_exact(validation_staging, manifest_content(cohort_rows["one_time_validation"]))
    readiness = evaluate_readiness(
        development_staging,
        validation_staging,
        args.used_registry,
        args.forbidden_registry,
    )
    readiness_path = quarantine / "readiness.prepublish.json"
    write_json_atomic(readiness_path, readiness)
    require(readiness["ready"], f"prepublish readiness failed: {readiness['errors']}")

    development_final = args.incoming_root / "development_v1/manifest.windows.jsonl"
    validation_final = args.incoming_root / "one_time_validation_v1/manifest.windows.jsonl"
    write_text_exact(development_final, development_staging.read_text(encoding="utf-8"))
    write_text_exact(validation_final, validation_staging.read_text(encoding="utf-8"))
    final_readiness = evaluate_readiness(
        development_final,
        validation_final,
        args.used_registry,
        args.forbidden_registry,
    )
    final_readiness_path = args.incoming_root / "readiness.new_scand_bags_v1.json"
    write_json_atomic(final_readiness_path, final_readiness)
    require(final_readiness["ready"], f"final readiness failed: {final_readiness['errors']}")

    result = {
        "schema": SCHEMA + "_result",
        "status": "complete_prelabel_ready",
        "plan": str(plan_path),
        "plan_sha256": sha256_file(plan_path),
        "downloaded_bytes": total_bytes,
        "bags": bag_audit,
        "development": {
            "routes": args.development_routes,
            "windows": len(cohort_rows["development"]),
            "manifest": str(development_final),
            "manifest_sha256": sha256_file(development_final),
        },
        "one_time_validation": {
            "routes": args.validation_routes,
            "windows": len(cohort_rows["one_time_validation"]),
            "manifest": str(validation_final),
            "manifest_sha256": sha256_file(validation_final),
            "labels_opened": False,
        },
        "camera_contract": plan["contract"],
        "environment": {
            "python_executable": sys.executable,
            "python_version": sys.version.split()[0],
            "numpy_version": np.__version__,
            "pillow_version": Image.__version__,
            "rosbags_required": True,
        },
        "readiness": final_readiness,
        "readiness_path": str(final_readiness_path),
        "readiness_sha256": sha256_file(final_readiness_path),
        "odom_or_truth_opened": False,
        "reserve_content_accessed": False,
        "training_started": False,
    }
    result_path = quarantine / "ingest.result.json"
    write_json_atomic(result_path, result)
    return {**result, "result_path": str(result_path), "result_sha256": sha256_file(result_path)}


def main() -> None:
    args = parse_args()
    reject_forbidden_path(args.incoming_root)
    require(args.retries >= 1, "retries must be positive")
    require(args.timeout_seconds > 0, "timeout must be positive")
    candidates, identity_audit, forbidden = filtered_candidates(args)
    selected = select_plan(args, candidates)
    plan_path = args.incoming_root / "quarantine/official_scand_selective_v1/download_plan.json"
    plan = freeze_plan(args, selected, identity_audit, plan_path)
    if not args.execute:
        print(
            canonical_json(
                {
                    "status": "plan_frozen_no_network",
                    "plan": str(plan_path),
                    "plan_sha256": sha256_file(plan_path),
                    "selected_bags": len(selected),
                    "selected_routes": len({row['route_group'] for row in selected}),
                    "development_routes": args.development_routes,
                    "one_time_validation_routes": args.validation_routes,
                    "execute_required_for_download": True,
                }
            )
        )
        return
    result = execute(args, plan, forbidden, plan_path)
    print(canonical_json(result))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(
            canonical_json(
                {
                    "schema": SCHEMA + "_failure",
                    "status": "fail_closed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "network_may_have_started": "--execute" in sys.argv,
                    "reserve_content_accessed": False,
                }
            ),
            file=sys.stderr,
        )
        raise SystemExit(2) from error
