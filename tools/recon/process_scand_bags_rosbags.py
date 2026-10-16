#!/usr/bin/env python3
"""Convert locked SCAND ROS1 bags into the ViNT/RECON trajectory layout."""

from __future__ import annotations

import argparse
import io
import json
import math
import pickle
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from recon_latents.split_lock import file_sha256, load_split_lock  # noqa: E402


IMAGE_TOPICS = (
    "/image_raw/compressed",
    "/camera/rgb/image_raw/compressed",
)
ODOM_TOPICS = (
    "/odom",
    "/jackal_velocity_controller/odom",
)
LTWM_HISTORY_FRAMES = 4
LTWM_FUTURE_FRAMES = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-lock", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--sample-rate", type=float, default=4.0)
    parser.add_argument(
        "--min-segment-frames",
        type=int,
        default=1,
        help=(
            "Minimum frames retained after NWM-compatible forward segmentation. "
            "The official NWM SCAND preprocessing retains short segments; the "
            "LTWM 4-context/4-future eligibility gate is audited separately."
        ),
    )
    parser.add_argument(
        "--roles",
        default="train,val,test",
        help="Comma-separated bag roles to convert. Reserve is excluded by default.",
    )
    parser.add_argument("--max-bags", type=int, default=None)
    parser.add_argument(
        "--bag-name",
        default=None,
        help="Optional exact bag filename for a bounded smoke conversion.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def quaternion_yaw(orientation: Any) -> float:
    x = float(orientation.x)
    y = float(orientation.y)
    z = float(orientation.z)
    w = float(orientation.w)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def process_compressed_image(message: Any) -> Image.Image:
    image = Image.open(io.BytesIO(bytes(message.data))).convert("RGB")
    width, height = image.size
    target_width = int(round(height * 4.0 / 3.0))
    if target_width < width:
        left = (width - target_width) // 2
        image = image.crop((left, 0, left + target_width, height))
    elif target_width > width:
        target_height = int(round(width * 3.0 / 4.0))
        top = max(0, (height - target_height) // 2)
        image = image.crop((0, top, width, top + target_height))
    return image.resize((320, 240), Image.Resampling.BICUBIC)


def choose_topic(connections: Iterable[Any], candidates: tuple[str, ...]) -> str:
    available = {connection.topic for connection in connections}
    for topic in candidates:
        if topic in available:
            return topic
    raise ValueError(f"none of the required topics are present: {candidates}")


def forward_segments(
    samples: list[tuple[Image.Image, np.ndarray, float, int]],
    min_segment_frames: int,
) -> list[list[tuple[Image.Image, np.ndarray, float, int]]]:
    segments: list[list[tuple[Image.Image, np.ndarray, float, int]]] = []
    current: list[tuple[Image.Image, np.ndarray, float, int]] = []
    for index in range(1, len(samples)):
        previous = samples[index - 1]
        present = samples[index]
        delta = present[1] - previous[1]
        forward = float(delta[0] * math.cos(previous[2]) + delta[1] * math.sin(previous[2]))
        if forward >= 1e-5:
            if not current:
                current.append(previous)
            current.append(present)
        elif current:
            if len(current) >= min_segment_frames:
                segments.append(current)
            current = []
    if len(current) >= min_segment_frames:
        segments.append(current)
    return segments


def eligible_ltwm_windows(frame_count: int) -> int:
    return max(0, frame_count - LTWM_HISTORY_FRAMES - LTWM_FUTURE_FRAMES + 1)


def select_official_segments(
    segments: list[list[tuple[Image.Image, np.ndarray, float, int]]],
    official_ids: list[str],
) -> tuple[
    list[list[tuple[Image.Image, np.ndarray, float, int]]],
    list[dict[str, int]],
]:
    """Match official NWM segment cardinality without a global length threshold."""

    if not official_ids:
        return segments, []
    expected_count = len(official_ids)
    if len(segments) < expected_count:
        raise ValueError(
            f"raw forward segmentation produced {len(segments)} segments, "
            f"but NWM lists {expected_count}"
        )
    ranked_indices = sorted(
        range(len(segments)),
        key=lambda index: (-len(segments[index]), index),
    )
    selected_indices = sorted(ranked_indices[:expected_count])
    selected_index_set = set(selected_indices)
    discarded = [
        {"raw_segment_index": index, "frames": len(segment)}
        for index, segment in enumerate(segments)
        if index not in selected_index_set
    ]
    return [segments[index] for index in selected_indices], discarded


def read_samples(
    bag_path: Path,
    sample_rate: float,
) -> tuple[list[tuple[Image.Image, np.ndarray, float, int]], dict[str, Any]]:
    try:
        from rosbags.highlevel import AnyReader
        from rosbags.typesys import Stores, get_typestore
    except ImportError as exc:
        raise ImportError(
            "SCAND conversion requires rosbags; install requirements-dinov3-scand.txt"
        ) from exc

    typestore = get_typestore(Stores.ROS1_NOETIC)
    with AnyReader([bag_path], default_typestore=typestore) as reader:
        image_topic = choose_topic(reader.connections, IMAGE_TOPICS)
        odom_topic = choose_topic(reader.connections, ODOM_TOPICS)
        selected = [
            connection
            for connection in reader.connections
            if connection.topic in {image_topic, odom_topic}
        ]
        latest_image = None
        latest_position = None
        latest_yaw = None
        next_sample_ns = None
        samples = []
        for connection, timestamp, rawdata in reader.messages(connections=selected):
            message = reader.deserialize(rawdata, connection.msgtype)
            if connection.topic == image_topic:
                latest_image = process_compressed_image(message)
            else:
                position = message.pose.pose.position
                latest_position = np.asarray(
                    [float(position.x), float(position.y)], dtype=np.float32
                )
                latest_yaw = quaternion_yaw(message.pose.pose.orientation)
            if next_sample_ns is None:
                next_sample_ns = int(timestamp)
            if (
                int(timestamp) >= next_sample_ns
                and latest_image is not None
                and latest_position is not None
                and latest_yaw is not None
            ):
                samples.append(
                    (
                        latest_image.copy(),
                        latest_position.copy(),
                        float(latest_yaw),
                        int(timestamp),
                    )
                )
                next_sample_ns = int(timestamp + round(1e9 / sample_rate))
        return samples, {
            "image_topic": image_topic,
            "odom_topic": odom_topic,
            "raw_sample_count": len(samples),
        }


def write_segment(
    output_root: Path,
    trajectory_id: str,
    segment: list[tuple[Image.Image, np.ndarray, float, int]],
    overwrite: bool,
) -> dict[str, Any]:
    output = output_root / trajectory_id
    if output.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True, exist_ok=True)
    for index, (image, _, _, _) in enumerate(segment):
        image.save(output / f"{index}.jpg", quality=95)
    positions = np.stack([item[1] for item in segment]).astype(np.float32)
    yaw = np.asarray([item[2] for item in segment], dtype=np.float32)
    timestamps = np.asarray([item[3] for item in segment], dtype=np.int64)
    with (output / "traj_data.pkl").open("wb") as handle:
        pickle.dump(
            {
                "position": positions,
                "yaw": yaw,
                "timestamps_ns": timestamps,
            },
            handle,
        )
    return {
        "trajectory_id": trajectory_id,
        "frames": len(segment),
        "output_dir": str(output),
        "first_timestamp_ns": int(timestamps[0]),
        "last_timestamp_ns": int(timestamps[-1]),
    }


def main() -> None:
    args = parse_args()
    if args.sample_rate <= 0:
        raise ValueError("--sample-rate must be positive")
    roles = {item.strip() for item in args.roles.split(",") if item.strip()}
    lock, lock_sha = load_split_lock(args.split_lock)
    if args.min_segment_frames < 1:
        raise ValueError("--min-segment-frames must be >= 1")
    bags = [entry for entry in lock["scand"]["bags"] if entry["split"] in roles]
    if args.bag_name is not None:
        bags = [entry for entry in bags if entry["bag_name"] == args.bag_name]
        if not bags:
            raise ValueError(f"bag is absent from selected roles: {args.bag_name}")
    if args.max_bags is not None:
        bags = bags[: args.max_bags]
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    for entry in bags:
        bag_path = Path(entry["bag_path"])
        actual_hash = file_sha256(bag_path)
        if actual_hash != entry["bag_sha256"]:
            raise ValueError(f"bag changed after split lock: {bag_path}")
        samples, topic_audit = read_samples(bag_path, args.sample_rate)
        raw_segments = forward_segments(samples, args.min_segment_frames)
        official_ids = sorted(
            set(entry["nwm_official"]["train"]) | set(entry["nwm_official"]["test"])
        )
        try:
            segments, discarded_segments = select_official_segments(
                raw_segments, official_ids
            )
        except ValueError as exc:
            raise ValueError(f"{entry['bag_name']}: {exc}") from exc
        derived_ids = official_ids or [
            f"{entry['trajectory_prefix']}_{index}" for index in range(len(segments))
        ]
        trajectories = [
            write_segment(
                args.output_root,
                trajectory_id,
                segment,
                args.overwrite,
            )
            for trajectory_id, segment in zip(derived_ids, segments, strict=True)
        ]
        manifest_rows.append(
            {
                **entry,
                **topic_audit,
                "sample_rate_hz": args.sample_rate,
                "min_segment_frames": args.min_segment_frames,
                "raw_forward_segment_count": len(raw_segments),
                "raw_forward_segment_lengths": [
                    len(segment) for segment in raw_segments
                ],
                "official_segment_selection": (
                    "nwm_cardinality_longest_then_chronological"
                    if official_ids
                    else "not_applicable"
                ),
                "discarded_raw_segments": discarded_segments,
                "ltwm_protocol": {
                    "history_frames": LTWM_HISTORY_FRAMES,
                    "future_frames": LTWM_FUTURE_FRAMES,
                    "minimum_window_frames": (
                        LTWM_HISTORY_FRAMES + LTWM_FUTURE_FRAMES
                    ),
                    "eligible_windows": sum(
                        eligible_ltwm_windows(len(segment)) for segment in segments
                    ),
                },
                "trajectory_count": len(trajectories),
                "trajectories": trajectories,
            }
        )
        print(
            json.dumps(
                {
                    "bag": entry["bag_name"],
                    "samples": len(samples),
                    "raw_segments": len(raw_segments),
                    "segments": len(segments),
                    "discarded_segments": discarded_segments,
                }
            ),
            flush=True,
        )
    manifest = {
        "schema_version": 1,
        "split_lock_sha256": lock_sha,
        "converter": "rosbags_anyreader_scand_v1",
        "bags": manifest_rows,
    }
    manifest_path = args.output_root / "conversion_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "PASS",
                "bags": len(manifest_rows),
                "trajectories": sum(row["trajectory_count"] for row in manifest_rows),
                "manifest": str(manifest_path),
                "manifest_sha256": file_sha256(manifest_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
