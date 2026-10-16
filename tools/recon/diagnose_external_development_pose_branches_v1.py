#!/usr/bin/env python3
"""Find natural same-pose/different-future branches in two development bags."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
PROCESSED_ROOT = (
    ROOT
    / "data/proposal_tail_collection_v1/incoming/quarantine"
    / "official_scand_selective_v4_contract_compatible/processed/development"
)
DEFAULT_OUTPUT = (
    ROOT
    / "outputs/highres_dual_ltwm_v1/external_development_pose_branches_v1/result.json"
)
SAMPLE_RATE_HZ = 4.0
MIN_SEPARATION_FRAMES = 16
POSITION_TOLERANCE_M = 0.20
YAW_TOLERANCE_DEG = 5.0
FUTURE_FRAMES = 4
MIN_PATH_RMS_DIVERGENCE_M = 0.08
MIN_ENDPOINT_DIVERGENCE_M = 0.10
MIN_YAW_DIVERGENCE_DEG = 8.0

BAGS = (
    (
        "random_mdps_A_Jackal_Libraryfront_Librarydoors_Thu_Nov_4_20",
        Path("/mnt/d/VEDIOS/A_Jackal_Libraryfront_Librarydoors_Thu_Nov_4_20.bag"),
    ),
    (
        "random_mdps_A_Spot_Thompson_Butler_Sat_Nov_13_103",
        Path("/mnt/d/VEDIOS/A_Spot_Thompson_Butler_Sat_Nov_13_103.bag"),
    ),
)
ODOM_TOPICS = ("/odom", "/jackal_velocity_controller/odom")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def wrap_angle(value: np.ndarray | float) -> np.ndarray | float:
    return (value + np.pi) % (2.0 * np.pi) - np.pi


def yaw_from_quaternion(orientation: Any) -> float:
    x = float(orientation.x)
    y = float(orientation.y)
    z = float(orientation.z)
    w = float(orientation.w)
    return math.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )


def read_odometry(path: Path) -> dict[str, np.ndarray | str]:
    from rosbags.highlevel import AnyReader
    from rosbags.typesys import Stores, get_typestore

    typestore = get_typestore(Stores.ROS1_NOETIC)
    with AnyReader([path], default_typestore=typestore) as reader:
        available = {connection.topic for connection in reader.connections}
        topic = next((name for name in ODOM_TOPICS if name in available), None)
        require(topic is not None, f"no supported odometry topic in {path}")
        selected = [
            connection for connection in reader.connections if connection.topic == topic
        ]
        next_sample_ns = None
        timestamps = []
        positions = []
        yaws = []
        period_ns = int(round(1.0e9 / SAMPLE_RATE_HZ))
        for connection, timestamp, rawdata in reader.messages(connections=selected):
            if next_sample_ns is not None and int(timestamp) < next_sample_ns:
                continue
            message = reader.deserialize(rawdata, connection.msgtype)
            pose = message.pose.pose
            timestamps.append(int(timestamp))
            positions.append((float(pose.position.x), float(pose.position.y)))
            yaws.append(yaw_from_quaternion(pose.orientation))
            next_sample_ns = int(timestamp) + period_ns
    require(len(timestamps) > FUTURE_FRAMES, f"too few odometry samples: {path}")
    return {
        "topic": str(topic),
        "timestamps_ns": np.asarray(timestamps, dtype=np.int64),
        "position": np.asarray(positions, dtype=np.float64),
        "yaw": np.asarray(yaws, dtype=np.float64),
    }


def local_future(
    position: np.ndarray,
    yaw: np.ndarray,
    index: int,
) -> tuple[np.ndarray, np.ndarray]:
    delta = position[index + 1 : index + FUTURE_FRAMES + 1] - position[index]
    cosine = math.cos(float(yaw[index]))
    sine = math.sin(float(yaw[index]))
    rotation = np.asarray(((cosine, sine), (-sine, cosine)))
    xy = delta @ rotation.T
    dyaw = wrap_angle(yaw[index + 1 : index + FUTURE_FRAMES + 1] - yaw[index])
    return xy, np.asarray(dyaw)


def analyze(data: dict[str, np.ndarray | str]) -> dict[str, Any]:
    position = np.asarray(data["position"])
    yaw = np.asarray(data["yaw"])
    valid = len(position) - FUTURE_FRAMES
    pairs = []
    position_candidates = 0
    heading_candidates = 0
    for left in range(valid):
        start = left + MIN_SEPARATION_FRAMES
        if start >= valid:
            continue
        distance = np.linalg.norm(position[start:valid] - position[left], axis=1)
        for right in (np.flatnonzero(distance <= POSITION_TOLERANCE_M) + start):
            position_candidates += 1
            yaw_error = abs(float(wrap_angle(yaw[right] - yaw[left])))
            if yaw_error > math.radians(YAW_TOLERANCE_DEG):
                continue
            heading_candidates += 1
            left_xy, left_yaw = local_future(position, yaw, left)
            right_xy, right_yaw = local_future(position, yaw, int(right))
            path_rms = float(np.sqrt(np.mean((left_xy - right_xy) ** 2)))
            endpoint = float(np.linalg.norm(left_xy[-1] - right_xy[-1]))
            yaw_divergence = float(
                np.degrees(np.max(np.abs(wrap_angle(left_yaw - right_yaw))))
            )
            different_future = (
                path_rms >= MIN_PATH_RMS_DIVERGENCE_M
                or endpoint >= MIN_ENDPOINT_DIVERGENCE_M
                or yaw_divergence >= MIN_YAW_DIVERGENCE_DEG
            )
            if different_future:
                pairs.append(
                    {
                        "left_index": left,
                        "right_index": int(right),
                        "separation_seconds": float(
                            (
                                np.asarray(data["timestamps_ns"])[right]
                                - np.asarray(data["timestamps_ns"])[left]
                            )
                            / 1.0e9
                        ),
                        "start_position_distance_m": float(
                            np.linalg.norm(position[right] - position[left])
                        ),
                        "start_yaw_difference_deg": float(np.degrees(yaw_error)),
                        "future_path_rms_divergence_m": path_rms,
                        "future_endpoint_divergence_m": endpoint,
                        "future_yaw_divergence_deg": yaw_divergence,
                    }
                )
    pairs.sort(
        key=lambda row: (
            row["future_endpoint_divergence_m"],
            row["future_path_rms_divergence_m"],
        ),
        reverse=True,
    )
    selected = []
    used: list[int] = []
    for row in pairs:
        indices = (row["left_index"], row["right_index"])
        if any(abs(index - prior) < 8 for index in indices for prior in used):
            continue
        selected.append(row)
        used.extend(indices)
    return {
        "odometry_samples": len(position),
        "duration_seconds": float(
            (
                np.asarray(data["timestamps_ns"])[-1]
                - np.asarray(data["timestamps_ns"])[0]
            )
            / 1.0e9
        ),
        "position_candidates": position_candidates,
        "position_and_heading_candidates": heading_candidates,
        "different_future_pairs": len(pairs),
        "independent_anchor_pairs": len(selected),
        "top_independent_pairs": selected[:30],
    }


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    require(not output.exists(), f"refusing to overwrite {output}")
    started = time.perf_counter()
    rows = {}
    for route_id, bag_path in BAGS:
        metadata_path = PROCESSED_ROOT / route_id / "route_metadata.json"
        require(metadata_path.is_file(), f"missing development metadata: {metadata_path}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        require(metadata.get("cohort_role") == "development", f"non-development bag: {route_id}")
        require(
            metadata["camera_preprocessing"].get("truth_or_labels_opened") is False,
            f"labels already opened: {route_id}",
        )
        require(bag_path.is_file(), f"missing raw bag: {bag_path}")
        before = time.perf_counter()
        data = read_odometry(bag_path)
        rows[route_id] = {
            "bag_path": str(bag_path),
            "bag_size_bytes": bag_path.stat().st_size,
            "odometry_topic": data["topic"],
            **analyze(data),
            "runtime_seconds": time.perf_counter() - before,
        }
        print(
            json.dumps(
                {
                    "route": route_id,
                    "independent_anchor_pairs": rows[route_id][
                        "independent_anchor_pairs"
                    ],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    total = sum(row["independent_anchor_pairs"] for row in rows.values())
    result = {
        "schema": "external_development_pose_branches_result_v1",
        "status": "complete_bounded_two_bag_development_diagnostic",
        "decision": (
            "GO_ENCODE_MINIMAL_NATURAL_BRANCH_PILOT"
            if total >= 10
            else "NO_GO_EXISTING_DEVELOPMENT_BAGS_LACK_BRANCH_ANCHORS"
        ),
        "thresholds": {
            "sample_rate_hz": SAMPLE_RATE_HZ,
            "minimum_temporal_separation_frames": MIN_SEPARATION_FRAMES,
            "position_tolerance_m": POSITION_TOLERANCE_M,
            "yaw_tolerance_deg": YAW_TOLERANCE_DEG,
            "future_frames": FUTURE_FRAMES,
            "minimum_path_rms_divergence_m": MIN_PATH_RMS_DIVERGENCE_M,
            "minimum_endpoint_divergence_m": MIN_ENDPOINT_DIVERGENCE_M,
            "minimum_yaw_divergence_deg": MIN_YAW_DIVERGENCE_DEG,
            "minimum_independent_anchors_for_latent_pilot": 10,
        },
        "routes": rows,
        "total_independent_anchor_pairs": total,
        "protection": {
            "development_bags_opened": 2,
            "one_time_validation_bags_opened": 0,
            "reserve_bags_opened": 0,
            "image_topics_opened": 0,
            "outcome_labels_opened": False,
            "sealed_validation_opened": False,
            "formal_v70_mutated": False,
            "frozen_ltwm_mutated": False,
        },
        "runner": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "runtime_seconds": time.perf_counter() - started,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"result": str(output), "decision": result["decision"]}, sort_keys=True))


if __name__ == "__main__":
    main()
