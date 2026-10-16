#!/usr/bin/env python3
"""Held-route LiDAR path-observability probe on three existing SCAND bags.

The current point cloud is held fixed while one factual and eleven observed
alternative H1-H4 ego trajectories are scored for geometric clearance.  This
probe never loads CEM outcomes, ATE, objective labels, or sealed validation.
It is a feasibility gate for an explicit robot-frame occupancy latent, not a
deployment selector.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np


ODOM_TOPIC = "/odom"
POINT_TOPIC = "/velodyne_points"
TF_TOPICS = {"/tf", "/tf_static"}
HORIZON_SECONDS = np.asarray((0.5, 1.0, 1.5, 2.0), dtype=np.float64)
CANDIDATES = 12
NEGATIVE_STRIDE = 17
VOXEL_METERS = 0.10
ROBOT_RADIUS_METERS = 0.38
PATH_SAMPLE_METERS = 0.10
OBSTACLE_Z_RANGE_METERS = (0.10, 1.60)
MAX_RANGE_METERS = 6.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-bag", type=Path, action="append", required=True)
    parser.add_argument("--held-bag", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def quaternion_yaw(orientation: Any) -> float:
    x = float(orientation.x)
    y = float(orientation.y)
    z = float(orientation.z)
    w = float(orientation.w)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def quaternion_matrix(rotation: Any) -> np.ndarray:
    x = float(rotation.x)
    y = float(rotation.y)
    z = float(rotation.z)
    w = float(rotation.w)
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    require(norm > 1.0e-8, "invalid TF quaternion")
    x /= norm
    y /= norm
    z /= norm
    w /= norm
    return np.asarray(
        (
            (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )


def read_velodyne_transform(path: Path) -> tuple[np.ndarray, np.ndarray]:
    from rosbags.highlevel import AnyReader
    from rosbags.typesys import Stores, get_typestore

    typestore = get_typestore(Stores.ROS1_NOETIC)
    with AnyReader([path], default_typestore=typestore) as reader:
        connections = [
            connection
            for connection in reader.connections
            if connection.topic in TF_TOPICS
        ]
        for connection, _, rawdata in reader.messages(connections=connections):
            message = reader.deserialize(rawdata, connection.msgtype)
            for transform in message.transforms:
                parent = str(transform.header.frame_id).lstrip("/")
                child = str(transform.child_frame_id).lstrip("/")
                if parent == "base_link" and child == "velodyne":
                    translation = transform.transform.translation
                    return (
                        quaternion_matrix(transform.transform.rotation),
                        np.asarray(
                            (
                                float(translation.x),
                                float(translation.y),
                                float(translation.z),
                            ),
                            dtype=np.float64,
                        ),
                    )
    raise RuntimeError(f"{path.name}: missing base_link -> velodyne TF")


def pointcloud_xyz(message: Any) -> np.ndarray:
    fields = {str(field.name): field for field in message.fields}
    require({"x", "y", "z"} <= set(fields), "point cloud lacks xyz fields")
    for name in ("x", "y", "z"):
        require(int(fields[name].datatype) == 7, f"{name} is not float32")
        require(int(fields[name].count) == 1, f"{name} count is not one")
    endian = ">" if bool(message.is_bigendian) else "<"
    dtype = np.dtype(
        {
            "names": ("x", "y", "z"),
            "formats": (endian + "f4", endian + "f4", endian + "f4"),
            "offsets": (
                int(fields["x"].offset),
                int(fields["y"].offset),
                int(fields["z"].offset),
            ),
            "itemsize": int(message.point_step),
        }
    )
    count = int(message.width) * max(1, int(message.height))
    raw = np.asarray(message.data, dtype=np.uint8).tobytes()
    points = np.frombuffer(raw, dtype=dtype, count=count)
    return np.column_stack((points["x"], points["y"], points["z"])).astype(
        np.float64,
        copy=False,
    )


def filter_and_voxelize(points: np.ndarray) -> np.ndarray:
    finite = np.isfinite(points).all(axis=1)
    radius = np.linalg.vector_norm(points[:, :2], axis=1)
    keep = (
        finite
        & (points[:, 2] >= OBSTACLE_Z_RANGE_METERS[0])
        & (points[:, 2] <= OBSTACLE_Z_RANGE_METERS[1])
        & (radius >= 0.55)
        & (radius <= MAX_RANGE_METERS)
        & (points[:, 0] >= -0.50)
    )
    xy = points[keep, :2].astype(np.float32)
    if len(xy) == 0:
        return xy
    cells = np.floor(xy / VOXEL_METERS).astype(np.int32)
    _, indices = np.unique(cells, axis=0, return_index=True)
    return xy[np.sort(indices)]


def read_bag(path: Path) -> dict[str, Any]:
    from rosbags.highlevel import AnyReader
    from rosbags.typesys import Stores, get_typestore

    rotation, translation = read_velodyne_transform(path)
    typestore = get_typestore(Stores.ROS1_NOETIC)
    odometry: list[tuple[int, float, float, float]] = []
    clouds: list[tuple[int, np.ndarray]] = []
    with AnyReader([path], default_typestore=typestore) as reader:
        available = {connection.topic for connection in reader.connections}
        require(
            {ODOM_TOPIC, POINT_TOPIC} <= available,
            f"{path.name}: missing odom or point cloud",
        )
        connections = [
            connection
            for connection in reader.connections
            if connection.topic in {ODOM_TOPIC, POINT_TOPIC}
        ]
        for connection, timestamp, rawdata in reader.messages(connections=connections):
            message = reader.deserialize(rawdata, connection.msgtype)
            if connection.topic == ODOM_TOPIC:
                pose = message.pose.pose
                odometry.append(
                    (
                        int(timestamp),
                        float(pose.position.x),
                        float(pose.position.y),
                        quaternion_yaw(pose.orientation),
                    )
                )
            else:
                sensor_points = pointcloud_xyz(message)
                base_points = sensor_points @ rotation.T + translation[None]
                clouds.append(
                    (int(timestamp), filter_and_voxelize(base_points))
                )
    require(len(odometry) >= 3, f"{path.name}: too few odometry messages")
    require(len(clouds) >= CANDIDATES, f"{path.name}: too few point clouds")
    odom = np.asarray(odometry, dtype=np.float64)
    order = np.argsort(odom[:, 0], kind="stable")
    odom = odom[order]
    odom[:, 3] = np.unwrap(odom[:, 3])
    origin = min(int(odom[0, 0]), int(clouds[0][0]))
    odom[:, 0] = (odom[:, 0] - origin) * 1.0e-9
    cloud_times = np.asarray(
        [(timestamp - origin) * 1.0e-9 for timestamp, _ in clouds],
        dtype=np.float64,
    )
    return {
        "path": path,
        "sha256": sha256_file(path),
        "odom": odom,
        "cloud_times": cloud_times,
        "clouds": [points for _, points in clouds],
        "rotation": rotation,
        "translation": translation,
    }


def interpolate_pose(odom: np.ndarray, times: np.ndarray) -> np.ndarray:
    require(times.min() >= odom[0, 0], "query before odometry")
    require(times.max() <= odom[-1, 0], "query after odometry")
    return np.column_stack(
        [np.interp(times, odom[:, 0], odom[:, column]) for column in (1, 2, 3)]
    )


def local_future_paths(raw: dict[str, Any]) -> dict[str, Any]:
    odom = raw["odom"]
    valid_indices = np.flatnonzero(
        (raw["cloud_times"] >= odom[0, 0])
        & (raw["cloud_times"] + HORIZON_SECONDS[-1] <= odom[-1, 0])
    )
    require(len(valid_indices) >= CANDIDATES, "too few valid LiDAR windows")
    paths = []
    for index in valid_indices:
        current_time = raw["cloud_times"][index]
        current = interpolate_pose(odom, np.asarray((current_time,)))[0]
        future = interpolate_pose(odom, current_time + HORIZON_SECONDS)
        delta = future[:, :2] - current[None, :2]
        cosine = math.cos(float(current[2]))
        sine = math.sin(float(current[2]))
        paths.append(
            np.column_stack(
                (
                    cosine * delta[:, 0] + sine * delta[:, 1],
                    -sine * delta[:, 0] + cosine * delta[:, 1],
                    (future[:, 2] - current[2] + np.pi) % (2 * np.pi) - np.pi,
                )
            )
        )
    return {
        **raw,
        "valid_cloud_indices": valid_indices,
        "paths": np.asarray(paths, dtype=np.float32),
    }


def candidate_indices(index: int, count: int) -> np.ndarray:
    values = [index]
    step = 1
    while len(values) < CANDIDATES:
        candidate = (index + step * NEGATIVE_STRIDE) % count
        if candidate not in values:
            values.append(candidate)
        step += 1
    return np.asarray(values, dtype=np.int64)


def sample_path(path: np.ndarray) -> np.ndarray:
    waypoints = np.concatenate(
        (np.zeros((1, 2), dtype=np.float32), path[:, :2]),
        axis=0,
    )
    rows = []
    traveled = 0.0
    for start, end in zip(waypoints[:-1], waypoints[1:]):
        length = float(np.linalg.vector_norm(end - start))
        samples = max(1, int(math.ceil(length / PATH_SAMPLE_METERS)))
        for fraction in np.linspace(0.0, 1.0, samples + 1, endpoint=True)[1:]:
            point = start + float(fraction) * (end - start)
            traveled += length / samples
            if traveled >= 0.45:
                rows.append(point)
    if not rows:
        rows.append(waypoints[-1])
    return np.asarray(rows, dtype=np.float32)


def clearance_score(points: np.ndarray, path: np.ndarray) -> float:
    if len(points) == 0:
        return 0.0
    samples = sample_path(path)
    squared = np.square(samples[:, None] - points[None]).sum(axis=-1)
    clearance = np.sqrt(np.maximum(squared.min(axis=1), 0.0))
    proximity = np.exp(-clearance / ROBOT_RADIUS_METERS)
    penetration = np.maximum(ROBOT_RADIUS_METERS - clearance, 0.0)
    return float(
        proximity.mean()
        + proximity.max()
        + 4.0 * np.square(penetration).mean()
        + 8.0 * np.square(penetration).max()
    )


def evaluate(raw: dict[str, Any]) -> dict[str, Any]:
    paths = raw["paths"]
    score_rows = []
    cloud_point_counts = []
    for index, cloud_index in enumerate(raw["valid_cloud_indices"]):
        points = raw["clouds"][int(cloud_index)]
        sources = candidate_indices(index, len(paths))
        score_rows.append(
            [clearance_score(points, paths[source]) for source in sources]
        )
        cloud_point_counts.append(len(points))
    scores = np.asarray(score_rows, dtype=np.float64)
    true = scores[:, :1]
    negative = scores[:, 1:]
    wins = true < negative
    ties = true == negative
    rank = 1.0 + wins.shape[1] - wins.sum(axis=1) - 0.5 * ties.sum(axis=1)
    random_top1 = 1.0 / CANDIDATES
    return {
        "windows": len(scores),
        "candidate_count_including_true": CANDIDATES,
        "random_top1": random_top1,
        "retrieval_top1": float(np.mean(rank == 1.0)),
        "pairwise_action_win_rate": float(np.mean(wins + 0.5 * ties)),
        "mean_rank": float(rank.mean()),
        "mean_reciprocal_rank": float(np.mean(1.0 / rank)),
        "true_score_mean": float(true.mean()),
        "negative_score_mean": float(negative.mean()),
        "positive_clearance_gap": float(negative.mean() - true.mean()),
        "point_count": {
            "mean": float(np.mean(cloud_point_counts)),
            "minimum": int(np.min(cloud_point_counts)),
            "maximum": int(np.max(cloud_point_counts)),
        },
    }


def main() -> None:
    args = parse_args()
    require(len(args.train_bag) == 2, "exactly two existing train bags required")
    started = time.perf_counter()
    train = [local_future_paths(read_bag(path)) for path in args.train_bag]
    held = local_future_paths(read_bag(args.held_bag))
    train_metrics = [evaluate(route) for route in train]
    held_metrics = evaluate(held)
    go = (
        held_metrics["retrieval_top1"] >= 2.0 * held_metrics["random_top1"]
        and held_metrics["pairwise_action_win_rate"] >= 0.65
        and all(metric["pairwise_action_win_rate"] >= 0.55 for metric in train_metrics)
    )
    result = {
        "schema": "scand_lidar_path_observability_result_v1",
        "status": "complete_existing_bag_lidar_probe",
        "decision": (
            "go_build_robot_frame_occupancy_latent"
            if go
            else "no_go_lidar_clearance_not_path_identifiable"
        ),
        "go": go,
        "protocol": {
            "train_bags": [str(path) for path in args.train_bag],
            "held_bag": str(args.held_bag),
            "current_point_cloud_fixed_across_candidates": True,
            "negative_paths_are_observed_h4_trajectories": True,
            "horizon_seconds": HORIZON_SECONDS.tolist(),
            "candidate_count_including_true": CANDIDATES,
            "negative_stride": NEGATIVE_STRIDE,
            "voxel_meters": VOXEL_METERS,
            "robot_radius_meters": ROBOT_RADIUS_METERS,
            "path_sample_meters": PATH_SAMPLE_METERS,
            "obstacle_z_range_meters": list(OBSTACLE_Z_RANGE_METERS),
            "maximum_range_meters": MAX_RANGE_METERS,
            "parameters_selected_without_held_route": True,
            "ate_or_objective_loaded": False,
            "sealed_validation_opened": False,
            "formal_v70_modified": False,
            "frozen_highres_ltwm_modified": False,
        },
        "train": [
            {
                "bag": str(route["path"]),
                "bag_sha256": route["sha256"],
                "velodyne_translation_in_base": route["translation"].tolist(),
                "velodyne_rotation_in_base": route["rotation"].tolist(),
                "metrics": metrics,
            }
            for route, metrics in zip(train, train_metrics)
        ],
        "held": {
            "bag": str(held["path"]),
            "bag_sha256": held["sha256"],
            "velodyne_translation_in_base": held["translation"].tolist(),
            "velodyne_rotation_in_base": held["rotation"].tolist(),
            "metrics": held_metrics,
        },
        "protection": {
            "robot_executor_used": False,
            "outcome_labels_loaded": False,
            "ate_or_objective_gradients": False,
            "sealed_validation_opened": False,
            "formal_v70_unchanged": True,
            "frozen_highres_ltwm_unchanged": True,
        },
        "runner": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "runtime_seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    require(not args.output.exists(), f"refusing to overwrite: {args.output}")
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
