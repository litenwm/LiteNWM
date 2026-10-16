#!/usr/bin/env python3
"""Read one lidar cloud and explain forward-corridor blocking points."""

from __future__ import annotations

import argparse
import math
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2

from safe_bridge_core import evaluate_ground_aware_clearance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", default="/utlidar/cloud_base")
    parser.add_argument("--half-width", type=float, default=0.32)
    parser.add_argument("--min-height", type=float, default=-0.25)
    parser.add_argument("--max-height", type=float, default=0.60)
    parser.add_argument("--clearance", type=float, default=0.65)
    parser.add_argument("--nearest", type=int, default=20)
    parser.add_argument("--timeout", type=float, default=5.0)
    return parser.parse_args()


class OneCloud(Node):
    def __init__(self, topic: str) -> None:
        super().__init__("probe_lidar_corridor_shadow")
        self.message: PointCloud2 | None = None
        self.create_subscription(
            PointCloud2,
            topic,
            self.callback,
            qos_profile_sensor_data,
        )

    def callback(self, message: PointCloud2) -> None:
        if self.message is None:
            self.message = message


def main() -> None:
    args = parse_args()
    print("READ-ONLY lidar corridor probe; no publisher is created.")

    rclpy.init()
    node = OneCloud(args.topic)
    deadline = time.monotonic() + args.timeout
    try:
        while node.message is None and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.05)
        if node.message is None:
            raise TimeoutError(f"no cloud received from {args.topic}")
        message = node.message
    finally:
        node.destroy_node()
        rclpy.shutdown()

    finite: list[tuple[float, float, float, float]] = []
    xyz_points: list[tuple[float, float, float]] = []
    corridor: list[tuple[float, float, float, float]] = []
    for raw in point_cloud2.read_points(
        message,
        field_names=("x", "y", "z"),
        skip_nans=False,
    ):
        x, y, z = (float(value) for value in raw)
        if not all(math.isfinite(value) for value in (x, y, z)):
            continue
        distance = math.hypot(x, y)
        record = (distance, x, y, z)
        finite.append(record)
        xyz_points.append((x, y, z))
        if (
            x >= 0.0
            and abs(y) <= args.half_width
            and args.min_height <= z <= args.max_height
        ):
            corridor.append(record)

    corridor.sort()
    blocked = [
        record
        for record in corridor
        if record[0] < args.clearance
    ]

    ground_aware = evaluate_ground_aware_clearance(
        xyz_points,
        min_forward_clearance_m=args.clearance,
        corridor_half_width_m=args.half_width,
        rotation_clearance_radius_m=0.48,
        min_height_above_ground_m=0.08,
        max_height_above_ground_m=0.60,
        min_finite_points=50,
        min_ground_points=30,
    )

    obstacle_band = []
    ground_z = float(ground_aware["ground_height_m"])
    if math.isfinite(ground_z):
        for x, y, z in xyz_points:
            height = z - ground_z
            if not 0.08 <= height <= 0.60:
                continue
            front_self = (
                0.05 <= x <= 0.38
                and 0.18 <= abs(y) <= 0.38
            )
            rear_self = (
                -0.42 <= x <= -0.05
                and 0.34 <= abs(y) <= 0.46
                and height <= 0.24
            )
            if front_self or rear_self:
                continue
            obstacle_band.append((math.hypot(x, y), x, y, z, height))
    obstacle_band.sort()

    print("frame_id:", message.header.frame_id)
    print("finite_points:", len(finite))
    print("corridor_points:", len(corridor))
    print("points_inside_clearance:", len(blocked))
    print(
        "nearest_distance_m:",
        corridor[0][0] if corridor else float("inf"),
    )

    print("ground-aware V5 result:")
    for name in (
        "ground_valid",
        "ground_height_m",
        "ground_candidate_points",
        "ground_inlier_points",
        "obstacle_points",
        "self_filtered_points",
        "forward_clear",
        "rotation_clear",
        "minimum_forward_distance_m",
        "minimum_rotation_distance_m",
        "raw_minimum_rotation_distance_m",
        "rotation_ground_skimming_points",
        "ignored_rotation_ground_skimming_points",
    ):
        print(f"  {name}: {ground_aware[name]}")

    bands = (
        (-0.25, -0.10),
        (-0.10, 0.00),
        (0.00, 0.10),
        (0.10, 0.30),
        (0.30, 0.60),
    )
    print("inside-clearance height bands:")
    for lower, upper in bands:
        values = [
            record
            for record in blocked
            if lower <= record[3] < upper
        ]
        nearest = min(
            (record[0] for record in values),
            default=float("inf"),
        )
        print(
            f"  z=[{lower:+.2f},{upper:+.2f}): "
            f"count={len(values)} nearest={nearest:.4f}"
        )

    print("nearest ground-aware obstacle points: distance x y z height")
    for distance, x, y, z, height in obstacle_band[: max(1, args.nearest)]:
        print(
            f"  obstacle {distance:.6f} {x:+.6f} {y:+.6f} "
            f"{z:+.6f} h={height:+.6f}"
        )

    print("nearest configured-corridor points: distance x y z")
    for distance, x, y, z in corridor[: max(1, args.nearest)]:
        print(f"  {distance:.6f} {x:+.6f} {y:+.6f} {z:+.6f}")


if __name__ == "__main__":
    main()
