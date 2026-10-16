#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data


@dataclass
class PoseSample:
    stamp: float
    x: float
    y: float
    yaw: float


def quaternion_to_yaw(x: float, y: float, z: float, w: float) -> float:
    sin_yaw = 2.0 * (w * z + x * y)
    cos_yaw = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(sin_yaw, cos_yaw)


def wrap_angle(value: np.ndarray) -> np.ndarray:
    return (value + np.pi) % (2.0 * np.pi) - np.pi


class PoseHistoryReader(Node):
    def __init__(self, topic: str) -> None:
        super().__init__("v70_pose_history_shadow")
        self.latest: PoseSample | None = None
        self.latest_seq = 0
        self.create_subscription(
            Odometry,
            topic,
            self.callback,
            qos_profile_sensor_data,
        )

    def callback(self, msg: Odometry) -> None:
        position = msg.pose.pose.position
        orientation = msg.pose.pose.orientation

        self.latest = PoseSample(
            # 某些 Unitree topic 会复用 header.stamp，因此使用本机接收时间。
            stamp=time.monotonic(),
            x=float(position.x),
            y=float(position.y),
            yaw=quaternion_to_yaw(
                float(orientation.x),
                float(orientation.y),
                float(orientation.z),
                float(orientation.w),
            ),
        )
        self.latest_seq += 1


def build_history_state(samples: list[PoseSample]) -> tuple[np.ndarray, dict]:
    if len(samples) != 4:
        raise ValueError(f"expected 4 pose samples, got {len(samples)}")

    positions = np.asarray(
        [[sample.x, sample.y] for sample in samples],
        dtype=np.float32,
    )
    yaw = np.asarray([sample.yaw for sample in samples], dtype=np.float32)

    # 与 build_recon100_k17_source_v1.motion_parts() 完全一致。
    global_velocity = np.diff(positions, axis=0)

    cosine = math.cos(float(yaw[-1]))
    sine = math.sin(float(yaw[-1]))
    rotation = np.asarray(
        [[cosine, sine], [-sine, cosine]],
        dtype=np.float32,
    )

    local_velocity = global_velocity @ rotation.T
    speed = np.linalg.norm(local_velocity, axis=1)
    yaw_delta = wrap_angle(np.diff(yaw)).astype(np.float32)

    weights = np.asarray([0.2, 0.3, 0.5], dtype=np.float32)
    velocity_weighted = (local_velocity * weights[:, None]).sum(axis=0)
    speed_weighted = float((speed * weights).sum())
    yaw_rate = float((yaw_delta * weights).sum())

    state = np.concatenate(
        (
            local_velocity.reshape(-1),                  # 6
            speed,                                       # 3
            yaw_delta,                                   # 3
            np.asarray(
                [
                    speed.mean(),
                    speed[-1],
                    speed_weighted,
                    speed[-1] - speed[-2],
                    yaw_delta.mean(),
                    yaw_rate,
                ],
                dtype=np.float32,
            ),                                           # 6
        )
    ).astype(np.float32)

    if state.shape != (18,):
        raise RuntimeError(f"history state shape changed: {state.shape}")

    diagnostics = {
        "positions": positions,
        "yaw": yaw,
        "global_velocity": global_velocity,
        "local_velocity": local_velocity,
        "speed": speed,
        "yaw_delta": yaw_delta,
        "velocity_weighted": velocity_weighted,
        "speed_weighted": speed_weighted,
        "yaw_rate": yaw_rate,
    }
    return state, diagnostics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", default="/utlidar/robot_odom")
    parser.add_argument("--interval", type=float, default=0.20)
    parser.add_argument("--timeout", type=float, default=10.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    rclpy.init()
    node = PoseHistoryReader(args.topic)

    samples: list[PoseSample] = []
    last_seq = -1
    next_capture = time.monotonic()
    deadline = time.monotonic() + args.timeout

    print(f"Reading {args.topic}")
    print(f"Collecting 4 samples at {args.interval:.3f} s spacing...")

    try:
        while len(samples) < 4:
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"did not receive 4 fresh pose samples within {args.timeout}s"
                )

            rclpy.spin_once(node, timeout_sec=0.05)
            latest = node.latest

            if latest is None or time.monotonic() < next_capture:
                continue

            # 必须至少收到一次新的回调，但不依赖 ROS header.stamp。
            if node.latest_seq == last_seq:
                continue

            samples.append(latest)
            last_seq = node.latest_seq
            next_capture = time.monotonic() + args.interval

            print(
                f"sample {len(samples) - 1}: "
                f"stamp={latest.stamp:.6f}, "
                f"x={latest.x:+.6f}, "
                f"y={latest.y:+.6f}, "
                f"yaw={latest.yaw:+.6f}"
            )

        state, diag = build_history_state(samples)

        timestamps = np.asarray([sample.stamp for sample in samples])
        print()
        print("Pose history shadow: PASS")
        print("timestamp deltas:", np.diff(timestamps))
        print("local velocity:")
        print(np.array2string(diag["local_velocity"], precision=6))
        print("yaw delta:")
        print(np.array2string(diag["yaw_delta"], precision=6))
        print("history state shape:", state.shape)
        print("history state:")
        print(np.array2string(state, precision=6, suppress_small=True))

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
