#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from planning.nomad_action_adapter import (  # noqa: E402
    nomad_waypoints_to_local_actions,
)
from tools.evaluation.build_recon100_k17_source_v1 import (  # noqa: E402
    action_set_features,
    trimmed_mean75,
)
from v70_latency_protocol import StageTimer, write_json_atomic  # noqa: E402

HORIZON = 4
SPACING_M = 0.25


@dataclass
class PoseSample:
    receive_time: float
    x: float
    y: float
    yaw: float


def quaternion_to_yaw(x: float, y: float, z: float, w: float) -> float:
    sin_yaw = 2.0 * (w * z + x * y)
    cos_yaw = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(sin_yaw, cos_yaw)


def wrap_angle(value: np.ndarray) -> np.ndarray:
    return (value + np.pi) % (2.0 * np.pi) - np.pi


class OdomReader(Node):
    def __init__(self, topic: str) -> None:
        super().__init__("v70_online_feature_shadow")

        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.latest: PoseSample | None = None
        self.latest_seq = 0

        self.create_subscription(
            Odometry,
            topic,
            self.callback,
            qos,
        )

    def callback(self, msg: Odometry) -> None:
        position = msg.pose.pose.position
        orientation = msg.pose.pose.orientation

        self.latest = PoseSample(
            receive_time=time.monotonic(),
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


def collect_pose_history(
    topic: str,
    interval: float,
    timeout: float,
) -> list[PoseSample]:
    node = OdomReader(topic)
    samples: list[PoseSample] = []

    deadline = time.monotonic() + timeout
    next_capture = time.monotonic()
    last_seq = -1

    try:
        while len(samples) < 4:
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"没有在 {timeout:.1f}s 内取得四个 odom 样本；"
                    f"received callbacks={node.latest_seq}"
                )

            rclpy.spin_once(node, timeout_sec=0.02)

            latest = node.latest
            if latest is None:
                continue

            if time.monotonic() < next_capture:
                continue

            if node.latest_seq == last_seq:
                continue

            samples.append(latest)
            last_seq = node.latest_seq
            next_capture = time.monotonic() + interval

            print(
                f"pose {len(samples) - 1}: "
                f"x={latest.x:+.6f}, "
                f"y={latest.y:+.6f}, "
                f"yaw={latest.yaw:+.6f}"
            )
    finally:
        node.destroy_node()

    return samples


def load_nomad_x16(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"NoMaD candidate file not found: {path}")

    if path.suffix == ".npy":
        value = np.load(path, allow_pickle=False)
    elif path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as payload:
            possible = (
                "nomad_x16_waypoint_units",
                "waypoints",
                "candidates",
                "arr_0",
            )
            key = next((name for name in possible if name in payload.files), None)
            if key is None:
                raise RuntimeError(
                    f"NPZ 中没有候选数组，可用字段：{payload.files}"
                )
            value = payload[key]
    else:
        raise ValueError(f"unsupported candidate format: {path.suffix}")

    value = np.asarray(value, dtype=np.float32)

    if value.ndim == 4 and value.shape[0] == 1:
        value = value[0]

    if value.ndim != 3 or value.shape[-1] != 2:
        raise RuntimeError(
            f"expected candidate shape [K,H,2], got {value.shape}"
        )

    if value.shape[0] < 16:
        raise RuntimeError(
            f"至少需要 16 条 NoMaD 候选，实际只有 {value.shape[0]}"
        )

    if value.shape[1] < HORIZON:
        raise RuntimeError(
            f"至少需要 H={HORIZON}，实际为 {value.shape[1]}"
        )

    # 旧 shadow 文件里的 17 条全部是 NoMaD sample，
    # 不能把第 0 条误认为正式 incumbent。
    reference = value[:16, :HORIZON, :].copy()

    if not np.isfinite(reference).all():
        raise RuntimeError("NoMaD candidates contain NaN/Inf")

    return reference


def history_motion_parts(
    candidates_m: np.ndarray,
    samples: list[PoseSample],
) -> tuple[np.ndarray, np.ndarray]:
    if candidates_m.ndim != 3 or candidates_m.shape[1:] != (HORIZON, 2):
        raise RuntimeError(
            f"expected candidates [K,{HORIZON},2], got {candidates_m.shape}"
        )

    if len(samples) != 4:
        raise RuntimeError(f"expected four poses, got {len(samples)}")

    positions = np.asarray(
        [[sample.x, sample.y] for sample in samples],
        dtype=np.float32,
    )
    yaw = np.asarray(
        [sample.yaw for sample in samples],
        dtype=np.float32,
    )

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
    velocity_weighted = (
        local_velocity * weights[:, None]
    ).sum(axis=0)
    speed_weighted = float((speed * weights).sum())
    yaw_rate = float((yaw_delta * weights).sum())

    state = np.concatenate(
        (
            local_velocity.reshape(-1),
            speed,
            yaw_delta,
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
            ),
        )
    ).astype(np.float32)

    if state.shape != (18,):
        raise RuntimeError(f"18D history state changed: {state.shape}")

    steps = np.arange(
        1,
        HORIZON + 1,
        dtype=np.float32,
    )[:, None]

    expected_last = steps * local_velocity[-1]
    expected_weighted = steps * velocity_weighted

    expected_twist = np.zeros((HORIZON, 2), dtype=np.float32)
    for future_step in range(HORIZON):
        heading = (future_step + 0.5) * yaw_rate
        increment = speed_weighted * np.asarray(
            [math.cos(heading), math.sin(heading)],
            dtype=np.float32,
        )
        if future_step:
            expected_twist[future_step] = expected_twist[future_step - 1]
        expected_twist[future_step] += increment

    candidate_steps = np.diff(
        np.concatenate(
            (
                np.zeros(
                    (len(candidates_m), 1, 2),
                    dtype=np.float32,
                ),
                candidates_m,
            ),
            axis=1,
        ),
        axis=1,
    )
    candidate_speed = np.linalg.norm(candidate_steps, axis=-1)

    def residual(expected: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        distance = np.linalg.norm(
            candidates_m - expected[None],
            axis=-1,
        )
        return distance.mean(axis=1), distance[:, -1]

    last_mean, last_final = residual(expected_last)
    weighted_mean, weighted_final = residual(expected_weighted)
    twist_mean, twist_final = residual(expected_twist)

    expected_direction = velocity_weighted / max(
        float(np.linalg.norm(velocity_weighted)),
        1.0e-6,
    )
    candidate_direction = candidates_m[:, -1] / np.maximum(
        np.linalg.norm(
            candidates_m[:, -1],
            axis=1,
            keepdims=True,
        ),
        1.0e-6,
    )

    candidate_specific = np.stack(
        (
            last_mean,
            last_final,
            weighted_mean,
            weighted_final,
            twist_mean,
            twist_final,
            np.abs(candidate_speed.mean(axis=1) - speed_weighted),
            np.abs(candidate_speed[:, 0] - speed[-1]),
            np.abs(
                np.linalg.norm(candidates_m[:, -1], axis=1)
                / HORIZON
                - speed_weighted
            ),
            1.0 - candidate_direction @ expected_direction,
        ),
        axis=-1,
    ).astype(np.float32)

    if candidate_specific.shape != (len(candidates_m), 10):
        raise RuntimeError(
            f"candidate-specific feature changed: "
            f"{candidate_specific.shape}"
        )

    return state, candidate_specific


def build_history_features(
    candidates_raw: np.ndarray,
    reference_raw: np.ndarray,
    samples: list[PoseSample],
) -> tuple[np.ndarray, np.ndarray]:
    state, candidate_specific = history_motion_parts(
        candidates_raw * SPACING_M,
        samples,
    )
    _, reference_specific = history_motion_parts(
        reference_raw * SPACING_M,
        samples,
    )

    center = np.median(reference_specific, axis=0, keepdims=True)
    scale = np.maximum(
        reference_specific.std(axis=0, keepdims=True),
        1.0e-5,
    )
    relative = np.clip(
        (candidate_specific - center) / scale,
        -8.0,
        8.0,
    )

    repeated_state = np.repeat(
        state[None],
        len(candidates_raw),
        axis=0,
    )

    result = np.concatenate(
        (
            repeated_state,
            candidate_specific,
            relative,
        ),
        axis=-1,
    ).astype(np.float32)

    if result.shape != (len(candidates_raw), 38):
        raise RuntimeError(
            f"38D history feature changed: {result.shape}"
        )

    return result, state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--nomad-candidates",
        type=Path,
        default=Path("/home/unitree/v70_nomad_live_candidates.npy"),
    )
    parser.add_argument(
        "--odom-topic",
        default="/utlidar/robot_odom",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.20,
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/home/unitree/v70_online_features_shadow.npz"),
    )
    parser.add_argument(
        "--latency-output",
        type=Path,
        help="Optional exact stage-timing JSON for the Orin benchmark.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    timer = StageTimer("online_features")
    candidate_prepare_started_ns = time.perf_counter_ns()
    print("===== V70 online feature shadow =====")
    print("No motion publisher is created.")
    print("No SportClient command is issued.")
    print()

    reference = load_nomad_x16(args.nomad_candidates)
    incumbent = trimmed_mean75(reference)
    candidates = np.concatenate(
        (incumbent[None], reference),
        axis=0,
    ).astype(np.float32)

    if candidates.shape != (17, HORIZON, 2):
        raise RuntimeError(f"formal K17 shape changed: {candidates.shape}")

    print("NoMaD x16:", reference.shape)
    print("formal K17:", candidates.shape)
    print("candidate 0: trimmed_mean75 incumbent")

    timer.add_ms(
        "candidate_pool_prepare_ms",
        (time.perf_counter_ns() - candidate_prepare_started_ns) / 1.0e6,
    )
    odom_started_ns = time.perf_counter_ns()
    rclpy.init()
    try:
        poses = collect_pose_history(
            topic=args.odom_topic,
            interval=args.interval,
            timeout=args.timeout,
        )
    finally:
        rclpy.shutdown()
    timer.add_ms(
        "odom_collect_4_samples_ms",
        (time.perf_counter_ns() - odom_started_ns) / 1.0e6,
    )
    feature_started_ns = time.perf_counter_ns()

    action_set = action_set_features(
        candidates[None],
        reference[None],
    )[0].astype(np.float32)

    if action_set.shape != (17, 83):
        raise RuntimeError(f"83D action-set changed: {action_set.shape}")

    history, history_state = build_history_features(
        candidates,
        reference,
        poses,
    )

    raw = np.concatenate(
        (action_set, history),
        axis=-1,
    ).astype(np.float32)

    if raw.shape != (17, 121):
        raise RuntimeError(f"121D raw feature changed: {raw.shape}")

    action_features = np.concatenate(
        (
            raw,
            raw - raw[:1],
        ),
        axis=-1,
    ).astype(np.float32)

    if action_features.shape != (17, 242):
        raise RuntimeError(
            f"242D action feature contract changed: "
            f"{action_features.shape}"
        )

    actions_h4 = nomad_waypoints_to_local_actions(
        candidates,
        dataset="recon",
    ).astype(np.float32)

    if actions_h4.shape != (17, HORIZON, 3):
        raise RuntimeError(f"H4 action shape changed: {actions_h4.shape}")

    pose_array = np.asarray(
        [
            [
                sample.receive_time,
                sample.x,
                sample.y,
                sample.yaw,
            ]
            for sample in poses
        ],
        dtype=np.float64,
    )
    timer.add_ms(
        "action_and_history_feature_build_ms",
        (time.perf_counter_ns() - feature_started_ns) / 1.0e6,
    )
    artifact_started_ns = time.perf_counter_ns()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        nomad_x16_waypoint_units=reference,
        candidates_waypoint_units=candidates,
        actions_h4=actions_h4,
        action_set_features=action_set,
        history_features=history,
        history_state=history_state,
        action_features=action_features,
        pose_samples=pose_array,
    )

    timer.add_ms(
        "artifact_write_ms",
        (time.perf_counter_ns() - artifact_started_ns) / 1.0e6,
    )
    timer.details.update(
        {
            "candidate_count": int(candidates.shape[0]),
            "pose_sample_count": int(len(poses)),
            "odom_topic": args.odom_topic,
            "odom_interval_s": float(args.interval),
            "compute_scope": (
                "candidate pool construction plus action/history feature "
                "construction; odometry waiting and artifact I/O are separate"
            ),
        }
    )
    if args.latency_output is not None:
        write_json_atomic(args.latency_output, timer.payload())
    print()
    print("===== RESULT =====")
    print("action_set_features:", action_set.shape)
    print("history_features:", history.shape)
    print("history_state:", history_state.shape)
    print("action_features:", action_features.shape)
    print("actions_h4:", actions_h4.shape)
    print("first action of incumbent:", actions_h4[0, 0])
    print("finite:", bool(np.isfinite(action_features).all()))
    print("saved:", args.output)

    print()
    print("V70 ONLINE FEATURE SHADOW: PASS")
    if args.latency_output is not None:
        print("latency:", args.latency_output)


if __name__ == "__main__":
    main()
