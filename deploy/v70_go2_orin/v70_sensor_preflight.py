#!/usr/bin/env python3
"""Zero-motion standing sensor acceptance for the Go2 V70 safety chain.

This process creates subscriptions only. It never creates a velocity,
motion-enable, sport-mode, or Unitree command publisher.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import socket
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

import rclpy
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2

from safe_bridge_core import evaluate_ground_aware_clearance
from v70_sensor_freshness import SourceStampGuard, ros_stamp_ns

DEFAULT_DURATION_S = 10.0
MIN_DURATION_S = 2.5
MAX_DURATION_S = 10.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interface", default="enP8p1s0")
    parser.add_argument("--expected-ip", default="192.168.123.18")
    parser.add_argument("--odom-topic", default="/utlidar/robot_odom")
    parser.add_argument("--cloud-topic", default="/utlidar/cloud_base")
    parser.add_argument("--min-odom-hz", type=float, default=20.0)
    parser.add_argument("--min-cloud-hz", type=float, default=5.0)
    parser.add_argument(
        "--duration-s",
        type=float,
        default=DEFAULT_DURATION_S,
        help=(
            "Observation window in seconds. The default preserves the full "
            "main-method preflight; fast baselines may use the validated "
            "2.5 second minimum."
        ),
    )
    parser.add_argument(
        "--min-ground-valid-ratio",
        type=float,
        default=0.80,
    )
    parser.add_argument(
        "--posture",
        choices=("standing", "prone", "unknown"),
        default="unknown",
        help="Operator-confirmed posture; only standing can pass.",
    )
    parser.add_argument(
        "--stop-file",
        type=Path,
        default=Path("/tmp/go2_nav_stop"),
    )
    parser.add_argument(
        "--arm-file",
        type=Path,
        default=Path("/run/go2-safety/ARMED"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/v70_sensor_preflight.json"),
    )
    return parser.parse_args()


def finite_or_none(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {
            str(key): finite_or_none(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [finite_or_none(item) for item in value]
    return value


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            finite_or_none(payload),
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    temporary.replace(destination)


def measured_rate_hz(times: list[float]) -> float:
    if len(times) < 2 or times[-1] <= times[0]:
        return 0.0
    return (len(times) - 1) / (times[-1] - times[0])


def interface_ipv4(interface: str) -> tuple[bool, list[str], str | None]:
    try:
        socket.if_nametoindex(interface)
    except OSError as exc:
        return False, [], f"{type(exc).__name__}:{exc}"
    try:
        completed = subprocess.run(
            ["ip", "-j", "-4", "address", "show", "dev", interface],
            check=True,
            capture_output=True,
            text=True,
            timeout=3.0,
        )
        records = json.loads(completed.stdout)
        addresses = [
            str(info["local"])
            for record in records
            for info in record.get("addr_info", [])
            if info.get("family") == "inet" and "local" in info
        ]
        return True, addresses, None
    except Exception as exc:
        return True, [], f"{type(exc).__name__}:{exc}"


def uri_binds_interface(uri: str, interface: str) -> bool:
    pattern = rf"name\s*=\s*['\"]{re.escape(interface)}['\"]"
    return bool(re.search(pattern, uri))


class SensorProbe(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("v70_zero_motion_sensor_preflight")
        self.odom_times: list[float] = []
        self.cloud_times: list[float] = []
        self.cloud_results: list[dict[str, Any]] = []
        self.odom_source = SourceStampGuard()
        self.cloud_source = SourceStampGuard()
        self.create_subscription(
            Odometry,
            args.odom_topic,
            self.odom_callback,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            PointCloud2,
            args.cloud_topic,
            self.cloud_callback,
            qos_profile_sensor_data,
        )

    def odom_callback(self, message: Odometry) -> None:
        receive_time = time.monotonic()
        if not self.odom_source.observe(
            ros_stamp_ns(message),
            receive_time,
        ):
            return
        values = (
            message.pose.pose.position.x,
            message.pose.pose.position.y,
            message.pose.pose.position.z,
            message.pose.pose.orientation.x,
            message.pose.pose.orientation.y,
            message.pose.pose.orientation.z,
            message.pose.pose.orientation.w,
        )
        if all(math.isfinite(float(value)) for value in values):
            self.odom_times.append(receive_time)

    def cloud_callback(self, message: PointCloud2) -> None:
        receive_time = time.monotonic()
        if not self.cloud_source.observe(
            ros_stamp_ns(message),
            receive_time,
        ):
            return
        self.cloud_times.append(receive_time)
        try:
            points = point_cloud2.read_points(
                message,
                field_names=("x", "y", "z"),
                skip_nans=False,
            )
            result = evaluate_ground_aware_clearance(
                points,
                min_forward_clearance_m=0.65,
                corridor_half_width_m=0.32,
                rotation_clearance_radius_m=0.48,
                min_height_above_ground_m=0.08,
                max_height_above_ground_m=0.60,
                min_finite_points=50,
                ground_x_min_m=0.30,
                ground_x_max_m=1.50,
                ground_half_width_m=0.45,
                ground_z_min_m=-0.60,
                ground_z_max_m=0.10,
                min_ground_points=30,
                ground_inlier_tolerance_m=0.05,
            )
            self.cloud_results.append(
                {
                    "receive_time": receive_time,
                    "error": None,
                    **result,
                }
            )
        except Exception as exc:
            self.cloud_results.append(
                {
                    "receive_time": receive_time,
                    "error": f"{type(exc).__name__}:{exc}",
                    "cloud_valid": False,
                    "ground_valid": False,
                    "forward_clear": False,
                    "rotation_clear": False,
                }
            )


def build_report(
    args: argparse.Namespace,
    probe: SensorProbe,
    *,
    started_monotonic: float,
    finished_monotonic: float,
) -> dict[str, Any]:
    interface_exists, addresses, _ = interface_ipv4(
        args.interface
    )
    cyclone_uri = os.environ.get("CYCLONEDDS_URI", "")
    latest = probe.cloud_results[-1] if probe.cloud_results else None
    valid_ground = sum(
        bool(result.get("cloud_valid"))
        and bool(result.get("ground_valid"))
        for result in probe.cloud_results
    )
    ground_ratio = (
        valid_ground / len(probe.cloud_results)
        if probe.cloud_results
        else 0.0
    )
    ground_heights = [
        float(result["ground_height_m"])
        for result in probe.cloud_results
        if result.get("ground_valid")
        and isinstance(result.get("ground_height_m"), (int, float))
        and math.isfinite(float(result["ground_height_m"]))
    ]
    odom_rate = measured_rate_hz(probe.odom_times)
    cloud_rate = measured_rate_hz(probe.cloud_times)
    stop_present = args.stop_file.expanduser().is_file()
    arm_present = args.arm_file.expanduser().exists()
    interface_ready = (
        interface_exists
        and args.expected_ip in addresses
        and uri_binds_interface(cyclone_uri, args.interface)
    )
    checks = {
        "operator_declared_standing": args.posture == "standing",
        "stop_present": stop_present,
        "arm_absent": not arm_present,
        "interface_ready": interface_ready,
        "odom_rate_ok": odom_rate >= args.min_odom_hz,
        "cloud_rate_ok": cloud_rate >= args.min_cloud_hz,
        "odom_source_monotonic": (
            probe.odom_source.accepted_count >= 2
            and probe.odom_source.invalid_count == 0
            and probe.odom_source.regression_count == 0
        ),
        "cloud_source_monotonic": (
            probe.cloud_source.accepted_count >= 2
            and probe.cloud_source.invalid_count == 0
            and probe.cloud_source.regression_count == 0
        ),
        "ground_valid": (
            ground_ratio >= args.min_ground_valid_ratio
            and bool(latest and latest.get("cloud_valid"))
            and bool(latest and latest.get("ground_valid"))
        ),
        "forward_clear": bool(
            latest and latest.get("forward_clear")
        ),
        "rotation_clear": bool(
            latest and latest.get("rotation_clear")
        ),
    }
    return {
        "schema": "v70_sensor_preflight_min_v1",
        "operator_posture": args.posture,
        "timeout_s": args.duration_s,
        "elapsed_s": finished_monotonic - started_monotonic,
        "subscriptions_only": True,
        "motion_api_initialized": False,
        "publishers_created": 0,
        "stop_present": stop_present,
        "arm_present": arm_present,
        "interface": args.interface,
        "interface_ready": interface_ready,
        "odom_hz": odom_rate,
        "cloud_hz": cloud_rate,
        "odom_source": probe.odom_source.report(),
        "cloud_source": probe.cloud_source.report(),
        "ground_valid_ratio": ground_ratio,
        "ground_height_m": (
            statistics.median(ground_heights)
            if ground_heights
            else None
        ),
        "forward_clear": checks["forward_clear"],
        "rotation_clear": checks["rotation_clear"],
        "failed_checks": [
            name for name, passed in checks.items() if not passed
        ],
        "accepted": all(checks.values()),
    }


def main() -> int:
    args = parse_args()
    if args.min_odom_hz <= 0.0 or args.min_cloud_hz <= 0.0:
        raise ValueError("minimum rates must be positive")
    if not (
        math.isfinite(args.duration_s)
        and MIN_DURATION_S <= args.duration_s <= MAX_DURATION_S
    ):
        raise ValueError(
            f"--duration-s must be in [{MIN_DURATION_S}, {MAX_DURATION_S}]"
        )
    if not 0.0 < args.min_ground_valid_ratio <= 1.0:
        raise ValueError("--min-ground-valid-ratio must be in (0, 1]")

    started = time.monotonic()
    deadline = started + args.duration_s
    rclpy.init()
    probe = SensorProbe(args)
    try:
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(probe, timeout_sec=0.10)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        finished = time.monotonic()
        report = build_report(
            args,
            probe,
            started_monotonic=started,
            finished_monotonic=finished,
        )
        atomic_write_json(args.output, report)
        print(json.dumps(finite_or_none(report), indent=2))
        probe.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0 if report["accepted"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
