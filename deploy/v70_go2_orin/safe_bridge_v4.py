#!/usr/bin/env python3
"""Fail-closed ROS velocity bridge for zero-motion validation.

The default configuration can publish only zero velocity. Non-zero output
requires dry_run=false, a fresh command/odometry/cloud, a clear measured
forward corridor, an absent stop file, and a present explicit arm file.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Optional

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2

from safe_bridge_core import evaluate_forward_corridor


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


class SafeBridgeV4(Node):
    def __init__(self) -> None:
        super().__init__("go2_safe_control_bridge_v4")

        self.declare_parameter("dry_run", True)
        self.declare_parameter("input_topic", "/nav/cmd_vel_raw")
        self.declare_parameter("output_topic", "/nav/cmd_vel_safe")
        self.declare_parameter("stop_file", "/tmp/go2_nav_stop")
        self.declare_parameter("arm_file", "/run/go2-safety/ARMED")

        self.declare_parameter("max_vx", 0.05)
        self.declare_parameter("max_wz", 0.10)
        self.declare_parameter("max_ax", 0.10)
        self.declare_parameter("max_awz", 0.20)

        self.declare_parameter("cmd_timeout", 0.25)
        self.declare_parameter("odom_timeout", 0.30)
        self.declare_parameter("cloud_timeout", 0.50)
        self.declare_parameter("min_forward_clearance_m", 0.65)
        self.declare_parameter("corridor_half_width_m", 0.32)
        self.declare_parameter("obstacle_min_height_m", -0.25)
        self.declare_parameter("obstacle_max_height_m", 0.60)
        self.declare_parameter("min_finite_cloud_points", 50)

        self.dry_run = bool(self.get_parameter("dry_run").value)
        input_topic = str(self.get_parameter("input_topic").value)
        output_topic = str(self.get_parameter("output_topic").value)
        self.stop_file = Path(
            str(self.get_parameter("stop_file").value)
        )
        self.arm_file = Path(
            str(self.get_parameter("arm_file").value)
        )

        self.max_vx = float(self.get_parameter("max_vx").value)
        self.max_wz = float(self.get_parameter("max_wz").value)
        self.max_ax = float(self.get_parameter("max_ax").value)
        self.max_awz = float(self.get_parameter("max_awz").value)
        self.cmd_timeout = float(
            self.get_parameter("cmd_timeout").value
        )
        self.odom_timeout = float(
            self.get_parameter("odom_timeout").value
        )
        self.cloud_timeout = float(
            self.get_parameter("cloud_timeout").value
        )
        self.min_forward_clearance_m = float(
            self.get_parameter("min_forward_clearance_m").value
        )
        self.corridor_half_width_m = float(
            self.get_parameter("corridor_half_width_m").value
        )
        self.obstacle_min_height_m = float(
            self.get_parameter("obstacle_min_height_m").value
        )
        self.obstacle_max_height_m = float(
            self.get_parameter("obstacle_max_height_m").value
        )
        self.min_finite_cloud_points = int(
            self.get_parameter("min_finite_cloud_points").value
        )

        self.last_cmd_time: Optional[float] = None
        self.last_odom_time: Optional[float] = None
        self.last_cloud_time: Optional[float] = None
        self.cloud_valid = False
        self.corridor_clear = False
        self.minimum_distance_m = math.inf

        self.requested = Twist()
        self.output = Twist()
        self.publisher = self.create_publisher(
            Twist,
            output_topic,
            10,
        )
        self.create_subscription(
            Twist,
            input_topic,
            self.cmd_callback,
            10,
        )
        self.create_subscription(
            Odometry,
            "/utlidar/robot_odom",
            self.odom_callback,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            PointCloud2,
            "/utlidar/cloud_base",
            self.cloud_callback,
            qos_profile_sensor_data,
        )

        self.timer_period = 0.05
        self.create_timer(self.timer_period, self.control_timer)
        self.last_status_log = 0.0
        self.get_logger().warn(
            "SAFE BRIDGE V4 STARTED: dry_run=%s stop_file=%s "
            "arm_file=%s; defaults permit zero output only."
            % (self.dry_run, self.stop_file, self.arm_file)
        )

    def cmd_callback(self, msg: Twist) -> None:
        values = (
            msg.linear.x,
            msg.linear.y,
            msg.linear.z,
            msg.angular.x,
            msg.angular.y,
            msg.angular.z,
        )
        if not all(math.isfinite(value) for value in values):
            self.get_logger().error("Rejected non-finite command")
            self.requested = Twist()
            self.last_cmd_time = None
            return
        self.requested = msg
        self.last_cmd_time = time.monotonic()

    def odom_callback(self, _: Odometry) -> None:
        self.last_odom_time = time.monotonic()

    def cloud_callback(self, msg: PointCloud2) -> None:
        try:
            points = point_cloud2.read_points(
                msg,
                field_names=("x", "y", "z"),
                skip_nans=False,
            )
            result = evaluate_forward_corridor(
                points,
                min_forward_clearance_m=(
                    self.min_forward_clearance_m
                ),
                corridor_half_width_m=self.corridor_half_width_m,
                min_height_m=self.obstacle_min_height_m,
                max_height_m=self.obstacle_max_height_m,
                min_finite_points=self.min_finite_cloud_points,
            )
        except Exception as exc:
            self.cloud_valid = False
            self.corridor_clear = False
            self.minimum_distance_m = math.inf
            self.last_cloud_time = None
            self.get_logger().error(
                f"Point cloud parsing failed closed: {exc}"
            )
            return

        self.cloud_valid = bool(result["cloud_valid"])
        self.corridor_clear = bool(result["corridor_clear"])
        self.minimum_distance_m = float(
            result["minimum_distance_m"]
        )
        self.last_cloud_time = time.monotonic()

    @staticmethod
    def fresh(
        stamp: Optional[float],
        timeout: float,
        now: float,
    ) -> bool:
        return stamp is not None and now - stamp <= timeout

    @staticmethod
    def approach(
        current: float,
        target: float,
        max_change: float,
    ) -> float:
        return current + clamp(
            target - current,
            -max_change,
            max_change,
        )

    def publish_immediate_zero(self) -> None:
        self.output = Twist()
        self.publisher.publish(self.output)

    def control_timer(self) -> None:
        now = time.monotonic()
        cmd_ok = self.fresh(
            self.last_cmd_time,
            self.cmd_timeout,
            now,
        )
        odom_ok = self.fresh(
            self.last_odom_time,
            self.odom_timeout,
            now,
        )
        cloud_fresh = self.fresh(
            self.last_cloud_time,
            self.cloud_timeout,
            now,
        )
        stop_absent = not self.stop_file.exists()
        armed = self.arm_file.is_file()
        enabled = (
            not self.dry_run
            and stop_absent
            and armed
            and cmd_ok
            and odom_ok
            and cloud_fresh
            and self.cloud_valid
            and self.corridor_clear
        )

        if not enabled:
            self.publish_immediate_zero()
        else:
            target_vx = clamp(
                self.requested.linear.x,
                0.0,
                self.max_vx,
            )
            target_wz = clamp(
                self.requested.angular.z,
                -self.max_wz,
                self.max_wz,
            )
            dt = self.timer_period
            next_vx = self.approach(
                self.output.linear.x,
                target_vx,
                self.max_ax * dt,
            )
            next_wz = self.approach(
                self.output.angular.z,
                target_wz,
                self.max_awz * dt,
            )
            self.output = Twist()
            self.output.linear.x = next_vx
            self.output.angular.z = next_wz
            self.publisher.publish(self.output)

        if now - self.last_status_log >= 1.0:
            self.last_status_log = now
            distance = (
                f"{self.minimum_distance_m:.3f}"
                if math.isfinite(self.minimum_distance_m)
                else "inf"
            )
            self.get_logger().info(
                "enabled=%s dry_run=%s stop_absent=%s armed=%s "
                "cmd=%s odom=%s cloud_fresh=%s cloud_valid=%s "
                "corridor_clear=%s min_distance_m=%s "
                "safe=(%.3f, %.3f, %.3f)"
                % (
                    enabled,
                    self.dry_run,
                    stop_absent,
                    armed,
                    cmd_ok,
                    odom_ok,
                    cloud_fresh,
                    self.cloud_valid,
                    self.corridor_clear,
                    distance,
                    self.output.linear.x,
                    self.output.linear.y,
                    self.output.angular.z,
                )
            )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SafeBridgeV4()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if rclpy.ok():
            node.publish_immediate_zero()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
