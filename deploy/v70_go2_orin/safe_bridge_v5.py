#!/usr/bin/env python3
"""Ground-aware, fail-closed ROS velocity bridge for Go2 V70.

The bridge publishes only zero velocity by default. Non-zero output requires
an explicit arm file, no stop file, fresh command/odometry/cloud data, a valid
ground estimate, and clearance for the requested motion. Forward and yaw
clearance are gated independently so a floor return cannot masquerade as an
obstacle and a blocked forward corridor does not automatically forbid a safe
in-place turn.
"""

from __future__ import annotations

import math
import signal
import time
from pathlib import Path
from typing import Optional

import rclpy
from rclpy._rclpy_pybind11 import RCLError
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2

from gait_command_shaping import (
    forward_gait_entry_for_command,
    shape_forward_gait_entry,
)
from safe_bridge_core import (
    ClearanceRecoveryLatch,
    evaluate_ground_aware_clearance,
    motion_clearance_allows,
    shape_forward_steering_for_clearance,
)
from v70_sensor_freshness import SourceStampGuard, ros_stamp_ns


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


class SafeBridgeV5(Node):
    def __init__(self) -> None:
        super().__init__("go2_safe_control_bridge_v5")

        self.declare_parameter("dry_run", True)
        self.declare_parameter("input_topic", "/nav/cmd_vel_raw")
        self.declare_parameter("output_topic", "/nav/cmd_vel_safe")
        self.declare_parameter("odom_topic", "/utlidar/robot_odom")
        self.declare_parameter("cloud_topic", "/utlidar/cloud_base")
        self.declare_parameter("stop_file", "/tmp/go2_nav_stop")
        self.declare_parameter("arm_file", "/run/go2-safety/ARMED")

        self.declare_parameter("max_vx", 0.05)
        self.declare_parameter("max_wz", 0.10)
        self.declare_parameter("max_ax", 0.10)
        self.declare_parameter("max_awz", 0.20)
        # Zero preserves the shared/baseline behavior.  The main-method
        # wrapper explicitly enables a calibrated walking-band entry so a
        # forward command is not decomposed into body-lean micro-commands.
        self.declare_parameter("min_forward_gait_entry_vx", 0.0)
        self.declare_parameter("command_epsilon", 1e-4)
        self.declare_parameter("forward_steering_wz_threshold", 0.08)

        self.declare_parameter("cmd_timeout", 0.25)
        self.declare_parameter("odom_timeout", 0.30)
        self.declare_parameter("cloud_timeout", 0.50)
        self.declare_parameter("clear_recovery_s", 1.50)
        self.declare_parameter("min_clear_observations", 5)
        self.declare_parameter("min_forward_clearance_m", 0.65)
        self.declare_parameter("corridor_half_width_m", 0.32)
        self.declare_parameter("rotation_clearance_radius_m", 0.48)
        self.declare_parameter(
            "min_height_above_ground_m",
            0.08,
        )
        self.declare_parameter(
            "max_height_above_ground_m",
            0.60,
        )
        self.declare_parameter("min_finite_cloud_points", 50)
        self.declare_parameter("ground_x_min_m", 0.30)
        self.declare_parameter("ground_x_max_m", 1.50)
        self.declare_parameter("ground_half_width_m", 0.45)
        self.declare_parameter("ground_z_min_m", -0.60)
        self.declare_parameter("ground_z_max_m", 0.10)
        self.declare_parameter("min_ground_points", 30)
        self.declare_parameter("ground_inlier_tolerance_m", 0.05)
        self.declare_parameter("self_front_foot_x_min_m", 0.05)
        self.declare_parameter("self_front_foot_x_max_m", 0.38)
        self.declare_parameter("self_front_foot_abs_y_min_m", 0.18)
        self.declare_parameter("self_front_foot_abs_y_max_m", 0.38)
        self.declare_parameter("self_rear_foot_x_min_m", -0.42)
        self.declare_parameter("self_rear_foot_x_max_m", -0.05)
        self.declare_parameter("self_rear_foot_abs_y_min_m", 0.34)
        self.declare_parameter("self_rear_foot_abs_y_max_m", 0.46)
        self.declare_parameter(
            "self_rear_foot_max_height_above_ground_m",
            0.24,
        )

        self.dry_run = bool(self.get_parameter("dry_run").value)
        input_topic = str(self.get_parameter("input_topic").value)
        output_topic = str(self.get_parameter("output_topic").value)
        odom_topic = str(self.get_parameter("odom_topic").value)
        cloud_topic = str(self.get_parameter("cloud_topic").value)
        self.stop_file = Path(
            str(self.get_parameter("stop_file").value)
        )
        self.arm_file = Path(
            str(self.get_parameter("arm_file").value)
        )

        scalar_parameters = (
            "max_vx",
            "max_wz",
            "max_ax",
            "max_awz",
            "min_forward_gait_entry_vx",
            "command_epsilon",
            "forward_steering_wz_threshold",
            "cmd_timeout",
            "odom_timeout",
            "cloud_timeout",
            "clear_recovery_s",
            "min_forward_clearance_m",
            "corridor_half_width_m",
            "rotation_clearance_radius_m",
            "min_height_above_ground_m",
            "max_height_above_ground_m",
            "ground_x_min_m",
            "ground_x_max_m",
            "ground_half_width_m",
            "ground_z_min_m",
            "ground_z_max_m",
            "ground_inlier_tolerance_m",
            "self_front_foot_x_min_m",
            "self_front_foot_x_max_m",
            "self_front_foot_abs_y_min_m",
            "self_front_foot_abs_y_max_m",
            "self_rear_foot_x_min_m",
            "self_rear_foot_x_max_m",
            "self_rear_foot_abs_y_min_m",
            "self_rear_foot_abs_y_max_m",
            "self_rear_foot_max_height_above_ground_m",
        )
        for name in scalar_parameters:
            setattr(
                self,
                name,
                float(self.get_parameter(name).value),
            )
        if not 0.0 <= self.min_forward_gait_entry_vx <= self.max_vx:
            raise ValueError(
                "min_forward_gait_entry_vx must be in [0, max_vx]"
            )
        self.min_finite_cloud_points = int(
            self.get_parameter("min_finite_cloud_points").value
        )
        self.min_ground_points = int(
            self.get_parameter("min_ground_points").value
        )
        self.min_clear_observations = int(
            self.get_parameter("min_clear_observations").value
        )
        self.forward_clear_latch = ClearanceRecoveryLatch(
            recovery_s=self.clear_recovery_s,
            min_clear_observations=self.min_clear_observations,
        )
        self.rotation_clear_latch = ClearanceRecoveryLatch(
            recovery_s=self.clear_recovery_s,
            min_clear_observations=self.min_clear_observations,
        )

        self.last_cmd_time: Optional[float] = None
        self.last_odom_time: Optional[float] = None
        self.last_cloud_time: Optional[float] = None
        self.odom_source = SourceStampGuard()
        self.cloud_source = SourceStampGuard()
        self.cloud_valid = False
        self.ground_valid = False
        self.observed_forward_clear = False
        self.observed_rotation_clear = False
        self.forward_clear = False
        self.rotation_clear = False
        self.ground_height_m = math.nan
        self.minimum_forward_distance_m = math.inf
        self.minimum_rotation_distance_m = math.inf
        self.self_filtered_points = 0
        self.front_self_filtered_points = 0
        self.rear_self_filtered_points = 0
        self.obstacle_points = 0

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
            odom_topic,
            self.odom_callback,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            PointCloud2,
            cloud_topic,
            self.cloud_callback,
            qos_profile_sensor_data,
        )

        self.timer_period = 0.05
        self.create_timer(self.timer_period, self.control_timer)
        self.last_status_log = 0.0
        self.get_logger().warn(
            "SAFE BRIDGE V5 STARTED: dry_run=%s stop_file=%s "
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

    def odom_callback(self, msg: Odometry) -> None:
        now = time.monotonic()
        if self.odom_source.observe(ros_stamp_ns(msg), now):
            self.last_odom_time = now

    def cloud_callback(self, msg: PointCloud2) -> None:
        receive_time = time.monotonic()
        if not self.cloud_source.observe(
            ros_stamp_ns(msg),
            receive_time,
        ):
            return
        try:
            points = point_cloud2.read_points(
                msg,
                field_names=("x", "y", "z"),
                skip_nans=False,
            )
            result = evaluate_ground_aware_clearance(
                points,
                min_forward_clearance_m=(
                    self.min_forward_clearance_m
                ),
                corridor_half_width_m=self.corridor_half_width_m,
                rotation_clearance_radius_m=(
                    self.rotation_clearance_radius_m
                ),
                min_height_above_ground_m=(
                    self.min_height_above_ground_m
                ),
                max_height_above_ground_m=(
                    self.max_height_above_ground_m
                ),
                min_finite_points=(
                    self.min_finite_cloud_points
                ),
                ground_x_min_m=self.ground_x_min_m,
                ground_x_max_m=self.ground_x_max_m,
                ground_half_width_m=self.ground_half_width_m,
                ground_z_min_m=self.ground_z_min_m,
                ground_z_max_m=self.ground_z_max_m,
                min_ground_points=self.min_ground_points,
                ground_inlier_tolerance_m=(
                    self.ground_inlier_tolerance_m
                ),
                self_foot_x_min_m=self.self_front_foot_x_min_m,
                self_foot_x_max_m=self.self_front_foot_x_max_m,
                self_foot_abs_y_min_m=(
                    self.self_front_foot_abs_y_min_m
                ),
                self_foot_abs_y_max_m=(
                    self.self_front_foot_abs_y_max_m
                ),
                self_rear_foot_x_min_m=(
                    self.self_rear_foot_x_min_m
                ),
                self_rear_foot_x_max_m=(
                    self.self_rear_foot_x_max_m
                ),
                self_rear_foot_abs_y_min_m=(
                    self.self_rear_foot_abs_y_min_m
                ),
                self_rear_foot_abs_y_max_m=(
                    self.self_rear_foot_abs_y_max_m
                ),
                self_rear_foot_max_height_above_ground_m=(
                    self.self_rear_foot_max_height_above_ground_m
                ),
            )
        except Exception as exc:
            self.cloud_valid = False
            self.ground_valid = False
            self.observed_forward_clear = False
            self.observed_rotation_clear = False
            self.forward_clear_latch.observe(False, receive_time)
            self.rotation_clear_latch.observe(False, receive_time)
            self.forward_clear = False
            self.rotation_clear = False
            self.last_cloud_time = None
            self.get_logger().error(
                f"Point cloud parsing failed closed: {exc}"
            )
            return

        self.cloud_valid = bool(result["cloud_valid"])
        self.ground_valid = bool(result["ground_valid"])
        self.observed_forward_clear = bool(result["forward_clear"])
        self.observed_rotation_clear = bool(result["rotation_clear"])
        self.forward_clear = self.forward_clear_latch.observe(
            self.observed_forward_clear,
            receive_time,
        )
        self.rotation_clear = self.rotation_clear_latch.observe(
            self.observed_rotation_clear,
            receive_time,
        )
        self.ground_height_m = float(result["ground_height_m"])
        self.minimum_forward_distance_m = float(
            result["minimum_forward_distance_m"]
        )
        self.minimum_rotation_distance_m = float(
            result["minimum_rotation_distance_m"]
        )
        self.self_filtered_points = int(
            result["self_filtered_points"]
        )
        self.front_self_filtered_points = int(
            result["front_self_filtered_points"]
        )
        self.rear_self_filtered_points = int(
            result["rear_self_filtered_points"]
        )
        self.obstacle_points = int(result["obstacle_points"])
        self.last_cloud_time = receive_time

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

        requested_values = (
            float(self.requested.linear.x),
            float(self.requested.linear.y),
            float(self.requested.linear.z),
            float(self.requested.angular.x),
            float(self.requested.angular.y),
            float(self.requested.angular.z),
        )
        explicit_zero_requested = all(
            abs(value) <= self.command_epsilon
            for value in requested_values
        )
        target_vx = clamp(
            requested_values[0],
            0.0,
            self.max_vx,
        )
        target_wz = clamp(
            requested_values[5],
            -self.max_wz,
            self.max_wz,
        )
        raw_target_wz = target_wz
        target_vx, target_wz, steering_degraded = (
            shape_forward_steering_for_clearance(
                target_vx=target_vx,
                target_wz=target_wz,
                rotation_clear=self.rotation_clear,
                forward_steering_wz_threshold=(
                    self.forward_steering_wz_threshold
                ),
                command_epsilon=self.command_epsilon,
            )
        )
        motion_requested = (
            target_vx > self.command_epsilon
            or abs(target_wz) > self.command_epsilon
        )
        clearance_ok = motion_clearance_allows(
            target_vx=target_vx,
            target_wz=target_wz,
            forward_clear=self.forward_clear,
            rotation_clear=self.rotation_clear,
            command_epsilon=self.command_epsilon,
        )
        base_enabled = (
            not self.dry_run
            and stop_absent
            and armed
            and cmd_ok
            and odom_ok
            and cloud_fresh
            and self.cloud_valid
            and self.ground_valid
        )
        enabled = False
        if not base_enabled:
            self.publish_immediate_zero()
        elif explicit_zero_requested:
            # Planned neutral is executed downstream as Move(0), without a
            # Sport StopMove transition. Publish it in this cycle so the Go2
            # does not spend several control ticks below its stepping band,
            # loading the rear legs and recoiling behind the endpoint.
            self.publish_immediate_zero()
        elif steering_degraded or not motion_requested or not clearance_ok:
            # Non-zero intent that cannot pass its corresponding clearance
            # gate is a safety stop, never a planned deceleration.
            self.publish_immediate_zero()
        else:
            dt = self.timer_period
            gait_entry_vx = forward_gait_entry_for_command(
                configured_entry_vx=float(
                    self.min_forward_gait_entry_vx
                ),
                target_wz=float(target_wz),
                max_abs_wz=float(
                    self.forward_steering_wz_threshold
                ),
            )
            next_vx = shape_forward_gait_entry(
                current_vx=float(self.output.linear.x),
                target_vx=float(target_vx),
                max_change=float(self.max_ax * dt),
                minimum_entry_vx=gait_entry_vx,
                epsilon=float(self.command_epsilon),
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
            enabled = True

        if now - self.last_status_log >= 1.0:
            self.last_status_log = now
            self.get_logger().info(
                "enabled=%s dry_run=%s stop_absent=%s armed=%s "
                "cmd=%s odom=%s cloud_fresh=%s cloud_valid=%s "
                "ground_valid=%s ground_z=%.3f forward_clear=%s "
                "rotation_clear=%s steering_degraded=%s raw_wz=%.3f "
                "min_forward=%.3f min_rotation=%.3f "
                "observed_clear=(%s,%s) clear_streak=(%d,%d) "
                "obstacles=%d self_filtered=%d(front=%d rear=%d) "
                "odom_stamp=(ok=%d dup=%d reg=%d invalid=%d) "
                "cloud_stamp=(ok=%d dup=%d reg=%d invalid=%d) "
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
                    self.ground_valid,
                    self.ground_height_m,
                    self.forward_clear,
                    self.rotation_clear,
                    steering_degraded,
                    raw_target_wz,
                    self.minimum_forward_distance_m,
                    self.minimum_rotation_distance_m,
                    self.observed_forward_clear,
                    self.observed_rotation_clear,
                    self.forward_clear_latch.clear_observations,
                    self.rotation_clear_latch.clear_observations,
                    self.obstacle_points,
                    self.self_filtered_points,
                    self.front_self_filtered_points,
                    self.rear_self_filtered_points,
                    self.odom_source.accepted_count,
                    self.odom_source.duplicate_count,
                    self.odom_source.regression_count,
                    self.odom_source.invalid_count,
                    self.cloud_source.accepted_count,
                    self.cloud_source.duplicate_count,
                    self.cloud_source.regression_count,
                    self.cloud_source.invalid_count,
                    self.output.linear.x,
                    self.output.linear.y,
                    self.output.angular.z,
                )
            )


def signal_interrupt(_signum, _frame) -> None:
    """Interrupt rclpy.spin without invalidating its context first."""
    raise KeyboardInterrupt


def main(args=None) -> None:
    rclpy.init(
        args=args,
        signal_handler_options=SignalHandlerOptions.NO,
    )
    previous_sigint = signal.signal(
        signal.SIGINT,
        signal_interrupt,
    )
    previous_sigterm = signal.signal(
        signal.SIGTERM,
        signal_interrupt,
    )
    node = None
    try:
        node = SafeBridgeV5()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except RCLError:
        # ROS Humble can surface an invalid-context RCLError instead of
        # ExternalShutdownException when SIGTERM races with wait-set setup.
        # Suppress only that shutdown case; an RCLError while the context is
        # still healthy remains a real fault.
        if rclpy.ok():
            raise
    finally:
        # Do not let a repeated termination signal interrupt the final zero
        # publication and ROS resource cleanup.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        if node is not None and rclpy.ok():
            node.publish_immediate_zero()
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        signal.signal(signal.SIGINT, previous_sigint)
        signal.signal(signal.SIGTERM, previous_sigterm)


if __name__ == "__main__":
    main()
