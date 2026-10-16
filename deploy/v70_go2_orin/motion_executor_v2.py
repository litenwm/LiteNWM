#!/usr/bin/env python3
"""Defense-in-depth Unitree Go2 velocity executor for V70.

Dry-run is the default and does not initialize SportClient.  Live non-zero
Move calls require all of: dry_run=false, stop file absent, arm file present,
fresh safe velocity, fresh true enable heartbeat, finite bounded command, and
no latched executor fault.
"""

from __future__ import annotations

import math
import signal
import threading
import time
from pathlib import Path
from typing import Any, Optional

import rclpy
from geometry_msgs.msg import Twist
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import Bool


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def neutral_hold_allowed(
    *,
    moving: bool,
    dry_run: bool,
    stop_absent: bool,
    armed: bool,
    cmd_ok: bool,
    healthy: bool,
    neutral_hold_s: float,
    last_nonzero_time: Optional[float],
    now: float,
) -> bool:
    """Allow only a short benign planner-zero hold, never a safety hold."""

    return bool(
        moving
        and not dry_run
        and stop_absent
        and armed
        and cmd_ok
        and healthy
        and neutral_hold_s > 0.0
        and last_nonzero_time is not None
        and 0.0 <= now - last_nonzero_time <= neutral_hold_s
    )


def planned_neutral_allowed(
    *,
    enabled: bool,
    moving: bool,
    dry_run: bool,
    stop_absent: bool,
    armed: bool,
    cmd_ok: bool,
    healthy: bool,
    nonzero: bool,
) -> bool:
    """Keep Sport gait active only for a fresh intentional zero command.

    This mirrors OpenTrackVLA's ordinary control loop: inference/planning
    gaps use Move(0, 0, 0), while STOP, disarm, stale input and faults still
    take the hard StopMove path.
    """

    return bool(
        enabled
        and moving
        and not dry_run
        and stop_absent
        and armed
        and cmd_ok
        and healthy
        and not nonzero
    )


class MotionExecutorV2(Node):
    def __init__(self) -> None:
        super().__init__("go2_motion_executor_v2")
        self.declare_parameter("dry_run", True)
        self.declare_parameter("interface", "enP8p1s0")
        self.declare_parameter("input_topic", "/nav/cmd_vel_safe")
        self.declare_parameter(
            "enable_topic",
            "/nav/motion_enable",
        )
        self.declare_parameter(
            "stop_file",
            "/tmp/go2_nav_stop",
        )
        self.declare_parameter(
            "arm_file",
            "/run/go2-safety/ARMED",
        )
        self.declare_parameter(
            "fault_file",
            "/tmp/v70_motion_executor_fault",
        )
        self.declare_parameter("cmd_timeout", 0.30)
        self.declare_parameter("enable_timeout", 0.30)
        self.declare_parameter("control_period", 0.05)
        self.declare_parameter("max_vx", 0.05)
        self.declare_parameter("max_wz", 0.10)
        self.declare_parameter("command_epsilon", 1e-5)
        # Optional main-method-only gait continuity across a short planned
        # zero interval. The default preserves the baseline behavior.
        self.declare_parameter("neutral_hold_s", 0.0)
        self.declare_parameter("planned_neutral_move", False)

        self.dry_run = bool(self.get_parameter("dry_run").value)
        self.interface = str(
            self.get_parameter("interface").value
        )
        input_topic = str(
            self.get_parameter("input_topic").value
        )
        enable_topic = str(
            self.get_parameter("enable_topic").value
        )
        self.stop_file = Path(
            str(self.get_parameter("stop_file").value)
        )
        self.arm_file = Path(
            str(self.get_parameter("arm_file").value)
        )
        self.fault_file = Path(
            str(self.get_parameter("fault_file").value)
        )
        self.cmd_timeout = float(
            self.get_parameter("cmd_timeout").value
        )
        self.enable_timeout = float(
            self.get_parameter("enable_timeout").value
        )
        self.control_period = float(
            self.get_parameter("control_period").value
        )
        self.max_vx = float(
            self.get_parameter("max_vx").value
        )
        self.max_wz = float(
            self.get_parameter("max_wz").value
        )
        self.command_epsilon = float(
            self.get_parameter("command_epsilon").value
        )
        self.neutral_hold_s = float(
            self.get_parameter("neutral_hold_s").value
        )
        self.planned_neutral_move = bool(
            self.get_parameter("planned_neutral_move").value
        )
        positive = (
            self.cmd_timeout,
            self.enable_timeout,
            self.control_period,
            self.max_vx,
            self.max_wz,
            self.command_epsilon,
        )
        if not all(
            math.isfinite(value) and value > 0.0
            for value in positive
        ):
            raise ValueError("executor limits/timeouts must be positive")
        if (
            not math.isfinite(self.neutral_hold_s)
            or not 0.0 <= self.neutral_hold_s <= 1.0
        ):
            raise ValueError("neutral_hold_s must be finite and in [0, 1]")

        self.lock = threading.Lock()
        self.last_cmd_time: Optional[float] = None
        # Track every Bool heartbeat separately from its requested state.  A
        # fresh False is an intentional planner neutral; absence of heartbeats
        # is still a hard timeout.
        self.last_enable_time: Optional[float] = None
        self.enable_requested = False
        self.safe_command_generation = 0
        self.minimum_active_command_generation = 1
        self.last_stop_time = 0.0
        self.last_nonzero_time: Optional[float] = None
        self.last_status_log = 0.0
        self.requested = Twist()
        self.moving = False
        self.fault_reason: str | None = None
        self.client: Any | None = None

        if self.fault_file.exists():
            raise RuntimeError(
                f"stale executor fault file exists: {self.fault_file}"
            )
        if not self.dry_run:
            self._initialize_sport_client()

        self.create_subscription(
            Twist,
            input_topic,
            self.command_callback,
            10,
        )
        self.create_subscription(
            Bool,
            enable_topic,
            self.enable_callback,
            10,
        )
        self.create_timer(
            self.control_period,
            self.control_timer,
        )
        self.get_logger().warn(
            "MOTION EXECUTOR V2 READY: dry_run=%s stop=%s arm=%s; "
            "default output is no SportClient Move call."
            % (self.dry_run, self.stop_file, self.arm_file)
        )

    def _initialize_sport_client(self) -> None:
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize
        from unitree_sdk2py.go2.sport.sport_client import (
            SportClient,
        )

        ChannelFactoryInitialize(0, self.interface)
        client = SportClient(enableLease=True)
        client.SetTimeout(3.0)
        client.Init()
        lease_code = client.WaitLeaseApplied()
        if lease_code not in (None, 0):
            raise RuntimeError(
                f"WaitLeaseApplied failed: code={lease_code}"
            )
        self.client = client
        code = self.stop_move("startup", force=True)
        if code not in (None, 0):
            raise RuntimeError(
                f"startup StopMove failed: code={code}"
            )

    @staticmethod
    def fresh(
        stamp: Optional[float],
        timeout: float,
        now: float,
    ) -> bool:
        return stamp is not None and now - stamp <= timeout

    def command_callback(self, msg: Twist) -> None:
        values = (
            msg.linear.x,
            msg.linear.y,
            msg.linear.z,
            msg.angular.x,
            msg.angular.y,
            msg.angular.z,
        )
        if not all(math.isfinite(value) for value in values):
            self.requested = Twist()
            self.last_cmd_time = None
            self._latch_fault("nonfinite_safe_command")
            return
        # V70 execution is deliberately non-holonomic.  Any lateral or
        # unsupported-axis command is treated as a fault, not ignored.
        unsupported = (
            abs(msg.linear.y) > self.command_epsilon
            or abs(msg.linear.z) > self.command_epsilon
            or abs(msg.angular.x) > self.command_epsilon
            or abs(msg.angular.y) > self.command_epsilon
        )
        if unsupported:
            self.requested = Twist()
            self.last_cmd_time = None
            self._latch_fault("unsupported_command_axis")
            return
        self.requested = msg
        self.last_cmd_time = time.monotonic()
        self.safe_command_generation += 1

    def enable_callback(self, msg: Bool) -> None:
        now = time.monotonic()
        requested = bool(msg.data)
        was_requested = self.enable_requested
        self.last_enable_time = now
        self.enable_requested = requested

        if requested:
            if not was_requested:
                # Commands received during the preceding False interval may be
                # delayed bridge output.  Require one safe Twist received after
                # this rising edge before non-zero execution can resume.
                self.minimum_active_command_generation = (
                    self.safe_command_generation + 1
                )
            return

        if not self.moving:
            return
        cmd_ok = self.fresh(
            self.last_cmd_time,
            self.cmd_timeout,
            now,
        )
        continuity_safe = (
            not self.dry_run
            and not self.stop_file.exists()
            and self.arm_file.is_file()
            and self.fault_reason is None
            and cmd_ok
        )
        continuity_requested = (
            self.planned_neutral_move
            or self.neutral_hold_s > 0.0
        )
        if not (continuity_safe and continuity_requested):
            self.stop_move("enable_false")

    def _latch_fault(self, reason: str) -> None:
        if self.fault_reason is not None:
            return
        self.fault_reason = reason
        self.stop_file.parent.mkdir(parents=True, exist_ok=True)
        self.stop_file.touch(exist_ok=True)
        self.fault_file.parent.mkdir(parents=True, exist_ok=True)
        self.fault_file.touch(exist_ok=True)
        self.get_logger().fatal(
            f"executor fault latched: {reason}"
        )
        self.stop_move(reason, force=True)

    def stop_move(
        self,
        reason: str,
        *,
        force: bool = False,
    ) -> int | None:
        if self.client is None:
            self.moving = False
            return 0
        now = time.monotonic()
        if (
            not force
            and not self.moving
            and now - self.last_stop_time < 1.0
        ):
            return 0
        try:
            with self.lock:
                code = self.client.StopMove()
        except Exception as exc:
            self.get_logger().error(
                f"StopMove exception ({reason}): {exc}"
            )
            return -1
        self.last_stop_time = time.monotonic()
        self.moving = False
        if code not in (None, 0):
            self.get_logger().error(
                f"StopMove failed ({reason}): code={code}"
            )
        return code

    def control_timer(self) -> None:
        now = time.monotonic()
        cmd_ok = self.fresh(
            self.last_cmd_time,
            self.cmd_timeout,
            now,
        )
        enable_fresh = self.fresh(
            self.last_enable_time,
            self.enable_timeout,
            now,
        )
        command_generation_ready = (
            self.safe_command_generation
            >= self.minimum_active_command_generation
        )
        enable_ok = (
            enable_fresh
            and self.enable_requested
            and command_generation_ready
        )
        stop_absent = not self.stop_file.exists()
        armed = self.arm_file.is_file()
        healthy = self.fault_reason is None

        requested_vx = clamp(
            float(self.requested.linear.x),
            0.0,
            self.max_vx,
        )
        requested_wz = clamp(
            float(self.requested.angular.z),
            -self.max_wz,
            self.max_wz,
        )
        force_planned_neutral = (
            self.planned_neutral_move
            and enable_fresh
            and (
                not self.enable_requested
                or not command_generation_ready
            )
        )
        if force_planned_neutral:
            # Bool False arrives directly while safe Twist traverses the bridge.
            # Ignore any cached or delayed non-zero Twist until a new rising
            # edge has also received a new safe command.
            vx = 0.0
            wz = 0.0
        else:
            vx = requested_vx
            wz = requested_wz
        nonzero = (
            vx > self.command_epsilon
            or abs(wz) > self.command_epsilon
        )
        active = (
            not self.dry_run
            and stop_absent
            and armed
            and cmd_ok
            and enable_ok
            and healthy
            and nonzero
        )

        if active:
            try:
                assert self.client is not None
                with self.lock:
                    code = self.client.Move(vx, 0.0, wz)
                if code not in (None, 0):
                    self._latch_fault(
                        f"move_return_error:{code}"
                    )
                else:
                    self.moving = True
                    self.last_nonzero_time = now
            except Exception as exc:
                self._latch_fault(
                    f"move_exception:{type(exc).__name__}"
                )
        else:
            planned_neutral = planned_neutral_allowed(
                enabled=(self.planned_neutral_move and enable_fresh),
                moving=self.moving,
                dry_run=self.dry_run,
                stop_absent=stop_absent,
                armed=armed,
                cmd_ok=cmd_ok,
                healthy=healthy,
                nonzero=nonzero,
            )
            neutral_hold = planned_neutral or neutral_hold_allowed(
                moving=self.moving,
                dry_run=self.dry_run,
                stop_absent=stop_absent,
                armed=armed,
                cmd_ok=cmd_ok,
                healthy=(healthy and enable_fresh),
                neutral_hold_s=self.neutral_hold_s,
                last_nonzero_time=self.last_nonzero_time,
                now=now,
            )
            if neutral_hold:
                try:
                    assert self.client is not None
                    with self.lock:
                        code = self.client.Move(0.0, 0.0, 0.0)
                    if code not in (None, 0):
                        self._latch_fault(
                            f"neutral_move_return_error:{code}"
                        )
                except Exception as exc:
                    self._latch_fault(
                        f"neutral_move_exception:{type(exc).__name__}"
                    )
            elif self.moving:
                if self.dry_run:
                    reason = "dry_run"
                elif not stop_absent:
                    reason = "stop_file"
                elif not armed:
                    reason = "arm_missing"
                elif not cmd_ok:
                    reason = "command_timeout"
                elif not enable_fresh:
                    reason = "enable_timeout"
                elif not self.enable_requested:
                    reason = "enable_false"
                elif not command_generation_ready:
                    reason = "waiting_for_new_safe_command"
                elif not healthy:
                    reason = "fault"
                else:
                    reason = "zero_command"
                code = self.stop_move(reason)
                if code not in (None, 0):
                    self._latch_fault(
                        f"stop_move_failed:{code}"
                    )

        if now - self.last_status_log >= 1.0:
            self.last_status_log = now
            self.get_logger().info(
                "active=%s dry_run=%s stop_absent=%s armed=%s "
                "cmd=%s enable=%s healthy=%s moving=%s "
                "safe=(%.3f, 0.000, %.3f)"
                % (
                    active,
                    self.dry_run,
                    stop_absent,
                    armed,
                    cmd_ok,
                    enable_ok,
                    healthy,
                    self.moving,
                    vx,
                    wz,
                )
            )

    def shutdown_stop(self) -> None:
        if self.client is not None:
            self.stop_move("shutdown", force=True)


def main(args=None) -> None:
    rclpy.init(args=args)

    def request_shutdown(_signum, _frame) -> None:
        if rclpy.ok():
            rclpy.shutdown()

    signal.signal(signal.SIGTERM, request_shutdown)
    signal.signal(signal.SIGINT, request_shutdown)
    node: MotionExecutorV2 | None = None
    try:
        node = MotionExecutorV2()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.shutdown_stop()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
