"""Pure fail-closed state machines for the V70 Go2 live loop.

This module has no ROS, CUDA, or Unitree dependency.  It turns the first
selected V70 local SE(2) prefix into bounded turn-then-drive executions,
and it parses a strict sequence of image goals for long-range navigation.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Sequence

import numpy as np


ROUTE_SCHEMA = "v70_image_goal_route_v1"

ACTIVE_HISTORY_SCAN_CAPTURE_SLOTS = {
    # Three semantic viewpoints for adaptive recovery.  The fourth turn is a
    # return-to-centre motion and deliberately has no capture slot.
    "adaptive_three": {
        0: 0,
        1: 1,
        2: 2,
    },
    "symmetric_compact": {
        0: 0,
        1: 1,
        2: 2,
        3: 3,
    },
    "symmetric": {
        0: 0,
        1: 1,
        3: 2,
        4: 3,
    },
    "left_sweep": {
        0: 0,
        1: 1,
        2: 2,
        3: 3,
    },
    "left_only": {
        0: 0,
        1: 1,
        2: 2,
        3: 3,
    },
    "right_to_left": {
        2: 0,
        3: 1,
        4: 2,
        5: 3,
    },
}

ACTIVE_HISTORY_SCAN_STEP_COUNTS = {
    "adaptive_three": 4,
    "symmetric_compact": 5,
    "symmetric": 6,
    "left_sweep": 5,
    "left_only": 4,
    "right_to_left": 6,
}

# A short Go2 burst must not be declared complete after only the generic
# absolute tolerance has been reached: body lean can create a transient
# odometry peak before a foot is replaced.
SHORT_ACTION_COMPLETION_FRACTION = 0.20
SHORT_ACTION_PARTIAL_PROGRESS_FRACTION = 0.50


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def wrap_angle(value: float) -> float:
    return float((value + math.pi) % (2.0 * math.pi) - math.pi)


def left_pad_temporal_history(
    values: Sequence[Any],
    *,
    count: int = 4,
) -> tuple[Any, ...]:
    """Return the newest temporal samples, padding startup with the oldest.

    A visual navigation model expects an ordered motion history.  At startup
    only one post-action, forward-facing sample exists, so repeating that
    sample is the neutral history: it represents zero prior camera motion.
    Panoramic scan viewpoints must never be used to fill this history.
    """

    if not isinstance(count, int) or count < 1:
        raise ValueError("temporal history count must be a positive integer")
    recent = tuple(values)[-count:]
    if not recent:
        raise ValueError("temporal history requires at least one sample")
    return (recent[0],) * (count - len(recent)) + recent


def bounded_plan_hold_fault_reason(
    *,
    consecutive_holds: int,
    max_consecutive_holds: int,
    cumulative_scan_translation_m: float,
    max_cumulative_scan_translation_m: float,
) -> str | None:
    """Return a fail-closed reason when replanning can no longer rescan safely."""

    if not isinstance(consecutive_holds, int) or consecutive_holds < 0:
        raise ValueError("consecutive plan holds must be a non-negative integer")
    if not isinstance(max_consecutive_holds, int) or max_consecutive_holds < 1:
        raise ValueError("maximum consecutive plan holds must be positive")
    values = (
        float(cumulative_scan_translation_m),
        float(max_cumulative_scan_translation_m),
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("scan translation limits must be finite")
    if cumulative_scan_translation_m < 0.0:
        raise ValueError("cumulative scan translation must be non-negative")
    if max_cumulative_scan_translation_m <= 0.0:
        raise ValueError("maximum cumulative scan translation must be positive")
    if cumulative_scan_translation_m >= max_cumulative_scan_translation_m:
        return (
            "scan_translation_budget_exhausted:"
            f"total_m={cumulative_scan_translation_m:.6f}:"
            f"limit_m={max_cumulative_scan_translation_m:.6f}"
        )
    if consecutive_holds >= max_consecutive_holds:
        return (
            "plan_hold_budget_exhausted:"
            f"consecutive={consecutive_holds}:"
            f"limit={max_consecutive_holds}"
        )
    return None


def active_history_scan_targets(
    step_yaw_rad: float,
    pattern: str = "symmetric",
) -> tuple[float, ...]:
    """Return bounded absolute yaw offsets for an active visual scan."""

    step = float(step_yaw_rad)
    if pattern not in ACTIVE_HISTORY_SCAN_CAPTURE_SLOTS:
        raise ValueError(
            "scan pattern must be adaptive_three, symmetric_compact, symmetric, "
            "left_sweep, left_only or right_to_left"
        )
    multiplier = {
        "adaptive_three": 1.0,
        "symmetric_compact": 1.5,
        "symmetric": 2.0,
        "left_sweep": 4.0,
        "left_only": 4.0,
        "right_to_left": 3.0,
    }[pattern]
    maximum_step = 0.75 / multiplier
    if (
        not math.isfinite(step)
        or step <= 0.0
        or multiplier * step > 0.75
    ):
        raise ValueError(
            f"{pattern} scan step yaw must be in (0, {maximum_step}] rad"
        )
    if pattern == "adaptive_three":
        # Capture left, centre, right, then return to the original centre.
        # Unlike the old four-slot sweep, every image has an unambiguous yaw
        # meaning and the final recenter turn never masquerades as evidence.
        return (step, 0.0, -step, 0.0)
    if pattern == "symmetric_compact":
        # Capture four views on both sides of the current heading and return
        # to that heading in five turns.  Compared with the old one-sided
        # right_to_left scan this removes a non-capture turn, avoids a lateral
        # observation bias, and keeps the final frame/action reference aligned.
        return (
            1.5 * step,
            0.5 * step,
            -0.5 * step,
            -1.5 * step,
            0.0,
        )
    if pattern == "left_sweep":
        return (step, 2.0 * step, 3.0 * step, 4.0 * step, 0.0)
    if pattern == "left_only":
        return (step, 2.0 * step, 3.0 * step, 4.0 * step)
    if pattern == "right_to_left":
        # Walk to the right edge in bounded increments without capturing,
        # then record four chronological views while sweeping left. The final
        # history frame is at the original heading used to start planning.
        return (
            -step,
            -2.0 * step,
            -3.0 * step,
            -2.0 * step,
            -step,
            0.0,
        )
    return (step, 2.0 * step, 0.0, -step, -2.0 * step, 0.0)


def active_history_scan_capture_slot(
    step_index: int,
    pattern: str = "symmetric",
) -> int | None:
    """Map a completed scan turn to its ordered V70 history-frame slot."""

    if pattern not in ACTIVE_HISTORY_SCAN_CAPTURE_SLOTS:
        raise ValueError(
            "scan pattern must be adaptive_three, symmetric_compact, symmetric, "
            "left_sweep, left_only or right_to_left"
        )
    step_count = ACTIVE_HISTORY_SCAN_STEP_COUNTS[pattern]
    if not isinstance(step_index, int) or not 0 <= step_index < step_count:
        raise ValueError(
            f"{pattern} scan step index must be in [0, {step_count - 1}]"
        )
    return ACTIVE_HISTORY_SCAN_CAPTURE_SLOTS[pattern].get(step_index)


def active_history_scan_next_step(
    step_index: int,
    pattern: str = "symmetric",
) -> int | None:
    """Return the next scan step, or ``None`` after the final step."""

    active_history_scan_capture_slot(step_index, pattern)
    next_index = step_index + 1
    if next_index == ACTIVE_HISTORY_SCAN_STEP_COUNTS[pattern]:
        return None
    return next_index


def active_history_scan_fault_is_recoverable(reason: str) -> bool:
    """Allow a point-cloud-blocked scan turn to degrade, not abort nav.

    The independent safety bridge continues to command zero throughout the
    block.  Only its bounded turn-block timeout is recoverable here; all
    other follower, sensor and interlock faults remain fail-closed.
    """

    return str(reason) == "turn_safety_block_timeout"


def external_control_rebase_update(
    *,
    anchor: Pose2D | None,
    stable_since: float | None,
    current: Pose2D,
    now: float,
    translation_tolerance_m: float,
    yaw_tolerance_rad: float,
    stable_duration_s: float,
) -> tuple[Pose2D, float, bool]:
    """Track a zero-command stabilization window after external motion."""

    values = (
        now,
        translation_tolerance_m,
        yaw_tolerance_rad,
        stable_duration_s,
    )
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("external-control rebase values must be finite")
    if now < 0.0 or translation_tolerance_m <= 0.0:
        raise ValueError("invalid external-control translation window")
    if yaw_tolerance_rad <= 0.0 or stable_duration_s <= 0.0:
        raise ValueError("invalid external-control stability window")
    if anchor is None:
        return current, float(now), False
    moved = bool(
        math.hypot(current.x - anchor.x, current.y - anchor.y)
        > translation_tolerance_m
        or abs(wrap_angle(current.yaw - anchor.yaw)) > yaw_tolerance_rad
    )
    if moved:
        return current, float(now), False
    since = float(now if stable_since is None else stable_since)
    return anchor, since, bool(now - since >= stable_duration_s)


def active_history_scan_min_wz(
    action_yaw_rad: float,
    *,
    base_min_wz: float,
    right_min_wz: float,
    max_wz: float,
) -> float:
    """Select the calibrated asymmetric minimum scan yaw command."""

    values = (action_yaw_rad, base_min_wz, right_min_wz, max_wz)
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("scan yaw command values must be finite")
    if not 0.0 < base_min_wz <= max_wz:
        raise ValueError("base scan min_wz must be in (0, max_wz]")
    if not base_min_wz <= right_min_wz <= max_wz:
        raise ValueError("right scan min_wz must be in [base_min_wz, max_wz]")
    return float(right_min_wz if action_yaw_rad < 0.0 else base_min_wz)


def active_scan_camera_recovery_allowed(
    *,
    camera_only: bool,
    scan_enabled: bool,
    scan_state: str,
) -> bool:
    """Allow a stopped scan to recover without relaxing drive safety."""

    return bool(
        camera_only
        and scan_enabled
        and scan_state in ("TURNING", "WAIT_FRAME")
    )


def update_directional_visual_confirmation(
    previous_direction: int,
    previous_streak: int,
    previous_misses: int,
    *,
    strong: bool,
    direction: int,
    max_misses: int = 1,
) -> tuple[int, int, int]:
    """Update bounded visual-direction hysteresis for recovery gating.

    A single weak/invalid four-frame estimate may be caused by blur between
    physical steps. Preserve the previous evidence for that one miss, but
    authorize recovery only when the current estimate is strong. A strong
    opposite estimate always starts a new streak immediately.
    """

    if previous_direction not in (-1, 0, 1):
        raise ValueError("previous_direction must be -1, 0 or 1")
    if direction not in (-1, 0, 1):
        raise ValueError("direction must be -1, 0 or 1")
    if previous_streak < 0 or previous_misses < 0 or max_misses < 0:
        raise ValueError("visual confirmation counters must be nonnegative")
    if strong:
        if direction not in (-1, 1):
            raise ValueError("strong visual direction must be -1 or 1")
        streak = (
            previous_streak + 1
            if direction == previous_direction
            else 1
        )
        return direction, streak, 0
    if (
        previous_direction in (-1, 1)
        and previous_streak > 0
        and previous_misses < max_misses
    ):
        return previous_direction, previous_streak, previous_misses + 1
    return 0, 0, 0


@dataclass(frozen=True)
class Pose2D:
    receive_time: float
    x: float
    y: float
    yaw: float

    def validate(self) -> None:
        if not all(
            math.isfinite(float(value))
            for value in (
                self.receive_time,
                self.x,
                self.y,
                self.yaw,
            )
        ):
            raise ValueError("pose contains NaN/Inf")


def robust_pose_median(
    samples: Sequence[Pose2D],
    *,
    window_size: int = 9,
) -> Pose2D:
    """Return a short-window median pose while preserving freshness.

    The Unitree lidar odometry can oscillate around a stationary pose at a
    much higher rate than the control loop.  Feeding the newest raw sample to
    the follower made a single yaw outlier repeatedly cross the TURN/DRIVE
    boundary.  Position medians and a wrapped yaw median reject those
    outliers without changing the timestamp used by stale-odometry checks.
    """

    if not isinstance(window_size, int) or window_size < 3:
        raise ValueError("pose median window_size must be an integer >= 3")
    if not samples:
        raise ValueError("pose median requires at least one sample")
    selected = tuple(samples[-window_size:])
    for sample in selected:
        sample.validate()
    newest = selected[-1]
    yaw_offsets = np.asarray(
        [wrap_angle(sample.yaw - newest.yaw) for sample in selected],
        dtype=np.float64,
    )
    return Pose2D(
        receive_time=newest.receive_time,
        x=float(np.median([sample.x for sample in selected])),
        y=float(np.median([sample.y for sample in selected])),
        yaw=wrap_angle(newest.yaw + float(np.median(yaw_offsets))),
    )


class FollowerPhase(str, Enum):
    IDLE = "IDLE"
    TURN = "TURN"
    TURN_SETTLE = "TURN_SETTLE"
    DRIVE = "DRIVE"
    SETTLE = "SETTLE"
    COMPLETE = "COMPLETE"
    FAULT = "FAULT"


def rebase_local_action_sequence(
    actions: np.ndarray,
    *,
    snapshot_pose: Pose2D,
    current_pose: Pose2D,
    max_translation_drift_m: float,
    max_yaw_drift_rad: float,
    drive_tolerance_m: float,
    turn_tolerance_rad: float,
    max_action_distance_m: float,
    max_abs_action_yaw_rad: float,
    heading_consistency_rad: float,
    coalesce_short_remainder: bool = False,
    promoted_first_max_abs_yaw_rad: float | None = None,
    align_rebased_forward_endpoint: bool = False,
    aligned_endpoint_max_heading_error_rad: float = 0.10,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Re-express a prefetched first action at the current robot pose.

    The planner output is not resampled or rescored. Its first SE(2) endpoint
    is held fixed in the odometry frame and expressed relative to the pose at
    handoff. A first transition that has already been completed is removed.
    Any result outside the existing follower contract is rejected so the
    caller can obtain a fresh observation instead.
    """

    values = np.asarray(actions, dtype=np.float64)
    snapshot_pose.validate()
    current_pose.validate()
    limits = (
        max_translation_drift_m,
        max_yaw_drift_rad,
        drive_tolerance_m,
        turn_tolerance_rad,
        max_action_distance_m,
        max_abs_action_yaw_rad,
        heading_consistency_rad,
    )
    if values.ndim != 2 or values.shape[1] != 3 or not len(values):
        raise ValueError("actions must have shape [H, 3] with H >= 1")
    if not np.isfinite(values).all() or not all(
        math.isfinite(float(value)) and float(value) >= 0.0
        for value in limits
    ):
        raise ValueError("prefetch rebase inputs must be finite")
    promoted_yaw_limit = float(max_abs_action_yaw_rad)
    if promoted_first_max_abs_yaw_rad is not None:
        promoted_yaw_limit = float(promoted_first_max_abs_yaw_rad)
        if (
            not math.isfinite(promoted_yaw_limit)
            or promoted_yaw_limit <= 0.0
            or promoted_yaw_limit > max_abs_action_yaw_rad
        ):
            raise ValueError("promoted first-action yaw limit is invalid")
    alignment_error_limit = float(aligned_endpoint_max_heading_error_rad)
    if (
        not math.isfinite(alignment_error_limit)
        or alignment_error_limit < heading_consistency_rad
        or alignment_error_limit > promoted_yaw_limit
    ):
        raise ValueError("aligned endpoint heading-error limit is invalid")

    drift_x = current_pose.x - snapshot_pose.x
    drift_y = current_pose.y - snapshot_pose.y
    drift_m = math.hypot(drift_x, drift_y)
    drift_yaw = abs(wrap_angle(current_pose.yaw - snapshot_pose.yaw))
    audit: dict[str, Any] = {
        "schema": "v70_prefetch_se2_rebase_v1",
        "drift_m": float(drift_m),
        "drift_yaw_rad": float(drift_yaw),
        "max_translation_drift_m": float(max_translation_drift_m),
        "max_yaw_drift_rad": float(max_yaw_drift_rad),
        "original_first_action": values[0].astype(np.float32).tolist(),
    }
    if (
        drift_m > max_translation_drift_m
        or drift_yaw > max_yaw_drift_rad
    ):
        audit.update({"accepted": False, "reason": "drift_envelope_exceeded"})
        return None, audit

    # A planner may retain an exact zero transition at the front of H4.  It is
    # a temporal placeholder, not a world-space endpoint.  Rebasing it after
    # the robot has moved makes the placeholder appear behind the robot and
    # unnecessarily discards the useful tail.  Drop exact leading no-ops before
    # applying SE(2) age compensation; no endpoint is changed by doing so.
    leading_noop_count = 0
    while (
        coalesce_short_remainder
        and leading_noop_count < len(values)
        and math.hypot(
            float(values[leading_noop_count][0]),
            float(values[leading_noop_count][1]),
        )
        <= 1.0e-6
        and abs(float(values[leading_noop_count][2])) <= 1.0e-6
    ):
        leading_noop_count += 1
    if leading_noop_count:
        audit["leading_noop_actions_dropped"] = int(leading_noop_count)
        if leading_noop_count == len(values):
            audit.update({
                "accepted": False,
                "reason": "leading_noop_actions_exhausted",
            })
            return None, audit
        values = values[leading_noop_count:]
        audit["effective_first_action"] = values[0].astype(np.float32).tolist()

    first = values[0]
    snapshot_cos = math.cos(snapshot_pose.yaw)
    snapshot_sin = math.sin(snapshot_pose.yaw)
    endpoint_world_x = (
        snapshot_pose.x
        + snapshot_cos * float(first[0])
        - snapshot_sin * float(first[1])
    )
    endpoint_world_y = (
        snapshot_pose.y
        + snapshot_sin * float(first[0])
        + snapshot_cos * float(first[1])
    )
    endpoint_world_yaw = wrap_angle(snapshot_pose.yaw + float(first[2]))
    world_x = endpoint_world_x - current_pose.x
    world_y = endpoint_world_y - current_pose.y
    current_cos = math.cos(current_pose.yaw)
    current_sin = math.sin(current_pose.yaw)
    rebased_first = np.asarray(
        [
            current_cos * world_x + current_sin * world_y,
            -current_sin * world_x + current_cos * world_y,
            wrap_angle(endpoint_world_yaw - current_pose.yaw),
        ],
        dtype=np.float64,
    )
    distance_m = math.hypot(
        float(rebased_first[0]), float(rebased_first[1])
    )
    heading_error = (
        0.0
        if distance_m <= 1.0e-6
        else abs(
            wrap_angle(
                math.atan2(
                    float(rebased_first[1]), float(rebased_first[0])
                )
                - float(rebased_first[2])
            )
        )
    )

    # Motion prefetch can finish its first endpoint before planning returns.
    # Promote the untouched tail from that WORLD endpoint when the remainder
    # is within 5 cm, or when the endpoint has just passed behind the robot by
    # at most 10 cm. Recursive rebasing retains the forward/distance/heading
    # contracts; the additional yaw bound prevents promotion into pure TURN.
    consumed_first_endpoint = bool(
        coalesce_short_remainder
        and len(values) > 1
        and (
            distance_m <= max(0.05, 2.0 * drive_tolerance_m)
            or (
                float(rebased_first[0]) <= 0.0
                and distance_m <= 0.10
            )
        )
    )
    if consumed_first_endpoint:
        rebased, tail_audit = rebase_local_action_sequence(
            values[1:],
            snapshot_pose=Pose2D(
                receive_time=current_pose.receive_time,
                x=endpoint_world_x,
                y=endpoint_world_y,
                yaw=endpoint_world_yaw,
            ),
            current_pose=current_pose,
            max_translation_drift_m=max_translation_drift_m,
            max_yaw_drift_rad=max_yaw_drift_rad,
            drive_tolerance_m=drive_tolerance_m,
            turn_tolerance_rad=turn_tolerance_rad,
            max_action_distance_m=max_action_distance_m,
            max_abs_action_yaw_rad=max_abs_action_yaw_rad,
            heading_consistency_rad=heading_consistency_rad,
            coalesce_short_remainder=False,
            promoted_first_max_abs_yaw_rad=promoted_yaw_limit,
            align_rebased_forward_endpoint=align_rebased_forward_endpoint,
            aligned_endpoint_max_heading_error_rad=alignment_error_limit,
        )
        promoted_distance_m = (
            math.hypot(float(rebased[0, 0]), float(rebased[0, 1]))
            if rebased is not None
            else math.inf
        )
        if (
            rebased is None
            or float(rebased[0, 0]) <= 1.0e-4
            or promoted_distance_m <= drive_tolerance_m + 1.0e-6
            or abs(float(rebased[0, 2])) > promoted_yaw_limit + 1.0e-7
        ):
            audit.update(
                {
                    "accepted": False,
                    "reason": "consumed_first_endpoint_tail_outside_contract",
                    "remaining_distance_m": float(distance_m),
                    "promoted_first_distance_m": float(promoted_distance_m),
                    "tail": tail_audit,
                    "promoted_first_max_abs_yaw_rad": promoted_yaw_limit,
                }
            )
            return None, audit
        audit.update(
            {
                "accepted": True,
                "reason": "consumed_first_endpoint_tail_promoted",
                "dropped_first_action": True,
                "remaining_distance_m": float(distance_m),
                "promoted_first_distance_m": float(promoted_distance_m),
                "rebased_first_action": rebased[0].tolist(),
                "promoted_first_max_abs_yaw_rad": promoted_yaw_limit,
                "tail": tail_audit,
            }
        )
        return rebased, audit

    # Join only nearly collinear forward segments. Preserve the second
    # WORLD endpoint instead of shifting an unchanged tail to the new pose.
    # This removes centimetre-long steps without widening follower limits.
    if coalesce_short_remainder and len(values) > 1 and distance_m <= 0.10:
        tail = values[1]
        c, s = math.cos(float(rebased_first[2])), math.sin(float(rebased_first[2]))
        merged = np.asarray([
            rebased_first[0] + c * tail[0] - s * tail[1],
            rebased_first[1] + s * tail[0] + c * tail[1],
            wrap_angle(float(rebased_first[2] + tail[2])),
        ])
        merged_distance = math.hypot(float(merged[0]), float(merged[1]))
        merged_heading = math.atan2(float(merged[1]), float(merged[0]))
        corner_cross_track = abs(
            -math.sin(merged_heading) * float(rebased_first[0])
            + math.cos(merged_heading) * float(rebased_first[1])
        )
        safe_join = bool(
            # Permit a bounded longitudinal overrun of the first endpoint.
            # The merged second endpoint must still be forward, close,
            # cross-track bounded, and heading-consistent below.
            float(rebased_first[0]) >= -min(0.10, 2.0 * drive_tolerance_m)
            and float(first[0]) >= 0.10
            and abs(float(first[2])) <= 0.15
            and float(tail[0]) >= 0.10
            and abs(float(tail[2])) <= 0.10
            # A small odometry yaw residual at a just-passed junction is safe
            # to absorb when the merged endpoint still passes the stricter
            # final heading and cross-track contract below.
            and abs(float(rebased_first[2])) <= min(
                0.15,
                max(0.10, heading_consistency_rad + turn_tolerance_rad),
            )
            and abs(wrap_angle(math.atan2(float(tail[1]), float(tail[0])) - float(tail[2]))) <= heading_consistency_rad
            and corner_cross_track <= drive_tolerance_m
            and merged[0] > 0.0
            and 0.10 <= merged_distance <= max_action_distance_m
            and abs(float(merged[2])) <= max_abs_action_yaw_rad
            and abs(wrap_angle(merged_heading - float(merged[2]))) <= heading_consistency_rad
        )
        if safe_join:
            rebased = values[1:].astype(np.float32)
            rebased[0] = merged
            audit.update({
                "accepted": True,
                "reason": "short_remainder_joined_to_second_world_endpoint",
                "dropped_first_action": True,
                "remaining_distance_m": float(distance_m),
                "joined_distance_m": float(merged_distance),
                "corner_cross_track_m": float(corner_cross_track),
                "rebased_first_action": rebased[0].tolist(),
            })
            return rebased, audit

    if (
        distance_m <= drive_tolerance_m
        and abs(float(rebased_first[2])) <= turn_tolerance_rad
    ):
        if len(values) == 1:
            audit.update(
                {
                    "accepted": False,
                    "reason": "first_action_already_completed_no_tail",
                    "rebased_first_action": rebased_first.astype(
                        np.float32
                    ).tolist(),
                }
            )
            return None, audit
        # Keep the second endpoint fixed even when the first was completed
        # within tolerance rather than reached exactly.
        rebased, tail_audit = rebase_local_action_sequence(
            values[1:],
            snapshot_pose=Pose2D(receive_time=current_pose.receive_time,
                                 x=endpoint_world_x, y=endpoint_world_y,
                                 yaw=endpoint_world_yaw),
            current_pose=current_pose,
            max_translation_drift_m=max_translation_drift_m,
            max_yaw_drift_rad=max_yaw_drift_rad,
            drive_tolerance_m=drive_tolerance_m,
            turn_tolerance_rad=turn_tolerance_rad,
            max_action_distance_m=max_action_distance_m,
            max_abs_action_yaw_rad=max_abs_action_yaw_rad,
            heading_consistency_rad=heading_consistency_rad,
            promoted_first_max_abs_yaw_rad=promoted_yaw_limit,
            align_rebased_forward_endpoint=align_rebased_forward_endpoint,
            aligned_endpoint_max_heading_error_rad=alignment_error_limit,
        )
        promoted_distance_m = (
            math.hypot(float(rebased[0, 0]), float(rebased[0, 1]))
            if rebased is not None
            else math.inf
        )
        if (
            rebased is None
            or float(rebased[0, 0]) <= 1.0e-4
            or promoted_distance_m <= drive_tolerance_m + 1.0e-6
            or abs(float(rebased[0, 2])) > promoted_yaw_limit + 1.0e-7
        ):
            audit.update({"accepted": False, "reason": "completed_tail_outside_contract", "tail": tail_audit})
            return None, audit
        audit.update(
            {
                "accepted": True,
                "reason": "first_action_already_completed_tail_promoted",
                "dropped_first_action": True,
                "rebased_first_action": rebased[0].tolist(),
                "promoted_first_distance_m": float(promoted_distance_m),
            }
        )
        return rebased, audit

    contract_ok = bool(
        float(rebased_first[0]) >= -1.0e-4
        and distance_m <= max_action_distance_m
        and abs(float(rebased_first[2])) <= max_abs_action_yaw_rad
        and (
            distance_m <= drive_tolerance_m
            or heading_error <= heading_consistency_rad
        )
    )
    if not contract_ok:
        # A moving snapshot can leave the planned position safely ahead while
        # its old endpoint yaw no longer matches the arc from the handoff pose.
        # Preserve that position, align H1 yaw to its bearing, and discard the
        # stale tail. This keeps the existing forward/distance/yaw envelope and
        # forces a fresh observation after one executable action.
        endpoint_heading = wrap_angle(
            math.atan2(float(rebased_first[1]), float(rebased_first[0]))
        )
        if (
            align_rebased_forward_endpoint
            and float(rebased_first[0]) > 1.0e-4
            and distance_m >= max(0.08, 3.0 * drive_tolerance_m)
            and distance_m <= max_action_distance_m
            and abs(endpoint_heading) <= promoted_yaw_limit + 1.0e-7
            and heading_error <= alignment_error_limit + 1.0e-7
        ):
            aligned = rebased_first.astype(np.float32)
            aligned[2] = np.float32(endpoint_heading)
            audit.update(
                {
                    "accepted": True,
                    "reason": "rebased_endpoint_yaw_aligned_for_h1",
                    "rebased_first_action": aligned.tolist(),
                    "remaining_distance_m": float(distance_m),
                    "original_rebased_yaw_rad": float(rebased_first[2]),
                    "aligned_endpoint_yaw_rad": float(endpoint_heading),
                    "aligned_endpoint_max_heading_error_rad": (
                        alignment_error_limit
                    ),
                    "tail_actions_dropped": int(max(0, len(values) - 1)),
                }
            )
            return aligned.reshape(1, 3), audit
        audit.update(
            {
                "accepted": False,
                "reason": "rebased_action_outside_follower_contract",
                "rebased_first_action": rebased_first.astype(
                    np.float32
                ).tolist(),
                "remaining_distance_m": float(distance_m),
                "heading_error_rad": float(heading_error),
            }
        )
        return None, audit

    rebased = values.astype(np.float32)
    rebased[0] = rebased_first.astype(np.float32)
    audit.update(
        {
            "accepted": True,
            "reason": "first_endpoint_preserved_in_odometry_frame",
            "dropped_first_action": False,
            "rebased_first_action": rebased[0].tolist(),
            "remaining_distance_m": float(distance_m),
            "heading_error_rad": float(heading_error),
        }
    )
    return rebased, audit


@dataclass(frozen=True)
class ActionFollowerConfig:
    max_vx: float = 0.05
    min_vx: float = 0.02
    max_wz: float = 0.10
    min_wz: float = 0.025
    # Optional physical Go2 turn assist. The planner action is unchanged;
    # these values only make execution robust to elastic-foot yaw deadband.
    turn_arc_vx: float = 0.0
    turn_arc_max_distance_m: float = 0.0
    continuous_forward_arc: bool = False
    continuous_forward_arc_max_yaw_rad: float = 0.0
    continuous_forward_arc_max_vx: float = 0.0
    # A forward arc may finish with a small elastic-foot yaw residual.  Keep
    # this independent from TURN tolerance: the next camera plan can correct
    # the residual without forcing a stop after the requested distance has
    # already been reached. Zero preserves the generic follower behavior.
    continuous_forward_arc_completion_yaw_tolerance_rad: float = 0.0
    drive_min_wz: float = 0.0
    drive_steering_deadband_rad: float = 0.0
    # Optional velocity taper before a planned DRIVE endpoint.  It mirrors the
    # downstream Safe Bridge acceleration limit so a normal endpoint does not
    # change from walking speed to zero in one control cycle.
    drive_braking_deceleration_mps2: float = 0.0
    # Optional tighter physical endpoint than the generic action/no-op
    # tolerance. Zero preserves the generic follower behavior. A live Go2 can
    # keep a gait-valid velocity until this smaller endpoint instead of
    # entering a sub-stride speed band several centimetres early.
    drive_completion_tolerance_m: float = 0.0
    drive_kp: float = 0.8
    yaw_kp: float = 1.2
    turn_tolerance_rad: float = 0.045
    # DRIVE uses a wider threshold than TURN completion so odometry noise
    # cannot cause rapid TURN/DRIVE oscillation.  A sustained violation is
    # still handled fail-closed by stopping before re-alignment.
    drive_realign_threshold_rad: float = 0.08
    turn_stall_margin_rad: float = 0.030
    drive_tolerance_m: float = 0.025
    max_cross_track_m: float = 0.12
    max_action_distance_m: float = 0.38
    max_abs_action_yaw_rad: float = 0.75
    heading_consistency_rad: float = 0.08
    max_odom_age_s: float = 0.30
    turn_timeout_s: float = 10.0
    turn_settle_time_s: float = 0.30
    drive_timeout_s: float = 12.0
    no_progress_timeout_s: float = 2.0
    # A bounded physical execution horizon keeps long V70 transitions from
    # holding one Sport Move command after the Go2 has stopped replacing
    # foot contacts.  Zero disables either recovery for generic profiles.
    max_drive_burst_m: float = 0.0
    partial_progress_replan_min_m: float = 0.0
    partial_progress_replan_timeout_s: float = 0.0
    safety_block_timeout_s: float = 8.0
    progress_epsilon_m: float = 0.004
    progress_epsilon_rad: float = 0.015
    settle_time_s: float = 0.60
    required_turn_confirmations: int = 3
    required_drive_realign_confirmations: int = 3
    minimum_turn_motion_time_s: float = 0.0
    minimum_drive_motion_time_s: float = 0.0
    # Optional main-method-only evidence that a commanded forward burst has
    # produced a real stride instead of body lean / odometry micro-drift.
    drive_gait_commit_timeout_s: float = 0.0
    drive_gait_commit_min_progress_m: float = 0.0

    def validate(self) -> None:
        positive = asdict(self)
        nonnegative = {
            "turn_arc_vx": positive.pop("turn_arc_vx"),
            "turn_arc_max_distance_m": positive.pop(
                "turn_arc_max_distance_m"
            ),
            "continuous_forward_arc": positive.pop(
                "continuous_forward_arc"
            ),
            "continuous_forward_arc_max_yaw_rad": positive.pop(
                "continuous_forward_arc_max_yaw_rad"
            ),
            "continuous_forward_arc_max_vx": positive.pop(
                "continuous_forward_arc_max_vx"
            ),
            "continuous_forward_arc_completion_yaw_tolerance_rad": positive.pop(
                "continuous_forward_arc_completion_yaw_tolerance_rad"
            ),
            "drive_min_wz": positive.pop("drive_min_wz"),
            "drive_steering_deadband_rad": positive.pop(
                "drive_steering_deadband_rad"
            ),
            "drive_braking_deceleration_mps2": positive.pop(
                "drive_braking_deceleration_mps2"
            ),
            "drive_completion_tolerance_m": positive.pop(
                "drive_completion_tolerance_m"
            ),
            "max_drive_burst_m": positive.pop(
                "max_drive_burst_m"
            ),
            "partial_progress_replan_min_m": positive.pop(
                "partial_progress_replan_min_m"
            ),
            "partial_progress_replan_timeout_s": positive.pop(
                "partial_progress_replan_timeout_s"
            ),
            "minimum_turn_motion_time_s": positive.pop(
                "minimum_turn_motion_time_s"
            ),
            "minimum_drive_motion_time_s": positive.pop(
                "minimum_drive_motion_time_s"
            ),
            "drive_gait_commit_timeout_s": positive.pop(
                "drive_gait_commit_timeout_s"
            ),
            "drive_gait_commit_min_progress_m": positive.pop(
                "drive_gait_commit_min_progress_m"
            ),
        }
        positive.pop("required_turn_confirmations")
        positive.pop("required_drive_realign_confirmations")
        if not isinstance(self.continuous_forward_arc, bool):
            raise ValueError("continuous_forward_arc must be boolean")
        for name, value in positive.items():
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        for name, value in nonnegative.items():
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.min_vx > self.max_vx:
            raise ValueError("min_vx must be <= max_vx")
        if self.min_wz > self.max_wz:
            raise ValueError("min_wz must be <= max_wz")
        if self.turn_arc_vx > self.max_vx:
            raise ValueError("turn_arc_vx must be <= max_vx")
        if self.turn_arc_vx > 0.0 and self.turn_arc_max_distance_m <= 0.0:
            raise ValueError(
                "turn_arc_max_distance_m must be positive when turn arc is enabled"
            )
        if self.drive_min_wz > self.max_wz:
            raise ValueError("drive_min_wz must be <= max_wz")
        if self.continuous_forward_arc:
            if (
                self.continuous_forward_arc_max_yaw_rad
                <= self.drive_realign_threshold_rad
            ):
                raise ValueError(
                    "continuous arc max yaw must exceed the drive realign threshold"
                )
            if not 0.0 < self.continuous_forward_arc_max_vx <= self.max_vx:
                raise ValueError(
                    "continuous arc max vx must be in (0, max_vx]"
                )
            if self.continuous_forward_arc_max_yaw_rad > self.max_abs_action_yaw_rad:
                raise ValueError(
                    "continuous arc max yaw must not exceed max action yaw"
                )
            if (
                self.continuous_forward_arc_completion_yaw_tolerance_rad
                > self.continuous_forward_arc_max_yaw_rad
            ):
                raise ValueError(
                    "continuous arc completion yaw must not exceed arc max yaw"
                )
            if self.drive_min_wz <= 0.0:
                raise ValueError(
                    "continuous forward arc requires positive drive_min_wz"
                )
            maximum_lateral_excursion = (
                self.continuous_forward_arc_max_vx
                / self.drive_min_wz
                * (1.0 - math.cos(self.continuous_forward_arc_max_yaw_rad))
            )
            if maximum_lateral_excursion > self.drive_tolerance_m:
                raise ValueError(
                    "continuous arc lateral excursion exceeds drive tolerance"
                )
        if self.drive_tolerance_m >= self.max_action_distance_m:
            raise ValueError("drive tolerance is too large")
        if (
            self.drive_completion_tolerance_m > 0.0
            and self.drive_completion_tolerance_m
            > self.drive_tolerance_m
        ):
            raise ValueError(
                "drive completion tolerance must not exceed drive tolerance"
            )
        if (
            self.max_drive_burst_m > 0.0
            and self.max_drive_burst_m <= self.drive_tolerance_m
        ):
            raise ValueError(
                "max_drive_burst_m must exceed drive_tolerance_m"
            )
        partial_enabled = (
            self.partial_progress_replan_min_m > 0.0
            or self.partial_progress_replan_timeout_s > 0.0
        )
        if partial_enabled and not (
            self.partial_progress_replan_min_m > 0.0
            and self.partial_progress_replan_timeout_s > 0.0
        ):
            raise ValueError(
                "partial-progress recovery requires both thresholds"
            )
        if (
            partial_enabled
            and self.partial_progress_replan_timeout_s
            >= self.no_progress_timeout_s
        ):
            raise ValueError(
                "partial-progress timeout must precede hard no-progress timeout"
            )
        if self.required_turn_confirmations < 2:
            raise ValueError(
                "required_turn_confirmations must be at least 2"
            )
        if self.required_drive_realign_confirmations < 2:
            raise ValueError(
                "required_drive_realign_confirmations must be at least 2"
            )
        if self.drive_realign_threshold_rad <= self.turn_tolerance_rad:
            raise ValueError(
                "drive_realign_threshold_rad must exceed turn_tolerance_rad"
            )
        gait_commit_enabled = (
            self.drive_gait_commit_timeout_s > 0.0
            or self.drive_gait_commit_min_progress_m > 0.0
        )
        if gait_commit_enabled and not (
            self.drive_gait_commit_timeout_s > 0.0
            and self.drive_gait_commit_min_progress_m > 0.0
        ):
            raise ValueError(
                "gait commit requires both timeout and progress thresholds"
            )
        if (
            gait_commit_enabled
            and self.drive_gait_commit_timeout_s >= self.drive_timeout_s
        ):
            raise ValueError("gait commit timeout must precede drive timeout")


def go2_live_action_follower_config(
    *,
    max_odom_age_s: float,
) -> ActionFollowerConfig:
    """Return the calibrated Go2 live actuation profile.

    V70 still supplies the target local SE(2) action.  These values only map
    that target to SportClient velocities above the physical Go2 deadband.
    """
    config = ActionFollowerConfig(
        max_vx=0.20,
        min_vx=0.20,
        max_wz=0.35,
        min_wz=0.30,
        # Match the physically proven scan behavior: a bounded 3 cm forward
        # arc lets the elastic-foot Go2 replace support contacts while yawing.
        # The dedicated drive phase still starts only after the full turn.
        turn_arc_vx=0.20,
        turn_arc_max_distance_m=0.03,
        continuous_forward_arc=False,
        drive_min_wz=0.0,
        drive_steering_deadband_rad=0.015,
        turn_tolerance_rad=0.015,
        drive_realign_threshold_rad=0.06,
        minimum_turn_motion_time_s=0.20,
        minimum_drive_motion_time_s=0.20,
        max_odom_age_s=max_odom_age_s,
        no_progress_timeout_s=3.0,
        # Do not impose an artificial distance cut on a selected transition.
        # Only the measured no-progress recovery below may end it early.
        # This leaves V70 proposals, scores, and trained weights unchanged.
        max_drive_burst_m=0.0,
        # The last live run advanced 0.0978 m and then stalled.  Replan from
        # rolling observations after useful 8 cm progress instead of missing
        # the old 10 cm boundary by millimetres and hard-faulting later.
        partial_progress_replan_min_m=0.08,
        partial_progress_replan_timeout_s=0.80,
    )
    config.validate()
    return config


def bounded_directional_execution_prefix(
    actions_h4: np.ndarray,
    *,
    enabled: bool,
    endpoint_lateral_threshold_m: float = 0.06,
    realized_lateral_threshold_m: float = 0.01,
    max_prefix_steps: int = 3,
    max_prefix_distance_m: float = 0.85,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Commit only enough selected H4 steps to realize its lateral intent.

    A one-step receding horizon is preserved unless a strong H4 endpoint is
    not yet represented by the first cumulative waypoint.  In that case the
    exact planned transitions are retained, but a bounded prefix is executed
    before replanning.  No action is mirrored, rescaled or synthesized.
    """

    actions = np.asarray(actions_h4, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] != 3:
        raise ValueError("actions_h4 must have shape [H,3]")
    if not 1 <= actions.shape[0] <= 4:
        raise ValueError("actions_h4 horizon must be in [1,4]")
    if not np.isfinite(actions).all():
        raise ValueError("actions_h4 contains NaN/Inf")
    if not (
        math.isfinite(endpoint_lateral_threshold_m)
        and endpoint_lateral_threshold_m > 0.0
        and math.isfinite(realized_lateral_threshold_m)
        and realized_lateral_threshold_m > 0.0
        and 1 <= max_prefix_steps <= actions.shape[0]
        and math.isfinite(max_prefix_distance_m)
        and max_prefix_distance_m > 0.0
    ):
        raise ValueError("invalid directional prefix limits")

    x = y = yaw = 0.0
    cumulative_distance = 0.0
    waypoints: list[tuple[float, float, float]] = []
    for dx_local, dy_local, dyaw in actions:
        cosine = math.cos(yaw)
        sine = math.sin(yaw)
        x += cosine * float(dx_local) - sine * float(dy_local)
        y += sine * float(dx_local) + cosine * float(dy_local)
        yaw = wrap_angle(yaw + float(dyaw))
        cumulative_distance += math.hypot(
            float(dx_local),
            float(dy_local),
        )
        waypoints.append((x, y, cumulative_distance))

    endpoint_y_m = float(waypoints[-1][1])
    direction = (
        1 if endpoint_y_m > 0.0 else (-1 if endpoint_y_m < 0.0 else 0)
    )
    active = bool(
        enabled
        and direction
        and abs(endpoint_y_m) >= endpoint_lateral_threshold_m
        and direction * waypoints[0][1]
        < realized_lateral_threshold_m
    )
    prefix_steps = 1
    reason = "single_step_default"
    if active:
        reason = "bounded_prefix_not_found"
        for index in range(1, min(max_prefix_steps, len(waypoints))):
            if waypoints[index][2] > max_prefix_distance_m:
                break
            if (
                direction * waypoints[index][1]
                >= realized_lateral_threshold_m
            ):
                prefix_steps = index + 1
                reason = "lateral_intent_realized"
                break

    audit = {
        "schema": "v70_directional_execution_prefix_v1",
        "enabled": bool(enabled),
        "active": active,
        "reason": reason,
        "prefix_steps": prefix_steps,
        "endpoint_y_m": endpoint_y_m,
        "first_waypoint_y_m": float(waypoints[0][1]),
        "prefix_distance_m": float(
            waypoints[prefix_steps - 1][2]
        ),
    }
    return actions[:prefix_steps].astype(np.float32), audit


def gated_h2_execution_prefix(
    actions_h4: np.ndarray,
    *,
    enabled: bool,
    approach_mode: bool,
    visual_direction: int = 0,
    visual_conflict: bool = False,
    max_total_distance_m: float = 0.45,
    max_total_abs_yaw_rad: float = 0.15,
    max_internal_bearing_change_rad: float = 0.10,
    endpoint_lateral_threshold_m: float = 0.03,
    lateral_tolerance_m: float = 0.01,
    yaw_tolerance_rad: float = 0.03,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Execute H2 only when the recorded real-robot coherence gate passes.

    V70 still selects the complete H4 trajectory.  This gate only decides
    whether its exact second transition may be handed to the odometry follower
    without another camera scan.  It never changes, mirrors, or rescales an
    action, and it falls back to the normal one-step receding horizon.
    """

    actions = np.asarray(actions_h4, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] != 3:
        raise ValueError("actions_h4 must have shape [H,3]")
    if not 1 <= actions.shape[0] <= 4:
        raise ValueError("actions_h4 horizon must be in [1,4]")
    if not np.isfinite(actions).all():
        raise ValueError("actions_h4 contains NaN/Inf")
    if visual_direction not in (-1, 0, 1):
        raise ValueError("visual_direction must be -1, 0, or 1")
    limits = (
        max_total_distance_m,
        max_total_abs_yaw_rad,
        max_internal_bearing_change_rad,
        endpoint_lateral_threshold_m,
        lateral_tolerance_m,
        yaw_tolerance_rad,
    )
    if not all(
        math.isfinite(float(value)) and float(value) > 0.0
        for value in limits
    ):
        raise ValueError("gated H2 limits must be finite and positive")

    reason = "disabled"
    metrics: dict[str, Any] = {}
    authorized = bool(enabled and actions.shape[0] >= 2)
    if not enabled:
        reason = "disabled"
    elif actions.shape[0] < 2:
        reason = "horizon_shorter_than_two"
    elif visual_conflict:
        reason = "visual_conflict"
        authorized = False
    else:
        first = actions[0]
        second = actions[1]
        first_bearing = math.atan2(float(first[1]), float(first[0]))
        second_bearing = math.atan2(float(second[1]), float(second[0]))
        internal_bearing_change = abs(
            wrap_angle(first_bearing - second_bearing)
        )
        total_distance = sum(
            math.hypot(float(action[0]), float(action[1]))
            for action in (first, second)
        )
        total_yaw = float(first[2]) + float(second[2])

        x = y = yaw = 0.0
        h2_y = 0.0
        endpoint_y = 0.0
        for index, (dx_local, dy_local, dyaw) in enumerate(actions):
            cosine = math.cos(yaw)
            sine = math.sin(yaw)
            x += cosine * float(dx_local) - sine * float(dy_local)
            y += sine * float(dx_local) + cosine * float(dy_local)
            yaw = wrap_angle(yaw + float(dyaw))
            if index == 1:
                h2_y = y
            endpoint_y = y
        endpoint_direction = (
            1
            if endpoint_y >= endpoint_lateral_threshold_m
            else (
                -1
                if endpoint_y <= -endpoint_lateral_threshold_m
                else 0
            )
        )
        endpoint_consistent = bool(
            endpoint_direction == 0
            or (
                endpoint_direction * h2_y >= -lateral_tolerance_m
                and endpoint_direction * total_yaw >= -yaw_tolerance_rad
            )
        )
        visual_consistent = bool(
            visual_direction == 0
            or (
                visual_direction * h2_y >= -lateral_tolerance_m
                and visual_direction * total_yaw >= -yaw_tolerance_rad
            )
        )
        metrics = {
            "h2_total_distance_m": total_distance,
            "h2_total_yaw_rad": total_yaw,
            "h2_internal_bearing_change_rad": internal_bearing_change,
            "h2_y_m": h2_y,
            "h4_endpoint_y_m": endpoint_y,
            "h4_endpoint_direction": endpoint_direction,
            "visual_direction": visual_direction,
            "endpoint_consistent": endpoint_consistent,
            "visual_consistent": visual_consistent,
        }
        checks = (
            (
                total_distance <= max_total_distance_m + 1.0e-9,
                "distance_limit",
            ),
            (
                abs(total_yaw) <= max_total_abs_yaw_rad + 1.0e-9,
                "yaw_limit",
            ),
            (
                internal_bearing_change
                <= max_internal_bearing_change_rad + 1.0e-9,
                "internal_bearing_change",
            ),
            (endpoint_consistent, "h4_endpoint_conflict"),
            (visual_consistent, "visual_direction_conflict"),
        )
        for passed, failed_reason in checks:
            if not passed:
                authorized = False
                reason = failed_reason
                break
        else:
            reason = "coherent_h2"

    prefix_steps = 2 if authorized else 1
    audit = {
        "schema": "v70_gated_h2_execution_prefix_v1",
        "enabled": bool(enabled),
        "active": authorized,
        "reason": reason,
        "prefix_steps": prefix_steps,
        "approach_mode": bool(approach_mode),
        "visual_conflict": bool(visual_conflict),
        "limits": {
            "max_total_distance_m": max_total_distance_m,
            "max_total_abs_yaw_rad": max_total_abs_yaw_rad,
            "max_internal_bearing_change_rad": (
                max_internal_bearing_change_rad
            ),
        },
        **metrics,
    }
    return actions[:prefix_steps].astype(np.float32), audit


def operator_guided_turn_first_action(
    action: Sequence[float],
    *,
    enabled: bool,
    direction: int,
    turn_gain: float = 6.0,
    minimum_abs_yaw_rad: float = 0.08,
    maximum_abs_yaw_rad: float = 0.60,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Make a fresh operator annotation an explicit turn-first action.

    The selected transition distance is preserved, while its heading and
    yaw are aligned with the annotated side. The model yaw magnitude is
    amplified by ``turn_gain`` with bounded floor/ceiling values. This keeps
    the action inside the follower SE(2) contract, so the existing non-arc
    follower must finish the turn before it can command forward motion.
    """

    values = np.asarray(action, dtype=np.float64)
    if values.shape != (3,) or not np.isfinite(values).all():
        raise ValueError("action must be one finite SE(2) transition")
    if direction not in (-1, 0, 1):
        raise ValueError("direction must be -1, 0, or 1")
    limits = (turn_gain, minimum_abs_yaw_rad, maximum_abs_yaw_rad)
    if not all(math.isfinite(float(value)) for value in limits):
        raise ValueError("operator turn-first limits must be finite")
    if turn_gain <= 0.0:
        raise ValueError("turn_gain must be positive")
    if not 0.0 < minimum_abs_yaw_rad <= maximum_abs_yaw_rad:
        raise ValueError("invalid operator turn-first yaw bounds")

    original = values.astype(np.float32)
    active = bool(enabled and direction in (-1, 1))
    audit: dict[str, Any] = {
        "schema": "v70_operator_turn_first_v1",
        "active": active,
        "direction": int(direction),
        "turn_gain": float(turn_gain),
        "minimum_abs_yaw_rad": float(minimum_abs_yaw_rad),
        "maximum_abs_yaw_rad": float(maximum_abs_yaw_rad),
        "original_action": original.tolist(),
    }
    if not active:
        audit["reason"] = "no_fresh_operator_direction"
        audit["execution_action"] = original.tolist()
        return original, audit

    distance_m = math.hypot(float(values[0]), float(values[1]))
    amplified_abs_yaw = abs(float(values[2])) * float(turn_gain)
    target_abs_yaw = clamp(
        max(amplified_abs_yaw, float(minimum_abs_yaw_rad)),
        float(minimum_abs_yaw_rad),
        float(maximum_abs_yaw_rad),
    )
    target_yaw = math.copysign(target_abs_yaw, float(direction))
    execution = np.asarray(
        [
            distance_m * math.cos(target_yaw),
            distance_m * math.sin(target_yaw),
            target_yaw,
        ],
        dtype=np.float32,
    )
    audit.update(
        {
            "reason": "fresh_annotation_turn_first_gain",
            "distance_m": float(distance_m),
            "model_abs_yaw_rad": abs(float(values[2])),
            "amplified_abs_yaw_rad": float(amplified_abs_yaw),
            "execution_abs_yaw_rad": float(target_abs_yaw),
            "execution_action": execution.tolist(),
        }
    )
    return execution, audit


def operator_guided_segmented_turn_then_drive(
    action: Sequence[float],
    *,
    enabled: bool,
    direction: int,
    turn_gain: float = 6.0,
    minimum_abs_yaw_rad: float = 0.08,
    maximum_abs_yaw_rad: float = 0.60,
    maximum_turn_segment_rad: float = 0.10,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Split an operator-guided turn into settle-separated yaw segments.

    Each yaw-only transition completes and settles before the next begins.
    The final transition drives straight in the newly established body
    heading, so no forward command can occur before the full guided turn.
    """

    guided, guided_audit = operator_guided_turn_first_action(
        action,
        enabled=enabled,
        direction=direction,
        turn_gain=turn_gain,
        minimum_abs_yaw_rad=minimum_abs_yaw_rad,
        maximum_abs_yaw_rad=maximum_abs_yaw_rad,
    )
    if not math.isfinite(float(maximum_turn_segment_rad)) or not (
        0.0 < maximum_turn_segment_rad <= maximum_abs_yaw_rad
    ):
        raise ValueError("invalid maximum turn segment")
    if not guided_audit["active"]:
        audit = dict(guided_audit)
        audit.update(
            {
                "schema": "v70_operator_segmented_turn_v1",
                "segmented": False,
                "prefix_steps": 1,
                "execution_prefix": [guided.tolist()],
            }
        )
        return guided.reshape(1, 3), audit

    total_yaw = float(guided[2])
    distance_m = math.hypot(float(guided[0]), float(guided[1]))
    segment_count = max(
        1,
        int(
            math.ceil(
                max(0.0, abs(total_yaw) - 1.0e-6)
                / maximum_turn_segment_rad
            )
        ),
    )
    segment_yaw = total_yaw / float(segment_count)
    prefix = np.zeros((segment_count + 1, 3), dtype=np.float32)
    prefix[:segment_count, 2] = segment_yaw
    prefix[segment_count, 0] = distance_m
    audit = dict(guided_audit)
    audit.update(
        {
            "schema": "v70_operator_segmented_turn_v1",
            "segmented": True,
            "reason": "fresh_annotation_segmented_turn_then_drive",
            "maximum_turn_segment_rad": float(maximum_turn_segment_rad),
            "turn_segment_count": int(segment_count),
            "turn_segment_yaw_rad": float(segment_yaw),
            "prefix_steps": int(prefix.shape[0]),
            "drive_after_all_turns": True,
            "execution_prefix": prefix.tolist(),
        }
    )
    return prefix, audit


def model_amplified_segmented_turn_then_drive(
    action: Sequence[float],
    *,
    enabled: bool,
    turn_gain: float = 6.0,
    trigger_abs_yaw_rad: float = 0.01,
    minimum_abs_yaw_rad: float = 0.08,
    maximum_abs_yaw_rad: float = 0.60,
    maximum_turn_segment_rad: float = 0.10,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Amplify the model's own turn without any operator annotation.

    Near-straight model actions remain unchanged. Once the model predicts a
    meaningful signed yaw, preserve that sign, apply the configured gain and
    finish settle-separated yaw segments before the original drive distance.
    """

    values = np.asarray(action, dtype=np.float64)
    if values.shape != (3,) or not np.isfinite(values).all():
        raise ValueError("model action must contain three finite values")
    if not math.isfinite(float(trigger_abs_yaw_rad)) or (
        trigger_abs_yaw_rad < 0.0
    ):
        raise ValueError("model turn trigger must be finite and non-negative")
    model_yaw = float(values[2])
    active = bool(enabled and abs(model_yaw) >= trigger_abs_yaw_rad)
    direction = 1 if model_yaw > 0.0 else -1 if model_yaw < 0.0 else 0
    prefix, inherited = operator_guided_segmented_turn_then_drive(
        values,
        enabled=active,
        direction=direction,
        turn_gain=turn_gain,
        minimum_abs_yaw_rad=minimum_abs_yaw_rad,
        maximum_abs_yaw_rad=maximum_abs_yaw_rad,
        maximum_turn_segment_rad=maximum_turn_segment_rad,
    )
    audit = dict(inherited)
    audit.update(
        {
            "schema": "v70_model_amplified_segmented_turn_v1",
            "active": active,
            "source": "world_model_first_action",
            "reason": (
                "model_yaw_gain_segmented_turn_then_drive"
                if active
                else "model_action_below_turn_trigger"
            ),
            "model_yaw_rad": model_yaw,
            "model_direction": direction,
            "trigger_abs_yaw_rad": float(trigger_abs_yaw_rad),
            "turn_gain": float(turn_gain),
        }
    )
    return prefix, audit


@dataclass(frozen=True)
class FollowerCommand:
    phase: FollowerPhase
    vx: float
    wz: float
    enable_motion: bool
    complete: bool
    fault: bool
    reason: str
    progress_m: float
    cross_track_m: float
    yaw_error_rad: float
    replan_required: bool = False

    @classmethod
    def zero(
        cls,
        phase: FollowerPhase,
        reason: str,
        *,
        complete: bool = False,
        fault: bool = False,
        progress_m: float = 0.0,
        cross_track_m: float = 0.0,
        yaw_error_rad: float = 0.0,
        replan_required: bool = False,
    ) -> "FollowerCommand":
        return cls(
            phase=phase,
            vx=0.0,
            wz=0.0,
            enable_motion=False,
            complete=complete,
            fault=fault,
            reason=reason,
            progress_m=progress_m,
            cross_track_m=cross_track_m,
            yaw_error_rad=yaw_error_rad,
            replan_required=replan_required,
        )


def should_continue_prefix_without_settle(
    command: FollowerCommand,
    *,
    pending_actions: int,
    interlock_allows_motion: bool,
    approach_mode: bool,
) -> bool:
    """Continue an exact V70 H4 prefix without an elastic recoil pause."""

    return bool(
        command.phase is FollowerPhase.SETTLE
        and command.reason == "drive_complete_settle_started"
        and pending_actions > 0
        and interlock_allows_motion
    )


def should_handoff_ready_plan_without_settle(
    command: FollowerCommand,
    *,
    interlock_allows_motion: bool,
    normal_drive_completion: bool,
) -> bool:
    """Use a ready, fully validated plan during the normal DRIVE settle."""

    return bool(
        command.phase is FollowerPhase.SETTLE
        and command.reason in {
            "drive_complete_settle_started",
            "settling",
        }
        and normal_drive_completion
        and not command.replan_required
        and interlock_allows_motion
    )


def visual_turn_override_allowed(
    *,
    confirmed_direction: int,
    center_x_norm: float | None,
    minimum_center_offset: float,
    motion_prefetch: bool,
    observed_scale_ratio: float | None = None,
    minimum_scale_ratio: float = 0.0,
) -> bool:
    """Require a material, sufficiently large fresh cue before replacing yaw."""

    if confirmed_direction not in (-1, 1) or motion_prefetch:
        return False
    threshold = float(minimum_center_offset)
    if not math.isfinite(threshold) or not 0.0 < threshold < 0.5:
        raise ValueError("minimum center offset must be finite and in (0, 0.5)")
    scale_threshold = float(minimum_scale_ratio)
    if not math.isfinite(scale_threshold) or scale_threshold < 0.0:
        raise ValueError("minimum scale ratio must be finite and nonnegative")
    if scale_threshold > 0.0:
        if observed_scale_ratio is None:
            return False
        scale = float(observed_scale_ratio)
        if not math.isfinite(scale) or scale < scale_threshold:
            return False
    if center_x_norm is None:
        return False
    center = float(center_x_norm)
    return bool(
        math.isfinite(center)
        and 0.0 <= center <= 1.0
        and abs(center - 0.5) >= threshold
    )


def far_target_model_reacquire_allowed(
    *,
    continuation_reason: str,
    planner_abstained: bool,
    trusted_scale_ratio: float | None,
    approach_scale_ratio: float = 0.30,
) -> bool:
    """Release a stale far visual lock after its bounded forward budget ends.

    ``forward_budget_exhausted`` is emitted only after the current H1 has
    passed the forward, curvature, and route-direction checks.  The caller can
    therefore execute that fresh H1 once, without reusing a tiny color blob or
    repeating several full plans from the same stationary pose.
    """

    approach = float(approach_scale_ratio)
    if not math.isfinite(approach) or approach <= 0.0:
        raise ValueError("approach scale ratio must be finite and positive")
    if continuation_reason != "forward_budget_exhausted" or planner_abstained:
        return False
    if trusted_scale_ratio is None:
        return False
    scale = float(trusted_scale_ratio)
    return bool(math.isfinite(scale) and 0.0 <= scale < approach)


def rolling_motion_prefetch_ready(
    command: FollowerCommand,
    *,
    action_elapsed_s: float,
    final_prefix_action: bool,
    buffered_prefix_action: bool = False,
    drive_trigger_progress_m: float,
    buffered_drive_trigger_progress_m: float = 0.04,
    turn_trigger_delay_s: float,
    maximum_drive_yaw_error_rad: float | None = None,
) -> bool:
    """Start inference after a stable final or gait-committed action moves."""

    elapsed = float(action_elapsed_s)
    drive_trigger = float(drive_trigger_progress_m)
    buffered_drive_trigger = float(buffered_drive_trigger_progress_m)
    turn_delay = float(turn_trigger_delay_s)
    if not all(
        math.isfinite(value) and value >= 0.0
        for value in (
            elapsed,
            drive_trigger,
            buffered_drive_trigger,
            turn_delay,
        )
    ):
        raise ValueError("rolling prefetch thresholds must be finite and nonnegative")
    drive_yaw_limit = None
    if maximum_drive_yaw_error_rad is not None:
        drive_yaw_limit = float(maximum_drive_yaw_error_rad)
        if not math.isfinite(drive_yaw_limit) or drive_yaw_limit < 0.0:
            raise ValueError(
                "rolling prefetch drive yaw limit must be finite and nonnegative"
            )
    if not (
        final_prefix_action or buffered_prefix_action
    ) or not command.enable_motion:
        return False
    if command.phase is FollowerPhase.DRIVE:
        trigger = (
            buffered_drive_trigger
            if buffered_prefix_action
            else drive_trigger
        )
        heading_stable = bool(
            drive_yaw_limit is None
            or (
                math.isfinite(float(command.yaw_error_rad))
                and abs(float(command.yaw_error_rad))
                <= drive_yaw_limit + 1.0e-9
            )
        )
        return bool(command.progress_m >= trigger and heading_stable)
    return bool(
        command.phase is FollowerPhase.TURN
        and elapsed >= turn_delay
    )


def short_buffered_motion_prefetch_allowed(
    actions: np.ndarray,
    *,
    approach_mode: bool,
    precision_goal_hold: bool,
    max_total_distance_m: float = 0.55,
    max_total_abs_yaw_rad: float = 0.30,
) -> tuple[bool, dict[str, Any]]:
    """Allow early prefetch only for a short, coherent two-action buffer."""

    values = np.asarray(actions, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("buffered prefetch actions must have shape [H,3]")
    if not np.isfinite(values).all():
        raise ValueError("buffered prefetch actions contain NaN/Inf")
    if not all(
        math.isfinite(float(value)) and float(value) > 0.0
        for value in (max_total_distance_m, max_total_abs_yaw_rad)
    ):
        raise ValueError("buffered prefetch limits must be finite and positive")

    total_distance = float(
        sum(math.hypot(float(row[0]), float(row[1])) for row in values)
    )
    total_abs_yaw = float(sum(abs(float(row[2])) for row in values))
    reason = "eligible"
    allowed = True
    if len(values) != 2:
        allowed = False
        reason = "requires_exact_h2"
    elif precision_goal_hold:
        allowed = False
        reason = "precision_goal_hold"
    elif total_distance > max_total_distance_m + 1.0e-9:
        allowed = False
        reason = "distance_limit"
    elif total_abs_yaw > max_total_abs_yaw_rad + 1.0e-9:
        allowed = False
        reason = "yaw_limit"
    return allowed, {
        "schema": "v70_short_buffered_motion_prefetch_v1",
        "allowed": allowed,
        "reason": reason,
        "prefix_steps": int(len(values)),
        "total_distance_m": total_distance,
        "total_abs_yaw_rad": total_abs_yaw,
        "max_total_distance_m": float(max_total_distance_m),
        "max_total_abs_yaw_rad": float(max_total_abs_yaw_rad),
    }


def bounded_inflight_forward_bridge_distance(
    *,
    normal_drive_completion: bool,
    motion_prefetch_inflight: bool,
    bridge_already_started: bool,
    translation_since_snapshot_m: float,
    approach_mode: bool,
    precision_goal_hold: bool,
    target_loss_holds: int,
    target_locked: bool,
    trusted_scale_ratio: float | None,
    nomad_temporal_distance: float | None,
    dino_cosine_distance: float | None,
    visual_valid: bool,
    visual_center_x_norm: float | None,
    yaw_error_rad: float,
    cross_track_m: float,
    safe_motion_applied: bool,
    max_bridge_distance_m: float = 0.10,
    rebase_motion_budget_m: float = 0.28,
    minimum_bridge_distance_m: float = 0.04,
    locked_scale_limit: float = 0.20,
    unobserved_temporal_minimum: float = 8.0,
    unobserved_dino_minimum: float = 0.48,
    maximum_visual_center_offset: float = 0.10,
    maximum_yaw_error_rad: float = 0.04,
    maximum_cross_track_m: float = 0.03,
) -> tuple[float | None, dict[str, Any]]:
    """Budget one straight far-target bridge while a motion plan is inflight."""

    numeric = (
        translation_since_snapshot_m,
        yaw_error_rad,
        cross_track_m,
        max_bridge_distance_m,
        rebase_motion_budget_m,
        minimum_bridge_distance_m,
        locked_scale_limit,
        unobserved_temporal_minimum,
        unobserved_dino_minimum,
        maximum_visual_center_offset,
        maximum_yaw_error_rad,
        maximum_cross_track_m,
    )
    if not all(math.isfinite(float(value)) for value in numeric):
        raise ValueError("forward bridge inputs must be finite")
    if target_loss_holds < 0:
        raise ValueError("target_loss_holds must be nonnegative")
    if not (
        0.0 < minimum_bridge_distance_m <= max_bridge_distance_m
        and max_bridge_distance_m < rebase_motion_budget_m
        and 0.0 < locked_scale_limit
        and 0.0 <= maximum_visual_center_offset < 0.5
        and maximum_yaw_error_rad >= 0.0
        and maximum_cross_track_m >= 0.0
    ):
        raise ValueError("forward bridge limits are invalid")

    audit: dict[str, Any] = {
        "schema": "v70_inflight_forward_bridge_v1",
        "allowed": False,
        "reason": "not_evaluated",
        "translation_since_snapshot_m": float(
            translation_since_snapshot_m
        ),
    }
    gates = (
        (normal_drive_completion, "not_normal_drive_completion"),
        (motion_prefetch_inflight, "motion_prefetch_not_inflight"),
        (not bridge_already_started, "already_started_for_plan"),
        (not approach_mode, "approach_mode"),
        (not precision_goal_hold, "precision_goal_hold"),
        (target_loss_holds == 0, "target_loss_active"),
        (safe_motion_applied, "safe_motion_not_applied"),
        (
            abs(float(yaw_error_rad)) <= maximum_yaw_error_rad + 1.0e-9,
            "yaw_error_limit",
        ),
        (
            abs(float(cross_track_m)) <= maximum_cross_track_m + 1.0e-9,
            "cross_track_limit",
        ),
    )
    for passed, reason in gates:
        if not passed:
            audit["reason"] = reason
            return None, audit

    if target_locked:
        far_evidence = bool(
            trusted_scale_ratio is not None
            and math.isfinite(float(trusted_scale_ratio))
            and float(trusted_scale_ratio) < locked_scale_limit
        )
        if not far_evidence:
            audit["reason"] = "locked_target_not_far"
            return None, audit
        audit["far_evidence"] = "locked_target_scale"
    else:
        far_evidence = bool(
            nomad_temporal_distance is not None
            and dino_cosine_distance is not None
            and math.isfinite(float(nomad_temporal_distance))
            and math.isfinite(float(dino_cosine_distance))
            and float(nomad_temporal_distance)
            >= unobserved_temporal_minimum
            and float(dino_cosine_distance) >= unobserved_dino_minimum
        )
        if not far_evidence:
            audit["reason"] = "unobserved_target_not_proven_far"
            return None, audit
        audit["far_evidence"] = "nomad_and_dino_far"

    if visual_valid:
        if (
            visual_center_x_norm is None
            or not math.isfinite(float(visual_center_x_norm))
            or abs(float(visual_center_x_norm) - 0.5)
            > maximum_visual_center_offset + 1.0e-9
        ):
            audit["reason"] = "visual_target_off_center"
            return None, audit

    available = min(
        max_bridge_distance_m,
        rebase_motion_budget_m - float(translation_since_snapshot_m),
    )
    audit["available_distance_m"] = float(max(0.0, available))
    if available < minimum_bridge_distance_m - 1.0e-9:
        audit["reason"] = "rebase_budget_exhausted"
        return None, audit
    distance = float(max(minimum_bridge_distance_m, available))
    audit.update(
        {
            "allowed": True,
            "reason": "bounded_far_target_bridge",
            "commanded_distance_m": distance,
        }
    )
    return distance, audit


def should_preserve_prefetched_plan_after_partial_completion(
    *,
    inflight_sequence: int | None,
    settle_prefetch: bool,
    motion_prefetch: bool,
) -> bool:
    """Keep prefetched work for the normal age, pose and authorization gates."""

    return bool(
        inflight_sequence is not None
        and (settle_prefetch or motion_prefetch)
    )


def fit_action_to_continuous_arc(
    action: Sequence[float],
    *,
    continuous_arc_max_vx: float,
    continuous_arc_min_wz: float,
    continuous_arc_turn_tolerance_rad: float,
    continuous_arc_drive_tolerance_m: float,
    continuous_arc_completion_fraction: float = SHORT_ACTION_COMPLETION_FRACTION,
    maximum_distance_m: float | None = None,
    maximum_abs_yaw_rad: float | None = None,
    distance_margin_m: float = 1.0e-4,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Reduce heading and yaw only enough for a forward-arc execution.

    The endpoint distance and turn sign are preserved except for an optional
    sub-millimetre contract margin at the maximum distance.  This mirrors the
    feasibility inequality in :meth:`ActionFollower.start`, so a short steering
    action does not fall back to pure TURN.
    """

    values = np.asarray(action, dtype=np.float64)
    if values.shape != (3,) or not np.isfinite(values).all():
        raise ValueError("continuous arc action must be one finite SE(2) transition")
    limits = (
        continuous_arc_max_vx,
        continuous_arc_min_wz,
        continuous_arc_turn_tolerance_rad,
        continuous_arc_drive_tolerance_m,
        continuous_arc_completion_fraction,
    )
    if not all(math.isfinite(float(value)) and float(value) > 0.0 for value in limits):
        raise ValueError("continuous arc limits must be finite and positive")
    if not 0.0 < continuous_arc_completion_fraction < 1.0:
        raise ValueError("continuous arc completion fraction must be in (0, 1)")
    margin = float(distance_margin_m)
    if not math.isfinite(margin) or margin < 0.0:
        raise ValueError("continuous arc distance margin must be nonnegative")
    maximum_distance = None
    if maximum_distance_m is not None:
        maximum_distance = float(maximum_distance_m)
        if not math.isfinite(maximum_distance) or maximum_distance <= margin:
            raise ValueError("continuous arc maximum distance is invalid")
    maximum_abs_yaw = None
    if maximum_abs_yaw_rad is not None:
        maximum_abs_yaw = float(maximum_abs_yaw_rad)
        if not math.isfinite(maximum_abs_yaw) or maximum_abs_yaw <= 0.0:
            raise ValueError("continuous arc maximum yaw is invalid")

    dx, dy, dyaw = (float(value) for value in values)
    distance = math.hypot(dx, dy)
    heading = math.atan2(dy, dx) if distance > 1.0e-9 else dyaw
    output_distance = (
        min(distance, maximum_distance - margin)
        if maximum_distance is not None
        else distance
    )
    completion_tolerance = min(
        continuous_arc_drive_tolerance_m,
        continuous_arc_completion_fraction * output_distance,
    )
    usable_forward_distance = max(0.0, output_distance - completion_tolerance)
    sine_limit = clamp(
        math.sin(continuous_arc_turn_tolerance_rad)
        + usable_forward_distance
        * continuous_arc_min_wz
        / continuous_arc_max_vx,
        -1.0,
        1.0,
    )
    if output_distance <= continuous_arc_drive_tolerance_m:
        # A sub-tolerance translation must never turn in place. Keep its yaw
        # strictly below TURN entry so the caller can discard/promote the noop.
        feasible_abs_yaw = max(
            0.0,
            continuous_arc_turn_tolerance_rad - 1.0e-4,
        )
    else:
        feasible_abs_yaw = max(
            continuous_arc_turn_tolerance_rad,
            math.asin(sine_limit) - 1.0e-4,
        )
    if maximum_abs_yaw is not None:
        feasible_abs_yaw = min(feasible_abs_yaw, maximum_abs_yaw)
    curvature_scale = (
        min(1.0, feasible_abs_yaw / abs(dyaw))
        if abs(dyaw) > 1.0e-9
        else 1.0
    )
    fitted_heading = heading * curvature_scale
    fitted_dyaw = dyaw * curvature_scale
    fitted = np.asarray(
        [
            output_distance * math.cos(fitted_heading),
            output_distance * math.sin(fitted_heading),
            fitted_dyaw,
        ],
        dtype=np.float32,
    )
    return fitted, {
        "schema": "v70_continuous_arc_fit_v1",
        "input_action": values.tolist(),
        "output_action": fitted.tolist(),
        "input_distance_m": distance,
        "output_distance_m": output_distance,
        "maximum_distance_m": maximum_distance,
        "maximum_abs_yaw_rad": maximum_abs_yaw,
        "distance_margin_m": margin,
        "input_heading_rad": heading,
        "output_heading_rad": fitted_heading,
        "input_dyaw_rad": dyaw,
        "output_dyaw_rad": fitted_dyaw,
        "curvature_scale": curvature_scale,
        "completion_tolerance_m": completion_tolerance,
        "max_feasible_abs_yaw_rad": feasible_abs_yaw,
    }


def bounded_target_loss_forward_action(
    action: Sequence[float],
    *,
    planner_abstained: bool,
    trusted_scale_ratio: float | None,
    accumulated_distance_m: float,
    route_direction: int,
    far_step_limit_m: float = 0.14,
    far_total_limit_m: float = 0.24,
    approach_step_limit_m: float = 0.06,
    approach_total_limit_m: float = 0.08,
    approach_scale_ratio: float = 0.30,
    stop_scale_ratio: float = 0.48,
    max_abs_heading_rad: float = 0.20,
    approach_max_abs_heading_rad: float = 0.12,
    max_heading_mismatch_rad: float = 0.08,
    minimum_distance_m: float = 0.04,
    continuous_arc_max_vx: float = 0.20,
    continuous_arc_min_wz: float = 0.35,
    continuous_arc_turn_tolerance_rad: float = 0.05,
    continuous_arc_drive_tolerance_m: float = 0.025,
    continuous_arc_completion_fraction: float = SHORT_ACTION_COMPLETION_FRACTION,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Bound a fresh low-curvature H1 while a locked target is missed.

    The recovery never invents a search turn. It can only shorten the current
    model action and preserve its turn direction, advancing the camera along
    the last trusted route while the next plan is computed. If the remaining
    distance budget is short, curvature is reduced enough to keep the follower
    in its forward-arc phase instead of falling back to a stationary turn.
    """

    values = np.asarray(action, dtype=np.float64)
    audit: dict[str, Any] = {
        "schema": "v70_target_loss_forward_continuation_v1",
        "active": False,
        "reason": "not_evaluated",
        "accumulated_distance_m": float(accumulated_distance_m),
        "trusted_scale_ratio": trusted_scale_ratio,
        "route_direction": int(route_direction),
    }
    if values.shape != (3,) or not np.isfinite(values).all():
        audit["reason"] = "invalid_h1"
        return None, audit
    if route_direction not in (-1, 0, 1):
        raise ValueError("route_direction must be -1, 0, or 1")
    numeric_limits = (
        accumulated_distance_m,
        far_step_limit_m,
        far_total_limit_m,
        approach_step_limit_m,
        approach_total_limit_m,
        approach_scale_ratio,
        stop_scale_ratio,
        max_abs_heading_rad,
        approach_max_abs_heading_rad,
        max_heading_mismatch_rad,
        minimum_distance_m,
        continuous_arc_max_vx,
        continuous_arc_min_wz,
        continuous_arc_turn_tolerance_rad,
        continuous_arc_drive_tolerance_m,
        continuous_arc_completion_fraction,
    )
    if not all(
        math.isfinite(float(value)) and float(value) >= 0.0
        for value in numeric_limits
    ):
        raise ValueError(
            "target-loss continuation limits must be finite and nonnegative"
        )
    if not 0.0 < approach_scale_ratio < stop_scale_ratio:
        raise ValueError("target-loss scale thresholds are invalid")
    if approach_max_abs_heading_rad > max_abs_heading_rad:
        raise ValueError(
            "approach target-loss heading limit must not exceed far limit"
        )
    if (
        continuous_arc_max_vx <= 0.0
        or continuous_arc_min_wz <= 0.0
        or continuous_arc_turn_tolerance_rad <= 0.0
        or continuous_arc_drive_tolerance_m <= 0.0
        or not 0.0 < continuous_arc_completion_fraction < 1.0
    ):
        raise ValueError("target-loss continuous arc limits are invalid")
    if planner_abstained:
        audit["reason"] = "planner_abstained"
        return None, audit
    if trusted_scale_ratio is None or not math.isfinite(float(trusted_scale_ratio)):
        audit["reason"] = "trusted_scale_unavailable"
        return None, audit
    scale = float(trusted_scale_ratio)
    if scale >= stop_scale_ratio:
        audit["reason"] = "target_too_close_for_blind_forward_motion"
        return None, audit

    approach = scale >= approach_scale_ratio
    effective_heading_limit = (
        approach_max_abs_heading_rad if approach else max_abs_heading_rad
    )
    dx, dy, dyaw = (float(value) for value in values)
    distance = math.hypot(dx, dy)
    heading = math.atan2(dy, dx) if distance > 1.0e-9 else dyaw
    audit.update(
        {
            "input_action": values.tolist(),
            "input_distance_m": distance,
            "heading_rad": heading,
            "dyaw_rad": dyaw,
        }
    )
    if dx <= 0.0 or distance < minimum_distance_m:
        audit["reason"] = "h1_not_forward"
        return None, audit
    if (
        abs(heading) > effective_heading_limit + 1.0e-9
        or abs(dyaw) > effective_heading_limit + 1.0e-9
        or abs(wrap_angle(heading - dyaw)) > max_heading_mismatch_rad
    ):
        audit["reason"] = "h1_curvature_too_large"
        return None, audit
    if route_direction and (
        route_direction * dy < -0.01
        or route_direction * dyaw < -0.03
    ):
        audit["reason"] = "h1_opposes_route_memory"
        return None, audit

    step_limit = approach_step_limit_m if approach else far_step_limit_m
    total_limit = approach_total_limit_m if approach else far_total_limit_m
    remaining_budget = max(0.0, total_limit - float(accumulated_distance_m))
    commanded_distance = min(distance, step_limit, remaining_budget)
    audit.update(
        {
            "approach_budget": approach,
            "effective_heading_limit_rad": effective_heading_limit,
            "step_limit_m": step_limit,
            "total_limit_m": total_limit,
            "remaining_budget_m": remaining_budget,
        }
    )
    if commanded_distance < minimum_distance_m - 1.0e-9:
        audit["reason"] = "forward_budget_exhausted"
        return None, audit

    scale_factor = commanded_distance / distance

    bounded, continuous_arc_fit = fit_action_to_continuous_arc(
        [
            commanded_distance * math.cos(heading),
            commanded_distance * math.sin(heading),
            dyaw,
        ],
        continuous_arc_max_vx=continuous_arc_max_vx,
        continuous_arc_min_wz=continuous_arc_min_wz,
        continuous_arc_turn_tolerance_rad=(
            continuous_arc_turn_tolerance_rad
        ),
        continuous_arc_drive_tolerance_m=(
            continuous_arc_drive_tolerance_m
        ),
        continuous_arc_completion_fraction=(
            continuous_arc_completion_fraction
        ),
    )
    curvature_scale = float(continuous_arc_fit["curvature_scale"])
    bounded_heading = float(continuous_arc_fit["output_heading_rad"])
    bounded_dyaw = float(continuous_arc_fit["output_dyaw_rad"])
    completion_tolerance = float(
        continuous_arc_fit["completion_tolerance_m"]
    )
    feasible_abs_yaw = float(
        continuous_arc_fit["max_feasible_abs_yaw_rad"]
    )
    bounded_heading_mismatch = abs(
        wrap_angle(bounded_heading - bounded_dyaw)
    )
    if (
        abs(bounded_heading) > effective_heading_limit + 1.0e-9
        or abs(bounded_dyaw) > effective_heading_limit + 1.0e-9
        or bounded_heading_mismatch > max_heading_mismatch_rad + 1.0e-9
    ):
        audit["reason"] = "bounded_h1_curvature_too_large"
        return None, audit
    if route_direction and (
        route_direction * float(bounded[1]) < -0.01
        or route_direction * bounded_dyaw < -0.03
    ):
        audit["reason"] = "bounded_h1_opposes_route_memory"
        return None, audit
    audit.update(
        {
            "active": True,
            "reason": "bounded_fresh_h1",
            "output_action": bounded.tolist(),
            "commanded_distance_m": commanded_distance,
            "distance_scale": scale_factor,
            "curvature_scale": curvature_scale,
            "bounded_heading_rad": bounded_heading,
            "bounded_dyaw_rad": bounded_dyaw,
            "continuous_arc_completion_tolerance_m": completion_tolerance,
            "continuous_arc_max_feasible_abs_yaw_rad": feasible_abs_yaw,
            "continuous_arc_fit": continuous_arc_fit,
        }
    )
    return bounded, audit


def target_loss_budget_after_prefetch_motion(
    *,
    accumulated_distance_m: float,
    snapshot_translation_m: float,
    motion_prefetch: bool,
    previous_continuity_allowed: bool,
) -> tuple[float, dict[str, Any]]:
    """Charge motion after the first snapshot that no longer sees a target."""

    accumulated = float(accumulated_distance_m)
    translation = float(snapshot_translation_m)
    if not (
        math.isfinite(accumulated)
        and accumulated >= 0.0
        and math.isfinite(translation)
        and translation >= 0.0
    ):
        raise ValueError("target-loss motion budget inputs must be nonnegative")
    charged = bool(motion_prefetch and previous_continuity_allowed)
    added = translation if charged else 0.0
    total = accumulated + added
    return total, {
        "schema": "v70_target_loss_prefetch_motion_budget_v1",
        "charged": charged,
        "previous_continuity_allowed": bool(previous_continuity_allowed),
        "motion_prefetch": bool(motion_prefetch),
        "previous_accumulated_distance_m": accumulated,
        "snapshot_translation_m": translation,
        "added_distance_m": added,
        "updated_accumulated_distance_m": total,
    }


def safe_velocity_feedback_applied(
    requested: FollowerCommand,
    *,
    safe_vx: float,
    safe_wz: float,
    feedback_fresh: bool,
    epsilon: float = 1.0e-3,
) -> bool:
    """Return whether the safe bridge is applying the requested phase."""

    if (
        not feedback_fresh
        or not requested.enable_motion
        or not math.isfinite(float(safe_vx))
        or not math.isfinite(float(safe_wz))
        or not math.isfinite(float(epsilon))
        or epsilon <= 0.0
    ):
        return False
    if requested.phase is FollowerPhase.TURN:
        return bool(
            abs(requested.wz) > epsilon
            and abs(safe_wz) > epsilon
            and requested.wz * safe_wz > 0.0
        )
    if requested.phase is FollowerPhase.DRIVE:
        forward_applied = bool(
            requested.vx > epsilon
            and safe_vx > epsilon
        )
        # Safe Bridge owns the final steering limit.  During a DRIVE arc its
        # yaw slew can cross zero later than the raw request while forward gait
        # remains authorized and physically applied.  Treat that as DRIVE
        # motion; odometry heading and cross-track guards still supervise it.
        return forward_applied
    return False


class ActionFollower:
    """Execute one selected V70 transition with odometry feedback."""

    def __init__(self, config: ActionFollowerConfig) -> None:
        config.validate()
        self.config = config
        self.phase = FollowerPhase.IDLE
        self.reason = "not_started"
        self._start_pose: Pose2D | None = None
        self._target_yaw = 0.0
        self._target_distance_m = 0.0
        self._phase_started = 0.0
        self._last_progress_time = 0.0
        self._best_progress = 0.0
        self._initial_turn_error = 0.0
        self._turn_confirmations = 0
        self._drive_realign_confirmations = 0
        self._last_update_time = 0.0
        self._motion_blocked_since: float | None = None
        self._settle_next_phase: FollowerPhase | None = None
        self._completion_reason = "action_complete"
        self._completion_requires_replan = False
        self._drive_gait_committed = False
        self._settle_completion_tolerance_m = 0.0
        self._continuous_arc_active = False
        self._continuous_arc_steering_direction = 0

    @property
    def active(self) -> bool:
        return self.phase in {
            FollowerPhase.TURN,
            FollowerPhase.TURN_SETTLE,
            FollowerPhase.DRIVE,
            FollowerPhase.SETTLE,
        }

    def reset(self, reason: str = "reset") -> None:
        self.phase = FollowerPhase.IDLE
        self.reason = reason
        self._start_pose = None
        self._turn_confirmations = 0
        self._drive_realign_confirmations = 0
        self._motion_blocked_since = None
        self._settle_next_phase = None
        self._completion_reason = "action_complete"
        self._completion_requires_replan = False
        self._drive_gait_committed = False
        self._continuous_arc_active = False
        self._continuous_arc_steering_direction = 0
        self._settle_completion_tolerance_m = 0.0

    def _fault(
        self,
        reason: str,
        *,
        progress_m: float = 0.0,
        cross_track_m: float = 0.0,
        yaw_error_rad: float = 0.0,
    ) -> FollowerCommand:
        self.phase = FollowerPhase.FAULT
        self.reason = reason
        return FollowerCommand.zero(
            self.phase,
            reason,
            fault=True,
            progress_m=progress_m,
            cross_track_m=cross_track_m,
            yaw_error_rad=yaw_error_rad,
        )

    def start(
        self,
        action: Sequence[float],
        pose: Pose2D,
        now: float,
    ) -> FollowerCommand:
        pose.validate()
        values = np.asarray(action, dtype=np.float64)
        if values.shape != (3,) or not np.isfinite(values).all():
            return self._fault("invalid_selected_action")
        if not math.isfinite(float(now)):
            return self._fault("invalid_start_time")

        dx, dy, dyaw = (float(value) for value in values)
        distance = math.hypot(dx, dy)
        heading = math.atan2(dy, dx) if distance > 1.0e-6 else dyaw
        heading_error = abs(wrap_angle(heading - dyaw))

        if dx < -1.0e-4:
            return self._fault("reverse_action_not_supported")
        if distance > self.config.max_action_distance_m:
            return self._fault("action_distance_exceeds_executor_contract")
        if abs(dyaw) > self.config.max_abs_action_yaw_rad:
            return self._fault("action_yaw_exceeds_executor_contract")
        if (
            distance > self.config.drive_tolerance_m
            and heading_error > self.config.heading_consistency_rad
        ):
            return self._fault("action_heading_contract_mismatch")

        self._start_pose = pose
        self._target_yaw = wrap_angle(pose.yaw + dyaw)
        self._target_distance_m = distance
        self._phase_started = float(now)
        self._last_progress_time = float(now)
        self._best_progress = 0.0
        self._initial_turn_error = abs(
            wrap_angle(self._target_yaw - pose.yaw)
        )
        self._turn_confirmations = 0
        self._drive_realign_confirmations = 0
        self._last_update_time = float(now)
        self._motion_blocked_since = None
        self._completion_reason = "action_complete"
        self._completion_requires_replan = False
        self._drive_gait_committed = False
        self._settle_completion_tolerance_m = 0.0
        arc_completion_yaw_tolerance = (
            self._continuous_arc_completion_yaw_tolerance()
        )
        continuous_arc_forward_requirement = max(
            0.0,
            self.config.continuous_forward_arc_max_vx
            / max(self.config.drive_min_wz, 1.0e-9)
            * (
                math.sin(self._initial_turn_error)
                - math.sin(arc_completion_yaw_tolerance)
            ),
        )
        continuous_arc_completion_tolerance = (
            self._drive_completion_tolerance()
        )
        self._continuous_arc_active = bool(
            distance > self.config.drive_tolerance_m
            and self.config.continuous_forward_arc
            and self._initial_turn_error
            <= self.config.continuous_forward_arc_max_yaw_rad + 1.0e-7
            and continuous_arc_forward_requirement
            + continuous_arc_completion_tolerance
            <= distance
        )
        initial_yaw_error = wrap_angle(self._target_yaw - pose.yaw)
        self._continuous_arc_steering_direction = (
            1
            if self._continuous_arc_active
            and initial_yaw_error > self.config.drive_steering_deadband_rad
            else (
                -1
                if self._continuous_arc_active
                and initial_yaw_error < -self.config.drive_steering_deadband_rad
                else 0
            )
        )
        self.reason = "action_started"

        if self._continuous_arc_active:
            self.phase = FollowerPhase.DRIVE
            return FollowerCommand.zero(
                self.phase,
                "continuous_arc_drive_started",
                yaw_error_rad=wrap_angle(
                    self._target_yaw - pose.yaw
                ),
            )
        if self._initial_turn_error > self.config.turn_tolerance_rad:
            self.phase = FollowerPhase.TURN
            return FollowerCommand.zero(
                self.phase,
                "turn_phase_started",
                yaw_error_rad=wrap_angle(
                    self._target_yaw - pose.yaw
                ),
            )
        if distance > self.config.drive_tolerance_m:
            self.phase = FollowerPhase.DRIVE
            return FollowerCommand.zero(
                self.phase,
                "drive_phase_started",
            )
        self.phase = FollowerPhase.SETTLE
        return FollowerCommand.zero(
            self.phase,
            "zero_length_action_settle",
        )

    def _pose_progress(
        self,
        pose: Pose2D,
    ) -> tuple[float, float]:
        assert self._start_pose is not None
        delta_x = pose.x - self._start_pose.x
        delta_y = pose.y - self._start_pose.y
        cosine = math.cos(self._target_yaw)
        sine = math.sin(self._target_yaw)
        forward = cosine * delta_x + sine * delta_y
        cross_track = -sine * delta_x + cosine * delta_y
        return forward, cross_track

    @staticmethod
    def _signed_minimum(
        value: float,
        minimum: float,
        maximum: float,
    ) -> float:
        if abs(value) <= 1.0e-12:
            return 0.0
        magnitude = clamp(abs(value), minimum, maximum)
        return math.copysign(magnitude, value)

    def _turn_stall_tolerance_rad(self) -> float:
        return max(
            self.config.turn_tolerance_rad,
            min(
                self.config.heading_consistency_rad,
                self.config.turn_tolerance_rad
                + self.config.turn_stall_margin_rad,
            ),
        )

    def telemetry(
        self,
        pose: Pose2D,
        now: float,
    ) -> dict[str, Any]:
        """Return control evidence without mutating follower state."""

        pose.validate()
        if not math.isfinite(float(now)):
            raise ValueError("invalid telemetry time")
        yaw_error = wrap_angle(self._target_yaw - pose.yaw)
        turn_progress = self._initial_turn_error - abs(yaw_error)
        return {
            "phase": self.phase.value,
            "reason": self.reason,
            "odom_yaw_rad": float(pose.yaw),
            "target_yaw_rad": float(self._target_yaw),
            "initial_turn_error_rad": float(
                self._initial_turn_error
            ),
            "yaw_error_rad": float(yaw_error),
            "turn_progress_rad": float(turn_progress),
            "best_turn_progress_rad": float(self._best_progress),
            "turn_tolerance_rad": float(
                self.config.turn_tolerance_rad
            ),
            "turn_stall_tolerance_rad": float(
                self._turn_stall_tolerance_rad()
            ),
            "phase_motion_elapsed_s": float(
                max(0.0, now - self._phase_started)
            ),
            "no_progress_motion_elapsed_s": float(
                max(0.0, now - self._last_progress_time)
            ),
            "turn_confirmations": int(self._turn_confirmations),
            "drive_realign_confirmations": int(
                self._drive_realign_confirmations
            ),
            "drive_realign_threshold_rad": float(
                self.config.drive_realign_threshold_rad
            ),
            "continuous_arc_active": bool(self._continuous_arc_active),
            "continuous_arc_steering_direction": int(
                self._continuous_arc_steering_direction
            ),
            "drive_gait_committed": bool(self._drive_gait_committed),
            "drive_gait_commit_timeout_s": float(
                self.config.drive_gait_commit_timeout_s
            ),
            "drive_gait_commit_min_progress_m": float(
                self.config.drive_gait_commit_min_progress_m
            ),
            "motion_blocked_wall_s": (
                None
                if self._motion_blocked_since is None
                else float(
                    max(0.0, now - self._motion_blocked_since)
                )
            ),
        }

    def _turn_arc_vx(self, pose: Pose2D) -> float:
        """Return a bounded forward assist while the Go2 is turning.

        A small forward arc helps the elastic-foot platform replace support
        contacts when pure in-place yaw stalls. It is capped by odometry
        distance and never changes the selected V70 action.
        """

        if (
            self.config.turn_arc_vx <= 0.0
            or self.config.turn_arc_max_distance_m <= 0.0
            or self._start_pose is None
        ):
            return 0.0
        traveled = math.hypot(
            pose.x - self._start_pose.x,
            pose.y - self._start_pose.y,
        )
        if traveled >= self.config.turn_arc_max_distance_m:
            return 0.0
        return self.config.turn_arc_vx

    def _rebase_drive_after_turn(self, pose: Pose2D) -> None:
        """Exclude the bounded turn-assist arc from the drive distance."""

        if self.config.turn_arc_vx <= 0.0 or self._start_pose is None:
            return
        if self._target_distance_m <= self.config.drive_tolerance_m:
            # A segmented yaw-only transition must remain yaw-only.  The
            # support-contact arc is incidental motion, not a new reverse or
            # forward drive target to be compensated after the turn.
            self._target_distance_m = 0.0
            self._start_pose = pose
            return
        forward, _ = self._pose_progress(pose)
        self._target_distance_m = max(
            self.config.drive_tolerance_m,
            self._target_distance_m - max(0.0, forward),
        )
        self._start_pose = pose

    def _complete_turn(
        self,
        now: float,
        yaw_error: float,
        *,
        reason_prefix: str,
        pose: Pose2D,
    ) -> FollowerCommand:
        self._rebase_drive_after_turn(pose)
        self._best_progress = 0.0
        self._last_progress_time = now
        self._phase_started = now
        self._motion_blocked_since = None
        if self._target_distance_m > self.config.drive_tolerance_m:
            self._settle_next_phase = FollowerPhase.DRIVE
            self.phase = FollowerPhase.TURN_SETTLE
            self.reason = (
                f"{reason_prefix}_stability_settle_started"
            )
            return FollowerCommand.zero(
                self.phase,
                self.reason,
                yaw_error_rad=yaw_error,
            )
        self._settle_next_phase = None
        self.phase = FollowerPhase.SETTLE
        self.reason = f"{reason_prefix}_settle_started"
        return FollowerCommand.zero(
            self.phase,
            self.reason,
            yaw_error_rad=yaw_error,
        )

    def _start_drive_replan_settle(
        self,
        now: float,
        *,
        completion_reason: str,
        progress_m: float,
        cross_track_m: float,
        yaw_error_rad: float,
    ) -> FollowerCommand:
        """Stop cleanly before replanning after useful physical progress."""

        self.phase = FollowerPhase.SETTLE
        self.reason = f"{completion_reason}_settle_started"
        self._completion_reason = completion_reason
        self._completion_requires_replan = True
        self._phase_started = now
        self._motion_blocked_since = None
        return FollowerCommand.zero(
            self.phase,
            self.reason,
            progress_m=progress_m,
            cross_track_m=cross_track_m,
            yaw_error_rad=yaw_error_rad,
        )

    def _continuous_arc_completion_yaw_tolerance(self) -> float:
        configured = (
            self.config.continuous_forward_arc_completion_yaw_tolerance_rad
        )
        return float(
            configured if configured > 0.0 else self.config.turn_tolerance_rad
        )

    def _drive_completion_tolerance(self) -> float:
        configured = self.config.drive_completion_tolerance_m
        tolerance = (
            configured
            if configured > 0.0
            else self.config.drive_tolerance_m
        )
        if self._target_distance_m > self.config.drive_tolerance_m:
            tolerance = min(
                tolerance,
                SHORT_ACTION_COMPLETION_FRACTION
                * self._target_distance_m,
            )
        return float(tolerance)

    def _start_normal_drive_settle(
        self,
        now: float,
        *,
        progress_m: float,
        cross_track_m: float,
        yaw_error_rad: float,
        completion_tolerance_m: float | None = None,
    ) -> FollowerCommand:
        self.phase = FollowerPhase.SETTLE
        self.reason = "drive_complete_settle_started"
        self._completion_reason = "action_complete"
        self._completion_requires_replan = False
        self._phase_started = now
        self._motion_blocked_since = None
        self._settle_completion_tolerance_m = float(
            0.0
            if completion_tolerance_m is None
            else max(0.0, completion_tolerance_m)
        )
        return FollowerCommand.zero(
            self.phase,
            self.reason,
            progress_m=progress_m,
            cross_track_m=cross_track_m,
            yaw_error_rad=yaw_error_rad,
        )

    def update(
        self,
        pose: Pose2D,
        now: float,
        *,
        interlock_allows_motion: bool,
        motion_applied: bool = True,
    ) -> FollowerCommand:
        if not interlock_allows_motion:
            self.reset("motion_interlock_closed")
            return FollowerCommand.zero(
                self.phase,
                "motion_interlock_closed",
            )
        if self.phase is FollowerPhase.IDLE:
            return FollowerCommand.zero(self.phase, self.reason)
        if self.phase is FollowerPhase.COMPLETE:
            return FollowerCommand.zero(
                self.phase,
                self.reason,
                complete=True,
                replan_required=self._completion_requires_replan,
            )
        if self.phase is FollowerPhase.FAULT:
            return FollowerCommand.zero(
                self.phase,
                self.reason,
                fault=True,
            )

        try:
            pose.validate()
        except ValueError:
            return self._fault("invalid_odom")
        if not math.isfinite(float(now)):
            return self._fault("invalid_control_time")
        if (
            now < pose.receive_time
            or now - pose.receive_time > self.config.max_odom_age_s
        ):
            return self._fault("stale_odom")
        if now < self._last_update_time:
            return self._fault("control_time_regression")

        elapsed_since_update = now - self._last_update_time
        self._last_update_time = now

        yaw_error = wrap_angle(self._target_yaw - pose.yaw)
        progress_m, cross_track_m = self._pose_progress(pose)
        drive_completion_tolerance = self._drive_completion_tolerance()
        planned_endpoint_hold = bool(
            self.phase is FollowerPhase.DRIVE
            and self._target_distance_m - progress_m
            <= drive_completion_tolerance
            and (
                not self._continuous_arc_active
                or abs(yaw_error)
                <= self._continuous_arc_completion_yaw_tolerance()
            )
        )
        motion_blocked = bool(
            self.phase in {FollowerPhase.TURN, FollowerPhase.DRIVE}
            and not motion_applied
            and not planned_endpoint_hold
        )
        if motion_blocked:
            if self._motion_blocked_since is None:
                self._motion_blocked_since = now
            # Count action and no-progress time only while the independent
            # safety bridge actually applies a non-zero command.
            self._phase_started += elapsed_since_update
            self._last_progress_time += elapsed_since_update
        else:
            self._motion_blocked_since = None

        if self.phase is FollowerPhase.TURN_SETTLE:
            if (
                now - self._phase_started
                >= self.config.turn_settle_time_s
            ):
                if self._settle_next_phase is FollowerPhase.DRIVE:
                    self._settle_next_phase = None
                    self.phase = FollowerPhase.DRIVE
                    self.reason = "turn_stability_settle_complete"
                    self._phase_started = now
                    self._last_progress_time = now
                    self._best_progress = 0.0
                    self._drive_gait_committed = False
                    self._drive_realign_confirmations = 0
                    return FollowerCommand.zero(
                        self.phase,
                        self.reason,
                        yaw_error_rad=yaw_error,
                    )
                return self._fault("turn_settle_target_invalid")
            return FollowerCommand.zero(
                self.phase,
                "turn_stability_settling",
                yaw_error_rad=yaw_error,
            )

        if self.phase is FollowerPhase.TURN:
            if now - self._phase_started > self.config.turn_timeout_s:
                return self._fault(
                    "turn_timeout",
                    yaw_error_rad=yaw_error,
                )
            turn_progress = self._initial_turn_error - abs(yaw_error)
            if (
                turn_progress
                >= self._best_progress
                + self.config.progress_epsilon_rad
            ):
                self._best_progress = turn_progress
                self._last_progress_time = now
            turn_motion_time_ok = bool(
                now - self._phase_started
                >= self.config.minimum_turn_motion_time_s
            )
            if (
                abs(yaw_error) <= self.config.turn_tolerance_rad
                and turn_motion_time_ok
            ):
                self._turn_confirmations += 1
                if (
                    self._turn_confirmations
                    >= self.config.required_turn_confirmations
                ):
                    return self._complete_turn(
                        now,
                        yaw_error,
                        reason_prefix="turn_complete",
                        pose=pose,
                    )
            else:
                self._turn_confirmations = 0

            raw_wz = self.config.yaw_kp * yaw_error
            wz = self._signed_minimum(
                raw_wz,
                self.config.min_wz,
                self.config.max_wz,
            )
            turn_vx = self._turn_arc_vx(pose)
            if motion_blocked:
                assert self._motion_blocked_since is not None
                if (
                    now - self._motion_blocked_since
                    > self.config.safety_block_timeout_s
                ):
                    return self._fault(
                        "turn_safety_block_timeout",
                        yaw_error_rad=yaw_error,
                    )
                return FollowerCommand(
                    phase=self.phase,
                    vx=turn_vx,
                    wz=wz,
                    enable_motion=True,
                    complete=False,
                    fault=False,
                    reason="turn_waiting_for_safe_output",
                    progress_m=0.0,
                    cross_track_m=0.0,
                    yaw_error_rad=yaw_error,
                )
            if (
                now - self._last_progress_time
                > self.config.no_progress_timeout_s
            ):
                if (
                    self._best_progress
                    >= self.config.progress_epsilon_rad
                    and abs(yaw_error)
                    <= self._turn_stall_tolerance_rad()
                ):
                    return self._complete_turn(
                        now,
                        yaw_error,
                        reason_prefix="turn_stall_near_target",
                        pose=pose,
                    )
                return self._fault(
                    "turn_no_progress",
                    yaw_error_rad=yaw_error,
                )
            return FollowerCommand(
                phase=self.phase,
                vx=turn_vx,
                wz=wz,
                enable_motion=True,
                complete=False,
                fault=False,
                reason="turn_tracking",
                progress_m=0.0,
                cross_track_m=0.0,
                yaw_error_rad=yaw_error,
            )

        if self.phase is FollowerPhase.DRIVE:
            if now - self._phase_started > self.config.drive_timeout_s:
                return self._fault(
                    "drive_timeout",
                    progress_m=progress_m,
                    cross_track_m=cross_track_m,
                    yaw_error_rad=yaw_error,
                )
            if abs(cross_track_m) > self.config.max_cross_track_m:
                return self._fault(
                    "drive_cross_track_exceeded",
                    progress_m=progress_m,
                    cross_track_m=cross_track_m,
                    yaw_error_rad=yaw_error,
                )
            continuous_arc_tracks_heading = bool(
                self._continuous_arc_active
                and abs(yaw_error)
                <= self.config.continuous_forward_arc_max_yaw_rad
                + self.config.drive_steering_deadband_rad
                + 1.0e-9
            )
            if (
                abs(yaw_error) > self.config.drive_realign_threshold_rad
                and not continuous_arc_tracks_heading
            ):
                self._drive_realign_confirmations += 1
                if (
                    self._drive_realign_confirmations
                    >= self.config.required_drive_realign_confirmations
                ):
                    if self._continuous_arc_active:
                        return self._start_drive_replan_settle(
                            now,
                            completion_reason=(
                                "continuous_arc_heading_envelope_replan"
                            ),
                            progress_m=progress_m,
                            cross_track_m=cross_track_m,
                            yaw_error_rad=yaw_error,
                        )
                    # Stop before correcting a sustained heading error.
                    # Re-enter pure TURN only after hysteresis confirmation.
                    self.phase = FollowerPhase.TURN
                    self.reason = "drive_heading_realign_started"
                    self._initial_turn_error = abs(yaw_error)
                    self._turn_confirmations = 0
                    self._drive_realign_confirmations = 0
                    self._best_progress = 0.0
                    self._phase_started = now
                    self._last_progress_time = now
                    self._motion_blocked_since = None
                    return FollowerCommand.zero(
                        self.phase,
                        self.reason,
                        progress_m=progress_m,
                        cross_track_m=cross_track_m,
                        yaw_error_rad=yaw_error,
                    )
                return FollowerCommand.zero(
                    self.phase,
                    "drive_heading_realign_confirming",
                    progress_m=progress_m,
                    cross_track_m=cross_track_m,
                    yaw_error_rad=yaw_error,
                )
            self._drive_realign_confirmations = 0
            if (
                progress_m
                >= self._best_progress
                + self.config.progress_epsilon_m
            ):
                self._best_progress = progress_m
                self._last_progress_time = now

            gait_commit_threshold = min(
                self.config.drive_gait_commit_min_progress_m,
                0.5 * self._target_distance_m,
            )
            if (
                gait_commit_threshold > 0.0
                and self._best_progress >= gait_commit_threshold
            ):
                self._drive_gait_committed = True
            gait_commit_required = bool(
                self.config.drive_gait_commit_timeout_s > 0.0
                and gait_commit_threshold > 0.0
            )
            if (
                gait_commit_required
                and not self._drive_gait_committed
                and now - self._phase_started
                > self.config.drive_gait_commit_timeout_s
            ):
                return self._start_drive_replan_settle(
                    now,
                    completion_reason=(
                        "drive_gait_commit_unconfirmed_replan"
                    ),
                    progress_m=progress_m,
                    cross_track_m=cross_track_m,
                    yaw_error_rad=yaw_error,
                )

            remaining = self._target_distance_m - progress_m
            arc_completion_yaw_tolerance = (
                self._continuous_arc_completion_yaw_tolerance()
            )
            completion_tolerance = drive_completion_tolerance
            if (
                self._continuous_arc_active
                and progress_m
                >= self._target_distance_m + self.config.drive_tolerance_m
                and abs(yaw_error) > arc_completion_yaw_tolerance
            ):
                return self._start_drive_replan_settle(
                    now,
                    completion_reason=(
                        "continuous_arc_heading_unresolved_replan"
                    ),
                    progress_m=progress_m,
                    cross_track_m=cross_track_m,
                    yaw_error_rad=yaw_error,
                )
            if (
                remaining <= completion_tolerance
                and (
                    not self._continuous_arc_active
                    or abs(yaw_error) <= arc_completion_yaw_tolerance
                )
                and (
                    not gait_commit_required
                    or self._drive_gait_committed
                )
                and now - self._phase_started
                >= self.config.minimum_drive_motion_time_s
            ):
                if (
                    self._continuous_arc_active
                    and abs(cross_track_m)
                    > min(
                        self.config.max_cross_track_m,
                        2.0 * self.config.drive_tolerance_m,
                    )
                ):
                    return self._start_drive_replan_settle(
                        now,
                        completion_reason=(
                            "continuous_arc_cross_track_replan"
                        ),
                        progress_m=progress_m,
                        cross_track_m=cross_track_m,
                        yaw_error_rad=yaw_error,
                    )
                return self._start_normal_drive_settle(
                    now,
                    progress_m=progress_m,
                    cross_track_m=cross_track_m,
                    yaw_error_rad=yaw_error,
                    completion_tolerance_m=completion_tolerance,
                )

            vx = clamp(
                self.config.drive_kp * remaining,
                self.config.min_vx,
                self.config.max_vx,
            )
            if (
                self._continuous_arc_active
                and abs(yaw_error)
                > self.config.drive_steering_deadband_rad
            ):
                vx = min(vx, self.config.continuous_forward_arc_max_vx)
            if self.config.drive_braking_deceleration_mps2 > 0.0:
                # Taper below the walking-band floor only inside the physical
                # stopping envelope. Safe Bridge applies the same bounded
                # deceleration, so the endpoint does not receive a one-tick
                # walking-speed-to-zero transition.
                braking_distance = max(0.0, remaining - completion_tolerance)
                if (
                    self._continuous_arc_active
                    and abs(yaw_error) > arc_completion_yaw_tolerance
                ):
                    # Keep a small forward component while the requested arc
                    # is still closing its heading. Otherwise the braking
                    # equation reaches zero at the position tolerance and
                    # silently turns DRIVE into a planted-foot yaw command.
                    # The existing overshoot guard bounds this extra arc to
                    # one drive tolerance before forcing a fresh observation.
                    braking_distance = max(
                        braking_distance,
                        max(
                            0.0,
                            remaining + self.config.drive_tolerance_m,
                        ),
                    )
                braking_vx = math.sqrt(
                    2.0
                    * self.config.drive_braking_deceleration_mps2
                    * braking_distance
                )
                vx = min(vx, braking_vx)
            # Heading has already been aligned in TURN. DRIVE is translation
            # only; any later heading error re-enters TURN above.
            wz = 0.0
            if (
                self._continuous_arc_active
                and self._continuous_arc_steering_direction
                and yaw_error * self._continuous_arc_steering_direction
                <= self.config.drive_steering_deadband_rad
            ):
                # Once this arc reaches or crosses its requested heading, do
                # not counter-steer in the same action.  The next fresh plan
                # can make any small correction without producing a snake.
                self._continuous_arc_steering_direction = 0
            continuous_arc_steering = bool(
                self._continuous_arc_active
                and self._continuous_arc_steering_direction
                and yaw_error * self._continuous_arc_steering_direction
                > self.config.drive_steering_deadband_rad
            )
            ordinary_drive_steering = bool(
                not self._continuous_arc_active
                and abs(yaw_error) > self.config.drive_steering_deadband_rad
            )
            if (
                self.config.drive_min_wz > 0.0
                and (continuous_arc_steering or ordinary_drive_steering)
            ):
                wz = self._signed_minimum(
                    self.config.yaw_kp * yaw_error,
                    self.config.drive_min_wz,
                    self.config.max_wz,
                )
            if motion_blocked:
                assert self._motion_blocked_since is not None
                blocked_wall_s = now - self._motion_blocked_since
                if blocked_wall_s <= 0.35:
                    # Safe feedback can trail a new request or briefly report
                    # zero while its independent limiter crosses a boundary.
                    # Keep the request alive through one bounded handshake
                    # before treating the burst as blocked.
                    return FollowerCommand(
                        phase=self.phase,
                        vx=vx,
                        wz=wz,
                        enable_motion=True,
                        complete=False,
                        fault=False,
                        reason="drive_waiting_for_safe_output",
                        progress_m=progress_m,
                        cross_track_m=cross_track_m,
                        yaw_error_rad=yaw_error,
                    )
                useful_progress_threshold = min(
                    self.config.partial_progress_replan_min_m,
                    SHORT_ACTION_PARTIAL_PROGRESS_FRACTION
                    * self._target_distance_m,
                )
                completion_reason = (
                    "drive_safety_block_after_progress_replan"
                    if (
                    useful_progress_threshold > 0.0
                    and self._best_progress
                    >= useful_progress_threshold
                    )
                    else "drive_safety_handshake_replan"
                )
                # A rejected request cannot become fresh again after we drop
                # it. End this burst as soon as the handshake window expires
                # instead of waiting another 0.85 s for the old block timeout.
                return self._start_drive_replan_settle(
                    now,
                    completion_reason=completion_reason,
                    progress_m=progress_m,
                    cross_track_m=cross_track_m,
                    yaw_error_rad=yaw_error,
                )
            if (
                self.config.max_drive_burst_m > 0.0
                and progress_m >= self.config.max_drive_burst_m
            ):
                return self._start_drive_replan_settle(
                    now,
                    completion_reason=(
                        "drive_bounded_progress_replan"
                    ),
                    progress_m=progress_m,
                    cross_track_m=cross_track_m,
                    yaw_error_rad=yaw_error,
                )
            if (
                self.config.partial_progress_replan_timeout_s > 0.0
                and self._best_progress
                >= min(
                    self.config.partial_progress_replan_min_m,
                    SHORT_ACTION_PARTIAL_PROGRESS_FRACTION
                    * self._target_distance_m,
                )
                and now - self._last_progress_time
                > self.config.partial_progress_replan_timeout_s
            ):
                return self._start_drive_replan_settle(
                    now,
                    completion_reason=(
                        "drive_partial_progress_replan"
                    ),
                    progress_m=progress_m,
                    cross_track_m=cross_track_m,
                    yaw_error_rad=yaw_error,
                )
            if (
                now - self._last_progress_time
                > self.config.no_progress_timeout_s
            ):
                return self._fault(
                    "drive_no_progress",
                    progress_m=progress_m,
                    cross_track_m=cross_track_m,
                    yaw_error_rad=yaw_error,
                )
            return FollowerCommand(
                phase=self.phase,
                vx=vx,
                wz=wz,
                enable_motion=True,
                complete=False,
                fault=False,
                reason="drive_tracking",
                progress_m=progress_m,
                cross_track_m=cross_track_m,
                yaw_error_rad=yaw_error,
            )

        if self.phase is FollowerPhase.SETTLE:
            if now - self._phase_started >= self.config.settle_time_s:
                remaining = (
                    self._target_distance_m
                    - progress_m
                )
                completion_recoiled = (
                    self._target_distance_m > self.config.drive_tolerance_m
                    and not self._completion_requires_replan
                    and self._completion_reason == "action_complete"
                    and remaining
                    > max(
                        self._settle_completion_tolerance_m,
                        min(
                            self.config.drive_tolerance_m,
                            SHORT_ACTION_COMPLETION_FRACTION
                            * self._target_distance_m,
                        ),
                    )
                    + self.config.progress_epsilon_m
                )
                if completion_recoiled:
                    # The body can lean ahead of the supporting feet and
                    # spring back after zero velocity. The old behavior
                    # restarted the same short action, which repeatedly
                    # produced a lean/recoil cycle without a footstep and
                    # eventually raised drive_no_progress. End this
                    # physical burst and require fresh observations; the
                    # unchanged V70 planner must choose the next action.
                    self.phase = FollowerPhase.COMPLETE
                    self.reason = (
                        "drive_settle_recoil_requires_replan"
                    )
                    self._completion_requires_replan = True
                    return FollowerCommand.zero(
                        self.phase,
                        self.reason,
                        complete=True,
                        replan_required=True,
                        progress_m=progress_m,
                        cross_track_m=cross_track_m,
                        yaw_error_rad=yaw_error,
                    )
                self.phase = FollowerPhase.COMPLETE
                self.reason = self._completion_reason
                return FollowerCommand.zero(
                    self.phase,
                    self.reason,
                    complete=True,
                    replan_required=self._completion_requires_replan,
                    progress_m=progress_m,
                    cross_track_m=cross_track_m,
                    yaw_error_rad=yaw_error,
                )
            return FollowerCommand.zero(
                self.phase,
                "settling",
                progress_m=progress_m,
                cross_track_m=cross_track_m,
                yaw_error_rad=yaw_error,
            )

        return self._fault("unknown_follower_phase")


@dataclass(frozen=True)
class RouteGoal:
    image: Path
    name: str
    arrival_calibrated: bool
    approach_temporal_distance: float
    temporal_distance_enter: float
    temporal_distance_exit: float
    cosine_distance_enter: float
    cosine_distance_exit: float
    required_confirmations: int

    def validate(self) -> None:
        if not self.image.is_file():
            raise FileNotFoundError(self.image)
        numeric = (
            self.approach_temporal_distance,
            self.temporal_distance_enter,
            self.temporal_distance_exit,
            self.cosine_distance_enter,
            self.cosine_distance_exit,
        )
        if not all(
            math.isfinite(float(value)) and float(value) >= 0.0
            for value in numeric
        ):
            raise ValueError(f"invalid arrival thresholds for {self.name}")
        if self.temporal_distance_exit < self.temporal_distance_enter:
            raise ValueError("temporal exit must be >= enter")
        if self.cosine_distance_exit < self.cosine_distance_enter:
            raise ValueError("cosine exit must be >= enter")
        if self.approach_temporal_distance < self.temporal_distance_exit:
            raise ValueError("approach threshold must be >= temporal exit")
        if self.required_confirmations < 2:
            raise ValueError("required confirmations must be >= 2")


@dataclass(frozen=True)
class RouteManifest:
    route_id: str
    goals: tuple[RouteGoal, ...]

    @classmethod
    def load(
        cls,
        path: Path,
        *,
        require_calibrated: bool,
    ) -> "RouteManifest":
        manifest_path = path.expanduser().resolve()
        payload = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )
        if payload.get("schema") != ROUTE_SCHEMA:
            raise RuntimeError("unexpected route manifest schema")
        route_id = str(payload.get("route_id", "")).strip()
        if not route_id:
            raise ValueError("route_id must be non-empty")
        raw_goals = payload.get("goals")
        if not isinstance(raw_goals, list) or not raw_goals:
            raise ValueError("route must contain at least one goal")

        goals: list[RouteGoal] = []
        for index, raw in enumerate(raw_goals):
            if not isinstance(raw, dict):
                raise ValueError(f"goal {index} must be an object")
            image = Path(str(raw["image"])).expanduser()
            if not image.is_absolute():
                image = manifest_path.parent / image
            goal = RouteGoal(
                image=image.resolve(),
                name=str(raw.get("name", f"goal_{index}")),
                arrival_calibrated=bool(
                    raw.get("arrival_calibrated", False)
                ),
                approach_temporal_distance=float(
                    raw["approach_temporal_distance"]
                ),
                temporal_distance_enter=float(
                    raw["temporal_distance_enter"]
                ),
                temporal_distance_exit=float(
                    raw["temporal_distance_exit"]
                ),
                cosine_distance_enter=float(
                    raw["cosine_distance_enter"]
                ),
                cosine_distance_exit=float(
                    raw["cosine_distance_exit"]
                ),
                required_confirmations=int(
                    raw.get("required_confirmations", 4)
                ),
            )
            goal.validate()
            if require_calibrated and not goal.arrival_calibrated:
                raise RuntimeError(
                    f"goal {goal.name!r} is not arrival-calibrated"
                )
            goals.append(goal)
        return cls(route_id=route_id, goals=tuple(goals))


class RouteProgress:
    def __init__(self, manifest: RouteManifest) -> None:
        self.manifest = manifest
        self.index = 0
        self.complete = False

    @property
    def current(self) -> RouteGoal:
        return self.manifest.goals[self.index]

    @property
    def is_final(self) -> bool:
        return self.index == len(self.manifest.goals) - 1

    def mark_current_reached(self) -> bool:
        if self.complete:
            return False
        if self.is_final:
            self.complete = True
            return False
        self.index += 1
        return True


__all__ = [
    "ActionFollower",
    "ActionFollowerConfig",
    "FollowerCommand",
    "FollowerPhase",
    "Pose2D",
    "ROUTE_SCHEMA",
    "RouteGoal",
    "RouteManifest",
    "RouteProgress",
    "bounded_inflight_forward_bridge_distance",
    "bounded_target_loss_forward_action",
    "clamp",
    "far_target_model_reacquire_allowed",
    "fit_action_to_continuous_arc",
    "left_pad_temporal_history",
    "model_amplified_segmented_turn_then_drive",
    "operator_guided_turn_first_action",
    "operator_guided_segmented_turn_then_drive",
    "rolling_motion_prefetch_ready",
    "short_buffered_motion_prefetch_allowed",
    "target_loss_budget_after_prefetch_motion",
    "should_handoff_ready_plan_without_settle",
    "should_preserve_prefetched_plan_after_partial_completion",
    "visual_turn_override_allowed",
    "wrap_angle",
]
