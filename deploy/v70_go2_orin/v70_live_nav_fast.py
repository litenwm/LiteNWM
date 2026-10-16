#!/usr/bin/env python3
"""Persistent, fail-closed V70 image-goal navigation node for Unitree Go2.

The node executes selected V70 transitions with odometry feedback and may
prefetch the next plan during the final verified drive segment. Prefetched
SE(2) actions are rebased before handoff. It publishes raw ROS velocity requests;
the independent V5 safety bridge and motion executor remain mandatory.

Startup always requires the stop file to be present and the arm file absent.
The node never removes the stop file or creates the arm file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import queue
import signal
import socket
import struct
import subprocess
import sys
import threading
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
from PIL import Image

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import Bool

from v70_closed_loop_core import (
    ActionFollower,
    FollowerCommand,
    FollowerPhase,
    Pose2D,
    active_scan_camera_recovery_allowed,
    active_history_scan_capture_slot,
    active_history_scan_fault_is_recoverable,
    active_history_scan_min_wz,
    active_history_scan_next_step,
    active_history_scan_targets,
    bounded_inflight_forward_bridge_distance,
    bounded_target_loss_forward_action,
    bounded_plan_hold_fault_reason,
    bounded_directional_execution_prefix,
    go2_live_action_follower_config,
    gated_h2_execution_prefix,
    external_control_rebase_update,
    far_target_model_reacquire_allowed,
    fit_action_to_continuous_arc,
    left_pad_temporal_history,
    model_amplified_segmented_turn_then_drive,
    operator_guided_turn_first_action,
    robust_pose_median,
    rebase_local_action_sequence,
    rolling_motion_prefetch_ready,
    RouteManifest,
    RouteProgress,
    safe_velocity_feedback_applied,
    short_buffered_motion_prefetch_allowed,
    should_continue_prefix_without_settle,
    should_handoff_ready_plan_without_settle,
    should_preserve_prefetched_plan_after_partial_completion,
    target_loss_budget_after_prefetch_motion,
    update_directional_visual_confirmation,
    visual_turn_override_allowed,
    wrap_angle,
)
from v70_goal_stop import (
    GoalEvidence,
    GoalPhase,
    GoalStopConfig,
    GoalStopSupervisor,
)
from v70_goal_profile import (
    apply_goal_visual_profile,
    load_goal_visual_profile,
)
from v70_pose_features import (
    PoseSample,
    quaternion_to_yaw,
)
import v70_persistent_planner_fast as persistent_planner_module
from v70_persistent_planner_fast import (
    PersistentV70Planner,
    ROUTE_INTENT_FIRST_STEP_CONFLICT_THRESHOLD_M,
    ROUTE_INTENT_FIRST_STEP_HEADING_THRESHOLD_RAD,
    ROUTE_INTENT_OPPOSITE_TOLERANCE_M,
)
from v70_process_utils import reap_process_bounded
from fresh_plan_gate import FreshPlanGate
from v70_runtime_contract import motion_interlock_allows_nonzero
from v70_scan_only_artifacts import (
    write_partial_scan_artifacts,
    write_scan_only_artifacts,
)
from v70_scan_annotation import (
    active_scan_required_for_plan,
    autonomous_consensus_with_operator_prior,
    confirmed_prior_after_annotation_phase,
    create_scan_annotation_request,
    direct_operator_direction_override,
    operator_direction_prior as derive_operator_direction_prior,
    read_operator_hint,
    scan_annotation_required,
)
from v70_unitree_camera_worker import jpeg_structure_error
from v70_visual_bearing import (
    estimate_goal_visual_bearing,
    unresolved_frame_direction_conflict,
)


EXIT_OK = 0
EXIT_CONFIG = 20
EXIT_MODEL = 21
EXIT_SENSOR = 22
EXIT_PLANNER = 23
EXIT_RUNTIME = 24
EXIT_SIGNAL = 130
# The optional inflight bridge is a synthesized recovery action, so it keeps
# its own conservative speed cap. Model-selected near-target actions use the
# same calibrated gait speeds as every other navigation action.
INFLIGHT_FORWARD_BRIDGE_MAX_VX = 0.20
FOLLOWER_TELEMETRY_PERIOD_S = 0.50
ROUTE_INTENT_LATCH_THRESHOLD_M = 0.06
ROUTE_INTENT_REVERSAL_CONFIRMATIONS = 3
VISUAL_LATERAL_HOLD_ENDPOINT_M = 0.06
VISUAL_LATERAL_HOLD_FIRST_DY_M = 0.05
VISUAL_LATERAL_HOLD_FIRST_DYAW_RAD = 0.12
DIRECTIONAL_RECOVERY_MIN_CENTER_OFFSET = 0.10
VISUAL_TURN_OVERRIDE_MIN_CENTER_OFFSET = 0.18
VISUAL_TURN_OVERRIDE_MAX_ABS_YAW_RAD = 0.27
DIRECTIONAL_RECOVERY_VISUAL_CONFIRMATIONS = 2
EXTERNAL_CONTROL_STABLE_DURATION_S = 1.0
DIRECTIONAL_RECOVERY_MAX_VISUAL_MISSES = 0
LIVE_DIRECT_DRIVE_YAW_TOLERANCE_RAD = 0.05
LIVE_DRIVE_MIN_VX = 0.25
LIVE_DRIVE_KP = 1.20
LIVE_H2_MAX_TOTAL_DISTANCE_M = 0.65
LIVE_H2_MAX_TOTAL_ABS_YAW_RAD = 0.30
LIVE_H2_MAX_INTERNAL_BEARING_CHANGE_RAD = 0.18
LIVE_BUFFERED_PREFETCH_MAX_DISTANCE_M = 0.55
LIVE_BUFFERED_PREFETCH_MAX_ABS_YAW_RAD = 0.30
ABSTENTION_FORWARD_STEP_M = 0.08
ABSTENTION_FORWARD_TOTAL_M = 0.24
ABSTENTION_FORWARD_MIN_TEMPORAL_DISTANCE = 9.0
ABSTENTION_FORWARD_MIN_DINO_DISTANCE = 0.42
ABSTENTION_FORWARD_MAX_ABS_YAW_RAD = 0.12
ABSTENTION_NEAR_TARGET_STEP_M = 0.06
ABSTENTION_NEAR_TARGET_MAX_CENTER_OFFSET = 0.08
ABSTENTION_NEAR_TARGET_MAX_ABS_YAW_RAD = 0.08
ABSTENTION_NEAR_TARGET_MAX_SCALE_RATIO = 0.45
ABSTENTION_NEAR_TARGET_MAX_LATEST_SCALE_RATIO = 0.50
# Start inference as soon as heading is stable. Recorded timing replay shows
# that delaying this trigger moves every reusable result farther past the
# action endpoint and directly lengthens the zero-motion handoff.
ROLLING_PREFETCH_TRIGGER_PROGRESS_M = 0.02
ROLLING_PREFETCH_BUFFERED_TRIGGER_PROGRESS_M = 0.04
ROLLING_TURN_PREFETCH_DELAY_S = 0.20
ROLLING_PREFETCH_MAX_DRIVE_YAW_ERROR_RAD = 0.10
ROLLING_PREFETCH_MAX_AGE_S = 4.50
ROLLING_PREFETCH_MAX_TRANSLATION_M = 0.32
ROLLING_PREFETCH_MAX_YAW_RAD = 0.35
# A fresh endpoint plan can finish after a few centimetres of passive gait
# settling.  Rebase only this tightly bounded residual drift instead of
# classifying it as external control and throwing away a full inference.
COMPLETED_PLAN_RESIDUAL_DRIFT_MAX_TRANSLATION_M = 0.05
COMPLETED_PLAN_RESIDUAL_DRIFT_MAX_YAW_RAD = 0.05
TAUGHT_PATH_ANCHOR_STRIDE = 12
TAUGHT_PATH_LOOKAHEAD_ANCHORS = 1
TAUGHT_PATH_MIN_MATCH_SCORE = 12.0
TAUGHT_PATH_MODERATE_MATCH_SCORE = 6.0
TAUGHT_PATH_MODERATE_CONFIRMATIONS = 2
OUTDOOR_JUNCTION_LANDMARK_FRAME = 69
OUTDOOR_JUNCTION_LANDMARK_MIN_SCORE = 120.0
OUTDOOR_JUNCTION_DISTANCE_FROM_START_M = 4.20
VISUAL_APPROACH_SCALE_RATIO = 0.30
VISUAL_HOLD_SCALE_RATIO = 0.48
VISUAL_HOLD_LATEST_SCALE_RATIO = 0.58
VISUAL_APPROACH_DINO_DISTANCE = 0.43
VISUAL_HOLD_DINO_DISTANCE = 0.36
VISUAL_APPROACH_CENTER_OFFSET = 0.12
VISUAL_HOLD_CENTER_OFFSET = 0.09
TARGET_LOCK_ACQUIRE_SCALE_RATIO = 0.08
TARGET_LOCK_RETAIN_SCALE_RATIO = 0.04
TARGET_LOCK_MAX_SHRINK_FACTOR = 0.35
TARGET_VISUAL_RECHECK_MAX_FRAMES = 4
TARGET_VISUAL_RECHECK_MIN_NEW_FRAMES = 2
TARGET_VISUAL_RECHECK_EDGE_OFFSET = 0.18
TARGET_VISUAL_RECHECK_FORWARD_STEP_M = 0.04
INFLIGHT_FORWARD_BRIDGE_ENABLED = False


def completed_plan_residual_drift_rebase_allowed(
    *,
    prefetched: bool,
    operator_stop_only: bool,
    follower_active: bool,
    pending_execution_actions: bool,
    external_control_rebase_pending: bool,
    drift_m: float,
    drift_yaw: float,
) -> bool:
    return bool(
        not prefetched
        and operator_stop_only
        and not follower_active
        and not pending_execution_actions
        and not external_control_rebase_pending
        and drift_m <= COMPLETED_PLAN_RESIDUAL_DRIFT_MAX_TRANSLATION_M
        and drift_yaw <= COMPLETED_PLAN_RESIDUAL_DRIFT_MAX_YAW_RAD
    )


@dataclass(frozen=True)
class ImageSample:
    receive_time: float
    image: Image.Image
    sha256: str
    width: int
    height: int


@dataclass(frozen=True)
class PlanningJob:
    sequence: int
    route_index: int
    goal_name: str
    goal_path: Path
    goal_sha256: str
    observations: tuple[Image.Image, ...]
    image_sha256: tuple[str, ...]
    poses: tuple[PoseSample, ...]
    visual_observations: tuple[Image.Image, ...]
    visual_image_sha256: tuple[str, ...]
    visual_poses: tuple[PoseSample, ...]
    history_source: str
    snapshot_time: float
    snapshot_pose: Pose2D
    interlock_generation: int
    motion_authorized_at_snapshot: bool
    route_intent_direction: int
    operator_visual_hint: dict[str, Any] | None
    operator_direction_prior: int
    settle_prefetch: bool
    motion_prefetch: bool
    visual_recheck_only: bool = False


@dataclass(frozen=True)
class PlanningOutput:
    job: PlanningJob
    result: dict[str, Any] | None
    completed_time: float
    error: str | None = None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reliable_centered_visual(
    visual_bearing: dict[str, Any],
    *,
    operator_direction: int = 0,
    maximum_center_offset: float = DIRECTIONAL_RECOVERY_MIN_CENTER_OFFSET,
) -> bool:
    """Release an old side preference after a reliable centred observation."""

    if operator_direction not in (-1, 0, 1):
        raise ValueError("operator_direction must be -1, 0, or 1")
    center_x = visual_bearing.get("center_x_norm")
    return bool(
        not operator_direction
        and visual_bearing.get("valid")
        and int(visual_bearing.get("direction", 0)) == 0
        and int(visual_bearing.get("valid_frame_count", 0)) >= 3
        and int(visual_bearing.get("consensus_count", 0)) >= 3
        and int(visual_bearing.get("opposing_count", 0)) == 0
        and center_x is not None
        and math.isfinite(float(center_x))
        and abs(float(center_x) - 0.5) < maximum_center_offset
    )


def visual_goal_proximity(
    visual_bearing: dict[str, Any],
    *,
    dino_cosine_distance: float,
) -> dict[str, Any]:
    """Return conservative, target-scale-based approach/hold evidence.

    Direction and scale must be supported by at least two recent frames.  The
    DINO distance is deliberately only a corroborating signal: it can never
    declare proximity without a geometrically centred target observation.
    """

    audit: dict[str, Any] = {
        "schema": "v70_visual_goal_proximity_v1",
        "available": False,
        "approach": False,
        "precision_hold": False,
        "reason": "visual_bearing_unavailable",
        "scale_source": None,
        "scale_ratio": None,
        "latest_scale_ratio": None,
    }
    if not isinstance(visual_bearing, dict) or not visual_bearing.get("valid"):
        return audit
    center_x = visual_bearing.get("center_x_norm")
    if center_x is None or not math.isfinite(float(center_x)):
        audit["reason"] = "visual_center_unavailable"
        return audit
    if int(visual_bearing.get("opposing_count", 0)) != 0:
        audit["reason"] = "opposing_visual_votes"
        return audit
    frames = [
        frame
        for frame in visual_bearing.get("frames", [])
        if isinstance(frame, dict) and frame.get("valid")
    ]
    ratios: list[float] = []
    scale_source: str | None = None
    if visual_bearing.get("cue") == "warm_target_color":
        profile = visual_bearing.get("goal_color_profile", {})
        reference_fraction = float(profile.get("goal_area_fraction", 0.0))
        if math.isfinite(reference_fraction) and reference_fraction > 0.0:
            for frame in frames:
                fraction = frame.get("component_area_fraction")
                if fraction is not None and math.isfinite(float(fraction)):
                    ratios.append(float(fraction) / reference_fraction)
            scale_source = "warm_component_area_ratio"
    else:
        reference_fraction = float(
            visual_bearing.get("goal_reference_area_fraction", 0.0)
        )
        if math.isfinite(reference_fraction) and reference_fraction > 0.0:
            for frame in frames:
                fraction = frame.get("projected_area_fraction")
                if fraction is not None and math.isfinite(float(fraction)):
                    ratios.append(float(fraction) / reference_fraction)
            scale_source = "feature_homography_area_ratio"
    if len(ratios) < 2:
        audit["reason"] = "insufficient_scale_frames"
        return audit
    recent_ratios = ratios[-2:]
    scale_ratio = float(np.median(np.asarray(recent_ratios, dtype=np.float64)))
    latest_scale_ratio = float(recent_ratios[-1])
    center_offset = abs(float(center_x) - 0.5)
    dino_distance = float(dino_cosine_distance)
    numeric = all(
        math.isfinite(value)
        for value in (
            scale_ratio,
            latest_scale_ratio,
            center_offset,
            dino_distance,
        )
    )
    if not numeric:
        audit["reason"] = "proximity_evidence_nonfinite"
        return audit
    approach = bool(
        scale_ratio >= VISUAL_APPROACH_SCALE_RATIO
        and center_offset <= VISUAL_APPROACH_CENTER_OFFSET
        and dino_distance <= VISUAL_APPROACH_DINO_DISTANCE
    )
    precision_hold = bool(
        approach
        and len(frames) >= 3
        and scale_ratio >= VISUAL_HOLD_SCALE_RATIO
        and latest_scale_ratio >= VISUAL_HOLD_LATEST_SCALE_RATIO
        and center_offset <= VISUAL_HOLD_CENTER_OFFSET
        and dino_distance <= VISUAL_HOLD_DINO_DISTANCE
    )
    audit.update(
        {
            "available": True,
            "approach": approach,
            "precision_hold": precision_hold,
            "reason": (
                "centred_target_at_goal_scale"
                if precision_hold
                else (
                    "centred_target_approach"
                    if approach
                    else "proximity_thresholds_not_met"
                )
            ),
            "scale_source": scale_source,
            "scale_ratio": scale_ratio,
            "latest_scale_ratio": latest_scale_ratio,
            "center_x_norm": float(center_x),
            "center_offset": center_offset,
            "dino_cosine_distance": dino_distance,
            "valid_scale_frames": len(ratios),
        }
    )
    return audit


def visual_target_continuity(
    visual_bearing: dict[str, Any],
    visual_proximity: dict[str, Any],
    *,
    locked: bool,
    trusted_scale_ratio: float | None,
) -> tuple[dict[str, Any], bool, float | None]:
    """Reject an implausibly vanishing warm target after it was acquired.

    A target that occupied roughly ten percent of its goal-image area cannot
    shrink to one percent during one forward action. The latest real run did
    exactly that because a 35-pixel warm component was accepted after the
    chair left view. This check acquires only from two scale-bearing frames.
    """

    audit: dict[str, Any] = {
        "schema": "v70_visual_target_continuity_v1",
        "locked_before": bool(locked),
        "locked_after": bool(locked),
        "action_allowed": True,
        "reason": "target_not_yet_locked",
        "observed_scale_ratio": None,
        "trusted_scale_ratio": trusted_scale_ratio,
        "minimum_retained_scale_ratio": None,
    }
    if not isinstance(visual_bearing, dict):
        visual_bearing = {}
    if not isinstance(visual_proximity, dict):
        visual_proximity = {}

    ratios: list[float] = []
    if visual_bearing.get("cue") == "warm_target_color":
        profile = visual_bearing.get("goal_color_profile", {})
        reference_fraction = float(profile.get("goal_area_fraction", 0.0))
        area_key = "component_area_fraction"
    else:
        reference_fraction = float(
            visual_bearing.get("goal_reference_area_fraction", 0.0)
        )
        area_key = "projected_area_fraction"
    frames = [
        frame
        for frame in visual_bearing.get("frames", [])
        if isinstance(frame, dict)
    ]
    recent_frames = frames[-2:]
    if math.isfinite(reference_fraction) and reference_fraction > 0.0:
        for frame in recent_frames:
            if not frame.get("valid"):
                continue
            fraction = frame.get(area_key)
            if fraction is None or not math.isfinite(float(fraction)):
                continue
            ratios.append(float(fraction) / reference_fraction)
    observed_scale = max(ratios) if ratios else None
    audit["observed_scale_ratio"] = observed_scale
    audit["recent_valid_frame_count"] = sum(
        bool(frame.get("valid")) for frame in recent_frames
    )

    valid_frames = int(visual_bearing.get("valid_frame_count", 0) or 0)
    acquisition_scale = visual_proximity.get("scale_ratio")
    can_acquire = bool(
        visual_bearing.get("valid")
        and valid_frames >= 2
        and audit["recent_valid_frame_count"] >= 1
        and acquisition_scale is not None
        and math.isfinite(float(acquisition_scale))
        and float(acquisition_scale) >= TARGET_LOCK_ACQUIRE_SCALE_RATIO
    )
    if not locked:
        if not can_acquire:
            return audit, False, trusted_scale_ratio
        # The proximity summary is a two-frame median, while observed_scale is
        # the largest real target footprint in those same frames.  Preserve the
        # larger value so a just-seen near target cannot fall back into the
        # longer blind-continuation budget when the next frame loses it.
        trusted = max(
            float(acquisition_scale),
            float(observed_scale)
            if observed_scale is not None
            else float(acquisition_scale),
        )
        audit.update(
            {
                "locked_after": True,
                "reason": "reliable_target_scale_acquired",
                "trusted_scale_ratio": trusted,
            }
        )
        return audit, True, trusted

    previous = float(trusted_scale_ratio or TARGET_LOCK_ACQUIRE_SCALE_RATIO)
    retained_minimum = max(
        TARGET_LOCK_RETAIN_SCALE_RATIO,
        previous * TARGET_LOCK_MAX_SHRINK_FACTOR,
    )
    audit["minimum_retained_scale_ratio"] = retained_minimum
    if (
        not visual_bearing.get("valid")
        or observed_scale is None
        or observed_scale < retained_minimum
    ):
        audit.update(
            {
                "action_allowed": False,
                "reason": "locked_target_missing_or_implausibly_small",
            }
        )
        return audit, True, previous

    trusted = max(previous, float(observed_scale))
    audit.update(
        {
            "reason": "locked_target_continuity_confirmed",
            "trusted_scale_ratio": trusted,
        }
    )
    return audit, True, trusted


def visual_target_loss_budget_reset_allowed(
    continuity: dict[str, Any],
) -> bool:
    """Reset blind-motion distance only after two current confirmations."""

    return bool(
        continuity.get("action_allowed")
        and continuity.get("reason")
        == "locked_target_continuity_confirmed"
        and int(continuity.get("recent_valid_frame_count", 0) or 0) >= 2
    )


def reliable_visual_recheck_direction(
    visual_bearing: dict[str, Any],
    *,
    route_direction: int = 0,
) -> int:
    """Return a side only when the two newest real frames agree exactly."""

    if route_direction not in (-1, 0, 1):
        raise ValueError("route_direction must be -1, 0, or 1")
    if not isinstance(visual_bearing, dict):
        return 0
    direction = int(visual_bearing.get("direction", 0) or 0)
    center_x = visual_bearing.get("center_x_norm")
    frames = [
        frame
        for frame in visual_bearing.get("frames", [])[-2:]
        if isinstance(frame, dict)
    ]
    frame_directions = [
        int(frame.get("direction", 0) or 0)
        for frame in frames
        if frame.get("valid")
    ]
    return (
        direction
        if (
            visual_bearing.get("valid")
            and visual_bearing.get("cue") == "warm_target_color"
            and direction in (-1, 1)
            and route_direction in (0, direction)
            and int(visual_bearing.get("opposing_count", 0) or 0) == 0
            and len(frame_directions) == 2
            and all(item == direction for item in frame_directions)
            and center_x is not None
            and abs(float(center_x) - 0.5)
            >= TARGET_VISUAL_RECHECK_EDGE_OFFSET
        )
        else 0
    )


def visual_recheck_forward_direction(
    source_heading_rad: float,
    source_yaw_rad: float,
    *,
    edge_direction: int,
    route_direction: int,
) -> tuple[int, bool]:
    """Choose a lateral guard for a short visual-recheck forward step.

    A reliable fresh-frame edge cue has priority, followed by the latched route
    direction. If neither exists, zero means straight/no lateral preference;
    the caller still requires a previously safety-scored H1 and applies all
    target-loss distance, scale and curvature bounds.
    """

    if edge_direction not in (-1, 0, 1):
        raise ValueError("edge_direction must be -1, 0, or 1")
    if route_direction not in (-1, 0, 1):
        raise ValueError("route_direction must be -1, 0, or 1")
    direction = edge_direction if edge_direction else route_direction
    compatible = bool(
        not direction
        or (
            direction * float(source_heading_rad) >= -0.02
            and direction * float(source_yaw_rad) >= -0.02
        )
    )
    return int(direction), compatible


class TaughtPathGuide:
    """Choose a nearby future demonstration view as NoMaD's local goal.

    The terminal route goal remains unchanged for visual bearing and arrival
    telemetry.  This guide only replaces the image passed to the planner when
    a goal-bound visual profile explicitly names a ``source_teach_run``.
    """

    def __init__(
        self,
        source_teach_run: Path,
        anchor_frames: list[str] | tuple[str, ...] | None = None,
        straight_corridors: list[list[int]] | tuple[tuple[int, int], ...] | None = None,
    ) -> None:
        source_teach_run = source_teach_run.expanduser().resolve()
        frame_dir = source_teach_run / "path_frames"
        all_paths = sorted(frame_dir.glob("frame_*.jpg"))
        if len(all_paths) < 4:
            raise RuntimeError(
                f"taught path has too few frames: {frame_dir}"
            )
        if anchor_frames is None:
            selected = list(all_paths[::TAUGHT_PATH_ANCHOR_STRIDE])
            if selected[-1] != all_paths[-1]:
                selected.append(all_paths[-1])
            self.anchor_source = "automatic_stride"
        else:
            if not isinstance(anchor_frames, (list, tuple)):
                raise RuntimeError("taught_path_anchor_frames must be a list")
            if not 2 <= len(anchor_frames) <= 12:
                raise RuntimeError(
                    "explicit taught path must contain 2 to 12 anchors"
                )
            selected = []
            for entry in anchor_frames:
                relative = Path(str(entry))
                if relative.is_absolute() or ".." in relative.parts:
                    raise RuntimeError(
                        f"unsafe taught path anchor: {entry!r}"
                    )
                if len(relative.parts) == 2 and relative.parts[0] == "path_frames":
                    relative = Path(relative.name)
                if len(relative.parts) != 1 or not relative.name.startswith("frame_"):
                    raise RuntimeError(
                        f"invalid taught path anchor: {entry!r}"
                    )
                candidate = (frame_dir / relative.name).resolve()
                if candidate.parent != frame_dir or not candidate.is_file():
                    raise RuntimeError(
                        f"missing taught path anchor: {candidate}"
                    )
                selected.append(candidate)
            if len(set(selected)) != len(selected):
                raise RuntimeError("explicit taught path anchors contain duplicates")
            frame_numbers = [
                int(path.stem.rsplit("_", 1)[-1]) for path in selected
            ]
            if frame_numbers != sorted(frame_numbers):
                raise RuntimeError(
                    "explicit taught path anchors must be chronological"
                )
            self.anchor_source = "profile_explicit"
        self.paths = tuple(selected)
        self.frame_numbers = tuple(
            int(path.stem.rsplit("_", 1)[-1]) for path in self.paths
        )
        self._teach_pose_rows = self._load_teach_pose_rows(source_teach_run)
        pose_by_frame = {
            int(row[0]): row for row in self._teach_pose_rows
        }
        self._teach_pose_by_frame = pose_by_frame
        parsed_corridors: list[tuple[int, int]] = []
        if straight_corridors is not None:
            if not isinstance(straight_corridors, (list, tuple)):
                raise RuntimeError("taught path straight corridors must be a list")
            for entry in straight_corridors:
                if not isinstance(entry, (list, tuple)) or len(entry) != 2:
                    raise RuntimeError(
                        "each taught path straight corridor must be [start, stop]"
                    )
                if any(isinstance(value, bool) for value in entry):
                    raise RuntimeError("straight corridor frames cannot be boolean")
                start_frame, stop_frame = (int(value) for value in entry)
                if (
                    start_frame < int(self._teach_pose_rows[0][0])
                    or stop_frame > int(self._teach_pose_rows[-1][0])
                    or start_frame >= stop_frame
                ):
                    raise RuntimeError("invalid taught path straight corridor range")
                parsed_corridors.append((start_frame, stop_frame))
            if parsed_corridors != sorted(parsed_corridors):
                raise RuntimeError("straight corridors must be chronological")
            if any(
                current[0] < previous[1]
                for previous, current in zip(
                    parsed_corridors, parsed_corridors[1:]
                )
            ):
                raise RuntimeError("straight corridors must not overlap")
        self.straight_corridors = tuple(parsed_corridors)
        self._completed_straight_corridors: set[tuple[int, int]] = set()
        if any(number not in pose_by_frame for number in self.frame_numbers):
            raise RuntimeError("taught path anchor is missing odometry")
        self._anchor_arc_m = tuple(
            float(pose_by_frame[number][4])
            for number in self.frame_numbers
        )
        self._navigation_start_pose: tuple[float, float, float] | None = None
        self._odometry_progress_frame = int(self.frame_numbers[0])
        self._odometry_progress_arc_m = float(self._anchor_arc_m[0])
        self._sift = cv2.SIFT_create(nfeatures=800)
        self._matcher = cv2.BFMatcher(cv2.NORM_L2)
        self._anchor_features = tuple(
            self._features(Image.open(path).convert("RGB"))
            for path in self.paths
        )
        self.progress_anchor = 0
        self.pending_anchor: int | None = None
        self.pending_anchor_confirmations = 0

    def _route_intent_audit(
        self,
        planning_goal_frame: int,
        odometry: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Derive a short-horizon turn prior from the sealed teach odometry."""

        current_frame = int(self._odometry_progress_frame)
        current = self._teach_pose_by_frame[current_frame]
        goal = self._teach_pose_by_frame[int(planning_goal_frame)]
        target_arc_m = min(float(goal[4]), float(current[4]) + 2.0)
        future = min(
            (
                row for row in self._teach_pose_rows
                if current_frame <= int(row[0]) <= int(planning_goal_frame)
            ),
            key=lambda row: (
                abs(float(row[4]) - target_arc_m),
                -int(row[0]),
            ),
        )
        dx = float(future[1]) - float(current[1])
        dy = float(future[2]) - float(current[2])
        cosine = math.cos(float(current[3]))
        sine = math.sin(float(current[3]))
        forward_m = cosine * dx + sine * dy
        left_m = -sine * dx + cosine * dy
        yaw_change_rad = wrap_angle(float(future[3]) - float(current[3]))
        signal = left_m + 0.5 * yaw_change_rad
        mode = "future_path_tangent"
        corridor = next(
            (
                bounds for bounds in self.straight_corridors
                if bounds[0] <= current_frame < bounds[1]
                and bounds not in self._completed_straight_corridors
            ),
            None,
        )
        if corridor is not None and odometry is not None:
            signed_cross_track_m = float(
                odometry.get("signed_cross_track_m", 0.0)
            )
            signed_yaw_error_rad = float(
                odometry.get("signed_yaw_error_rad", 0.0)
            )
            signal = (
                -0.8 * signed_cross_track_m
                -0.7 * signed_yaw_error_rad
            )
            mode = "straight_corridor_centerline_correction"
            if (
                abs(signed_cross_track_m) <= 0.08
                and abs(signed_yaw_error_rad) <= 0.08
            ):
                signal = 0.0
        direction_threshold = 0.08 if corridor is not None else 0.05
        direction = (
            0
            if abs(signal) < direction_threshold
            else (1 if signal > 0.0 else -1)
        )
        return {
            "route_intent_mode": mode,
            "route_intent_corridor_frames": (
                list(corridor) if corridor is not None else None
            ),
            "route_intent_direction": int(direction),
            "route_intent_future_frame": int(future[0]),
            "route_intent_forward_m": float(forward_m),
            "route_intent_left_m": float(left_m),
            "route_intent_yaw_change_rad": float(yaw_change_rad),
            "route_intent_signal": float(signal),
        }

    @staticmethod
    def _load_teach_pose_rows(
        source_teach_run: Path,
    ) -> tuple[tuple[int, float, float, float, float], ...]:
        frames_path = source_teach_run / "frames.jsonl"
        rows: list[tuple[int, float, float, float]] = []
        with frames_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                payload = json.loads(line)
                if payload.get("kind") != "frame":
                    continue
                pose = payload.get("pose")
                if not isinstance(pose, dict):
                    continue
                rows.append((
                    int(payload["index"]),
                    float(pose["x_m"]),
                    float(pose["y_m"]),
                    float(pose["yaw_rad"]),
                ))
        if len(rows) < 4:
            raise RuntimeError("taught path has too few odometry samples")
        origin_x, origin_y, origin_yaw = rows[0][1:]
        cosine = math.cos(origin_yaw)
        sine = math.sin(origin_yaw)
        arc_m = 0.0
        previous_x = rows[0][1]
        previous_y = rows[0][2]
        result: list[tuple[int, float, float, float, float]] = []
        for frame, x_m, y_m, yaw_rad in rows:
            arc_m += math.hypot(x_m - previous_x, y_m - previous_y)
            dx = x_m - origin_x
            dy = y_m - origin_y
            result.append((
                frame,
                cosine * dx + sine * dy,
                -sine * dx + cosine * dy,
                wrap_angle(yaw_rad - origin_yaw),
                arc_m,
            ))
            previous_x, previous_y = x_m, y_m
        return tuple(result)

    def _update_progress_from_odometry(self, pose: Pose2D) -> dict[str, Any]:
        current = (float(pose.x), float(pose.y), float(pose.yaw))
        if self._navigation_start_pose is None:
            # A fragment route can deliberately begin at a non-zero teach
            # frame (for example, from inside a corridor).  Align the live
            # start pose to that first anchor instead of incorrectly mapping
            # it onto teach frame zero.
            self._relocalize_odometry_to_anchor(pose, self.progress_anchor)
        start_x, start_y, start_yaw = self._navigation_start_pose
        dx = current[0] - start_x
        dy = current[1] - start_y
        cosine = math.cos(start_yaw)
        sine = math.sin(start_yaw)
        local_x = cosine * dx + sine * dy
        local_y = -sine * dx + cosine * dy
        local_yaw = wrap_angle(current[2] - start_yaw)
        candidates = [
            row
            for row in self._teach_pose_rows
            if (
                row[0] >= self._odometry_progress_frame - 8
                and row[0] <= self._odometry_progress_frame + 24
            )
        ]
        nearest = min(
            candidates,
            key=lambda row: (
                math.hypot(local_x - row[1], local_y - row[2])
                + 0.35 * abs(wrap_angle(local_yaw - row[3])),
                -row[0],
            ),
        )
        cross_track_m = math.hypot(local_x - nearest[1], local_y - nearest[2])
        path_error_x = local_x - nearest[1]
        path_error_y = local_y - nearest[2]
        signed_cross_track_m = (
            -math.sin(nearest[3]) * path_error_x
            + math.cos(nearest[3]) * path_error_y
        )
        signed_yaw_error_rad = wrap_angle(local_yaw - nearest[3])
        yaw_error_rad = abs(signed_yaw_error_rad)
        trusted = bool(cross_track_m <= 2.0 and yaw_error_rad <= 1.25)
        active_corridor = next(
            (
                bounds for bounds in self.straight_corridors
                if bounds[0] <= self._odometry_progress_frame < bounds[1]
                and bounds not in self._completed_straight_corridors
            ),
            None,
        )
        corridor_exit_latched = bool(
            active_corridor is not None
            and int(nearest[0]) >= int(active_corridor[1])
            and yaw_error_rad <= 0.8
        )
        if corridor_exit_latched:
            exit_frame = int(active_corridor[1])
            self._completed_straight_corridors.add(active_corridor)
            self._odometry_progress_frame = max(
                self._odometry_progress_frame, exit_frame
            )
            self._odometry_progress_arc_m = max(
                self._odometry_progress_arc_m,
                float(self._teach_pose_by_frame[exit_frame][4]),
            )
        if trusted and nearest[0] >= self._odometry_progress_frame:
            self._odometry_progress_frame = int(nearest[0])
            self._odometry_progress_arc_m = max(
                self._odometry_progress_arc_m,
                float(nearest[4]),
            )
        previous = self.progress_anchor
        if trusted or corridor_exit_latched:
            reached = max(
                index
                for index, frame in enumerate(self.frame_numbers)
                if frame <= self._odometry_progress_frame
            )
            self.progress_anchor = max(self.progress_anchor, reached)
        return {
            "available": True,
            "trusted": trusted,
            "nearest_frame": int(nearest[0]),
            "progress_frame": int(self._odometry_progress_frame),
            "progress_arc_m": float(self._odometry_progress_arc_m),
            "cross_track_m": float(cross_track_m),
            "signed_cross_track_m": float(signed_cross_track_m),
            "yaw_error_rad": float(yaw_error_rad),
            "signed_yaw_error_rad": float(signed_yaw_error_rad),
            "distance_from_start_m": float(math.hypot(local_x, local_y)),
            "corridor_exit_latched": corridor_exit_latched,
            "completed_straight_corridors": [
                list(bounds)
                for bounds in sorted(self._completed_straight_corridors)
            ],
            "advanced_anchor": self.progress_anchor > previous,
        }

    def _relocalize_odometry_to_anchor(
        self, pose: Pose2D, anchor_index: int
    ) -> None:
        """Align the live odometry origin to a strongly matched route anchor."""
        frame = int(self.frame_numbers[anchor_index])
        taught = self._teach_pose_by_frame[frame]
        start_yaw = wrap_angle(float(pose.yaw) - float(taught[3]))
        cosine = math.cos(start_yaw)
        sine = math.sin(start_yaw)
        offset_x = cosine * float(taught[1]) - sine * float(taught[2])
        offset_y = sine * float(taught[1]) + cosine * float(taught[2])
        self._navigation_start_pose = (
            float(pose.x) - offset_x,
            float(pose.y) - offset_y,
            start_yaw,
        )
        self._odometry_progress_frame = frame
        self._odometry_progress_arc_m = float(taught[4])
        self.progress_anchor = int(anchor_index)

    @staticmethod
    def _gray(image: Image.Image) -> np.ndarray:
        rgb = np.asarray(image.convert("RGB"))
        height, width = rgb.shape[:2]
        scale = min(1.0, 640.0 / max(width, height))
        if scale < 1.0:
            rgb = cv2.resize(
                rgb,
                (int(round(width * scale)), int(round(height * scale))),
                interpolation=cv2.INTER_AREA,
            )
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    def _features(
        self,
        image: Image.Image,
    ) -> tuple[tuple[cv2.KeyPoint, ...], np.ndarray | None]:
        keypoints, descriptors = self._sift.detectAndCompute(
            self._gray(image), None
        )
        return tuple(keypoints), descriptors

    def _score(
        self,
        current: tuple[tuple[cv2.KeyPoint, ...], np.ndarray | None],
        anchor: tuple[tuple[cv2.KeyPoint, ...], np.ndarray | None],
    ) -> tuple[float, int, int]:
        current_kp, current_desc = current
        anchor_kp, anchor_desc = anchor
        if current_desc is None or anchor_desc is None:
            return 0.0, 0, 0
        pairs = self._matcher.knnMatch(current_desc, anchor_desc, k=2)
        good = [
            first
            for pair in pairs
            if len(pair) == 2
            for first, second in [pair]
            if first.distance < 0.72 * second.distance
        ]
        inliers = 0
        if len(good) >= 4:
            source = np.float32(
                [current_kp[item.queryIdx].pt for item in good]
            ).reshape(-1, 1, 2)
            target = np.float32(
                [anchor_kp[item.trainIdx].pt for item in good]
            ).reshape(-1, 1, 2)
            _, mask = cv2.findHomography(
                source, target, cv2.RANSAC, 4.0
            )
            if mask is not None:
                inliers = int(mask.sum())
        return float(inliers) + 0.10 * len(good), len(good), inliers

    def select(
        self,
        current_image: Image.Image,
        current_pose: Pose2D | None = None,
    ) -> tuple[Image.Image, dict[str, Any]]:
        odometry = (
            self._update_progress_from_odometry(current_pose)
            if current_pose is not None
            else {"available": False, "trusted": False}
        )
        current = self._features(current_image)
        start = max(0, self.progress_anchor - 1)
        stop = min(len(self.paths), self.progress_anchor + 4)
        scored = [
            (self._score(current, self._anchor_features[index]), index)
            for index in range(start, stop)
        ]
        (best_score, good_matches, inliers), matched = max(
            scored, key=lambda item: item[0][0]
        )
        previous = self.progress_anchor
        junction_landmark_relocalized = False
        junction_relocalization_source: str | None = None
        if (
            current_pose is not None
            and self._odometry_progress_frame < OUTDOOR_JUNCTION_LANDMARK_FRAME
            and OUTDOOR_JUNCTION_LANDMARK_FRAME in self.frame_numbers
        ):
            landmark_index = self.frame_numbers.index(
                OUTDOOR_JUNCTION_LANDMARK_FRAME
            )
            landmark_score = self._score(
                current,
                self._anchor_features[landmark_index],
            )
            distance_landmark = bool(
                float(odometry.get("distance_from_start_m", 0.0))
                >= OUTDOOR_JUNCTION_DISTANCE_FROM_START_M
            )
            visual_landmark = bool(
                landmark_score[0] >= OUTDOOR_JUNCTION_LANDMARK_MIN_SCORE
            )
            if visual_landmark or distance_landmark:
                best_score, good_matches, inliers = landmark_score
                matched = landmark_index
                self._relocalize_odometry_to_anchor(current_pose, matched)
                odometry = self._update_progress_from_odometry(current_pose)
                junction_landmark_relocalized = True
                junction_relocalization_source = (
                    "visual"
                    if visual_landmark
                    else "distance_from_start"
                )
        strong_match = best_score >= TAUGHT_PATH_MIN_MATCH_SCORE
        moderate_forward_match = bool(
            best_score >= TAUGHT_PATH_MODERATE_MATCH_SCORE
            and matched > self.progress_anchor
        )
        if moderate_forward_match:
            if self.pending_anchor == matched:
                self.pending_anchor_confirmations += 1
            else:
                self.pending_anchor = matched
                self.pending_anchor_confirmations = 1
        else:
            self.pending_anchor = None
            self.pending_anchor_confirmations = 0
        repeated_moderate_match = bool(
            moderate_forward_match
            and self.pending_anchor_confirmations
            >= TAUGHT_PATH_MODERATE_CONFIRMATIONS
        )
        startup_relocalized = bool(
            current_pose is not None
            and self._odometry_progress_frame
            <= int(self.frame_numbers[min(1, len(self.frame_numbers) - 1)]) + 2
            and matched >= self.progress_anchor + 2
            and strong_match
            and self.pending_anchor_confirmations
            >= TAUGHT_PATH_MODERATE_CONFIRMATIONS
        )
        if startup_relocalized:
            # Fragment tests and interrupted long runs may resume well after
            # frame zero. Two strong, consecutive visual matches are enough
            # to establish the route anchor; the live odometry is then
            # transformed into that anchor's taught coordinate frame.
            self._relocalize_odometry_to_anchor(current_pose, matched)
            odometry = self._update_progress_from_odometry(current_pose)
            self.pending_anchor = None
            self.pending_anchor_confirmations = 0
        if (
            (
                not odometry.get("available")
                or not odometry.get("trusted")
            )
            and (strong_match or repeated_moderate_match)
        ):
            # Progress remains monotonic. A single moderate match cannot
            # teleport through a repeated door/window. Two consecutive
            # matches may catch up by two anchors so that the local planning
            # goal does not remain behind a robot that has already moved.
            advance_limit = 2 if repeated_moderate_match else 1
            self.progress_anchor = max(
                self.progress_anchor,
                min(matched, self.progress_anchor + advance_limit),
            )
            self.pending_anchor = None
            self.pending_anchor_confirmations = 0
        goal_anchor = min(
            len(self.paths) - 1,
            self.progress_anchor + TAUGHT_PATH_LOOKAHEAD_ANCHORS,
        )
        with Image.open(self.paths[goal_anchor]) as raw:
            local_goal = raw.convert("RGB").copy()
        route_intent = self._route_intent_audit(
            self.frame_numbers[goal_anchor],
            odometry,
        )
        route_intent["route_intent_trusted"] = bool(
            odometry.get("trusted")
            or odometry.get("corridor_exit_latched")
        )
        if not route_intent["route_intent_trusted"]:
            route_intent["route_intent_direction"] = 0
        return local_goal, {
            "enabled": True,
            "previous_anchor": previous,
            "matched_anchor": matched,
            "matched_frame": self.frame_numbers[matched],
            "match_score": best_score,
            "good_matches": good_matches,
            "homography_inliers": inliers,
            "progress_match_reason": (
                "outdoor_junction_visual_landmark"
                if junction_landmark_relocalized
                else "startup_visual_relocalization"
                if startup_relocalized
                else "strong_match"
                if strong_match
                else (
                    "repeated_moderate_match"
                    if repeated_moderate_match
                    else "unconfirmed_match"
                )
            ),
            "pending_anchor": self.pending_anchor,
            "pending_anchor_confirmations": (
                self.pending_anchor_confirmations
            ),
            "startup_relocalized": startup_relocalized,
            "junction_landmark_relocalized": (
                junction_landmark_relocalized
            ),
            "junction_relocalization_source": (
                junction_relocalization_source
            ),
            "junction_distance_trigger_m": (
                OUTDOOR_JUNCTION_DISTANCE_FROM_START_M
            ),
            "progress_anchor": self.progress_anchor,
            "planning_goal_anchor": goal_anchor,
            "planning_goal_frame": self.frame_numbers[goal_anchor],
            "anchor_count": len(self.paths),
            "anchor_source": self.anchor_source,
            "terminal_anchor_active": goal_anchor == len(self.paths) - 1,
            "odometry_progress": odometry,
            **route_intent,
        }


def suppress_terminal_bearing_during_taught_path(
    visual_bearing: dict[str, Any],
    taught_path_audit: dict[str, Any],
) -> dict[str, Any]:
    """Keep the far terminal image from steering between path anchors."""
    if (
        not taught_path_audit.get("enabled")
        or taught_path_audit.get("terminal_anchor_active", False)
    ):
        return visual_bearing
    result = dict(visual_bearing)
    result["taught_path_raw_terminal_direction"] = result.get("direction")
    result["taught_path_raw_terminal_reason"] = result.get("reason")
    result.update({
        "valid": False,
        "direction": 0,
        "direction_name": "center",
        "center_x_norm": 0.5,
        "bearing_rad": 0.0,
        "reason": "taught_path_intermediate_terminal_bearing_suppressed",
    })
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route-manifest", required=True, type=Path)
    parser.add_argument("--interface", default="enP8p1s0")
    parser.add_argument("--odom-topic", default="/utlidar/robot_odom")
    parser.add_argument("--cmd-topic", default="/nav/cmd_vel_raw")
    parser.add_argument(
        "--safe-cmd-topic",
        default="/nav/cmd_vel_safe",
    )
    parser.add_argument(
        "--motion-enable-topic",
        default="/nav/motion_enable",
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
        "--ready-file",
        type=Path,
        default=Path("/tmp/v70_nav_ready"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.home() / "litenwm_runs/current",
    )
    parser.add_argument("--live", action="store_true")
    parser.add_argument(
        "--operator-stop-only",
        action="store_true",
        help=(
            "allow an uncalibrated single goal in live mode, disable "
            "automatic route completion, and require the operator to stop"
        ),
    )
    parser.add_argument("--shadow-once", action="store_true")
    parser.add_argument(
        "--scan-only",
        action="store_true",
        help=(
            "perform one live active scan, save four ordered frames, "
            "assert STOP, and exit before planning"
        ),
    )
    parser.add_argument("--camera-interval", type=float, default=0.20)
    parser.add_argument("--camera-max-age", type=float, default=0.60)
    parser.add_argument("--odom-interval", type=float, default=0.20)
    parser.add_argument("--odom-max-age", type=float, default=0.30)
    parser.add_argument(
        "--safe-feedback-max-age",
        type=float,
        default=0.25,
    )
    parser.add_argument("--planner-timeout", type=float, default=20.0)
    parser.add_argument(
        "--cem-iterations",
        type=int,
        default=3,
        help="number of online CEM refinement iterations (formal V70: 3)",
    )
    parser.add_argument(
        "--adaptive-cem",
        action="store_true",
        help=(
            "stop after CEM1 when a full safe elite set passes all existing "
            "planning gates; otherwise continue through the configured rounds"
        ),
    )
    parser.add_argument(
        "--adaptive-cem-min-safe",
        type=int,
        default=4,
        help="minimum safe CEM1 candidates required for the fast path",
    )
    parser.add_argument(
        "--fixed-pool-fast-path",
        action="store_true",
        help=(
            "skip CEM when the fixed K17 pool already contains a complete "
            "safe elite set under all existing planning gates"
        ),
    )
    parser.add_argument(
        "--dino-feature-cache",
        action="store_true",
        help=(
            "reuse content-addressed DINO features for unchanged history "
            "and goal images"
        ),
    )
    parser.add_argument(
        "--settle-plan-prefetch",
        action="store_true",
        help=(
            "start the next plan from a fresh endpoint image while the "
            "completed physical action is settling at zero velocity"
        ),
    )
    parser.add_argument(
        "--rolling-plan-prefetch",
        action="store_true",
        help=(
            "prefetch the next plan during the final verified DRIVE and "
            "rebase its first SE(2) endpoint at handoff"
        ),
    )
    parser.add_argument(
        "--async-plan-images",
        action="store_true",
        help=(
            "save diagnostic plan images on a dedicated worker and hard-link "
            "identical history/visual frames"
        ),
    )
    parser.add_argument(
        "--live-max-vx",
        type=float,
        default=0.24,
        help="main-navigation forward speed cap in m/s",
    )
    parser.add_argument("--startup-timeout", type=float, default=30.0)
    parser.add_argument(
        "--sensor-loss-timeout",
        type=float,
        default=2.0,
    )
    parser.add_argument(
        "--stationary-translation-tolerance",
        type=float,
        default=0.025,
    )
    parser.add_argument(
        "--stationary-yaw-tolerance",
        type=float,
        default=0.050,
    )
    parser.add_argument("--retry-delay", type=float, default=1.0)
    parser.add_argument("--shadow-period", type=float, default=15.0)
    parser.add_argument("--control-period", type=float, default=0.05)
    parser.add_argument("--save-plan-images", action="store_true")
    parser.add_argument(
        "--active-history-scan",
        action="store_true",
        help=(
            "enable bounded startup/recovery yaw scans while temporal history "
            "comes from real action-endpoint centre frames"
        ),
    )
    parser.add_argument(
        "--scan-step-yaw-rad",
        type=float,
        default=0.15,
        help="yaw increment for each active history scan step",
    )
    parser.add_argument(
        "--scan-pattern",
        choices=(
            "adaptive_three",
            "symmetric_compact",
            "symmetric",
            "left_sweep",
            "left_only",
            "right_to_left",
        ),
        default="adaptive_three",
        help=(
            "adaptive_three captures left/centre/right only when startup or "
            "recovery needs it; symmetric_compact captures four balanced views over a "
            "shorter pure-yaw sweep and returns to center; symmetric "
            "captures left-left/right-right; left_sweep "
            "captures four increasing left views before a closed-loop "
            "return; left_only captures the same views and starts planning "
            "without a right-turn return; right_to_left prepositions to "
            "the right, captures four views while sweeping left, and "
            "finishes at the original heading"
        ),
    )
    parser.add_argument(
        "--scan-left-min-wz",
        type=float,
        default=0.30,
        help=(
            "minimum magnitude for positive-yaw scan turns; keep this "
            "scan-specific so normal navigation tuning is unchanged"
        ),
    )
    parser.add_argument(
        "--scan-turn-arc-vx",
        type=float,
        default=0.0,
        help=(
            "bounded forward assist for scan turns; zero disables it"
        ),
    )
    parser.add_argument(
        "--scan-turn-arc-max-distance-m",
        type=float,
        default=0.0,
        help=(
            "maximum forward assist distance per scan step; zero "
            "disables it"
        ),
    )
    parser.add_argument(
        "--max-consecutive-plan-holds",
        type=int,
        default=5,
        help=(
            "fail closed after this many consecutive plans are held or "
            "rejected before any navigation action starts"
        ),
    )
    parser.add_argument(
        "--aggressive-navigation",
        action="store_true",
        help=(
            "continue planning through visual frame direction conflicts and "
            "degrade scan turn no-progress faults to a skipped viewpoint"
        ),
    )
    parser.add_argument(
        "--max-cumulative-scan-translation-m",
        type=float,
        default=2.0,
        help=(
            "maximum summed active-scan translation before a navigation "
            "action starts; exceeding it asserts STOP"
        ),
    )
    parser.add_argument(
        "--scan-right-min-wz",
        type=float,
        default=0.35,
        help=(
            "minimum magnitude for negative-yaw scan turns; this Go2 has "
            "a measured asymmetric in-place right-turn deadband"
        ),
    )
    parser.add_argument(
        "--scan-frame-timeout",
        type=float,
        default=6.0,
        help="maximum wait for a fresh camera frame after each scan turn",
    )
    parser.add_argument(
        "--scan-annotation",
        action="store_true",
        help=(
            "pause at zero velocity after every four-frame scan until the "
            "operator marks the visible target or confirms it is absent"
        ),
    )
    parser.add_argument(
        "--scan-annotation-root",
        type=Path,
        default=Path.home() / "litenwm_scan_annotations",
    )
    parser.add_argument(
        "--scan-annotation-timeout",
        type=float,
        default=300.0,
        help="fail closed if no scan annotation response arrives in time",
    )
    parser.add_argument(
        "--scan-annotation-max-rounds",
        type=int,
        default=2,
        help=(
            "request operator annotation for only the first N scan/plan "
            "rounds; all later rounds use autonomous inference"
        ),
    )
    parser.add_argument(
        "--camera-restart-delay",
        type=float,
        default=3.0,
        help="stationary camera-stale time before restarting its worker",
    )
    parser.add_argument(
        "--camera-recovery-timeout",
        type=float,
        default=12.0,
        help="maximum bounded stationary camera recovery window",
    )
    parser.add_argument(
        "--camera-restart-limit",
        type=int,
        default=0,
        help=(
            "maximum camera worker restarts in one navigation run; the "
            "Go2 videohub leaks short-lived RPC sessions, so live defaults "
            "to one persistent worker with no automatic restart"
        ),
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive = (
        "camera_interval",
        "camera_max_age",
        "odom_interval",
        "odom_max_age",
        "safe_feedback_max_age",
        "planner_timeout",
        "startup_timeout",
        "sensor_loss_timeout",
        "stationary_translation_tolerance",
        "stationary_yaw_tolerance",
        "retry_delay",
        "shadow_period",
        "control_period",
        "scan_frame_timeout",
        "scan_annotation_timeout",
        "camera_restart_delay",
        "camera_recovery_timeout",
    )
    for name in positive:
        value = float(getattr(args, name))
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.operator_stop_only and not args.live:
        raise ValueError("--operator-stop-only requires --live")
    if args.settle_plan_prefetch and (
        not args.live or not args.operator_stop_only
    ):
        raise ValueError(
            "--settle-plan-prefetch requires live operator-stop-only mode"
        )
    if args.rolling_plan_prefetch and (
        not args.live or not args.operator_stop_only
    ):
        raise ValueError(
            "--rolling-plan-prefetch requires live operator-stop-only mode"
        )
    if args.async_plan_images and not args.save_plan_images:
        raise ValueError(
            "--async-plan-images requires --save-plan-images"
        )
    if args.scan_only and not args.live:
        raise ValueError("--scan-only requires --live")
    if args.scan_only and not args.active_history_scan:
        raise ValueError("--scan-only requires --active-history-scan")
    if args.scan_only and args.shadow_once:
        raise ValueError("--scan-only is incompatible with --shadow-once")
    if args.scan_only and args.operator_stop_only:
        raise ValueError(
            "--scan-only is incompatible with --operator-stop-only"
        )
    if not 0 <= args.camera_restart_limit <= 4:
        raise ValueError("--camera-restart-limit must be in [0, 4]")
    if args.camera_restart_delay >= args.camera_recovery_timeout:
        raise ValueError(
            "--camera-restart-delay must precede recovery timeout"
        )
    active_history_scan_targets(args.scan_step_yaw_rad, args.scan_pattern)
    if (
        not math.isfinite(args.scan_left_min_wz)
        or not 0.30 <= args.scan_left_min_wz <= 0.35
    ):
        raise ValueError("--scan-left-min-wz must be in [0.30, 0.35]")
    if (
        not math.isfinite(args.scan_turn_arc_vx)
        or not 0.0 <= args.scan_turn_arc_vx <= 0.20
    ):
        raise ValueError("--scan-turn-arc-vx must be in [0.0, 0.20]")
    if (
        not math.isfinite(args.scan_turn_arc_max_distance_m)
        or not 0.0 <= args.scan_turn_arc_max_distance_m <= 0.05
    ):
        raise ValueError(
            "--scan-turn-arc-max-distance-m must be in [0.0, 0.05]"
        )
    arc_enabled = args.scan_turn_arc_vx > 0.0
    arc_distance_enabled = args.scan_turn_arc_max_distance_m > 0.0
    if arc_enabled != arc_distance_enabled:
        raise ValueError(
            "scan turn arc velocity and distance must be enabled together"
        )
    if not 1 <= args.max_consecutive_plan_holds <= 5:
        raise ValueError("--max-consecutive-plan-holds must be in [1, 5]")
    if (
        not math.isfinite(args.max_cumulative_scan_translation_m)
        or not 0.10 <= args.max_cumulative_scan_translation_m <= 2.00
    ):
        raise ValueError(
            "--max-cumulative-scan-translation-m must be in [0.10, 2.00]"
        )
    if (
        not math.isfinite(args.scan_right_min_wz)
        or not 0.30 <= args.scan_right_min_wz <= 0.35
    ):
        raise ValueError("--scan-right-min-wz must be in [0.30, 0.35]")
    if args.active_history_scan and not args.live:
        raise ValueError("--active-history-scan requires --live")
    if args.scan_annotation and not args.active_history_scan:
        raise ValueError("--scan-annotation requires --active-history-scan")
    if not 10.0 <= args.scan_annotation_timeout <= 900.0:
        raise ValueError("--scan-annotation-timeout must be in [10, 900]")
    if not 2 <= args.scan_annotation_max_rounds <= 10:
        raise ValueError("--scan-annotation-max-rounds must be in [2, 10]")
    if not 1 <= args.cem_iterations <= 3:
        raise ValueError("--cem-iterations must be in [1, 3]")
    if not 1 <= args.adaptive_cem_min_safe <= 17:
        raise ValueError("--adaptive-cem-min-safe must be in [1, 17]")
    if not 0.20 <= args.live_max_vx <= 0.50:
        raise ValueError("--live-max-vx must be in [0.20, 0.50]")
    if not args.stop_file.is_file():
        raise RuntimeError(
            f"startup requires stop file: {args.stop_file}"
        )
    if args.arm_file.exists():
        raise RuntimeError(
            f"startup requires arm file absent: {args.arm_file}"
        )
    if args.ready_file.exists():
        raise RuntimeError(
            f"stale ready file exists: {args.ready_file}"
        )


def write_startup_fault_status(
    args: argparse.Namespace,
    *,
    stage: str,
    error: BaseException,
    exit_code: int,
) -> None:
    stop_file = args.stop_file.expanduser().resolve()
    stop_file.parent.mkdir(parents=True, exist_ok=True)
    stop_file.touch(exist_ok=True)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "v70_live_nav_status_v1",
        "updated_unix": time.time(),
        "mode": (
            "scan_only"
            if args.scan_only
            else ("live" if args.live else "shadow")
        ),
        "scan_only": args.scan_only,
        "operator_stop_only": args.operator_stop_only,
        "manual_stop_required": args.operator_stop_only,
        "ready": False,
        "fault": (
            f"startup_{stage}:{type(error).__name__}:{error}"
        ),
        "exit_code": int(exit_code),
        "stop_present": stop_file.is_file(),
        "arm_present": args.arm_file.expanduser().exists(),
        "plan_inflight": False,
        "route_complete": False,
        "follower_phase": "IDLE",
        "last_command": {
            "vx": 0.0,
            "wz": 0.0,
            "enable": False,
            "reason": "startup_fault_stop_latched",
        },
        "last_plan": None,
        "last_goal_decision": None,
    }
    temporary = output_dir / "status.json.tmp"
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(output_dir / "status.json")


class UnitreeCameraBuffer:
    def __init__(
        self,
        interface: str,
        *,
        interval_s: float,
        corrupt_dir: Path,
        maxlen: int = 64,
    ) -> None:
        self.interface = str(interface)
        self.interval_s = float(interval_s)
        self.corrupt_dir = corrupt_dir.expanduser().resolve()
        self.samples: deque[ImageSample] = deque(maxlen=maxlen)
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.last_error: str | None = None
        self.parent_socket: socket.socket | None = None
        self.process: subprocess.Popen[bytes] | None = None
        self.restart_thread: threading.Thread | None = None
        self.restart_error: str | None = None

    def start(self) -> None:
        if self.process is not None:
            raise RuntimeError("camera worker is already started")
        if self.thread is not None:
            raise RuntimeError("camera reader thread is already started")
        self.stop_event.clear()
        parent_socket, child_socket = socket.socketpair(
            socket.AF_UNIX,
            socket.SOCK_STREAM,
        )
        worker = Path(__file__).resolve().with_name(
            "v70_unitree_camera_worker.py"
        )
        if not worker.is_file():
            parent_socket.close()
            child_socket.close()
            raise FileNotFoundError(worker)
        self.process = subprocess.Popen(
            [
                sys.executable,
                str(worker),
                "--fd",
                str(child_socket.fileno()),
                "--interface",
                self.interface,
                "--interval",
                str(self.interval_s),
                "--client-timeout",
                "3.0",
                "--corrupt-dir",
                str(self.corrupt_dir),
                "--max-corrupt-frames",
                "32",
            ],
            pass_fds=(child_socket.fileno(),),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=True,
        )
        child_socket.close()
        parent_socket.settimeout(1.0)
        self.parent_socket = parent_socket
        self.thread = threading.Thread(
            target=self._run,
            name="v70-camera-buffer",
            daemon=True,
        )
        self.thread.start()

    def stop(self) -> None:
        restart_thread = self.restart_thread
        if (
            restart_thread is not None
            and restart_thread is not threading.current_thread()
            and restart_thread.is_alive()
        ):
            restart_thread.join(timeout=5.0)
        self.stop_event.set()
        process = self.process
        if process is not None and process.poll() is None:
            try:
                # Signal before closing the socket so the SDK call starts
                # leaving while the reader thread is being unblocked.
                process.terminate()
            except ProcessLookupError:
                pass
        if self.parent_socket is not None:
            try:
                self.parent_socket.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        thread = self.thread
        if thread is not None:
            thread.join(timeout=1.0)
            self.thread = None
        if self.parent_socket is not None:
            self.parent_socket.close()
            self.parent_socket = None
        if process is not None:
            report = reap_process_bounded(
                process,
                terminate_timeout_s=2.0,
                kill_timeout_s=1.0,
                send_terminate=False,
            )
            if report.error is not None:
                with self.lock:
                    self.last_error = (
                        f"camera_worker_shutdown:{report.error}"
                    )
            self.process = None

    def restart(self) -> None:
        """Recreate the isolated SDK participant after a bounded stall."""

        self.stop()
        with self.lock:
            self.samples.clear()
            self.last_error = None
        self.start()

    def restart_async(self) -> bool:
        """Restart without blocking the ROS control/spin thread."""

        existing = self.restart_thread
        if existing is not None and existing.is_alive():
            return False
        self.restart_error = None

        def run_restart() -> None:
            try:
                self.restart()
            except Exception as exc:
                self.restart_error = f"{type(exc).__name__}:{exc}"

        self.restart_thread = threading.Thread(
            target=run_restart,
            name="v70-camera-restart",
            daemon=True,
        )
        self.restart_thread.start()
        return True

    def diagnostics(self) -> dict[str, Any]:
        with self.lock:
            samples = list(self.samples)
            error = self.last_error
        process = self.process
        restart_thread = self.restart_thread
        return {
            "sample_count": len(samples),
            "latest_age_s": (
                None
                if not samples
                else max(0.0, time.monotonic() - samples[-1].receive_time)
            ),
            "last_error": error,
            "worker_pid": None if process is None else process.pid,
            "worker_returncode": (
                None if process is None else process.poll()
            ),
            "restart_in_progress": bool(
                restart_thread is not None and restart_thread.is_alive()
            ),
            "restart_error": self.restart_error,
        }

    @staticmethod
    def _recv_exact(channel: socket.socket, size: int) -> bytes:
        chunks: list[bytes] = []
        remaining = int(size)
        while remaining > 0:
            chunk = channel.recv(remaining)
            if not chunk:
                raise EOFError("camera worker socket closed")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def _run(self) -> None:
        if self.parent_socket is None or self.process is None:
            with self.lock:
                self.last_error = "camera worker was not initialized"
            return
        channel = self.parent_socket
        header_size = struct.calcsize("!BdI")
        while not self.stop_event.is_set():
            try:
                header = self._recv_exact(channel, header_size)
                kind, receive_time, payload_size = struct.unpack(
                    "!BdI",
                    header,
                )
                if payload_size > 16 * 1024 * 1024:
                    raise RuntimeError("camera payload exceeds 16 MiB")
                raw = self._recv_exact(channel, payload_size)
                if kind == 2:
                    raise RuntimeError(
                        raw.decode("utf-8", errors="replace")
                    )
                if kind != 1 or not raw:
                    raise RuntimeError("invalid camera worker message")
                structure_error = jpeg_structure_error(raw)
                if structure_error is not None:
                    raise RuntimeError(
                        f"camera JPEG integrity failed: {structure_error}"
                    )
                bgr = cv2.imdecode(
                    np.frombuffer(raw, dtype=np.uint8),
                    cv2.IMREAD_COLOR,
                )
                if bgr is None:
                    raise RuntimeError("camera JPEG decode failed")
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                sample = ImageSample(
                    receive_time=float(receive_time),
                    image=Image.fromarray(rgb, mode="RGB"),
                    sha256=hashlib.sha256(raw).hexdigest(),
                    width=int(rgb.shape[1]),
                    height=int(rgb.shape[0]),
                )
                with self.lock:
                    self.samples.append(sample)
                    self.last_error = None
            except socket.timeout:
                if self.process.poll() is not None:
                    with self.lock:
                        self.last_error = (
                            "camera worker exited with code "
                            f"{self.process.returncode}"
                        )
            except Exception as exc:
                if self.process.poll() is not None:
                    with self.lock:
                        self.last_error = (
                            "camera worker exited with code "
                            f"{self.process.returncode}: {exc}"
                        )
                    return
                with self.lock:
                    self.last_error = str(exc)

    def snapshot(
        self,
        *,
        count: int,
        max_age_s: float,
    ) -> tuple[ImageSample, ...]:
        with self.lock:
            values = list(self.samples)
            error = self.last_error
        if len(values) < count:
            raise RuntimeError(
                f"camera buffer has {len(values)}/{count} frames"
            )
        selected = tuple(values[-count:])
        now = time.monotonic()
        if now - selected[-1].receive_time > max_age_s:
            raise RuntimeError("camera buffer is stale")
        minimum_span = self.interval_s * (count - 1) * 0.65
        if (
            selected[-1].receive_time
            - selected[0].receive_time
            < minimum_span
        ):
            raise RuntimeError("camera history span is too short")
        if error:
            raise RuntimeError(f"camera error: {error}")
        return selected

    def latest_after(
        self,
        *,
        receive_time: float,
        max_age_s: float,
    ) -> ImageSample:
        """Return the first decoded frame captured after a motion settled."""

        with self.lock:
            values = list(self.samples)
            error = self.last_error
        selected = next(
            (
                sample
                for sample in values
                if sample.receive_time > float(receive_time)
            ),
            None,
        )
        if selected is None:
            raise RuntimeError("camera has no fresh post-turn frame")
        if time.monotonic() - selected.receive_time > max_age_s:
            raise RuntimeError("post-turn camera frame is stale")
        if error:
            raise RuntimeError(f"camera error: {error}")
        return selected

    def newest_after(
        self,
        *,
        receive_time: float,
        max_age_s: float,
    ) -> ImageSample:
        """Return the newest decoded frame after a rolling-motion cutoff."""

        with self.lock:
            values = list(self.samples)
            error = self.last_error
        selected = next(
            (
                sample
                for sample in reversed(values)
                if sample.receive_time > float(receive_time)
            ),
            None,
        )
        if selected is None:
            raise RuntimeError("camera has no fresh rolling-motion frame")
        if time.monotonic() - selected.receive_time > max_age_s:
            raise RuntimeError("rolling-motion camera frame is stale")
        if error:
            raise RuntimeError(f"camera error: {error}")
        return selected


class OdomHistory:
    def __init__(
        self,
        *,
        maxlen: int = 512,
    ) -> None:
        self.samples: deque[PoseSample] = deque(maxlen=maxlen)
        self.lock = threading.Lock()

    def append(self, msg: Odometry) -> None:
        position = msg.pose.pose.position
        orientation = msg.pose.pose.orientation
        sample = PoseSample(
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
        values = (
            sample.receive_time,
            sample.x,
            sample.y,
            sample.yaw,
        )
        if not all(math.isfinite(value) for value in values):
            return
        with self.lock:
            self.samples.append(sample)

    def latest(self) -> Pose2D | None:
        with self.lock:
            if not self.samples:
                return None
            samples = tuple(self.samples)[-9:]
        return robust_pose_median(
            tuple(
                Pose2D(
                    receive_time=sample.receive_time,
                    x=sample.x,
                    y=sample.y,
                    yaw=sample.yaw,
                )
                for sample in samples
            ),
            window_size=9,
        )

    def snapshot(
        self,
        *,
        count: int,
        interval_s: float,
        max_age_s: float,
    ) -> tuple[PoseSample, ...]:
        with self.lock:
            values = list(self.samples)
        if not values:
            raise RuntimeError("odometry buffer is empty")
        now = time.monotonic()
        if now - values[-1].receive_time > max_age_s:
            raise RuntimeError("odometry buffer is stale")

        selected = [values[-1]]
        minimum_gap = interval_s * 0.75
        last_time = values[-1].receive_time
        for sample in reversed(values[:-1]):
            if last_time - sample.receive_time >= minimum_gap:
                selected.append(sample)
                last_time = sample.receive_time
                if len(selected) == count:
                    break
        if len(selected) != count:
            raise RuntimeError(
                f"odometry buffer has no {count}-sample spaced history"
            )
        selected.reverse()
        return tuple(selected)


def goal_supervisor_for(
    route_goal,
    *,
    max_evidence_age_s: float,
) -> GoalStopSupervisor:
    return GoalStopSupervisor(
        GoalStopConfig(
            approach_temporal_distance=(
                route_goal.approach_temporal_distance
            ),
            temporal_distance_enter=(
                route_goal.temporal_distance_enter
            ),
            temporal_distance_exit=(
                route_goal.temporal_distance_exit
            ),
            cosine_distance_enter=(
                route_goal.cosine_distance_enter
            ),
            cosine_distance_exit=(
                route_goal.cosine_distance_exit
            ),
            required_confirmations=(
                route_goal.required_confirmations
            ),
            max_evidence_age_s=max_evidence_age_s,
            # Every image in the explicit route is a topological node.
            # RouteProgress decides whether it is the final global node.
            require_final_topological_node=False,
        )
    )


class V70LiveNavNode(Node):
    def __init__(
        self,
        *,
        args: argparse.Namespace,
        planner: PersistentV70Planner,
        camera: UnitreeCameraBuffer,
        route: RouteManifest,
    ) -> None:
        super().__init__("v70_live_navigation")
        self.args = args
        self.planner = planner
        self.camera = camera
        self.route = RouteProgress(route)
        self.goal_supervisor = goal_supervisor_for(
            self.route.current,
            max_evidence_age_s=args.planner_timeout,
        )
        self.odom = OdomHistory()
        navigation_follower_config = replace(
            go2_live_action_follower_config(
                max_odom_age_s=args.odom_max_age,
            ),
            max_vx=float(args.live_max_vx),
            min_vx=min(LIVE_DRIVE_MIN_VX, float(args.live_max_vx)),
            drive_kp=LIVE_DRIVE_KP,
            # The 2026-09-01 run showed that a nominal 3 cm turn assist moved
            # 13-18 cm before DRIVE, then that incidental motion was subtracted
            # from the selected action.  Small headings now drive directly;
            # larger headings use pure yaw before a deliberate forward step.
            turn_tolerance_rad=LIVE_DIRECT_DRIVE_YAW_TOLERANCE_RAD,
            drive_realign_threshold_rad=0.08,
            turn_arc_vx=0.0,
            turn_arc_max_distance_m=0.0,
            # Low and medium heading changes remain a forward gait.  DRIVE
            # steers a bounded arc and only falls back to a stationary turn
            # for unusually large model headings.
            continuous_forward_arc=True,
            continuous_forward_arc_max_yaw_rad=0.27,
            continuous_forward_arc_max_vx=min(
                0.24, float(args.live_max_vx)
            ),
            continuous_forward_arc_completion_yaw_tolerance_rad=0.13,
            drive_min_wz=0.35,
            drive_steering_deadband_rad=0.07,
            drive_braking_deceleration_mps2=0.0,
            drive_completion_tolerance_m=0.005,
            # Never leave a stale forward request armed while Safe Bridge is
            # rejecting it.  A short block is enough evidence to stop this
            # burst and observe again; the bridge itself remains authoritative.
            safety_block_timeout_s=1.20,
            # Distinguish a real first stride from chassis pitch / foot
            # shuffling.  If measured translation does not commit promptly,
            # end only this burst and replan from a fresh camera frame.
            drive_gait_commit_timeout_s=1.00,
            drive_gait_commit_min_progress_m=0.04,
            # Once a stride has committed, do not keep loading the rear legs
            # for the old three-second stall window if odometry stops moving.
            no_progress_timeout_s=1.20,
        )
        navigation_follower_config.validate()
        self.follower = ActionFollower(navigation_follower_config)
        self.scan_follower_config = replace(
            go2_live_action_follower_config(
                max_odom_age_s=args.odom_max_age,
            ),
            # A compact scan step is at most 0.36 rad. If pure yaw makes no
            # measurable progress, skip that auxiliary view quickly instead
            # of spending the normal navigation stall window on it.
            no_progress_timeout_s=1.20,
        )
        self.scan_follower = ActionFollower(self.scan_follower_config)
        self.active_history_scan_enabled = bool(
            args.live
            and args.active_history_scan
            and not args.operator_stop_only
        )
        # Active yaw is now an exception, not the temporal-history source.
        # Startup gets one alignment scan; subsequent plans consume true
        # action-endpoint centre frames unless an explicit conflict/block asks
        # for recovery evidence.
        self.active_scan_pending = self.active_history_scan_enabled
        self.active_scan_reason = "initial_alignment"
        self.scan_pattern = str(args.scan_pattern)
        self.scan_targets = active_history_scan_targets(
            args.scan_step_yaw_rad,
            self.scan_pattern,
        )
        self.scan_expected_frames = sum(
            active_history_scan_capture_slot(index, self.scan_pattern)
            is not None
            for index in range(len(self.scan_targets))
        )
        self.scan_state = "IDLE"
        self.scan_step_index = 0
        self.scan_origin_pose: Pose2D | None = None
        self.scan_images: list[ImageSample] = []
        self.scan_poses: list[PoseSample] = []
        self.planning_history_images: deque[ImageSample] = deque(maxlen=4)
        self.planning_history_poses: deque[PoseSample] = deque(maxlen=4)
        self.planning_history_last_sequence: int | None = None
        self.scan_frame_after = 0.0
        self.scan_frame_wait_started = 0.0
        self.scan_camera_paused = False
        self.scan_annotation_root = (
            args.scan_annotation_root.expanduser().resolve()
        )
        self.scan_annotation_request: dict[str, Any] | None = None
        self.scan_annotation_deadline = 0.0
        self.scan_annotation_requests_created = 0
        self.operator_visual_hint: dict[str, Any] | None = None
        self.operator_hint_directions: list[int] = []
        self.operator_direction_prior = 0
        self.consecutive_plan_holds = 0
        self.outdoor_junction_forward_turn_committed = False
        self.cumulative_scan_translation_m = 0.0
        self.external_control_rebase_pending = False
        self.external_control_rebase_reason: str | None = None
        self.external_control_anchor_pose: Pose2D | None = None
        self.external_control_stable_since: float | None = None
        self.safe_feedback_receive_time: float | None = None
        self.safe_feedback_vx = 0.0
        self.safe_feedback_wz = 0.0
        self.safe_feedback_motion_applied = False

        self.cmd_publisher = self.create_publisher(
            Twist,
            args.cmd_topic,
            10,
        )
        self.enable_publisher = self.create_publisher(
            Bool,
            args.motion_enable_topic,
            10,
        )
        self.create_subscription(
            Odometry,
            args.odom_topic,
            self._odom_callback,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Twist,
            args.safe_cmd_topic,
            self._safe_cmd_callback,
            10,
        )
        self.create_timer(
            args.control_period,
            self._control_timer,
        )

        self.output_dir = args.output_dir.expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.output_dir / "live_nav.jsonl"
        self.status_path = self.output_dir / "status.json"
        self.stop_file = args.stop_file.expanduser().resolve()
        self.arm_file = args.arm_file.expanduser().resolve()
        self.ready_file = args.ready_file.expanduser().resolve()

        self.snapshot_executor: ThreadPoolExecutor | None = (
            ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="v70-plan-images",
            )
            if args.save_plan_images and args.async_plan_images
            else None
        )
        self.snapshot_futures: deque[Future[None]] = deque()

        self.job_queue: queue.Queue[PlanningJob | None] = queue.Queue(
            maxsize=1
        )
        self.output_queue: queue.Queue[PlanningOutput] = queue.Queue()
        self.worker_stop = threading.Event()
        self.worker = threading.Thread(
            target=self._planning_worker,
            name="v70-planning-worker",
            daemon=True,
        )
        self.worker.start()

        self.plan_sequence = 0
        self.plan_inflight = False
        self.inflight_plan_sequence: int | None = None
        self.inflight_plan_settle_prefetch = False
        self.inflight_plan_motion_prefetch = False
        self.inflight_plan_snapshot_pose: Pose2D | None = None
        self.invalidated_plan_sequences: set[int] = set()
        self.next_plan_time = time.monotonic()
        self.last_plan_summary: dict[str, Any] | None = None
        self.last_goal_decision: dict[str, Any] | None = None
        self.started_time = time.monotonic()
        self.last_sensor_ready_time: float | None = None
        self.last_sensor_error: str | None = None
        self.camera_recovery_started: float | None = None
        self.camera_last_restart_time: float | None = None
        self.camera_restart_attempts = 0
        self.camera_recovery_count = 0
        self.last_plan_request_error: str | None = None
        self.fault_reason: str | None = None
        self.last_fault_details: dict[str, Any] | None = None
        self.exit_code = EXIT_OK
        self.done = False
        self.ready = False
        self.shutdown_started = False
        self.shutdown_complete = False
        self.planning_worker_alive_after_join = False
        self.last_status_write = 0.0
        self.last_follower_telemetry_log = 0.0
        self.last_follower_telemetry_reason: str | None = None
        self.interlock_generation = 0
        self.last_interlock = False
        self.approach_mode = False
        self.precision_goal_hold = False
        self.target_visual_locked = False
        self.target_visual_trusted_scale_ratio: float | None = None
        self.target_visual_loss_holds = 0
        self.target_visual_loss_forward_distance_m = 0.0
        self.abstention_forward_distance_m = 0.0
        self.abstention_near_target_step_used = False
        self.target_visual_last_plan_continuity_allowed = True
        self.target_visual_unconfirmed_motion_anchor_pose: Pose2D | None = None
        self.target_visual_recheck_active = False
        self.target_visual_recheck_after = 0.0
        self.target_visual_recheck_count = 0
        self.target_visual_recheck_step_used = False
        self.target_visual_recheck_plan: PlanningOutput | None = None
        self.route_intent_direction = 0
        self.route_intent_endpoint_y_m: float | None = None
        self.route_intent_opposite_streak = 0
        self.fresh_plan_gate = FreshPlanGate()
        self.pending_execution_actions: list[np.ndarray] = []
        self.execution_prefix_step = 0
        self.execution_prefix_audit: dict[str, Any] | None = None
        self.short_buffered_prefetch_allowed = False
        self.planning_gap_bridge_active = False
        self.planning_gap_bridge_sequence: int | None = None
        self.planning_gap_bridge_started_time: float | None = None
        self.settle_prefetch_after: float | None = None
        # A settle-prefetch is beneficial only while the platform actually
        # stays inside the snapshot validity envelope.  One measured drift
        # failure suspends further prefetches for this run; fresh endpoint
        # planning is faster than repeatedly computing plans that are known
        # to be discarded on the current floor/gait state.
        self.settle_prefetch_suspended = False
        self.rolling_prefetch_after: float | None = None
        self.last_command = FollowerCommand.zero(
            FollowerPhase.IDLE,
            "startup",
        )

        self.get_logger().warn(
            "V70 LIVE NAV STARTED: mode=%s operator_stop_only=%s "
            "stop=%s arm=%s route=%s goals=%d; startup output is zero."
            % (
                (
                    "scan_only"
                    if args.scan_only
                    else ("live" if args.live else "shadow")
                ),
                args.operator_stop_only,
                self.stop_file,
                self.arm_file,
                route.route_id,
                len(route.goals),
            )
        )
        if args.operator_stop_only:
            self.get_logger().warn(
                "OPERATOR-STOP-ONLY: automatic arrival completion is "
                "disabled; motion is low-speed and a physical operator "
                "must assert STOP."
            )

    def _odom_callback(self, msg: Odometry) -> None:
        self.odom.append(msg)

    def _safe_cmd_callback(self, msg: Twist) -> None:
        vx = float(msg.linear.x)
        wz = float(msg.angular.z)
        if not math.isfinite(vx) or not math.isfinite(wz):
            self.safe_feedback_receive_time = None
            self.safe_feedback_vx = 0.0
            self.safe_feedback_wz = 0.0
            return
        self.safe_feedback_receive_time = time.monotonic()
        self.safe_feedback_vx = vx
        self.safe_feedback_wz = wz

    def _safe_feedback_fresh(self, now: float) -> bool:
        receive_time = self.safe_feedback_receive_time
        return bool(
            receive_time is not None
            and receive_time <= now
            and now - receive_time
            <= self.args.safe_feedback_max_age
        )

    def _safe_motion_applied(self, now: float) -> bool:
        return safe_velocity_feedback_applied(
            self.last_command,
            safe_vx=self.safe_feedback_vx,
            safe_wz=self.safe_feedback_wz,
            feedback_fresh=self._safe_feedback_fresh(now),
        )

    def _publish(
        self,
        vx: float,
        wz: float,
        *,
        enable: bool,
    ) -> bool:
        if not rclpy.ok():
            return False
        msg = Twist()
        if enable:
            msg.linear.x = float(vx)
            msg.angular.z = float(wz)
        heartbeat = Bool()
        heartbeat.data = bool(enable)
        try:
            self.cmd_publisher.publish(msg)
            self.enable_publisher.publish(heartbeat)
        except Exception as exc:
            self.stop_file.parent.mkdir(parents=True, exist_ok=True)
            self.stop_file.touch(exist_ok=True)
            if self.fault_reason is None:
                self.fault_reason = (
                    "ros_publish_failed:"
                    f"{type(exc).__name__}:{exc}"
                )
                self.exit_code = EXIT_RUNTIME
            self.done = True
            return False
        return True

    def _publish_zero(self) -> bool:
        return self._publish(0.0, 0.0, enable=False)

    def _log(self, event: str, **fields: Any) -> None:
        record = {
            "schema": "v70_live_nav_event_v1",
            "time_unix": time.time(),
            "time_monotonic": time.monotonic(),
            "event": event,
            "route_id": self.route.manifest.route_id,
            "route_index": self.route.index,
            "route_goal": self.route.current.name,
            "route_complete": self.route.complete,
            "route_intent_direction": self.route_intent_direction,
            "route_intent_endpoint_y_m": self.route_intent_endpoint_y_m,
            "route_intent_opposite_streak": self.route_intent_opposite_streak,
            **fields,
        }
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
            handle.write("\n")
            handle.flush()

    @staticmethod
    def _command_payload(command: FollowerCommand) -> dict[str, Any]:
        return {
            "phase": command.phase.value,
            "vx": float(command.vx),
            "wz": float(command.wz),
            "enable": bool(command.enable_motion),
            "complete": bool(command.complete),
            "fault": bool(command.fault),
            "reason": command.reason,
            "progress_m": float(command.progress_m),
            "cross_track_m": float(command.cross_track_m),
            "yaw_error_rad": float(command.yaw_error_rad),
        }

    def _follower_control_evidence(
        self,
        *,
        command: FollowerCommand,
        pose: Pose2D,
        now: float,
        motion_applied: bool,
    ) -> dict[str, Any]:
        return {
            "execution_prefix_step": int(self.execution_prefix_step),
            "pose": {
                "x_m": float(pose.x),
                "y_m": float(pose.y),
                "yaw_rad": float(pose.yaw),
                "odom_age_s": float(max(0.0, now - pose.receive_time)),
            },
            "requested": self._command_payload(command),
            "safe_feedback": {
                "fresh": self._safe_feedback_fresh(now),
                "vx": float(self.safe_feedback_vx),
                "wz": float(self.safe_feedback_wz),
                "motion_applied": bool(motion_applied),
            },
            "follower": self.follower.telemetry(pose, now),
        }

    def _maybe_log_follower_control(
        self,
        *,
        command: FollowerCommand,
        pose: Pose2D,
        now: float,
        motion_applied: bool,
    ) -> dict[str, Any]:
        evidence = self._follower_control_evidence(
            command=command,
            pose=pose,
            now=now,
            motion_applied=motion_applied,
        )
        reason_changed = (
            command.reason != self.last_follower_telemetry_reason
        )
        if (
            reason_changed
            or command.fault
            or now - self.last_follower_telemetry_log
            >= FOLLOWER_TELEMETRY_PERIOD_S
        ):
            self._log("follower_control", **evidence)
            self.last_follower_telemetry_log = now
            self.last_follower_telemetry_reason = command.reason
        return evidence

    def _runtime_state(self) -> str:
        if self.shutdown_complete:
            return "STOPPED"
        if self.shutdown_started:
            return "SHUTTING_DOWN"
        if self.fault_reason is not None:
            return "FAULT"
        if self.route.complete:
            return "ARRIVED"
        if not self.ready:
            error = (self.last_sensor_error or "").lower()
            if "camera" in error:
                return "WAIT_CAMERA"
            if "odom" in error:
                return "WAIT_ODOM"
            return "STARTING"
        if self.plan_inflight:
            return "PLANNING"
        if self.scan_state == "WAIT_ANNOTATION":
            return "WAIT_ANNOTATION"
        if self.scan_state != "IDLE":
            return "SCANNING"
        if self.follower.active:
            return "FOLLOWING"
        return "WAIT_PLAN"

    def _active_scan_required_for_next_plan(self) -> bool:
        if self.active_history_scan_enabled and not self.args.scan_only:
            return bool(
                self.scan_state != "IDLE" or self.active_scan_pending
            )
        return active_scan_required_for_plan(
            enabled=self.active_history_scan_enabled,
            annotation_enabled=bool(self.args.scan_annotation),
            requests_created=self.scan_annotation_requests_created,
            max_rounds=self.args.scan_annotation_max_rounds,
            scan_in_progress=self.scan_state != "IDLE",
        )

    def _write_status(
        self,
        now: float,
        *,
        force: bool = False,
    ) -> None:
        if not force and now - self.last_status_write < 1.0:
            return
        self.last_status_write = now
        payload = {
            "schema": "v70_live_nav_status_v1",
            "updated_unix": time.time(),
            "mode": (
                "scan_only"
                if self.args.scan_only
                else ("live" if self.args.live else "shadow")
            ),
            "scan_only": self.args.scan_only,
            "operator_stop_only": self.args.operator_stop_only,
            "manual_stop_required": self.args.operator_stop_only,
            "state": self._runtime_state(),
            "ready": self.ready,
            "fault": self.fault_reason,
            "exit_code": self.exit_code,
            "last_sensor_error": self.last_sensor_error,
            "camera": {
                **self.camera.diagnostics(),
                "recovery_active": self.camera_recovery_started is not None,
                "restart_attempts": self.camera_restart_attempts,
                "recovery_count": self.camera_recovery_count,
            },
            "last_plan_request_error": self.last_plan_request_error,
            "shutdown_started": self.shutdown_started,
            "shutdown_complete": self.shutdown_complete,
            "planning_worker_alive_after_join": (
                self.planning_worker_alive_after_join
            ),
            "stop_present": self.stop_file.exists(),
            "arm_present": self.arm_file.is_file(),
            "plan_inflight": self.plan_inflight,
            "inflight_plan_sequence": self.inflight_plan_sequence,
            "inflight_plan_settle_prefetch": (
                self.inflight_plan_settle_prefetch
            ),
            "inflight_plan_motion_prefetch": (
                self.inflight_plan_motion_prefetch
            ),
            "settle_plan_prefetch": {
                "enabled": bool(self.args.settle_plan_prefetch),
                "suspended_after_drift": bool(
                    self.settle_prefetch_suspended
                ),
                "effective": bool(
                    self.args.settle_plan_prefetch
                    and not self.settle_prefetch_suspended
                ),
                "waiting_for_endpoint_frame": bool(
                    self.settle_prefetch_after is not None
                    and not self.plan_inflight
                ),
                "snapshot_writer_pending": len(
                    self.snapshot_futures
                ),
            },
            "rolling_plan_prefetch": {
                "enabled": bool(self.args.rolling_plan_prefetch),
                "action_frame_after": self.rolling_prefetch_after,
                "max_handoff_age_s": ROLLING_PREFETCH_MAX_AGE_S,
                "short_buffer_allowed": bool(
                    self.short_buffered_prefetch_allowed
                ),
                "gap_bridge_active": bool(
                    self.planning_gap_bridge_active
                ),
                "gap_bridge_sequence": (
                    self.planning_gap_bridge_sequence
                ),
            },
            "active_history_scan": {
                "enabled": self.active_history_scan_enabled,
                "policy": (
                    "disabled_operator_stop_only"
                    if self.args.operator_stop_only
                    else "startup_then_on_demand"
                ),
                "pending": self.active_scan_pending,
                "pending_reason": self.active_scan_reason,
                "required_for_next_plan": (
                    self._active_scan_required_for_next_plan()
                ),
                "history_source_after_annotation": (
                    "rolling_camera_four_frame"
                    if self.args.scan_annotation
                    and self.scan_annotation_requests_created
                    >= self.args.scan_annotation_max_rounds
                    and self.scan_state == "IDLE"
                    else "action_endpoint_center_frames"
                ),
                "planner_temporal_frames": len(
                    self.planning_history_images
                ),
                "visual_scan_frames": len(self.scan_images),
                "state": self.scan_state,
                "step_index": self.scan_step_index,
                "captured_frames": len(self.scan_images),
                "expected_frames": self.scan_expected_frames,
                "step_yaw_rad": self.args.scan_step_yaw_rad,
                "pattern": self.scan_pattern,
                "left_min_wz": self.args.scan_left_min_wz,
                "turn_arc_vx": self.args.scan_turn_arc_vx,
                "turn_arc_max_distance_m": (
                    self.args.scan_turn_arc_max_distance_m
                ),
                "right_min_wz": self.args.scan_right_min_wz,
                "consecutive_plan_holds": self.consecutive_plan_holds,
                "max_consecutive_plan_holds": (
                    self.args.max_consecutive_plan_holds
                ),
                "cumulative_translation_m": (
                    self.cumulative_scan_translation_m
                ),
                "max_cumulative_translation_m": (
                    self.args.max_cumulative_scan_translation_m
                ),
                "annotation_enabled": bool(self.args.scan_annotation),
                "annotation_rounds_completed": (
                    self.scan_annotation_requests_created
                ),
                "annotation_max_rounds": (
                    self.args.scan_annotation_max_rounds
                ),
                "annotation_phase_complete": bool(
                    self.scan_annotation_requests_created
                    >= self.args.scan_annotation_max_rounds
                ),
                "annotation_waiting": self.scan_state == "WAIT_ANNOTATION",
                "annotation_request_id": (
                    self.scan_annotation_request.get("request_id")
                    if self.scan_annotation_request is not None
                    else None
                ),
                "operator_visual_hint": self.operator_visual_hint,
                "operator_direction_prior": self.operator_direction_prior,
                "operator_hint_directions": list(
                    self.operator_hint_directions
                ),
                "external_control_rebase_pending": (
                    self.external_control_rebase_pending
                ),
                "external_control_rebase_reason": (
                    self.external_control_rebase_reason
                ),
            },
            "route_id": self.route.manifest.route_id,
            "route_index": self.route.index,
            "route_goal": self.route.current.name,
            "route_final": self.route.is_final,
            "route_complete": self.route.complete,
            "route_intent_direction": self.route_intent_direction,
            "route_intent_endpoint_y_m": self.route_intent_endpoint_y_m,
            "follower_phase": self.follower.phase.value,
            "approach_mode": self.approach_mode,
            "precision_goal_hold": self.precision_goal_hold,
            "target_visual_continuity": {
                "locked": self.target_visual_locked,
                "trusted_scale_ratio": (
                    self.target_visual_trusted_scale_ratio
                ),
                "loss_holds": self.target_visual_loss_holds,
                "loss_forward_distance_m": (
                    self.target_visual_loss_forward_distance_m
                ),
                "last_plan_continuity_allowed": bool(
                    self.target_visual_last_plan_continuity_allowed
                ),
                "unconfirmed_motion_anchor_active": bool(
                    self.target_visual_unconfirmed_motion_anchor_pose
                    is not None
                ),
            },
            "last_command": {
                "vx": self.last_command.vx,
                "wz": self.last_command.wz,
                "enable": self.last_command.enable_motion,
                "phase": self.last_command.phase.value,
                "reason": self.last_command.reason,
                "complete": self.last_command.complete,
                "fault": self.last_command.fault,
                "progress_m": self.last_command.progress_m,
                "cross_track_m": self.last_command.cross_track_m,
                "yaw_error_rad": self.last_command.yaw_error_rad,
            },
            "safe_feedback": {
                "fresh": self._safe_feedback_fresh(now),
                "vx": self.safe_feedback_vx,
                "wz": self.safe_feedback_wz,
                "motion_applied": (
                    self.safe_feedback_motion_applied
                ),
            },
            "last_fault_details": self.last_fault_details,
            "last_plan": self.last_plan_summary,
            "last_goal_decision": self.last_goal_decision,
        }
        temporary = self.status_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        temporary.replace(self.status_path)

    def _latch_fault(
        self,
        reason: str,
        exit_code: int = EXIT_RUNTIME,
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        if self.fault_reason is not None:
            return
        self.fault_reason = reason
        self.exit_code = int(exit_code)
        self.done = True
        self.last_fault_details = details
        self.follower.reset(reason)
        self.pending_execution_actions.clear()
        self.execution_prefix_step = 0
        self.execution_prefix_audit = None
        self.short_buffered_prefetch_allowed = False
        self.planning_gap_bridge_active = False
        self.planning_gap_bridge_started_time = None
        self.settle_prefetch_after = None
        self.stop_file.parent.mkdir(parents=True, exist_ok=True)
        self.stop_file.touch(exist_ok=True)
        self._publish_zero()
        if (
            self.args.scan_only
            and self.scan_images
            and self.scan_origin_pose is not None
        ):
            try:
                capture_targets = tuple(
                    float(target)
                    for index, target in enumerate(self.scan_targets)
                    if active_history_scan_capture_slot(
                        index,
                        self.scan_pattern,
                    )
                    is not None
                )[: len(self.scan_images)]
                partial_manifest = write_partial_scan_artifacts(
                    output_dir=self.output_dir / "partial_scan",
                    pattern=self.scan_pattern,
                    step_yaw_rad=self.args.scan_step_yaw_rad,
                    scan_targets=self.scan_targets,
                    capture_target_yaw_offsets=capture_targets,
                    images=[sample.image for sample in self.scan_images],
                    source_sha256=[
                        sample.sha256 for sample in self.scan_images
                    ],
                    poses=[
                        {
                            "receive_time_monotonic": pose.receive_time,
                            "x_m": pose.x,
                            "y_m": pose.y,
                            "yaw_rad": pose.yaw,
                        }
                        for pose in self.scan_poses
                    ],
                    origin_pose={
                        "receive_time_monotonic": (
                            self.scan_origin_pose.receive_time
                        ),
                        "x_m": self.scan_origin_pose.x,
                        "y_m": self.scan_origin_pose.y,
                        "yaw_rad": self.scan_origin_pose.yaw,
                    },
                    failure_reason=reason,
                    failed_step_index=self.scan_step_index,
                    stop_asserted=self.stop_file.is_file(),
                )
                self._log(
                    "active_history_scan_partial_artifacts_saved",
                    artifact_dir=str(self.output_dir / "partial_scan"),
                    manifest=partial_manifest,
                )
            except Exception as exc:
                try:
                    self._log(
                        "active_history_scan_partial_artifacts_failed",
                        reason=f"{type(exc).__name__}:{exc}",
                    )
                except Exception:
                    pass
        self._reset_active_scan(reason)
        try:
            self._log(
                "fault_latched",
                reason=reason,
                details=self.last_fault_details,
            )
        except Exception:
            pass
        try:
            self._write_status(time.monotonic(), force=True)
        except Exception:
            pass
        try:
            self.get_logger().fatal(
                f"V70 navigation fault latched: {reason}"
            )
        except Exception:
            pass

    def _reset_active_scan(self, reason: str) -> None:
        was_active = self.scan_state != "IDLE" or bool(self.scan_images)
        self.scan_follower.reset(reason)
        self.scan_state = "IDLE"
        self.scan_step_index = 0
        self.scan_origin_pose = None
        self.scan_images.clear()
        self.scan_poses.clear()
        self.scan_frame_after = 0.0
        self.scan_frame_wait_started = 0.0
        self.scan_camera_paused = False
        self.scan_annotation_request = None
        self.scan_annotation_deadline = 0.0
        if was_active:
            try:
                self._log("active_history_scan_reset", reason=reason)
            except Exception:
                pass

    def _request_adaptive_scan(self, reason: str) -> None:
        """Request one bounded recovery scan before the next plan."""

        if (
            self.args.operator_stop_only
            or not self.active_history_scan_enabled
            or self.args.scan_only
        ):
            return
        already_pending = self.active_scan_pending
        self.active_scan_pending = True
        self.active_scan_reason = str(reason)
        if not already_pending:
            self._log("adaptive_visual_scan_requested", reason=reason)

    def _start_scan_step(self, latest: Pose2D, now: float) -> None:
        if self.scan_origin_pose is None:
            raise RuntimeError("active scan has no origin pose")
        if not 0 <= self.scan_step_index < len(self.scan_targets):
            raise RuntimeError("active scan step index is invalid")
        target_offset = self.scan_targets[self.scan_step_index]
        target_yaw = wrap_angle(
            self.scan_origin_pose.yaw + target_offset
        )
        action_yaw = wrap_angle(target_yaw - latest.yaw)
        scan_min_wz = active_history_scan_min_wz(
            action_yaw,
            base_min_wz=float(self.args.scan_left_min_wz),
            right_min_wz=float(self.args.scan_right_min_wz),
            max_wz=self.scan_follower_config.max_wz,
        )
        scan_config = replace(
            self.scan_follower_config,
            min_wz=scan_min_wz,
            turn_arc_vx=float(self.args.scan_turn_arc_vx),
            turn_arc_max_distance_m=float(
                self.args.scan_turn_arc_max_distance_m
            ),
        )
        scan_config.validate()
        self.scan_follower = ActionFollower(scan_config)
        start = self.scan_follower.start(
            np.asarray([0.0, 0.0, action_yaw], dtype=np.float32),
            latest,
            now,
        )
        self.last_command = start
        if start.fault:
            raise RuntimeError(
                f"scan_step_rejected:{self.scan_step_index}:{start.reason}"
            )
        self.scan_state = "TURNING"
        self._log(
            "active_history_scan_step_started",
            scan_step=self.scan_step_index,
            target_offset_yaw_rad=target_offset,
            action_yaw_rad=action_yaw,
            commanded_min_wz=scan_config.min_wz,
            capture_slot=active_history_scan_capture_slot(
                self.scan_step_index,
                self.scan_pattern,
            ),
        )

    def _start_active_scan(self, latest: Pose2D, now: float) -> None:
        self._reset_active_scan("new_scan")
        # A hint is valid for exactly one scan/plan pair and must never leak
        # into a later planning cycle.
        self.operator_visual_hint = None
        self.scan_origin_pose = latest
        self.scan_state = "STARTING"
        self._log(
            "active_history_scan_started",
            origin_yaw_rad=latest.yaw,
            target_offsets_yaw_rad=list(self.scan_targets),
            scan_pattern=self.scan_pattern,
            left_min_wz=self.args.scan_left_min_wz,
            turn_arc_vx=self.args.scan_turn_arc_vx,
            turn_arc_max_distance_m=(
                self.args.scan_turn_arc_max_distance_m
            ),
            right_min_wz=self.args.scan_right_min_wz,
            trigger_reason=self.active_scan_reason,
            expected_frames=self.scan_expected_frames,
        )
        self._start_scan_step(latest, now)

    def _finish_active_scan(self, latest: Pose2D) -> None:
        if (
            len(self.scan_images) != self.scan_expected_frames
            or len(self.scan_poses) != self.scan_expected_frames
        ):
            raise RuntimeError(
                "active scan completed without expected frames:"
                f"expected={self.scan_expected_frames}:"
                f"images={len(self.scan_images)}:poses={len(self.scan_poses)}"
            )
        origin = self.scan_origin_pose
        if origin is None:
            raise RuntimeError("active scan has no origin pose")
        self.scan_step_index = len(self.scan_targets)
        self.scan_state = "READY"
        self._publish_zero()
        # The four scan images are auxiliary panoramic evidence only.  Wait
        # for a new forward-facing frame after the final return-to-center turn
        # before extending the planner's true temporal motion history.
        center_frame_after = time.monotonic()
        self.scan_frame_after = center_frame_after
        self.scan_frame_wait_started = center_frame_after
        final_yaw_error = wrap_angle(latest.yaw - origin.yaw)
        translation_drift = math.hypot(
            latest.x - origin.x,
            latest.y - origin.y,
        )
        self.cumulative_scan_translation_m += translation_drift
        if self.args.scan_only:
            # Fail closed before any artifact write.  The next control tick
            # cannot enter planning because done is latched below.
            self.stop_file.parent.mkdir(parents=True, exist_ok=True)
            self.stop_file.touch(exist_ok=True)
        self._log(
            "active_history_scan_ready",
            image_sha256=[sample.sha256 for sample in self.scan_images],
            final_yaw_error_rad=final_yaw_error,
            translation_drift_m=translation_drift,
            cumulative_translation_m=self.cumulative_scan_translation_m,
            planner_center_frame_after_monotonic=center_frame_after,
        )
        drift_fault = bounded_plan_hold_fault_reason(
            consecutive_holds=self.consecutive_plan_holds,
            max_consecutive_holds=self.args.max_consecutive_plan_holds,
            cumulative_scan_translation_m=(
                self.cumulative_scan_translation_m
            ),
            max_cumulative_scan_translation_m=(
                self.args.max_cumulative_scan_translation_m
            ),
        )
        if drift_fault is not None:
            if drift_fault.startswith("scan_translation_budget_exhausted:"):
                self._begin_external_control_rebase(
                    latest,
                    time.monotonic(),
                    reason=drift_fault,
                )
                self._record_plan_hold("scan_translation_pose_rebase")
                return
            raise RuntimeError(drift_fault)
        if not self.args.scan_only:
            if scan_annotation_required(
                enabled=self.args.scan_annotation,
                requests_created=self.scan_annotation_requests_created,
                max_rounds=self.args.scan_annotation_max_rounds,
            ):
                yaw_offsets = [
                    wrap_angle(pose.yaw - origin.yaw)
                    for pose in self.scan_poses
                ]
                request_id = (
                    f"{self.output_dir.name}-plan-{self.plan_sequence:06d}"
                )
                self.scan_annotation_request = (
                    create_scan_annotation_request(
                        root=self.scan_annotation_root,
                        request_id=request_id,
                        images=[sample.image for sample in self.scan_images],
                        source_sha256=[
                            sample.sha256 for sample in self.scan_images
                        ],
                        yaw_offsets_rad=yaw_offsets,
                        timeout_s=self.args.scan_annotation_timeout,
                    )
                )
                self.scan_annotation_deadline = (
                    time.monotonic() + self.args.scan_annotation_timeout
                )
                self.scan_annotation_requests_created += 1
                self.scan_state = "WAIT_ANNOTATION"
                self._log(
                    "scan_annotation_wait_started",
                    request_id=request_id,
                    timeout_s=self.args.scan_annotation_timeout,
                    annotation_round=(
                        self.scan_annotation_requests_created
                    ),
                    annotation_max_rounds=(
                        self.args.scan_annotation_max_rounds
                    ),
                )
            elif self.args.scan_annotation:
                self._log(
                    "scan_annotation_skipped_autonomous_phase",
                    completed_annotation_rounds=(
                        self.scan_annotation_requests_created
                    ),
                    annotation_max_rounds=(
                        self.args.scan_annotation_max_rounds
                    ),
                )
            return

        capture_targets = tuple(
            float(target)
            for index, target in enumerate(self.scan_targets)
            if active_history_scan_capture_slot(
                index,
                self.scan_pattern,
            )
            is not None
        )
        artifact_dir = self.output_dir / "scan_only"
        manifest = write_scan_only_artifacts(
            output_dir=artifact_dir,
            pattern=self.scan_pattern,
            step_yaw_rad=self.args.scan_step_yaw_rad,
            scan_targets=self.scan_targets,
            capture_target_yaw_offsets=capture_targets,
            images=[sample.image for sample in self.scan_images],
            source_sha256=[sample.sha256 for sample in self.scan_images],
            poses=[
                {
                    "receive_time_monotonic": pose.receive_time,
                    "x_m": pose.x,
                    "y_m": pose.y,
                    "yaw_rad": pose.yaw,
                }
                for pose in self.scan_poses
            ],
            origin_pose={
                "receive_time_monotonic": origin.receive_time,
                "x_m": origin.x,
                "y_m": origin.y,
                "yaw_rad": origin.yaw,
            },
            stop_asserted=self.stop_file.is_file(),
        )
        self.last_command = FollowerCommand.zero(
            self.scan_follower.phase,
            "scan_only_complete",
        )
        self._log(
            "active_history_scan_only_complete",
            artifact_dir=str(artifact_dir),
            manifest=manifest,
        )
        self.exit_code = EXIT_OK
        self.done = True

    def _complete_scan_step(self, latest: Pose2D, now: float) -> None:
        capture_slot = active_history_scan_capture_slot(
            self.scan_step_index,
            self.scan_pattern,
        )
        self.scan_follower.reset("scan_step_complete")
        if capture_slot is not None:
            self.scan_state = "WAIT_FRAME"
            self.scan_frame_after = now
            self.scan_frame_wait_started = now
            self._publish_zero()
            return
        next_step = active_history_scan_next_step(
            self.scan_step_index,
            self.scan_pattern,
        )
        if next_step is None:
            self._finish_active_scan(latest)
            return
        self.scan_step_index = next_step
        self._start_scan_step(latest, now)

    def _capture_scan_frame(self, latest: Pose2D, now: float) -> None:
        try:
            sample = self.camera.latest_after(
                receive_time=self.scan_frame_after,
                max_age_s=self.args.camera_max_age,
            )
        except Exception:
            if now - self.scan_frame_wait_started > self.args.scan_frame_timeout:
                raise RuntimeError(
                    f"scan_frame_timeout:step={self.scan_step_index}"
                )
            self._publish_zero()
            return
        slot = active_history_scan_capture_slot(
            self.scan_step_index,
            self.scan_pattern,
        )
        if slot is None or slot != len(self.scan_images):
            raise RuntimeError("active scan frame order contract failed")
        self.scan_images.append(sample)
        self.scan_poses.append(
            PoseSample(
                receive_time=latest.receive_time,
                x=latest.x,
                y=latest.y,
                yaw=latest.yaw,
            )
        )
        assert self.scan_origin_pose is not None
        self._log(
            "active_history_scan_frame_captured",
            scan_step=self.scan_step_index,
            capture_slot=slot,
            image_sha256=sample.sha256,
            yaw_offset_rad=wrap_angle(
                latest.yaw - self.scan_origin_pose.yaw
            ),
        )
        next_step = active_history_scan_next_step(
            self.scan_step_index,
            self.scan_pattern,
        )
        if next_step is None:
            self._finish_active_scan(latest)
            return
        self.scan_step_index = next_step
        self._start_scan_step(latest, now)

    def _update_active_scan(
        self,
        *,
        latest: Pose2D,
        now: float,
        interlock: bool,
    ) -> None:
        if not interlock:
            self._reset_active_scan("interlock_closed")
            self._publish_zero()
            return
        if self.scan_state == "IDLE":
            if now < self.next_plan_time:
                self._publish_zero()
                return
            self._start_active_scan(latest, now)
        if self.scan_state == "WAIT_FRAME":
            self._capture_scan_frame(latest, now)
            return
        if self.scan_state == "WAIT_ANNOTATION":
            self._publish_zero()
            if self.scan_annotation_request is None:
                raise RuntimeError("scan annotation request is missing")
            if now >= self.scan_annotation_deadline:
                raise RuntimeError("scan_annotation_timeout")
            hint = read_operator_hint(
                root=self.scan_annotation_root,
                request=self.scan_annotation_request,
            )
            if hint is None:
                return
            self.operator_visual_hint = hint
            hint_direction = int(hint.get("direction", 0))
            if bool(hint.get("visible")) and hint_direction in (-1, 1):
                self.operator_hint_directions.append(hint_direction)
            self.operator_direction_prior = derive_operator_direction_prior(
                self.operator_hint_directions,
                required=self.args.scan_annotation_max_rounds,
            )
            self.scan_state = "READY"
            self._log(
                "scan_annotation_received",
                operator_visual_hint=hint,
                operator_hint_directions=list(
                    self.operator_hint_directions
                ),
                operator_direction_prior=(
                    self.operator_direction_prior
                ),
            )
            return
        if self.scan_state == "READY":
            self._publish_zero()
            self._request_plan(now)
            return
        if self.scan_state != "TURNING":
            raise RuntimeError(f"invalid active scan state: {self.scan_state}")
        motion_applied = self._safe_motion_applied(now)
        self.safe_feedback_motion_applied = motion_applied
        command = self.scan_follower.update(
            latest,
            now,
            interlock_allows_motion=interlock,
            motion_applied=motion_applied,
        )
        self.last_command = command
        self._publish(
            command.vx,
            command.wz,
            enable=command.enable_motion,
        )
        if command.fault:
            if (
                active_history_scan_fault_is_recoverable(command.reason)
                or (
                    self.args.aggressive_navigation
                    and command.reason == "turn_no_progress"
                )
            ):
                # A frame captured after a failed turn does not represent its
                # named left/centre/right slot.  The old implementation still
                # accepted it, which fabricated panoramic evidence and was
                # observed to steer the 06:27 run toward the door frame.
                # Abort this auxiliary scan and plan once from the actual
                # current centre pose; Safe Bridge remains authoritative.
                self._log(
                    "adaptive_visual_scan_aborted",
                    scan_step=self.scan_step_index,
                    reason=command.reason,
                    capture_slot=active_history_scan_capture_slot(
                        self.scan_step_index,
                        self.scan_pattern,
                    ),
                    captured_frames=len(self.scan_images),
                )
                self._reset_active_scan(
                    f"unreached_semantic_view:{command.reason}"
                )
                self.active_scan_pending = False
                self.active_scan_reason = ""
                self.next_plan_time = now
                return
            raise RuntimeError(
                f"scan_follower:{self.scan_step_index}:{command.reason}"
            )
        if command.complete:
            self._complete_scan_step(latest, now)

    def _write_plan_images(self, job: PlanningJob) -> None:
        destination = (
            self.output_dir / "snapshots" / f"{job.sequence:06d}"
        )
        destination.mkdir(parents=True, exist_ok=False)
        history_paths: list[Path] = []
        for index, image in enumerate(job.observations):
            path = destination / f"history_{index}.jpg"
            image.save(path, quality=95)
            history_paths.append(path)
        if job.visual_image_sha256 == job.image_sha256:
            for index, source in enumerate(history_paths):
                os.link(
                    source,
                    destination / f"visual_scan_{index}.jpg",
                )
        else:
            for index, image in enumerate(job.visual_observations):
                image.save(
                    destination / f"visual_scan_{index}.jpg",
                    quality=95,
                )
        with Image.open(job.goal_path) as image:
            image.convert("RGB").save(
                destination / "goal.jpg",
                quality=95,
            )

    @staticmethod
    def _json_array(value: Any) -> Any:
        """Convert the small planning audit arrays to JSON-safe values."""
        if isinstance(value, np.ndarray):
            return V70LiveNavNode._json_array(value.tolist())
        if isinstance(value, np.generic):
            return V70LiveNavNode._json_array(value.item())
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if isinstance(value, dict):
            return {
                str(key): V70LiveNavNode._json_array(item)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [V70LiveNavNode._json_array(item) for item in value]
        return value

    def _build_plan_decision_payload(
        self,
        sequence: int,
        result: dict[str, Any],
        summary: dict[str, Any],
    ) -> dict[str, Any]:
        """Copy only small candidate-level evidence out of the planner result."""
        pools: list[dict[str, Any]] = []

        def append_pool(
            name: str,
            source_round: int,
            candidates: Any,
            evaluation: Any,
        ) -> None:
            candidate_array = np.asarray(candidates, dtype=np.float32)
            if candidate_array.ndim == 4 and candidate_array.shape[0] == 1:
                candidate_array = candidate_array[0]
            audit = evaluation if isinstance(evaluation, dict) else {}
            energy = np.asarray(audit.get("energy", []), dtype=np.float32)
            safe = np.asarray(audit.get("safe", []), dtype=bool)
            planning_safe = np.asarray(
                audit.get("planning_safe", safe), dtype=bool
            )
            if energy.ndim == 2 and energy.shape[0] == 1:
                energy = energy[0]
            if safe.ndim == 2 and safe.shape[0] == 1:
                safe = safe[0]
            if planning_safe.ndim == 2 and planning_safe.shape[0] == 1:
                planning_safe = planning_safe[0]
            pools.append(
                {
                    "name": name,
                    "source_round": int(source_round),
                    "candidate_waypoints_m": (
                        candidate_array
                        * float(persistent_planner_module.SPACING_M)
                    ).tolist(),
                    "energy_lower_is_better": energy.tolist(),
                    "model_safe": safe.tolist(),
                    "planning_safe": planning_safe.tolist(),
                }
            )

        append_pool(
            "fixed",
            -1,
            result.get("base_candidates", []),
            result.get("fixed", {}),
        )
        for round_audit in result.get("rounds", []):
            append_pool(
                f"cem_{int(round_audit.get('iteration', 0))}",
                int(round_audit.get("iteration", 0)),
                round_audit.get("candidates", []),
                round_audit.get("evaluation", {}),
            )

        return self._json_array({
            "schema": "v70_goal_only_plan_decision_v1",
            "sequence": int(sequence),
            "score_semantics": (
                "lower energy is better; only planning_safe candidates "
                "are executable"
            ),
            "selected_source_round": int(
                result.get("archive_source_round", -2)
            ),
            "selected_source_candidate": int(
                result.get("archive_source_candidate", 0)
            ),
            "selected_waypoints_h4_m": summary.get(
                "selected_waypoints_h4_m", []
            ),
            "selected_actions_h4": summary.get("selected_actions_h4", []),
            "selection_reason": summary.get("selection_reason"),
            "selected_energy": summary.get("archive_energy"),
            "abstained": bool(summary.get("abstained", False)),
            "pools": pools,
        })

    def _write_plan_decision_artifact(
        self,
        sequence: int,
        payload: dict[str, Any],
    ) -> None:
        """Persist a pre-extracted audit without retaining model tensors."""
        destination = self.output_dir / "snapshots" / f"{sequence:06d}"
        destination.mkdir(parents=True, exist_ok=True)
        temporary = destination / "decision.json.tmp"
        temporary.write_text(
            json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(destination / "decision.json")

    def _queue_plan_decision_artifact(
        self,
        sequence: int,
        result: dict[str, Any],
        summary: dict[str, Any],
    ) -> None:
        if not self.args.save_plan_images:
            return
        payload = self._build_plan_decision_payload(
            sequence, result, summary
        )
        self._reap_plan_image_futures()
        if self.snapshot_executor is None:
            self._write_plan_decision_artifact(sequence, payload)
        else:
            self.snapshot_futures.append(
                self.snapshot_executor.submit(
                    self._write_plan_decision_artifact,
                    sequence,
                    payload,
                )
            )

    def _reap_plan_image_futures(self, *, block: bool = False) -> None:
        while self.snapshot_futures and (
            block or self.snapshot_futures[0].done()
        ):
            self.snapshot_futures.popleft().result()

    def _snapshot_job(
        self,
        *,
        minimum_camera_receive_time: float | None = None,
        settle_prefetch: bool = False,
        motion_prefetch: bool = False,
    ) -> PlanningJob:
        latest = self.odom.latest()
        if latest is None:
            raise RuntimeError("odometry is unavailable at planning snapshot")
        if (
            self.active_history_scan_enabled
            and self.last_interlock
            and self.scan_state == "READY"
        ):
            if (
                len(self.scan_images) != self.scan_expected_frames
                or len(self.scan_poses) != self.scan_expected_frames
            ):
                raise RuntimeError(
                    "active visual scan frame count mismatch:"
                    f"expected={self.scan_expected_frames}"
                )
            visual_samples = tuple(self.scan_images)
            visual_poses = tuple(self.scan_poses)
            try:
                center_sample = self.camera.latest_after(
                    receive_time=self.scan_frame_after,
                    max_age_s=self.args.camera_max_age,
                )
            except Exception as exc:
                if (
                    time.monotonic() - self.scan_frame_wait_started
                    > self.args.scan_frame_timeout
                ):
                    raise RuntimeError(
                        "post_scan_center_frame_timeout"
                    ) from exc
                raise RuntimeError(
                    "waiting_for_post_scan_center_frame"
                ) from exc
            if self.planning_history_last_sequence != self.plan_sequence:
                self.planning_history_images.append(center_sample)
                self.planning_history_poses.append(
                    PoseSample(
                        receive_time=latest.receive_time,
                        x=latest.x,
                        y=latest.y,
                        yaw=latest.yaw,
                    )
                )
                self.planning_history_last_sequence = self.plan_sequence
            image_samples = left_pad_temporal_history(
                tuple(self.planning_history_images),
                count=4,
            )
            poses = left_pad_temporal_history(
                tuple(self.planning_history_poses),
                count=4,
            )
            history_source = "action_endpoint_center_frames"
        elif self.last_interlock:
            # Normal live cycles do not move the body for visual collection.
            # Add exactly one fresh centre image after the preceding physical
            # action, pad the true endpoint sequence to NoMaD's four-frame
            # temporal contract, and use only the current centre view for the
            # optional visual-bearing estimator.  It has a strict four-frame
            # temporal interface, so provide the same real endpoint history
            # used by NoMaD.  Duplicate left-padding is rejected in the worker
            # until at least two distinct endpoints exist.
            if minimum_camera_receive_time is not None:
                camera_method = (
                    self.camera.newest_after
                    if motion_prefetch
                    else self.camera.latest_after
                )
                center_sample = camera_method(
                    receive_time=minimum_camera_receive_time,
                    max_age_s=self.args.camera_max_age,
                )
            else:
                center_sample = self.camera.snapshot(
                    count=1,
                    max_age_s=self.args.camera_max_age,
                )[-1]
            if self.planning_history_last_sequence != self.plan_sequence:
                self.planning_history_images.append(center_sample)
                self.planning_history_poses.append(
                    PoseSample(
                        receive_time=latest.receive_time,
                        x=latest.x,
                        y=latest.y,
                        yaw=latest.yaw,
                    )
                )
                self.planning_history_last_sequence = self.plan_sequence
            image_samples = left_pad_temporal_history(
                tuple(self.planning_history_images),
                count=4,
            )
            poses = left_pad_temporal_history(
                tuple(self.planning_history_poses),
                count=4,
            )
            visual_samples = tuple(image_samples)
            visual_poses = tuple(poses)
            history_source = "action_endpoint_center_frames_no_scan"
        else:
            image_samples = self.camera.snapshot(
                count=4,
                max_age_s=self.args.camera_max_age,
            )
            poses = self.odom.snapshot(
                count=4,
                interval_s=self.args.odom_interval,
                max_age_s=self.args.odom_max_age,
            )
            visual_samples = tuple(image_samples)
            visual_poses = tuple(poses)
            history_source = "rolling_camera_four_frame"
        # latest_after() can wait for a new camera frame. Pair that frame with
        # odometry sampled after the wait instead of the pose from entry.
        captured_pose = self.odom.latest()
        if captured_pose is None:
            raise RuntimeError("odometry unavailable after camera snapshot")
        latest = captured_pose
        goal = self.route.current
        now = time.monotonic()
        job = PlanningJob(
            sequence=self.plan_sequence,
            route_index=self.route.index,
            goal_name=goal.name,
            goal_path=goal.image,
            goal_sha256=sha256_file(goal.image),
            observations=tuple(
                sample.image.copy() for sample in image_samples
            ),
            image_sha256=tuple(
                sample.sha256 for sample in image_samples
            ),
            poses=tuple(poses),
            visual_observations=tuple(
                sample.image.copy() for sample in visual_samples
            ),
            visual_image_sha256=tuple(
                sample.sha256 for sample in visual_samples
            ),
            visual_poses=tuple(visual_poses),
            history_source=history_source,
            snapshot_time=now,
            snapshot_pose=Pose2D(
                receive_time=latest.receive_time,
                x=latest.x,
                y=latest.y,
                yaw=latest.yaw,
            ),
            interlock_generation=self.interlock_generation,
            motion_authorized_at_snapshot=self.last_interlock,
            route_intent_direction=(
                self.route_intent_direction
                if self.args.operator_stop_only
                else 0
            ),
            operator_visual_hint=(
                dict(self.operator_visual_hint)
                if self.operator_visual_hint is not None
                else None
            ),
            operator_direction_prior=int(
                self.operator_direction_prior
            ),
            settle_prefetch=bool(settle_prefetch),
            motion_prefetch=bool(motion_prefetch),
            visual_recheck_only=bool(
                self.target_visual_recheck_active
                and self.args.operator_stop_only
            ),
        )
        if self.args.save_plan_images:
            self._reap_plan_image_futures()
            if self.snapshot_executor is None:
                self._write_plan_images(job)
            else:
                self.snapshot_futures.append(
                    self.snapshot_executor.submit(
                        self._write_plan_images,
                        job,
                    )
                )
        return job

    def _request_plan(
        self,
        now: float,
        *,
        allow_follower_settle: bool = False,
        allow_follower_motion: bool = False,
        minimum_camera_receive_time: float | None = None,
    ) -> None:
        if (
            minimum_camera_receive_time is None
            and self.target_visual_recheck_active
        ):
            minimum_camera_receive_time = self.target_visual_recheck_after
        motion_prefetch_allowed = bool(
            allow_follower_motion
            and self.args.rolling_plan_prefetch
            and self.follower.phase in {
                FollowerPhase.TURN,
                FollowerPhase.DRIVE,
            }
            and (
                not self.pending_execution_actions
                or self.short_buffered_prefetch_allowed
            )
        )
        follower_blocks_planning = bool(
            self.follower.active
            and not (
                allow_follower_settle
                and self.args.settle_plan_prefetch
                and self.follower.phase is FollowerPhase.SETTLE
                and not self.pending_execution_actions
            )
            and not motion_prefetch_allowed
        )
        if (
            self.plan_inflight
            or follower_blocks_planning
            or self.route.complete
            or self.precision_goal_hold
            or self.fault_reason is not None
            or now < self.next_plan_time
            # A live plan collected before ARM is unusable: opening the
            # interlock deliberately invalidates it and requires a fresh
            # active scan.  Suppress that wasted CEM load so videohub stays
            # healthy for the four operator-requested scan frames.
            or (self.args.live and not self.last_interlock)
            or (
                allow_follower_settle
                and self.settle_prefetch_suspended
            )
        ):
            return
        try:
            job = self._snapshot_job(
                minimum_camera_receive_time=(
                    minimum_camera_receive_time
                ),
                settle_prefetch=bool(allow_follower_settle),
                motion_prefetch=bool(motion_prefetch_allowed),
            )
        except Exception as exc:
            error = f"{type(exc).__name__}:{exc}"
            if error != self.last_plan_request_error:
                try:
                    self._log(
                        "plan_request_waiting_for_snapshot",
                        error=error,
                    )
                except Exception:
                    pass
            self.last_plan_request_error = error
            return
        try:
            self.job_queue.put_nowait(job)
        except queue.Full:
            return
        self.plan_inflight = True
        self.inflight_plan_sequence = job.sequence
        self.inflight_plan_settle_prefetch = bool(job.settle_prefetch)
        self.inflight_plan_motion_prefetch = bool(job.motion_prefetch)
        self.inflight_plan_snapshot_pose = job.snapshot_pose
        if (
            job.motion_prefetch
            and self.target_visual_locked
            and self.target_visual_last_plan_continuity_allowed
            and self.target_visual_unconfirmed_motion_anchor_pose is None
        ):
            self.target_visual_unconfirmed_motion_anchor_pose = (
                job.snapshot_pose
            )
        self.last_plan_request_error = None
        self.plan_sequence += 1
        if self.active_history_scan_enabled and self.scan_state == "READY":
            self._reset_active_scan("scan_consumed_by_plan")
            self.active_scan_pending = False
            self.active_scan_reason = ""
        # A visible box is a one-plan instruction.  The persistent direction
        # prior remains available, but the strict fresh annotation must not
        # leak into later rolling-history autonomous plans.
        self.operator_visual_hint = None
        self._log(
            "plan_requested",
            sequence=job.sequence,
            goal_sha256=job.goal_sha256,
            image_sha256=list(job.image_sha256),
            visual_image_sha256=list(job.visual_image_sha256),
            history_source=job.history_source,
            route_intent_direction=job.route_intent_direction,
            operator_visual_hint=job.operator_visual_hint,
            operator_direction_prior=job.operator_direction_prior,
            settle_prefetch=job.settle_prefetch,
            motion_prefetch=job.motion_prefetch,
            visual_recheck_only=job.visual_recheck_only,
        )

    def _planning_worker(self) -> None:
        recovery_visual_direction = 0
        recovery_visual_streak = 0
        recovery_visual_misses = 0
        taught_path_goal_sha256: str | None = None
        taught_path_guide: TaughtPathGuide | None = None
        cached_goal_key: tuple[str, str] | None = None
        cached_goal_image: Image.Image | None = None
        cached_goal_visual_profile: dict[str, Any] | None = None
        while not self.worker_stop.is_set():
            try:
                job = self.job_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if job is None:
                return
            try:
                goal_key = (str(job.goal_path), job.goal_sha256)
                if cached_goal_key != goal_key or cached_goal_image is None:
                    if sha256_file(job.goal_path) != job.goal_sha256:
                        raise RuntimeError(
                            "goal image changed after planning snapshot"
                        )
                    with Image.open(job.goal_path) as raw_goal:
                        cached_goal_image = raw_goal.convert("RGB").copy()
                    cached_goal_visual_profile = load_goal_visual_profile(
                        job.goal_path,
                        job.goal_sha256,
                    )
                    cached_goal_key = goal_key
                goal_image = cached_goal_image
                planner_goal_image = goal_image
                taught_path_audit: dict[str, Any] = {
                    "enabled": False,
                    "reason": "no_goal_bound_teach_path",
                }
                try:
                    goal_visual_profile = cached_goal_visual_profile
                    source_teach_run = (
                        goal_visual_profile.get("source_teach_run")
                        if goal_visual_profile is not None
                        else None
                    )
                    taught_path_anchor_frames = (
                        goal_visual_profile.get("taught_path_anchor_frames")
                        if goal_visual_profile is not None
                        else None
                    )
                    taught_path_straight_corridors = (
                        goal_visual_profile.get(
                            "taught_path_straight_corridors"
                        )
                        if goal_visual_profile is not None
                        else None
                    )
                    if source_teach_run:
                        if taught_path_goal_sha256 != job.goal_sha256:
                            taught_path_guide = TaughtPathGuide(
                                Path(str(source_teach_run)),
                                taught_path_anchor_frames,
                                taught_path_straight_corridors,
                            )
                            taught_path_goal_sha256 = job.goal_sha256
                        if taught_path_guide is not None:
                            (
                                planner_goal_image,
                                taught_path_audit,
                            ) = taught_path_guide.select(
                                job.observations[-1],
                                job.snapshot_pose,
                            )
                    else:
                        taught_path_guide = None
                        taught_path_goal_sha256 = None
                    unique_visual_frames = len(
                        set(job.visual_image_sha256)
                    )
                    if unique_visual_frames < 2:
                        visual_bearing = {
                            "schema": "v70_visual_bearing_v2",
                            "valid": False,
                            "direction": 0,
                            "reason": (
                                "insufficient_unique_endpoint_frames"
                            ),
                            "unique_frame_count": unique_visual_frames,
                            "frames": [],
                        }
                    else:
                        visual_bearing = estimate_goal_visual_bearing(
                            goal_image,
                            job.visual_observations,
                            observation_yaw_offsets_rad=tuple(
                                wrap_angle(
                                    pose.yaw - job.snapshot_pose.yaw
                                )
                                for pose in job.visual_poses
                            ),
                            goal_roi_norm=(
                                goal_visual_profile["target_roi_norm"]
                                if goal_visual_profile is not None
                                else None
                            ),
                            goal_target_center_x_norm=(
                                goal_visual_profile["target_center_x_norm"]
                                if goal_visual_profile is not None
                                else 0.5
                            ),
                            bearing_cue=(
                                goal_visual_profile["bearing_cue"]
                                if goal_visual_profile is not None
                                else "auto"
                            ),
                            exposure_recovery=bool(self.args.rolling_plan_prefetch),
                        )
                    if goal_visual_profile is not None:
                        visual_bearing = apply_goal_visual_profile(
                            visual_bearing,
                            goal_visual_profile,
                        )
                    visual_bearing = (
                        suppress_terminal_bearing_during_taught_path(
                            visual_bearing,
                            taught_path_audit,
                        )
                    )
                except Exception as exc:
                    visual_bearing = {
                        "schema": "v70_visual_bearing_v2",
                        "valid": False,
                        "direction": 0,
                        "reason": (
                            f"estimator_exception:"
                            f"{type(exc).__name__}:{exc}"
                        ),
                        "frames": [],
                    }
                if job.visual_recheck_only:
                    # A locked-target miss is decided entirely by the
                    # lightweight colour/geometry continuity gate.  Do not
                    # rerun DINO, NoMaD, LTWM or CEM while the same gate is
                    # certain to reject every action.  A later confirmed frame
                    # requests a fresh full plan before any model action runs.
                    result = {
                        "schema": "v70_visual_recheck_result_v1",
                        "visual_recheck_only": True,
                        "visual_bearing": visual_bearing,
                        "operator_visual_hint": dict(
                            job.operator_visual_hint or {}
                        ),
                    }
                    self.output_queue.put(
                        PlanningOutput(
                            job=job,
                            result=result,
                            completed_time=time.monotonic(),
                        )
                    )
                    continue
                visual_direction = int(
                    visual_bearing.get("direction", 0)
                )
                operator_hint = job.operator_visual_hint or {}
                operator_hint_direction = int(
                    operator_hint.get("direction", 0)
                ) if operator_hint.get("visible") else 0
                operator_guidance_direction = (
                    operator_hint_direction
                    or int(job.operator_direction_prior)
                )
                operator_guided_consensus = (
                    autonomous_consensus_with_operator_prior(
                        prior_direction=operator_guidance_direction,
                        visual_bearing=visual_bearing,
                        required_agreeing_frames=2,
                    )
                )
                center_x_norm = visual_bearing.get("center_x_norm")
                strong_visual_direction = bool(
                    visual_bearing.get("valid")
                    and visual_direction in (-1, 1)
                    and int(visual_bearing.get("consensus_count", 0)) >= 2
                    and int(visual_bearing.get("opposing_count", 0)) <= 1
                    and center_x_norm is not None
                    and abs(float(center_x_norm) - 0.5)
                    >= DIRECTIONAL_RECOVERY_MIN_CENTER_OFFSET
                )
                (
                    recovery_visual_direction,
                    recovery_visual_streak,
                    recovery_visual_misses,
                ) = update_directional_visual_confirmation(
                    recovery_visual_direction,
                    recovery_visual_streak,
                    recovery_visual_misses,
                    strong=strong_visual_direction,
                    direction=visual_direction,
                    max_misses=DIRECTIONAL_RECOVERY_MAX_VISUAL_MISSES,
                )

                operator_direction_override = (
                    direct_operator_direction_override(
                        operator_hint_direction
                    )
                )
                visual_confirmation_requirement = (
                    1
                    if (
                        visual_bearing.get("cue") == "warm_target_color"
                        and int(visual_bearing.get("consensus_count", 0)) >= 1
                        and int(visual_bearing.get("opposing_count", 0)) == 0
                    )
                    else DIRECTIONAL_RECOVERY_VISUAL_CONFIRMATIONS
                )
                warm_immediate_direction = bool(
                    strong_visual_direction
                    and visual_confirmation_requirement == 1
                )
                route_memory_visual_agreement = bool(
                    strong_visual_direction
                    and int(job.route_intent_direction) in (-1, 1)
                    and int(job.route_intent_direction) == visual_direction
                )
                confirmed_visual_direction = (
                    int(visual_direction)
                    if (
                        warm_immediate_direction
                        or route_memory_visual_agreement
                    )
                    else (
                        int(recovery_visual_direction)
                        if (
                        strong_visual_direction
                        and recovery_visual_streak
                        >= visual_confirmation_requirement
                        )
                        else 0
                    )
                )
                centered_visual = reliable_centered_visual(
                    visual_bearing,
                    operator_direction=(
                        operator_hint_direction
                        or int(job.operator_direction_prior)
                    ),
                )
                taught_route_direction = int(
                    taught_path_audit.get("route_intent_direction", 0)
                )
                # Generic visual cues must pass the temporal gate before they
                # steer CEM.  A strong warm-colour cue may act immediately only
                # after the tiny-speck and opposing-vote checks above; this also
                # prevents a discarded prefetch job from erasing a reliable
                # current-frame direction.
                # A centred terminal-object cue is not a valid reason to erase
                # the geometric turn between taught intermediate anchors.  In
                # the real run this zeroed a strong -2.07 right-turn signal and
                # allowed the robot to continue toward the wall.  The taught
                # direction remains bounded to the current local route goal;
                # centred visual evidence may neutralize steering only after
                # the taught guide no longer requests a turn.
                planner_direction = (
                    operator_hint_direction
                    or taught_route_direction
                    or confirmed_visual_direction
                    or (
                        0
                        if centered_visual
                        else int(job.route_intent_direction)
                    )
                )
                taught_route_authorized = bool(
                    taught_route_direction in (-1, 1)
                    and planner_direction == taught_route_direction
                )
                allow_directional_recovery = bool(
                    operator_direction_override
                    or confirmed_visual_direction
                    or operator_guided_consensus.get("authorized")
                    or taught_route_authorized
                )
                result = self.planner.plan(
                    observations=job.observations,
                    goal_image=planner_goal_image,
                    poses=job.poses,
                    route_intent_direction=planner_direction,
                    strict_route_intent=bool(
                        operator_direction_override
                        or confirmed_visual_direction
                        or taught_route_authorized
                    ),
                    allow_directional_recovery=(
                        allow_directional_recovery
                    ),
                    # The taught route is an independent, bounded recovery
                    # authorization. It may fill K16 from three distinct
                    # sealed support templates immediately, or two after one
                    # failed planning cycle. Every repeated candidate still
                    # passes the unchanged V70/LTWM safety gates.
                    allow_taught_path_support_recovery=bool(
                        taught_path_audit.get("enabled")
                    ),
                    taught_path_support_minimum_unique_templates=(
                        2 if self.consecutive_plan_holds >= 1 else 3
                    ),
                    taught_path_support_distance_multiplier=(
                        2.0 if self.consecutive_plan_holds >= 1 else 1.5
                    ),
                    fixed_pool_lateral_execution_authorized=bool(
                        allow_directional_recovery
                        or centered_visual
                        or visual_bearing.get(
                            "profile_planner_fallback_authorized", False
                        )
                    ),
                    # A moving result loses value faster than it gains from a
                    # second safe challenger. Return the first candidate that
                    # already passed every unchanged gate; stationary plans
                    # retain the configured two-candidate precision policy.
                    fixed_pool_fast_path_min_safe=(
                        1 if job.motion_prefetch else None
                    ),
                    # The first abstention has already exhausted the full
                    # search. Keep each stationary retry to one CEM round.
                    # Moving prefetch retains the full fallback search when
                    # neither its fixed pool nor exact support produces an
                    # action; board replay found valid actions in CEM2.
                    maximum_cem_iterations=(
                        1
                        if (
                            self.last_plan_summary is not None
                            and self.last_plan_summary.get("abstained")
                            and self.consecutive_plan_holds >= 1
                        )
                        else None
                    ),
                    allow_abstention_resample=bool(
                        self.args.operator_stop_only
                        and not job.motion_prefetch
                    ),
                )
                result["directional_recovery_gate"] = {
                    "strong_visual_direction": strong_visual_direction,
                    "confirmed_visual_direction": confirmed_visual_direction,
                    "visual_direction": visual_direction,
                    "visual_streak": recovery_visual_streak,
                    "visual_misses": recovery_visual_misses,
                    "max_visual_misses": (
                        DIRECTIONAL_RECOVERY_MAX_VISUAL_MISSES
                    ),
                    "required_visual_streak": (
                        visual_confirmation_requirement
                    ),
                    "route_memory_visual_agreement": (
                        route_memory_visual_agreement
                    ),
                    "center_x_norm": center_x_norm,
                    "reliable_centered_visual": centered_visual,
                    "bearing_rad": visual_bearing.get("bearing_rad"),
                    "visual_cue": visual_bearing.get("cue"),
                    "minimum_center_offset": (
                        DIRECTIONAL_RECOVERY_MIN_CENTER_OFFSET
                    ),
                    "authorized": allow_directional_recovery,
                    "operator_guided_consensus": (
                        operator_guided_consensus
                    ),
                    "operator_guidance_direction": (
                        operator_guidance_direction
                    ),
                    "operator_override_authorized": (
                        operator_direction_override
                    ),
                    "sensor_execution_gate": (
                        "safe_bridge_must_be_healthy_before_motion"
                    ),
                }
                result["visual_bearing"] = visual_bearing
                result["visual_goal_proximity"] = visual_goal_proximity(
                    visual_bearing,
                    dino_cosine_distance=float(
                        result["dino"]["current_goal_cosine_distance"]
                    ),
                )
                result["taught_path_guide"] = taught_path_audit
                result["operator_visual_hint"] = dict(operator_hint)
                result["planner_route_intent_direction"] = (
                    planner_direction
                )
                result["operator_direction_override"] = (
                    operator_direction_override
                )
                output = PlanningOutput(
                    job=job,
                    result=result,
                    completed_time=time.monotonic(),
                )
            except Exception as exc:
                output = PlanningOutput(
                    job=job,
                    result=None,
                    completed_time=time.monotonic(),
                    error=f"{type(exc).__name__}: {exc}",
                )
            self.output_queue.put(output)

    def _stationary_since_snapshot(
        self,
        job: PlanningJob,
        latest: Pose2D,
    ) -> tuple[bool, float, float]:
        distance = math.hypot(
            latest.x - job.snapshot_pose.x,
            latest.y - job.snapshot_pose.y,
        )
        yaw_delta = abs(
            wrap_angle(latest.yaw - job.snapshot_pose.yaw)
        )
        valid = (
            distance
            <= self.args.stationary_translation_tolerance
            and yaw_delta <= self.args.stationary_yaw_tolerance
        )
        return valid, distance, yaw_delta

    def _advance_route(self) -> None:
        advanced = self.route.mark_current_reached()
        if advanced:
            self.approach_mode = False
            self.precision_goal_hold = False
            self.target_visual_locked = False
            self.target_visual_trusted_scale_ratio = None
            self.target_visual_loss_holds = 0
            self.target_visual_loss_forward_distance_m = 0.0
            self.abstention_forward_distance_m = 0.0
            self.abstention_near_target_step_used = False
            self.target_visual_last_plan_continuity_allowed = True
            self.target_visual_unconfirmed_motion_anchor_pose = None
            self.target_visual_recheck_active = False
            self.target_visual_recheck_after = 0.0
            self.target_visual_recheck_count = 0
            self.target_visual_recheck_step_used = False
            self.target_visual_recheck_plan = None
            self.route_intent_direction = 0
            self.route_intent_endpoint_y_m = None
            self.route_intent_opposite_streak = 0
            self.short_buffered_prefetch_allowed = False
            self.planning_gap_bridge_active = False
            self.planning_gap_bridge_sequence = None
            self.planning_gap_bridge_started_time = None
            self.goal_supervisor = goal_supervisor_for(
                self.route.current,
                max_evidence_age_s=self.args.planner_timeout,
            )
            self.next_plan_time = time.monotonic()
            self._log(
                "route_advanced",
                new_goal=self.route.current.name,
            )
        else:
            # A terminal image-goal arrival is a hard zero latch.
            self.stop_file.parent.mkdir(parents=True, exist_ok=True)
            self.stop_file.touch(exist_ok=True)
            self._log("route_complete")
            self._publish_zero()

    def _record_plan_hold(
        self,
        reason: str,
        *,
        continue_replanning_on_budget: bool = False,
    ) -> bool:
        """Count a non-executable plan and apply the configured retry policy."""

        self._request_adaptive_scan(f"plan_hold:{reason}")
        self.consecutive_plan_holds += 1
        self._log(
            "plan_hold_budget_updated",
            reason=reason,
            consecutive_holds=self.consecutive_plan_holds,
            max_consecutive_holds=self.args.max_consecutive_plan_holds,
            cumulative_scan_translation_m=(
                self.cumulative_scan_translation_m
            ),
            max_cumulative_scan_translation_m=(
                self.args.max_cumulative_scan_translation_m
            ),
        )
        fault = bounded_plan_hold_fault_reason(
            consecutive_holds=self.consecutive_plan_holds,
            max_consecutive_holds=self.args.max_consecutive_plan_holds,
            cumulative_scan_translation_m=(
                self.cumulative_scan_translation_m
            ),
            max_cumulative_scan_translation_m=(
                self.args.max_cumulative_scan_translation_m
            ),
        )
        if fault is None:
            return False
        if (
            continue_replanning_on_budget
            and fault.startswith("plan_hold_budget_exhausted:")
        ):
            self._log(
                "plan_hold_budget_recycled_for_replanning",
                reason=reason,
                exhausted_fault=fault,
                previous_consecutive_holds=(
                    self.consecutive_plan_holds
                ),
                operator_stop_only=bool(self.args.operator_stop_only),
            )
            # A planning abstention is not a robot-health fault. Recycle only
            # this counter so the node can rescan and replan; scan drift,
            # stale sensors, odometry faults and operator STOP remain hard.
            self.consecutive_plan_holds = 0
            return False
        self._latch_fault(fault)
        return True

    def _begin_external_control_rebase(
        self,
        latest: Pose2D,
        now: float,
        *,
        reason: str,
    ) -> None:
        """Hold our output at zero until external pose changes settle."""

        previous_translation = self.cumulative_scan_translation_m
        self.follower.reset("external_control_pose_rebase")
        self._reset_active_scan("external_control_pose_rebase")
        self.planning_history_images.clear()
        self.planning_history_poses.clear()
        self.planning_history_last_sequence = None
        self.pending_execution_actions.clear()
        self.execution_prefix_step = 0
        self.execution_prefix_audit = None
        self.short_buffered_prefetch_allowed = False
        self.planning_gap_bridge_active = False
        self.planning_gap_bridge_sequence = None
        self.planning_gap_bridge_started_time = None
        self.settle_prefetch_after = None
        self.cumulative_scan_translation_m = 0.0
        self.target_visual_locked = False
        self.target_visual_trusted_scale_ratio = None
        self.target_visual_loss_holds = 0
        self.target_visual_loss_forward_distance_m = 0.0
        self.abstention_forward_distance_m = 0.0
        self.abstention_near_target_step_used = False
        self.target_visual_last_plan_continuity_allowed = True
        self.target_visual_unconfirmed_motion_anchor_pose = None
        self.target_visual_recheck_active = False
        self.target_visual_recheck_after = 0.0
        self.target_visual_recheck_count = 0
        self.target_visual_recheck_step_used = False
        self.target_visual_recheck_plan = None
        self.route_intent_direction = 0
        self.route_intent_endpoint_y_m = None
        self.route_intent_opposite_streak = 0
        self.external_control_rebase_pending = True
        self.external_control_rebase_reason = str(reason)
        self.external_control_anchor_pose = latest
        self.external_control_stable_since = float(now)
        self.next_plan_time = float(now)
        self._publish_zero()
        self._log(
            "external_control_rebase_started",
            reason=reason,
            previous_cumulative_scan_translation_m=(
                previous_translation
            ),
            anchor_pose={
                "x_m": latest.x,
                "y_m": latest.y,
                "yaw_rad": latest.yaw,
            },
            stable_duration_s=EXTERNAL_CONTROL_STABLE_DURATION_S,
        )

    def _update_external_control_rebase(
        self,
        latest: Pose2D,
        now: float,
    ) -> bool:
        """Return true once the robot is stable at its externally set pose."""

        anchor, stable_since, ready = external_control_rebase_update(
            anchor=self.external_control_anchor_pose,
            stable_since=self.external_control_stable_since,
            current=latest,
            now=now,
            translation_tolerance_m=(
                self.args.stationary_translation_tolerance
            ),
            yaw_tolerance_rad=self.args.stationary_yaw_tolerance,
            stable_duration_s=EXTERNAL_CONTROL_STABLE_DURATION_S,
        )
        moved_again = anchor is not self.external_control_anchor_pose
        self.external_control_anchor_pose = anchor
        self.external_control_stable_since = stable_since
        if moved_again:
            self._log(
                "external_control_motion_observed",
                reason=self.external_control_rebase_reason,
                pose={
                    "x_m": latest.x,
                    "y_m": latest.y,
                    "yaw_rad": latest.yaw,
                },
            )
        if not ready:
            return False
        reason = self.external_control_rebase_reason
        self.external_control_rebase_pending = False
        self.external_control_rebase_reason = None
        self.external_control_anchor_pose = None
        self.external_control_stable_since = None
        self.next_plan_time = now
        self._log(
            "external_control_rebase_complete_reanalysis_requested",
            reason=reason,
            stable_duration_s=EXTERNAL_CONTROL_STABLE_DURATION_S,
            new_origin_pose={
                "x_m": latest.x,
                "y_m": latest.y,
                "yaw_rad": latest.yaw,
            },
        )
        return True

    def _reset_plan_hold_budget(self, reason: str) -> None:
        if (
            not self.consecutive_plan_holds
            and not self.cumulative_scan_translation_m
        ):
            return
        self._log(
            "plan_hold_budget_reset",
            reason=reason,
            previous_consecutive_holds=self.consecutive_plan_holds,
            previous_cumulative_scan_translation_m=(
                self.cumulative_scan_translation_m
            ),
        )
        self.consecutive_plan_holds = 0
        self.cumulative_scan_translation_m = 0.0





    def _handle_target_visual_recheck(
        self, *, output: PlanningOutput, result: dict[str, Any], latest: Pose2D,
        stationary: bool, drift_m: float, drift_yaw: float, now: float,
    ) -> bool:
        """A visual-only result can request fresh inference, never restore a plan."""
        if not result.get("visual_recheck_only"):
            return False
        self.target_visual_recheck_active = False
        self.target_visual_recheck_after = 0.0
        self.target_visual_recheck_count = 0
        self.target_visual_recheck_step_used = False
        self.target_visual_recheck_plan = None
        self.pending_execution_actions.clear()
        self._publish_zero()
        self.next_plan_time = now
        self._log("visual_recheck_requires_fresh_full_plan",
                  sequence=output.job.sequence, stationary=stationary,
                  drift_m=drift_m, drift_yaw_rad=drift_yaw)
        return True

    def _handle_plan_output(
        self,
        output: PlanningOutput,
        now: float,
    ) -> None:
        self.plan_inflight = False
        self.inflight_plan_sequence = None
        self.inflight_plan_settle_prefetch = False
        self.inflight_plan_motion_prefetch = False
        self.inflight_plan_snapshot_pose = None
        self.planning_gap_bridge_active = False
        self.planning_gap_bridge_started_time = None
        if output.job.sequence in self.invalidated_plan_sequences:
            self.invalidated_plan_sequences.discard(output.job.sequence)
            self.next_plan_time = now
            self._log(
                "settle_prefetched_plan_discarded",
                sequence=output.job.sequence,
                reason="physical_completion_required_fresh_replan",
            )
            return
        if output.error is not None or output.result is None:
            self._latch_fault(
                f"planner_failed:{output.error}",
                EXIT_PLANNER,
            )
            return
        if output.job.route_index != self.route.index:
            self.next_plan_time = now
            self._log("stale_route_plan_discarded")
            return
        latest = self.odom.latest()
        if latest is None:
            self._latch_fault(
                "plan_completed_without_fresh_odom",
                EXIT_SENSOR,
            )
            return
        compute_age = output.completed_time - output.job.snapshot_time
        queue_wait = max(0.0, now - output.completed_time)
        age = (
            now - output.job.snapshot_time
            if output.job.settle_prefetch or output.job.motion_prefetch
            else compute_age
        )
        prefetched = bool(
            output.job.settle_prefetch or output.job.motion_prefetch
        )
        result = output.result
        stationary, drift_m, drift_yaw = (
            self._stationary_since_snapshot(
                output.job,
                latest,
            )
        )
        residual_drift_rebase = (
            completed_plan_residual_drift_rebase_allowed(
                prefetched=prefetched,
                operator_stop_only=bool(self.args.operator_stop_only),
                follower_active=bool(self.follower.active),
                pending_execution_actions=bool(
                    self.pending_execution_actions
                ),
                external_control_rebase_pending=bool(
                    self.external_control_rebase_pending
                ),
                drift_m=drift_m,
                drift_yaw=drift_yaw,
            )
        )
        if prefetched and age > ROLLING_PREFETCH_MAX_AGE_S:
            self.settle_prefetch_after = None
            self.rolling_prefetch_after = None
            self.target_visual_recheck_active = False
            self.target_visual_recheck_after = 0.0
            self.target_visual_recheck_count = 0
            self.target_visual_recheck_step_used = False
            self.target_visual_recheck_plan = None
            self.next_plan_time = now
            self._log(
                "prefetched_plan_discarded_requires_fresh_observation",
                sequence=output.job.sequence,
                settle_prefetch=bool(output.job.settle_prefetch),
                motion_prefetch=bool(output.job.motion_prefetch),
                age_s=age,
                reason="rolling_prefetch_age_limit",
            )
            return
        if age > self.args.planner_timeout:
            self._latch_fault(
                f"planner_timeout:age_s={age:.6f}",
                EXIT_PLANNER,
            )
            return
        if self._handle_target_visual_recheck(
            output=output,
            result=result,
            latest=latest,
            stationary=stationary,
            drift_m=drift_m,
            drift_yaw=drift_yaw,
            now=now,
        ):
            return
        if not stationary:
            if prefetched and bool(result.get("abstained")):
                # An all-zero abstention is a planning result, not a geometric
                # trajectory. Preserve it for the normal hold accounting
                # instead of trying to rebase four no-op placeholders and
                # restarting the same retry state from scratch.
                self._log(
                    "prefetched_abstention_preserved",
                    sequence=output.job.sequence,
                    settle_prefetch=bool(output.job.settle_prefetch),
                    motion_prefetch=bool(output.job.motion_prefetch),
                    age_s=age,
                    drift_m=drift_m,
                    drift_yaw_rad=drift_yaw,
                )
            elif prefetched and bool(
                result.get("visual_goal_proximity", {}).get(
                    "precision_hold"
                )
            ):
                # A hold carries no trajectory across coordinate frames. If
                # it was already true at the moving snapshot, later forward
                # progress can only make stopping more conservative. Preserve
                # the result so the hold below clears an H2 tail immediately.
                self._log(
                    "prefetched_precision_hold_preserved",
                    sequence=output.job.sequence,
                    settle_prefetch=bool(output.job.settle_prefetch),
                    motion_prefetch=bool(output.job.motion_prefetch),
                    age_s=age,
                    drift_m=drift_m,
                    drift_yaw_rad=drift_yaw,
                )
            elif prefetched or residual_drift_rebase:
                rebased_actions, rebase_audit = rebase_local_action_sequence(
                    np.asarray(result["selected_actions_h4"]),
                    snapshot_pose=output.job.snapshot_pose,
                    current_pose=latest,
                    max_translation_drift_m=(
                        ROLLING_PREFETCH_MAX_TRANSLATION_M
                    ),
                    max_yaw_drift_rad=ROLLING_PREFETCH_MAX_YAW_RAD,
                    drive_tolerance_m=self.follower.config.drive_tolerance_m,
                    turn_tolerance_rad=self.follower.config.turn_tolerance_rad,
                    max_action_distance_m=(
                        self.follower.config.max_action_distance_m
                    ),
                    max_abs_action_yaw_rad=(
                        self.follower.config.max_abs_action_yaw_rad
                    ),
                    heading_consistency_rad=(
                        self.follower.config.heading_consistency_rad
                    ),
                    coalesce_short_remainder=bool(
                        self.args.rolling_plan_prefetch
                        and output.job.motion_prefetch
                        and not result.get("visual_goal_proximity", {}).get("precision_hold")
                        and not result.get("taught_path_guide", {}).get("enabled")
                    ),
                    promoted_first_max_abs_yaw_rad=(
                        self.follower.config.continuous_forward_arc_max_yaw_rad
                    ),
                    align_rebased_forward_endpoint=True,
                    # A live-log replay shows three otherwise valid H1
                    # handoffs just above the legacy 0.10 rad residual. Keep
                    # the planned world endpoint, align yaw to its forward
                    # bearing, discard the stale tail, and re-observe after
                    # this one action. Larger residuals still require a fresh
                    # plan.
                    aligned_endpoint_max_heading_error_rad=0.15,
                )
                if (
                    age <= ROLLING_PREFETCH_MAX_AGE_S
                    and rebased_actions is not None
                ):
                    result["selected_actions_h4"] = rebased_actions
                    result["selected_first_action"] = rebased_actions[0].copy()
                    if residual_drift_rebase:
                        self._log(
                            "completed_plan_rebased_for_residual_drift",
                            sequence=output.job.sequence,
                            age_s=age,
                            queue_wait_s=queue_wait,
                            drift_m=drift_m,
                            drift_yaw_rad=drift_yaw,
                            rebase=rebase_audit,
                        )
                    else:
                        self._log(
                            "prefetched_plan_rebased_for_handoff",
                            sequence=output.job.sequence,
                            settle_prefetch=bool(output.job.settle_prefetch),
                            motion_prefetch=bool(output.job.motion_prefetch),
                            age_s=age,
                            queue_wait_s=queue_wait,
                            rebase=rebase_audit,
                        )
                else:
                    if residual_drift_rebase:
                        self._log(
                            "plan_discarded_robot_moved",
                            drift_m=drift_m,
                            drift_yaw_rad=drift_yaw,
                            settle_prefetch=False,
                            residual_drift_rebase_attempted=True,
                            rebase=rebase_audit,
                        )
                        self._begin_external_control_rebase(
                            latest,
                            now,
                            reason="robot_moved_during_planning",
                        )
                    else:
                        self.settle_prefetch_after = None
                        self.rolling_prefetch_after = None
                        self.next_plan_time = now
                        self._log(
                            "prefetched_plan_discarded_requires_fresh_observation",
                            sequence=output.job.sequence,
                            settle_prefetch=bool(output.job.settle_prefetch),
                            motion_prefetch=bool(output.job.motion_prefetch),
                            age_s=age,
                            rebase=rebase_audit,
                        )
                    return
            else:
                self._log(
                    "plan_discarded_robot_moved",
                    drift_m=drift_m,
                    drift_yaw_rad=drift_yaw,
                    settle_prefetch=False,
                )
                self._begin_external_control_rebase(
                    latest,
                    now,
                    reason="robot_moved_during_planning",
                )
                return

        taught_path_result = result.get("taught_path_guide", {})
        taught_start_reliably_matched = bool(
            isinstance(taught_path_result, dict)
            and int(taught_path_result.get("matched_anchor", -1)) == 0
            and taught_path_result.get("progress_match_reason")
            == "strong_match"
            and float(taught_path_result.get("match_score", 0.0))
            >= TAUGHT_PATH_MIN_MATCH_SCORE
        )
        taught_entry_heading_guard = bool(
            isinstance(taught_path_result, dict)
            and taught_path_result.get("enabled")
            and int(taught_path_result.get("progress_anchor", 0)) == 0
            # Protect only uncertain entries. A strong match to anchor zero
            # is positive evidence that the demonstrated initial heading is
            # already reproduced, so clipping its first turn defeats the
            # taught route exactly where it is most useful.
            and not taught_start_reliably_matched
            and int(output.job.sequence) < 2
        )
        if taught_entry_heading_guard:
            guarded_actions = np.asarray(
                result["selected_actions_h4"], dtype=np.float32
            ).copy()
            original_first = guarded_actions[0].copy()
            guarded_actions[0, 1] = max(
                -0.005, min(0.005, float(guarded_actions[0, 1]))
            )
            guarded_actions[0, 2] = max(
                -0.015, min(0.015, float(guarded_actions[0, 2]))
            )
            result["selected_actions_h4"] = guarded_actions
            result["selected_first_action"] = guarded_actions[0].copy()
            taught_path_result["entry_heading_guard"] = {
                "active": True,
                "reason": "first_teach_anchor_not_reliably_reproduced",
                "original_first_action": original_first.tolist(),
                "guarded_first_action": guarded_actions[0].tolist(),
                "max_abs_dy_m": 0.005,
                "max_abs_dyaw_rad": 0.015,
                "execution_prefix_limit": 1,
            }
        taught_direction = int(
            taught_path_result.get("route_intent_direction", 0) or 0
        )
        corridor_active = bool(
            isinstance(taught_path_result, dict)
            and taught_path_result.get("odometry_progress", {}).get(
                "route_intent_corridor_frames"
            )
        )
        if (
            taught_direction in (-1, 1)
            and not corridor_active
            and not bool(result.get("abstained"))
        ):
            guarded_actions = np.asarray(
                result["selected_actions_h4"], dtype=np.float32
            ).copy()
            original_first = guarded_actions[0].copy()
            opposing_lateral = bool(
                float(guarded_actions[0, 1]) * taught_direction < 0.0
            )
            opposing_yaw = bool(
                float(guarded_actions[0, 2]) * taught_direction < 0.0
            )
            if opposing_lateral:
                guarded_actions[0, 1] = 0.0
            if opposing_yaw:
                guarded_actions[0, 2] = 0.0
            if opposing_lateral or opposing_yaw:
                result["selected_actions_h4"] = guarded_actions
                result["selected_first_action"] = guarded_actions[0].copy()
                taught_path_result["direction_consistency_clamp"] = {
                    "active": True,
                    "route_intent_direction": taught_direction,
                    "original_first_action": original_first.tolist(),
                    "guarded_first_action": guarded_actions[0].tolist(),
                    "policy": "opposing_components_zeroed_without_vx_reduction",
                }
        # In a narrow taught corridor, route side is not a turn command:
        # it is a centerline error.  Preserve the planner's forward speed,
        # but apply a small bounded correction so a few centimetres of drift
        # do not send the robot into the white wall.  This is deliberately
        # applied after turn-direction clamping and is still safety-filtered.
        if (
            corridor_active
            and not bool(result.get("abstained"))
            and isinstance(taught_path_result.get("odometry_progress"), dict)
            and taught_path_result["odometry_progress"].get("trusted")
        ):
            progress = taught_path_result["odometry_progress"]
            signed_xtrack = float(
                progress.get("signed_cross_track_m", 0.0) or 0.0
            )
            signed_yaw = float(
                progress.get("signed_yaw_error_rad", 0.0) or 0.0
            )
            if abs(signed_xtrack) >= 0.025 or abs(signed_yaw) >= 0.06:
                guarded_actions = np.asarray(
                    result["selected_actions_h4"], dtype=np.float32
                ).copy()
                original_first = guarded_actions[0].copy()
                correction_y = float(np.clip(-0.35 * signed_xtrack, -0.05, 0.05))
                correction_yaw = float(np.clip(-0.35 * signed_yaw, -0.10, 0.10))
                guarded_actions[0, 1] = float(
                    np.clip(guarded_actions[0, 1] + correction_y, -0.06, 0.06)
                )
                guarded_actions[0, 2] = float(
                    np.clip(guarded_actions[0, 2] + correction_yaw, -0.15, 0.15)
                )
                result["selected_actions_h4"] = guarded_actions
                result["selected_first_action"] = guarded_actions[0].copy()
                taught_path_result["corridor_centerline_correction"] = {
                    "active": True,
                    "signed_cross_track_m": signed_xtrack,
                    "signed_yaw_error_rad": signed_yaw,
                    "correction_y_m": correction_y,
                    "correction_yaw_rad": correction_yaw,
                    "original_first_action": original_first.tolist(),
                    "guarded_first_action": guarded_actions[0].tolist(),
                    "deadband_m": 0.025,
                    "deadband_yaw_rad": 0.06,
                }
        summary = self.planner.summary(result)
        visual_bearing = result.get("visual_bearing", {})
        operator_visual_hint = result.get("operator_visual_hint") or {}
        operator_hint_visible = bool(
            operator_visual_hint.get("visible")
        )
        operator_hint_direction = (
            int(operator_visual_hint.get("direction", 0))
            if operator_hint_visible
            else 0
        )
        prior_consensus = autonomous_consensus_with_operator_prior(
            prior_direction=self.operator_direction_prior,
            visual_bearing=(
                visual_bearing
                if isinstance(visual_bearing, dict)
                else {}
            ),
            required_agreeing_frames=2,
        )
        prior_consensus_authorized = (
            confirmed_prior_after_annotation_phase(
                operator_hint_visible=operator_hint_visible,
                requests_created=(
                    self.scan_annotation_requests_created
                ),
                max_rounds=self.args.scan_annotation_max_rounds,
                prior_consensus=prior_consensus,
            )
        )
        summary["visual_bearing"] = visual_bearing
        visual_proximity = dict(
            result.get("visual_goal_proximity", {})
        )
        summary["visual_goal_proximity"] = visual_proximity
        summary["operator_visual_hint"] = dict(operator_visual_hint)
        summary["operator_direction_prior"] = (
            self.operator_direction_prior
        )
        summary["autonomous_prior_consensus"] = prior_consensus
        summary["directional_recovery_gate"] = dict(
            result.get("directional_recovery_gate", {})
        )
        summary["taught_path_guide"] = dict(
            result.get("taught_path_guide", {})
        )
        taught_path_audit = summary["taught_path_guide"]
        if (
            taught_path_audit.get("junction_landmark_relocalized")
            and not self.outdoor_junction_forward_turn_committed
        ):
            # This real-robot landmark is the recurring T junction where the
            # learned route progress used to remain stuck around frame 20.
            # Execute the demonstrated topology explicitly: enter the junction
            # straight, then begin the right arc.  Independent follower and
            # Safe Bridge contracts still validate and gate every command.
            junction_actions = np.asarray(
                [
                    [0.35, 0.0, 0.0],
                    [0.28, -0.018, -0.12],
                    [0.25, -0.018, -0.18],
                    [0.24, -0.018, -0.20],
                ],
                dtype=np.float32,
            )
            result["selected_actions_h4"] = junction_actions
            result["selected_first_action"] = junction_actions[0].copy()
            result["abstained"] = False
            summary["selected_actions_h4"] = junction_actions.tolist()
            summary["selected_first_action"] = junction_actions[0].tolist()
            summary["abstained"] = False
            summary["selection_reason"] = (
                "outdoor_junction_visual_landmark_forward_then_right"
            )
            summary["outdoor_junction_action_override"] = {
                "active": True,
                "landmark_frame": OUTDOOR_JUNCTION_LANDMARK_FRAME,
                "execution": "forward_then_right",
                "actions_h4": junction_actions.tolist(),
            }
            self.outdoor_junction_forward_turn_committed = True
            self._log(
                "outdoor_junction_forward_then_right_committed",
                landmark_frame=OUTDOOR_JUNCTION_LANDMARK_FRAME,
                actions_h4=junction_actions.tolist(),
            )
        taught_local_goal_active = bool(
            taught_path_audit.get("enabled")
            and int(taught_path_audit.get("planning_goal_anchor", 0))
            < int(taught_path_audit.get("anchor_count", 1)) - 1
        )
        summary["taught_local_goal_active"] = taught_local_goal_active
        summary["planner_route_intent_direction"] = int(
            result.get(
                "planner_route_intent_direction",
                output.job.route_intent_direction,
            )
        )
        summary.update(
            {
                "sequence": output.job.sequence,
                "goal_sha256": output.job.goal_sha256,
                "plan_age_s": age,
                "planner_compute_age_s": compute_age,
                "prefetch_queue_wait_s": queue_wait,
                "settle_prefetch": bool(output.job.settle_prefetch),
                "stationary_drift_m": drift_m,
                "stationary_drift_yaw_rad": drift_yaw,
            }
        )
        visual_frame_conflict, visual_frame_directions = (
            unresolved_frame_direction_conflict(visual_bearing)
            if isinstance(visual_bearing, dict)
            else (False, set())
        )
        summary["visual_frame_direction_disagreement"] = (
            len(visual_frame_directions) > 1
        )
        summary["visual_frame_conflict"] = visual_frame_conflict
        summary["aggressive_navigation"] = bool(
            self.args.aggressive_navigation
        )
        taught_odometry = taught_path_audit.get("odometry_progress", {})
        # The route hold is a lateral/yaw guard.  The unsigned distance to the
        # nearest taught pose also contains along-track phase error and caused
        # a false stop at frame 65 (0.247 m pose distance but only 0.187 m
        # lateral error).  Gate on the signed lateral/yaw components instead.
        taught_signed_cross_track_m = float(
            taught_odometry.get("signed_cross_track_m") or 0.0
        )
        taught_signed_yaw_error_rad = float(
            taught_odometry.get("signed_yaw_error_rad") or 0.0
        )
        taught_cross_track_m = abs(taught_signed_cross_track_m)
        taught_yaw_error_rad = abs(taught_signed_yaw_error_rad)
        active_straight_corridor = bool(
            taught_path_audit.get("route_intent_corridor_frames")
        )
        maximum_cross_track_m = (
            0.22 if active_straight_corridor else 0.30
        )
        maximum_yaw_error_rad = (
            0.40 if active_straight_corridor else 0.65
        )
        taught_route_deviation_hold = bool(
            self.args.live
            and taught_path_audit.get("enabled")
            and taught_odometry.get("available")
            and (
                taught_cross_track_m > maximum_cross_track_m
                or taught_yaw_error_rad > maximum_yaw_error_rad
            )
        )
        selected_deviation_action = np.asarray(
            result.get("selected_first_action", [0.0, 0.0, 0.0]),
            dtype=np.float32,
        )
        taught_recovery_direction = int(
            taught_path_audit.get("route_intent_direction", 0) or 0
        )
        deviation_recovery_action_allowed = bool(
            taught_route_deviation_hold
            and not bool(result.get("abstained"))
            and selected_deviation_action.shape == (3,)
            and np.all(np.isfinite(selected_deviation_action))
            and float(selected_deviation_action[0]) > 0.05
            and taught_recovery_direction in (-1, 1)
            and (
                taught_recovery_direction
                * float(selected_deviation_action[2])
                >= 0.015
                or taught_recovery_direction
                * float(selected_deviation_action[1])
                >= 0.006
            )
        )
        summary["taught_route_deviation_gate"] = {
            "active": taught_route_deviation_hold,
            "cross_track_m": taught_cross_track_m,
            "signed_cross_track_m": taught_signed_cross_track_m,
            "maximum_cross_track_m": maximum_cross_track_m,
            "yaw_error_rad": taught_yaw_error_rad,
            "signed_yaw_error_rad": taught_signed_yaw_error_rad,
            "maximum_yaw_error_rad": maximum_yaw_error_rad,
            "straight_corridor": active_straight_corridor,
            "recovery_direction": taught_recovery_direction,
            "aligned_fresh_action_allowed": (
                deviation_recovery_action_allowed
            ),
        }
        if taught_route_deviation_hold and not deviation_recovery_action_allowed:
            self.pending_execution_actions.clear()
            self.execution_prefix_step = 0
            self.execution_prefix_audit = {
                "active": True,
                "reason": "taught_route_deviation_hold",
                **summary["taught_route_deviation_gate"],
            }
            self.last_plan_summary = summary
            self._publish_zero()
            self.next_plan_time = now + self.args.retry_delay
            self._log(
                "plan_held_taught_route_deviation",
                summary=summary,
                policy="operator_reposition_or_restart_required",
            )
            # A taught-route mismatch is recoverable in operator-stop-only
            # outdoor runs: continue stationary replanning instead of converting a
            # transient visual/odom mismatch into a terminal stop.
            self._record_plan_hold(
                "taught_route_deviation",
                continue_replanning_on_budget=bool(
                    self.args.operator_stop_only
                    or taught_path_audit.get("enabled")
                ),
            )
            return
        if deviation_recovery_action_allowed:
            self._log(
                "taught_route_deviation_aligned_action_released",
                recovery_direction=taught_recovery_direction,
                selected_first_action=selected_deviation_action.tolist(),
                deviation=summary["taught_route_deviation_gate"],
            )
        if (
            self.args.live
            and visual_frame_conflict
            and not operator_hint_visible
            and not prior_consensus_authorized
            and not self.args.aggressive_navigation
            and not bool(visual_bearing.get(
                "profile_planner_fallback_authorized"
            ))
            # While following a verified taught-path intermediate image, the
            # terminal object's bearing is intentionally suppressed. Requiring
            # terminal four-frame consensus here deadlocks every legitimate
            # corner proposal. The local taught goal and planner safety remain
            # authoritative until the terminal anchor becomes active.
            and not taught_local_goal_active
        ):
            summary["visual_bearing_hold"] = (
                "conflicting_four_frame_direction"
            )
            self.pending_execution_actions = []
            self.execution_prefix_step = 0
            self.execution_prefix_audit = {
                "active": True,
                "reason": "visual_frame_direction_conflict",
                "directions": sorted(visual_frame_directions),
            }
            self.last_plan_summary = summary
            self._publish_zero()
            self.next_plan_time = now + self.args.retry_delay
            self._log(
                "plan_held_visual_frame_direction_conflict",
                visual_bearing=visual_bearing,
                directions=sorted(visual_frame_directions),
            )
            self._begin_external_control_rebase(
                latest,
                now,
                reason="visual_frames_changed_during_external_control",
            )
            self._record_plan_hold("visual_frame_direction_conflict")
            return
        route_intent_endpoint_y_m = None
        if summary.get("selected_waypoints_h4_m"):
            route_intent_endpoint_y_m = float(
                summary["selected_waypoints_h4_m"][-1][1]
            )
        first_action = summary.get("selected_first_action") or []
        first_dy = (
            float(first_action[1])
            if len(first_action) == 3
            else 0.0
        )
        first_dyaw = (
            float(first_action[2])
            if len(first_action) == 3
            else 0.0
        )
        strong_lateral_endpoint = bool(
            route_intent_endpoint_y_m is not None
            and abs(route_intent_endpoint_y_m)
            >= VISUAL_LATERAL_HOLD_ENDPOINT_M
        )
        recovery_gate = summary.get("directional_recovery_gate") or {}
        centered_visual_authorized = bool(
            recovery_gate.get("reliable_centered_visual")
        )
        if self.args.operator_stop_only and centered_visual_authorized:
            previous_direction = self.route_intent_direction
            self.route_intent_direction = 0
            self.route_intent_endpoint_y_m = None
            self.route_intent_opposite_streak = 0
            if previous_direction:
                self._log(
                    "route_intent_released_by_centered_visual",
                    previous_direction=previous_direction,
                    center_x_norm=visual_bearing.get("center_x_norm"),
                    valid_frame_count=visual_bearing.get(
                        "valid_frame_count"
                    ),
                    consensus_count=visual_bearing.get("consensus_count"),
                )
        automatic_visual_direction = int(
            recovery_gate.get("confirmed_visual_direction", 0)
        )
        visual_direction = (
            operator_hint_direction
            if operator_hint_visible
            else (
                self.operator_direction_prior
                if prior_consensus_authorized
                else automatic_visual_direction
            )
        )
        visual_lateral_consensus = bool(
            visual_direction in (-1, 1)
            and (
                operator_hint_visible
                or prior_consensus_authorized
                or (
                    isinstance(visual_bearing, dict)
                    and visual_bearing.get("valid")
                )
            )
        )
        taught_route_lateral_authorized = bool(
            isinstance(taught_path_result, dict)
            and taught_path_result.get("enabled")
            and int(taught_path_result.get("route_intent_direction", 0) or 0)
            in (-1, 1)
            and isinstance(taught_path_result.get("odometry_progress"), dict)
            and taught_path_result["odometry_progress"].get("trusted")
        )
        visual_endpoint_conflict = bool(
            visual_lateral_consensus
            and (
                operator_hint_visible
                or prior_consensus_authorized
                or recovery_gate.get("confirmed_visual_direction")
            )
            and route_intent_endpoint_y_m is not None
            and visual_direction * route_intent_endpoint_y_m
            < -ROUTE_INTENT_OPPOSITE_TOLERANCE_M
        )
        summary["visual_endpoint_conflict"] = visual_endpoint_conflict
        visual_first_step_conflict = bool(
            visual_lateral_consensus
            and (
                visual_direction * first_dy
                < -ROUTE_INTENT_FIRST_STEP_CONFLICT_THRESHOLD_M
                or visual_direction * first_dyaw
                < -ROUTE_INTENT_FIRST_STEP_HEADING_THRESHOLD_RAD
            )
        )
        summary["visual_first_step_conflict"] = visual_first_step_conflict
        if (
            self.args.live
            and visual_first_step_conflict
            and not bool(recovery_gate.get("confirmed_visual_direction"))
        ):
            self.pending_execution_actions = []
            self.execution_prefix_step = 0
            self.execution_prefix_audit = {
                "active": True,
                "reason": "visual_bearing_first_step_conflict",
                "visual_direction": visual_direction,
                "first_dy_m": first_dy,
                "first_dyaw_rad": first_dyaw,
            }
            self.last_plan_summary = summary
            self._publish_zero()
            self.next_plan_time = now + self.args.retry_delay
            self._log(
                "action_rejected_visual_first_step_conflict",
                visual_bearing=visual_bearing,
                first_dy_m=first_dy,
                first_dyaw_rad=first_dyaw,
            )
            self._record_plan_hold("visual_first_step_conflict")
            return
        if self.args.live and visual_endpoint_conflict:
            self.pending_execution_actions = []
            self.execution_prefix_step = 0
            self.execution_prefix_audit = {
                "active": True,
                "reason": "strong_visual_endpoint_conflict",
                "visual_direction": visual_direction,
                "endpoint_y_m": route_intent_endpoint_y_m,
                "opposite_tolerance_m": (
                    ROUTE_INTENT_OPPOSITE_TOLERANCE_M
                ),
            }
            self.last_plan_summary = summary
            self._publish_zero()
            self.next_plan_time = now + self.args.retry_delay
            self._log(
                "action_rejected_strong_visual_endpoint_conflict",
                visual_bearing=visual_bearing,
                endpoint_y_m=route_intent_endpoint_y_m,
                opposite_tolerance_m=(
                    ROUTE_INTENT_OPPOSITE_TOLERANCE_M
                ),
            )
            self._record_plan_hold("strong_visual_endpoint_conflict")
            return
        lateral_first_step = bool(
            abs(first_dy) >= VISUAL_LATERAL_HOLD_FIRST_DY_M
            or abs(first_dyaw)
            >= VISUAL_LATERAL_HOLD_FIRST_DYAW_RAD
        )
        # If a taught turn is clearly lateral but the learned first sample is
        # almost straight, give it a small signed steering seed.  This fixes
        # the real-robot failure mode where the first narrow-door turn is
        # skipped even though the offline route intent is correct.  The seed
        # is limited to the first action and is not used in straight corridors.
        taught_yaw_change_rad = float(
            taught_path_audit.get("route_intent_yaw_change_rad") or 0.0
        )
        taught_left_m = float(
            taught_path_audit.get("route_intent_left_m") or 0.0
        )
        geometric_turn_requested = bool(
            abs(taught_yaw_change_rad) >= 0.12
            or abs(taught_left_m) >= 0.12
        )
        if (
            not corridor_active
            and not bool(summary.get("abstained"))
            and taught_direction in (-1, 1)
            and (
                geometric_turn_requested
                or (
                    route_intent_endpoint_y_m is not None
                    and abs(route_intent_endpoint_y_m) >= 0.12
                )
            )
            and abs(first_dy) < 0.012
            and abs(first_dyaw) < 0.045
        ):
            guarded_actions = np.asarray(
                result["selected_actions_h4"], dtype=np.float32
            ).copy()
            original_first = guarded_actions[0].copy()
            # Must clear the live DRIVE steering deadband (0.07 rad).  The
            # previous 0.065 seed looked like a turn in planner telemetry but
            # produced continuous_arc_steering_direction=0 and physical wz=0.
            seeded_yaw = float(
                taught_direction
                * max(abs(float(guarded_actions[0, 2])), 0.12)
            )
            # Keep the waypoint bearing consistent with the requested yaw so
            # the final follower contract accepts short turn-entry actions.
            seeded_dy = float(
                np.clip(
                    float(guarded_actions[0, 0]) * math.tan(seeded_yaw),
                    -0.018,
                    0.018,
                )
            )
            guarded_actions[0, 1] = seeded_dy
            guarded_actions[0, 2] = seeded_yaw
            result["selected_actions_h4"] = guarded_actions
            result["selected_first_action"] = guarded_actions[0].copy()
            summary["selected_first_action"] = guarded_actions[0].tolist()
            first_dy = float(guarded_actions[0, 1])
            first_dyaw = float(guarded_actions[0, 2])
            taught_path_audit["turn_steering_seed"] = {
                "active": True,
                "route_intent_direction": taught_direction,
                "endpoint_y_m": route_intent_endpoint_y_m,
                "taught_yaw_change_rad": taught_yaw_change_rad,
                "taught_left_m": taught_left_m,
                "geometric_turn_requested": geometric_turn_requested,
                "original_first_action": original_first.tolist(),
                "seeded_first_action": guarded_actions[0].tolist(),
                "seeded_dy_m": seeded_dy,
                "minimum_abs_dyaw_rad": 0.12,
                "execution_steering_deadband_rad": 0.07,
            }
        if (
            self.args.live
            and strong_lateral_endpoint
            and lateral_first_step
            and not visual_lateral_consensus
            and not centered_visual_authorized
            and not taught_route_lateral_authorized
            and not bool(visual_bearing.get(
                "profile_planner_fallback_authorized"
            ))
        ):
            summary["visual_bearing_hold"] = (
                "lateral_action_without_four_frame_consensus"
            )
            self.pending_execution_actions = []
            self.execution_prefix_step = 0
            self.execution_prefix_audit = {
                "active": True,
                "reason": "lateral_action_without_visual_consensus",
                "endpoint_y_m": route_intent_endpoint_y_m,
                "first_dy_m": first_dy,
                "first_dyaw_rad": first_dyaw,
                "visual_bearing": visual_bearing,
            }
            self.last_plan_summary = summary
            self._publish_zero()
            self.next_plan_time = now + self.args.retry_delay
            self._log(
                "plan_held_lateral_action_without_visual_consensus",
                visual_bearing=visual_bearing,
                endpoint_y_m=route_intent_endpoint_y_m,
                first_dy_m=first_dy,
                first_dyaw_rad=first_dyaw,
            )
            self._record_plan_hold(
                "lateral_action_without_visual_consensus"
            )
            return
        candidate_direction = 0
        if (
            self.args.operator_stop_only
            and visual_direction in (-1, 1)
        ):
            previous_direction = self.route_intent_direction
            self.route_intent_direction = visual_direction
            self.route_intent_endpoint_y_m = (
                route_intent_endpoint_y_m
            )
            self.route_intent_opposite_streak = 0
            if previous_direction != visual_direction:
                self._log(
                    "visual_bearing_direction_applied",
                    previous_direction=previous_direction,
                    direction=(
                        "left" if visual_direction > 0
                        else "right"
                    ),
                    center_x_norm=visual_bearing.get(
                        "center_x_norm"
                    ),
                    valid_frame_count=visual_bearing.get(
                        "valid_frame_count"
                    ),
                    reason=visual_bearing.get("reason"),
                    source=(
                        "operator_scan_annotation"
                        if operator_hint_visible
                        else "automatic_visual_bearing"
                    ),
                )
        if (
            not bool(summary.get("abstained"))
            and route_intent_endpoint_y_m is not None
            and abs(route_intent_endpoint_y_m)
            >= ROUTE_INTENT_LATCH_THRESHOLD_M
        ):
            candidate_direction = (
                1 if route_intent_endpoint_y_m > 0.0 else -1
            )
        if (
            self.args.operator_stop_only
            and visual_direction in (-1, 1)
            and candidate_direction
        ):
            if self.route_intent_direction == 0:
                self.route_intent_direction = candidate_direction
                self.route_intent_endpoint_y_m = (
                    route_intent_endpoint_y_m
                )
                self.route_intent_opposite_streak = 0
                self._log(
                    "route_intent_latched",
                    direction=(
                        "left" if candidate_direction > 0
                        else "right"
                    ),
                    endpoint_y_m=route_intent_endpoint_y_m,
                    policy="soft_rank_only",
                )
            elif candidate_direction != self.route_intent_direction:
                self.route_intent_opposite_streak += 1
                self._log(
                    "route_intent_reversal_observed",
                    direction=(
                        "left" if candidate_direction > 0
                        else "right"
                    ),
                    endpoint_y_m=route_intent_endpoint_y_m,
                    consecutive=self.route_intent_opposite_streak,
                    required=ROUTE_INTENT_REVERSAL_CONFIRMATIONS,
                )
                if (
                    not self.route.is_final
                    and self.route_intent_opposite_streak
                    >= ROUTE_INTENT_REVERSAL_CONFIRMATIONS
                ):
                    previous_direction = self.route_intent_direction
                    self.route_intent_direction = candidate_direction
                    self.route_intent_endpoint_y_m = (
                        route_intent_endpoint_y_m
                    )
                    self.route_intent_opposite_streak = 0
                    self._log(
                        "route_intent_switched_after_consensus",
                        previous_direction=previous_direction,
                        direction=candidate_direction,
                        endpoint_y_m=route_intent_endpoint_y_m,
                        confirmations=ROUTE_INTENT_REVERSAL_CONFIRMATIONS,
                        policy="multi_goal_only",
                    )
                elif self.route.is_final:
                    self._log(
                        "route_intent_reversal_held_for_final_goal",
                        direction=(
                            "left" if candidate_direction > 0
                            else "right"
                        ),
                        endpoint_y_m=route_intent_endpoint_y_m,
                        consecutive=self.route_intent_opposite_streak,
                        policy="latched_final_goal_side",
                    )
            else:
                self.route_intent_opposite_streak = 0
        elif not bool(summary.get("abstained")):
            self.route_intent_opposite_streak = 0
        summary["route_intent_direction"] = (
            int(self.route_intent_direction)
        )
        summary["route_intent_endpoint_y_m"] = (
            route_intent_endpoint_y_m
        )
        summary["route_intent_opposite_streak"] = (
            int(self.route_intent_opposite_streak)
        )

        visual_route_conflict = bool(
            self.args.live
            and visual_direction in (-1, 1)
            and candidate_direction in (-1, 1)
            and candidate_direction != visual_direction
        )
        if visual_route_conflict:
            self.pending_execution_actions = []
            self.execution_prefix_step = 0
            self.execution_prefix_audit = {
                "active": True,
                "reason": "visual_bearing_route_conflict",
                "visual_direction": visual_direction,
                "candidate_direction": candidate_direction,
            }
            self._publish_zero()
            self.next_plan_time = now + self.args.retry_delay
            self._log(
                "action_rejected_visual_bearing_conflict",
                visual_bearing=visual_bearing,
                candidate_direction=candidate_direction,
                endpoint_y_m=route_intent_endpoint_y_m,
            )
            self.last_plan_summary = summary
            self._record_plan_hold("visual_bearing_route_conflict")
            return

        self.last_plan_summary = summary

        route_intent_conflict = bool(
            self.args.operator_stop_only
            and self.route_intent_direction
            and candidate_direction
            and candidate_direction != self.route_intent_direction
        )
        if route_intent_conflict:
            self._log(
                "route_intent_local_correction_allowed",
                direction=(
                    "left" if candidate_direction > 0
                    else "right"
                ),
                endpoint_y_m=route_intent_endpoint_y_m,
                consecutive=self.route_intent_opposite_streak,
            )

        current_interlock = (
            self.args.live
            and motion_interlock_allows_nonzero(
                stop_file=self.stop_file,
                arm_file=self.arm_file,
            )
        )
        if self.args.live and not current_interlock:
            # Live mode may warm up and report plans while stopped, but an
            # unarmed plan must never mutate route state or later execute.
            self._publish_zero()
            self.next_plan_time = now + self.args.shadow_period
            self._log(
                "plan_complete_unarmed_shadow",
                summary=summary,
            )
            return
        if self.args.live and (
            not output.job.motion_authorized_at_snapshot
            or output.job.interlock_generation
            != self.interlock_generation
        ):
            self._publish_zero()
            self.next_plan_time = now
            self._log(
                "plan_discarded_authorization_epoch",
                plan_generation=output.job.interlock_generation,
                current_generation=self.interlock_generation,
            )
            return

        far_target_model_reacquire = False
        far_target_model_reacquire_action: np.ndarray | None = None
        continuity, locked, trusted_scale = visual_target_continuity(
            visual_bearing,
            visual_proximity,
            locked=self.target_visual_locked,
            trusted_scale_ratio=self.target_visual_trusted_scale_ratio,
        )
        summary["visual_target_continuity"] = continuity
        if self.args.operator_stop_only:
            self.target_visual_locked = locked
            self.target_visual_trusted_scale_ratio = trusted_scale
            if not continuity["action_allowed"]:
                unconfirmed_anchor = (
                    self.target_visual_unconfirmed_motion_anchor_pose
                )
                unconfirmed_translation = (
                    math.hypot(
                        latest.x - unconfirmed_anchor.x,
                        latest.y - unconfirmed_anchor.y,
                    )
                    if unconfirmed_anchor is not None
                    else drift_m
                )
                (
                    self.target_visual_loss_forward_distance_m,
                    prefetch_motion_budget,
                ) = target_loss_budget_after_prefetch_motion(
                    accumulated_distance_m=(
                        self.target_visual_loss_forward_distance_m
                    ),
                    snapshot_translation_m=unconfirmed_translation,
                    motion_prefetch=bool(
                        output.job.motion_prefetch
                        or unconfirmed_anchor is not None
                    ),
                    previous_continuity_allowed=bool(
                        self.target_visual_last_plan_continuity_allowed
                    ),
                )
                summary["target_loss_prefetch_motion_budget"] = (
                    {
                        **prefetch_motion_budget,
                        "anchor_survived_discarded_plan": bool(
                            unconfirmed_anchor is not None
                            and not output.job.motion_prefetch
                        ),
                    }
                )
                self.target_visual_unconfirmed_motion_anchor_pose = None
                self.target_visual_last_plan_continuity_allowed = False
                self.target_visual_loss_holds += 1
                continuation_audit = summary.get(
                    "target_loss_forward_continuation", {}
                )
                far_target_model_reacquire = (
                    far_target_model_reacquire_allowed(
                        continuation_reason=str(
                            continuation_audit.get("reason", "")
                        ),
                        planner_abstained=bool(result.get("abstained", True)),
                        trusted_scale_ratio=trusted_scale,
                        approach_scale_ratio=VISUAL_APPROACH_SCALE_RATIO,
                    )
                )
                if far_target_model_reacquire:
                    reacquire_distance_limit = max(
                        1.0e-3,
                        self.follower.config.max_action_distance_m - 1.0e-4,
                    )
                    selected_actions = np.asarray(
                        result.get("selected_actions_h4", []),
                        dtype=np.float64,
                    )
                    first_action = (
                        selected_actions[0]
                        if selected_actions.ndim == 2 and len(selected_actions)
                        else np.asarray([], dtype=np.float64)
                    )
                    (
                        far_target_model_reacquire_action,
                        far_target_model_reacquire_audit,
                    ) = bounded_target_loss_forward_action(
                        first_action,
                        planner_abstained=bool(result.get("abstained", True)),
                        trusted_scale_ratio=trusted_scale,
                        accumulated_distance_m=0.0,
                        route_direction=int(self.route_intent_direction),
                        far_step_limit_m=reacquire_distance_limit,
                        far_total_limit_m=reacquire_distance_limit,
                        continuous_arc_max_vx=(
                            self.follower.config.continuous_forward_arc_max_vx
                        ),
                        continuous_arc_min_wz=self.follower.config.drive_min_wz,
                        continuous_arc_turn_tolerance_rad=(
                            self.follower.config.turn_tolerance_rad
                        ),
                        continuous_arc_drive_tolerance_m=(
                            self.follower.config.drive_tolerance_m
                        ),
                    )
                    summary["far_target_model_reacquire_candidate"] = (
                        far_target_model_reacquire_audit
                    )
                    far_target_model_reacquire = bool(
                        far_target_model_reacquire_action is not None
                    )
                if far_target_model_reacquire:
                    # The bounded blind allowance is exhausted, but this fresh
                    # H1 has already passed the same forward, curvature and
                    # route checks. The second shaping pass preserves the full
                    # model H1 distance while reducing yaw only as needed to
                    # keep a short action in DRIVE. Drop the stale far color
                    # lock so the normal goal-conditioned model can reacquire.
                    self.target_visual_locked = False
                    self.target_visual_trusted_scale_ratio = None
                    self.target_visual_loss_holds = 0
                    self.target_visual_loss_forward_distance_m = 0.0
                    self.target_visual_last_plan_continuity_allowed = True
                    self._reset_plan_hold_budget(
                        "far_target_model_reacquire_plan_allowed"
                    )
                    summary["far_target_model_reacquire"] = {
                        "active": True,
                        "reason": "validated_fresh_h1_after_visual_budget",
                        "trusted_scale_ratio": trusted_scale,
                        "h1_only": True,
                        "visual_lock_released": True,
                        "execution_action": (
                            far_target_model_reacquire_action.tolist()
                        ),
                    }
                    self._log(
                        "far_target_model_reacquire_plan_allowed",
                        sequence=output.job.sequence,
                        trusted_scale_ratio=trusted_scale,
                        continuation=continuation_audit,
                    )
                else:
                    self.pending_execution_actions.clear()
                    self.execution_prefix_step = 0
                    self.target_visual_recheck_active = True
                    self.target_visual_recheck_after = (
                        output.completed_time
                    )
                    self.target_visual_recheck_count = 0
                    self.target_visual_recheck_plan = output
                    self.execution_prefix_audit = {
                        "active": True,
                        "reason": "locked_target_visual_continuity_lost",
                        "visual_target_continuity": continuity,
                    }
                    self.last_plan_summary = summary
                    self._publish_zero()
                    self.next_plan_time = now + self.args.retry_delay
                    self._log(
                        "plan_held_locked_target_visual_continuity_lost",
                        sequence=output.job.sequence,
                        loss_holds=self.target_visual_loss_holds,
                        visual_target_continuity=continuity,
                        visual_bearing=visual_bearing,
                        next_stage="fresh_frame_visual_recheck",
                    )
                    self._record_plan_hold(
                        "locked_target_visual_continuity_lost"
                    )
                    return
            if continuity["action_allowed"]:
                self.target_visual_recheck_active = False
                self.target_visual_recheck_after = 0.0
                self.target_visual_recheck_count = 0
                self.target_visual_recheck_step_used = False
                self.target_visual_recheck_plan = None
                self.target_visual_unconfirmed_motion_anchor_pose = None
                self.target_visual_last_plan_continuity_allowed = True
                reset_target_loss_budget = (
                    visual_target_loss_budget_reset_allowed(continuity)
                )
                if self.target_visual_loss_holds and reset_target_loss_budget:
                    self._log(
                        "locked_target_visual_continuity_recovered",
                        previous_loss_holds=self.target_visual_loss_holds,
                        previous_loss_forward_distance_m=(
                            self.target_visual_loss_forward_distance_m
                        ),
                        visual_target_continuity=continuity,
                    )
                    self.target_visual_loss_holds = 0
                    self.target_visual_loss_forward_distance_m = 0.0
                elif self.target_visual_loss_holds:
                    self._log(
                        "locked_target_single_frame_seen_budget_preserved",
                        loss_holds=self.target_visual_loss_holds,
                        loss_forward_distance_m=(
                            self.target_visual_loss_forward_distance_m
                        ),
                        visual_target_continuity=continuity,
                    )

        evidence = GoalEvidence(
            age_s=age,
            temporal_distance=float(
                result["nomad"]["temporal_distance"]
            ),
            cosine_distance=float(
                result["dino"][
                    "current_goal_cosine_distance"
                ]
            ),
            final_topological_node=self.route.is_final,
            sensor_valid=True,
            safety_hold=False,
        )
        decision = self.goal_supervisor.update(evidence)
        decision_payload = decision.to_dict()
        if taught_path_audit.get("enabled"):
            decision_payload["score_reference"] = (
                "taught_local_planning_goal"
                if taught_local_goal_active
                else "terminal_route_goal"
            )
            decision_payload["taught_path_guide"] = dict(
                taught_path_audit
            )
        if self.args.operator_stop_only:
            # Preserve uncalibrated scores as advisory telemetry only. They
            # must never end the route or command a goal-candidate hold.
            decision_payload.update(
                {
                    "advisory_only": True,
                    "automatic_arrival_disabled": True,
                    "manual_stop_required": True,
                    "visual_precision_approach": bool(
                        visual_proximity.get("approach")
                        and not taught_local_goal_active
                    ),
                    "visual_precision_hold": bool(
                        visual_proximity.get("precision_hold")
                        and not taught_local_goal_active
                    ),
                }
            )
            # The operator still owns route termination. A conservative
            # target-scale cue may switch to H1 and a gait-valid approach
            # speed, or hold zero once the centred target has reached the
            # demonstrated image scale. This prevents overshoot without
            # trusting uncalibrated NoMaD distance as an arrival signal.
            self.approach_mode = bool(
                visual_proximity.get("approach")
                and not taught_local_goal_active
            )
        else:
            self.approach_mode = bool(decision.slow_approach)
        self.last_goal_decision = decision_payload
        self._queue_plan_decision_artifact(
            output.job.sequence,
            result,
            summary,
        )
        self._log(
            "plan_complete",
            summary=summary,
            goal_decision=decision_payload,
        )
        if not self.args.live and self.args.shadow_once:
            self.done = True

        if (
            self.args.operator_stop_only
            and bool(visual_proximity.get("precision_hold"))
            and not taught_local_goal_active
        ):
            # In operator-stop-only mode this cue is telemetry, not authority
            # to stop navigation.  The operator explicitly owns termination;
            # continue planning/executing until manual STOP or an independent
            # physical safety interlock fires.
            self.precision_goal_hold = False
            self._log(
                "visual_precision_goal_hold_ignored_operator_stop_only",
                visual_goal_proximity=visual_proximity,
                policy="manual_stop_required_continue_navigation",
            )

        if (
            not self.args.operator_stop_only
            and decision.phase is GoalPhase.STOPPED
        ):
            self._publish_zero()
            self._advance_route()
            return
        if (
            not self.args.operator_stop_only
            and decision.command_zero
        ):
            self._publish_zero()
            self.next_plan_time = now + self.args.retry_delay
            return
        if bool(result["abstained"]):
            hold_exhausted = self._record_plan_hold(
                "planner_abstained",
                continue_replanning_on_budget=bool(
                    self.args.operator_stop_only
                    or taught_path_audit.get("enabled")
                ),
            )
            self._publish_zero()
            self.next_plan_time = now + self.args.retry_delay
            return

        interlock = current_interlock
        if not interlock:
            self._publish_zero()
            self.next_plan_time = now + self.args.shadow_period
            if self.args.shadow_once:
                self.done = True
            return

        directional_prefix_authorized = bool(
            operator_hint_visible
            or prior_consensus_authorized
            or recovery_gate.get("confirmed_visual_direction")
        )
        selected_endpoint_y_m = float(
            summary.get("route_intent_endpoint_y_m") or 0.0
        )
        selected_endpoint_direction = (
            1
            if selected_endpoint_y_m >= 0.06
            else (-1 if selected_endpoint_y_m <= -0.06 else 0)
        )
        taught_execution_direction = int(
            taught_path_audit.get("route_intent_direction", 0) or 0
        )
        planner_execution_direction = int(
            result.get("planner_route_intent_direction", 0) or 0
        )
        trusted_taught_turn = bool(
            taught_local_goal_active
            and taught_path_audit.get("route_intent_trusted", False)
            and taught_execution_direction in (-1, 1)
            and planner_execution_direction == taught_execution_direction
            and selected_endpoint_direction == taught_execution_direction
            and not taught_entry_heading_guard
            and not corridor_active
        )
        execution_prefix, prefix_audit = (
            bounded_directional_execution_prefix(
                np.asarray(result["selected_actions_h4"]),
                enabled=(
                    self.args.operator_stop_only
                    and directional_prefix_authorized
                    and not far_target_model_reacquire
                ),
                max_prefix_steps=min(2, len(result["selected_actions_h4"])),
            )
        )
        h2_prefix, h2_audit = gated_h2_execution_prefix(
            np.asarray(result["selected_actions_h4"]),
            enabled=(
                self.args.operator_stop_only
                and not taught_entry_heading_guard
                and not far_target_model_reacquire
            ),
            # Near-target trajectories use the same H2 coherence gate as the
            # rest of the route. Precision hold above remains authoritative.
            approach_mode=self.approach_mode,
            visual_direction=int(visual_direction),
            visual_conflict=bool(
                visual_endpoint_conflict
                or visual_first_step_conflict
            ),
            max_total_distance_m=(
                0.72 if trusted_taught_turn else LIVE_H2_MAX_TOTAL_DISTANCE_M
            ),
            max_total_abs_yaw_rad=(
                0.40 if trusted_taught_turn else LIVE_H2_MAX_TOTAL_ABS_YAW_RAD
            ),
            max_internal_bearing_change_rad=(
                0.30
                if trusted_taught_turn
                else LIVE_H2_MAX_INTERNAL_BEARING_CHANGE_RAD
            ),
        )
        directional_audit = prefix_audit
        if h2_audit.get("active"):
            execution_prefix = h2_prefix
            prefix_audit = {
                **h2_audit,
                "authorization": (
                    "trusted_taught_turn_h2"
                    if trusted_taught_turn
                    else "default_h2_gate"
                ),
                "taught_execution_direction": taught_execution_direction,
                "directional_prefix": directional_audit,
            }
        elif (
            directional_audit.get("active")
            and directional_audit.get("reason")
            != "bounded_prefix_not_found"
        ):
            # A directional prefix is still subject to the same empirical H2
            # limits.  Falling back here means execute exactly H1 and replan.
            execution_prefix = np.asarray(
                result["selected_actions_h4"],
                dtype=np.float32,
            )[:1]
            prefix_audit = {
                "schema": "v70_gated_h2_execution_prefix_v1",
                "enabled": True,
                "active": False,
                "reason": "gated_h2_fallback",
                "prefix_steps": 1,
                "gated_h2": h2_audit,
                "directional_prefix": directional_audit,
            }
        else:
            prefix_audit = {
                **directional_audit,
                "gated_h2": h2_audit,
            }
        if far_target_model_reacquire:
            assert far_target_model_reacquire_action is not None
            execution_prefix = far_target_model_reacquire_action.reshape(1, 3)
            prefix_audit = {
                "schema": "v70_directional_execution_prefix_v1",
                "enabled": True,
                "active": True,
                "reason": "far_target_model_reacquire_h1",
                "prefix_steps": 1,
                "visual_target_continuity": continuity,
            }
        model_turn_prefix, model_turn_audit = (
            model_amplified_segmented_turn_then_drive(
                execution_prefix[0],
                enabled=False,
                turn_gain=6.0,
                trigger_abs_yaw_rad=0.01,
                minimum_abs_yaw_rad=0.12,
                maximum_abs_yaw_rad=0.60,
                maximum_turn_segment_rad=0.10,
            )
        )
        autonomous_visual_direction = int(
            recovery_gate.get("confirmed_visual_direction", 0)
        )
        visual_turn_override_enabled = bool(
            not far_target_model_reacquire
            and visual_turn_override_allowed(
                confirmed_direction=autonomous_visual_direction,
                center_x_norm=recovery_gate.get("center_x_norm"),
                minimum_center_offset=(
                    VISUAL_TURN_OVERRIDE_MIN_CENTER_OFFSET
                ),
                motion_prefetch=bool(output.job.motion_prefetch),
                observed_scale_ratio=visual_proximity.get("scale_ratio"),
                minimum_scale_ratio=TARGET_LOCK_ACQUIRE_SCALE_RATIO,
            )
        )
        visual_turn_action = np.asarray(
            execution_prefix[0],
            dtype=np.float32,
        ).copy()
        visual_bearing_rad = recovery_gate.get("bearing_rad")
        if (
            autonomous_visual_direction
            and isinstance(visual_bearing_rad, (int, float))
            and math.isfinite(float(visual_bearing_rad))
        ):
            visual_turn_action[2] = float(
                np.clip(
                    float(visual_bearing_rad),
                    -VISUAL_TURN_OVERRIDE_MAX_ABS_YAW_RAD,
                    VISUAL_TURN_OVERRIDE_MAX_ABS_YAW_RAD,
                )
            )
        visual_turn_action, visual_turn_audit = (
            operator_guided_turn_first_action(
                visual_turn_action,
                enabled=visual_turn_override_enabled,
                direction=autonomous_visual_direction,
                turn_gain=1.0,
                minimum_abs_yaw_rad=0.08,
                maximum_abs_yaw_rad=VISUAL_TURN_OVERRIDE_MAX_ABS_YAW_RAD,
            )
        )
        if visual_turn_audit.get("active"):
            pre_fit_action = visual_turn_action.copy()
            visual_turn_action, visual_arc_fit = fit_action_to_continuous_arc(
                visual_turn_action,
                continuous_arc_max_vx=(
                    self.follower.config.continuous_forward_arc_max_vx
                ),
                continuous_arc_min_wz=self.follower.config.drive_min_wz,
                continuous_arc_turn_tolerance_rad=(
                    self.follower.config.turn_tolerance_rad
                ),
                continuous_arc_drive_tolerance_m=(
                    self.follower.config.drive_tolerance_m
                ),
                maximum_distance_m=self.follower.config.max_action_distance_m,
            )
            visual_turn_audit = {
                **visual_turn_audit,
                "pre_arc_fit_execution_action": pre_fit_action.tolist(),
                "execution_action": visual_turn_action.tolist(),
                "continuous_arc_fit": visual_arc_fit,
            }
        summary["model_turn_gain_override"] = model_turn_audit
        summary["autonomous_visual_turn_override"] = {
            **visual_turn_audit,
            "schema": "v70_autonomous_visual_turn_v1",
            "source": "reliable_visual_bearing",
            "planner_minimum_center_offset": (
                DIRECTIONAL_RECOVERY_MIN_CENTER_OFFSET
            ),
            "execution_minimum_center_offset": (
                VISUAL_TURN_OVERRIDE_MIN_CENTER_OFFSET
            ),
            "motion_prefetch_excluded": bool(
                output.job.motion_prefetch
            ),
        }
        self.last_plan_summary = summary
        if visual_turn_audit.get("active"):
            # A reliable, unopposed visual bearing has priority over a CEM
            # first-step sign. Execute its exact aligned endpoint as one
            # follower action. The follower performs the bounded turn assist,
            # turn stability check, and then the drive without throwing away
            # an approach prefix after its first artificial yaw segment.
            execution_prefix = visual_turn_action.reshape(1, 3)
            prefix_audit = {
                "schema": "v70_directional_execution_prefix_v1",
                "enabled": True,
                "active": True,
                "reason": "autonomous_visual_turn_single_action",
                "prefix_steps": 1,
                "visual_direction": autonomous_visual_direction,
                "visual_turn": summary[
                    "autonomous_visual_turn_override"
                ],
            }
        elif model_turn_audit.get("active"):
            # Preserve the autonomous model path: no annotation is consulted.
            # Only the model's own signed yaw is amplified, with every turn
            # segment settled before the final straight drive.
            execution_prefix = model_turn_prefix
            prefix_audit = {
                "schema": "v70_directional_execution_prefix_v1",
                "enabled": True,
                "active": True,
                "reason": "model_turn_gain_override",
                "prefix_steps": int(model_turn_prefix.shape[0]),
                "model_turn_gain": model_turn_audit,
            }
        arc_fit_audits: list[dict[str, Any]] = []
        first_arc_change: int | None = None
        fitted_first_action: np.ndarray | None = None
        for action_index, candidate_action in enumerate(execution_prefix):
            fitted_action, arc_fit = fit_action_to_continuous_arc(
                candidate_action,
                continuous_arc_max_vx=(
                    self.follower.config.continuous_forward_arc_max_vx
                ),
                continuous_arc_min_wz=self.follower.config.drive_min_wz,
                continuous_arc_turn_tolerance_rad=(
                    self.follower.config.turn_tolerance_rad
                ),
                continuous_arc_drive_tolerance_m=(
                    self.follower.config.drive_tolerance_m
                ),
                maximum_distance_m=self.follower.config.max_action_distance_m,
                maximum_abs_yaw_rad=(
                    self.follower.config.continuous_forward_arc_max_yaw_rad
                ),
            )
            arc_fit_audits.append(arc_fit)
            if not np.allclose(
                fitted_action,
                candidate_action,
                rtol=0.0,
                atol=1.0e-6,
            ):
                first_arc_change = action_index
                if action_index == 0:
                    fitted_first_action = fitted_action
                break
        if first_arc_change is not None:
            # If H2's second transition needs reshaping, stop the prefix at H1
            # so its local frame is never applied after a modified predecessor.
            # If H1 needs reshaping, execute that one bounded forward arc and
            # re-observe. This removes stationary model TURN commands without
            # silently chaining a geometrically stale tail.
            if first_arc_change == 0:
                assert fitted_first_action is not None
                execution_prefix = fitted_first_action.reshape(1, 3)
            else:
                execution_prefix = np.asarray(
                    execution_prefix[:first_arc_change],
                    dtype=np.float32,
                )
            prefix_audit = {
                **prefix_audit,
                "reason": "continuous_arc_execution_fit",
                "active": True,
                "prefix_steps": int(len(execution_prefix)),
                "continuous_arc_changed_step": int(first_arc_change),
                "continuous_arc_fits": arc_fit_audits,
            }
        else:
            prefix_audit = {
                **prefix_audit,
                "continuous_arc_fits": arc_fit_audits,
            }
        if (
            prefix_audit.get("active")
            and prefix_audit.get("reason") == "bounded_prefix_not_found"
        ):
            # Do not execute a first transition whose lateral sign still
            # contradicts the selected H4 endpoint. Replan from the same
            # stationary observation; this is a generic trajectory-coherence
            # guard and does not mirror or synthesize an action.
            self.pending_execution_actions = []
            self.execution_prefix_step = 0
            self.execution_prefix_audit = prefix_audit
            self._publish_zero()
            self.next_plan_time = now + self.args.retry_delay
            self._log(
                "action_rejected_directional_prefix_not_realized",
                prefix_audit=prefix_audit,
            )
            self._record_plan_hold("directional_prefix_not_realized")
            return
        # Public release: one freshly selected H1 per unique planning sequence.
        # H2/H4 buffers, old H1 replay and synthesized gap motion are excluded.
        execution_prefix = [np.asarray(result["selected_actions_h4"], dtype=np.float32)[0].copy()]
        prefix_audit = {"enabled": True, "active": False, "prefix_steps": 1,
                        "reason": "fresh_h1_only", "action_reuse": False}
        short_prefetch_allowed, short_prefetch_audit = (
            short_buffered_motion_prefetch_allowed(
                execution_prefix,
                approach_mode=self.approach_mode,
                precision_goal_hold=self.precision_goal_hold,
                max_total_distance_m=(
                    LIVE_BUFFERED_PREFETCH_MAX_DISTANCE_M
                ),
                max_total_abs_yaw_rad=(
                    LIVE_BUFFERED_PREFETCH_MAX_ABS_YAW_RAD
                ),
            )
        )
        prefix_audit = {
            **prefix_audit,
            "short_buffered_motion_prefetch": {
                **short_prefetch_audit,
                # An H2 prefetch launched in H1 produced most of the measured
                # endpoint-tail rebase rejects. Start it from H2 instead,
                # after the continuous handoff resets rolling_prefetch_after.
                "runtime_scheduled": False,
                "runtime_reason": "defer_until_final_prefix",
            },
        }
        self.short_buffered_prefetch_allowed = False
        self.pending_execution_actions = [
            action.copy() for action in execution_prefix[1:]
        ]
        self.execution_prefix_step = 1
        self.execution_prefix_audit = prefix_audit
        self.abstention_forward_distance_m = 0.0
        self.abstention_near_target_step_used = False
        execution_action = execution_prefix[0]
        self.settle_prefetch_after = None
        self.rolling_prefetch_after = now
        authorization = self.fresh_plan_gate.consume(
            sequence=output.job.sequence, interlock=interlock,
            abstained=bool(result["abstained"]), action=execution_action,
        )
        if not authorization.allowed:
            self._publish_zero()
            self.next_plan_time = now + self.args.retry_delay
            self._log("fresh_plan_rejected", sequence=output.job.sequence,
                      reason=authorization.reason)
            return
        start = self.follower.start(execution_action, latest, now)
        self.last_command = start
        if start.fault:
            # A fresh action rejected by the follower latches a stop.
            self._latch_fault(
                f"selected_action_rejected:{start.reason}"
            )
            return
        self._reset_plan_hold_budget("navigation_action_started")
        self._log(
            "action_started",
            selected_first_action=np.asarray(
                result["selected_first_action"]
            ).tolist(),
            execution_action=execution_action.tolist(),
            execution_prefix=prefix_audit,
            follower_phase=start.phase.value,
        )

    def _handoff_ready_plan_at_drive_endpoint(
        self, command: FollowerCommand, latest: Pose2D, now: float, interlock: bool,
    ) -> FollowerCommand:
        """Use a ready plan at a verified DRIVE endpoint.

        Held, stale, invalidated or unauthorized results leave zero output;
        a result can start motion only through the normal plan validator.
        """
        if (
            self.args.rolling_plan_prefetch
            and should_handoff_ready_plan_without_settle(
                command,
                interlock_allows_motion=interlock,
                normal_drive_completion=bool(
                    self.follower.reason
                    == "drive_complete_settle_started"
                ),
            )
        ):
            try:
                handoff_output = self.output_queue.get_nowait()
            except queue.Empty:
                handoff_output = None
            if handoff_output is not None:
                completed_progress = command.progress_m
                completed_step = self.execution_prefix_step
                self.follower.reset("ready_plan_drive_endpoint_handoff")
                self._log(
                    "action_complete",
                    execution_prefix_step=completed_step,
                    progress_m=completed_progress,
                    cross_track_m=command.cross_track_m,
                    yaw_error_rad=command.yaw_error_rad,
                    completion_reason="ready_plan_drive_endpoint_handoff",
                    replan_required=False,
                )
                self._handle_plan_output(handoff_output, now)
                if self.follower.active and self.fault_reason is None:
                    command = self.follower.update(
                        latest, now,
                        interlock_allows_motion=interlock,
                        motion_applied=False,
                    )
                    self._log(
                        "rolling_plan_continuous_handoff",
                        sequence=handoff_output.job.sequence,
                        follower_phase=command.phase.value,
                        completed_progress_m=completed_progress,
                    )
                else:
                    command = FollowerCommand.zero(
                        self.follower.phase,
                        "ready_plan_handoff_held_or_rejected",
                    )
        return command

    def _drain_planner_output(self, now: float) -> None:
        # A settle-prefetched result must never replace an action follower
        # that is still active.  Leave it queued until the zero-velocity
        # settle completes; the normal snapshot drift and authorization
        # checks still run before the result can start the next action.
        if self.follower.active and not self.planning_gap_bridge_active:
            return
        try:
            output = self.output_queue.get_nowait()
        except queue.Empty:
            return
        if self.planning_gap_bridge_active:
            self.follower.reset("inflight_forward_bridge_preempted_by_plan")
            self.planning_gap_bridge_active = False
            self.planning_gap_bridge_started_time = None
            self._log(
                "inflight_forward_bridge_preempted",
                sequence=output.job.sequence,
            )
        self._handle_plan_output(output, now)

    def _control_timer(self) -> None:
        now = time.monotonic()
        latest = self.odom.latest()
        camera_error: str | None = None
        odom_error: str | None = None
        try:
            self.camera.snapshot(
                count=4,
                max_age_s=self.args.camera_max_age,
            )
        except Exception as exc:
            camera_error = f"{type(exc).__name__}:{exc}"
        try:
            self.odom.snapshot(
                count=4,
                interval_s=self.args.odom_interval,
                max_age_s=self.args.odom_max_age,
            )
        except Exception as exc:
            odom_error = f"{type(exc).__name__}:{exc}"
        sensor_ready = camera_error is None and odom_error is None
        sensor_error = camera_error or odom_error
        self.last_sensor_error = sensor_error

        if sensor_ready:
            self.last_sensor_ready_time = now
            if self.camera_recovery_started is not None:
                self.camera_recovery_count += 1
                self._log(
                    "camera_recovered",
                    recovery_elapsed_s=(
                        now - self.camera_recovery_started
                    ),
                    restart_attempts=self.camera_restart_attempts,
                    camera=self.camera.diagnostics(),
                )
                self.camera_recovery_started = None
                if self.scan_camera_paused:
                    self.scan_camera_paused = False
                    if latest is None:
                        self._latch_fault(
                            "missing_odom_after_scan_camera_recovery",
                            EXIT_SENSOR,
                        )
                        return
                    # The camera outage can last several seconds, so the old
                    # follower deadline and its open-loop action are no
                    # longer valid.  Recompute only the unfinished scan step
                    # from fresh odometry; already captured history remains.
                    self._start_scan_step(latest, now)
                    self._log(
                        "active_history_scan_resumed_after_camera_recovery",
                        scan_step=self.scan_step_index,
                        captured_frames=len(self.scan_images),
                    )
        elif not self.ready and (
            now - self.started_time > self.args.startup_timeout
        ):
            self._latch_fault(
                f"sensor_startup_timeout:{sensor_error}",
                EXIT_SENSOR,
            )
            return
        elif self.ready and self.last_sensor_ready_time is not None:
            loss_elapsed = now - self.last_sensor_ready_time
            motion_active = bool(
                self.follower.active or self.scan_state == "TURNING"
            )
            camera_only = camera_error is not None and odom_error is None
            planning_busy = bool(
                self.plan_inflight and self.output_queue.empty()
            )
            scan_camera_recoverable = active_scan_camera_recovery_allowed(
                camera_only=camera_only,
                scan_enabled=self.active_history_scan_enabled,
                scan_state=self.scan_state,
            )
            if camera_only and (not motion_active or scan_camera_recoverable):
                self._publish_zero()
                if self.camera_recovery_started is None:
                    self.camera_recovery_started = now
                    if scan_camera_recoverable:
                        self.scan_camera_paused = True
                        self.scan_follower.reset(
                            "scan_camera_recovery_pause"
                        )
                    self._log(
                        "camera_recovery_started",
                        reason=camera_error,
                        planning_busy=planning_busy,
                        active_scan_paused=scan_camera_recoverable,
                        camera=self.camera.diagnostics(),
                    )
                recovery_elapsed = now - self.camera_recovery_started
                # CEM planning contains Python-heavy sections that can
                # temporarily starve the camera reader thread.  No action is
                # running in this phase, so hold zero and let the queued JPEG
                # data drain before considering a worker restart.
                planning_grace = self.args.planner_timeout + 1.0
                if planning_busy and recovery_elapsed <= planning_grace:
                    self._write_status(now)
                    return
                restart_due = (
                    recovery_elapsed >= self.args.camera_restart_delay
                    and self.camera_restart_attempts
                    < self.args.camera_restart_limit
                    and not self.camera.diagnostics()[
                        "restart_in_progress"
                    ]
                    and (
                        self.camera_last_restart_time is None
                        or now - self.camera_last_restart_time
                        >= self.args.camera_restart_delay
                    )
                )
                if restart_due:
                    if self.camera.restart_async():
                        self.camera_restart_attempts += 1
                        self.camera_last_restart_time = now
                        self._log(
                            "camera_worker_restart_requested",
                            attempt=self.camera_restart_attempts,
                            camera=self.camera.diagnostics(),
                        )
                    self._write_status(time.monotonic(), force=True)
                    return
                if recovery_elapsed > self.args.camera_recovery_timeout:
                    self._latch_fault(
                        f"camera_recovery_timeout:{camera_error}",
                        EXIT_SENSOR,
                    )
                    return
                self._write_status(now)
                return

            # Any stale sensor immediately gates motion to zero.  Persistent
            # loss while turning/driving still fails closed on the original
            # short timeout; it is never masked by camera recovery.
            self._publish_zero()
            if loss_elapsed > self.args.sensor_loss_timeout:
                self._latch_fault(
                    f"sensor_loss_timeout:{sensor_error}",
                    EXIT_SENSOR,
                )
                return
            self._write_status(now)
            return

        if sensor_ready and not self.ready:
            self.ready = True
            mode = "live" if self.args.live else "shadow"
            self.ready_file.parent.mkdir(
                parents=True,
                exist_ok=True,
            )
            self.ready_file.write_text(
                f"v70_live_ready_v1 {mode} {os.getpid()}\n",
                encoding="utf-8",
            )
            self._log(
                "ready",
                load_timings_ms=self.planner.load_timings_ms,
            )

        # Never turn a completed plan into motion until both camera and odom
        # are fresh again.  In particular, this keeps planner-side GIL stalls
        # from racing the active scan or follower start.
        self._drain_planner_output(now)

        interlock = (
            self.args.live
            and self.ready
            and self.fault_reason is None
            and not self.route.complete
            and motion_interlock_allows_nonzero(
                stop_file=self.stop_file,
                arm_file=self.arm_file,
            )
        )
        if interlock != self.last_interlock:
            self.interlock_generation += 1
        if self.last_interlock and not interlock:
            self.follower.reset("interlock_closed")
            self._reset_active_scan("interlock_closed")
            self.target_visual_locked = False
            self.target_visual_trusted_scale_ratio = None
            self.target_visual_loss_holds = 0
            self.target_visual_loss_forward_distance_m = 0.0
            self.abstention_forward_distance_m = 0.0
            self.abstention_near_target_step_used = False
            self.target_visual_last_plan_continuity_allowed = True
            self.target_visual_unconfirmed_motion_anchor_pose = None
            self.target_visual_recheck_active = False
            self.target_visual_recheck_after = 0.0
            self.target_visual_recheck_count = 0
            self.target_visual_recheck_step_used = False
            self.target_visual_recheck_plan = None
            self.route_intent_direction = 0
            self.route_intent_endpoint_y_m = None
            self.route_intent_opposite_streak = 0
            self.pending_execution_actions.clear()
            self.execution_prefix_step = 0
            self.execution_prefix_audit = None
            self.short_buffered_prefetch_allowed = False
            self.planning_gap_bridge_active = False
            self.planning_gap_bridge_sequence = None
            self.planning_gap_bridge_started_time = None
            self.settle_prefetch_after = None
            self.next_plan_time = now
            self._log("motion_interlock_closed")
        if interlock and not self.last_interlock:
            # Never execute a shadow/stale plan after arming.
            self.follower.reset("interlock_opened_requires_fresh_plan")
            self._reset_active_scan(
                "interlock_opened_requires_fresh_scan"
            )
            self.pending_execution_actions.clear()
            self.execution_prefix_step = 0
            self.execution_prefix_audit = None
            self.short_buffered_prefetch_allowed = False
            self.planning_gap_bridge_active = False
            self.planning_gap_bridge_sequence = None
            self.planning_gap_bridge_started_time = None
            self.settle_prefetch_after = None
            self.rolling_prefetch_after = None
            self.target_visual_recheck_active = False
            self.target_visual_recheck_after = 0.0
            self.target_visual_recheck_count = 0
            self.target_visual_recheck_step_used = False
            self.target_visual_recheck_plan = None
            self.next_plan_time = now
            self.active_scan_pending = self.active_history_scan_enabled
            self.active_scan_reason = "interlock_opened_initial_alignment"
            self._log("motion_interlock_opened")
        self.last_interlock = interlock

        if self.fault_reason is not None or self.route.complete:
            self.last_command = FollowerCommand.zero(
                self.follower.phase,
                self.fault_reason or "route_complete",
            )
            self._publish_zero()
            self._write_status(now)
            return

        if self.external_control_rebase_pending and interlock:
            if latest is None:
                self._latch_fault("missing_odom_during_external_rebase")
                return
            self._publish_zero()
            if not self._update_external_control_rebase(latest, now):
                self._write_status(now)
                return
            # Stability has just been confirmed. Keep this control tick at
            # zero and begin the fresh scan/reanalysis on the next tick.
            self._write_status(now)
            return

        if (
            self._active_scan_required_for_next_plan()
            and interlock
            and not self.plan_inflight
            and not self.follower.active
            and self.follower.phase is not FollowerPhase.COMPLETE
        ):
            if latest is None:
                self._latch_fault("missing_odom_during_active_scan")
                return
            try:
                self._update_active_scan(
                    latest=latest,
                    now=now,
                    interlock=interlock,
                )
            except Exception as exc:
                self._latch_fault(
                    "active_history_scan_failed:"
                    f"{type(exc).__name__}:{exc}",
                    EXIT_SENSOR,
                )
            self._write_status(now)
            return

        if self.follower.active or (
            self.follower.phase is FollowerPhase.COMPLETE
        ):
            if latest is None:
                self._latch_fault("missing_odom_during_action")
                return
            motion_applied = self._safe_motion_applied(now)
            self.safe_feedback_motion_applied = motion_applied
            if self.planning_gap_bridge_active:
                bridge_elapsed = (
                    now - self.planning_gap_bridge_started_time
                    if self.planning_gap_bridge_started_time is not None
                    else math.inf
                )
                bridge_stop_reason = None
                if bridge_elapsed >= 0.65:
                    bridge_stop_reason = "wall_time_limit"
                elif not motion_applied and bridge_elapsed >= 0.15:
                    bridge_stop_reason = "safe_motion_not_applied"
                if bridge_stop_reason is not None:
                    self.follower.reset(
                        f"inflight_forward_bridge_{bridge_stop_reason}"
                    )
                    self.planning_gap_bridge_active = False
                    self.planning_gap_bridge_started_time = None
                    self.last_command = FollowerCommand.zero(
                        self.follower.phase,
                        f"inflight_forward_bridge_{bridge_stop_reason}",
                    )
                    self._publish_zero()
                    self._log(
                        "inflight_forward_bridge_stopped",
                        sequence=self.planning_gap_bridge_sequence,
                        reason=bridge_stop_reason,
                        elapsed_s=bridge_elapsed,
                    )
                    self._write_status(now)
                    return
            command = self.follower.update(
                latest,
                now,
                interlock_allows_motion=interlock,
                motion_applied=motion_applied,
            )
            # Prefer a newer fully validated plan at every real action
            # endpoint. If none is ready, the old bounded prefix continues
            # below without inserting a zero command.
            command = self._handoff_ready_plan_at_drive_endpoint(
                command, latest, now, interlock,
            )
            if (
                INFLIGHT_FORWARD_BRIDGE_ENABLED
                and command.phase is FollowerPhase.SETTLE
                and command.reason == "drive_complete_settle_started"
                and not self.pending_execution_actions
            ):
                bridge_command = None
                if bridge_command is not None:
                    command = bridge_command
            if self.planning_gap_bridge_active and command.enable_motion:
                command = replace(
                    command,
                    vx=min(command.vx, INFLIGHT_FORWARD_BRIDGE_MAX_VX),
                    wz=0.0,
                    reason=f"{command.reason}:inflight_forward_bridge",
                )
            self.last_command = command
            follower_evidence = self._maybe_log_follower_control(
                command=command,
                pose=latest,
                now=now,
                motion_applied=motion_applied,
            )
            planned_motion_session = bool(
                self.args.operator_stop_only
                and interlock
                and self.follower.active
                and not command.complete
                and not command.fault
            )
            self._publish(
                command.vx,
                command.wz,
                # Keep the executor in Move mode through follower phase-entry
                # and planned settle zeros. Safe Bridge owns the velocity ramp;
                # independent STOP/ARM and sensor gates still cut it instantly.
                enable=bool(
                    command.enable_motion or planned_motion_session
                ),
            )
            if (
                self.args.rolling_plan_prefetch
                and self.rolling_prefetch_after is not None
                and rolling_motion_prefetch_ready(
                    command,
                    action_elapsed_s=max(
                        0.0,
                        now - self.rolling_prefetch_after,
                    ),
                    final_prefix_action=(
                        not self.pending_execution_actions
                    ),
                    buffered_prefix_action=bool(
                        self.pending_execution_actions
                        and self.short_buffered_prefetch_allowed
                    ),
                    drive_trigger_progress_m=(
                        ROLLING_PREFETCH_TRIGGER_PROGRESS_M
                    ),
                    buffered_drive_trigger_progress_m=(
                        ROLLING_PREFETCH_BUFFERED_TRIGGER_PROGRESS_M
                    ),
                    turn_trigger_delay_s=(
                        ROLLING_TURN_PREFETCH_DELAY_S
                    ),
                    maximum_drive_yaw_error_rad=(
                        ROLLING_PREFETCH_MAX_DRIVE_YAW_ERROR_RAD
                    ),
                )
                and not self.plan_inflight
                and interlock
            ):
                self._request_plan(
                    now,
                    allow_follower_motion=True,
                    minimum_camera_receive_time=self.rolling_prefetch_after,
                )
            if (
                self.args.settle_plan_prefetch
                and not self.settle_prefetch_suspended
                and command.phase is FollowerPhase.SETTLE
                and command.reason == "drive_complete_settle_started"
                and not self.pending_execution_actions
                and interlock
            ):
                self.settle_prefetch_after = now
                self._log(
                    "settle_plan_prefetch_waiting_for_endpoint_frame",
                    sequence=self.plan_sequence,
                    minimum_camera_receive_time=now,
                )
            if (
                self.args.settle_plan_prefetch
                and not self.settle_prefetch_suspended
                and self.settle_prefetch_after is not None
                and command.phase is FollowerPhase.SETTLE
                and not self.pending_execution_actions
                and not self.plan_inflight
                and interlock
            ):
                self._request_plan(
                    now,
                    allow_follower_settle=True,
                    minimum_camera_receive_time=(
                        self.settle_prefetch_after
                    ),
                )
            if command.fault:
                self.settle_prefetch_after = None
                if self.inflight_plan_sequence is not None:
                    self.invalidated_plan_sequences.add(
                        self.inflight_plan_sequence
                    )
                if (
                    self.args.aggressive_navigation
                    and command.reason
                    in {
                        "drive_no_progress",
                        "drive_safety_block_timeout",
                        "turn_no_progress",
                        "turn_safety_block_timeout",
                    }
                ):
                    # In aggressive mode, a physical drive/turn stall or a
                    # sustained Safe Bridge block abandons only the current
                    # action. Publish zero first, discard the stale execution
                    # prefix, then re-observe and replan. This avoids latching
                    # the whole navigation run on a transient gait stall.
                    # Independent STOP/ARM, Safe Bridge, LiDAR and executor
                    # protections remain authoritative.
                    self._publish_zero()
                    self.pending_execution_actions.clear()
                    self.execution_prefix_step = 0
                    self.execution_prefix_audit = None
                    self.follower.reset(
                        f"aggressive_{command.reason}_replan"
                    )
                    self.next_plan_time = now
                    if command.reason in {
                        "turn_safety_block_timeout",
                        "turn_no_progress",
                    }:
                        self._request_adaptive_scan(
                            f"action_fault:{command.reason}"
                        )
                    self._log(
                        "action_recoverable_fault_replan_requested",
                        recovery_reason=command.reason,
                        details=follower_evidence,
                    )
                    return
                self._latch_fault(
                    f"action_follower:{command.reason}",
                    details=follower_evidence,
                )
                return
            if command.complete:
                self.settle_prefetch_after = None
                self._log(
                    "action_complete",
                    execution_prefix_step=(
                        self.execution_prefix_step
                    ),
                    completion_reason=command.reason,
                    replan_required=command.replan_required,
                    progress_m=command.progress_m,
                    cross_track_m=command.cross_track_m,
                    yaw_error_rad=command.yaw_error_rad,
                )
                if command.replan_required:
                    preserve_prefetch = (
                        should_preserve_prefetched_plan_after_partial_completion(
                            inflight_sequence=(
                                self.inflight_plan_sequence
                            ),
                            settle_prefetch=(
                                self.inflight_plan_settle_prefetch
                            ),
                            motion_prefetch=(
                                self.inflight_plan_motion_prefetch
                            ),
                        )
                    )
                    if (
                        self.inflight_plan_sequence is not None
                        and not preserve_prefetch
                    ):
                        self.invalidated_plan_sequences.add(
                            self.inflight_plan_sequence
                        )
                    # The physical controller deliberately ended only this
                    # execution burst. Never continue the old H4 prefix. A
                    # Prefetched work still passes the normal age, SE(2)
                    # rebase, route, target and authorization checks. Preserve
                    # it instead of forcing another 2--4 second inference gap.
                    self.pending_execution_actions.clear()
                    self.execution_prefix_step = 0
                    self.execution_prefix_audit = None
                    self.follower.reset(
                        (
                            "physical_completion_waiting_for_prefetch"
                            if preserve_prefetch
                            else "physical_progress_requires_replan"
                        )
                    )
                    self.next_plan_time = now
                    if preserve_prefetch:
                        self._log(
                            "prefetch_preserved_after_physical_completion",
                            sequence=self.inflight_plan_sequence,
                            settle_prefetch=(
                                self.inflight_plan_settle_prefetch
                            ),
                            motion_prefetch=(
                                self.inflight_plan_motion_prefetch
                            ),
                            completion_reason=command.reason,
                            progress_m=command.progress_m,
                        )
                    else:
                        self._log(
                            "physical_progress_replan_requested",
                            completion_reason=command.reason,
                            progress_m=command.progress_m,
                        )
                self.pending_execution_actions.clear()
                self.execution_prefix_step = 0
                self.execution_prefix_audit = None
                self.rolling_prefetch_after = None
                self.follower.reset("ready_for_replan")
                self.next_plan_time = now
        else:
            self.safe_feedback_motion_applied = False
            self.last_command = FollowerCommand.zero(
                self.follower.phase,
                "planning_or_waiting",
            )
            self._publish_zero()
            if sensor_ready:
                self._request_plan(now)

        self._write_status(now)

    def shutdown(self) -> None:
        if self.shutdown_started:
            return
        self.shutdown_started = True
        self._reset_active_scan("shutdown")
        self.stop_file.parent.mkdir(parents=True, exist_ok=True)
        self.stop_file.touch(exist_ok=True)
        if rclpy.ok():
            for _ in range(8):
                if not self._publish_zero():
                    break
                time.sleep(0.03)
        self.worker_stop.set()
        try:
            self.job_queue.put_nowait(None)
        except queue.Full:
            pass
        # A CUDA planning call cannot be cancelled safely from a Python
        # thread.  Let the already-running call finish under the same
        # bounded planner timeout instead of tearing the interpreter down
        # around an active PyTorch/CUDA thread (which can enter D state).
        planning_join_timeout_s = self.args.planner_timeout + 2.0
        self.worker.join(timeout=planning_join_timeout_s)
        self.planning_worker_alive_after_join = self.worker.is_alive()
        if self.planning_worker_alive_after_join:
            try:
                self._log(
                    "planning_worker_shutdown_timeout",
                    join_timeout_s=planning_join_timeout_s,
                )
            except Exception:
                pass
        if self.snapshot_executor is not None:
            self.snapshot_executor.shutdown(
                wait=True,
                cancel_futures=False,
            )
            self.snapshot_executor = None
            while self.snapshot_futures:
                future = self.snapshot_futures.popleft()
                try:
                    future.result()
                except Exception as exc:
                    try:
                        self._log(
                            "plan_image_writer_shutdown_error",
                            error=f"{type(exc).__name__}:{exc}",
                        )
                    except Exception:
                        pass
        if self.ready_file.exists():
            self.ready_file.unlink()
        self.ready = False
        self.plan_inflight = False
        self.last_command = FollowerCommand.zero(
            self.follower.phase,
            self.fault_reason or "shutdown_stop_latched",
        )
        self.shutdown_complete = True
        self._write_status(time.monotonic(), force=True)
        self._log("shutdown", exit_code=self.exit_code)


def main() -> int:
    args = parse_args()
    shutdown_signal: int | None = None
    camera: UnitreeCameraBuffer | None = None
    node: V70LiveNavNode | None = None
    ros_initialized = False
    exit_code = EXIT_OK
    stage = "configuration"
    force_context_shutdown = (
        os.environ.get("V70_TEST_FORCE_CONTEXT_SHUTDOWN", "0") == "1"
    )

    def request_shutdown(signum, _frame) -> None:
        nonlocal shutdown_signal
        shutdown_signal = int(signum)
        stop_file = args.stop_file.expanduser().resolve()
        stop_file.parent.mkdir(parents=True, exist_ok=True)
        stop_file.touch(exist_ok=True)

    signal.signal(signal.SIGTERM, request_shutdown)
    signal.signal(signal.SIGINT, request_shutdown)
    try:
        validate_args(args)
        if args.live and force_context_shutdown:
            raise RuntimeError(
                "context fault injection is forbidden in live mode"
            )
        route = RouteManifest.load(
            args.route_manifest,
            require_calibrated=(
                args.live
                and not args.operator_stop_only
                and not args.scan_only
            ),
        )
        if len(route.goals) != 1:
            raise RuntimeError(
                "this V4 entry point supports exactly one goal image"
            )
        goal_profile_startup = []
        for goal in route.goals:
            goal_sha256 = sha256_file(goal.image)
            profile = load_goal_visual_profile(goal.image, goal_sha256)
            goal_profile_startup.append(
                {
                    "goal": goal.name,
                    "image": str(goal.image),
                    "goal_sha256": goal_sha256,
                    "visual_profile": (
                        {
                            "target_label": profile.get("target_label"),
                            "minimum_lateral_consensus": profile[
                                "minimum_lateral_consensus"
                            ],
                            "planner_fallback_when_target_unobserved": (
                                profile[
                                    "planner_fallback_when_target_unobserved"
                                ]
                            ),
                        }
                        if profile is not None
                        else None
                    ),
                }
            )
        print(json.dumps(
            {"goal_profile_startup": goal_profile_startup},
            indent=2,
        ))

        stage = "model_load"
        print("Loading persistent navigation models...")
        shadow_recovery_value = os.environ.get(
            "V70_ENABLE_SUPPORT_RECOVERY_SHADOW",
            "0",
        )
        if shadow_recovery_value not in ("0", "1"):
            raise ValueError(
                "V70_ENABLE_SUPPORT_RECOVERY_SHADOW must be 0 or 1"
            )
        support_recovery_enabled = bool(
            args.operator_stop_only
            or (
                not args.live
                and shadow_recovery_value == "1"
            )
        )
        persistent_planner_module.cem.ITERATIONS = int(
            args.cem_iterations
        )
        planner = PersistentV70Planner(
            enable_support_recovery=support_recovery_enabled,
            # Route-intent compatibility is advisory in live operation.  The
            # physical safety filters remain active, but a transient visual
            # side disagreement must not discard every otherwise-safe action.
            enable_reference_intent_guard=(
                args.operator_stop_only and not args.live
            ),
            adaptive_cem=bool(args.adaptive_cem),
            adaptive_cem_min_safe=int(args.adaptive_cem_min_safe),
            enable_fixed_pool_fast_path=bool(
                args.fixed_pool_fast_path
            ),
            enable_dino_feature_cache=bool(args.dino_feature_cache),
        )
        precision_contract = getattr(
            planner,
            "startup_contract",
            {
                "weights": "unchanged sealed V70",
                "dino_batch": "4 observations + 1 goal",
                "ltwm_members": 3,
                "selector_members": 3,
                "dino_feature_cache": bool(args.dino_feature_cache),
                "fixed_pool_fast_path": bool(
                    args.fixed_pool_fast_path
                ),
                "settle_plan_prefetch": bool(
                    args.settle_plan_prefetch
                ),
                "async_plan_images": bool(args.async_plan_images),
                "cem": (
                    f"{args.cem_iterations} iteration(s) x "
                    "population 16"
                ),
            },
        )
        print(json.dumps(
            {
                "planner_backend": getattr(planner, "mode", "v70"),
                "load_timings_ms": planner.load_timings_ms,
                "support_recovery_enabled": (
                    support_recovery_enabled
                ),
                "reference_intent_guard_enabled": (
                    args.operator_stop_only
                ),
                "precision_contract": precision_contract,
            },
            indent=2,
        ))
        if shutdown_signal is not None:
            raise InterruptedError(
                f"signal_{shutdown_signal}_during_model_load"
            )

        stage = "camera_start"
        camera = UnitreeCameraBuffer(
            args.interface,
            interval_s=args.camera_interval,
            corrupt_dir=(
                args.output_dir.expanduser().resolve()
                / "corrupt_camera_frames"
            ),
        )
        camera.start()
        rclpy.init()
        ros_initialized = True
        # rclpy.init() installs its own signal handlers.  Restore the V70
        # handler so SIGTERM is recorded as an explicit bounded shutdown
        # instead of an ambiguous ExternalShutdownException.
        signal.signal(signal.SIGTERM, request_shutdown)
        signal.signal(signal.SIGINT, request_shutdown)
        node = V70LiveNavNode(
            args=args,
            planner=planner,
            camera=camera,
            route=route,
        )
        stage = "runtime"
        while rclpy.ok() and not node.done:
            if shutdown_signal is not None:
                if args.operator_stop_only:
                    node._log(
                        "manual_stop_requested",
                        signal=int(shutdown_signal),
                        stop_source="operator_controller",
                    )
                node._latch_fault(
                    f"signal_{shutdown_signal}",
                    EXIT_SIGNAL,
                )
                break
            rclpy.spin_once(node, timeout_sec=0.05)
            if force_context_shutdown and node.ready:
                node._log(
                    "test_force_ros_context_shutdown"
                )
                rclpy.shutdown()
        if shutdown_signal is not None and node.fault_reason is None:
            node._latch_fault(
                f"signal_{shutdown_signal}",
                EXIT_SIGNAL,
            )
        elif not node.done and not rclpy.ok():
            node._latch_fault(
                "ros_context_became_invalid",
                EXIT_RUNTIME,
            )
        exit_code = node.exit_code
    except ExternalShutdownException as exc:
        exit_code = EXIT_RUNTIME
        if node is not None:
            node._latch_fault(
                f"external_shutdown:{type(exc).__name__}:{exc}",
                exit_code,
            )
        else:
            write_startup_fault_status(
                args,
                stage=stage,
                error=exc,
                exit_code=exit_code,
            )
    except BaseException as exc:
        if isinstance(exc, (KeyboardInterrupt, InterruptedError)):
            exit_code = EXIT_SIGNAL
        elif stage == "configuration":
            exit_code = EXIT_CONFIG
        elif stage == "model_load":
            exit_code = EXIT_MODEL
        elif stage == "camera_start":
            exit_code = EXIT_SENSOR
        else:
            exit_code = EXIT_RUNTIME
        if node is not None:
            node._latch_fault(
                f"{stage}_failed:{type(exc).__name__}:{exc}",
                exit_code,
            )
        else:
            write_startup_fault_status(
                args,
                stage=stage,
                error=exc,
                exit_code=exit_code,
            )
    finally:
        if node is not None:
            try:
                node.shutdown()
                exit_code = node.exit_code
            except Exception as exc:
                exit_code = EXIT_RUNTIME
                node.exit_code = exit_code
                node.fault_reason = node.fault_reason or (
                    f"shutdown_failed:{type(exc).__name__}:{exc}"
                )
                node.stop_file.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )
                node.stop_file.touch(exist_ok=True)
                try:
                    node._write_status(time.monotonic(), force=True)
                except Exception:
                    pass
        if camera is not None:
            try:
                camera.stop()
            except Exception:
                exit_code = EXIT_RUNTIME
        if node is not None:
            try:
                node.destroy_node()
            except Exception:
                exit_code = EXIT_RUNTIME
        if ros_initialized and rclpy.ok():
            rclpy.shutdown()
    return int(exit_code)


if __name__ == "__main__":
    raise SystemExit(main())
