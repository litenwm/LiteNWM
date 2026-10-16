#!/usr/bin/env python3
"""Validated, hash-bound visual profiles for V70 image goals.

Profiles are advisory target evidence.  They may constrain a lateral action
only when enough frames agree; they never encode a fixed route turn.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Sequence


PROFILE_SCHEMA = "v70_goal_visual_profile_v1"


def _validated_roi(value: object) -> list[float]:
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError("goal visual profile target_roi_norm is invalid")
    if any(isinstance(item, bool) for item in value):
        raise ValueError("goal visual profile ROI contains a boolean")
    try:
        left, top, right, bottom = (float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise ValueError("goal visual profile ROI is not numeric") from exc
    if not (
        0.0 <= left < right <= 1.0
        and 0.0 <= top < bottom <= 1.0
        and right - left >= 0.08
        and bottom - top >= 0.08
    ):
        raise ValueError("goal visual profile ROI is outside image bounds")
    return [left, top, right, bottom]


def validate_goal_visual_profile(
    payload: object,
    *,
    goal_sha256: str,
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("goal visual profile must be a JSON object")
    result = dict(payload)
    if result.get("schema") != PROFILE_SCHEMA:
        raise ValueError("unsupported goal visual profile schema")
    profile_sha = str(result.get("goal_sha256", "")).lower()
    expected_sha = str(goal_sha256).lower()
    if profile_sha != expected_sha:
        raise ValueError(
            "goal visual profile SHA-256 does not match goal pixels: "
            f"profile={profile_sha or '<missing>'} goal={expected_sha}"
        )
    result["goal_sha256"] = expected_sha
    result["target_roi_norm"] = _validated_roi(
        result.get("target_roi_norm")
    )
    minimum = result.get("minimum_lateral_consensus", 1)
    if isinstance(minimum, bool) or not isinstance(minimum, int):
        raise ValueError("minimum_lateral_consensus must be an integer")
    if not 1 <= minimum <= 4:
        raise ValueError("minimum_lateral_consensus must be 1..4")
    result["minimum_lateral_consensus"] = minimum
    bearing_cue = result.get("bearing_cue", "auto")
    if bearing_cue not in {"auto", "feature_only", "profile_color"}:
        raise ValueError(
            "bearing_cue must be 'auto', 'feature_only', or 'profile_color'"
        )
    result["bearing_cue"] = bearing_cue
    target_center = result.get("target_center_x_norm", 0.5)
    if isinstance(target_center, bool):
        raise ValueError("target_center_x_norm must be numeric")
    try:
        target_center = float(target_center)
    except (TypeError, ValueError) as exc:
        raise ValueError("target_center_x_norm must be numeric") from exc
    if not math.isfinite(target_center) or not 0.0 <= target_center <= 1.0:
        raise ValueError("target_center_x_norm must be finite and in [0, 1]")
    result["target_center_x_norm"] = target_center
    fallback = result.get("planner_fallback_when_target_unobserved", False)
    if not isinstance(fallback, bool):
        raise ValueError(
            "planner_fallback_when_target_unobserved must be boolean"
        )
    result["planner_fallback_when_target_unobserved"] = fallback
    return result


def load_goal_visual_profile(
    goal_path: Path,
    goal_sha256: str,
) -> dict[str, Any] | None:
    profile_path = goal_path.with_name(goal_path.name + ".visual.json")
    if not profile_path.is_file():
        return None
    try:
        payload = json.loads(profile_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read goal visual profile: {exc}") from exc
    return validate_goal_visual_profile(
        payload,
        goal_sha256=goal_sha256,
    )


def apply_goal_visual_profile(
    visual_bearing: dict[str, Any],
    profile: dict[str, Any],
) -> dict[str, Any]:
    """Apply confidence gating while preserving raw evidence for audits."""

    result = dict(visual_bearing)
    minimum = int(profile["minimum_lateral_consensus"])
    fallback = bool(profile["planner_fallback_when_target_unobserved"])
    result["goal_visual_profile"] = {
        "schema": profile["schema"],
        "target_label": profile.get("target_label"),
        "goal_sha256": profile["goal_sha256"],
        "target_roi_norm": list(profile["target_roi_norm"]),
        "minimum_lateral_consensus": minimum,
        "bearing_cue": profile["bearing_cue"],
        "target_center_x_norm": profile["target_center_x_norm"],
        "planner_fallback_when_target_unobserved": fallback,
    }
    raw_direction = int(result.get("direction", 0))
    consensus = int(result.get("consensus_count", 0))
    opposing = int(result.get("opposing_count", 0))
    if raw_direction in (-1, 1) and (
        consensus < minimum or opposing > 0
    ):
        result["profile_lateral_direction_suppressed"] = True
        result["raw_profile_direction"] = raw_direction
        result["raw_profile_direction_name"] = result.get("direction_name")
        result["raw_profile_reason"] = result.get("reason")
        result["direction"] = 0
        result["direction_name"] = "center"
        result["center_x_norm"] = 0.5
        result["bearing_rad"] = 0.0
        result["reason"] = (
            "goal_profile_opposing_direction_evidence"
            if opposing > 0
            else "goal_profile_minimum_lateral_consensus_not_met"
        )
    elif not bool(result.get("valid")):
        result["profile_target_unobserved"] = True
    elif (
        raw_direction == 0
        and int(result.get("valid_frame_count", 0)) < minimum
    ):
        result["profile_center_evidence_suppressed"] = True
        result["raw_profile_reason"] = result.get("reason")
        result["reason"] = "goal_profile_center_evidence_below_minimum"

    result["profile_planner_fallback_authorized"] = bool(
        fallback
        and (
            result.get("profile_lateral_direction_suppressed")
            or result.get("profile_target_unobserved")
            or result.get("profile_center_evidence_suppressed")
        )
    )
    return result


def apply_profile_to_replay_direction(
    visual_bearing: dict[str, Any],
    profile: dict[str, Any],
) -> int:
    """Small pure helper used by deployment regression tests."""

    return int(apply_goal_visual_profile(visual_bearing, profile)["direction"])


__all__: Sequence[str] = (
    "PROFILE_SCHEMA",
    "apply_goal_visual_profile",
    "apply_profile_to_replay_direction",
    "load_goal_visual_profile",
    "validate_goal_visual_profile",
)
