#!/usr/bin/env python3
"""Pure fail-closed goal-arrival supervisor for V70 navigation.

This module does not import ROS or any Unitree SDK.  It converts synchronized
navigation evidence into a latched state decision.  The thresholds are
deployment parameters and must be calibrated from real approach sequences.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any


class GoalPhase(str, Enum):
    NAVIGATE = "NAVIGATE"
    APPROACH = "APPROACH"
    GOAL_CANDIDATE = "GOAL_CANDIDATE"
    SAFETY_HOLD = "SAFETY_HOLD"
    STOPPED = "STOPPED"


@dataclass(frozen=True)
class GoalStopConfig:
    approach_temporal_distance: float
    temporal_distance_enter: float
    temporal_distance_exit: float
    cosine_distance_enter: float
    cosine_distance_exit: float
    required_confirmations: int = 4
    max_evidence_age_s: float = 0.50
    require_final_topological_node: bool = True
    odom_distance_enter_m: float | None = None
    odom_distance_exit_m: float | None = None
    yaw_error_enter_rad: float | None = None
    yaw_error_exit_rad: float | None = None

    def validate(self) -> None:
        finite_positive = {
            "approach_temporal_distance": self.approach_temporal_distance,
            "temporal_distance_enter": self.temporal_distance_enter,
            "temporal_distance_exit": self.temporal_distance_exit,
            "cosine_distance_enter": self.cosine_distance_enter,
            "cosine_distance_exit": self.cosine_distance_exit,
            "max_evidence_age_s": self.max_evidence_age_s,
        }
        for name, value in finite_positive.items():
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.temporal_distance_exit < self.temporal_distance_enter:
            raise ValueError("temporal-distance exit must be >= enter")
        if self.cosine_distance_exit < self.cosine_distance_enter:
            raise ValueError("cosine-distance exit must be >= enter")
        if self.approach_temporal_distance < self.temporal_distance_exit:
            raise ValueError(
                "approach threshold must be >= temporal-distance exit"
            )
        if self.required_confirmations < 2:
            raise ValueError("required_confirmations must be at least 2")
        self._validate_optional_pair(
            "odom_distance",
            self.odom_distance_enter_m,
            self.odom_distance_exit_m,
        )
        self._validate_optional_pair(
            "yaw_error",
            self.yaw_error_enter_rad,
            self.yaw_error_exit_rad,
        )

    @staticmethod
    def _validate_optional_pair(
        name: str,
        enter: float | None,
        exit_value: float | None,
    ) -> None:
        if (enter is None) != (exit_value is None):
            raise ValueError(f"{name} enter/exit must be set together")
        if enter is None:
            return
        if (
            not math.isfinite(float(enter))
            or not math.isfinite(float(exit_value))
            or float(enter) < 0.0
            or float(exit_value) < float(enter)
        ):
            raise ValueError(f"invalid {name} enter/exit thresholds")


@dataclass(frozen=True)
class GoalEvidence:
    age_s: float
    temporal_distance: float
    cosine_distance: float
    final_topological_node: bool
    sensor_valid: bool = True
    safety_hold: bool = False
    odom_distance_m: float | None = None
    yaw_error_rad: float | None = None


@dataclass(frozen=True)
class GoalDecision:
    phase: GoalPhase
    goal_reached: bool
    command_zero: bool
    slow_approach: bool
    confirmations: int
    reason: str

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["phase"] = self.phase.value
        return value


class GoalStopSupervisor:
    """Require multi-signal, consecutive confirmation and latch zero at goal."""

    def __init__(self, config: GoalStopConfig) -> None:
        config.validate()
        self.config = config
        self.phase = GoalPhase.NAVIGATE
        self.confirmations = 0

    def reset(self) -> None:
        self.phase = GoalPhase.NAVIGATE
        self.confirmations = 0

    @staticmethod
    def _finite(value: float | None) -> bool:
        return value is not None and math.isfinite(float(value))

    def _evidence_is_numeric(self, evidence: GoalEvidence) -> bool:
        required = (
            evidence.age_s,
            evidence.temporal_distance,
            evidence.cosine_distance,
        )
        if not all(math.isfinite(float(value)) for value in required):
            return False
        if self.config.odom_distance_enter_m is not None and not self._finite(
            evidence.odom_distance_m
        ):
            return False
        if self.config.yaw_error_enter_rad is not None and not self._finite(
            evidence.yaw_error_rad
        ):
            return False
        return True

    def _within(
        self,
        evidence: GoalEvidence,
        *,
        use_exit: bool,
    ) -> bool:
        temporal_limit = (
            self.config.temporal_distance_exit
            if use_exit
            else self.config.temporal_distance_enter
        )
        cosine_limit = (
            self.config.cosine_distance_exit
            if use_exit
            else self.config.cosine_distance_enter
        )
        passed = (
            evidence.temporal_distance <= temporal_limit
            and evidence.cosine_distance <= cosine_limit
        )
        if self.config.odom_distance_enter_m is not None:
            odom_limit = (
                self.config.odom_distance_exit_m
                if use_exit
                else self.config.odom_distance_enter_m
            )
            passed = passed and float(evidence.odom_distance_m) <= float(
                odom_limit
            )
        if self.config.yaw_error_enter_rad is not None:
            yaw_limit = (
                self.config.yaw_error_exit_rad
                if use_exit
                else self.config.yaw_error_enter_rad
            )
            passed = passed and abs(float(evidence.yaw_error_rad)) <= float(
                yaw_limit
            )
        return bool(passed)

    def _decision(self, reason: str) -> GoalDecision:
        reached = self.phase is GoalPhase.STOPPED
        zero_hold = self.phase in {
            GoalPhase.GOAL_CANDIDATE,
            GoalPhase.SAFETY_HOLD,
        }
        return GoalDecision(
            phase=self.phase,
            goal_reached=reached,
            command_zero=reached or zero_hold,
            slow_approach=self.phase is GoalPhase.APPROACH,
            confirmations=self.confirmations,
            reason=reason,
        )

    def update(self, evidence: GoalEvidence) -> GoalDecision:
        if self.phase is GoalPhase.STOPPED:
            return self._decision("goal_stop_latched")

        if not evidence.sensor_valid:
            self.phase = GoalPhase.SAFETY_HOLD
            self.confirmations = 0
            return self._decision("sensor_invalid")
        if not self._evidence_is_numeric(evidence):
            self.phase = GoalPhase.SAFETY_HOLD
            self.confirmations = 0
            return self._decision("evidence_nonfinite_or_missing")
        if (
            evidence.age_s < 0.0
            or evidence.age_s > self.config.max_evidence_age_s
        ):
            self.phase = GoalPhase.SAFETY_HOLD
            self.confirmations = 0
            return self._decision("evidence_stale")
        if evidence.safety_hold:
            self.phase = GoalPhase.SAFETY_HOLD
            self.confirmations = 0
            return self._decision("independent_safety_hold")

        if (
            self.config.require_final_topological_node
            and not evidence.final_topological_node
        ):
            self.phase = GoalPhase.NAVIGATE
            self.confirmations = 0
            return self._decision("not_final_topological_node")

        if self._within(evidence, use_exit=False):
            self.confirmations += 1
            if self.confirmations >= self.config.required_confirmations:
                self.phase = GoalPhase.STOPPED
                return self._decision("goal_confirmed")
            self.phase = GoalPhase.GOAL_CANDIDATE
            return self._decision("goal_candidate_confirming")

        self.confirmations = 0
        if (
            self.phase is GoalPhase.GOAL_CANDIDATE
            and self._within(evidence, use_exit=True)
        ):
            return self._decision("goal_candidate_hysteresis")
        if (
            evidence.temporal_distance
            <= self.config.approach_temporal_distance
        ):
            self.phase = GoalPhase.APPROACH
            return self._decision("approach_final_goal")

        self.phase = GoalPhase.NAVIGATE
        return self._decision("goal_not_close")


__all__ = [
    "GoalDecision",
    "GoalEvidence",
    "GoalPhase",
    "GoalStopConfig",
    "GoalStopSupervisor",
]
