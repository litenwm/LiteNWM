from __future__ import annotations

import sys
from pathlib import Path

import pytest

DEPLOY_DIR = (
    Path(__file__).resolve().parents[2]
    / "deploy"
    / "v70_go2_orin"
)
sys.path.insert(0, str(DEPLOY_DIR))

from v70_goal_stop import (  # noqa: E402
    GoalEvidence,
    GoalPhase,
    GoalStopConfig,
    GoalStopSupervisor,
)


def config(**overrides) -> GoalStopConfig:
    values = {
        "approach_temporal_distance": 5.0,
        "temporal_distance_enter": 2.5,
        "temporal_distance_exit": 3.5,
        "cosine_distance_enter": 0.20,
        "cosine_distance_exit": 0.28,
        "required_confirmations": 3,
        "max_evidence_age_s": 0.5,
    }
    values.update(overrides)
    return GoalStopConfig(**values)


def evidence(**overrides) -> GoalEvidence:
    values = {
        "age_s": 0.05,
        "temporal_distance": 2.0,
        "cosine_distance": 0.15,
        "final_topological_node": True,
    }
    values.update(overrides)
    return GoalEvidence(**values)


def test_requires_final_topological_node() -> None:
    supervisor = GoalStopSupervisor(config())
    result = supervisor.update(
        evidence(final_topological_node=False)
    )
    assert result.phase is GoalPhase.NAVIGATE
    assert not result.goal_reached
    assert not result.command_zero
    assert result.confirmations == 0


def test_requires_consecutive_confirmation_and_latches_stop() -> None:
    supervisor = GoalStopSupervisor(config())
    first = supervisor.update(evidence())
    assert first.confirmations == 1
    assert first.phase is GoalPhase.GOAL_CANDIDATE
    assert first.command_zero
    assert not first.slow_approach
    second = supervisor.update(evidence())
    assert second.confirmations == 2
    assert second.command_zero
    reached = supervisor.update(evidence())
    assert reached.phase is GoalPhase.STOPPED
    assert reached.goal_reached
    assert reached.command_zero

    latched = supervisor.update(
        evidence(temporal_distance=20.0, cosine_distance=1.0)
    )
    assert latched.phase is GoalPhase.STOPPED
    assert latched.reason == "goal_stop_latched"


def test_failed_enter_resets_consecutive_counter_but_keeps_hysteresis() -> None:
    supervisor = GoalStopSupervisor(config())
    assert supervisor.update(evidence()).confirmations == 1
    result = supervisor.update(
        evidence(temporal_distance=3.0, cosine_distance=0.24)
    )
    assert result.phase is GoalPhase.GOAL_CANDIDATE
    assert result.reason == "goal_candidate_hysteresis"
    assert result.confirmations == 0
    assert result.command_zero


@pytest.mark.parametrize(
    ("sample", "reason"),
    [
        (evidence(sensor_valid=False), "sensor_invalid"),
        (evidence(age_s=0.8), "evidence_stale"),
        (evidence(safety_hold=True), "independent_safety_hold"),
        (
            evidence(temporal_distance=float("nan")),
            "evidence_nonfinite_or_missing",
        ),
    ],
)
def test_invalid_or_safety_evidence_is_zero_hold_not_success(
    sample: GoalEvidence,
    reason: str,
) -> None:
    supervisor = GoalStopSupervisor(config())
    result = supervisor.update(sample)
    assert result.phase is GoalPhase.SAFETY_HOLD
    assert result.command_zero
    assert not result.goal_reached
    assert result.reason == reason


def test_optional_odom_gate_is_required_when_configured() -> None:
    supervisor = GoalStopSupervisor(
        config(
            odom_distance_enter_m=0.4,
            odom_distance_exit_m=0.6,
        )
    )
    missing = supervisor.update(evidence())
    assert missing.phase is GoalPhase.SAFETY_HOLD
    assert not missing.goal_reached

    supervisor.reset()
    far = supervisor.update(evidence(odom_distance_m=0.5))
    assert far.phase is GoalPhase.APPROACH
    assert far.confirmations == 0


def test_config_rejects_missing_hysteresis_or_single_frame_stop() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        config(required_confirmations=1).validate()
    with pytest.raises(ValueError, match="exit"):
        config(
            temporal_distance_enter=3.0,
            temporal_distance_exit=2.0,
        ).validate()
