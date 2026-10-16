from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

DEPLOY_DIR = (
    Path(__file__).resolve().parents[2]
    / "deploy"
    / "v70_go2_orin"
)
sys.path.insert(0, str(DEPLOY_DIR))

from v70_closed_loop_core import (  # noqa: E402
    ActionFollower,
    ActionFollowerConfig,
    FollowerPhase,
    Pose2D,
    RouteManifest,
    RouteProgress,
)


def pose(
    t: float,
    *,
    x: float = 0.0,
    y: float = 0.0,
    yaw: float = 0.0,
) -> Pose2D:
    return Pose2D(receive_time=t, x=x, y=y, yaw=yaw)


def test_turn_drive_settle_completes_with_bounded_commands() -> None:
    follower = ActionFollower(
        ActionFollowerConfig(
            no_progress_timeout_s=5.0,
            settle_time_s=0.2,
            turn_settle_time_s=0.10,
        )
    )
    started = follower.start(
        (0.20, 0.04, math.atan2(0.04, 0.20)),
        pose(0.0),
        0.0,
    )
    assert started.phase is FollowerPhase.TURN
    command = follower.update(
        pose(0.1, yaw=0.10),
        0.1,
        interlock_allows_motion=True,
    )
    assert command.enable_motion
    assert abs(command.wz) <= 0.10

    target_yaw = math.atan2(0.04, 0.20)
    for index in range(3):
        command = follower.update(
            pose(0.2 + index * 0.05, yaw=target_yaw),
            0.2 + index * 0.05,
            interlock_allows_motion=True,
        )
    assert follower.phase is FollowerPhase.TURN_SETTLE
    assert not command.enable_motion
    # The current controller waits for the body to settle after turning.
    command = follower.update(pose(0.48, yaw=target_yaw), 0.48,
                              interlock_allows_motion=True)
    assert follower.phase is FollowerPhase.DRIVE

    command = follower.update(
        pose(
            0.5,
            x=0.10 * math.cos(target_yaw),
            y=0.10 * math.sin(target_yaw),
            yaw=target_yaw,
        ),
        0.5,
        interlock_allows_motion=True,
    )
    assert command.enable_motion
    assert 0.0 < command.vx <= 0.05

    distance = math.hypot(0.20, 0.04)
    command = follower.update(
        pose(
            1.0,
            x=distance * math.cos(target_yaw),
            y=distance * math.sin(target_yaw),
            yaw=target_yaw,
        ),
        1.0,
        interlock_allows_motion=True,
    )
    assert command.phase is FollowerPhase.SETTLE
    assert not command.enable_motion

    complete = follower.update(
        pose(
            1.3,
            x=distance * math.cos(target_yaw),
            y=distance * math.sin(target_yaw),
            yaw=target_yaw,
        ),
        1.3,
        interlock_allows_motion=True,
    )
    assert complete.complete
    assert complete.phase is FollowerPhase.COMPLETE


def test_interlock_closure_aborts_without_resuming() -> None:
    follower = ActionFollower(ActionFollowerConfig())
    follower.start((0.20, 0.0, 0.0), pose(0.0), 0.0)
    stopped = follower.update(
        pose(0.1),
        0.1,
        interlock_allows_motion=False,
    )
    assert stopped.phase is FollowerPhase.IDLE
    assert not stopped.enable_motion
    assert not follower.active


@pytest.mark.parametrize(
    ("action", "reason"),
    [
        ((-0.10, 0.0, 0.0), "reverse_action_not_supported"),
        ((0.50, 0.0, 0.0), "action_distance_exceeds"),
        ((0.10, 0.0, 1.0), "action_yaw_exceeds"),
        ((0.10, 0.10, 0.0), "heading_contract_mismatch"),
    ],
)
def test_executor_rejects_out_of_contract_actions(
    action,
    reason: str,
) -> None:
    follower = ActionFollower(ActionFollowerConfig())
    result = follower.start(action, pose(0.0), 0.0)
    assert result.fault
    assert reason in result.reason


def test_stale_odom_and_no_progress_fail_closed() -> None:
    follower = ActionFollower(
        ActionFollowerConfig(no_progress_timeout_s=0.2)
    )
    follower.start((0.20, 0.0, 0.0), pose(0.0), 0.0)
    stale = follower.update(
        pose(0.0),
        1.0,
        interlock_allows_motion=True,
    )
    assert stale.fault
    assert stale.reason == "stale_odom"

    follower = ActionFollower(
        ActionFollowerConfig(
            no_progress_timeout_s=0.2,
            max_odom_age_s=1.0,
        )
    )
    follower.start((0.20, 0.0, 0.0), pose(0.0), 0.0)
    blocked = follower.update(
        pose(0.3),
        0.3,
        interlock_allows_motion=True,
    )
    assert blocked.fault
    assert blocked.reason == "drive_no_progress"


def write_route(
    path: Path,
    goal_paths: list[Path],
    *,
    calibrated: bool,
) -> None:
    path.write_text(
        json.dumps(
            {
                "schema": "v70_image_goal_route_v1",
                "route_id": "test-route",
                "goals": [
                    {
                        "name": f"goal-{index}",
                        "image": str(goal),
                        "arrival_calibrated": calibrated,
                        "approach_temporal_distance": 5.0,
                        "temporal_distance_enter": 2.5,
                        "temporal_distance_exit": 3.5,
                        "cosine_distance_enter": 0.20,
                        "cosine_distance_exit": 0.28,
                        "required_confirmations": 3,
                    }
                    for index, goal in enumerate(goal_paths)
                ],
            }
        ),
        encoding="utf-8",
    )


def test_route_requires_calibration_only_for_live_mode(
    tmp_path: Path,
) -> None:
    goal = tmp_path / "goal.jpg"
    goal.write_bytes(b"image-placeholder")
    route = tmp_path / "route.json"
    write_route(route, [goal], calibrated=False)

    shadow = RouteManifest.load(
        route,
        require_calibrated=False,
    )
    assert shadow.route_id == "test-route"
    with pytest.raises(RuntimeError, match="not arrival-calibrated"):
        RouteManifest.load(route, require_calibrated=True)


def test_route_progress_advances_and_latches_final(
    tmp_path: Path,
) -> None:
    goals = [tmp_path / "a.jpg", tmp_path / "b.jpg"]
    for goal in goals:
        goal.write_bytes(b"image-placeholder")
    route = tmp_path / "route.json"
    write_route(route, goals, calibrated=True)

    progress = RouteProgress(
        RouteManifest.load(route, require_calibrated=True)
    )
    assert not progress.is_final
    assert progress.mark_current_reached()
    assert progress.index == 1
    assert progress.is_final
    assert not progress.mark_current_reached()
    assert progress.complete
