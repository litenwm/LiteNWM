"""No ROS, robot, model weights or motion API required."""
from __future__ import annotations
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "deploy/v70_go2_orin"))
from fresh_plan_gate import FreshPlanGate
from v70_closed_loop_core import ActionFollower, Pose2D, go2_live_action_follower_config


def main() -> int:
    gate = FreshPlanGate()
    follower = ActionFollower(go2_live_action_follower_config(max_odom_age_s=0.30))
    action = [0.12, 0.0, 0.0]
    auth = gate.consume(sequence=0, interlock=True, abstained=False, action=action)
    assert auth.allowed
    command = follower.start(action, Pose2D(0.0, 0.0, 0.0, 0.0), 0.0)
    assert not command.fault
    # A rejected new plan and a duplicate old plan cannot start another action.
    assert not gate.consume(sequence=1, interlock=True, abstained=True, action=action).allowed
    assert not gate.consume(sequence=0, interlock=True, abstained=False, action=action).allowed
    follower.reset("no_fresh_plan")
    command = follower.update(Pose2D(1.0, 0.12, 0.0, 0.0), 1.0,
                              interlock_allows_motion=True, motion_applied=False)
    assert command.vx == 0.0 and command.wz == 0.0
    print(json.dumps({"passed": True, "motion_api_imported": False,
                      "execution_policy": "fresh_h1_once", "waiting_command": [0, 0]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
