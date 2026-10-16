"""Behavioral checks for no-reuse Go2 execution; no ROS or weights needed."""
from pathlib import Path
import ast
import sys
import math

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "deploy/v70_go2_orin"))
from fresh_plan_gate import FreshPlanGate


def test_accepted_action_cannot_execute_again_when_planner_abstains():
    gate = FreshPlanGate()
    action = [0.12, 0.01, 0.05]
    assert gate.consume(sequence=4, interlock=True, abstained=False, action=action).allowed
    assert not gate.consume(sequence=5, interlock=True, abstained=True, action=action).allowed
    assert not gate.consume(sequence=4, interlock=True, abstained=False, action=action).allowed
    assert gate.consume(sequence=6, interlock=True, abstained=False, action=action).allowed
    assert not gate.consume(sequence=5, interlock=True, abstained=False, action=action).allowed


def test_disarm_invalid_and_zero_actions_cannot_consume_or_start_motion():
    gate = FreshPlanGate()
    assert not gate.consume(sequence=0, interlock=False, abstained=False, action=[.1, 0, 0]).allowed
    assert not gate.consume(sequence=0, interlock=True, abstained=False, action=[math.nan, 0, 0]).allowed
    assert not gate.consume(sequence=0, interlock=True, abstained=False, action=[0, 0, 0]).allowed
    assert gate.last_consumed_sequence == -1
    assert gate.consume(sequence=0, interlock=True, abstained=False, action=[.1, 0, 0]).allowed


def test_live_node_has_no_alternate_action_recovery_start():
    # Integration contract: every main action start must be behind the consumed
    # plan gate. This also catches accidental reintroduction of a fallback path.
    text = (ROOT / "deploy/v70_go2_orin/v70_live_nav_fast.py").read_text(encoding="utf-8")
    tree = ast.parse(text)
    starts = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
              and isinstance(n.func, ast.Attribute) and n.func.attr == "start"
              and ast.unparse(n.func.value) == "self.follower"]
    assert len(starts) == 1
    disabled = {"_start_last_valid_action_replay", "_start_inflight_forward_bridge",
                "_start_abstention_forward_reposition", "_start_target_loss_forward_continuation"}
    assert not any(isinstance(n, ast.Attribute) and n.attr in disabled for n in ast.walk(tree))
    assert "self.fresh_plan_gate.consume(" in text
    assert 'execution_prefix = [np.asarray(result["selected_actions_h4"]' in text


def test_offline_smoke_uses_real_follower_and_never_imports_ros():
    import runpy
    module = runpy.run_path(str(ROOT / "robot/smoke_test.py"))
    assert module["main"]() == 0
    assert "rclpy" not in sys.modules
    assert "unitree_sdk2py" not in sys.modules


def test_abstention_report_is_standard_json():
    from robot.offline_plan import json_safe
    import json
    value = json_safe({"abstained": True, "archive_energy": math.inf,
                       "actions": [[0.0, 0.0, 0.0]]})
    encoded = json.dumps(value, allow_nan=False)
    assert json.loads(encoded)["archive_energy"] is None
