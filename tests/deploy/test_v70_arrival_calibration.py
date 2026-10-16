from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pytest

DEPLOY_DIR = (
    Path(__file__).resolve().parents[2]
    / "deploy"
    / "v70_go2_orin"
)
sys.path.insert(0, str(DEPLOY_DIR))

from v70_arrival_calibrate import calibrate  # noqa: E402
from v70_arrival_collect import append_unique, build_record  # noqa: E402


def make_sample(
    label: str,
    index: int,
    *,
    goal_sha256: str,
    temporal: float,
    cosine: float,
) -> dict:
    return {
        "schema": "v70_arrival_sample_v1",
        "operator_label": label,
        "operator_confirmed_standing_stationary": True,
        "goal_sha256": goal_sha256,
        "observation_sha256": [
            hashlib.sha256(f"{label}-{index}-{frame}".encode()).hexdigest()
            for frame in range(4)
        ],
        "nomad_temporal_distance": temporal,
        "dinov3_current_goal_cosine_distance": cosine,
    }


def make_dataset(goal_sha256: str) -> list[dict]:
    samples = []
    bases = {
        "far": (9.0, 0.70),
        "approach": (3.0, 0.25),
        "stop": (1.0, 0.10),
        "overshoot": (1.4, 0.14),
        "lookalike": (1.0, 0.50),
    }
    for label, (temporal, cosine) in bases.items():
        for index in range(5):
            samples.append(
                make_sample(
                    label,
                    index,
                    goal_sha256=goal_sha256,
                    temporal=temporal + index * 0.01,
                    cosine=cosine + index * 0.001,
                )
            )
    return samples


def route_for(goal: Path, *, calibrated: bool = True) -> dict:
    return {
        "schema": "v70_image_goal_route_v1",
        "route_id": "single-goal-test",
        "goals": [
            {
                "name": "goal",
                "image": str(goal),
                "arrival_calibrated": calibrated,
                "approach_temporal_distance": 5.0,
                "temporal_distance_enter": 2.5,
                "temporal_distance_exit": 3.5,
                "cosine_distance_enter": 0.20,
                "cosine_distance_exit": 0.28,
                "required_confirmations": 3,
            }
        ],
    }


def test_candidate_requires_balanced_separation_and_never_promotes(
    tmp_path: Path,
) -> None:
    goal = tmp_path / "goal.jpg"
    goal.write_bytes(b"goal-image")
    goal_sha256 = hashlib.sha256(goal.read_bytes()).hexdigest()
    report, candidate = calibrate(
        make_dataset(goal_sha256),
        route_for(goal),
        goal,
        min_per_label=5,
    )
    assert report["ready_for_manual_review"] is True
    assert report["arrival_calibrated_was_not_promoted"] is True
    assert candidate is not None
    assert candidate["goals"][0]["arrival_calibrated"] is False
    assert candidate["goals"][0]["required_confirmations"] == 4
    assert candidate["calibration_candidate"][
        "held_out_validation_required"
    ] is True


def test_lookalike_false_positive_rejects_candidate(
    tmp_path: Path,
) -> None:
    goal = tmp_path / "goal.jpg"
    goal.write_bytes(b"goal-image")
    goal_sha256 = hashlib.sha256(goal.read_bytes()).hexdigest()
    samples = make_dataset(goal_sha256)
    lookalike = next(
        sample
        for sample in samples
        if sample["operator_label"] == "lookalike"
    )
    lookalike["dinov3_current_goal_cosine_distance"] = 0.09
    report, candidate = calibrate(
        samples,
        route_for(goal),
        goal,
        min_per_label=5,
    )
    assert report["ready_for_manual_review"] is False
    assert candidate is None
    assert any(
        reason.startswith("dangerous_enter_false_positives")
        for reason in report["rejection_reasons"]
    )


def test_collector_requires_successful_shadow_and_prevents_duplicate(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    hashes = [hashlib.sha256(str(i).encode()).hexdigest() for i in range(4)]
    goal_hash = hashlib.sha256(b"goal").hexdigest()
    status = {
        "mode": "shadow",
        "fault": None,
        "exit_code": 0,
        "stop_present": True,
        "arm_present": False,
        "route_id": "route",
        "route_goal": "goal",
        "last_goal_decision": {"phase": "NAVIGATE"},
        "last_plan": {
            "sequence": 0,
            "goal_sha256": goal_hash,
            "nomad_temporal_distance": 3.0,
            "dinov3_current_goal_cosine_distance": 0.2,
            "selected_first_action": [0.2, 0.0, 0.0],
            "abstained": False,
            "timings_ms": {"persistent_total": 8.0},
        },
    }
    (run_dir / "status.json").write_text(json.dumps(status))
    events = [
        {
            "event": "plan_requested",
            "sequence": 0,
            "image_sha256": hashes,
        },
        {
            "event": "plan_complete",
            "summary": {"sequence": 0},
        },
    ]
    (run_dir / "live_nav.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in events)
    )
    args = argparse.Namespace(
        run_dir=run_dir,
        label="approach",
        dataset=tmp_path / "dataset.jsonl",
        confirm_standing_stationary=True,
        note="test",
    )
    record = build_record(args)
    append_unique(args.dataset, record)
    with pytest.raises(RuntimeError, match="duplicate"):
        append_unique(args.dataset, record)
