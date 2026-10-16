from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path

import pytest

DEPLOY_DIR = (
    Path(__file__).resolve().parents[2]
    / "deploy"
    / "v70_go2_orin"
)
sys.path.insert(0, str(DEPLOY_DIR))

from v70_board_latency_benchmark import (  # noqa: E402
    require_zero_motion_interlock,
    summarize_samples,
)
from v70_latency_protocol import (  # noqa: E402
    SCHEMA,
    StageTimer,
    read_latency,
    summarize_ms,
    write_json_atomic,
)


def test_stage_timer_rejects_duplicate_and_invalid_values() -> None:
    timer = StageTimer("unit")
    timer.add_ms("first_ms", 1.25)

    with pytest.raises(ValueError, match="duplicate"):
        timer.add_ms("first_ms", 2.0)
    with pytest.raises(ValueError, match="invalid"):
        timer.add_ms("negative_ms", -1.0)


def test_stage_timer_measure_and_atomic_roundtrip(tmp_path: Path) -> None:
    timer = StageTimer("unit")
    with timer.measure("work_ms"):
        sum(range(100))
    payload = timer.payload()
    output = tmp_path / "latency.json"
    write_json_atomic(output, payload)

    restored = read_latency(output, expected_stage="unit")
    assert restored["schema"] == SCHEMA
    assert restored["timings_ms"]["work_ms"] >= 0.0


def test_summarize_ms_uses_linear_percentiles() -> None:
    result = summarize_ms([0.0, 10.0, 20.0, 30.0])

    assert result["count"] == 4
    assert result["p50_ms"] == pytest.approx(15.0)
    assert result["p90_ms"] == pytest.approx(27.0)
    assert result["p95_ms"] == pytest.approx(28.5)


def test_zero_motion_interlock_is_fail_closed(tmp_path: Path) -> None:
    stop_file = tmp_path / "stop"
    arm_file = tmp_path / "armed"
    args = Namespace(stop_file=stop_file, arm_file=arm_file)

    with pytest.raises(RuntimeError, match="stop file is missing"):
        require_zero_motion_interlock(args)

    stop_file.touch()
    require_zero_motion_interlock(args)

    arm_file.touch()
    with pytest.raises(RuntimeError, match="arm file must be absent"):
        require_zero_motion_interlock(args)


def test_benchmark_summary_excludes_warmup() -> None:
    samples = [
        {
            "warmup": True,
            "dinov3_encode_ms": 1000.0,
            "complete_planning_cycle_wall_ms": 2000.0,
        },
        {
            "warmup": False,
            "dinov3_encode_ms": 10.0,
            "complete_planning_cycle_wall_ms": 20.0,
        },
        {
            "warmup": False,
            "dinov3_encode_ms": 30.0,
            "complete_planning_cycle_wall_ms": 40.0,
        },
    ]

    summary = summarize_samples(samples)
    assert summary["warmup_samples"] == 1
    assert summary["measured_samples"] == 2
    assert (
        summary["metrics_ms"]["dinov3_encode_ms"]["p50_ms"]
        == pytest.approx(20.0)
    )
