from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from deploy.v70_go2_orin.v70_runtime_contract import (
    SNAPSHOT_CAPTURE_NS_KEY,
    SNAPSHOT_GOAL_SHA256_KEY,
    SNAPSHOT_RUN_ID_KEY,
    motion_interlock_allows_nonzero,
    require_domain_calibration,
    require_same_snapshot,
    select_executable_action,
    zero_actions_if_abstained,
)


def tri_result(*, sacson_strict: bool = False) -> dict:
    return {
        "schema": "proposal_aware_v70_tridomain_training_result_v1",
        "status": "frozen_after_v69_spent_failure_before_v70_spent_replay",
        "seal_status": "RELAXED_FEASIBLE_V1",
        "relaxed_all_domains_feasible_v1": True,
        "relaxed_domain_feasible_v1": {
            "recon": True,
            "scand": True,
            "sacson": True,
        },
        "model_domain_index": {"recon": 0, "scand": 1, "sacson": 2},
        "calibration": {
            domain: {
                "feasible": domain != "sacson" or sacson_strict,
                "strict_feasible": domain != "sacson" or sacson_strict,
                "relaxed_feasible_v1": True,
                "policy": {},
                "dual_temperature": 1.0,
                "harm_temperature": 1.0,
            }
            for domain in ("recon", "scand", "sacson")
        },
    }


def test_strict_and_explicit_relaxed_calibration_contracts() -> None:
    recon = require_domain_calibration(tri_result(), "recon")
    assert recon["domain_id"] == 0
    assert recon["spacing_m"] == pytest.approx(0.25)
    assert recon["calibration_contract"] == "strict"

    with pytest.raises(RuntimeError, match="not strictly feasible"):
        require_domain_calibration(tri_result(), "sacson")

    sacson = require_domain_calibration(
        tri_result(),
        "sacson",
        allow_relaxed=True,
    )
    assert sacson["domain_id"] == 2
    assert sacson["spacing_m"] == pytest.approx(0.255)
    assert sacson["calibration_contract"] == "relaxed_v1"


def test_relaxed_calibration_fails_closed_if_any_seal_field_is_missing() -> None:
    result = tri_result()
    result["relaxed_all_domains_feasible_v1"] = False
    with pytest.raises(RuntimeError, match="not strictly feasible"):
        require_domain_calibration(result, "sacson", allow_relaxed=True)


def test_no_safe_challenger_produces_explicit_zero_action() -> None:
    energy = np.asarray([[0.0, np.inf, 0.3]], dtype=np.float32)
    safe = np.asarray([[True, False, False]])
    actions = np.ones((1, 3, 4, 3), dtype=np.float32)
    selection = select_executable_action(energy, safe, actions)

    assert selection["selected_index"] == 0
    assert selection["abstained"] is True
    assert selection["reason"] == "no_finite_safe_challenger"
    assert np.array_equal(
        selection["selected_actions_h4"],
        np.zeros((4, 3), dtype=np.float32),
    )
    assert np.array_equal(
        selection["selected_first_action"],
        np.zeros(3, dtype=np.float32),
    )


def test_candidate_zero_is_never_selected_as_motion() -> None:
    energy = np.asarray([[-100.0, 3.0, 1.0]], dtype=np.float32)
    safe = np.asarray([[True, True, True]])
    actions = np.zeros((1, 3, 4, 3), dtype=np.float32)
    actions[0, 0] = 9.0
    actions[0, 1] = 1.0
    actions[0, 2] = 2.0

    selection = select_executable_action(energy, safe, actions)
    assert selection["selected_index"] == 2
    assert selection["abstained"] is False
    assert np.array_equal(
        selection["selected_actions_h4"],
        np.full((4, 3), 2.0, dtype=np.float32),
    )


def test_archived_abstention_is_zeroed() -> None:
    actions = np.full((4, 3), 7.0, dtype=np.float32)
    assert np.array_equal(
        zero_actions_if_abstained(actions, abstained=True),
        np.zeros((4, 3), dtype=np.float32),
    )


def artifact(run: str, goal: str, capture_ns: int) -> dict[str, np.ndarray]:
    return {
        SNAPSHOT_RUN_ID_KEY: np.asarray(run),
        SNAPSHOT_GOAL_SHA256_KEY: np.asarray(goal),
        SNAPSHOT_CAPTURE_NS_KEY: np.asarray(capture_ns, dtype=np.int64),
    }


def test_snapshot_contract_accepts_one_coherent_capture() -> None:
    value = require_same_snapshot(
        [
            artifact("run-1", "abc", 10_000_000_000),
            artifact("run-1", "abc", 10_200_000_000),
        ],
        max_capture_skew_s=0.25,
    )
    assert value["snapshot_run_id"] == "run-1"
    assert value["capture_skew_s"] == pytest.approx(0.2)


@pytest.mark.parametrize(
    "second,match",
    [
        (artifact("run-2", "abc", 10_000_000_000), "different snapshot"),
        (artifact("run-1", "def", 10_000_000_000), "different goals"),
        (artifact("run-1", "abc", 11_000_000_000), "allowed skew"),
    ],
)
def test_snapshot_contract_rejects_mixed_artifacts(
    second: dict[str, np.ndarray],
    match: str,
) -> None:
    with pytest.raises(RuntimeError, match=match):
        require_same_snapshot(
            [artifact("run-1", "abc", 10_000_000_000), second],
            max_capture_skew_s=0.25,
        )


def test_motion_interlock_is_fail_closed(tmp_path: Path) -> None:
    stop = tmp_path / "stop"
    arm = tmp_path / "arm"
    assert not motion_interlock_allows_nonzero(
        stop_file=stop,
        arm_file=arm,
    )
    arm.touch()
    assert motion_interlock_allows_nonzero(
        stop_file=stop,
        arm_file=arm,
    )
    stop.touch()
    assert not motion_interlock_allows_nonzero(
        stop_file=stop,
        arm_file=arm,
    )
