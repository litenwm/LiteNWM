#!/usr/bin/env python3
"""Fail-closed runtime contracts for the tri-domain V70 shadow pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


TRIDOMAIN_RESULT_SCHEMA = "proposal_aware_v70_tridomain_training_result_v1"
TRIDOMAIN_FROZEN_STATUS = (
    "frozen_after_v69_spent_failure_before_v70_spent_replay"
)
RELAXED_SEAL_STATUS = "RELAXED_FEASIBLE_V1"

DOMAIN_INDEX = {
    "recon": 0,
    "scand": 1,
    "sacson": 2,
}
DOMAIN_SPACING_M = {
    "recon": 0.25,
    "scand": 0.38,
    "sacson": 0.255,
}

SNAPSHOT_RUN_ID_KEY = "snapshot_run_id"
SNAPSHOT_GOAL_SHA256_KEY = "snapshot_goal_sha256"
SNAPSHOT_CAPTURE_NS_KEY = "snapshot_capture_unix_ns"


def require_domain_calibration(
    result: Mapping[str, Any],
    domain: str,
    *,
    allow_relaxed: bool = False,
) -> dict[str, Any]:
    """Validate the frozen tri-domain calibration and return runtime fields."""

    if result.get("schema") != TRIDOMAIN_RESULT_SCHEMA:
        raise RuntimeError(
            "unexpected selector result schema: "
            f"{result.get('schema')!r}"
        )
    if result.get("status") != TRIDOMAIN_FROZEN_STATUS:
        raise RuntimeError("tri-domain selector is not in its frozen state")
    if domain not in DOMAIN_INDEX:
        raise RuntimeError(f"unsupported runtime domain: {domain!r}")

    model_domain_index = result.get("model_domain_index")
    if not isinstance(model_domain_index, Mapping):
        raise RuntimeError("selector result missing model_domain_index")
    actual_index = model_domain_index.get(domain)
    expected_index = DOMAIN_INDEX[domain]
    if actual_index != expected_index:
        raise RuntimeError(
            f"{domain} domain index mismatch: "
            f"expected {expected_index}, got {actual_index!r}"
        )

    calibrations = result.get("calibration")
    if not isinstance(calibrations, Mapping) or domain not in calibrations:
        raise RuntimeError(f"selector result missing {domain} calibration")
    calibration = calibrations[domain]
    if not isinstance(calibration, Mapping):
        raise RuntimeError(f"invalid {domain} calibration record")

    strict_feasible = bool(
        calibration.get(
            "strict_feasible",
            calibration.get("feasible", False),
        )
    )
    contract = "strict"
    if not strict_feasible:
        relaxed_domain = result.get("relaxed_domain_feasible_v1")
        relaxed_ok = (
            allow_relaxed
            and result.get("seal_status") == RELAXED_SEAL_STATUS
            and bool(result.get("relaxed_all_domains_feasible_v1"))
            and isinstance(relaxed_domain, Mapping)
            and bool(relaxed_domain.get(domain))
            and bool(calibration.get("relaxed_feasible_v1"))
        )
        if not relaxed_ok:
            raise RuntimeError(
                f"{domain} calibration is not strictly feasible and the "
                "complete RELAXED_FEASIBLE_V1 contract was not accepted"
            )
        contract = "relaxed_v1"

    for key in ("policy", "dual_temperature", "harm_temperature"):
        if key not in calibration:
            raise RuntimeError(
                f"{domain} calibration missing required field {key!r}"
            )

    return {
        "domain": domain,
        "domain_id": expected_index,
        "spacing_m": DOMAIN_SPACING_M[domain],
        "calibration_contract": contract,
        "calibration": dict(calibration),
    }


def select_executable_action(
    energy: np.ndarray,
    safe: np.ndarray,
    actions_h4: np.ndarray,
) -> dict[str, Any]:
    """Select only a finite safe challenger; otherwise return explicit zeros.

    Candidate zero is an inference baseline/incumbent, not an executable
    fallback. The returned action is therefore zero whenever no candidate in
    indices 1..K-1 passes every gate.
    """

    energy_array = np.asarray(energy)
    safe_array = np.asarray(safe, dtype=bool)
    actions_array = np.asarray(actions_h4, dtype=np.float32)

    if energy_array.ndim == 2:
        if energy_array.shape[0] != 1:
            raise ValueError("only a single live state is supported")
        energy_array = energy_array[0]
    if safe_array.ndim == 2:
        if safe_array.shape[0] != 1:
            raise ValueError("only a single live state is supported")
        safe_array = safe_array[0]
    if actions_array.ndim == 4:
        if actions_array.shape[0] != 1:
            raise ValueError("only a single live state is supported")
        actions_array = actions_array[0]

    if energy_array.ndim != 1 or safe_array.ndim != 1:
        raise ValueError("energy and safe must be one-dimensional")
    if actions_array.ndim != 3 or actions_array.shape[-1] != 3:
        raise ValueError("actions_h4 must have shape (K, H, 3)")
    if (
        energy_array.shape != safe_array.shape
        or energy_array.shape[0] != actions_array.shape[0]
    ):
        raise ValueError("candidate dimensions do not match")
    if energy_array.shape[0] < 2:
        raise ValueError("at least one challenger is required")
    if not np.isfinite(actions_array).all():
        raise ValueError("actions_h4 contains NaN/Inf")

    eligible = safe_array & np.isfinite(energy_array)
    eligible[0] = False
    challenger_indices = np.flatnonzero(eligible)

    if challenger_indices.size == 0:
        selected_index = 0
        abstained = True
        reason = "no_finite_safe_challenger"
        selected_actions = np.zeros_like(actions_array[0])
    else:
        local = int(np.argmin(energy_array[challenger_indices]))
        selected_index = int(challenger_indices[local])
        abstained = False
        reason = "finite_safe_challenger"
        selected_actions = actions_array[selected_index].copy()

    return {
        "selected_index": selected_index,
        "abstained": abstained,
        "reason": reason,
        "selected_actions_h4": selected_actions,
        "selected_first_action": selected_actions[0].copy(),
    }


def zero_actions_if_abstained(
    selected_actions_h4: np.ndarray,
    *,
    abstained: bool,
) -> np.ndarray:
    """Apply the final fail-closed action gate to an archived CEM result."""

    actions = np.asarray(selected_actions_h4, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[-1] != 3:
        raise ValueError("selected_actions_h4 must have shape (H, 3)")
    if not np.isfinite(actions).all():
        raise ValueError("selected_actions_h4 contains NaN/Inf")
    return np.zeros_like(actions) if abstained else actions.copy()


def _scalar_text(value: np.ndarray, key: str) -> str:
    array = np.asarray(value)
    if array.size != 1:
        raise RuntimeError(f"{key} must be a scalar")
    text = str(array.reshape(-1)[0])
    if not text:
        raise RuntimeError(f"{key} must be non-empty")
    return text


def _scalar_int(value: np.ndarray, key: str) -> int:
    array = np.asarray(value)
    if array.size != 1:
        raise RuntimeError(f"{key} must be a scalar")
    try:
        return int(array.reshape(-1)[0])
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{key} must be an integer") from exc


def require_same_snapshot(
    artifacts: Sequence[Mapping[str, np.ndarray]],
    *,
    max_capture_skew_s: float = 0.5,
) -> dict[str, Any]:
    """Require all stage artifacts to refer to one goal and one capture run."""

    if not artifacts:
        raise RuntimeError("no artifacts were supplied")
    if max_capture_skew_s < 0.0:
        raise ValueError("max_capture_skew_s must be non-negative")

    run_ids: list[str] = []
    goal_hashes: list[str] = []
    capture_ns: list[int] = []
    for artifact in artifacts:
        missing = {
            SNAPSHOT_RUN_ID_KEY,
            SNAPSHOT_GOAL_SHA256_KEY,
            SNAPSHOT_CAPTURE_NS_KEY,
        }.difference(artifact)
        if missing:
            raise RuntimeError(
                f"artifact missing snapshot metadata: {sorted(missing)}"
            )
        run_ids.append(
            _scalar_text(artifact[SNAPSHOT_RUN_ID_KEY], SNAPSHOT_RUN_ID_KEY)
        )
        goal_hashes.append(
            _scalar_text(
                artifact[SNAPSHOT_GOAL_SHA256_KEY],
                SNAPSHOT_GOAL_SHA256_KEY,
            )
        )
        capture_ns.append(
            _scalar_int(
                artifact[SNAPSHOT_CAPTURE_NS_KEY],
                SNAPSHOT_CAPTURE_NS_KEY,
            )
        )

    if len(set(run_ids)) != 1:
        raise RuntimeError("artifacts were produced by different snapshot runs")
    if len(set(goal_hashes)) != 1:
        raise RuntimeError("artifacts were produced for different goals")

    skew_ns = max(capture_ns) - min(capture_ns)
    if skew_ns > int(max_capture_skew_s * 1_000_000_000):
        raise RuntimeError(
            "artifact capture timestamps exceed allowed skew: "
            f"{skew_ns / 1_000_000_000:.6f}s"
        )

    return {
        "snapshot_run_id": run_ids[0],
        "snapshot_goal_sha256": goal_hashes[0],
        "capture_skew_s": skew_ns / 1_000_000_000,
    }


def motion_interlock_allows_nonzero(
    *,
    stop_file: Path,
    arm_file: Path,
) -> bool:
    """Return true only when stop is absent and the explicit arm file exists."""

    return not stop_file.exists() and arm_file.is_file()
