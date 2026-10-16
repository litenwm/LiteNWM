#!/usr/bin/env python3
"""Diagnose v66 scorer-cal failure without touching A--H or reserve."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from tools.evaluation.audit_cem_proposal_shift_v65 import (  # noqa: E402
    binary_summary,
    within_state_rank,
)
from tools.planning.run_cem_safe_elite_v55 import (  # noqa: E402
    candidate_safety_mask,
    select_lowest_energy_safe,
)
from tools.scoring import run_train_dualdomain_cem_tail_v66_frozen as frozen  # noqa: E402
from tools.scoring.run_train_dualdomain_cem_tail_v66_candidate_level import (  # noqa: E402
    candidate_energy,
)
from tools.scoring.train_universal_dualdomain_raw_v2 import (  # noqa: E402
    ensemble_predict,
    policy_grid,
    safety_metrics,
    sigmoid,
)


implementation = frozen.implementation
OUTPUT = (
    ROOT
    / "outputs/cem_development_v1/dualdomain_cem_tail_v66/"
    "scorer_cal_failure_diagnosis.json"
)
EXTENDED_TEMPERATURES = (5.0, 8.0, 12.0, 16.0, 24.0, 32.0)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def labels(targets: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    delta = np.asarray(targets, dtype=np.float32)[..., :4]
    dual = (delta[..., 0] < -1.0e-7) & (delta[..., 3] < -1.0e-7)
    harm = (delta[..., 0] > 1.0e-7) | (delta[..., 3] > 1.0e-7)
    cost = (
        delta[..., 0]
        + 0.25 * delta[..., 1]
        + 0.25 * np.maximum(delta[..., 2], 0.0)
        + 0.50 * delta[..., 3]
        + 0.10 * harm
    )
    return -cost.astype(np.float32), dual, harm


def selected_summary(
    selected: np.ndarray,
    targets: np.ndarray,
    p_dual: np.ndarray,
    p_harm: np.ndarray,
) -> dict[str, Any]:
    metrics = safety_metrics(selected, targets)
    promoted = selected != 0
    row = np.flatnonzero(promoted)
    column = selected[promoted]
    metrics["mean_selected_p_dual"] = (
        float(p_dual[row, column].mean()) if len(row) else None
    )
    metrics["mean_selected_p_harm"] = (
        float(p_harm[row, column].mean()) if len(row) else None
    )
    return metrics


def set_audits(
    prediction: dict[str, np.ndarray],
    p_dual: np.ndarray,
    p_harm: np.ndarray,
    valid: np.ndarray,
    targets: np.ndarray,
    policy: dict[str, float],
    states: int,
) -> list[dict[str, Any]]:
    energy = candidate_energy(prediction, p_dual, p_harm, valid, policy)
    safe = candidate_safety_mask(prediction, p_dual, p_harm, valid, policy)
    true_score, dual, harm = labels(targets)
    names = ("fixed", "cem_iteration_0", "cem_iteration_1", "cem_iteration_2")
    result = []
    for index, name in enumerate(names):
        start = index * states
        stop = start + states
        local_valid = valid[start:stop].copy()
        local_valid[:, 0] = False
        local_safe = safe[start:stop]
        valid_row, valid_column = np.nonzero(local_valid)
        safe_row, safe_column = np.nonzero(local_safe)
        safe_row_global = safe_row + start
        selected = select_lowest_energy_safe(
            energy[start:stop], local_safe
        )
        result.append({
            "set_type": name,
            "all_valid": binary_summary(
                p_dual[start:stop][valid_row, valid_column],
                p_harm[start:stop][valid_row, valid_column],
                dual[start:stop][valid_row, valid_column],
                harm[start:stop][valid_row, valid_column],
            ),
            "all_safe": binary_summary(
                p_dual[safe_row_global, safe_column],
                p_harm[safe_row_global, safe_column],
                dual[safe_row_global, safe_column],
                harm[safe_row_global, safe_column],
            ),
            "safe_top": selected_summary(
                selected,
                targets[start:stop],
                p_dual[start:stop],
                p_harm[start:stop],
            ),
            "within_state_rank": within_state_rank(
                -energy[start:stop], true_score[start:stop], valid[start:stop]
            ),
        })
    return result


def frontier_summary(
    prediction: dict[str, np.ndarray],
    targets: np.ndarray,
    valid: np.ndarray,
    dual_temperature: float,
    harm_temperature: float,
) -> dict[str, Any]:
    p_dual = sigmoid(prediction["dual_logit"] / dual_temperature)
    p_harm = sigmoid(prediction["harm_logit"] / harm_temperature)
    rows = []
    for policy in policy_grid():
        energy = candidate_energy(prediction, p_dual, p_harm, valid, policy)
        safe = candidate_safety_mask(prediction, p_dual, p_harm, valid, policy)
        selected = select_lowest_energy_safe(energy, safe)
        metrics = safety_metrics(selected, targets)
        rows.append({"policy": policy, "metrics": metrics})
    coverage = [row for row in rows if row["metrics"]["coverage"] >= 0.40]
    joint = [
        row
        for row in coverage
        if row["metrics"]["dual_precision"] is not None
        and row["metrics"]["dual_precision"] >= 0.85
        and row["metrics"]["harm_rate"] is not None
        and row["metrics"]["harm_rate"] <= 0.15
        and row["metrics"]["dual_wilson_lower_95"] is not None
        and row["metrics"]["dual_wilson_lower_95"] >= 0.80
    ]
    best_dual = max(
        coverage,
        key=lambda row: (
            row["metrics"]["dual_precision"] or -1.0,
            -(row["metrics"]["harm_rate"] or 1.0),
            row["metrics"]["coverage"],
        ),
        default=None,
    )
    best_harm = min(
        coverage,
        key=lambda row: (
            row["metrics"]["harm_rate"] if row["metrics"]["harm_rate"] is not None else 1.0,
            -(row["metrics"]["dual_precision"] or 0.0),
            -row["metrics"]["coverage"],
        ),
        default=None,
    )
    best_joint = max(
        joint,
        key=lambda row: (
            row["metrics"]["coverage"],
            -row["metrics"]["net_delta_per_state_m"]["objective"],
        ),
        default=None,
    )
    return {
        "dual_temperature": dual_temperature,
        "harm_temperature": harm_temperature,
        "policies": len(rows),
        "policies_with_coverage_at_least_0_40": len(coverage),
        "feasible_policies": len(joint),
        "best_dual_at_coverage": best_dual,
        "best_harm_at_coverage": best_harm,
        "maximum_coverage_feasible": best_joint,
    }


def model_audit(
    domain: str,
    part: dict[str, np.ndarray],
    states: int,
    checkpoints: list[dict[str, Any]],
    policy: dict[str, float],
    dual_temperature: float,
    harm_temperature: float,
    device: str,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    prediction = ensemble_predict(
        checkpoints, part["feature"], part["domain"], device=device
    )
    p_dual = sigmoid(prediction["dual_logit"] / dual_temperature)
    p_harm = sigmoid(prediction["harm_logit"] / harm_temperature)
    energy = candidate_energy(prediction, p_dual, p_harm, part["valid"], policy)
    safe = candidate_safety_mask(
        prediction, p_dual, p_harm, part["valid"], policy
    )
    selected = select_lowest_energy_safe(energy, safe)
    return {
        "domain": domain,
        "dual_temperature": dual_temperature,
        "harm_temperature": harm_temperature,
        "policy": policy,
        "pooled": selected_summary(selected, part["targets_raw"], p_dual, p_harm),
        "sets": set_audits(
            prediction,
            p_dual,
            p_harm,
            part["valid"],
            part["targets_raw"],
            policy,
            states,
        ),
    }, prediction


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    result_path = implementation.OUTPUT / "result.json"
    trained = read_json(result_path)
    if trained.get("status") != "trained_and_frozen_before_A_to_H_replay":
        raise RuntimeError("v66 training is not frozen")
    if trained.get("A_to_H_accessed") or trained.get("reserve_accessed"):
        raise RuntimeError("v66 training reports forbidden access")
    base = read_json(implementation.BASE_RESULT)
    envelopes = implementation.load_envelopes(Path(base["support"]))

    domains = {}
    extension = []
    for domain in implementation.DOMAINS:
        part = implementation.load_domain_split(domain, "scorer_cal", envelopes)
        states = len(part["feature"]) // 4
        base_arm = base["arms"]["D"]
        base_temperature = base_arm["temperature"]["domains"][domain]
        original, _ = model_audit(
            domain,
            part,
            states,
            base_arm["checkpoints"]["full"],
            base_arm["calibration"]["frontier"]["domains"][domain]["policy"],
            float(base_temperature["deployment_dual_temperature"]),
            float(base_temperature["deployment_harm_temperature"]),
            "cuda",
        )
        calibration = trained["calibration"][domain]
        current, prediction = model_audit(
            domain,
            part,
            states,
            trained["checkpoints"]["full"],
            calibration["policy"],
            float(calibration["dual_temperature"]),
            float(calibration["harm_temperature"]),
            "cuda",
        )
        current["frontier_at_frozen_temperature"] = frontier_summary(
            prediction,
            part["targets_raw"],
            part["valid"],
            float(calibration["dual_temperature"]),
            float(calibration["harm_temperature"]),
        )
        if domain == "scand":
            extension = [
                frontier_summary(
                    prediction,
                    part["targets_raw"],
                    part["valid"],
                    temperature,
                    temperature,
                )
                for temperature in EXTENDED_TEMPERATURES
            ]
        domains[domain] = {"original_arm_d": original, "proposal_aware_v66": current}

    payload = {
        "schema": "dualdomain_cem_tail_v66_calibration_failure_diagnosis",
        "status": "complete_without_A_to_H_or_reserve",
        "training_result": str(result_path),
        "training_result_sha256": sha256(result_path),
        "domains": domains,
        "scand_equal_temperature_extension_diagnostic": extension,
        "temperature_extension_is_diagnostic_not_frozen_policy": True,
        "A_to_H_accessed": False,
        "reserve_accessed": False,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "output": str(OUTPUT), "sha256": sha256(OUTPUT)}, indent=2))


if __name__ == "__main__":
    main()
