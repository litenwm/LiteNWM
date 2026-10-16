#!/usr/bin/env python3
"""Tri-domain V70 runtime compatibility helpers.

This module intentionally does not monkey-patch or execute the historical
dual-domain replay entry point. It exposes only the frozen-model contract and
the standard proposal-aware ensemble loader needed by the Orin shadow stages.
"""

from __future__ import annotations

import json
from pathlib import Path

from deploy.v70_go2_orin.v70_runtime_contract import (
    require_domain_calibration,
)
from tools.planning import run_dualdomain_cem_tail_v66_replay as replay
from tools.scoring.train_recon_dense_meta_dinov3_v1 import file_sha256


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def frozen_model_contract(
    *,
    role: str,
    domain: str,
    base_result_path: Path,
    v70_result_path: Path,
    allow_relaxed: bool = False,
):
    """Load an explicitly sealed tri-domain calibration contract."""

    if role == "original_arm_d":
        return replay.frozen_model_contract(
            role=role,
            domain=domain,
            base_result_path=base_result_path,
            v66_result_path=v70_result_path,
        )

    value = read_json(v70_result_path)
    if value.get("A_to_H_used_by_v70_training_or_calibration"):
        raise RuntimeError("v70 reports forbidden A-H calibration use")
    if value.get("fresh_scand_used_by_v70_training_or_calibration"):
        raise RuntimeError("v70 reports forbidden fresh-SCAND calibration use")
    if value.get("reserve_accessed"):
        raise RuntimeError("v70 reports reserve access")

    runtime = require_domain_calibration(
        value,
        domain,
        allow_relaxed=allow_relaxed,
    )
    calibration = runtime["calibration"]
    return (
        value["checkpoints"]["full"],
        calibration["policy"],
        float(calibration["dual_temperature"]),
        float(calibration["harm_temperature"]),
        {
            "model_role": role,
            "model_version": "v70_tridomain",
            "model_result": str(v70_result_path),
            "model_result_sha256": file_sha256(v70_result_path),
            "model_trained_with_cem": True,
            "calibration_contract": runtime["calibration_contract"],
            "calibration_feasible": (
                runtime["calibration_contract"] == "strict"
            ),
            "relaxed_calibration_feasible_v1": (
                runtime["calibration_contract"] == "relaxed_v1"
            ),
            "A_to_H_used_by_calibration": False,
            "fresh_scand_used_by_calibration": False,
        },
    )


def compatible_ensemble_predict(
    records,
    feature,
    domain,
    *,
    device,
):
    """Use the proposal-aware loader required by tri-domain checkpoints."""

    return replay.ensemble_predict(
        records,
        feature,
        domain,
        device=device,
    )
