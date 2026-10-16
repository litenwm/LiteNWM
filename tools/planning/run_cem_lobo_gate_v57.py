#!/usr/bin/env python3
"""Run the frozen v57 LCB gate through the audited v55 CEM engine."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import tools.planning.run_cem_safe_elite_v55 as engine  # noqa: E402
from tools.scoring.train_recon_dense_meta_dinov3_v1 import (  # noqa: E402
    file_sha256,
)
from tools.scoring.train_universal_dualdomain_raw_v2 import sigmoid  # noqa: E402


def extract_calibration() -> Path:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--calibration", type=Path, required=True)
    known, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    return known.calibration


def main() -> None:
    calibration_path = extract_calibration()
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    if (
        calibration.get("status") != "one_shared_gate_frozen_before_E_F_outcomes"
        or calibration.get("E_F_outcomes_accessed")
        or calibration.get("reserve_accessed")
        or calibration.get("state_gate_used")
        or calibration.get("cohort_specific_thresholds")
    ):
        raise RuntimeError("v57 calibration freeze is invalid")
    config = calibration["frozen_gate"]

    def conservative_energy(
        prediction: dict[str, np.ndarray],
        *,
        policy: dict[str, float],
        dual_temperature: float,
        harm_temperature: float,
        valid: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        del policy
        p_dual = sigmoid(prediction["dual_logit"] / dual_temperature)
        p_harm = sigmoid(prediction["harm_logit"] / harm_temperature)
        mean = prediction["mean"]
        std = prediction["std"]
        benefit_lcb = -0.5 * (mean[..., 0] + mean[..., 3]) - float(
            config["beta"]
        ) * np.sqrt(std[..., 0] ** 2 + std[..., 3] ** 2)
        energy = -benefit_lcb + 0.25 * p_harm - 0.10 * p_dual
        energy = np.where(valid, energy, np.inf).astype(np.float32)
        energy[:, 0] = np.inf
        return energy, p_dual.astype(np.float32), p_harm.astype(np.float32)

    def candidate_safety_mask(
        prediction: dict[str, np.ndarray],
        p_dual: np.ndarray,
        p_harm: np.ndarray,
        valid: np.ndarray,
        policy: dict[str, float],
    ) -> np.ndarray:
        del policy
        mean = prediction["mean"]
        std = prediction["std"]
        benefit_lcb = -0.5 * (mean[..., 0] + mean[..., 3]) - float(
            config["beta"]
        ) * np.sqrt(std[..., 0] ** 2 + std[..., 3] ** 2)
        safe = (
            np.asarray(valid, dtype=bool)
            & (benefit_lcb >= float(config["benefit_threshold"]))
            & (p_dual >= float(config["dual_threshold"]))
            & (p_harm <= float(config["harm_threshold"]))
        )
        safe[:, 0] = False
        return safe

    engine.conservative_energy = conservative_energy
    engine.candidate_safety_mask = candidate_safety_mask
    engine.main()

    output_index = sys.argv.index("--output-dir") + 1
    output_dir = Path(sys.argv[output_index])
    binding = {
        "schema": "cem_lobo_gate_v57_runtime_binding",
        "calibration": str(calibration_path),
        "calibration_sha256": file_sha256(calibration_path),
        "frozen_gate": config,
        "formula": calibration["formula"],
        "E_F_outcomes_accessed": False,
        "reserve_accessed": False,
        "state_gate_used": False,
    }
    (output_dir / "v57_runtime_binding.json").write_text(
        json.dumps(binding, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
