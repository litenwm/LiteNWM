#!/usr/bin/env python3
"""Final v66 training entrypoint with all pretraining corrections sealed."""

from __future__ import annotations

from tools.scoring import run_train_dualdomain_cem_tail_v66_candidate_level as corrected
from tools.scoring.train_recon_dense_meta_dinov3_v1 import file_sha256


implementation = corrected.implementation
CORRECTION = (
    implementation.ROOT
    / "outputs/cem_development_v1/"
    "dualdomain_cem_tail_v66_candidate_level_calibration_correction.json"
)
_write_json_atomic = implementation.write_json_atomic


def write_json_with_candidate_level_seal(path, value):
    if value.get("schema") == "dualdomain_cem_tail_v66_training_result":
        value = {
            **value,
            "calibration_decision": (
                "candidate-level safety mask then lowest conservative energy"
            ),
            "candidate_level_calibration_correction": str(CORRECTION),
            "candidate_level_calibration_correction_sha256": file_sha256(CORRECTION),
        }
    _write_json_atomic(path, value)


implementation.write_json_atomic = write_json_with_candidate_level_seal


if __name__ == "__main__":
    implementation.main()
