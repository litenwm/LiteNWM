#!/usr/bin/env python3
"""Run the v66 evaluator with the user-corrected frozen acceptance contract."""

from __future__ import annotations

from pathlib import Path

from tools.evaluation import evaluate_dualdomain_cem_tail_v66_replay as implementation
from tools.scoring.train_recon_dense_meta_dinov3_v1 import file_sha256


ACCEPTANCE = (
    implementation.ROOT
    / "outputs/cem_development_v1/"
    "dualdomain_cem_tail_v66_development_acceptance_amendment.json"
)
_write_json_atomic = implementation.write_json_atomic


def write_json_with_acceptance(path: Path, value) -> None:
    if value.get("schema") == "dualdomain_cem_tail_v66_outcome_replay_result":
        checks = dict(value["development_checks"])
        diagnostic = checks.pop("coverage_gain_at_least_0_10")
        checks["coverage_gain_at_least_0_10_diagnostic_only"] = diagnostic
        checks["cohort_pass"] = all(
            item
            for key, item in checks.items()
            if key not in {
                "cohort_pass",
                "coverage_gain_at_least_0_10_diagnostic_only",
            }
        )
        value = {
            **value,
            "development_checks": checks,
            "development_acceptance_amendment": str(ACCEPTANCE),
            "development_acceptance_amendment_sha256": file_sha256(ACCEPTANCE),
        }
    _write_json_atomic(path, value)


def main() -> None:
    acceptance = implementation.read_json(ACCEPTANCE)
    if acceptance.get("status") != "frozen_before_A_to_H_replay":
        raise RuntimeError("development acceptance amendment is not frozen")
    if acceptance.get("A_to_H_accessed") or acceptance.get("reserve_accessed"):
        raise RuntimeError("acceptance amendment reports forbidden access")
    implementation.write_json_atomic = write_json_with_acceptance
    implementation.main()


if __name__ == "__main__":
    main()
