#!/usr/bin/env python3
"""Aggregate v66 replays under the user-corrected frozen acceptance contract."""

from __future__ import annotations

from pathlib import Path

from tools.evaluation import aggregate_dualdomain_cem_tail_v66_replays as implementation
from tools.scoring.train_recon_dense_meta_dinov3_v1 import file_sha256


ACCEPTANCE = (
    implementation.ROOT
    / "outputs/cem_development_v1/"
    "dualdomain_cem_tail_v66_development_acceptance_amendment.json"
)
_read_json = implementation.read_json
_write_json_atomic = implementation.write_json_atomic


def read_json_with_acceptance(path: Path):
    value = _read_json(path)
    if value.get("schema") == "dualdomain_cem_tail_v66_outcome_replay_result":
        if value.get("development_acceptance_amendment_sha256") != file_sha256(
            ACCEPTANCE
        ):
            raise RuntimeError(f"acceptance amendment binding changed: {path}")
    return value


def write_json_with_acceptance(path: Path, value) -> None:
    if value.get("schema") == "dualdomain_cem_tail_v66_aggregate_result":
        checks = dict(value["development_checks"])
        diagnostic = checks.pop("coverage_gain_at_least_0_10")
        promising = all(checks.values())
        value = {
            **value,
            "status": (
                "promising_development_result"
                if promising
                else "failed_frozen_development_replay"
            ),
            "development_checks": checks,
            "coverage_gain_at_least_0_10_diagnostic_only": diagnostic,
            "promising_development_result": promising,
            "development_acceptance_amendment": str(ACCEPTANCE),
            "development_acceptance_amendment_sha256": file_sha256(ACCEPTANCE),
        }
    _write_json_atomic(path, value)


def main() -> None:
    acceptance = _read_json(ACCEPTANCE)
    if acceptance.get("status") != "frozen_before_A_to_H_replay":
        raise RuntimeError("development acceptance amendment is not frozen")
    if acceptance.get("A_to_H_accessed") or acceptance.get("reserve_accessed"):
        raise RuntimeError("acceptance amendment reports forbidden access")
    implementation.read_json = read_json_with_acceptance
    implementation.write_json_atomic = write_json_with_acceptance
    implementation.main()


if __name__ == "__main__":
    main()
