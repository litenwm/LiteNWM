#!/usr/bin/env python3
"""Fresh-outer entry point for the shared resumable SCAND K17 generator."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tools.planning import build_scand_inner_nomad_k17_v1 as shared


_write_json = shared.write_json


def sealed_write_json(path: Path, value: Any) -> None:
    if isinstance(value, dict) and value.get("partition") == "fresh-outer":
        value = {
            **value,
            "schema": "scand_fresh_outer_nomad_k17_freeze_v1",
            "status": "candidate_pool_frozen_before_fresh_outer_labels",
            "partition_outcome_status": "sealed_during_candidate_generation",
            "fresh_outer_pose_outcomes_opened": False,
            "internal_holdout_pose_outcomes_opened": False,
        }
    _write_json(path, value)


def main() -> None:
    shared.PARTITIONS = (*shared.PARTITIONS, "fresh-outer")
    shared.write_json = sealed_write_json
    shared.main()


if __name__ == "__main__":
    main()
