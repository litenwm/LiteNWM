#!/usr/bin/env python3
"""Run the v70 reserve pair binder against the sealed runner config fields."""

from __future__ import annotations

import copy
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from tools.evaluation import bind_dualdomain_cem_tail_v70_reserve_pair as core  # noqa: E402


CONFIG_KEYS = (
    "iterations",
    "population_challengers",
    "set_cardinality_with_incumbent",
    "elite_fraction",
    "elite_count",
    "proposal_scale",
    "covariance_floor_waypoint_units",
    "support_quantile",
    "support_threshold",
    "seed",
    "spacing_m",
    "cross_round_safe_archive",
)


_read_json = core.read_json


def read_json(path: Path):
    value = _read_json(path)
    if value.get("schema") == "dualdomain_cem_tail_v70_reserve_sealed_manifest":
        value = copy.deepcopy(value)
        contract = value["cem_contract"]
        value["cem_contract"] = {key: contract[key] for key in CONFIG_KEYS}
    return value


def main() -> None:
    core.read_json = read_json
    core.main()


if __name__ == "__main__":
    main()
