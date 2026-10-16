#!/usr/bin/env python3
"""Print a compact, read-only summary of mechanism decomposition v1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result", type=Path)
    args = parser.parse_args()
    value = json.loads(args.result.read_text(encoding="utf-8"))
    stages = {}
    for reference, records in value["funnel"].items():
        stages[reference] = {}
        for name, record in records.items():
            if name == "scorer_archive_v70":
                stages[reference][name] = record
            else:
                stages[reference][name] = {
                    "candidates": record["candidate"]["candidates"],
                    "states_with_dual_opportunity": record["candidate"]["states_with_dual_opportunity"],
                    "dual_rate": record["candidate"]["dual_rate"],
                    "harm_rate": record["candidate"]["harm_rate"],
                    "oracle_objective_gain_m": record["oracle"]["objective_gain_vs_reference_m"],
                }
    summary = {
        "states": value["states"],
        "routes": value["routes"],
        "union_candidates": value["union_candidates"],
        "raw_draws": value["raw_draws"],
        "funnel": stages,
        "route_round": value["route_round"],
        "fixed_strata_deciles": value["fixed_strata_deciles"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
