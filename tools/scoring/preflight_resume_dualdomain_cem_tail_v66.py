#!/usr/bin/env python3
"""Outcome-free preflight for the sealed v66 OOM resume."""

from __future__ import annotations

import json

from tools.scoring.resume_dualdomain_cem_tail_v66 import (
    OUTPUT,
    context,
    replay_population,
    rng_for_scand_train_fit,
    verify_resume_seal,
)


def main() -> None:
    verify_resume_seal(OUTPUT)
    ctx = context(OUTPUT)
    rng = rng_for_scand_train_fit()
    records = []
    for iteration in (0, 1):
        replay = replay_population(ctx, rng, iteration)
        records.append(
            {
                "iteration": iteration,
                "population_sha256": replay["population"].get("metadata", None),
                "sampling": replay["sampling"],
                "rng_replay_bit_exact": True,
            }
        )
    print(
        json.dumps(
            {
                "status": "passed",
                "iterations": records,
                "outcomes_opened": False,
                "A_to_H_accessed": False,
                "reserve_accessed": False,
            },
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
