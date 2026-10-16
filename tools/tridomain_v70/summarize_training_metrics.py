#!/usr/bin/env python3
"""Print a compact summary of the latest proposal-aware training epoch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics", type=Path)
    args = parser.parse_args()

    lines = [
        line for line in args.metrics.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not lines:
        raise RuntimeError("metrics file has no completed epoch")
    latest = json.loads(lines[-1])
    calibration = latest["calibration"]
    domains = {}
    for name, value in calibration["domains"].items():
        metrics = value["metrics"]
        domains[name] = {
            "ate_ratio": value["ate_ratio"],
            "fde_ratio": value["fde_ratio"],
            "feasible": value["feasible"],
            "coverage": metrics["coverage"],
            "dual_precision": metrics["dual_precision"],
            "harm_rate": metrics["harm_rate"],
        }
    print(
        json.dumps(
            {
                "epoch": latest["epoch"],
                "elapsed_sec": latest["elapsed_sec"],
                "member_train_loss": latest["member_train_loss"],
                "macro_ate_ratio": calibration["macro_ate_ratio"],
                "macro_fde_ratio": calibration["macro_fde_ratio"],
                "all_domains_feasible": calibration[
                    "all_domains_feasible"
                ],
                "domains": domains,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
