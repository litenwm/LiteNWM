#!/usr/bin/env python3
"""Apply the frozen pose-branch diagnostic to the other four development bags."""

from __future__ import annotations

import json
from pathlib import Path

import diagnose_external_development_pose_branches_v1 as base


BAGS = (
    "A_Spot_EERC_Ahg_Thu_Nov_11_80.bag",
    "A_Spot_NHB_Ahg_Wed_Nov_10_55.bag",
    "B_Spot_RLM_RLM_Wed_Nov_10_44.bag",
    "B_Spot_SAC_SAC_Mon_Nov_15_109.bag",
)


def main() -> None:
    rows = {}
    for name in BAGS:
        result = base.analyze(
            base.read_odometry(Path("/mnt/d/VEDIOS") / name)
        )
        rows[name] = result
        print(
            json.dumps(
                {
                    "bag": name,
                    **{
                        key: result[key]
                        for key in (
                            "odometry_samples",
                            "position_candidates",
                            "position_and_heading_candidates",
                            "different_future_pairs",
                            "independent_anchor_pairs",
                        )
                    },
                },
                sort_keys=True,
            ),
            flush=True,
        )
    print(
        json.dumps(
            {
                "total_independent_anchor_pairs": sum(
                    row["independent_anchor_pairs"] for row in rows.values()
                ),
                "one_time_validation_bags_opened": 0,
                "reserve_bags_opened": 0,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
