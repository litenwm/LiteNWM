from __future__ import annotations

import argparse

from tools.planning import download_prepare_new_scand_bags_v1 as v1
from tools.planning.download_prepare_new_scand_bags_v5 import (
    EXCLUDED_FILE,
    corrected_selection,
)


def test_routefit_excludes_short_route_and_rebalances(monkeypatch) -> None:
    original = []
    for index in range(18):
        original.append(
            {
                "official_file_name": EXCLUDED_FILE if index == 0 else f"bag_{index}",
                "route_group": "gdc__gdc" if index == 0 else f"route_{index}",
                "cohort_role": "development" if index < 12 else "one_time_validation",
                "duration_seconds": 9.6 if index == 0 else 100.0,
            }
        )
    monkeypatch.setattr(v1, "select_plan", lambda args, candidates: [dict(row) for row in original])
    args = argparse.Namespace(
        development_routes=12,
        validation_routes=6,
        windows_per_route=40,
        seed=20260723,
    )
    selected, correction = corrected_selection(args, [])
    assert len(selected) == 17
    assert all(row["official_file_name"] != EXCLUDED_FILE for row in selected)
    assert sum(row["cohort_role"] == "development" for row in selected) == 12
    assert sum(row["cohort_role"] == "one_time_validation" for row in selected) == 5
    assert correction["observed_eligible_h4_windows"] == 29
    assert correction["required_windows"] == 40
