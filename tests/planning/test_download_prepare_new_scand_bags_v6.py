from __future__ import annotations

import argparse

from tools.planning.download_prepare_new_scand_bags_v6 import freeze_corrected_plan


def test_corrected_plan_is_byte_idempotent(tmp_path) -> None:
    args = argparse.Namespace(
        seed=20260723,
        windows_per_route=40,
        development_routes=12,
        validation_routes=5,
    )
    selected = [
        {
            "official_file_name": "example",
            "route_group": "example__route",
            "cohort_role": "development",
        }
    ]
    path = tmp_path / "plan.json"
    first = freeze_corrected_plan(args, selected, {"audit": "ok"}, {"reason": "test"}, path)
    content = path.read_bytes()
    second = freeze_corrected_plan(args, selected, {"audit": "ok"}, {"reason": "test"}, path)
    assert first == second
    assert path.read_bytes() == content
