from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from planning.proposal_tail_sidecar_v1 import canonical_json, identity_hash, sha256_file
from tools.planning.download_prepare_new_scand_bags_v1 import (
    PROCESSED_HEIGHT,
    PROCESSED_WIDTH,
    crop_resize,
    filtered_candidates,
    select_currents,
    select_plan,
    validate_url,
)


def registry(*, forbidden: bool) -> dict:
    value = {
        "schema": "proposal_tail_forbidden_identity_registry_v1",
        "status": "complete_synthetic_unit_test",
        "complete": True,
        "identity_values_are_hash_only": True,
        "route_hashes": [],
        "day_hashes": [],
        "bag_hashes": [],
        "trajectory_hashes": [],
        "window_hashes": [],
    }
    if forbidden:
        value.update({"generated_at": "unit-test", "custodian_id": "unit-test-only"})
    else:
        value["sources"] = []
    return value


def make_args(tmp_path: Path, rows: list[dict[str, str]]) -> argparse.Namespace:
    index = tmp_path / "index.csv"
    with index.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    used = tmp_path / "used.json"
    forbidden = tmp_path / "forbidden.json"
    used.write_text(canonical_json(registry(forbidden=False)) + "\n")
    forbidden.write_text(canonical_json(registry(forbidden=True)) + "\n")
    inventory_rows = []
    for offset, row in enumerate(rows, 2):
        bag = f"random_mdps_{row['FileName']}"
        route = "__".join(sorted(row["FileName"].split("_")[2:4])).lower()
        inventory_rows.append(
            {
                "row_number": offset,
                "public_file_name": row["FileName"],
                "canonical_bag_id": bag,
                "route_group": route,
                "identity_layer_clear": True,
            }
        )
    inventory = tmp_path / "inventory.json"
    inventory.write_text(
        canonical_json(
            {
                "status": "quarantine_fail_closed_missing_forbidden_registry",
                "inputs": {
                    "official_index": {"sha256": sha256_file(index)},
                    "used_registry": {"sha256": sha256_file(used)},
                },
                "rows": inventory_rows,
            }
        )
        + "\n"
    )
    return argparse.Namespace(
        official_index=index,
        identity_inventory=inventory,
        used_registry=used,
        forbidden_registry=forbidden,
        development_routes=2,
        validation_routes=1,
        windows_per_route=30,
        seed=17,
    )


def official_row(index: int, left: str, right: str) -> dict[str, str]:
    file_name = f"A_Spot_{left}_{right}_Mon_Jan_{index}_{index}"
    return {
        "Person": "A",
        "Robot": "Spot",
        "Start Location": left,
        "End Location": right,
        "Link to rosbag": f"https://utexas.box.com/shared/static/file{index}.bag",
        "Tags": "Sidewalk",
        "Distance Travelled (meters)": "100",
        "Time (secs)": str(100 + index),
        "Date": f"Monday Jan {index}",
        "FileName": file_name,
        "Train/Val": "Train",
    }


def test_identity_filter_and_route_disjoint_plan_are_prelabel(tmp_path: Path) -> None:
    args = make_args(
        tmp_path,
        [
            official_row(1, "AHG", "Library"),
            official_row(2, "Fountain", "Library"),
            official_row(3, "SAC", "NHB"),
        ],
    )
    candidates, audit, forbidden = filtered_candidates(args)
    assert audit["registry_clear_bags"] == 3
    assert not any(forbidden.values())
    selected = select_plan(args, candidates)
    assert len(selected) == 3
    development = {row["route_group"] for row in selected if row["cohort_role"] == "development"}
    validation = {row["route_group"] for row in selected if row["cohort_role"] == "one_time_validation"}
    assert not (development & validation)
    assert all("label" not in key and "truth" not in key for row in selected for key in row)


def test_resolution_and_window_contract() -> None:
    source = Image.new("RGB", (1280, 720), color=(10, 20, 30))
    processed = crop_resize(source)
    assert processed.size == (PROCESSED_WIDTH, PROCESSED_HEIGHT)
    currents = select_currents(100, 40)
    assert len(currents) == 40
    assert currents[0] >= 3
    assert currents[-1] + 4 < 100


def test_url_is_restricted_to_official_static_box() -> None:
    assert validate_url("https://utexas.box.com/shared/static/example.bag").endswith(".bag")
    with pytest.raises(RuntimeError):
        validate_url("https://example.com/shared/static/example.bag")
    with pytest.raises(RuntimeError):
        validate_url("http://utexas.box.com/shared/static/example.bag")

