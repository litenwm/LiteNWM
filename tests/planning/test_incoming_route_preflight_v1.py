"""Fail-closed tests for the incoming SCAND route data-access layer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from planning.proposal_tail_sidecar_v1 import canonical_json, identity_hash
from tools.planning.build_incoming_scand_route_manifest_v1 import build_rows
from tools.planning.check_proposal_tail_readiness_v1 import evaluate_readiness


def write_png(path: Path, width: int = 2, height: int = 3) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + (13).to_bytes(4, "big")
        + b"IHDR"
        + width.to_bytes(4, "big")
        + height.to_bytes(4, "big")
    )


def write_route(
    root: Path,
    *,
    role: str,
    route: str,
    trajectory: str,
    missing_timestamp: bool = False,
) -> Path:
    frames = []
    for index in range(4):
        relative = f"frames/{index}.png"
        write_png(root / relative)
        frames.append(
            {
                "frame_index": index,
                "timestamp": None if missing_timestamp and index == 2 else float(index),
                "image": relative,
                "encoding": "png",
            }
        )
    metadata = {
        "schema": "incoming_scand_route_source_v1",
        "example_only": False,
        "public_nonreserve": True,
        "labels_opened": False,
        "cohort_role": role,
        "route_group": route,
        "route_id": route + "_id",
        "collection_day": "2026-07-23" if role == "development" else "2026-07-24",
        "bag_id": route + "_bag",
        "trajectory_id": trajectory,
        "timestamp_unit": "seconds",
        "frames": frames,
        "windows": [
            {
                "window_uid": trajectory + "_w0",
                "history_timesteps": [0, 1],
                "current_t": 2,
                "goal_t": 3,
            }
        ],
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "route_metadata.json").write_text(
        json.dumps(metadata, sort_keys=True) + "\n", encoding="utf-8"
    )
    return root


def build_manifest(route_dir: Path, destination: Path) -> Path:
    rows, _ = build_rows(route_dir)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        "".join(canonical_json(row) + "\n" for row in rows), encoding="utf-8"
    )
    return destination


def registry(path: Path, *, complete: bool = True, route_hashes: list[str] | None = None) -> Path:
    value: dict[str, Any] = {
        "schema": "proposal_tail_forbidden_identity_registry_v1",
        "status": "test_hash_only",
        "complete": complete,
        "identity_values_are_hash_only": True,
        "route_hashes": route_hashes or [],
        "day_hashes": [],
        "bag_hashes": [],
        "trajectory_hashes": [],
        "window_hashes": [],
    }
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    return path


def valid_fixture(root: Path) -> tuple[Path, Path, Path, Path]:
    development = build_manifest(
        write_route(root / "dev", role="development", route="route_dev", trajectory="traj_dev"),
        root / "dev_manifest.windows.jsonl",
    )
    validation = build_manifest(
        write_route(
            root / "validation",
            role="one_time_validation",
            route="route_validation",
            trajectory="traj_validation",
        ),
        root / "validation_manifest.windows.jsonl",
    )
    used = registry(root / "used_hashes.json")
    forbidden = registry(root / "forbidden_hashes.json")
    return development, validation, used, forbidden


def test_valid_disjoint_prelabel_input_is_ready(tmp_path: Path) -> None:
    development, validation, used, forbidden = valid_fixture(tmp_path)
    result = evaluate_readiness(development, validation, used, forbidden)
    assert result["ready"] is True
    assert result["status"] == "ready_prelabel"
    assert result["reserve_content_accessed"] is False


def test_forbidden_path_token_fails_closed(tmp_path: Path) -> None:
    route = write_route(
        tmp_path / "reserve" / "route",
        role="development",
        route="route_dev",
        trajectory="traj_dev",
    )
    try:
        build_rows(route)
    except RuntimeError as exc:
        assert "forbidden cohort path" in str(exc)
    else:
        raise AssertionError("forbidden path did not fail closed")


def test_missing_timestamp_fails_closed(tmp_path: Path) -> None:
    route = write_route(
        tmp_path / "missing_timestamp",
        role="development",
        route="route_dev",
        trajectory="traj_dev",
        missing_timestamp=True,
    )
    try:
        build_rows(route)
    except RuntimeError as exc:
        assert "timestamp" in str(exc)
    else:
        raise AssertionError("missing timestamp did not fail closed")


def test_used_identity_overlap_fails_closed(tmp_path: Path) -> None:
    development, validation, _, forbidden = valid_fixture(tmp_path)
    used = registry(
        tmp_path / "used_overlap.json",
        route_hashes=[identity_hash("route", "route_dev")],
    )
    result = evaluate_readiness(development, validation, used, forbidden)
    assert result["ready"] is False
    assert "overlaps v70-v77" in result["errors"][0]


def test_incomplete_forbidden_registry_fails_closed(tmp_path: Path) -> None:
    development, validation, used, _ = valid_fixture(tmp_path)
    forbidden = registry(tmp_path / "forbidden_incomplete.json", complete=False)
    result = evaluate_readiness(development, validation, used, forbidden)
    assert result["ready"] is False
    assert "incomplete identity registry" in result["errors"][0]


def test_development_validation_route_overlap_fails_closed(tmp_path: Path) -> None:
    development = build_manifest(
        write_route(tmp_path / "dev", role="development", route="same_route", trajectory="traj_dev"),
        tmp_path / "dev_manifest.windows.jsonl",
    )
    validation = build_manifest(
        write_route(
            tmp_path / "validation",
            role="one_time_validation",
            route="same_route",
            trajectory="traj_validation",
        ),
        tmp_path / "validation_manifest.windows.jsonl",
    )
    result = evaluate_readiness(
        development,
        validation,
        registry(tmp_path / "used.json"),
        registry(tmp_path / "forbidden.json"),
    )
    assert result["ready"] is False
    assert "route overlap" in result["errors"][0]


def test_opened_label_or_changed_image_fails_closed(tmp_path: Path) -> None:
    development, validation, used, forbidden = valid_fixture(tmp_path)
    row = json.loads(validation.read_text(encoding="utf-8"))
    row["labels_opened"] = True
    validation.write_text(canonical_json(row) + "\n", encoding="utf-8")
    opened = evaluate_readiness(development, validation, used, forbidden)
    assert opened["ready"] is False
    assert "opened labels" in opened["errors"][0]

    development, validation, used, forbidden = valid_fixture(tmp_path / "changed")
    image = tmp_path / "changed/dev/frames/0.png"
    image.write_bytes(image.read_bytes() + b"changed")
    changed = evaluate_readiness(development, validation, used, forbidden)
    assert changed["ready"] is False
    assert "size changed" in changed["errors"][0]

