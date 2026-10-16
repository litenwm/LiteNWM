#!/usr/bin/env python3
"""Freeze K16/K32 old-V70 selected paths before fair metric labels are opened."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from planning.nomad_action_adapter import (  # noqa: E402
    nomad_waypoints_to_local_actions,
)


SPACING_M = {"recon": 0.25, "scand": 0.38, "sacson": 0.255}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", choices=tuple(SPACING_M), required=True
    )
    parser.add_argument("--cohort-id", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--k16-selected", type=Path, required=True)
    parser.add_argument("--k32-merged", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_rows(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        if isinstance(value, list):
            rows = value
        elif isinstance(value, dict):
            rows = value.get("records") or value.get("windows") or value.get("rows")
        else:
            rows = None
    if not isinstance(rows, list) or not rows:
        raise RuntimeError("manifest has no rows")
    return rows


def current_time(row: dict[str, Any]) -> int:
    value = row.get("current_t", row.get("curr_time"))
    if value is None:
        raise KeyError("manifest row misses current_t/curr_time")
    return int(value)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_npz(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    rows = read_rows(args.manifest)
    expected = np.asarray(
        [
            f"{row['trajectory_id']}:{current_time(row)}"
            for row in rows
        ]
    )
    k16 = load_npz(args.k16_selected)
    k32 = load_npz(args.k32_merged)
    if not (
        np.array_equal(k16["record_key"].astype(str), expected)
        and np.array_equal(k32["record_key"].astype(str), expected)
    ):
        raise RuntimeError("manifest/K16/K32 identity mismatch")
    row = np.arange(len(rows))
    k16_path_feature = k16["final_cem_feature_waypoint_units"][
        row, k16["final_cem_selected_index"].astype(np.int64)
    ].astype(np.float32)
    k32_path_feature = k32[
        "selected_feature_waypoint_units"
    ].astype(np.float32)
    spacing = SPACING_M[args.dataset]

    def convert(path_feature: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        path_m = path_feature * np.float32(0.25)
        native = path_m / np.float32(spacing)
        actions = nomad_waypoints_to_local_actions(
            native, dataset=args.dataset
        ).astype(np.float32)
        return path_m.astype(np.float32), actions

    k16_path_m, k16_actions = convert(k16_path_feature)
    k32_path_m, k32_actions = convert(k32_path_feature)
    args.output_dir.mkdir(parents=True)
    selection_path = args.output_dir / "selected_actions.npz"
    write_npz(
        selection_path,
        record_key=expected,
        k16_actions_h4=k16_actions,
        k16_path_m=k16_path_m,
        k32_actions_h4=k32_actions,
        k32_path_m=k32_path_m,
        k16_feature_waypoint_units=k16_path_feature,
        k32_feature_waypoint_units=k32_path_feature,
    )
    audit = {
        "schema": "proposal_aware_v70_tridomain_fair_selection_v1",
        "created_at": utc_now(),
        "status": "actions_frozen_before_future_poses",
        "dataset": args.dataset,
        "cohort_id": args.cohort_id,
        "states": len(rows),
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256(args.manifest),
        "k16_source": str(args.k16_selected.resolve()),
        "k16_source_sha256": sha256(args.k16_selected),
        "k32_source": str(args.k32_merged.resolve()),
        "k32_source_sha256": sha256(args.k32_merged),
        "selection": str(selection_path.resolve()),
        "selection_sha256": sha256(selection_path),
        "future_pose_opened": False,
        "test_labels_used_by_selection": False,
        "implementation": (
            "original V70 proposal-aware scorer, candidate-level safe elite, "
            "three-round CEM, cross-round safe archive, incumbent fallback"
        ),
    }
    write_json(args.output_dir / "selection_audit.json", audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
