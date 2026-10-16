#!/usr/bin/env python3
"""Build the label-free old-V70 dense source for a tri-domain K17 pool."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from tools.evaluation.build_recon100_k17_source_v1 import (  # noqa: E402
    action_set_features,
    history_features,
)


SPACING_M = {"recon": 0.25, "scand": 0.38, "sacson": 0.255}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", choices=tuple(SPACING_M), required=True
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--trajectory-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alias-manifest-output", type=Path)
    parser.add_argument("--alias-candidate-output", type=Path)
    return parser.parse_args()


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


def write_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def family_for(row: dict[str, Any], dataset: str) -> str:
    for key in ("route_id", "route_group", "bag_id"):
        if key in row and str(row[key]):
            return f"{dataset}::{row[key]}"
    return f"{dataset}::{row['trajectory_id']}"


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    rows = read_rows(args.manifest)
    with np.load(args.candidate, allow_pickle=False) as archive:
        candidate = {key: archive[key].copy() for key in archive.files}
    required = {
        "candidates_waypoint_units",
        "nomad_x16_waypoint_units",
        "record_key",
    }
    missing = required.difference(candidate)
    if missing:
        raise RuntimeError(f"candidate artifact misses {sorted(missing)}")
    expected = np.asarray(
        [
            f"{row['trajectory_id']}:{current_time(row)}"
            for row in rows
        ]
    )
    if not np.array_equal(candidate["record_key"].astype(str), expected):
        raise RuntimeError("manifest/candidate identity mismatch")

    bridge = np.float32(SPACING_M[args.dataset] / 0.25)
    candidates = (
        candidate["candidates_waypoint_units"].astype(np.float32) * bridge
    )
    reference = (
        candidate["nomad_x16_waypoint_units"].astype(np.float32) * bridge
    )
    legacy_rows = [
        {
            "traj_name": str(row["trajectory_id"]),
            "curr_time": current_time(row),
        }
        for row in rows
    ]
    action_set = action_set_features(candidates, reference)
    motion = history_features(
        candidates, reference, legacy_rows, args.trajectory_root
    )
    raw = np.concatenate((action_set, motion), axis=-1).astype(np.float32)
    action = np.concatenate((raw, raw - raw[:, :1]), axis=-1)
    if action.shape != (len(rows), 17, 242):
        raise RuntimeError(f"old V70 action feature shape changed: {action.shape}")
    write_npz(
        args.output,
        candidates=candidates,
        action_features=action.astype(np.float32),
        record_key=expected,
        family=np.asarray(
            [family_for(row, args.dataset) for row in rows]
        ),
    )

    # The legacy replay parser has only recon/scand CLI tokens.  SACSoN uses a
    # decision-equivalent alias: candidate native units are rescaled so the
    # replay's 0.38/0.25 bridge reconstructs the exact 0.255/0.25 feature path.
    if args.dataset == "sacson":
        if (
            args.alias_manifest_output is None
            or args.alias_candidate_output is None
        ):
            raise ValueError("SACSoN requires both alias outputs")
        alias_rows = []
        for row in rows:
            value = dict(row)
            value["route_group"] = str(
                row.get(
                    "route_group",
                    row.get("bag_id", row.get("trajectory_id")),
                )
            )
            alias_rows.append(value)
        args.alias_manifest_output.parent.mkdir(parents=True, exist_ok=True)
        args.alias_manifest_output.write_text(
            json.dumps(
                {"records": alias_rows},
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        scale = np.float32(0.255 / 0.38)
        alias_candidate = dict(candidate)
        alias_candidate["candidates_waypoint_units"] = (
            candidate["candidates_waypoint_units"].astype(np.float32)
            * scale
        )
        alias_candidate["nomad_x16_waypoint_units"] = (
            candidate["nomad_x16_waypoint_units"].astype(np.float32)
            * scale
        )
        write_npz(args.alias_candidate_output, **alias_candidate)

    print(
        json.dumps(
            {
                "status": "OLD_V70_DENSE_SOURCE_READY",
                "dataset": args.dataset,
                "states": len(rows),
                "labels_accessed": False,
                "output": str(args.output),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
