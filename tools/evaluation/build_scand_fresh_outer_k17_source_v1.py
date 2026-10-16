#!/usr/bin/env python3
"""Build the label-free dense feature source for the fresh SCAND outer set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tools.evaluation.build_recon100_k17_source_v1 import (  # noqa: E402
    action_set_features,
    history_features,
    write_npz,
)
from tools.planning.build_scand100_nomad_k17_v1 import sha256, write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--trajectory-root", type=Path, default=Path("/home/rainbow/nwm/data/scand")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit_path = args.output.with_suffix(".audit.json")
    if args.output.exists() or audit_path.exists():
        raise FileExistsError(args.output)
    rows = [
        json.loads(line)
        for line in args.manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != 100 or any(row.get("partition") != "fresh-outer" for row in rows):
        raise RuntimeError("fresh-outer H4-100 manifest contract failed")
    if len({str(row["trajectory_id"]) for row in rows}) != 25:
        raise RuntimeError("fresh-outer trajectory count changed")
    if len({str(row["route_group"]) for row in rows}) != 25:
        raise RuntimeError("fresh-outer route count changed")

    with np.load(args.candidate, allow_pickle=False) as payload:
        candidates = payload["candidates_waypoint_units"].astype(np.float32)
        reference = payload["nomad_x16_waypoint_units"].astype(np.float32)
        observed_uid = payload["window_uid"].astype(str)
        observed_record = payload["record_key"].astype(str)
    expected_uid = np.asarray([str(row["window_uid"]) for row in rows])
    expected_record = np.asarray(
        [f"{row['trajectory_id']}:{int(row['current_t'])}" for row in rows]
    )
    if not np.array_equal(observed_uid, expected_uid) or not np.array_equal(
        observed_record, expected_record
    ):
        raise RuntimeError("fresh candidate/manifest identity mismatch")
    if candidates.shape != (100, 17, 4, 2) or reference.shape != (100, 16, 4, 2):
        raise RuntimeError("fresh-outer K17 shape changed")

    bridge = np.float32(0.38 / 0.25)
    feature_candidates = candidates * bridge
    feature_reference = reference * bridge
    legacy_rows = [
        {"traj_name": row["trajectory_id"], "curr_time": int(row["current_t"])}
        for row in rows
    ]
    action_set = action_set_features(feature_candidates, feature_reference)
    motion = history_features(
        feature_candidates, feature_reference, legacy_rows, args.trajectory_root
    )
    raw_action = np.concatenate((action_set, motion), axis=-1).astype(np.float32)
    action_features = np.concatenate(
        (raw_action, raw_action - raw_action[:, :1]), axis=-1
    ).astype(np.float32)
    if action_features.shape != (100, 17, 242):
        raise RuntimeError("fresh-outer action feature contract changed")
    family = np.asarray([f"scand::{row['route_group']}" for row in rows])
    write_npz(
        args.output,
        candidates=feature_candidates,
        candidates_scand_waypoint_units=candidates,
        action_features=action_features,
        record_key=expected_record,
        family=family,
        trajectory=np.asarray([str(row["trajectory_id"]) for row in rows]),
        route_group=np.asarray([str(row["route_group"]) for row in rows]),
        bag_id=np.asarray([str(row["bag_id"]) for row in rows]),
        window_uid=expected_uid,
    )
    audit = {
        "schema": "scand_fresh_outer_k17_dense_source_v1",
        "partition": "fresh-outer",
        "states": 100,
        "trajectories": 25,
        "route_groups": 25,
        "bags": 25,
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256(args.manifest),
        "candidate": str(args.candidate.resolve()),
        "candidate_sha256": sha256(args.candidate),
        "output": str(args.output.resolve()),
        "output_sha256": sha256(args.output),
        "waypoint_spacing_m": 0.38,
        "legacy_feature_unit_bridge": "multiply waypoint units by 0.38/0.25",
        "past_motion_features_used": True,
        "future_pose_targets_computed": False,
        "future_pose_targets_opened": False,
        "labels_in_output": False,
        "fixed_pool_only": True,
        "cem": False,
    }
    write_json(audit_path, audit)
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
