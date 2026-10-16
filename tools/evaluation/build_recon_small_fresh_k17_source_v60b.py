#!/usr/bin/env python3
"""Build the label-free dense source for a frozen 11-route/44-state v60 bank."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.evaluation.build_recon100_k17_source_v1 import (  # noqa: E402
    action_set_features,
    history_features,
    write_npz,
)
from tools.planning.build_scand100_nomad_k17_v1 import sha256, write_json  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-audit", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--trajectory-root", type=Path, default=Path("/home/rainbow/nwm/data/recon")
    )
    args = parser.parse_args()
    output_audit = args.output.with_suffix(".audit.json")
    if args.output.exists() or output_audit.exists():
        raise FileExistsError(args.output)

    audit = json.loads(args.manifest_audit.read_text(encoding="utf-8"))
    if (
        audit.get("status") != "11route_44state_manifest_frozen_before_outcomes"
        or audit.get("labels_opened")
        or audit.get("future_pose_opened")
        or audit.get("outcomes_accessed")
        or audit.get("reserve_accessed")
    ):
        raise RuntimeError("v60 manifest is not label-free and reserve-safe")
    if audit.get("manifest_sha256") != sha256(args.manifest):
        raise RuntimeError("v60 manifest changed after identity freeze")

    rows = [
        json.loads(line)
        for line in args.manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    trajectories = {str(row["trajectory_id"]) for row in rows}
    routes = {str(row["route_id"]) for row in rows}
    if len(rows) != 44 or len(trajectories) != 11 or len(routes) != 11:
        raise RuntimeError("v60 bank must contain 11 routes x 4 states")
    if any(
        row.get("partition") != "fresh-outer"
        or row.get("protocol") != "small-fresh-development-v60-11route-44state"
        for row in rows
    ):
        raise RuntimeError("v60 small-fresh protocol changed")

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
        raise RuntimeError("candidate/manifest identity mismatch")
    if candidates.shape != (44, 17, 4, 2) or reference.shape != (44, 16, 4, 2):
        raise RuntimeError("v60 K17 shape changed")

    legacy_rows = [
        {"traj_name": row["trajectory_id"], "curr_time": int(row["current_t"])}
        for row in rows
    ]
    action_set = action_set_features(candidates, reference)
    motion = history_features(candidates, reference, legacy_rows, args.trajectory_root)
    raw_action = np.concatenate((action_set, motion), axis=-1).astype(np.float32)
    action_features = np.concatenate(
        (raw_action, raw_action - raw_action[:, :1]), axis=-1
    ).astype(np.float32)
    if action_features.shape != (44, 17, 242):
        raise RuntimeError("v60 action feature contract changed")

    write_npz(
        args.output,
        candidates=candidates,
        action_features=action_features,
        record_key=expected_record,
        family=np.asarray([f"recon::{row['route_id']}" for row in rows]),
        trajectory=np.asarray([str(row["trajectory_id"]) for row in rows]),
        route_group=np.asarray([str(row["route_id"]) for row in rows]),
        window_uid=expected_uid,
    )
    result = {
        "schema": "recon_small_fresh_k17_dense_source_v60b",
        "status": "label_free_source_frozen",
        "partition": "fresh-outer",
        "protocol": "small-fresh-development-v60-11route-44state",
        "states": 44,
        "trajectories": 11,
        "route_groups": 11,
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256(args.manifest),
        "manifest_audit": str(args.manifest_audit.resolve()),
        "manifest_audit_sha256": sha256(args.manifest_audit),
        "candidate": str(args.candidate.resolve()),
        "candidate_sha256": sha256(args.candidate),
        "output": str(args.output.resolve()),
        "output_sha256": sha256(args.output),
        "waypoint_spacing_m": 0.25,
        "past_motion_range": "t-3...t",
        "future_pose_targets_computed": False,
        "future_pose_targets_opened": False,
        "labels_in_output": False,
        "cem": False,
        "reserve_accessed": False,
    }
    write_json(output_audit, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
