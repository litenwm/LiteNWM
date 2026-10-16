#!/usr/bin/env python3
"""Build the label-free dense feature source for fresh RECON H4-100."""

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
from tools.planning.build_scand100_nomad_k17_v1 import (  # noqa: E402
    sha256,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--manifest-audit", type=Path, required=True
    )
    parser.add_argument(
        "--trajectory-root", type=Path, default=Path("/home/rainbow/nwm/data/recon")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit_path = args.output.with_suffix(".audit.json")
    if args.output.exists() or audit_path.exists():
        raise FileExistsError(args.output)
    manifest_audit = json.loads(args.manifest_audit.read_text(encoding="utf-8"))
    if (
        manifest_audit.get("status") != "manifest_frozen_rgb_only"
        or manifest_audit.get("labels_opened")
        or manifest_audit.get("future_pose_opened")
        or manifest_audit.get("selected_trajectory_overlap")
        or manifest_audit.get("selected_route_overlap")
    ):
        raise RuntimeError("fresh manifest provenance is not label-free/disjoint")
    if manifest_audit.get("manifest_sha256") != sha256(args.manifest):
        raise RuntimeError("fresh manifest changed after identity freeze")

    rows = [
        json.loads(line)
        for line in args.manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != 100 or any(
        row.get("partition") != "fresh-outer" for row in rows
    ):
        raise RuntimeError("fresh-outer H4-100 manifest contract failed")
    trajectories = {str(row["trajectory_id"]) for row in rows}
    routes = {str(row["route_id"]) for row in rows}
    if len(trajectories) != 25 or len(routes) != 25:
        raise RuntimeError("fresh-outer trajectory/route count changed")

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

    legacy_rows = [
        {"traj_name": row["trajectory_id"], "curr_time": int(row["current_t"])}
        for row in rows
    ]
    action_set = action_set_features(candidates, reference)
    # This uses only t-3...t past robot motion; t+1...t+4 outcomes remain sealed.
    motion = history_features(candidates, reference, legacy_rows, args.trajectory_root)
    raw_action = np.concatenate((action_set, motion), axis=-1).astype(np.float32)
    action_features = np.concatenate(
        (raw_action, raw_action - raw_action[:, :1]), axis=-1
    ).astype(np.float32)
    if action_features.shape != (100, 17, 242):
        raise RuntimeError("fresh-outer action feature contract changed")

    family = np.asarray([f"recon::{row['route_id']}" for row in rows])
    write_npz(
        args.output,
        candidates=candidates,
        action_features=action_features,
        record_key=expected_record,
        family=family,
        trajectory=np.asarray([str(row["trajectory_id"]) for row in rows]),
        route_group=np.asarray([str(row["route_id"]) for row in rows]),
        window_uid=expected_uid,
    )
    audit = {
        "schema": "recon_fresh_outer_k17_dense_source_v2",
        "partition": "fresh-outer",
        "states": 100,
        "trajectories": len(trajectories),
        "route_groups": len(routes),
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256(args.manifest),
        "manifest_audit": str(args.manifest_audit.resolve()),
        "manifest_audit_sha256": sha256(args.manifest_audit),
        "candidate": str(args.candidate.resolve()),
        "candidate_sha256": sha256(args.candidate),
        "output": str(args.output.resolve()),
        "output_sha256": sha256(args.output),
        "waypoint_spacing_m": 0.25,
        "past_motion_features_used": True,
        "past_motion_range": "t-3...t",
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
