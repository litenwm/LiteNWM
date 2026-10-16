#!/usr/bin/env python3
"""Build the causal, label-free fixed-K17 source for the old-train pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

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
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--trajectory-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit_path = args.output.with_suffix(".audit.json")
    if args.output.exists() or audit_path.exists():
        raise FileExistsError(args.output)
    prereg = json.loads(args.preregistration.read_text(encoding="utf-8"))
    if prereg.get("status") != "frozen_before_candidate_generation_and_outcomes":
        raise RuntimeError("pilot preregistration is not frozen")
    if sha256(args.manifest) != prereg["inputs"]["manifest"]["sha256"]:
        raise RuntimeError("pilot manifest changed after preregistration")
    rows = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    with np.load(args.candidate, allow_pickle=False) as payload:
        candidates_native = payload["candidates_waypoint_units"].astype(np.float32)
        reference_native = payload["nomad_x16_waypoint_units"].astype(np.float32)
        record = payload["record_key"].astype(str)
    expected = np.asarray([f"{row['trajectory_id']}:{int(row['current_t'])}" for row in rows])
    if not np.array_equal(record, expected):
        raise RuntimeError("candidate/manifest identity mismatch")
    bridge = np.float32(0.38 / 0.25)
    candidates = candidates_native * bridge
    reference = reference_native * bridge
    legacy_rows = [
        {"traj_name": row["trajectory_id"], "curr_time": int(row["current_t"])}
        for row in rows
    ]
    action_set = action_set_features(candidates, reference)
    # The frozen feature implementation reads only t-3..t from position/yaw;
    # t+1..t+4 are not sliced, copied, or returned by this stage.
    motion = history_features(candidates, reference, legacy_rows, args.trajectory_root)
    raw = np.concatenate((action_set, motion), axis=-1).astype(np.float32)
    action = np.concatenate((raw, raw - raw[:, :1]), axis=-1).astype(np.float32)
    if action.shape != (len(rows), 17, 242):
        raise RuntimeError(f"pilot action feature shape changed: {action.shape}")
    family = np.asarray([f"scand::{row['route_group']}::bag::{row['bag_id']}" for row in rows])
    write_npz(
        args.output,
        candidates=candidates,
        action_features=action,
        record_key=expected,
        family=family,
        trajectory=np.asarray([row["trajectory_id"] for row in rows]),
        bag_id=np.asarray([row["bag_id"] for row in rows]),
        route_group=np.asarray([row["route_group"] for row in rows]),
        window_uid=np.asarray([row["window_uid"] for row in rows]),
    )
    audit = {
        "schema": "scand_existing_train_pilot_label_free_source_v1",
        "status": "causal_source_frozen_before_offline_outcome_builder",
        "states": len(rows),
        "candidate_shape": list(candidates.shape),
        "action_feature_shape": list(action.shape),
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256(args.manifest),
        "candidate": str(args.candidate.resolve()),
        "candidate_sha256": sha256(args.candidate),
        "preregistration": str(args.preregistration.resolve()),
        "preregistration_sha256": sha256(args.preregistration),
        "output": str(args.output.resolve()),
        "output_sha256": sha256(args.output),
        "traj_data_pkl_opened_after_k17_freeze_for_causal_history": True,
        "history_slice": "position/yaw t-3..t only",
        "future_pose_values_used": False,
        "ground_truth_local_m_materialized": False,
        "targets_materialized": False,
        "reserve_accessed": False,
    }
    write_json(audit_path, audit)
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
