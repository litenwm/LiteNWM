#!/usr/bin/env python3
"""Build dense DINOv3 source features for labeled SCAND inner K17 pools."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import pickle
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tools.evaluation.build_recon100_k17_source_v1 import (  # noqa: E402
    action_set_features,
    history_features,
    metric_labels,
    write_npz,
)
from tools.planning.build_scand100_nomad_k17_v1 import sha256, write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partition", choices=("train-fit", "scorer-cal"), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--spent-manifest",
        type=Path,
        default=REPO_ROOT
        / "data/evaluation/scand/strict_test_h4_100_seed17/manifest.json",
    )
    parser.add_argument(
        "--trajectory-root", type=Path, default=Path("/home/rainbow/nwm/data/scand")
    )
    return parser.parse_args()


def local_ground_truth(path: Path, timestep: int) -> np.ndarray:
    with path.open("rb") as handle:
        raw = pickle.load(handle)
    position = np.asarray(raw["position"], dtype=np.float64)[timestep : timestep + 5, :2]
    yaw = np.asarray(raw["yaw"], dtype=np.float64).reshape(-1)[timestep : timestep + 5]
    if position.shape != (5, 2) or yaw.shape != (5,):
        raise RuntimeError(f"trajectory shorter than H4: {path}:{timestep}")
    angle = float(yaw[0])
    rotation = np.asarray(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]],
        dtype=np.float64,
    )
    return ((position - position[0]).dot(rotation))[1:].astype(np.float32)


def main() -> None:
    args = parse_args()
    audit_path = args.output.with_suffix(".audit.json")
    if args.output.exists() or audit_path.exists():
        raise FileExistsError(args.output)
    rows = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    if not rows or any(row.get("partition") != args.partition for row in rows):
        raise RuntimeError("manifest partition contract failed")
    spent = json.loads(args.spent_manifest.read_text())
    spent_trajectories = {str(row["trajectory_id"]) for row in spent["records"]}
    trajectories = {str(row["trajectory_id"]) for row in rows}
    if spent_trajectories.intersection(trajectories):
        raise RuntimeError("inner labeled source overlaps spent strict100 trajectories")

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
    if candidates.shape != (len(rows), 17, 4, 2) or reference.shape != (
        len(rows),
        16,
        4,
        2,
    ):
        raise RuntimeError("SCAND inner K17 shape changed")

    # The shared exporter uses the RECON 0.25 m adapter. Multiplying waypoint
    # coordinates by 0.38/0.25 makes every downstream metric/action exactly
    # equal to the native SCAND 0.38 m physical representation.
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
    if action_features.shape != (len(rows), 17, 242):
        raise RuntimeError("SCAND inner action feature contract changed")

    ground_truth = np.empty((len(rows), 4, 2), dtype=np.float32)
    for index, row in enumerate(rows):
        ground_truth[index] = local_ground_truth(
            args.trajectory_root / str(row["trajectory_id"]) / "traj_data.pkl",
            int(row["current_t"]),
        )
    labels = metric_labels(feature_candidates, ground_truth, 0.25, 0.4, 0.6)
    targets = labels["targets"]
    if targets.shape != (len(rows), 17, 8):
        raise RuntimeError("SCAND inner target contract changed")
    family = np.asarray([f"scand::{row['route_group']}" for row in rows])
    write_npz(
        args.output,
        candidates=feature_candidates,
        candidates_scand_waypoint_units=candidates,
        action_features=action_features,
        targets=targets,
        ground_truth_local_m=ground_truth,
        record_key=expected_record,
        family=family,
        trajectory=np.asarray([str(row["trajectory_id"]) for row in rows]),
        route_group=np.asarray([str(row["route_group"]) for row in rows]),
        bag_id=np.asarray([str(row["bag_id"]) for row in rows]),
        window_uid=expected_uid,
    )
    audit = {
        "schema": "scand_inner_k17_dense_source_v1",
        "partition": args.partition,
        "states": len(rows),
        "trajectories": len(trajectories),
        "route_groups": len(set(family)),
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256(args.manifest),
        "candidate": str(args.candidate.resolve()),
        "candidate_sha256": sha256(args.candidate),
        "output": str(args.output.resolve()),
        "output_sha256": sha256(args.output),
        "waypoint_spacing_m": 0.38,
        "legacy_feature_unit_bridge": "multiply waypoint units by 0.38/0.25",
        "objective": "0.4 * RMS ATE + 0.6 * endpoint",
        "labels_opened": True,
        "labels_scope": f"SCAND inner {args.partition} only",
        "spent_strict100_trajectory_overlap": 0,
        "strict100_outcomes_used": False,
        "cem_cal_accessed": False,
        "internal_holdout_accessed": False,
        "fixed_pool_only": True,
        "cem": False,
    }
    write_json(audit_path, audit)
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
