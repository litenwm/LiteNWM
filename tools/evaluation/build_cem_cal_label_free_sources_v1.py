#!/usr/bin/env python3
"""Build label-free feature sources for frozen cem-cal development K17 pools."""

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
from tools.scoring.train_recon_dense_meta_dinov3_v1 import (  # noqa: E402
    file_sha256,
    write_json,
)


ROOT = REPO_ROOT / "outputs/cem_development_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", choices=("recon", "scand"), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--trajectory-root",
        type=Path,
        default=None,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit_path = args.output.with_suffix(".audit.json")
    if args.output.exists() or audit_path.exists():
        raise FileExistsError(args.output)
    root = args.trajectory_root
    if root is None:
        root = Path(f"/home/rainbow/nwm/data/{args.domain}")
    rows = [
        json.loads(line)
        for line in args.manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != 100 or any(
        row.get("partition") != "cem-cal" for row in rows
    ):
        raise RuntimeError("cem-cal development manifest contract failed")
    if any(str(row.get("dataset")) != args.domain for row in rows):
        raise RuntimeError("cem-cal domain identity mismatch")
    with np.load(args.candidate, allow_pickle=False) as payload:
        candidates_native = payload["candidates_waypoint_units"].astype(
            np.float32
        )
        reference_native = payload["nomad_x16_waypoint_units"].astype(
            np.float32
        )
        record_key = payload["record_key"].astype(str)
        window_uid = payload["window_uid"].astype(str)
    expected_record = np.asarray(
        [f"{row['trajectory_id']}:{int(row['current_t'])}" for row in rows]
    )
    expected_uid = np.asarray([str(row["window_uid"]) for row in rows])
    if not np.array_equal(record_key, expected_record):
        raise RuntimeError("cem-cal candidate record identity mismatch")
    if not np.array_equal(window_uid, expected_uid):
        raise RuntimeError("cem-cal candidate UID identity mismatch")
    bridge = np.float32(1.0 if args.domain == "recon" else 0.38 / 0.25)
    candidates = candidates_native * bridge
    reference = reference_native * bridge
    legacy_rows = [
        {
            "traj_name": row["trajectory_id"],
            "curr_time": int(row["current_t"]),
        }
        for row in rows
    ]
    action_set = action_set_features(candidates, reference)
    motion = history_features(candidates, reference, legacy_rows, root)
    raw_action = np.concatenate((action_set, motion), axis=-1).astype(
        np.float32
    )
    action_features = np.concatenate(
        (raw_action, raw_action - raw_action[:, :1]), axis=-1
    ).astype(np.float32)
    if action_features.shape != (100, 17, 242):
        raise RuntimeError("cem-cal action feature contract changed")
    route = np.asarray(
        [
            str(row.get("route_group", row.get("route_id")))
            for row in rows
        ]
    )
    payload = {
        "candidates": candidates,
        "action_features": action_features,
        "record_key": expected_record,
        "family": np.asarray(
            [f"{args.domain}::{value}" for value in route]
        ),
        "trajectory": np.asarray(
            [str(row["trajectory_id"]) for row in rows]
        ),
        "route_group": route,
        "window_uid": expected_uid,
    }
    if args.domain == "scand":
        payload["candidates_scand_waypoint_units"] = candidates_native
        payload["bag_id"] = np.asarray(
            [str(row["bag_id"]) for row in rows]
        )
    write_npz(args.output, **payload)
    report = {
        "schema": "cem_cal_label_free_dense_source_v1",
        "domain": args.domain,
        "status": "label_free_source_frozen",
        "manifest": str(args.manifest),
        "manifest_sha256": file_sha256(args.manifest),
        "candidate": str(args.candidate),
        "candidate_sha256": file_sha256(args.candidate),
        "output": str(args.output),
        "output_sha256": file_sha256(args.output),
        "states": 100,
        "waypoint_spacing_m": 0.25 if args.domain == "recon" else 0.38,
        "legacy_feature_unit_bridge": float(bridge),
        "labels_opened": False,
        "labels_in_output": False,
        "outcomes_opened": False,
        "fixed_pool_only": True,
        "cem": False,
    }
    write_json(audit_path, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
