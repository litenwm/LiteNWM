#!/usr/bin/env python3
"""Build a label-free dense-meta source for the frozen strict SCAND100 K17 pool."""

from __future__ import annotations

import argparse
import hashlib
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


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate",
        type=Path,
        default=REPO_ROOT
        / "outputs/scand_strict_h4_100_three_arm_v1/candidates/frozen_k17.npz",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPO_ROOT
        / "data/evaluation/scand/strict_test_h4_100_seed17/manifest.json",
    )
    parser.add_argument(
        "--trajectory-root", type=Path, default=Path("/home/rainbow/nwm/data/scand")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(".audit.json").exists():
        raise FileExistsError(args.output)
    records = list(json.loads(args.manifest.read_text())["records"])
    with np.load(args.candidate, allow_pickle=False) as payload:
        candidates = payload["candidates_waypoint_units"].astype(np.float32)
        reference = payload["nomad_x16_waypoint_units"].astype(np.float32)
        observed_eval_index = payload["eval_index"].astype(np.int64)
    expected_eval_index = np.asarray(
        [int(row["eval_index"]) for row in records], dtype=np.int64
    )
    if candidates.shape != (100, 17, 4, 2) or reference.shape != (100, 16, 4, 2):
        raise RuntimeError("strict SCAND100 K17 candidate shape changed")
    if not np.array_equal(observed_eval_index, expected_eval_index):
        raise RuntimeError("candidate/manifest order mismatch")

    # The shared feature builder's legacy input unit is a 0.25 m RECON waypoint.
    # Scaling SCAND coordinates by 0.38/0.25 before that builder is algebraically
    # identical to using the 0.38 m SCAND spacing in every metric/action term.
    recon_unit_equivalent = np.float32(0.38 / 0.25)
    feature_candidates = candidates * recon_unit_equivalent
    feature_reference = reference * recon_unit_equivalent
    adapted_records = [
        {"traj_name": row["trajectory_id"], "curr_time": int(row["curr_time"])}
        for row in records
    ]
    action_set = action_set_features(feature_candidates, feature_reference)
    motion = history_features(
        feature_candidates, feature_reference, adapted_records, args.trajectory_root
    )
    raw = np.concatenate((action_set, motion), axis=-1).astype(np.float32)
    action = np.concatenate((raw, raw - raw[:, :1]), axis=-1).astype(np.float32)
    if action.shape != (100, 17, 242):
        raise RuntimeError(f"action feature contract changed: {action.shape}")
    record_key = np.asarray(
        [f"{row['trajectory_id']}:{int(row['curr_time'])}" for row in records]
    )
    family = np.asarray([str(row["trajectory_id"]) for row in records])
    write_npz(
        args.output,
        candidates=feature_candidates,
        candidates_scand_waypoint_units=candidates,
        action_features=action,
        record_key=record_key,
        family=family,
        eval_index=expected_eval_index,
    )
    audit = {
        "schema": "scand100_k17_dense_source_v1",
        "candidate_pool": str(args.candidate.resolve()),
        "candidate_pool_sha256": sha256(args.candidate),
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256(args.manifest),
        "output": str(args.output.resolve()),
        "output_sha256": sha256(args.output),
        "states": 100,
        "candidates": 17,
        "waypoint_spacing_m": 0.38,
        "legacy_feature_unit_bridge": "multiply waypoint units by 0.38/0.25",
        "candidates_field_units": "legacy 0.25m units; multiply by 0.25 for meters",
        "candidates_scand_waypoint_units_field": True,
        "pose_input_scope": "history t-3..t only",
        "future_pose_indices_accessed": False,
        "navigation_eval_deserialized": False,
        "strict_test_outcomes_opened": False,
        "fixed_pool_only": True,
        "cem": False,
    }
    args.output.with_suffix(".audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
