#!/usr/bin/env python3
"""Corrected frame/odometry alignment for the frozen development pilot."""

from __future__ import annotations

import json
from pathlib import Path
import pickle
import sys

import numpy as np
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from planning.proposal_tail_sidecar_v1 import sha256_file  # noqa: E402
from tools.evaluation.build_scand_new_development_pose_alignment_v1 import (  # noqa: E402
    existing_comparison,
    nearest_indices,
    yaw_from_quaternion,
)


BASE = ROOT / "outputs/cem_development_v1/scand_new_routefit10_v4_development_pilot_v2_strict_identity"
INGEST = ROOT / "data/proposal_tail_collection_v1/incoming/quarantine/official_scand_selective_v4_contract_compatible/local_ingest.result.json"
PROCESSED = ROOT / "data/proposal_tail_collection_v1/incoming/quarantine/official_scand_selective_v4_contract_compatible/processed"


def main() -> None:
    candidate = BASE / "pilot_k17.npz"
    candidate_audit = BASE / "pilot_k17.audit.json"
    if not candidate.is_file() or not candidate_audit.is_file():
        raise RuntimeError("K17 must be frozen first")
    rows = [json.loads(line) for line in (BASE / "pilot_manifest.windows.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    ingest = json.loads(INGEST.read_text(encoding="utf-8"))
    by_stem = {row["filename"].removesuffix(".bag"): row for row in ingest["bags"]}
    output = BASE / "trajectory_state_root_v2"
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    typestore = get_typestore(Stores.ROS1_NOETIC)
    audits = []
    for trajectory in sorted({row["trajectory_id"] for row in rows}):
        route_rows = [row for row in rows if row["trajectory_id"] == trajectory]
        bag_id = route_rows[0]["bag_id"]
        stem = bag_id.removeprefix("random_mdps_")
        ingest_row = by_stem[stem]
        bag = Path("/mnt/d/VEDIOS") / ingest_row["filename"]
        route_dir = PROCESSED / ingest_row["cohort_role"] / bag_id
        metadata = json.loads((route_dir / "route_metadata.json").read_text(encoding="utf-8"))
        frame_time = np.asarray([int(frame["timestamp"]) for frame in metadata["frames"]], dtype=np.int64)
        with AnyReader([bag], default_typestore=typestore) as reader:
            names = {connection.topic for connection in reader.connections}
            topic = "/odom" if "/odom" in names else "/jackal_velocity_controller/odom"
            connections = [connection for connection in reader.connections if connection.topic == topic]
            times, positions, yaws = [], [], []
            for connection, timestamp, rawdata in reader.messages(connections=connections):
                message = reader.deserialize(rawdata, connection.msgtype)
                pose = message.pose.pose
                times.append(int(timestamp))
                positions.append((float(pose.position.x), float(pose.position.y)))
                yaws.append(yaw_from_quaternion(pose.orientation))
        times = np.asarray(times, dtype=np.int64)
        positions = np.asarray(positions, dtype=np.float64)
        yaws = np.asarray(yaws, dtype=np.float64)
        if len(times) < 2 or not np.all(np.diff(times) >= 0):
            raise RuntimeError(f"bad odometry timestamps: {trajectory}")
        index = nearest_indices(times, frame_time)
        position, yaw = positions[index], yaws[index]
        offset = np.abs(times[index] - frame_time)
        if float(np.quantile(offset, 0.99)) > 0.15e9:
            raise RuntimeError(f"camera/odom p99 offset exceeds 150ms: {trajectory}")
        directory = output / trajectory
        directory.mkdir()
        path = directory / "traj_data.pkl"
        with path.open("wb") as handle:
            pickle.dump({"position": position, "yaw": yaw, "timestamps_ns": frame_time, "odom_timestamps_ns": times[index]}, handle, protocol=4)
        audits.append({
            "trajectory_id": trajectory,
            "route_group": route_rows[0]["route_group"],
            "bag": str(bag),
            "bag_sha256": ingest_row["source_bag_sha256"],
            "camera_topic": metadata["camera_preprocessing"]["camera_topic"],
            "odom_topic": topic,
            "frames": len(frame_time),
            "odom_messages": len(times),
            "alignment_abs_ms": {"mean": float(offset.mean()/1e6), "p90": float(np.quantile(offset,.9)/1e6), "p99": float(np.quantile(offset,.99)/1e6), "max": float(offset.max()/1e6)},
            "existing_preprocessed_comparison": existing_comparison(trajectory, position, yaw),
            "traj_data": str(path),
            "traj_data_sha256": sha256_file(path),
        })
    audit = {
        "schema": "scand_new_development_pose_alignment_v2",
        "status": "complete_after_k17_freeze",
        "implementation_correction": "processed route path derived from frozen cohort_role and bag_id",
        "candidate_sha256": sha256_file(candidate),
        "candidate_audit_sha256": sha256_file(candidate_audit),
        "manifest_sha256": sha256_file(BASE / "pilot_manifest.windows.jsonl"),
        "trajectories": audits,
        "alignment_rule": "nearest raw odometry bag timestamp to frozen 4Hz camera timestamp",
        "pose_values_opened_scope": "four development trajectories only",
        "sealed_validation_manifest_sha256": sha256_file(BASE / "sealed_validation_identity_clear.manifest.windows.jsonl"),
        "validation_pose_or_outcomes_opened": False,
        "reserve_accessed": False,
    }
    audit_path = BASE / "pose_alignment_v2.audit.json"
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": audit["status"], "trajectories": audits, "audit_sha256": sha256_file(audit_path)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
