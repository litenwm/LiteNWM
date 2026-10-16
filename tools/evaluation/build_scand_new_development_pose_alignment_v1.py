#!/usr/bin/env python3
"""Build frame-aligned odometry for the frozen new SCAND development pilot."""

from __future__ import annotations

import json
import math
from pathlib import Path
import pickle
import sys

import numpy as np
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from planning.proposal_tail_sidecar_v1 import sha256_file  # noqa: E402
from tools.planning.audit_local_scand_preprocessed_reuse_v1 import load_trajectory  # noqa: E402


BASE = ROOT / "outputs/cem_development_v1/scand_new_routefit10_v4_development_pilot_v2_strict_identity"
INGEST = ROOT / "data/proposal_tail_collection_v1/incoming/quarantine/official_scand_selective_v4_contract_compatible/local_ingest.result.json"
VALIDATION = BASE / "sealed_validation_identity_clear.manifest.windows.jsonl"


def yaw_from_quaternion(q) -> float:
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def nearest_indices(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    right = np.searchsorted(source, target, side="left")
    right = np.clip(right, 0, len(source) - 1)
    left = np.clip(right - 1, 0, len(source) - 1)
    choose_left = np.abs(target - source[left]) <= np.abs(source[right] - target)
    return np.where(choose_left, left, right)


def existing_comparison(trajectory: str, position: np.ndarray, yaw: np.ndarray) -> dict:
    path = Path("/home/rainbow/nwm/data/scand") / trajectory / "traj_data.pkl"
    if not path.is_file():
        return {"available": False}
    old = load_trajectory(path)
    old_position = np.asarray(old["position"], dtype=np.float64)
    old_yaw = np.asarray(old["yaw"], dtype=np.float64)
    candidates = []
    for offset in range(-4, 5):
        new_start, old_start = max(offset, 0), max(-offset, 0)
        count = min(len(position) - new_start, len(old_position) - old_start)
        if count <= 0:
            continue
        delta = position[new_start : new_start + count] - old_position[old_start : old_start + count]
        yaw_delta = np.angle(np.exp(1j * (yaw[new_start : new_start + count] - old_yaw[old_start : old_start + count])))
        candidates.append((float(np.sqrt(np.mean(delta * delta))), float(np.sqrt(np.mean(yaw_delta * yaw_delta))), offset, count))
    best = min(candidates)
    return {
        "available": True,
        "path": str(path),
        "path_sha256": sha256_file(path),
        "new_frames": len(position),
        "old_frames": len(old_position),
        "best_position_rmse_m": best[0],
        "best_yaw_rmse_rad": best[1],
        "best_new_minus_old_index_offset": best[2],
        "compared_frames": best[3],
    }


def main() -> None:
    candidate = BASE / "pilot_k17.npz"
    candidate_audit = BASE / "pilot_k17.audit.json"
    prereg = json.loads((BASE / "preregistration.json").read_text(encoding="utf-8"))
    if prereg["status"] != "frozen_before_candidate_generation_and_outcomes":
        raise RuntimeError("preregistration status changed")
    if not candidate.is_file() or not candidate_audit.is_file():
        raise RuntimeError("K17 candidate must be frozen before pose access")
    rows = [json.loads(line) for line in (BASE / "pilot_manifest.windows.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    ingest = json.loads(INGEST.read_text(encoding="utf-8"))
    by_bag = {row["filename"].removesuffix(".bag"): row for row in ingest["bags"]}
    trajectories = sorted({row["trajectory_id"] for row in rows})
    output = BASE / "trajectory_state_root"
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    typestore = get_typestore(Stores.ROS1_NOETIC)
    audits = []
    for trajectory in trajectories:
        selected_rows = [row for row in rows if row["trajectory_id"] == trajectory]
        bag_id = selected_rows[0]["bag_id"]
        filename_stem = bag_id.removeprefix("random_mdps_")
        ingest_row = by_bag[filename_stem]
        bag = Path("/mnt/d/VEDIOS") / ingest_row["filename"]
        metadata = json.loads((Path(ingest_row["processed_route"]) / "route_metadata.json").read_text(encoding="utf-8"))
        frame_timestamps = np.asarray([int(frame["timestamp"]) for frame in metadata["frames"]], dtype=np.int64)
        with AnyReader([bag], default_typestore=typestore) as reader:
            names = {connection.topic for connection in reader.connections}
            odom_topic = "/odom" if "/odom" in names else "/jackal_velocity_controller/odom"
            connections = [connection for connection in reader.connections if connection.topic == odom_topic]
            odom_timestamps, positions, yaws = [], [], []
            for connection, timestamp, rawdata in reader.messages(connections=connections):
                message = reader.deserialize(rawdata, connection.msgtype)
                pose = message.pose.pose
                odom_timestamps.append(int(timestamp))
                positions.append((float(pose.position.x), float(pose.position.y)))
                yaws.append(yaw_from_quaternion(pose.orientation))
        odom_timestamps = np.asarray(odom_timestamps, dtype=np.int64)
        positions = np.asarray(positions, dtype=np.float64)
        yaws = np.asarray(yaws, dtype=np.float64)
        if len(odom_timestamps) < 2 or not np.all(np.diff(odom_timestamps) >= 0):
            raise RuntimeError(f"bad odometry timestamps: {trajectory}")
        index = nearest_indices(odom_timestamps, frame_timestamps)
        aligned_position = positions[index]
        aligned_yaw = yaws[index]
        offsets = np.abs(odom_timestamps[index] - frame_timestamps)
        if float(np.quantile(offsets, 0.99)) > 0.15e9:
            raise RuntimeError(f"camera/odom p99 offset exceeds 150ms: {trajectory}")
        directory = output / trajectory
        directory.mkdir()
        payload = {
            "position": aligned_position,
            "yaw": aligned_yaw,
            "timestamps_ns": frame_timestamps,
            "odom_timestamps_ns": odom_timestamps[index],
        }
        path = directory / "traj_data.pkl"
        with path.open("wb") as handle:
            pickle.dump(payload, handle, protocol=4)
        audits.append({
            "trajectory_id": trajectory,
            "route_group": selected_rows[0]["route_group"],
            "bag": str(bag),
            "bag_sha256": ingest_row["source_bag_sha256"],
            "camera_topic": metadata["camera_preprocessing"]["camera_topic"],
            "odom_topic": odom_topic,
            "frames": len(frame_timestamps),
            "odom_messages": len(odom_timestamps),
            "alignment_abs_ms": {"mean": float(offsets.mean() / 1e6), "p90": float(np.quantile(offsets, 0.9) / 1e6), "p99": float(np.quantile(offsets, 0.99) / 1e6), "max": float(offsets.max() / 1e6)},
            "existing_preprocessed_comparison": existing_comparison(trajectory, aligned_position, aligned_yaw),
            "traj_data": str(path),
            "traj_data_sha256": sha256_file(path),
        })
    audit = {
        "schema": "scand_new_development_pose_alignment_v1",
        "status": "complete_after_k17_freeze",
        "candidate_sha256": sha256_file(candidate),
        "candidate_audit_sha256": sha256_file(candidate_audit),
        "manifest_sha256": sha256_file(BASE / "pilot_manifest.windows.jsonl"),
        "trajectories": audits,
        "alignment_rule": "nearest raw odometry bag timestamp to each frozen 4Hz camera timestamp",
        "pose_values_opened_scope": "four development trajectories only",
        "sealed_validation_manifest_sha256": sha256_file(VALIDATION),
        "validation_pose_or_outcomes_opened": False,
        "reserve_accessed": False,
    }
    audit_path = BASE / "pose_alignment.audit.json"
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": audit["status"], "trajectories": audits, "audit_sha256": sha256_file(audit_path)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
