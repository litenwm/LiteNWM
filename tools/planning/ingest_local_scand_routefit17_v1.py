#!/usr/bin/env python3
"""Ingest the frozen routefit17 SCAND bags from a local directory.

No network access is performed and no odometry, truth, or outcome values are
opened. Official byte counts and MD5 values are verified before the existing
camera-only processor and identity/readiness checks are reused.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from planning.proposal_tail_sidecar_v1 import canonical_json, sha256_file  # noqa: E402
from tools.planning import download_prepare_new_scand_bags_v1 as v1  # noqa: E402


SCHEMA = "ingest_local_scand_routefit17_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=Path("/mnt/d/VEDIOS"))
    parser.add_argument(
        "--plan",
        type=Path,
        default=ROOT
        / "data/proposal_tail_collection_v1/incoming/quarantine/official_scand_selective_v1/download_plan.routefit_v3.json",
    )
    parser.add_argument(
        "--download-manifest",
        type=Path,
        default=ROOT
        / "data/proposal_tail_collection_v1/incoming/quarantine/official_scand_selective_v1/scand_routefit17_download_manifest.csv",
    )
    parser.add_argument(
        "--used-registry",
        type=Path,
        default=ROOT / "outputs/proposal_tail_collection_v1/used_v70_v77_identities.json",
    )
    parser.add_argument(
        "--forbidden-registry",
        type=Path,
        default=ROOT / "data/proposal_tail_collection_v1/incoming/forbidden_identity_hashes.json",
    )
    parser.add_argument(
        "--incoming-root",
        type=Path,
        default=ROOT / "data/proposal_tail_collection_v1/incoming",
    )
    return parser.parse_args()


def md5_file(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_download_manifest(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    result = {row["filename"]: row for row in rows}
    v1.require(len(rows) == 17 and len(result) == 17, "download manifest must contain 17 unique bags")
    return result


def topic_inventory(path: Path) -> dict[str, Any]:
    try:
        from rosbags.highlevel import AnyReader
        from rosbags.typesys import Stores, get_typestore
    except ImportError as error:
        raise RuntimeError("nwm environment is missing rosbags") from error
    typestore = get_typestore(Stores.ROS1_NOETIC)
    with AnyReader([path], default_typestore=typestore) as reader:
        connections = list(reader.connections)
        camera_topic = v1.choose_image_topic(connections)
        topics = [
            {
                "topic": connection.topic,
                "msgtype": connection.msgtype,
                "message_count": int(getattr(connection, "msgcount", 0)),
            }
            for connection in connections
        ]
    lower = [(row["topic"].lower(), row) for row in topics]
    motion = [row for name, row in lower if any(token in name for token in ("cmd_vel", "velocity", "twist"))]
    odometry = [row for name, row in lower if any(token in name for token in ("odom", "odometry", "pose"))]
    return {
        "camera_topic_selected": camera_topic,
        "topics": topics,
        "motion_topic_metadata": motion,
        "odometry_or_pose_topic_metadata": odometry,
        "message_values_opened": False,
    }


def write_frozen(path: Path, value: dict[str, Any]) -> None:
    v1.write_text_exact(path, canonical_json(value) + "\n")


def main() -> None:
    args = parse_args()
    started = time.time()
    plan = v1.read_json(args.plan)
    v1.require(plan.get("schema") == "download_prepare_new_scand_bags_v6_plan", "unexpected plan schema")
    v1.require(len(plan.get("selected", [])) == 17, "frozen plan does not contain 17 bags")
    official = read_download_manifest(args.download_manifest)
    forbidden, forbidden_raw = v1.load_registry(args.forbidden_registry, forbidden=True)
    _, used_raw = v1.load_registry(args.used_registry, forbidden=False)
    identity_audit = plan["identity_audit"]
    v1.require(sha256_file(args.forbidden_registry) == identity_audit["forbidden_registry_sha256"], "forbidden registry changed")
    v1.require(sha256_file(args.used_registry) == identity_audit["used_registry_sha256"], "used registry changed")

    quarantine = args.incoming_root / "quarantine/official_scand_selective_v1"
    prereg_path = quarantine / "local_ingest.preregistration.json"
    prereg = {
        "schema": SCHEMA + "_preregistration",
        "status": "frozen_before_bag_content_processing",
        "source_dir": str(args.source_dir.resolve()),
        "plan": str(args.plan.resolve()),
        "plan_sha256": sha256_file(args.plan),
        "download_manifest_sha256": sha256_file(args.download_manifest),
        "selected_bags": 17,
        "development_routes": 12,
        "one_time_validation_routes": 5,
        "processing": "camera-only 4Hz, center-crop 4:3, 320x240 JPEG quality 95",
        "validation_outcomes_opened": False,
        "reserve_content_accessed": False,
        "network_allowed": False,
    }
    write_frozen(prereg_path, prereg)

    cohort_rows: dict[str, list[dict[str, Any]]] = {"development": [], "one_time_validation": []}
    bag_audit = []
    for position, selected in enumerate(plan["selected"], start=1):
        filename = selected["official_file_name"] + ".bag"
        source = args.source_dir / filename
        v1.require(source.is_file(), f"missing downloaded bag: {source}")
        expected = official.get(filename)
        v1.require(expected is not None, f"bag absent from official checksum manifest: {filename}")
        v1.require(source.stat().st_size == int(expected["bytes"]), f"size mismatch: {filename}")
        observed_md5 = md5_file(source)
        v1.require(observed_md5 == expected["md5"], f"official MD5 mismatch: {filename}")
        topics = topic_inventory(source)
        row = {**selected, "source_official_index_sha256": identity_audit["official_index_sha256"]}
        destination = quarantine / "processed" / selected["cohort_role"] / selected["bag_id"]
        processed = v1.process_camera_only(source, destination, row, forbidden)
        cohort_rows[selected["cohort_role"]].extend(processed["manifest_rows"])
        bag_audit.append(
            {
                "position": position,
                "filename": filename,
                "bag_id": selected["bag_id"],
                "route_group": selected["route_group"],
                "cohort_role": selected["cohort_role"],
                "bytes": source.stat().st_size,
                "official_md5": observed_md5,
                "source_path": str(source.resolve()),
                "processed_route": str(destination),
                "processed_frames": len(processed["metadata"]["frames"]),
                "processed_windows": len(processed["manifest_rows"]),
                "source_bag_sha256": processed["metadata"]["source_bag_sha256"],
                "camera_contract": processed["metadata"]["camera_preprocessing"],
                "topic_metadata_only": topics,
                "reused_processing": processed["reused"],
                "outcome_values_opened": False,
            }
        )
        print(canonical_json({"status": "processed", "position": position, "total": 17, "filename": filename, "role": selected["cohort_role"], "frames": len(processed["metadata"]["frames"])}), flush=True)

    staging = quarantine / "staging_manifests"
    development_staging = staging / "development.manifest.windows.jsonl"
    validation_staging = staging / "one_time_validation.manifest.windows.jsonl"
    v1.write_text_exact(development_staging, v1.manifest_content(cohort_rows["development"]))
    v1.write_text_exact(validation_staging, v1.manifest_content(cohort_rows["one_time_validation"]))
    readiness = v1.evaluate_readiness(development_staging, validation_staging, args.used_registry, args.forbidden_registry)
    v1.write_json_atomic(quarantine / "readiness.prepublish.json", readiness)
    v1.require(readiness["ready"], f"prepublish readiness failed: {readiness['errors']}")

    development_final = args.incoming_root / "development_v1/manifest.windows.jsonl"
    validation_final = args.incoming_root / "one_time_validation_v1/manifest.windows.jsonl"
    v1.write_text_exact(development_final, development_staging.read_text(encoding="utf-8"))
    v1.write_text_exact(validation_final, validation_staging.read_text(encoding="utf-8"))
    final_readiness = v1.evaluate_readiness(development_final, validation_final, args.used_registry, args.forbidden_registry)
    final_readiness_path = args.incoming_root / "readiness.new_scand_bags_v1.json"
    v1.write_json_atomic(final_readiness_path, final_readiness)
    v1.require(final_readiness["ready"], f"final readiness failed: {final_readiness['errors']}")
    result = {
        "schema": SCHEMA + "_result",
        "status": "complete_prelabel_ready",
        "preregistration": str(prereg_path),
        "preregistration_sha256": sha256_file(prereg_path),
        "plan_sha256": sha256_file(args.plan),
        "bags": bag_audit,
        "development": {"routes": 12, "windows": len(cohort_rows["development"]), "manifest": str(development_final), "manifest_sha256": sha256_file(development_final)},
        "one_time_validation": {"routes": 5, "windows": len(cohort_rows["one_time_validation"]), "manifest": str(validation_final), "manifest_sha256": sha256_file(validation_final), "labels_opened": False},
        "readiness": final_readiness,
        "readiness_path": str(final_readiness_path),
        "readiness_sha256": sha256_file(final_readiness_path),
        "used_registry_status": used_raw.get("status"),
        "forbidden_registry_status": forbidden_raw.get("status"),
        "elapsed_seconds": time.time() - started,
        "network_accessed": False,
        "odometry_or_truth_values_opened": False,
        "reserve_content_accessed": False,
        "training_started": False,
    }
    result_path = quarantine / "local_ingest.result.json"
    v1.write_json_atomic(result_path, result)
    print(canonical_json({**result, "result_path": str(result_path), "result_sha256": sha256_file(result_path)}))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(canonical_json({"schema": SCHEMA + "_failure", "status": "fail_closed", "error_type": type(error).__name__, "error": str(error), "network_accessed": False, "validation_outcomes_opened": False, "reserve_content_accessed": False}), file=sys.stderr)
        raise SystemExit(2) from error
