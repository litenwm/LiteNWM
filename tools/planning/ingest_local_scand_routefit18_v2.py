#!/usr/bin/env python3
"""Corrected local SCAND ingest with audited Jackal camera-topic support.

The earlier GDC exclusion is treated as invalid because the legacy camera
topic list omitted ``/left/image_color/compressed``.  This tool first audits
that raw topic and the exact 4 Hz/H4 eligibility rule without opening motion
or outcome values.  It restores the original frozen 18-route 12/6 split when
the corrected input passes, otherwise it falls back to the prelabel 17-route
input-eligibility plan.  Network and reserve access are forbidden.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from planning.proposal_tail_sidecar_v1 import canonical_json, sha256_file  # noqa: E402
from tools.planning import download_prepare_new_scand_bags_v1 as v1  # noqa: E402
from tools.planning import ingest_local_scand_routefit17_v1 as local_v1  # noqa: E402


SCHEMA = "ingest_local_scand_routefit18_v2"
JACKAL_CAMERA_TOPIC = "/left/image_color/compressed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=Path("/mnt/d/VEDIOS"))
    parser.add_argument(
        "--gdc-bag",
        type=Path,
        default=ROOT
        / "data/proposal_tail_collection_v1/incoming/quarantine/official_scand_selective_v1/bags/A_Jackal_GDC_GDC_Fri_Oct_29_11.bag",
    )
    parser.add_argument(
        "--plan18",
        type=Path,
        default=ROOT
        / "data/proposal_tail_collection_v1/incoming/quarantine/official_scand_selective_v1/download_plan.json",
    )
    parser.add_argument(
        "--plan17",
        type=Path,
        default=ROOT
        / "data/proposal_tail_collection_v1/incoming/quarantine/official_scand_selective_v1/download_plan.routefit_v3.json",
    )
    parser.add_argument(
        "--tdr-manifest",
        type=Path,
        default=ROOT / "configs/proposal_tail_collection/scand_tdr_selected_files_v1.json",
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


def gdc_camera_audit(path: Path) -> dict:
    try:
        from rosbags.highlevel import AnyReader
        from rosbags.typesys import Stores, get_typestore
    except ImportError as error:
        raise RuntimeError("nwm environment is missing rosbags") from error
    typestore = get_typestore(Stores.ROS1_NOETIC)
    with AnyReader([path], default_typestore=typestore) as reader:
        connections = list(reader.connections)
        selected = [connection for connection in connections if connection.topic == JACKAL_CAMERA_TOPIC]
        v1.require(selected, f"GDC lacks corrected Jackal camera topic: {JACKAL_CAMERA_TOPIC}")
        timestamps = [int(timestamp) for _, timestamp, _ in reader.messages(connections=selected)]
        bag_start = int(reader.start_time)
        bag_end = int(reader.end_time)
        topics = [
            {
                "topic": connection.topic,
                "msgtype": connection.msgtype,
                "message_count": int(getattr(connection, "msgcount", 0)),
            }
            for connection in connections
        ]
    v1.require(timestamps, "corrected GDC camera topic has no messages")
    v1.require(all(a <= b for a, b in zip(timestamps, timestamps[1:])), "GDC camera timestamps are nonmonotonic")
    sampled = []
    next_sample_ns = None
    step = int(round(1e9 / v1.SAMPLE_RATE_HZ))
    for timestamp in timestamps:
        if next_sample_ns is not None and timestamp < next_sample_ns:
            continue
        sampled.append(timestamp)
        next_sample_ns = timestamp + step
    eligible = len(sampled) - v1.GOAL_OFFSET_FRAMES - (v1.HISTORY_FRAMES - 1)
    return {
        "schema": "jackal_gdc_camera_topic_correction_audit_v1",
        "status": "old_short_route_conclusion_invalidated_and_recomputed",
        "bag": str(path.resolve()),
        "bag_sha256": sha256_file(path),
        "legacy_topic_omission": JACKAL_CAMERA_TOPIC,
        "legacy_observation_invalidated": {"sampled_frames": 36, "eligible_h4_windows": 29},
        "bag_start_ns": bag_start,
        "bag_end_ns": bag_end,
        "bag_duration_seconds": (bag_end - bag_start) / 1e9,
        "selected_camera_topic": JACKAL_CAMERA_TOPIC,
        "raw_camera_messages": len(timestamps),
        "camera_first_timestamp_ns": timestamps[0],
        "camera_last_timestamp_ns": timestamps[-1],
        "camera_duration_seconds": (timestamps[-1] - timestamps[0]) / 1e9,
        "sample_rule": "first message then timestamp >= previous_selected + 0.25s",
        "sampled_4hz_frames": len(sampled),
        "eligible_h4_windows": eligible,
        "required_h4_windows": 40,
        "input_eligible": eligible >= 40,
        "topics": topics,
        "message_values_opened": False,
        "outcomes_or_labels_opened": False,
        "reserve_content_accessed": False,
    }


def main() -> None:
    args = parse_args()
    started = time.time()
    v1.IMAGE_TOPICS = tuple(dict.fromkeys((*v1.IMAGE_TOPICS, JACKAL_CAMERA_TOPIC)))
    plan18 = v1.read_json(args.plan18)
    plan17 = v1.read_json(args.plan17)
    v1.require(len(plan18["selected"]) == 18, "original plan is not 18-route")
    v1.require(len(plan17["selected"]) == 17, "fallback plan is not 17-route")
    tdr = v1.read_json(args.tdr_manifest)
    official = {row["label"]: row for row in tdr["files"]}
    v1.require(len(official) == 18, "official TDR manifest is not the frozen 18-bag snapshot")
    forbidden, forbidden_raw = v1.load_registry(args.forbidden_registry, forbidden=True)
    _, used_raw = v1.load_registry(args.used_registry, forbidden=False)

    corrected_root = args.incoming_root / "quarantine/official_scand_selective_v2_corrected_topic"
    corrected_root.mkdir(parents=True, exist_ok=True)
    gdc_audit = gdc_camera_audit(args.gdc_bag)
    gdc_audit_path = corrected_root / "gdc_camera_topic_correction.audit.json"
    v1.write_text_exact(gdc_audit_path, canonical_json(gdc_audit) + "\n")
    restored = bool(gdc_audit["input_eligible"])
    plan = plan18 if restored else plan17
    split = {"development": 12, "one_time_validation": 6 if restored else 5}
    decision = {
        "schema": SCHEMA + "_preregistration",
        "status": "frozen_after_label_free_input_audit_before_outcomes",
        "legacy_exclusion_invalidated": True,
        "reason": "legacy processor omitted the observed Jackal camera topic",
        "gdc_audit": str(gdc_audit_path),
        "gdc_audit_sha256": sha256_file(gdc_audit_path),
        "deterministic_rule": "restore original 18-route 12/6 split iff corrected 4Hz stream supplies at least 40 H4 windows; otherwise use prelabel 17-route 12/5 fallback",
        "gdc_restored": restored,
        "selected_plan": str((args.plan18 if restored else args.plan17).resolve()),
        "selected_plan_sha256": sha256_file(args.plan18 if restored else args.plan17),
        "selected_routes": len(plan["selected"]),
        "split": split,
        "camera_topics": list(v1.IMAGE_TOPICS),
        "processing": "camera-only 4Hz, center-crop 4:3, 320x240 JPEG quality 95",
        "outcomes_or_labels_opened": False,
        "network_accessed": False,
        "reserve_content_accessed": False,
    }
    prereg_path = corrected_root / "local_ingest.preregistration.json"
    v1.write_text_exact(prereg_path, canonical_json(decision) + "\n")

    cohort_rows = {"development": [], "one_time_validation": []}
    bag_audit = []
    for position, selected in enumerate(plan["selected"], start=1):
        filename = selected["official_file_name"] + ".bag"
        source = args.gdc_bag if filename == args.gdc_bag.name else args.source_dir / filename
        v1.require(source.is_file(), f"missing downloaded bag: {source}")
        expected = official.get(filename)
        v1.require(expected is not None, f"bag absent from official TDR snapshot: {filename}")
        v1.require(source.stat().st_size == int(expected["filesize"]), f"size mismatch: {filename}")
        observed_md5 = local_v1.md5_file(source)
        v1.require(observed_md5 == expected["checksum"], f"official MD5 mismatch: {filename}")
        topics = local_v1.topic_inventory(source)
        row = {**selected, "source_official_index_sha256": plan["identity_audit"]["official_index_sha256"]}
        destination = corrected_root / "processed" / selected["cohort_role"] / selected["bag_id"]
        processed = v1.process_camera_only(source, destination, row, forbidden)
        cohort_rows[selected["cohort_role"]].extend(processed["manifest_rows"])
        bag_audit.append(
            {
                "position": position,
                "filename": filename,
                "route_group": selected["route_group"],
                "cohort_role": selected["cohort_role"],
                "bytes": source.stat().st_size,
                "official_md5": observed_md5,
                "source_bag_sha256": processed["metadata"]["source_bag_sha256"],
                "source_path": str(source.resolve()),
                "processed_route": str(destination),
                "processed_frames": len(processed["metadata"]["frames"]),
                "processed_windows": len(processed["manifest_rows"]),
                "camera_contract": processed["metadata"]["camera_preprocessing"],
                "topic_metadata_only": topics,
                "outcome_values_opened": False,
            }
        )
        print(canonical_json({"status": "processed", "position": position, "total": len(plan["selected"]), "filename": filename, "role": selected["cohort_role"], "frames": len(processed["metadata"]["frames"])}), flush=True)

    staging = corrected_root / "staging_manifests"
    development_staging = staging / "development.manifest.windows.jsonl"
    validation_staging = staging / "one_time_validation.manifest.windows.jsonl"
    v1.write_text_exact(development_staging, v1.manifest_content(cohort_rows["development"]))
    v1.write_text_exact(validation_staging, v1.manifest_content(cohort_rows["one_time_validation"]))
    readiness = v1.evaluate_readiness(development_staging, validation_staging, args.used_registry, args.forbidden_registry)
    v1.write_json_atomic(corrected_root / "readiness.prepublish.json", readiness)
    v1.require(readiness["ready"], f"prepublish readiness failed: {readiness['errors']}")

    suffix = "routefit18_v2" if restored else "routefit17_v2"
    development_final = args.incoming_root / f"development_{suffix}/manifest.windows.jsonl"
    validation_final = args.incoming_root / f"one_time_validation_{suffix}/manifest.windows.jsonl"
    v1.write_text_exact(development_final, development_staging.read_text(encoding="utf-8"))
    v1.write_text_exact(validation_final, validation_staging.read_text(encoding="utf-8"))
    final_readiness = v1.evaluate_readiness(development_final, validation_final, args.used_registry, args.forbidden_registry)
    final_readiness_path = args.incoming_root / f"readiness.{suffix}.json"
    v1.write_json_atomic(final_readiness_path, final_readiness)
    v1.require(final_readiness["ready"], f"final readiness failed: {final_readiness['errors']}")
    result = {
        "schema": SCHEMA + "_result",
        "status": "complete_prelabel_ready",
        "gdc_restored": restored,
        "legacy_short_route_conclusion_valid": False,
        "routes": len(plan["selected"]),
        "split": split,
        "bags": bag_audit,
        "development": {"routes": split["development"], "windows": len(cohort_rows["development"]), "manifest": str(development_final), "manifest_sha256": sha256_file(development_final)},
        "one_time_validation": {"routes": split["one_time_validation"], "windows": len(cohort_rows["one_time_validation"]), "manifest": str(validation_final), "manifest_sha256": sha256_file(validation_final), "labels_opened": False},
        "readiness": final_readiness,
        "readiness_path": str(final_readiness_path),
        "readiness_sha256": sha256_file(final_readiness_path),
        "preregistration": str(prereg_path),
        "preregistration_sha256": sha256_file(prereg_path),
        "used_registry_status": used_raw.get("status"),
        "forbidden_registry_status": forbidden_raw.get("status"),
        "elapsed_seconds": time.time() - started,
        "network_accessed": False,
        "odometry_or_truth_values_opened": False,
        "reserve_content_accessed": False,
        "training_started": False,
    }
    result_path = corrected_root / "local_ingest.result.json"
    v1.write_json_atomic(result_path, result)
    print(canonical_json({**result, "result_path": str(result_path), "result_sha256": sha256_file(result_path)}))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(canonical_json({"schema": SCHEMA + "_failure", "status": "fail_closed", "error_type": type(error).__name__, "error": str(error), "network_accessed": False, "outcomes_or_labels_opened": False, "reserve_content_accessed": False}), file=sys.stderr)
        raise SystemExit(2) from error
