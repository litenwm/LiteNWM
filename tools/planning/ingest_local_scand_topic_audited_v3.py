#!/usr/bin/env python3
"""Topic-audited, network-free local SCAND ingest after the Jackal fix."""

from __future__ import annotations

from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from planning.proposal_tail_sidecar_v1 import canonical_json, sha256_file  # noqa: E402
from tools.planning import download_prepare_new_scand_bags_v1 as v1  # noqa: E402
from tools.planning import ingest_local_scand_routefit17_v1 as local_v1  # noqa: E402
from tools.planning import ingest_local_scand_routefit18_v2 as v2  # noqa: E402


SCHEMA = "ingest_local_scand_topic_audited_v3"


def audit_gdc(path: Path) -> dict:
    from rosbags.highlevel import AnyReader
    from rosbags.typesys import Stores, get_typestore

    typestore = get_typestore(Stores.ROS1_NOETIC)
    with AnyReader([path], default_typestore=typestore) as reader:
        connections = list(reader.connections)
        selected_topic = v1.choose_image_topic(connections)
        selected = [connection for connection in connections if connection.topic == selected_topic]
        timestamps = [int(timestamp) for _, timestamp, _ in reader.messages(connections=selected)]
        start_ns, end_ns = int(reader.start_time), int(reader.end_time)
        topics = [
            {"topic": c.topic, "msgtype": c.msgtype, "message_count": int(getattr(c, "msgcount", 0))}
            for c in connections
        ]
    v1.require(timestamps, "GDC selected camera stream is empty")
    v1.require(all(a <= b for a, b in zip(timestamps, timestamps[1:])), "GDC camera timestamps are nonmonotonic")
    sampled = []
    next_ns = None
    step_ns = int(round(1e9 / v1.SAMPLE_RATE_HZ))
    for timestamp in timestamps:
        if next_ns is not None and timestamp < next_ns:
            continue
        sampled.append(timestamp)
        next_ns = timestamp + step_ns
    eligible = len(sampled) - v1.GOAL_OFFSET_FRAMES - (v1.HISTORY_FRAMES - 1)
    return {
        "schema": "jackal_gdc_full_topic_reaudit_v2",
        "status": "complete_prelabel_input_audit",
        "bag": str(path.resolve()),
        "bag_sha256": sha256_file(path),
        "bag_duration_seconds": (end_ns - start_ns) / 1e9,
        "selected_camera_topic": selected_topic,
        "left_camera_topic_present": any(c.topic == v2.JACKAL_CAMERA_TOPIC for c in connections),
        "raw_camera_messages": len(timestamps),
        "camera_duration_seconds": (timestamps[-1] - timestamps[0]) / 1e9,
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
    args = v2.parse_args()
    started = time.time()
    v1.IMAGE_TOPICS = tuple(dict.fromkeys((*v1.IMAGE_TOPICS, v2.JACKAL_CAMERA_TOPIC)))
    plan18, plan17 = v1.read_json(args.plan18), v1.read_json(args.plan17)
    v1.require(len(plan18["selected"]) == 18 and len(plan17["selected"]) == 17, "route plan cardinality mismatch")
    official = {row["label"]: row for row in v1.read_json(args.tdr_manifest)["files"]}
    forbidden, forbidden_raw = v1.load_registry(args.forbidden_registry, forbidden=True)
    _, used_raw = v1.load_registry(args.used_registry, forbidden=False)
    root = args.incoming_root / "quarantine/official_scand_selective_v3_topic_audited"
    root.mkdir(parents=True, exist_ok=True)
    gdc = audit_gdc(args.gdc_bag)
    gdc_path = root / "gdc_full_topic_reaudit.json"
    v1.write_text_exact(gdc_path, canonical_json(gdc) + "\n")
    restored = bool(gdc["input_eligible"])
    plan = plan18 if restored else plan17
    development_routes = 12
    validation_routes = 6 if restored else 5
    prereg = {
        "schema": SCHEMA + "_preregistration",
        "status": "frozen_after_label_free_topic_audit_before_outcomes",
        "processor_fix": "accept /left/image_color/compressed for Jackal bags",
        "gdc_old_exclusion_temporarily_invalidated_for_reaudit": True,
        "gdc_final_decision": "restore" if restored else "exclude_revalidated_short",
        "deterministic_rule": "restore original 18-route 12/6 split iff corrected full-topic 4Hz stream has >=40 H4 windows; otherwise retain 17-route 12/5 plan",
        "selected_plan": str((args.plan18 if restored else args.plan17).resolve()),
        "selected_plan_sha256": sha256_file(args.plan18 if restored else args.plan17),
        "selected_routes": len(plan["selected"]),
        "development_routes": development_routes,
        "one_time_validation_routes": validation_routes,
        "gdc_audit": str(gdc_path),
        "gdc_audit_sha256": sha256_file(gdc_path),
        "outcomes_or_labels_opened": False,
        "network_accessed": False,
        "reserve_content_accessed": False,
    }
    prereg_path = root / "local_ingest.preregistration.json"
    v1.write_text_exact(prereg_path, canonical_json(prereg) + "\n")

    cohort_rows = {"development": [], "one_time_validation": []}
    audits = []
    for position, selected in enumerate(plan["selected"], start=1):
        filename = selected["official_file_name"] + ".bag"
        source = args.gdc_bag if filename == args.gdc_bag.name else args.source_dir / filename
        expected = official.get(filename)
        v1.require(source.is_file() and expected is not None, f"missing source or checksum: {filename}")
        v1.require(source.stat().st_size == int(expected["filesize"]), f"size mismatch: {filename}")
        observed_md5 = local_v1.md5_file(source)
        v1.require(observed_md5 == expected["checksum"], f"official MD5 mismatch: {filename}")
        topics = local_v1.topic_inventory(source)
        row = {**selected, "source_official_index_sha256": plan["identity_audit"]["official_index_sha256"]}
        destination = root / "processed" / selected["cohort_role"] / selected["bag_id"]
        processed = v1.process_camera_only(source, destination, row, forbidden)
        cohort_rows[selected["cohort_role"]].extend(processed["manifest_rows"])
        audits.append({
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
        })
        print(canonical_json({"status": "processed", "position": position, "total": len(plan["selected"]), "filename": filename, "role": selected["cohort_role"], "frames": len(processed["metadata"]["frames"])}), flush=True)

    staging = root / "staging_manifests"
    dev_stage = staging / "development.manifest.windows.jsonl"
    val_stage = staging / "one_time_validation.manifest.windows.jsonl"
    v1.write_text_exact(dev_stage, v1.manifest_content(cohort_rows["development"]))
    v1.write_text_exact(val_stage, v1.manifest_content(cohort_rows["one_time_validation"]))
    readiness = v1.evaluate_readiness(dev_stage, val_stage, args.used_registry, args.forbidden_registry)
    v1.write_json_atomic(root / "readiness.prepublish.json", readiness)
    v1.require(readiness["ready"], f"prepublish readiness failed: {readiness['errors']}")

    suffix = "routefit18_v3" if restored else "routefit17_v3"
    dev_final = args.incoming_root / f"development_{suffix}/manifest.windows.jsonl"
    val_final = args.incoming_root / f"one_time_validation_{suffix}/manifest.windows.jsonl"
    v1.write_text_exact(dev_final, dev_stage.read_text(encoding="utf-8"))
    v1.write_text_exact(val_final, val_stage.read_text(encoding="utf-8"))
    final_ready = v1.evaluate_readiness(dev_final, val_final, args.used_registry, args.forbidden_registry)
    ready_path = args.incoming_root / f"readiness.{suffix}.json"
    v1.write_json_atomic(ready_path, final_ready)
    v1.require(final_ready["ready"], f"final readiness failed: {final_ready['errors']}")
    result = {
        "schema": SCHEMA + "_result",
        "status": "complete_prelabel_ready",
        "gdc_final_decision": prereg["gdc_final_decision"],
        "routes": len(plan["selected"]),
        "development": {"routes": development_routes, "windows": len(cohort_rows["development"]), "manifest": str(dev_final), "manifest_sha256": sha256_file(dev_final)},
        "one_time_validation": {"routes": validation_routes, "windows": len(cohort_rows["one_time_validation"]), "manifest": str(val_final), "manifest_sha256": sha256_file(val_final), "labels_opened": False},
        "bags": audits,
        "readiness": final_ready,
        "readiness_path": str(ready_path),
        "readiness_sha256": sha256_file(ready_path),
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
    result_path = root / "local_ingest.result.json"
    v1.write_json_atomic(result_path, result)
    print(canonical_json({**result, "result_path": str(result_path), "result_sha256": sha256_file(result_path)}))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(canonical_json({"schema": SCHEMA + "_failure", "status": "fail_closed", "error_type": type(error).__name__, "error": str(error), "network_accessed": False, "outcomes_or_labels_opened": False, "reserve_content_accessed": False}), file=sys.stderr)
        raise SystemExit(2) from error
