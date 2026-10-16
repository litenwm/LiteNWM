#!/usr/bin/env python3
"""Fail-closed ingest of only camera-contract-compatible routefit bags."""

from __future__ import annotations

from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from planning.proposal_tail_sidecar_v1 import canonical_json, sha256_file  # noqa: E402
from tools.planning import download_prepare_new_scand_bags_v1 as v1  # noqa: E402
from tools.planning import ingest_local_scand_routefit17_v1 as local_v1  # noqa: E402
from tools.planning.ingest_local_scand_routefit18_v2 import JACKAL_CAMERA_TOPIC  # noqa: E402


SCHEMA = "ingest_local_scand_contract_compatible_v4"


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=Path("/mnt/d/VEDIOS"))
    parser.add_argument(
        "--plan",
        type=Path,
        default=ROOT / "data/proposal_tail_collection_v1/incoming/quarantine/official_scand_selective_v1/download_plan.routefit_v3.json",
    )
    parser.add_argument(
        "--camera-inventory",
        type=Path,
        default=ROOT / "data/proposal_tail_collection_v1/incoming/quarantine/official_scand_selective_v3_topic_audited/camera_topic_inventory.json",
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
    args = parser.parse_args()
    started = time.time()
    v1.IMAGE_TOPICS = tuple(dict.fromkeys((*v1.IMAGE_TOPICS, JACKAL_CAMERA_TOPIC)))
    plan = v1.read_json(args.plan)
    inventory = v1.read_json(args.camera_inventory)
    availability = {row["filename"]: bool(row["canonical_available"]) for row in inventory["rows"]}
    selected = [row for row in plan["selected"] if availability.get(row["official_file_name"] + ".bag")]
    rejected = [row for row in plan["selected"] if not availability.get(row["official_file_name"] + ".bag")]
    development_routes = sum(row["cohort_role"] == "development" for row in selected)
    validation_routes = sum(row["cohort_role"] == "one_time_validation" for row in selected)
    v1.require((len(selected), development_routes, validation_routes) == (10, 6, 4), "camera-compatible subset changed")
    official = {row["label"]: row for row in v1.read_json(args.tdr_manifest)["files"]}
    forbidden, forbidden_raw = v1.load_registry(args.forbidden_registry, forbidden=True)
    _, used_raw = v1.load_registry(args.used_registry, forbidden=False)
    root = args.incoming_root / "quarantine/official_scand_selective_v4_contract_compatible"
    root.mkdir(parents=True, exist_ok=True)
    prereg = {
        "schema": SCHEMA + "_preregistration",
        "status": "frozen_before_outcomes",
        "selection_rule": "retain only prelabel routes with the established external RGB stream; reject body-fisheye-only Spot routes",
        "canonical_topics": list(v1.IMAGE_TOPICS),
        "pixel_contract_evidence": {
            "spot_external_rgb_best_psnr_db": 31.142839804659804,
            "spot_body_camera_best_psnr_db_range": [8.77895729756151, 10.812337799723311],
            "jackal_external_rgb_best_psnr_db": 31.928073337374528,
        },
        "camera_inventory": str(args.camera_inventory.resolve()),
        "camera_inventory_sha256": sha256_file(args.camera_inventory),
        "selected_routes": len(selected),
        "development_routes": development_routes,
        "one_time_validation_routes": validation_routes,
        "rejected_routes": [row["route_group"] for row in rejected],
        "underpowered_vs_original_12_5_target": True,
        "labels_or_outcomes_opened": False,
        "network_accessed": False,
        "reserve_content_accessed": False,
    }
    prereg_path = root / "local_ingest.preregistration.json"
    v1.write_text_exact(prereg_path, canonical_json(prereg) + "\n")
    cohorts = {"development": [], "one_time_validation": []}
    audits = []
    for position, row0 in enumerate(selected, start=1):
        filename = row0["official_file_name"] + ".bag"
        source = args.source_dir / filename
        expected = official[filename]
        v1.require(source.is_file(), f"missing source: {filename}")
        v1.require(source.stat().st_size == int(expected["filesize"]), f"size mismatch: {filename}")
        md5 = local_v1.md5_file(source)
        v1.require(md5 == expected["checksum"], f"official MD5 mismatch: {filename}")
        topics = local_v1.topic_inventory(source)
        row = {**row0, "source_official_index_sha256": plan["identity_audit"]["official_index_sha256"]}
        destination = root / "processed" / row0["cohort_role"] / row0["bag_id"]
        processed = v1.process_camera_only(source, destination, row, forbidden)
        cohorts[row0["cohort_role"]].extend(processed["manifest_rows"])
        audits.append({
            "position": position,
            "filename": filename,
            "route_group": row0["route_group"],
            "cohort_role": row0["cohort_role"],
            "bytes": source.stat().st_size,
            "official_md5": md5,
            "source_bag_sha256": processed["metadata"]["source_bag_sha256"],
            "processed_frames": len(processed["metadata"]["frames"]),
            "processed_windows": len(processed["manifest_rows"]),
            "camera_contract": processed["metadata"]["camera_preprocessing"],
            "topic_metadata_only": topics,
            "outcome_values_opened": False,
        })
        print(canonical_json({"status": "processed", "position": position, "total": len(selected), "filename": filename, "role": row0["cohort_role"], "frames": len(processed["metadata"]["frames"])}), flush=True)
    staging = root / "staging_manifests"
    dev_stage = staging / "development.manifest.windows.jsonl"
    val_stage = staging / "one_time_validation.manifest.windows.jsonl"
    v1.write_text_exact(dev_stage, v1.manifest_content(cohorts["development"]))
    v1.write_text_exact(val_stage, v1.manifest_content(cohorts["one_time_validation"]))
    readiness = v1.evaluate_readiness(dev_stage, val_stage, args.used_registry, args.forbidden_registry)
    v1.write_json_atomic(root / "readiness.prepublish.json", readiness)
    v1.require(readiness["ready"], f"prepublish readiness failed: {readiness['errors']}")
    dev_final = args.incoming_root / "development_routefit10_v4/manifest.windows.jsonl"
    val_final = args.incoming_root / "one_time_validation_routefit10_v4/manifest.windows.jsonl"
    v1.write_text_exact(dev_final, dev_stage.read_text(encoding="utf-8"))
    v1.write_text_exact(val_final, val_stage.read_text(encoding="utf-8"))
    final_ready = v1.evaluate_readiness(dev_final, val_final, args.used_registry, args.forbidden_registry)
    ready_path = args.incoming_root / "readiness.routefit10_v4.json"
    v1.write_json_atomic(ready_path, final_ready)
    v1.require(final_ready["ready"], f"final readiness failed: {final_ready['errors']}")
    result = {
        "schema": SCHEMA + "_result",
        "status": "complete_prelabel_ready_underpowered_pilot",
        "development": {"routes": development_routes, "windows": len(cohorts["development"]), "manifest": str(dev_final), "manifest_sha256": sha256_file(dev_final)},
        "one_time_validation": {"routes": validation_routes, "windows": len(cohorts["one_time_validation"]), "manifest": str(val_final), "manifest_sha256": sha256_file(val_final), "labels_opened": False},
        "rejected_body_camera_only_routes": [row["official_file_name"] for row in rejected],
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
    main()
