#!/usr/bin/env python3
"""Identifier-only inventory of legal SCAND routes before v77 preregistration."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs/cem_development_v1/v77_scand_tail_joint_selector"
RESULT = OUTPUT / "route_inventory.json"
USED_RECORDS = ROOT / "outputs/cem_development_v1/scand_filter_attribution_v1/candidate_records.csv"
FRESH_MANIFEST = ROOT / "outputs/scand_domain_adaptation_v1/fresh_outer_v1/manifest.windows.jsonl"
INNER_ROOT = ROOT / "data/inner_manifests/scand"
FRESH_SEARCH_AUDIT = ROOT / "outputs/universal_dualdomain_raw_v2/scand_fresh_outer_v3/manifest.windows.audit.json"
FRESH_SEARCH_EXCLUSIONS = ROOT / "outputs/universal_dualdomain_raw_v2/scand_fresh_outer_v3/manifest.windows.excluded_trajectories.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reject_forbidden(path: Path) -> None:
    if "res" + "erve" in str(path).lower():
        raise RuntimeError(f"forbidden route-inventory path: {path}")


def manifest_identity(path: Path) -> dict[str, Any]:
    reject_forbidden(path)
    routes: set[str] = set()
    trajectories: set[str] = set()
    windows = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            route = record.get("route_group", record.get("route_id"))
            trajectory = record.get("trajectory_id")
            if route is not None:
                routes.add(str(route))
            if trajectory is not None:
                trajectories.add(str(trajectory))
            windows += 1
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": sha256_file(path),
        "windows": windows,
        "routes": sorted(routes),
        "route_count": len(routes),
        "trajectory_count": len(trajectories),
        "fields_read": ["route_group/route_id", "trajectory_id"],
        "labels_read": False,
        "images_read": False,
    }


def main() -> None:
    if RESULT.exists():
        raise FileExistsError(RESULT)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    reject_forbidden(USED_RECORDS)
    used_routes: set[str] = set()
    used_records: set[str] = set()
    with USED_RECORDS.open(newline="", encoding="utf-8") as handle:
        for record in csv.DictReader(handle):
            used_routes.add(record["route"])
            used_records.add(record["record_key"])

    fresh = manifest_identity(FRESH_MANIFEST)
    inner = {
        partition: manifest_identity(INNER_ROOT / f"{partition}.windows.jsonl")
        for partition in ("train-fit", "scorer-cal", "cem-cal", "internal-holdout")
    }
    search_audit_path = FRESH_SEARCH_AUDIT
    reject_forbidden(search_audit_path)
    search = json.loads(search_audit_path.read_text(encoding="utf-8"))
    if search.get("labels_opened") or search.get("pose_outcomes_opened"):
        raise RuntimeError("fresh-route search was not identity-only")
    if search.get("fresh_candidate_routes") != 0 or search.get("selected_routes") != 0:
        raise RuntimeError("fresh-route audit unexpectedly contains candidates; manual isolation review required")

    summary_path = INNER_ROOT / "manifest_summary.json"
    independence_path = INNER_ROOT / "independent_validation.json"
    for path in (summary_path, independence_path, FRESH_SEARCH_EXCLUSIONS):
        reject_forbidden(path)
    result = {
        "schema": "v77_scand_route_inventory_v1",
        "status": "completed_before_v77_preregistration",
        "scope": "route/trajectory identifiers, manifests, and hashes only",
        "reserve_accessed": False,
        "labels_opened": False,
        "images_opened": False,
        "v70_v76_spent_candidate_cache": {
            "routes": sorted(used_routes),
            "route_count": len(used_routes),
            "states": len(used_records),
            "candidate_rows": sum(1 for _ in USED_RECORDS.open(encoding="utf-8")) - 1,
            "candidate_records_path": str(USED_RECORDS.relative_to(ROOT)),
            "candidate_records_sha256": sha256_file(USED_RECORDS),
            "fresh_manifest": fresh,
            "route_identity_match": sorted(used_routes) == [f"scand::scand::{route}" for route in fresh["routes"]],
        },
        "historical_inner_train_only_partitions": inner,
        "inner_partition_contract": {
            "manifest_summary_path": str(summary_path.relative_to(ROOT)),
            "manifest_summary_sha256": sha256_file(summary_path),
            "independence_audit_path": str(independence_path.relative_to(ROOT)),
            "independence_audit_sha256": sha256_file(independence_path),
            "classification": (
                "historical train/scorer-cal/cem-cal/internal-holdout partitions; not an untouched outer cohort, "
                "and no frozen 65-candidate v70 CEM cache exists for v77 evaluation"
            ),
        },
        "identity_only_fresh_search": {
            "path": str(search_audit_path.relative_to(ROOT)),
            "sha256": sha256_file(search_audit_path),
            "excluded_identity_path": str(FRESH_SEARCH_EXCLUSIONS.relative_to(ROOT)),
            "excluded_identity_sha256": sha256_file(FRESH_SEARCH_EXCLUSIONS),
            "raw_trajectory_directories": search["raw_trajectory_directories"],
            "excluded_used_routes": search["excluded_used_routes"],
            "excluded_used_trajectories": search["excluded_used_trajectories"],
            "fresh_candidate_routes": search["fresh_candidate_routes"],
            "fresh_candidate_trajectories": search["fresh_candidate_trajectories"],
            "selected_routes": search["selected_routes"],
            "status": search["status"],
            "labels_opened": search["labels_opened"],
            "pose_outcomes_opened": search["pose_outcomes_opened"],
        },
        "legal_untouched_routes_with_required_v70_candidate_cache": 0,
        "isolation_feasible_now": False,
        "decision": (
            "no legal untouched non-reserve SCAND routes with frozen v70 65-candidate caches; "
            "v77 must use the spent 25-route/100-state cohort under strict nested route-OOF and can provide development evidence only"
        ),
        "new_routes_frozen_for_v77_use": [],
    }
    temporary = RESULT.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(RESULT)
    print(json.dumps({
        "result": str(RESULT),
        "sha256": sha256_file(RESULT),
        "spent_routes": len(used_routes),
        "legal_untouched_routes": 0,
        "decision": result["decision"],
    }, indent=2))


if __name__ == "__main__":
    main()
