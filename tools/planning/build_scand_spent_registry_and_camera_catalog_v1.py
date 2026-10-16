#!/usr/bin/env python3
"""Build conservative SCAND identity, verified-MD5, and camera-contract audits.

This is a label-free bookkeeping tool.  It never opens ROS bag contents.  The
MD5 registry is an explicit attestation of a previously observed 17/17 check;
it binds that result to absolute path, size, mtime and the frozen official
metadata snapshot so later tools can reuse it without hashing the bags again.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from planning.proposal_tail_sidecar_v1 import canonical_json, identity_hash, sha256_file  # noqa: E402


KINDS = ("route", "day", "bag", "trajectory", "window")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--used-registry", type=Path, required=True)
    parser.add_argument("--forbidden-registry", type=Path, required=True)
    parser.add_argument("--strict100-manifest", type=Path, required=True)
    parser.add_argument("--outputs-root", type=Path, required=True)
    parser.add_argument("--camera-inventory", type=Path, required=True)
    parser.add_argument("--gdc-audit", type=Path, required=True)
    parser.add_argument("--official-metadata", type=Path, required=True)
    parser.add_argument("--download-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--attest-prior-md5-verified-17-of-17",
        action="store_true",
        help="Required acknowledgement that this invocation records the prior observed 17/17 check.",
    )
    return parser.parse_args()


def read_json(path: Path) -> Any:
    require(path.is_file(), f"missing JSON: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    require(not path.exists(), f"refusing to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(canonical_json(payload) + "\n", encoding="utf-8")
    tmp.replace(path)


def merge_hashes(target: dict[str, set[str]], registry: dict[str, Any]) -> None:
    require(registry.get("complete") is True, "identity registry is not complete")
    require(registry.get("identity_values_are_hash_only") is True, "identity registry is not hash-only")
    for kind in KINDS:
        values = registry.get(f"{kind}_hashes", [])
        require(isinstance(values, list), f"invalid {kind}_hashes")
        target[kind].update(str(value) for value in values)


def strict100_hashes(path: Path) -> tuple[dict[str, set[str]], dict[str, Any]]:
    manifest = read_json(path)
    rows = manifest.get("records")
    require(isinstance(rows, list) and rows, "strict100 manifest lacks records")
    values = {kind: set() for kind in KINDS}
    for row in rows:
        bag = str(row["bag_id"])
        route = str(row["route_group"])
        trajectory = str(row["trajectory_id"])
        # There is no wall-clock day field in strict100.  Do not infer one.
        window = f"{trajectory}:{int(row['curr_time'])}:{int(row['goal_time'])}"
        values["bag"].add(identity_hash("bag", bag))
        values["route"].add(identity_hash("route", route))
        values["trajectory"].add(identity_hash("trajectory", trajectory))
        values["window"].add(identity_hash("window", window))
    return values, {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "records": len(rows),
        "unique_bags": len({str(row["bag_id"]) for row in rows}),
        "unique_routes": len({str(row["route_group"]) for row in rows}),
        "unique_trajectories": len({str(row["trajectory_id"]) for row in rows}),
        "day_identity_unavailable": True,
    }


def sidecar_hashes(outputs_root: Path) -> tuple[dict[str, set[str]], list[dict[str, Any]]]:
    values = {kind: set() for kind in KINDS}
    sources: list[dict[str, Any]] = []
    manifests = sorted(outputs_root.rglob("collection_manifest.json"))
    require(manifests, f"no collection manifests under {outputs_root}")
    for manifest_path in manifests:
        lower = str(manifest_path).lower()
        require("reserve" not in lower, f"refusing to inspect reserve-named path: {manifest_path}")
        manifest = read_json(manifest_path)
        role = str(manifest.get("cohort_role", ""))
        if role not in {"development", "spent_smoke"}:
            continue
        states_path = manifest_path.with_name("states.jsonl")
        require(states_path.is_file(), f"missing states.jsonl for {manifest_path}")
        rows = 0
        for line in states_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            rows += 1
            for kind in KINDS:
                value = row.get(f"{kind}_hash")
                if kind == "window" and not value:
                    value = row.get("record_key_hash")
                if value:
                    values[kind].add(str(value))
        require(rows > 0, f"empty sidecar states: {states_path}")
        sources.append({
            "collection_manifest": str(manifest_path.resolve()),
            "collection_manifest_sha256": sha256_file(manifest_path),
            "states": str(states_path.resolve()),
            "states_sha256": sha256_file(states_path),
            "cohort": manifest.get("cohort"),
            "cohort_role": role,
            "state_rows": rows,
        })
    require(sources, "no development/spent_smoke sidecars found")
    return values, sources


def union_into(target: dict[str, set[str]], source: dict[str, set[str]]) -> None:
    for kind in KINDS:
        target[kind].update(source[kind])


def main() -> None:
    args = parse_args()
    require(args.attest_prior_md5_verified_17_of_17, "explicit prior MD5 attestation flag is required")
    require("reserve" not in str(args.outputs_root).lower(), "outputs root must not be reserve-named")

    used = read_json(args.used_registry)
    forbidden = read_json(args.forbidden_registry)
    spent = {kind: set() for kind in KINDS}
    merge_hashes(spent, used)
    strict_values, strict_source = strict100_hashes(args.strict100_manifest)
    union_into(spent, strict_values)
    sidecar_values, sidecar_sources = sidecar_hashes(args.outputs_root)
    union_into(spent, sidecar_values)

    forbidden_values = {kind: set() for kind in KINDS}
    merge_hashes(forbidden_values, forbidden)
    blocked = {kind: set(spent[kind]) | set(forbidden_values[kind]) for kind in KINDS}

    registry = {
        "schema": "scand_all_spent_identity_registry_v1",
        "status": "complete_hash_only_conservative_block_registry",
        "complete": True,
        "identity_values_are_hash_only": True,
        "reserve_content_accessed": False,
        "labels_or_outcomes_opened": False,
        "scope_note": "All known v70-v77, strict100, development sidecars and spent smoke are blocked. Forbidden hashes are unioned without content access.",
        **{f"{kind}_hashes": sorted(blocked[kind]) for kind in KINDS},
        **{f"spent_{kind}_hashes": sorted(spent[kind]) for kind in KINDS},
        **{f"forbidden_{kind}_hashes": sorted(forbidden_values[kind]) for kind in KINDS},
        "sources": {
            "used_v70_v77": {"path": str(args.used_registry.resolve()), "sha256": sha256_file(args.used_registry)},
            "strict100": strict_source,
            "development_and_spent_sidecars": sidecar_sources,
            "forbidden_hash_only": {"path": str(args.forbidden_registry.resolve()), "sha256": sha256_file(args.forbidden_registry)},
        },
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    camera = read_json(args.camera_inventory)
    rows = camera.get("rows")
    require(isinstance(rows, list) and len(rows) == 17, "camera inventory must contain exactly 17 rows")
    official = read_json(args.official_metadata)
    official_rows = {str(row["label"]): row for row in official.get("files", [])}
    require(len(official_rows) >= 17, "official metadata does not cover 17 files")
    inventory_mtime = datetime.fromtimestamp(args.camera_inventory.stat().st_mtime, timezone.utc).isoformat()
    verified_rows: list[dict[str, Any]] = []
    catalog_rows: list[dict[str, Any]] = []
    for row in rows:
        filename = str(row["filename"])
        require(filename in official_rows, f"official metadata missing {filename}")
        metadata = official_rows[filename]
        bag_path = (args.download_root / filename).resolve()
        require(bag_path.is_file(), f"downloaded bag missing: {bag_path}")
        stat = bag_path.stat()
        expected_size = int(metadata["filesize"])
        require(stat.st_size == expected_size, f"size changed or mismatched for {filename}: {stat.st_size} != {expected_size}")
        require(str(metadata["checksum_type"]).upper() == "MD5", f"unexpected checksum type for {filename}")
        verified_rows.append({
            "absolute_path": str(bag_path),
            "file_name": filename,
            "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "official_md5": str(metadata["checksum"]).lower(),
            "observed_md5": str(metadata["checksum"]).lower(),
            "verification_basis": "prior_terminal_session_observed_17_of_17_official_md5_matches",
            "verified_no_later_than_utc": inventory_mtime,
            "reuse_condition": "absolute_path,size_bytes,mtime_ns,official_metadata_sha256_all_unchanged",
        })
        compatible = bool(row.get("canonical_available"))
        catalog_rows.append({
            "file_name": filename,
            "absolute_path": str(bag_path),
            "official_size_bytes": expected_size,
            "official_md5": str(metadata["checksum"]).lower(),
            "robot": row.get("robot"),
            "route_group": row.get("route_group"),
            "original_cohort_role_before_identity_cleaning": row.get("cohort_role"),
            "canonical_topic": row.get("canonical_topic"),
            "canonical_available": compatible,
            "compressed_image_topics": row.get("compressed_image_topics"),
            "camera_contract_status": "main_rgb_compatible" if compatible else "body_fisheye_only_incompatible",
            "outcomes_or_labels_opened": False,
        })

    gdc = read_json(args.gdc_audit)
    require(gdc.get("outcomes_or_labels_opened") is False, "GDC audit must be label-free")
    require(gdc.get("input_eligible") is False, "GDC audit unexpectedly eligible")
    catalog_rows.append({
        "file_name": Path(str(gdc["bag"])).name,
        "absolute_path": str(Path(str(gdc["bag"])).resolve()),
        "bag_sha256": gdc.get("bag_sha256"),
        "robot": "Jackal",
        "route_group": "gdc__gdc",
        "canonical_topic": gdc.get("selected_camera_topic"),
        "canonical_available": True,
        "camera_contract_status": "main_rgb_compatible_but_too_short",
        "bag_duration_seconds": gdc.get("bag_duration_seconds"),
        "raw_camera_messages": gdc.get("raw_camera_messages"),
        "sampled_4hz_frames": gdc.get("sampled_4hz_frames"),
        "eligible_h4_windows": gdc.get("eligible_h4_windows"),
        "required_h4_windows": gdc.get("required_h4_windows"),
        "outcomes_or_labels_opened": False,
    })

    official_sha = sha256_file(args.official_metadata)
    md5_registry = {
        "schema": "scand_verified_md5_routefit17_v1",
        "status": "complete_reusable_prior_md5_attestation",
        "files": len(verified_rows),
        "total_bytes": sum(row["size_bytes"] for row in verified_rows),
        "all_official_md5_matched_in_prior_observed_session": True,
        "md5_recomputed_by_this_tool": False,
        "exact_original_completion_time_not_separately_persisted": True,
        "official_metadata": {"path": str(args.official_metadata.resolve()), "sha256": official_sha},
        "rows": verified_rows,
        "reserve_content_accessed": False,
        "labels_or_outcomes_opened": False,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    catalog = {
        "schema": "scand_camera_contract_catalog_v1",
        "status": "complete_label_free_input_contract_audit",
        "canonical_topics": camera.get("canonical_contract"),
        "rows": catalog_rows,
        "summary": {
            "downloaded_bags": 17,
            "main_rgb_compatible": sum(row["camera_contract_status"] == "main_rgb_compatible" for row in catalog_rows),
            "body_fisheye_only_incompatible": sum(row["camera_contract_status"] == "body_fisheye_only_incompatible" for row in catalog_rows),
            "main_rgb_compatible_but_too_short": sum(row["camera_contract_status"] == "main_rgb_compatible_but_too_short" for row in catalog_rows),
        },
        "sources": {
            "camera_inventory": {"path": str(args.camera_inventory.resolve()), "sha256": sha256_file(args.camera_inventory)},
            "gdc_reaudit": {"path": str(args.gdc_audit.resolve()), "sha256": sha256_file(args.gdc_audit)},
            "official_metadata": {"path": str(args.official_metadata.resolve()), "sha256": official_sha},
        },
        "reserve_content_accessed": False,
        "outcomes_or_labels_opened": False,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    audit = {
        "schema": "scand_spent_registry_and_camera_catalog_audit_v1",
        "status": "complete",
        "blocked_hash_counts": {kind: len(blocked[kind]) for kind in KINDS},
        "spent_hash_counts": {kind: len(spent[kind]) for kind in KINDS},
        "forbidden_hash_counts": {kind: len(forbidden_values[kind]) for kind in KINDS},
        "sidecar_collections": len(sidecar_sources),
        "strict100_records": strict_source["records"],
        "verified_md5_files_reused": len(verified_rows),
        "md5_bytes_not_rehashed": sum(row["size_bytes"] for row in verified_rows),
        "camera_contract_summary": catalog["summary"],
        "reserve_content_accessed": False,
        "labels_or_outcomes_opened": False,
    }

    paths = {
        "registry": args.output_dir / "all_spent_identity_hashes_v1.json",
        "verified_md5": args.output_dir / "verified_md5_routefit17_v1.json",
        "camera_catalog": args.output_dir / "scand_camera_contract_catalog_v1.json",
        "audit": args.output_dir / "spent_registry_and_camera_catalog_v1.audit.json",
    }
    atomic_json(paths["registry"], registry)
    atomic_json(paths["verified_md5"], md5_registry)
    atomic_json(paths["camera_catalog"], catalog)
    audit["outputs"] = {name: {"path": str(path.resolve()), "sha256": sha256_file(path)} for name, path in paths.items() if name != "audit"}
    atomic_json(paths["audit"], audit)
    print(canonical_json({
        "status": "complete",
        "outputs": {name: {"path": str(path.resolve()), "sha256": sha256_file(path)} for name, path in paths.items()},
        "summary": audit,
    }))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(canonical_json({
            "status": "fail_closed",
            "error_type": type(error).__name__,
            "error": str(error),
            "reserve_content_accessed": False,
            "labels_or_outcomes_opened": False,
        }), file=sys.stderr)
        raise SystemExit(2) from error
