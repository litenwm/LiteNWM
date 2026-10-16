#!/usr/bin/env python3
"""Single-command fail-closed readiness check for incoming SCAND cohorts.

Only public prelabel manifests and hash-only identity registries are read.
The forbidden registry is structurally restricted to hashes and audit metadata;
reserve content, paths, source manifests, labels, truth, and outcomes are never
opened by this checker.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from planning.proposal_tail_sidecar_v1 import (  # noqa: E402
    canonical_json,
    identity_hash,
    sha256_file,
)
from tools.planning.build_incoming_scand_route_manifest_v1 import (  # noqa: E402
    read_manifest,
)


SCHEMA = "proposal_tail_incoming_readiness_v1"
KINDS = ("route", "day", "bag", "trajectory", "window")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
FORBIDDEN_REGISTRY_KEYS = {
    "schema",
    "status",
    "complete",
    "identity_values_are_hash_only",
    "route_hashes",
    "day_hashes",
    "bag_hashes",
    "trajectory_hashes",
    "window_hashes",
    "generated_at",
    "custodian_id",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def read_json(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"missing JSON file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON object required: {path}")
    return value


def validate_hash_list(value: Any, field: str) -> set[str]:
    require(isinstance(value, list), f"{field} must be a list")
    result = {str(item) for item in value}
    require(len(result) == len(value), f"duplicate hashes in {field}")
    require(all(HEX64.fullmatch(item) for item in result), f"non-hash value in {field}")
    return result


def load_registry(path: Path, *, forbidden: bool) -> tuple[dict[str, set[str]], dict[str, Any]]:
    registry = read_json(path)
    require(registry.get("complete") is True, f"incomplete identity registry: {path}")
    require(registry.get("identity_values_are_hash_only") is True, f"registry is not hash-only: {path}")
    if forbidden:
        unexpected = set(registry) - FORBIDDEN_REGISTRY_KEYS
        require(not unexpected, f"forbidden registry contains non-hash/source fields: {sorted(unexpected)}")
        require("sources" not in registry, "forbidden registry must not expose source manifests")
    values = {
        kind: validate_hash_list(registry.get(f"{kind}_hashes"), f"{kind}_hashes")
        for kind in KINDS
    }
    return values, registry


def manifest_identity_sets(rows: Sequence[Mapping[str, Any]]) -> dict[str, set[str]]:
    mapping = {
        "route": "route_group",
        "day": "collection_day",
        "bag": "bag_id",
        "trajectory": "trajectory_id",
        "window": "window_uid",
    }
    return {
        kind: {identity_hash(kind, row[field]) for row in rows}
        for kind, field in mapping.items()
    }


def role_check(rows: Sequence[Mapping[str, Any]], role: str) -> None:
    require(all(row.get("cohort_role") == role for row in rows), f"manifest role is not {role}")
    require(all(row.get("labels_opened") is False for row in rows), "manifest reports opened labels")


def overlap(left: Mapping[str, set[str]], right: Mapping[str, set[str]]) -> dict[str, list[str]]:
    return {kind: sorted(left[kind] & right[kind]) for kind in KINDS}


def evaluate_readiness(
    development_manifest: Path,
    validation_manifest: Path,
    used_registry_path: Path,
    forbidden_registry_path: Path,
) -> dict[str, Any]:
    errors: list[str] = []
    evidence: dict[str, Any] = {}
    try:
        development = read_manifest(development_manifest)
        validation = read_manifest(validation_manifest)
        role_check(development, "development")
        role_check(validation, "one_time_validation")
        development_ids = manifest_identity_sets(development)
        validation_ids = manifest_identity_sets(validation)
        used, used_raw = load_registry(used_registry_path, forbidden=False)
        forbidden, forbidden_raw = load_registry(forbidden_registry_path, forbidden=True)
        dev_used = overlap(development_ids, used)
        val_used = overlap(validation_ids, used)
        dev_forbidden = overlap(development_ids, forbidden)
        val_forbidden = overlap(validation_ids, forbidden)
        cross = overlap(development_ids, validation_ids)
        require(not any(dev_used.values()), "development overlaps v70-v77 used identities")
        require(not any(val_used.values()), "validation overlaps v70-v77 used identities")
        require(not any(dev_forbidden.values()), "development overlaps forbidden identities")
        require(not any(val_forbidden.values()), "validation overlaps forbidden identities")
        require(not cross["route"], "development/validation route overlap")
        require(not cross["trajectory"], "development/validation trajectory overlap")
        require(not cross["window"], "development/validation window overlap")
        evidence = {
            "development": {
                "manifest": str(development_manifest.resolve()),
                "manifest_sha256": sha256_file(development_manifest),
                "states": len(development),
                "routes": len(development_ids["route"]),
                "trajectories": len(development_ids["trajectory"]),
            },
            "one_time_validation": {
                "manifest": str(validation_manifest.resolve()),
                "manifest_sha256": sha256_file(validation_manifest),
                "states": len(validation),
                "routes": len(validation_ids["route"]),
                "trajectories": len(validation_ids["trajectory"]),
            },
            "used_registry": {
                "path": str(used_registry_path.resolve()),
                "sha256": sha256_file(used_registry_path),
                "status": used_raw.get("status"),
            },
            "forbidden_registry": {
                "path": str(forbidden_registry_path.resolve()),
                "sha256": sha256_file(forbidden_registry_path),
                "status": forbidden_raw.get("status"),
                "hash_only_structure_verified": True,
            },
            "overlap": {
                "development_vs_used": dev_used,
                "validation_vs_used": val_used,
                "development_vs_forbidden": dev_forbidden,
                "validation_vs_forbidden": val_forbidden,
                "development_vs_validation": cross,
            },
        }
    except Exception as exc:  # fail-closed JSON is the command contract
        errors.append(f"{type(exc).__name__}: {exc}")
    return {
        "schema": SCHEMA,
        "status": "ready_prelabel" if not errors else "fail_closed",
        "ready": not errors,
        "labels_opened": False,
        "reserve_content_accessed": False,
        "sidecar_enabled": False,
        "checks": {
            "manifest_fields_images_and_sha": not errors,
            "used_identity_overlap_zero": not errors,
            "complete_hash_only_forbidden_registry": not errors,
            "development_validation_route_disjoint": not errors,
            "labels_unopened": not errors,
        },
        "errors": errors,
        "evidence": evidence,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--used-identity-registry", type=Path, required=True)
    parser.add_argument("--forbidden-identity-registry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = evaluate_readiness(
        args.development_manifest,
        args.validation_manifest,
        args.used_identity_registry,
        args.forbidden_identity_registry,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(canonical_json(result) + "\n", encoding="utf-8")
    print(canonical_json(result))
    if not result["ready"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
