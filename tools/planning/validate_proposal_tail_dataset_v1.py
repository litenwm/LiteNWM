#!/usr/bin/env python3
"""Fail-closed validator for SCAND proposal-tail sidecar collections."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from planning.proposal_tail_sidecar_v1 import (  # noqa: E402
    METRIC_NAMES,
    SCHEMA,
    array_sha256,
    canonical_json,
    reject_forbidden_path,
    sha256_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--used-identity-registry", type=Path, required=True)
    parser.add_argument("--reserve-identity-registry", type=Path, required=True)
    parser.add_argument("--against", action="append", type=Path, default=[])
    parser.add_argument("--audit-output", type=Path, default=None)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_states(collection: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in (collection / "states.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def identity_sets(states: list[dict[str, Any]]) -> dict[str, set[str]]:
    return {
        kind: {
            str(row[f"{kind}_hash"])
            for row in states
            if row.get(f"{kind}_hash") is not None
        }
        for kind in ("route", "day", "bag", "trajectory", "window")
    }


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def timestamps_complete_and_monotonic(state: dict[str, Any]) -> bool:
    timestamp = state["timestamps"]
    values = []
    history = timestamp["history"]["value"]
    if history is not None:
        values.extend(history)
    for key in ("current", "goal"):
        if timestamp[key]["value"] is not None:
            values.append(timestamp[key]["value"])
    if not values:
        return False
    numeric = np.asarray(values, dtype=np.float64)
    return bool(np.isfinite(numeric).all() and np.all(np.diff(numeric) >= 0.0))


def camera_complete(state: dict[str, Any]) -> bool:
    camera = state["camera"]
    return all(camera[key]["value"] is not None for key in ("width", "height", "encoding", "file_paths", "file_sizes", "file_sha256"))


def validate() -> dict[str, Any]:
    args = parse_args()
    reject_forbidden_path(args.collection)
    manifest_path = args.collection / "collection_manifest.json"
    manifest = read_json(manifest_path)
    require(manifest.get("schema") == SCHEMA, "collection schema mismatch")
    role = manifest.get("cohort_role")
    require(role in ("spent_smoke", "development", "one_time_validation"), "invalid cohort role")
    require(manifest.get("formal_behavior_influence") is False, "sidecar claims behavior influence")
    require(tuple(manifest.get("metric_names", ())) == METRIC_NAMES, "metric order changed")
    require(manifest.get("candidate_count") == 65, "candidate count must be 65")
    require(manifest.get("seed_count") == 3, "seed count must be three")
    require(manifest.get("feature_waypoint_unit_m") == 0.25, "waypoint unit changed")
    for relative, expected in manifest["files"].items():
        path = args.collection / relative
        require(path.is_file(), f"missing sidecar file: {relative}")
        require(sha256_file(path) == expected, f"sidecar SHA mismatch: {relative}")

    states = read_states(args.collection)
    require(len(states) == int(manifest["states"]), "state count mismatch")
    record_keys = [str(row["record_key"]) for row in states]
    require(len(record_keys) == len(set(record_keys)), "duplicate record key")
    windows = [str(row["window_hash"]) for row in states]
    require(len(windows) == len(set(windows)), "duplicate window identity")
    table = np.load(args.collection / "candidate_table.npz", allow_pickle=False)
    candidates = table["candidates_feature_waypoint_units"]
    require(candidates.shape == (len(states), 65, 4, 2), f"bad candidate table shape: {candidates.shape}")
    require(np.isfinite(candidates).all(), "candidate table has NaN/Inf")
    require(np.array_equal(table["record_key"].astype(str), np.asarray(record_keys)), "candidate record identity mismatch")
    require(array_sha256(table["record_key"]) == manifest["identity"]["record_key_sha256"], "record-key SHA mismatch")
    for pool_name in ("fixed_pool", "iteration_00", "iteration_01", "iteration_02"):
        pool = np.load(
            args.collection / pool_name / "candidates_and_scores.npz",
            allow_pickle=False,
        )
        candidate_key = (
            "candidates_feature_waypoint_units"
            if "candidates_feature_waypoint_units" in pool.files
            else "candidates"
        )
        require(
            pool[candidate_key].shape == (len(states), 17, 4, 2),
            f"{pool_name} candidate shape mismatch",
        )
        for seed_name in ("ate_by_seed", "rtp_by_seed", "support_by_seed"):
            key = f"ltwm_{seed_name}" if f"ltwm_{seed_name}" in pool.files else seed_name
            require(key in pool.files, f"{pool_name} missing {seed_name}")
            require(
                pool[key].shape == (3, len(states), 17),
                f"{pool_name} {seed_name} shape mismatch: {pool[key].shape}",
            )
        if manifest.get("raw_draws_observable") is True and pool_name != "fixed_pool":
            draws = np.load(args.collection / pool_name / "raw_draws.npz", allow_pickle=False)
            require(draws["raw_draw_unclipped"].shape[-1] == 8, "raw draw shape mismatch")
            require(draws["physical_coordinate_mask"].shape[-1] == 12, "physical submask shape mismatch")
            require(len(draws["support_score"]) == len(draws["accepted_mask"]), "draw mask length mismatch")

    used = read_json(args.used_identity_registry)
    reserve = read_json(args.reserve_identity_registry)
    require(used.get("complete") is True, "used identity registry is incomplete")
    current = identity_sets(states)
    overlaps = {}
    for registry_name, registry in (("used", used), ("reserve", reserve)):
        overlaps[registry_name] = {
            kind: sorted(current[kind] & set(registry.get(f"{kind}_hashes", [])))
            for kind in current
        }
    if role != "spent_smoke":
        require(reserve.get("complete") is True, "future cohort requires complete hash-only reserve registry")
        require(not any(overlaps["used"][kind] for kind in current), "future cohort overlaps v70-v77 used identity")
        require(not any(overlaps["reserve"][kind] for kind in current), "future cohort overlaps forbidden identity")
        require(manifest.get("raw_draws_observable") is True, "future cohort lacks prefilter raw draws")
        for kind in ("route", "day", "bag", "trajectory", "window"):
            require(
                all(row.get(f"{kind}_hash") for row in states),
                f"future cohort missing {kind} identity",
            )
        require(all(timestamps_complete_and_monotonic(row) for row in states), "future cohort timestamp contract incomplete/nonmonotonic")
        require(all(camera_complete(row) for row in states), "future cohort image metadata contract incomplete")
    else:
        require(manifest.get("evidence_role") == "pipeline_smoke_only_spent_nonreserve", "spent smoke evidence role mismatch")

    against_overlap = {}
    for path in args.against:
        reject_forbidden_path(path)
        other = identity_sets(read_states(path))
        against_overlap[str(path)] = {
            kind: sorted(current[kind] & other[kind]) for kind in current
        }
        require(not against_overlap[str(path)]["route"], "development/validation route overlap")
        require(not against_overlap[str(path)]["trajectory"], "development/validation trajectory overlap")

    audit = {
        "schema": "proposal_tail_dataset_validation_v1",
        "status": "pass",
        "collection": str(args.collection),
        "collection_manifest_sha256": sha256_file(manifest_path),
        "cohort_role": role,
        "states": len(states),
        "candidate_shape": list(candidates.shape),
        "raw_draws_observable": bool(manifest["raw_draws_observable"]),
        "used_identity_overlap_expected_only_for_spent_smoke": overlaps["used"],
        "reserve_identity_overlap": overlaps["reserve"],
        "against_overlap": against_overlap,
        "route_leakage": False,
        "reserve_content_accessed": False,
        "performance_evidence_allowed": False,
    }
    if args.audit_output is not None:
        args.audit_output.parent.mkdir(parents=True, exist_ok=True)
        args.audit_output.write_text(canonical_json(audit) + "\n", encoding="utf-8")
    return audit


if __name__ == "__main__":
    print(canonical_json(validate()))
