#!/usr/bin/env python3
"""Preregister and audit label-free SCAND replacement-route sufficiency."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from planning.proposal_tail_sidecar_v1 import canonical_json, sha256_file  # noqa: E402


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def read_json(path: Path) -> Any:
    require(path.is_file(), f"missing JSON: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    require(path.is_file(), f"missing JSONL: {path}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    require(not path.exists(), f"refusing to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(canonical_json(payload) + "\n", encoding="utf-8")
    tmp.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity-inventory", type=Path, required=True)
    parser.add_argument("--camera-catalog", type=Path, required=True)
    parser.add_argument("--all-spent-registry", type=Path, required=True)
    parser.add_argument("--development-manifest", type=Path, required=True)
    parser.add_argument("--sealed-validation-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def wilson_bounds(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total == 0:
        return 0.0, 1.0
    p = successes / total
    den = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / den
    radius = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / den
    return center - radius, center + radius


def main() -> None:
    args = parse_args()
    generated = datetime.now(timezone.utc).isoformat()
    prereg = {
        "schema": "scand_replacement_route_preregistration_v1",
        "status": "frozen_before_topic_duration_or_outcome_inspection_of_unknown_candidates",
        "purpose": "Determine whether public official candidates can supply enough clean development route groups for a materially new incremental-risk representation.",
        "selection_inputs_allowed": [
            "hash-only bag/route/day/trajectory/window identity",
            "robot",
            "route endpoints",
            "collection day",
            "official public split metadata",
            "camera topic names and message counts",
            "bag/camera duration",
        ],
        "selection_inputs_forbidden": [
            "ATE/RPE/endpoint/objective outcomes",
            "candidate dual/harm labels",
            "model scores",
            "validation labels",
            "reserve contents",
        ],
        "deterministic_input_eligibility": {
            "identity": "bag and route hashes absent from the all-spent and forbidden union",
            "lineage": "not part of the 111-bag original DWM train set",
            "camera": "one canonical main RGB compressed topic: /image_raw/compressed, /camera/rgb/image_raw/compressed, or /left/image_color/compressed",
            "duration": "at least 40 valid 4-Hz H4 windows under the frozen sampler",
            "contract": "timestamps, odometry/action and 320x240 decoded-frame contract validate fail-closed",
            "unknown_topic_rule": "inspect/download at most one bag at a time only while the remaining maximum eligible count can still meet the power floor",
        },
        "cohort_rule": {
            "development_only": True,
            "existing_development_routes": 4,
            "states_per_route_cap": 40,
            "minimum_total_development_routes": 10,
            "minimum_additional_routes": 6,
            "reason": "Five route-grouped folds need at least two groups/fold, and ~7% selective coverage needs >=22 all-dual/zero-harm replacements for Wilson LCB>=0.85 and harm UCB<=0.15.",
        },
        "unknown_candidate_stop_rule": "Do not download an unknown-topic candidate when even every remaining unknown becoming eligible cannot reach 10 total development routes.",
        "sealed_validation": "Existing validation manifest remains identity-only and labels unopened; it cannot fill the development-route floor.",
        "formal_v70_mutated": False,
        "training_started": False,
        "reserve_content_accessed": False,
        "outcomes_opened_for_candidate_selection": False,
        "generated_at_utc": generated,
    }
    prereg_path = args.output_dir / "preregistration.json"
    atomic_json(prereg_path, prereg)

    inventory = read_json(args.identity_inventory)
    catalog = read_json(args.camera_catalog)
    spent = read_json(args.all_spent_registry)
    require(inventory.get("status") == "identity_ready", "identity inventory is not ready")
    require(spent.get("complete") is True and spent.get("identity_values_are_hash_only") is True, "all-spent registry incomplete")
    dev = read_jsonl(args.development_manifest)
    val = read_jsonl(args.sealed_validation_manifest)
    require(dev and val, "development or validation manifest empty")
    dev_routes = sorted({str(row["route_group"]) for row in dev})
    val_routes = sorted({str(row["route_group"]) for row in val})
    require(len(dev_routes) == 4, f"expected four frozen development routes, got {len(dev_routes)}")
    require(len(val_routes) == 3, f"expected three sealed validation routes, got {len(val_routes)}")
    require(not (set(dev_routes) & set(val_routes)), "development/validation route overlap")

    catalog_by_name = {str(row["file_name"]): row for row in catalog["rows"]}
    admissible_identity = [row for row in inventory["rows"] if row["admissible_after_all_identity_checks"]]
    classified = []
    for row in admissible_identity:
        bag_name = str(row["public_file_name"]) + ".bag"
        known = catalog_by_name.get(bag_name)
        if known is None:
            status = "topic_and_duration_unknown"
            camera = None
        else:
            camera = known.get("camera_contract_status")
            if camera == "main_rgb_compatible":
                status = "known_input_compatible_but_already_assigned_or_requires_separate_identity_review"
            elif camera == "body_fisheye_only_incompatible":
                status = "ineligible_body_fisheye_only"
            elif camera == "main_rgb_compatible_but_too_short":
                status = "ineligible_too_short"
            else:
                status = "unknown_catalog_status_fail_closed"
        classified.append({
            "public_file_name": row["public_file_name"],
            "route_group": row["route_group"],
            "robot": str(row["public_file_name"]).split("_")[1] if "_" in str(row["public_file_name"]) else None,
            "collection_date": row["collection_date"],
            "public_split": row["public_split"],
            "identity_status": "clear",
            "input_contract_status": status,
            "known_camera_contract_status": camera,
        })

    incompatible = [row for row in classified if row["input_contract_status"].startswith("ineligible_")]
    unknown = [row for row in classified if row["input_contract_status"] == "topic_and_duration_unknown"]
    known_compatible = [row for row in classified if row["input_contract_status"].startswith("known_input_compatible")]
    maximum_additional = len(unknown) + len(known_compatible)
    maximum_total = len(dev_routes) + maximum_additional

    minimum_perfect_replacements = None
    for n in range(1, 10000):
        dual_lcb, _ = wilson_bounds(n, n)
        _, harm_ucb = wilson_bounds(0, n)
        if dual_lcb >= 0.85 and harm_ucb <= 0.15:
            minimum_perfect_replacements = n
            break
    require(minimum_perfect_replacements is not None, "Wilson floor search failed")
    observed_selective_coverage = 0.07
    states_needed_at_observed_coverage = math.ceil(minimum_perfect_replacements / observed_selective_coverage)
    routes_needed_at_observed_coverage = math.ceil(states_needed_at_observed_coverage / 40)
    target_routes = max(10, routes_needed_at_observed_coverage)
    sufficient = maximum_total >= target_routes

    result = {
        "schema": "scand_replacement_route_sufficiency_audit_v1",
        "status": "complete_label_free",
        "verdict": "candidate_pool_sufficient_for_sequential_topic_preflight" if sufficient else "official_candidate_pool_insufficient_under_frozen_contract",
        "current_cohorts": {
            "development": {"routes": len(dev_routes), "states": len(dev), "route_groups": dev_routes},
            "sealed_one_time_validation": {"routes": len(val_routes), "states": len(val), "route_groups": val_routes, "labels_opened": False},
        },
        "identity_inventory": {
            "official_public_bags": inventory["counts"]["public_unique_bags"],
            "original_dwm_train_bags_excluded": inventory["counts"]["old_train_bags"],
            "identity_clear_candidates": len(admissible_identity),
            "input_incompatible_known": len(incompatible),
            "input_compatible_known": len(known_compatible),
            "topic_duration_unknown": len(unknown),
            "maximum_additional_routes_if_every_unknown_passes": maximum_additional,
            "maximum_total_development_routes": maximum_total,
        },
        "power_floor": {
            "wilson_confidence": 0.95,
            "required_dual_lcb": 0.85,
            "required_harm_ucb": 0.15,
            "minimum_replacements_if_all_dual_and_zero_harm": minimum_perfect_replacements,
            "observed_strict_selective_coverage_used_for_planning": observed_selective_coverage,
            "minimum_states_at_observed_coverage": states_needed_at_observed_coverage,
            "minimum_routes_at_40_states_each_for_wilson_only": routes_needed_at_observed_coverage,
            "five_fold_route_group_floor": 10,
            "frozen_target_total_routes": target_routes,
            "frozen_target_additional_routes": target_routes - len(dev_routes),
        },
        "decision": {
            "download_unknown_candidate_now": sufficient,
            "reason": "Sequential topic preflight remains worthwhile." if sufficient else "Even every unknown candidate passing leaves fewer than the frozen ten development route groups; downloading cannot repair route-level power.",
            "training_new_representation_started": False,
            "validation_opened": False,
        },
        "candidates": classified,
        "inputs": {
            "preregistration": {"path": str(prereg_path.resolve()), "sha256": sha256_file(prereg_path)},
            "identity_inventory": {"path": str(args.identity_inventory.resolve()), "sha256": sha256_file(args.identity_inventory)},
            "camera_catalog": {"path": str(args.camera_catalog.resolve()), "sha256": sha256_file(args.camera_catalog)},
            "all_spent_registry": {"path": str(args.all_spent_registry.resolve()), "sha256": sha256_file(args.all_spent_registry)},
            "development_manifest": {"path": str(args.development_manifest.resolve()), "sha256": sha256_file(args.development_manifest)},
            "sealed_validation_manifest": {"path": str(args.sealed_validation_manifest.resolve()), "sha256": sha256_file(args.sealed_validation_manifest)},
        },
        "formal_v70_mutated": False,
        "labels_or_outcomes_opened": False,
        "reserve_content_accessed": False,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    result_path = args.output_dir / "result.json"
    atomic_json(result_path, result)
    print(canonical_json({
        "status": result["status"],
        "verdict": result["verdict"],
        "preregistration": {"path": str(prereg_path.resolve()), "sha256": sha256_file(prereg_path)},
        "result": {"path": str(result_path.resolve()), "sha256": sha256_file(result_path)},
        "identity_clear": len(admissible_identity),
        "incompatible": len(incompatible),
        "unknown": len(unknown),
        "maximum_total_routes": maximum_total,
        "target_total_routes": target_routes,
        "validation_opened": False,
    }))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(canonical_json({"status": "fail_closed", "error_type": type(error).__name__, "error": str(error)}), file=sys.stderr)
        raise SystemExit(2) from error
