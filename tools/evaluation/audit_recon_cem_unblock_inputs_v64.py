#!/usr/bin/env python3
"""Read-only inventory of lawful inputs that could unblock frozen CEM v63.

This audit never loads outcome artifacts, source_labeled files, or any reserve
array.  Raw RECON readiness is determined from image filenames only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np


REPO = Path(__file__).resolve().parents[2]
ROUTE_SUFFIX = re.compile(r"_r\d+$")
TRAJECTORY_PATTERN = re.compile(
    r"^jackal_\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d+_\d+_r\d+$"
)
SENSOR_TOKENS = (
    "odom",
    "odometry",
    "imu",
    "accelerometer",
    "gyroscope",
    "depth",
    "lidar",
    "laser",
    "scan",
    "pointcloud",
    "velodyne",
    "controller_state",
    "wheel_velocity",
    "twist",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def json_rows(path: Path) -> Iterable[dict[str, Any]]:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"non-object row in {path}")
            yield value


def route_id(trajectory: str) -> str:
    return ROUTE_SUFFIX.sub("", trajectory)


def numeric_images(path: Path) -> set[int]:
    return {
        int(item.stem)
        for item in path.glob("*.jpg")
        if item.stem.isdigit()
    }


def valid_h4_currents(path: Path) -> list[int]:
    images = numeric_images(path)
    return sorted(
        current
        for current in images
        if current >= 3
        and all(index in images for index in range(current - 3, current + 1))
        and current + 4 in images
    )


def directory_identities(path: Path) -> set[str]:
    if not path.is_dir():
        return set()
    return {
        item.name
        for item in path.iterdir()
        if item.is_dir() and TRAJECTORY_PATTERN.match(item.name)
    }


def npz_schema(path: Path) -> dict[str, list[int]]:
    with np.load(path, allow_pickle=False) as payload:
        return {key: list(payload[key].shape) for key in payload.files}


def find_sensor_keys(schema: dict[str, list[int]]) -> list[str]:
    return sorted(
        key
        for key in schema
        if any(token in key.lower() for token in SENSOR_TOKENS)
    )


def source_lines(path: Path, terms: tuple[str, ...]) -> list[dict[str, Any]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [
        {"line": number, "text": text.strip()}
        for number, text in enumerate(lines, 1)
        if any(term.lower() in text.lower() for term in terms)
    ]


def manifest_path(bank: str) -> Path:
    if bank in tuple("abcdef"):
        return (
            REPO
            / "outputs/cem_development_v1/recon_target_calibration_v20"
            / f"bank_{bank}"
            / "manifest.windows.jsonl"
        )
    return (
        REPO
        / "outputs/cem_development_v1/recon_cem_boundary_v60"
        / f"bank_{bank}_v60"
        / "manifest.windows.jsonl"
    )


def bank_root(bank: str) -> Path:
    if bank in tuple("abcdef"):
        return (
            REPO
            / "outputs/cem_development_v1/recon_target_calibration_v20"
            / f"bank_{bank}"
        )
    return (
        REPO
        / "outputs/cem_development_v1/recon_cem_boundary_v60"
        / f"bank_{bank}_v60"
    )


def lawful_npz_paths(root: Path) -> list[Path]:
    prohibited = ("outcome", "labeled", "pose_target")
    return sorted(
        path
        for path in root.rglob("*.npz")
        if not any(token in path.name.lower() for token in prohibited)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    v63_path = REPO / "outputs/cem_development_v1/recon_cem_identity_audit_v63/audit/result.json"
    v63 = read_json(v63_path)
    if v63.get("decision") != "fallback_to_frozen_67":
        raise RuntimeError("v63 authority is not frozen 67")
    reserve_path = (
        REPO
        / "outputs/cem_development_v1/recon_target_calibration_v20"
        / "final_confirmation_reserve/selection_v26/freeze.json"
    )
    reserve_sha_before = sha256(reserve_path)
    reserve = read_json(reserve_path)
    if reserve.get("labels_opened") or reserve.get("outcomes_accessed"):
        raise RuntimeError("reserve is not sealed")

    canonical_root = Path("/home/rainbow/nwm/data/recon")
    manifests: dict[str, Any] = {}
    cache_sensor_hits: dict[str, list[dict[str, Any]]] = {}
    all_records: list[dict[str, Any]] = []
    all_routes: set[str] = set()
    all_trajectories: set[str] = set()
    raw_auxiliary_files: dict[str, list[str]] = {}
    context_ready = 0
    k17_action_cache_states = 0
    cache_files_scanned = 0
    for bank in tuple("abcdefgh"):
        path = manifest_path(bank)
        rows = list(json_rows(path))
        root = bank_root(bank)
        bank_routes = {
            str(row.get("route_id", route_id(str(row["trajectory_id"]))))
            for row in rows
        }
        bank_trajectories = {str(row["trajectory_id"]) for row in rows}
        all_routes.update(bank_routes)
        all_trajectories.update(bank_trajectories)
        all_records.extend(rows)
        manifest_keys = sorted({key for row in rows for key in row})
        for row in rows:
            trajectory = str(row["trajectory_id"])
            current = int(row["current_t"])
            images = numeric_images(canonical_root / trajectory)
            if all(index in images for index in range(current - 3, current + 1)):
                context_ready += 1

        k17_path = root / "k17.npz"
        k17_schema = npz_schema(k17_path)
        candidate_keys = [
            key
            for key in k17_schema
            if "candidate" in key.lower() or "action" in key.lower()
        ]
        if candidate_keys:
            k17_action_cache_states += len(rows)

        hits: list[dict[str, Any]] = []
        schemas: dict[str, Any] = {}
        for cache_path in lawful_npz_paths(root):
            schema = npz_schema(cache_path)
            cache_files_scanned += 1
            relative = str(cache_path.relative_to(REPO))
            schemas[relative] = schema
            sensor_keys = find_sensor_keys(schema)
            if sensor_keys:
                hits.append({"path": relative, "keys": sensor_keys})
        cache_sensor_hits[bank] = hits
        manifests[bank] = {
            "path": str(path.relative_to(REPO)),
            "sha256": sha256(path),
            "states": len(rows),
            "routes": len(bank_routes),
            "trajectories": len(bank_trajectories),
            "row_keys": manifest_keys,
            "k17_path": str(k17_path.relative_to(REPO)),
            "k17_sha256": sha256(k17_path),
            "k17_keys": k17_schema,
            "candidate_or_action_keys": candidate_keys,
            "lawful_cache_files_scanned": len(schemas),
            "sensor_key_hits": hits,
        }

    for trajectory in sorted(all_trajectories):
        path = canonical_root / trajectory
        auxiliary = sorted(
            item.name
            for item in path.iterdir()
            if item.is_file()
            and not (item.suffix.lower() == ".jpg" and item.stem.isdigit())
            and item.name != "traj_data.pkl"
        )
        if auxiliary:
            raw_auxiliary_files[trajectory] = auxiliary

    old_audit_path = (
        REPO
        / "outputs/cem_development_v1/recon_nomad_goal_condition_v53"
        / "hard_blocker_reaudit_v54.json"
    )
    old_audit = read_json(old_audit_path)
    old_decomposition = old_audit["history_decomposition_within_strict_test"]
    prior_unused = set(old_decomposition["unused_route_ids"])
    prior_h4 = set(old_decomposition["unused_h4_ready_detail"])
    gh_routes = set()
    for bank in ("g", "h"):
        gh_routes.update(
            str(row.get("route_id", route_id(str(row["trajectory_id"]))))
            for row in json_rows(manifest_path(bank))
        )
    if gh_routes != prior_h4 or len(gh_routes) != 22:
        raise RuntimeError("spent G/H routes do not exactly consume prior 22 H4 routes")
    remaining_routes = prior_unused - gh_routes

    strict_path = REPO / "data/split_lock/recon_strict_splits.json"
    strict = read_json(strict_path)
    strict_test = [str(value) for value in strict["test"]]
    remaining_detail: dict[str, list[dict[str, Any]]] = {}
    for trajectory in strict_test:
        route = route_id(trajectory)
        if route not in remaining_routes:
            continue
        currents = valid_h4_currents(canonical_root / trajectory)
        remaining_detail.setdefault(route, []).append(
            {
                "trajectory_id": trajectory,
                "h4_ready_currents": len(currents),
                "has_at_least_four_h4_windows": len(currents) >= 4,
            }
        )
    remaining_h4_routes = sorted(
        route
        for route, entries in remaining_detail.items()
        if any(entry["has_at_least_four_h4_windows"] for entry in entries)
    )

    data_roots = [
        canonical_root,
        Path("/home/rainbow/rrnwm/data/recon"),
        Path("/home/rainbow/DWM/recon_data/recon"),
        Path("/home/rainbow/DWM/data/raw/recon"),
        Path("/home/rainbow/nwm/cdit_xl_results/nwm_cdit_xl_cdit_xl_ego4d_200000/recon"),
        Path("/home/rainbow/nwm/cdit_xl_results/gt/recon"),
    ]
    canonical_identities = directory_identities(canonical_root)
    root_inventory = {}
    for root in data_roots:
        identities = directory_identities(root)
        root_inventory[str(root)] = {
            "exists": root.is_dir(),
            "trajectory_directories": len(identities),
            "routes": len({route_id(value) for value in identities}),
            "new_trajectories_vs_canonical": sorted(identities - canonical_identities),
        }

    constraint_path = REPO / "src/planning/trajectory_constraints.py"
    scand_odom_path = REPO / "tools/recon/process_scand_bags_rosbags.py"
    recon_label_path = REPO / "tools/evaluation/build_recon100_k17_source_v1.py"
    feature_path = REPO / "src/scoring/features.py"
    sensor_cache_hit_count = sum(len(value) for value in cache_sensor_hits.values())
    states = len(all_records)
    if states != 688 or context_ready != states or k17_action_cache_states != states:
        raise RuntimeError("A-H state/action cache identity contract changed")

    signals = {
        "controller_or_kinematic_feasibility": {
            "exists": True,
            "online_observable": True,
            "coverage_states": states,
            "coverage_fraction": 1.0,
            "cached_input": "K17/CEM candidate actions",
            "independent_new_safety_input": False,
            "already_used": True,
            "why_not_an_unblock": (
                "The implementation uses candidate action bounds, train-fit support, "
                "roughness, jerk and curvature only. It has no measured controller state, "
                "tracking error, wheel speed, slip or actuator feedback. These action-only "
                "features/support were already used by the current candidate filter and the "
                "v35/v37-v40 physical/action family."
            ),
            "evidence": {
                "constraint_source": str(constraint_path.relative_to(REPO)),
                "constraint_source_sha256": sha256(constraint_path),
                "relevant_lines": source_lines(
                    constraint_path,
                    ("no pose", "roughness", "jerk", "curvature", "in_bounds"),
                ),
                "action_feature_source": str(feature_path.relative_to(REPO)),
                "action_feature_source_sha256": sha256(feature_path),
            },
        },
        "recon_odometry_or_imu": {
            "exists_in_A_to_H_cache": False,
            "coverage_states": 0,
            "time_aligned_at_or_before_t": False,
            "online_contract_present": False,
            "independent_new_safety_input": False,
            "raw_traj_data_note": (
                "RECON traj_data.pkl is retained for pose/outcome evaluation, but no "
                "timestamped current-only odometry/IMU field is exported into A-H. The "
                "audit does not open this pickle and does not treat a future-containing pose "
                "array as a causal input."
            ),
            "scand_only_interface": {
                "path": str(scand_odom_path.relative_to(REPO)),
                "sha256": sha256(scand_odom_path),
                "relevant_lines": source_lines(
                    scand_odom_path, ("ODOM_TOPICS", "odom_topic", "timestamp")
                ),
            },
            "recon_pose_label_evidence": {
                "path": str(recon_label_path.relative_to(REPO)),
                "sha256": sha256(recon_label_path),
                "relevant_lines": source_lines(
                    recon_label_path, ("traj_data.pkl", "position", "yaw")
                ),
            },
        },
        "depth_or_lidar": {
            "exists_in_A_to_H_cache": False,
            "coverage_states": 0,
            "time_aligned_at_or_before_t": False,
            "online_contract_present": False,
            "independent_new_safety_input": False,
            "raw_auxiliary_sensor_file_trajectories": len(raw_auxiliary_files),
        },
    }

    no_new_signal = (
        not signals["controller_or_kinematic_feasibility"][
            "independent_new_safety_input"
        ]
        and not signals["recon_odometry_or_imu"]["independent_new_safety_input"]
        and not signals["depth_or_lidar"]["independent_new_safety_input"]
        and sensor_cache_hit_count == 0
    )
    no_fresh_h4_routes = len(remaining_h4_routes) == 0
    status = "technically_blocked" if no_new_signal or no_fresh_h4_routes else "unblocked"

    reserve_sha_after = sha256(reserve_path)
    if reserve_sha_after != reserve_sha_before:
        raise RuntimeError("reserve changed during read-only audit")
    report = {
        "schema": "recon_cem_unblock_input_inventory_v64",
        "status": status,
        "authority": {
            "v63_result": str(v63_path.relative_to(REPO)),
            "v63_result_sha256": sha256(v63_path),
            "accepted_selection": "frozen_67_only",
            "coverage_gain": 67 / 688,
            "target_coverage_gain": 0.10,
            "minimum_additional_promotions_needed": 2,
        },
        "A_to_H": {
            "states": states,
            "routes": len(all_routes),
            "trajectories": len(all_trajectories),
            "context_RGB_at_or_before_t_ready_states": context_ready,
            "K17_action_cache_states": k17_action_cache_states,
            "lawful_npz_files_scanned": cache_files_scanned,
            "sensor_named_cache_hits": cache_sensor_hits,
            "sensor_named_cache_hit_count": sensor_cache_hit_count,
            "raw_auxiliary_files_excluding_numeric_RGB_and_traj_data": raw_auxiliary_files,
            "banks": manifests,
        },
        "signal_inventory": signals,
        "data_inventory": {
            "strict_split": str(strict_path.relative_to(REPO)),
            "strict_split_sha256": sha256(strict_path),
            "prior_v54_audit": str(old_audit_path.relative_to(REPO)),
            "prior_v54_audit_sha256": sha256(old_audit_path),
            "prior_unused_routes": len(prior_unused),
            "prior_unused_h4_ready_routes": len(prior_h4),
            "spent_GH_routes": len(gh_routes),
            "spent_GH_exactly_equals_prior_H4_ready": gh_routes == prior_h4,
            "remaining_unused_nonreserve_routes": len(remaining_routes),
            "remaining_unused_route_detail": remaining_detail,
            "remaining_unused_H4_ready_routes": len(remaining_h4_routes),
            "remaining_unused_H4_ready_route_ids": remaining_h4_routes,
            "data_roots": root_inventory,
        },
        "one_time_validation_protocol": None,
        "why_no_protocol_is_frozen": (
            "No independent, deployment-observable signal has complete A-H coverage and a "
            "past-only cache; no unused non-reserve H4-ready route remains. Freezing a test "
            "now would substitute an already-used action feature or consume the reserve."
        ),
        "minimum_external_requirements": {
            "signal_contract": {
                "required": (
                    "At least one independently measured, timestamped safety stream available "
                    "at t or earlier: preferably controller command-vs-measured tracking state "
                    "(linear/angular velocity, wheel speed/slip or tracking error); otherwise "
                    "odometry+IMU yaw-rate/acceleration, or synchronized depth/LiDAR."
                ),
                "per_frame_fields": [
                    "sensor_timestamp",
                    "image_timestamp",
                    "explicit synchronization error/tolerance",
                    "validity/missing-data mask",
                    "measured values in documented physical units",
                ],
                "causal_window": "only samples with timestamp <= current image time t",
                "coverage": "100% of development and confirmation windows, or an explicit causal abstain on missing data",
                "candidate_use": (
                    "The state measurement may condition candidate-level feasibility/safety; "
                    "safe elite updates must still require the frozen candidate scorer/support "
                    "and may never use future pose/outcome."
                ),
            },
            "route_contract_if_AH_can_be_backfilled": {
                "new_route_disjoint_H4_ready_routes": 22,
                "allocation": "two immutable 11-route x 4-window cohorts (44+44 states)",
                "reason": (
                    "A-H can serve as spent development/calibration only if the same new signal "
                    "can be causally backfilled for all 688 states. The 22 new routes then supply "
                    "two independent confirmation cohorts."
                ),
            },
            "route_contract_if_AH_cannot_be_backfilled": {
                "new_route_disjoint_H4_ready_routes": 44,
                "allocation": (
                    "22 routes for route-disjoint development/calibration plus two sealed "
                    "11-route x 4-window confirmation cohorts"
                ),
                "reason": "The new signal otherwise has no lawful labeled development bank on which to freeze its safety rule.",
            },
            "success_gate_unchanged": {
                "coverage_gain_at_least": 0.10,
                "dual_precision_at_least": 0.85,
                "harm_rate_at_most": 0.15,
                "all_four_mean_and_p90_nondegraded": True,
                "independent_cohort_direction_count_at_least": 2,
            },
        },
        "prohibitions_observed": {
            "outcome_artifacts_loaded": False,
            "source_labeled_loaded": False,
            "reserve_arrays_loaded": False,
            "reserve_labels_or_outcomes_loaded": False,
            "candidates_generated": False,
            "model_trained": False,
            "thresholds_tuned": False,
            "archive_reselected": False,
            "CEM_budget_changed": False,
        },
        "reserve": {
            "path": str(reserve_path.relative_to(REPO)),
            "sha256_before": reserve_sha_before,
            "sha256_after": reserve_sha_after,
            "labels_opened": bool(reserve.get("labels_opened", False)),
            "outcomes_accessed": bool(reserve.get("outcomes_accessed", False)),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "status": status,
                "states": states,
                "routes": len(all_routes),
                "sensor_cache_hits": sensor_cache_hit_count,
                "spent_GH_routes": len(gh_routes),
                "remaining_unused_routes": len(remaining_routes),
                "remaining_unused_H4_ready_routes": len(remaining_h4_routes),
                "reserve_sha256": reserve_sha_after,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
