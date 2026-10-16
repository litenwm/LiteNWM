#!/usr/bin/env python3
"""Freeze two 11-route small fresh-development manifests from unused RECON routes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scoring.strict_inner_partitions import recon_route_id  # noqa: E402
from tools.evaluation.build_recon_fresh_outer_manifest_v2 import (  # noqa: E402
    numeric_images,
    stable_key,
    used_identity,
    valid_currents,
    write_jsonl,
)
from tools.evaluation.export_recon_dense_dinov3_features_v1 import (  # noqa: E402
    write_json_atomic,
)
from tools.scoring.train_recon_dense_meta_dinov3_v1 import file_sha256  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--boundary-freeze", type=Path, required=True)
    parser.add_argument("--trajectory-root", type=Path, required=True)
    parser.add_argument("--split-lock", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261060)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    allocation_path = args.output_root / "GH_manifest_freeze.json"
    if allocation_path.exists():
        raise FileExistsError(allocation_path)
    inventory = json.loads(args.inventory.read_text(encoding="utf-8"))
    boundary = json.loads(args.boundary_freeze.read_text(encoding="utf-8"))
    if (
        boundary.get("status")
        != "unique_boundary_rule_frozen_before_new_GH_manifests"
        or boundary.get("G_H_outcomes_accessed")
        or boundary.get("reserve_accessed")
        or inventory["remaining_unused_recon_routes"]["remaining"] != 22
    ):
        raise RuntimeError("unused22/boundary contract invalid")
    split_lock = json.loads(args.split_lock.read_text(encoding="utf-8"))
    test_trajectories = set(split_lock["recon"]["splits"]["test"])
    used_trajectories, scan = used_identity(
        [ROOT / "outputs", Path("/home/rainbow/ndwm/DWM/reports")]
    )
    used_routes = {recon_route_id(value) for value in used_trajectories}
    inventory_routes = {
        str(item["route_id"]): item
        for item in inventory["remaining_unused_recon_routes"]["routes"]
    }
    if len(inventory_routes) != 22:
        raise RuntimeError("unused route inventory no longer has 22 unique routes")
    overlap = sorted(set(inventory_routes) & used_routes)
    if overlap:
        raise RuntimeError(f"unused22 routes were consumed after inventory: {overlap}")

    candidates = {}
    for route, item in inventory_routes.items():
        choices = []
        for trajectory in item["trajectories"]:
            trajectory_id = str(trajectory["trajectory_id"])
            path = args.trajectory_root / trajectory_id
            if trajectory_id not in test_trajectories or not path.is_dir():
                continue
            if recon_route_id(trajectory_id) != route:
                raise RuntimeError("inventory route identity changed")
            currents = valid_currents(numeric_images(path))
            if len(currents) >= 4 and (path / "traj_data.pkl").is_file():
                choices.append(
                    {"trajectory_id": trajectory_id, "valid_currents": currents}
                )
        if not choices:
            raise RuntimeError(f"route no longer H4-ready: {route}")
        candidates[route] = min(
            choices,
            key=lambda value: stable_key(
                args.seed, "v60-trajectory", route, value["trajectory_id"]
            ),
        )
    route_order = sorted(
        candidates,
        key=lambda route: stable_key(args.seed, "v60-route-allocation", route),
    )
    groups = {
        "bank_g_v60": route_order[:11],
        "bank_h_v60": route_order[11:],
    }
    if len(groups["bank_g_v60"]) != 11 or len(groups["bank_h_v60"]) != 11:
        raise RuntimeError("v60 route allocation shape changed")

    cohort_audits = {}
    route_sets = {}
    for cohort, routes in groups.items():
        rows = []
        for route in routes:
            candidate = candidates[route]
            currents = candidate["valid_currents"]
            positions = np.linspace(0, len(currents) - 1, 4).round().astype(int)
            chosen = [currents[index] for index in positions]
            if len(set(chosen)) != 4:
                raise RuntimeError("temporal bin selection duplicated")
            for temporal_bin, current in enumerate(chosen):
                window_uid = hashlib.sha256(
                    (
                        f"recon-cem-v60|{args.seed}|{cohort}|"
                        f"{candidate['trajectory_id']}|{current}"
                    ).encode()
                ).hexdigest()[:24]
                rows.append(
                    {
                        "dataset": "recon",
                        "partition": "fresh-outer",
                        "protocol": "small-fresh-development-v60-11route-44state",
                        "cohort": cohort,
                        "trajectory_id": candidate["trajectory_id"],
                        "current_t": int(current),
                        "history_timesteps": list(range(current - 3, current + 1)),
                        "goal_t": int(current + 4),
                        "route_id": route,
                        "route_group": route,
                        "temporal_bin": temporal_bin,
                        "window_uid": window_uid,
                    }
                )
        output_dir = args.output_root / cohort
        manifest_path = output_dir / "manifest.windows.jsonl"
        audit_path = output_dir / "manifest.windows.audit.json"
        if manifest_path.exists() or audit_path.exists():
            raise FileExistsError(manifest_path)
        write_jsonl(manifest_path, rows)
        audit = {
            "schema": "recon_small_fresh_development_manifest_v60",
            "status": "11route_44state_manifest_frozen_before_outcomes",
            "protocol_is_original_H4_100": False,
            "routes": 11,
            "states": 44,
            "windows_per_route": 4,
            "manifest": str(manifest_path),
            "manifest_sha256": file_sha256(manifest_path),
            "seed": args.seed,
            "route_ids": routes,
            "labels_opened": False,
            "outcomes_accessed": False,
            "future_pose_opened": False,
            "reserve_accessed": False,
        }
        write_json_atomic(audit_path, audit)
        cohort_audits[cohort] = {
            "manifest": str(manifest_path),
            "manifest_sha256": file_sha256(manifest_path),
            "audit": str(audit_path),
            "audit_sha256": file_sha256(audit_path),
            "routes": routes,
        }
        route_sets[cohort] = set(routes)
    if route_sets["bank_g_v60"] & route_sets["bank_h_v60"]:
        raise RuntimeError("new G/H route overlap")

    allocation = {
        "schema": "recon_small_fresh_GH_v60_manifest_freeze",
        "status": "all_22_routes_and_88_windows_frozen_before_any_outcome",
        "protocol_is_original_25route_H4_100": False,
        "seed": args.seed,
        "allocation_rule": "SHA256(seed, v60-route-allocation, route); first 11 G, remaining 11 H",
        "trajectory_rule": "minimum SHA256(seed, v60-trajectory, route, trajectory_id)",
        "window_rule": "four rounded linspace positions over H4-ready RGB currents",
        "cohorts": cohort_audits,
        "combined_routes": 22,
        "combined_states": 88,
        "route_overlap": 0,
        "used_identity_rescan": scan,
        "used_route_overlap_before_freeze": overlap,
        "inventory": str(args.inventory),
        "inventory_sha256": file_sha256(args.inventory),
        "boundary_freeze": str(args.boundary_freeze),
        "boundary_freeze_sha256": file_sha256(args.boundary_freeze),
        "split_lock": str(args.split_lock),
        "split_lock_sha256": file_sha256(args.split_lock),
        "success_added_windows_min": 9,
        "labels_opened": False,
        "outcomes_accessed": False,
        "post_outcome_regroup_or_window_fill_allowed": False,
        "all_22_routes_become_spent_after_evaluation": True,
        "sealed_25route_reserve_accessed": False,
        "production_safe_claim_allowed": False,
    }
    write_json_atomic(allocation_path, allocation)
    print(json.dumps(allocation, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
