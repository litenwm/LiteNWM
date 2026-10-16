#!/usr/bin/env python3
"""Freeze identity-disjoint cem-cal confirmation subsets without outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tools.evaluation.build_cem_cal_dev_subsets_v1 import (  # noqa: E402
    contaminated_recon_trajectories,
    load_rows,
    write_jsonl,
)
from tools.scoring.train_recon_dense_meta_dinov3_v1 import (  # noqa: E402
    file_sha256,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--identity-audit",
        type=Path,
        default=REPO_ROOT
        / "outputs/cem_development_v1/preflight/cem_cal_access_audit.json",
    )
    parser.add_argument(
        "--dev-audit",
        type=Path,
        default=REPO_ROOT
        / "outputs/cem_development_v1/manifests/cem_cal_dev_subsets.audit.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs/cem_development_v1/manifests",
    )
    parser.add_argument("--states", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260729)
    return parser.parse_args()


def identity_sets(
    rows: list[dict[str, Any]], domain: str
) -> dict[str, set[str]]:
    route_key = "route_id" if domain == "recon" else "route_group"
    return {
        "record": {
            f"{row['trajectory_id']}:{int(row['current_t'])}" for row in rows
        },
        "trajectory": {str(row["trajectory_id"]) for row in rows},
        "route": {str(row[route_key]) for row in rows},
        "bag": {
            str(row["bag_id"])
            for row in rows
            if row.get("bag_id") is not None
        },
    }


def hash_key(seed: int, row: dict[str, Any]) -> str:
    return hashlib.sha256(
        (
            f"{seed}|{row['trajectory_id']}|{int(row['current_t'])}|"
            f"{row.get('route_id', row.get('route_group'))}|{row.get('bag_id', '')}"
        ).encode()
    ).hexdigest()


def main() -> None:
    args = parse_args()
    if args.states < 40:
        raise ValueError("confirmation target must be at least 40 states")
    identity_audit = json.loads(
        args.identity_audit.read_text(encoding="utf-8")
    )
    dev_audit = json.loads(args.dev_audit.read_text(encoding="utf-8"))
    if (
        identity_audit.get("status") != "identity_only_preflight_complete"
        or dev_audit.get("labels_opened")
    ):
        raise RuntimeError("identity/development audit contract changed")
    contaminated = contaminated_recon_trajectories(identity_audit)
    domains: dict[str, Any] = {}
    for offset, domain in enumerate(("recon", "scand")):
        source = Path(identity_audit["domains"][domain]["cem_cal"]["manifest"])
        dev_manifest = Path(dev_audit["domains"][domain]["output"])
        source_rows = load_rows(source)
        dev_rows = load_rows(dev_manifest)
        used = identity_sets(dev_rows, domain)
        route_key = "route_id" if domain == "recon" else "route_group"
        clean = []
        exclusion_counts = {
            "record": 0,
            "trajectory": 0,
            "route": 0,
            "bag": 0,
            "historical_recon_trajectory": 0,
        }
        for row in source_rows:
            record = f"{row['trajectory_id']}:{int(row['current_t'])}"
            trajectory = str(row["trajectory_id"])
            route = str(row[route_key])
            bag = str(row.get("bag_id", ""))
            reasons = []
            if record in used["record"]:
                reasons.append("record")
            if trajectory in used["trajectory"]:
                reasons.append("trajectory")
            if route in used["route"]:
                reasons.append("route")
            if domain == "scand" and bag in used["bag"]:
                reasons.append("bag")
            if domain == "recon" and trajectory in contaminated:
                reasons.append("historical_recon_trajectory")
            if reasons:
                for reason in set(reasons):
                    exclusion_counts[reason] += 1
            else:
                clean.append(row)

        ordered = sorted(clean, key=lambda row: hash_key(args.seed + offset, row))
        selected: list[dict[str, Any]] = []
        selected_trajectory: set[str] = set()
        selected_route: set[str] = set()
        selected_bag: set[str] = set()
        for row in ordered:
            trajectory = str(row["trajectory_id"])
            route = str(row[route_key])
            bag = str(row.get("bag_id", ""))
            if trajectory in selected_trajectory or route in selected_route:
                continue
            if domain == "scand" and bag in selected_bag:
                continue
            selected.append(row)
            selected_trajectory.add(trajectory)
            selected_route.add(route)
            if domain == "scand":
                selected_bag.add(bag)
            if len(selected) == args.states:
                break

        output = (
            args.output_dir
            / f"{domain}_cem_cal_confirmation{args.states}.windows.jsonl"
        )
        output_audit = output.with_suffix(".audit.json")
        if output.exists() or output_audit.exists():
            raise FileExistsError(output)
        write_jsonl(output, selected)
        selected_identity = identity_sets(selected, domain)
        overlap = {
            key: len(selected_identity[key].intersection(used[key]))
            for key in ("record", "trajectory", "route", "bag")
        }
        status = (
            "label_sealed_confirmation_frozen"
            if len(selected) == args.states
            else "blocked_insufficient_strictly_disjoint_cem_cal"
        )
        report = {
            "schema": "cem_cal_confirmation_subset_v1",
            "domain": domain,
            "status": status,
            "requested_states": args.states,
            "states": len(selected),
            "trajectories": len(selected_identity["trajectory"]),
            "route_groups": len(selected_identity["route"]),
            "bags": len(selected_identity["bag"]),
            "source_manifest": str(source),
            "source_manifest_sha256": file_sha256(source),
            "development_manifest": str(dev_manifest),
            "development_manifest_sha256": file_sha256(dev_manifest),
            "strict_remaining_rows_before_unique_group_selection": len(clean),
            "development_overlap": overlap,
            "exclusion_counts_not_mutually_exclusive": exclusion_counts,
            "selection": (
                "hash_deterministic_greedy_unique_trajectory_route_bag"
            ),
            "seed": args.seed + offset,
            "output": str(output),
            "output_sha256": file_sha256(output),
            "labels_opened": False,
            "outcomes_opened": False,
            "cem": False,
        }
        write_json(output_audit, report)
        domains[domain] = report
    combined = args.output_dir / "cem_cal_confirmation_subsets.audit.json"
    write_json(
        combined,
        {
            "schema": "dual_domain_cem_cal_confirmation_subsets_v1",
            "identity_audit": str(args.identity_audit),
            "identity_audit_sha256": file_sha256(args.identity_audit),
            "development_audit": str(args.dev_audit),
            "development_audit_sha256": file_sha256(args.dev_audit),
            "domains": domains,
            "labels_opened": False,
            "outcomes_opened": False,
            "cem": False,
        },
    )
    print(
        json.dumps(
            {
                "combined": str(combined),
                "combined_sha256": file_sha256(combined),
                "domains": domains,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
