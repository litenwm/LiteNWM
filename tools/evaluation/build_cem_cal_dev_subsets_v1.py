#!/usr/bin/env python3
"""Freeze small label-sealed RECON/SCAND cem-cal development subsets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tools.scoring.train_recon_dense_meta_dinov3_v1 import (  # noqa: E402
    file_sha256,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audit",
        type=Path,
        default=REPO_ROOT
        / "outputs/cem_development_v1/preflight/cem_cal_access_audit.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs/cem_development_v1/manifests",
    )
    parser.add_argument("--states", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260727)
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def hash_key(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}|{value}".encode()).hexdigest()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(
            json.dumps(value, sort_keys=True) + "\n" for value in rows
        ),
        encoding="utf-8",
    )
    temporary.replace(path)


def contaminated_recon_trajectories(
    audit: dict[str, Any],
) -> set[str]:
    result = set()
    cem_rows = load_rows(
        Path(audit["domains"]["recon"]["cem_cal"]["manifest"])
    )
    record_to_trajectory = {
        f"{row['trajectory_id']}:{int(row['current_t'])}": str(
            row["trajectory_id"]
        )
        for row in cem_rows
    }
    for record in audit["domains"]["recon"][
        "historical_npz_identity_matches"
    ]:
        with np.load(record["path"], allow_pickle=False) as payload:
            observed = payload["record_key"].astype(str)
        result.update(
            record_to_trajectory[value]
            for value in observed
            if value in record_to_trajectory
        )
    return result


def select_route_spread(
    rows: list[dict[str, Any]],
    *,
    states: int,
    seed: int,
    route_key: str,
) -> list[dict[str, Any]]:
    by_route: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_route.setdefault(str(row[route_key]), []).append(row)
    for route, values in by_route.items():
        values.sort(
            key=lambda value: hash_key(
                seed,
                f"{route}|{value['trajectory_id']}|{value['current_t']}",
            )
        )
    routes = sorted(
        by_route,
        key=lambda value: hash_key(seed, f"route|{value}"),
    )
    selected = []
    depth = 0
    while len(selected) < states:
        added = False
        for route in routes:
            if depth < len(by_route[route]):
                selected.append(by_route[route][depth])
                added = True
                if len(selected) == states:
                    break
        if not added:
            break
        depth += 1
    if len(selected) != states:
        raise RuntimeError("insufficient cem-cal rows for subset")
    return selected


def main() -> None:
    args = parse_args()
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    if audit.get("status") != "identity_only_preflight_complete":
        raise RuntimeError("cem-cal identity audit is incomplete")
    if args.states < 40:
        raise ValueError("CEM development subset must have at least 40 states")
    outputs = {}
    contaminated = contaminated_recon_trajectories(audit)
    for offset, domain in enumerate(("recon", "scand")):
        manifest = Path(audit["domains"][domain]["cem_cal"]["manifest"])
        values = load_rows(manifest)
        if domain == "recon":
            clean = [
                row
                for row in values
                if str(row["trajectory_id"]) not in contaminated
            ]
            route_key = "route_id"
        else:
            clean = values
            route_key = "route_group"
        selected = select_route_spread(
            clean,
            states=args.states,
            seed=args.seed + offset,
            route_key=route_key,
        )
        output = args.output_dir / f"{domain}_cem_cal_dev{args.states}.windows.jsonl"
        output_audit = output.with_suffix(".audit.json")
        if output.exists() or output_audit.exists():
            raise FileExistsError(output)
        write_jsonl(output, selected)
        selected_trajectories = {
            str(value["trajectory_id"]) for value in selected
        }
        selected_routes = {str(value[route_key]) for value in selected}
        selected_bags = {
            str(value["bag_id"])
            for value in selected
            if value.get("bag_id") is not None
        }
        report = {
            "schema": "cem_cal_development_subset_v1",
            "domain": domain,
            "status": "label_sealed_subset_frozen",
            "source_manifest": str(manifest),
            "source_manifest_sha256": file_sha256(manifest),
            "source_windows": len(values),
            "output": str(output),
            "output_sha256": file_sha256(output),
            "states": len(selected),
            "trajectories": len(selected_trajectories),
            "route_groups": len(selected_routes),
            "bags": len(selected_bags),
            "selection": "hash_deterministic_route_spread",
            "seed": args.seed + offset,
            "recon_contaminated_trajectories_excluded": (
                sorted(contaminated) if domain == "recon" else []
            ),
            "labels_opened": False,
            "outcomes_opened": False,
            "fixed_pool_baseline_required": True,
            "cem": False,
        }
        write_json(output_audit, report)
        outputs[domain] = report
    combined = args.output_dir / "cem_cal_dev_subsets.audit.json"
    write_json(
        combined,
        {
            "schema": "dual_domain_cem_cal_development_subsets_v1",
            "identity_audit": str(args.audit),
            "identity_audit_sha256": file_sha256(args.audit),
            "domains": outputs,
            "labels_opened": False,
            "cem": False,
        },
    )
    print(
        json.dumps(
            {
                "combined": str(combined),
                "combined_sha256": file_sha256(combined),
                "domains": outputs,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
