#!/usr/bin/env python3
"""Audit whether sealed SCAND holdout trajectories appeared in prior artifacts."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tools.planning.build_scand100_nomad_k17_v1 import sha256, write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--holdout-manifest",
        type=Path,
        default=REPO_ROOT
        / "outputs/scand_domain_adaptation_v1/manifests/internal_holdout_rgb_available.windows.jsonl",
    )
    parser.add_argument(
        "--current-output-root",
        type=Path,
        default=REPO_ROOT / "outputs/scand_domain_adaptation_v1",
    )
    parser.add_argument(
        "--scan-root",
        type=Path,
        action="append",
        default=[
            REPO_ROOT / "outputs",
            Path("/home/rainbow/ndwm/DWM/reports"),
        ],
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT
        / "outputs/scand_domain_adaptation_v1/holdout_freeze_v1/usage_audit.json",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def row_sets(rows: list[dict[str, Any]]) -> dict[str, set[str]]:
    return {
        "record_key": {
            f"{row['trajectory_id']}:{int(row['current_t'])}" for row in rows
        },
        "trajectory": {str(row["trajectory_id"]) for row in rows},
        "route_group": {str(row.get("route_group", "")) for row in rows},
        "bag_id": {str(row.get("bag_id", "")) for row in rows},
    }


def split_audit(holdout: dict[str, set[str]], path: Path) -> dict[str, Any]:
    rows = read_jsonl(path)
    other = row_sets(rows)
    return {
        "path": str(path),
        "sha256": sha256(path),
        "windows": len(rows),
        "record_overlap": sorted(holdout["record_key"] & other["record_key"]),
        "trajectory_overlap": sorted(holdout["trajectory"] & other["trajectory"]),
        "route_group_overlap": sorted(holdout["route_group"] & other["route_group"]),
        "bag_id_overlap": sorted(holdout["bag_id"] & other["bag_id"]),
    }


def candidate_files(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    result = []
    for directory, _, files in os.walk(root):
        base = Path(directory)
        for name in files:
            path = base / name
            if path.suffix.lower() == ".npz":
                result.append(path)
    return sorted(result)


def npz_identity(path: Path) -> tuple[set[str], set[str]]:
    records: set[str] = set()
    trajectories: set[str] = set()
    with np.load(path, allow_pickle=False) as payload:
        if "record_key" in payload.files:
            records = set(payload["record_key"].astype(str).reshape(-1).tolist())
            trajectories.update(
                value.rpartition(":")[0]
                for value in records
                if value.rpartition(":")[1]
            )
        for key in ("trajectory", "trajectory_id", "traj_name"):
            if key in payload.files:
                trajectories.update(
                    payload[key].astype(str).reshape(-1).tolist()
                )
    return records, trajectories


def inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.holdout_manifest)
    holdout = row_sets(rows)
    split_paths = {
        "train_fit": REPO_ROOT / "data/inner_manifests/scand/train-fit.windows.jsonl",
        "scorer_cal": REPO_ROOT / "data/inner_manifests/scand/scorer-cal.windows.jsonl",
        "cem_cal": REPO_ROOT / "data/inner_manifests/scand/cem-cal.windows.jsonl",
    }
    split_results = {
        name: split_audit(holdout, path) for name, path in split_paths.items()
    }
    split_overlap = any(
        result["record_overlap"]
        or result["trajectory_overlap"]
        or result["route_group_overlap"]
        or result["bag_id_overlap"]
        for result in split_results.values()
    )

    artifacts = []
    failures = []
    scanned = 0
    seen: set[Path] = set()
    for root in args.scan_root:
        for path in candidate_files(root):
            resolved = path.resolve()
            if resolved in seen or inside(resolved, args.current_output_root):
                continue
            seen.add(resolved)
            scanned += 1
            try:
                records, trajectories = npz_identity(path)
            except Exception as error:  # audit failure is retained, not hidden
                failures.append({"path": str(path), "error": repr(error)})
                continue
            record_overlap = sorted(holdout["record_key"] & records)
            trajectory_overlap = sorted(holdout["trajectory"] & trajectories)
            if record_overlap or trajectory_overlap:
                artifacts.append(
                    {
                        "path": str(path),
                        "sha256": sha256(path),
                        "record_overlap": record_overlap,
                        "trajectory_overlap": trajectory_overlap,
                    }
                )
    audit = {
        "schema": "scand_internal_holdout_historical_usage_audit_v1",
        "status": "unspent" if not split_overlap and not artifacts else "spent",
        "holdout_manifest": str(args.holdout_manifest),
        "holdout_manifest_sha256": sha256(args.holdout_manifest),
        "windows": len(rows),
        "trajectories": len(holdout["trajectory"]),
        "route_groups": len(holdout["route_group"]),
        "split_audit": split_results,
        "split_overlap": split_overlap,
        "prior_npz_files_scanned": scanned,
        "prior_artifact_overlaps": artifacts,
        "scan_failures": failures,
        "historically_spent": bool(split_overlap or artifacts),
        "current_domain_adaptation_outputs_excluded_from_prior_scan": str(
            args.current_output_root
        ),
        "future_pose_outcomes_opened": False,
        "fixed_pool_only": True,
        "cem": False,
    }
    write_json(args.output, audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True))
    if audit["historically_spent"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
