#!/usr/bin/env python3
"""Fail closed when an evaluation manifest overlaps frozen v70 data identities."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY = (
    ROOT / "data_splits/v70_training_identity/v70_training_identities.jsonl"
)
DEFAULT_SPLITS = ROOT / "data_splits/v70_training_identity/strict_splits"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_candidates(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        return load_jsonl(path)
    if path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return payload
        if isinstance(payload.get("records"), list):
            return payload["records"]
        raise RuntimeError("JSON manifest has no records array")
    if path.suffix == ".pkl":
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        return [
            {
                "trajectory_id": row[0],
                "current_t": int(row[1]),
                "route_group": None,
            }
            for row in payload
        ]
    raise RuntimeError(f"unsupported candidate manifest: {path}")


def trajectory_id(row: dict[str, Any]) -> str:
    value = row.get("trajectory_id") or row.get("traj_name")
    if value is None:
        raise RuntimeError(f"candidate row has no trajectory identity: {row}")
    return str(value)


def current_t(row: dict[str, Any]) -> int:
    value = row.get("current_t", row.get("curr_time"))
    if value is None:
        raise RuntimeError(f"candidate row has no current time: {row}")
    return int(value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("recon", "scand"), required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--strict-splits-dir", type=Path, default=DEFAULT_SPLITS)
    args = parser.parse_args()

    training = [
        row
        for row in load_jsonl(args.registry)
        if row["dataset"] == args.dataset
    ]
    candidates = load_candidates(args.candidate)

    train_records = {row["record_key"] for row in training}
    train_trajectories = {row["trajectory_id"] for row in training}
    train_routes = {
        row["route_group"] for row in training if row["route_group"] is not None
    }
    candidate_records = {
        f"{trajectory_id(row)}:{current_t(row)}" for row in candidates
    }
    candidate_trajectories = {trajectory_id(row) for row in candidates}
    candidate_routes = {
        str(row.get("route_group") or row.get("route_id"))
        for row in candidates
        if row.get("route_group") is not None or row.get("route_id") is not None
    }

    split_path = args.strict_splits_dir / f"{args.dataset}_strict_splits.json"
    split = json.loads(split_path.read_text(encoding="utf-8"))
    split_membership = {
        name: len(candidate_trajectories & set(split[name]))
        for name in ("train", "val", "test")
    }
    listed = set(split["train"]) | set(split["val"]) | set(split["test"])
    split_membership["unlisted"] = len(candidate_trajectories - listed)

    exact_overlap = sorted(candidate_records & train_records)
    trajectory_overlap = sorted(candidate_trajectories & train_trajectories)
    route_overlap = sorted(candidate_routes & train_routes)
    passed = (
        not exact_overlap
        and not trajectory_overlap
        and not route_overlap
        and split_membership["train"] == 0
        and split_membership["val"] == 0
        and split_membership["unlisted"] == 0
        and split_membership["test"] == len(candidate_trajectories)
    )
    result = {
        "schema": "v70_training_overlap_check_v1",
        "status": "pass" if passed else "fail_closed",
        "dataset": args.dataset,
        "candidate_states": len(candidates),
        "candidate_trajectories": len(candidate_trajectories),
        "exact_record_overlap": len(exact_overlap),
        "trajectory_overlap": len(trajectory_overlap),
        "route_overlap": len(route_overlap),
        "split_membership": split_membership,
        "examples": {
            "exact_record_overlap": exact_overlap[:10],
            "trajectory_overlap": trajectory_overlap[:10],
            "route_overlap": route_overlap[:10],
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
