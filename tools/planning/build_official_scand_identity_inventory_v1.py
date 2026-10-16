#!/usr/bin/env python3
"""Build an identity-only quarantine inventory from the public SCAND index.

This tool never downloads bags and never opens images, labels, outcomes, or a
reserve.  A row that is not known-used is still *not* declared independent
unless a complete custodian-produced hash-only forbidden registry is supplied.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from planning.proposal_tail_sidecar_v1 import (  # noqa: E402
    canonical_json,
    identity_hash,
    sha256_file,
)
from scoring.strict_inner_partitions import parse_scand_trajectory_id  # noqa: E402
from tools.planning.check_proposal_tail_readiness_v1 import (  # noqa: E402
    load_registry,
)


SCHEMA = "official_scand_identity_inventory_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-index", type=Path, required=True)
    parser.add_argument("--train-trajectories", type=Path, required=True)
    parser.add_argument("--used-registry", type=Path, required=True)
    parser.add_argument("--forbidden-registry", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def canonical_public_bag(file_name: str) -> str:
    name = str(file_name).strip()
    require(bool(name), "public index row has empty FileName")
    require("/" not in name and "\\" not in name, f"invalid FileName: {name}")
    return f"random_mdps_{name}"


def load_train_bags(path: Path) -> tuple[set[str], int]:
    trajectories = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    # One historical entry uses Bags_ for the canonical random_mdps_ prefix.
    canonical = [
        "random_mdps_" + value[len("Bags_") :] if value.startswith("Bags_") else value
        for value in trajectories
    ]
    bags = {parse_scand_trajectory_id(value)["bag_id"] for value in canonical}
    return bags, len(trajectories)


def main() -> None:
    args = parse_args()
    require(args.official_index.is_file(), f"missing official index: {args.official_index}")
    train_bags, train_trajectories = load_train_bags(args.train_trajectories)
    used, used_raw = load_registry(args.used_registry, forbidden=False)

    forbidden = None
    forbidden_raw: dict[str, Any] | None = None
    if args.forbidden_registry is not None:
        forbidden, forbidden_raw = load_registry(args.forbidden_registry, forbidden=True)

    with args.official_index.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    require(rows, "official index is empty")

    inventory = []
    seen_bags: set[str] = set()
    for row_number, row in enumerate(rows, start=2):
        bag = canonical_public_bag(row.get("FileName", ""))
        require(bag not in seen_bags, f"duplicate public bag identity: {bag}")
        seen_bags.add(bag)
        # CSV location columns contain display annotations on some rows. The
        # split lock derives route identity from FileName tokens, so append a
        # synthetic segment solely to reuse the canonical trajectory parser.
        route = parse_scand_trajectory_id(bag + "_0")["route_group"]
        bag_hash = identity_hash("bag", bag)
        route_hash = identity_hash("route", route)
        old_train = bag in train_bags
        used_bag = bag_hash in used["bag"]
        used_route = route_hash in used["route"]
        forbidden_bag = forbidden is not None and bag_hash in forbidden["bag"]
        forbidden_route = forbidden is not None and route_hash in forbidden["route"]

        reasons = []
        if old_train:
            reasons.append("old_train_bag")
        if used_bag:
            reasons.append("used_v70_v77_bag_hash")
        if used_route:
            reasons.append("used_v70_v77_route_hash")
        if forbidden_bag:
            reasons.append("custodian_forbidden_bag_hash")
        if forbidden_route:
            reasons.append("custodian_forbidden_route_hash")
        if forbidden is None:
            reasons.append("complete_custodian_forbidden_registry_missing")

        identity_clear = not any(
            (old_train, used_bag, used_route, bool(forbidden_bag), bool(forbidden_route))
        )
        admissible = identity_clear and forbidden is not None
        inventory.append(
            {
                "row_number": row_number,
                "public_file_name": row.get("FileName"),
                "canonical_bag_id": bag,
                "route_group": route,
                "public_split": row.get("Train/Val"),
                "collection_date": row.get("Date"),
                "bag_hash": bag_hash,
                "route_hash": route_hash,
                "old_train_bag": old_train,
                "used_bag_hash_overlap": used_bag,
                "used_route_hash_overlap": used_route,
                "identity_layer_clear": identity_clear,
                "admissible_after_all_identity_checks": admissible,
                "quarantine_reasons": reasons,
            }
        )

    clear = [row for row in inventory if row["identity_layer_clear"]]
    admissible = [row for row in inventory if row["admissible_after_all_identity_checks"]]
    split_counts: dict[str, int] = {}
    for row in inventory:
        key = str(row["public_split"])
        split_counts[key] = split_counts.get(key, 0) + 1

    payload = {
        "schema": SCHEMA,
        "status": "identity_ready" if forbidden is not None else "quarantine_fail_closed_missing_forbidden_registry",
        "scope": "public metadata and hash comparison only; no bag download, labels, outcomes, or reserve content",
        "inputs": {
            "official_index": {"path": str(args.official_index.resolve()), "sha256": sha256_file(args.official_index)},
            "train_trajectories": {
                "path": str(args.train_trajectories.resolve()),
                "sha256": sha256_file(args.train_trajectories),
                "trajectory_count": train_trajectories,
                "bag_count": len(train_bags),
            },
            "used_registry": {
                "path": str(args.used_registry.resolve()),
                "sha256": sha256_file(args.used_registry),
                "status": used_raw.get("status"),
            },
            "forbidden_registry": None
            if args.forbidden_registry is None
            else {
                "path": str(args.forbidden_registry.resolve()),
                "sha256": sha256_file(args.forbidden_registry),
                "status": forbidden_raw.get("status") if forbidden_raw else None,
            },
        },
        "counts": {
            "public_rows": len(inventory),
            "public_unique_bags": len(seen_bags),
            "public_split": dict(sorted(split_counts.items())),
            "old_train_bags": sum(row["old_train_bag"] for row in inventory),
            "used_bag_hash_overlap": sum(row["used_bag_hash_overlap"] for row in inventory),
            "used_route_hash_overlap": sum(row["used_route_hash_overlap"] for row in inventory),
            "identity_layer_clear_but_forbidden_unresolved": len(clear) if forbidden is None else 0,
            "admissible_after_all_identity_checks": len(admissible),
        },
        "download_started": False,
        "labels_opened": False,
        "reserve_content_accessed": False,
        "rows": inventory,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(canonical_json(payload) + "\n", encoding="utf-8")
    print(canonical_json({"output": str(args.output), "sha256": sha256_file(args.output), **payload["counts"]}))


if __name__ == "__main__":
    main()
