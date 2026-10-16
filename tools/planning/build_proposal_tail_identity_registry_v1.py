#!/usr/bin/env python3
"""Build a hash-only deny registry from already-used non-reserve manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from planning.proposal_tail_sidecar_v1 import (  # noqa: E402
    canonical_json,
    identity_hash,
    reject_forbidden_path,
    sha256_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    values = {key: set() for key in ("route", "day", "bag", "trajectory", "window")}
    sources = []
    for path in args.manifest:
        reject_forbidden_path(path)
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not rows:
            raise RuntimeError(f"empty identity manifest: {path}")
        for row in rows:
            mapping = {
                "route": row.get("route_group", row.get("route_id")),
                "day": row.get("day", row.get("collection_day")),
                "bag": row.get("bag_id"),
                "trajectory": row.get("trajectory_id"),
                "window": row.get("window_uid"),
            }
            for kind, value in mapping.items():
                if value is not None:
                    values[kind].add(identity_hash(kind, value))
        sources.append({"path": str(path), "sha256": sha256_file(path), "rows": len(rows)})
    payload = {
        "schema": "proposal_tail_forbidden_identity_registry_v1",
        "status": "complete_used_v70_through_v77_nonreserve_identity_registry",
        "complete": True,
        "identity_values_are_hash_only": True,
        "sources": sources,
        **{f"{key}_hashes": sorted(value) for key, value in values.items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(canonical_json(payload) + "\n", encoding="utf-8")
    print(canonical_json({"output": str(args.output), "sha256": sha256_file(args.output)}))


if __name__ == "__main__":
    main()
