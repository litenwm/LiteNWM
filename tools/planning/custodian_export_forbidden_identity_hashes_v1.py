#!/usr/bin/env python3
"""Custodian-side exporter for a complete hash-only forbidden registry.

Run this tool outside the research/selection workflow, on the machine or
account that is allowed to inspect the forbidden cohort identity manifest.
Only canonical identity hashes are written.  Input paths, source hashes,
labels, outcomes, and record contents never appear in the output registry.

Supported inputs are JSONL rows, a JSON list, or a JSON object containing a
``records``/``rows``/``windows`` list.  Every row must expose route, bag,
trajectory, and window identity.  Collection day is included when available.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from planning.proposal_tail_sidecar_v1 import (  # noqa: E402
    canonical_json,
    identity_hash,
    sha256_file,
)
from tools.planning.check_proposal_tail_readiness_v1 import load_registry  # noqa: E402


SCHEMA = "proposal_tail_forbidden_identity_registry_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity-manifest", type=Path, action="append", required=True)
    parser.add_argument("--custodian-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def rows_from_value(value: Any, path: Path) -> list[Mapping[str, Any]]:
    if isinstance(value, list):
        rows = value
    elif isinstance(value, Mapping):
        rows = None
        for key in ("records", "rows", "windows"):
            if isinstance(value.get(key), list):
                rows = value[key]
                break
        if rows is None and all(
            key in value
            for key in ("bag_id", "trajectory_id", "window_uid")
        ):
            rows = [value]
        require(rows is not None, f"JSON object has no records/rows/windows list: {path}")
    else:
        raise RuntimeError(f"unsupported JSON root in {path}")
    require(rows and all(isinstance(row, Mapping) for row in rows), f"empty or invalid identity rows: {path}")
    return list(rows)


def read_rows(path: Path) -> list[Mapping[str, Any]]:
    require(path.is_file(), f"missing identity manifest: {path}")
    text = path.read_text(encoding="utf-8")
    try:
        return rows_from_value(json.loads(text), path)
    except json.JSONDecodeError:
        values = [json.loads(line) for line in text.splitlines() if line.strip()]
        require(values and all(isinstance(row, Mapping) for row in values), f"invalid JSONL identity manifest: {path}")
        return values


def identity_value(row: Mapping[str, Any], names: Sequence[str], field: str, *, required: bool) -> str | None:
    for name in names:
        value = row.get(name)
        if isinstance(value, str) and value.strip():
            return value.strip()
    require(not required, f"identity row lacks {field}: accepted keys={list(names)}")
    return None


def main() -> None:
    args = parse_args()
    require(str(args.custodian_id).strip(), "custodian-id must be nonempty")
    require(not args.output.exists(), f"refusing to overwrite registry: {args.output}")
    values = {kind: set() for kind in ("route", "day", "bag", "trajectory", "window")}
    rows_seen = 0
    for path in args.identity_manifest:
        for row in read_rows(path):
            rows_seen += 1
            mapping = {
                "route": identity_value(row, ("route_group", "route_id"), "route", required=True),
                "day": identity_value(row, ("collection_day", "day"), "day", required=False),
                "bag": identity_value(row, ("bag_id",), "bag", required=True),
                "trajectory": identity_value(
                    row,
                    ("trajectory_id", "trajectory", "traj_name"),
                    "trajectory",
                    required=True,
                ),
                "window": identity_value(row, ("window_uid",), "window", required=True),
            }
            for kind, value in mapping.items():
                if value is not None:
                    values[kind].add(identity_hash(kind, value))
    require(rows_seen > 0, "no identity rows found")
    for kind in ("route", "bag", "trajectory", "window"):
        require(values[kind], f"complete registry cannot have empty {kind}_hashes")
    payload = {
        "schema": SCHEMA,
        "status": "complete_custodian_hash_only_forbidden_registry",
        "complete": True,
        "identity_values_are_hash_only": True,
        "route_hashes": sorted(values["route"]),
        "day_hashes": sorted(values["day"]),
        "bag_hashes": sorted(values["bag"]),
        "trajectory_hashes": sorted(values["trajectory"]),
        "window_hashes": sorted(values["window"]),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "custodian_id": str(args.custodian_id).strip(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    temporary.write_text(canonical_json(payload) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    # Reuse the consumer's strict parser before reporting success.
    load_registry(args.output, forbidden=True)
    print(
        canonical_json(
            {
                "status": payload["status"],
                "output": str(args.output),
                "output_sha256": sha256_file(args.output),
                "rows_hashed": rows_seen,
                "hash_counts": {kind: len(items) for kind, items in values.items()},
                "source_paths_written_to_registry": False,
                "labels_or_outcomes_written_to_registry": False,
            }
        )
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(
            canonical_json(
                {
                    "status": "fail_closed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "registry_written": False,
                }
            ),
            file=sys.stderr,
        )
        raise SystemExit(2) from error
