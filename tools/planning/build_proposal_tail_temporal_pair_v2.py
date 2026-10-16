#!/usr/bin/env python3
"""Compatibility entrypoint for temporal-pair features from live sidecars.

Live pool files do not repeat ``record_key``.  This wrapper injects the
candidate-table key only when the pool state count matches; the v1 builder
then still verifies gathered candidate values and every candidate SHA in
exact table order before writing features.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from planning.proposal_tail_sidecar_v1 import canonical_json, sha256_file  # noqa: E402
from tools.planning import build_proposal_tail_temporal_pair_v1 as base  # noqa: E402


_original_load_npz = base.load_npz
_record_key = None
_injected: list[str] = []


def compatible_load_npz(path: Path):
    global _record_key
    value = _original_load_npz(path)
    if path.name == "candidate_table.npz":
        _record_key = value["record_key"].copy()
        return value
    if path.name == "candidates_and_scores.npz" and "record_key" not in value:
        if _record_key is None:
            raise RuntimeError("candidate table must be loaded before pool identity fallback")
        candidate_key = "candidates_feature_waypoint_units" if "candidates_feature_waypoint_units" in value else "candidates"
        if value[candidate_key].shape[0] != len(_record_key):
            raise RuntimeError(f"pool state count mismatch: {path}")
        value = {**value, "record_key": _record_key.copy()}
        _injected.append(str(path.resolve()))
    return value


def argument_path(name: str) -> Path:
    try:
        return Path(sys.argv[sys.argv.index(name) + 1])
    except (ValueError, IndexError) as error:
        raise RuntimeError(f"missing required argument {name}") from error


def main() -> None:
    base.load_npz = compatible_load_npz
    base.main()
    audit_path = argument_path("--audit")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["pool_record_key_observable"] = False
    audit["pool_record_key_compatibility"] = {
        "schema": "proposal_tail_pool_identity_compatibility_v2",
        "injected_from_candidate_table": _injected,
        "guard": "matching state count plus exact candidate values plus per-candidate SHA",
        "candidate_order_and_hash_verified_by_v1": bool(audit["candidate_order_and_hash_verified"]),
    }
    audit_path.write_text(canonical_json(audit) + "\n", encoding="utf-8")
    print(canonical_json({"status": "pass", "audit": str(audit_path), "audit_sha256": sha256_file(audit_path), "pool_record_key_injections": len(_injected)}))


if __name__ == "__main__":
    main()
