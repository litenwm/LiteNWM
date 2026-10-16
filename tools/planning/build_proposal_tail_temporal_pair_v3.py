#!/usr/bin/env python3
"""Cache-only fallback for legacy sidecars that omitted raw feature copies.

Legacy sidecar pool files retained decisions, scores, actions and seed outputs
but did not duplicate ``raw_ltwm_features`` or ``scorer_v2_aux_features``.
Those exact arrays remain in the frozen base/round feature caches.  This
wrapper joins them by state and exact float32 candidate value, then delegates
all candidate-hash/order and formal-reference checks to the v1 builder.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from planning.proposal_tail_sidecar_v1 import canonical_json, sha256_file  # noqa: E402
from tools.planning import build_proposal_tail_temporal_pair_v1 as base  # noqa: E402


_original_load_npz = base.load_npz
_record_key: np.ndarray | None = None
_sidecar: Path | None = None
_injected_identity: list[str] = []
_joined_features: list[dict[str, str]] = []


def argument_path(name: str) -> Path:
    try:
        return Path(sys.argv[sys.argv.index(name) + 1]).resolve()
    except (ValueError, IndexError) as error:
        raise RuntimeError(f"missing required argument {name}") from error


def legacy_feature_path(pool_path: Path) -> Path:
    assert _sidecar is not None
    cohort_root = _sidecar.parent
    relative = pool_path.relative_to(_sidecar)
    if relative.parts[0] == "fixed_pool":
        matches = sorted((cohort_root / "base_features").glob("*_features.npz"))
        if len(matches) != 1:
            raise RuntimeError(f"expected one frozen base feature file, got {matches}")
        return matches[0]
    if relative.parts[0].startswith("iteration_"):
        return cohort_root / "formal_logger_on" / relative.parts[0] / "cem_features.npz"
    raise RuntimeError(f"unrecognized legacy pool path: {pool_path}")


def compatible_load_npz(path: Path):
    global _record_key
    value = _original_load_npz(path)
    if path.name == "candidate_table.npz":
        _record_key = value["record_key"].copy()
        return value
    if path.name != "candidates_and_scores.npz":
        return value
    if _record_key is None:
        raise RuntimeError("candidate table must be loaded before pool fallback")
    candidate_key = "candidates_feature_waypoint_units" if "candidates_feature_waypoint_units" in value else "candidates"
    if value[candidate_key].shape[0] != len(_record_key):
        raise RuntimeError(f"pool state count mismatch: {path}")
    if "record_key" not in value:
        value = {**value, "record_key": _record_key.copy()}
        _injected_identity.append(str(path.resolve()))
    missing = [name for name in ("raw_ltwm_features", "scorer_v2_aux_features") if name not in value]
    if missing:
        source_path = legacy_feature_path(path)
        source = _original_load_npz(source_path)
        if not np.array_equal(source["record_key"].astype(str), _record_key.astype(str)):
            raise RuntimeError(f"record mismatch in legacy feature cache: {source_path}")
        if not np.array_equal(source["candidates"].astype(np.float32), value[candidate_key].astype(np.float32)):
            raise RuntimeError(f"candidate mismatch in legacy feature cache: {source_path}")
        value = {**value, **{name: source[name].copy() for name in missing}}
        _joined_features.append({"pool": str(path.resolve()), "feature_cache": str(source_path.resolve()), "feature_cache_sha256": sha256_file(source_path)})
    return value


def main() -> None:
    global _sidecar
    _sidecar = argument_path("--sidecar-dir")
    base.load_npz = compatible_load_npz
    base.main()
    audit_path = argument_path("--audit")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["legacy_feature_cache_join"] = {
        "schema": "proposal_tail_legacy_feature_cache_join_v3",
        "dwm_ltwm_rerun": False,
        "scorer_rerun": False,
        "record_key_injected_from_candidate_table": _injected_identity,
        "joined_feature_caches": _joined_features,
        "join_guard": "record_key equality, exact float32 candidate equality, then v1 per-candidate SHA/order verification",
    }
    audit_path.write_text(canonical_json(audit) + "\n", encoding="utf-8")
    print(canonical_json({
        "status": "pass",
        "audit": str(audit_path),
        "audit_sha256": sha256_file(audit_path),
        "feature_cache_joins": len(_joined_features),
        "dwm_ltwm_rerun": False,
    }))


if __name__ == "__main__":
    main()
