#!/usr/bin/env python3
"""Append cached static visual/state context to the spent pairwise table."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from planning.proposal_tail_sidecar_v1 import canonical_json, sha256_file  # noqa: E402
from tools.evaluation import run_v72_proposal_tail_reranker as v72  # noqa: E402


SOURCE = ROOT / "outputs/cem_development_v1/incremental_outcome_student_v1/spent_runtime_pair_table.npz"
OUTPUT_DIR = ROOT / "outputs/cem_development_v1/incremental_outcome_student_v1/static_context_student_v1"
OUTPUT = OUTPUT_DIR / "spent_runtime_pair_static_context_table.npz"
AUDIT = OUTPUT_DIR / "spent_runtime_pair_static_context_table.audit.json"
LEGACY = ROOT / "outputs/cem_development_v1/scand_existing_train_upperbound_pilot_v1_corrected"
NEW = ROOT / "outputs/cem_development_v1/scand_new_routefit10_v4_development_pilot_v2_strict_identity"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    require(path.is_file(), f"missing NPZ: {path}")
    with np.load(path, allow_pickle=False) as payload:
        return {key: payload[key].copy() for key in payload.files}


def feature_pool_paths(root: Path) -> dict[int, Path]:
    fixed_sidecar = root / "proposal_tail_sidecar/fixed_pool/candidates_and_scores.npz"
    fixed_payload = load_npz(fixed_sidecar)
    if "static_features" in fixed_payload:
        fixed = fixed_sidecar
    else:
        matches = sorted((root / "base_features").glob("*_features.npz"))
        require(len(matches) == 1, f"expected one base feature file: {root}")
        fixed = matches[0]
    result = {-1: fixed}
    for iteration in range(3):
        sidecar = root / f"proposal_tail_sidecar/iteration_{iteration:02d}/candidates_and_scores.npz"
        payload = load_npz(sidecar)
        result[iteration] = sidecar if "static_features" in payload else root / f"formal_logger_on/iteration_{iteration:02d}/cem_features.npz"
    return result


def sidecar_static(root: Path) -> tuple[np.ndarray, dict[str, str]]:
    table = load_npz(root / "proposal_tail_sidecar/candidate_table.npz")
    paths = feature_pool_paths(root)
    pools = {source: load_npz(path) for source, path in paths.items()}
    rounds = table["source_round"].astype(np.int64)
    local = table["source_candidate_index"].astype(np.int64)
    states, candidates = rounds.shape
    output = np.empty((states, candidates, 210), dtype=np.float32)
    reconstructed = np.empty_like(table["candidates_feature_waypoint_units"], dtype=np.float32)
    for state in range(states):
        for candidate in range(candidates):
            payload = pools[int(rounds[state, candidate])]
            index = int(local[state, candidate])
            candidate_key = "candidates_feature_waypoint_units" if "candidates_feature_waypoint_units" in payload else "candidates"
            reconstructed[state, candidate] = payload[candidate_key][state, index]
            output[state, candidate] = payload["static_features"][state, index]
    require(np.array_equal(reconstructed, table["candidates_feature_waypoint_units"].astype(np.float32)), f"candidate mapping mismatch: {root}")
    require(np.isfinite(output).all(), f"nonfinite static feature: {root}")
    return output, {str(source): sha256_file(path) for source, path in paths.items()}


def fresh_static() -> tuple[np.ndarray, dict[str, str]]:
    spec = next(item for item in v72.COHORTS if item["name"] == "scand_fresh_outer")
    cohort = v72.load_cohort(spec)
    replay = v72.V70_ROOT / "replays" / spec["name"] / "proposal_aware_v70"
    freeze = v72.read_json(replay / "prelabel_freeze.json")
    paths = [v72.resolve_path(freeze["base_features"])] + [replay / f"iteration_{iteration:02d}/cem_features.npz" for iteration in range(3)]
    pools = [load_npz(path) for path in paths]
    states, candidates = cohort["candidates"].shape[:2]
    output = np.empty((states, candidates, 210), dtype=np.float32)
    reconstructed = np.empty_like(cohort["candidates"], dtype=np.float32)
    for state in range(states):
        seen = set()
        cursor = 0
        for payload in pools:
            for index in range(payload["candidates"].shape[1]):
                key = np.ascontiguousarray(payload["candidates"][state, index], dtype=np.float32).tobytes()
                if key in seen:
                    continue
                seen.add(key)
                reconstructed[state, cursor] = payload["candidates"][state, index]
                output[state, cursor] = payload["static_features"][state, index]
                cursor += 1
        require(cursor == candidates, f"fresh union cardinality mismatch state={state}")
    require(np.array_equal(reconstructed, cohort["candidates"].astype(np.float32)), "fresh static candidate mapping mismatch")
    require(np.isfinite(output).all(), "fresh static feature contains NaN/Inf")
    return output, {str(index - 1): sha256_file(path) for index, path in enumerate(paths)}


def main() -> None:
    require(not OUTPUT.exists() and not AUDIT.exists(), "refusing to overwrite static-context table")
    source = load_npz(SOURCE)
    legacy, legacy_hash = sidecar_static(LEGACY)
    fresh, fresh_hash = fresh_static()
    new, new_hash = sidecar_static(NEW)
    static = np.concatenate((legacy, fresh, new), axis=0)
    require(static.shape == (800, 65, 210), f"static shape mismatch: {static.shape}")
    formal = source["formal_v70_index"].astype(np.int64)
    selected = static[np.arange(len(formal)), formal]
    static_pair = np.concatenate((static - selected[:, None], np.broadcast_to(selected[:, None], static.shape)), axis=-1).astype(np.float32)
    feature = np.concatenate((source["runtime_pair_feature"].astype(np.float32), static_pair), axis=-1)
    require(feature.shape == (800, 65, 771), f"extended feature shape mismatch: {feature.shape}")
    require(np.isfinite(feature).all(), "extended feature contains NaN/Inf")
    names = np.concatenate((
        source["feature_names"].astype(str),
        np.asarray([f"delta_static_context_{i}" for i in range(210)]),
        np.asarray([f"formal_static_context_{i}" for i in range(210)]),
    ))
    arrays = {key: value for key, value in source.items() if key not in {"runtime_pair_feature", "feature_names"}}
    arrays["runtime_pair_feature"] = feature
    arrays["feature_names"] = names
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("wb") as handle:
        np.savez_compressed(handle, **arrays)

    candidate_std = static.std(axis=1)
    invariant = np.median(candidate_std, axis=0) < 1.0e-8
    route = source["route_group"].astype(str)
    formal_static = selected.astype(np.float64)
    total_var = formal_static.var(axis=0)
    within = np.zeros(210, dtype=np.float64)
    for name in np.unique(route):
        local = route == name
        within += local.mean() * formal_static[local].var(axis=0)
    eta = np.where(total_var > 1.0e-12, np.clip(1.0 - within / total_var, 0.0, 1.0), 0.0)
    audit = {
        "schema": "scand_static_context_pair_table_audit_v1",
        "status": "complete_spent_development_materially_new_representation",
        "states": 800,
        "routes": len(np.unique(route)),
        "candidates_per_state": 65,
        "base_feature_dim": 351,
        "static_context_dim": 210,
        "static_pair_added_dim": 420,
        "output_feature_dim": 771,
        "static_candidate_invariant_dimensions": int(invariant.sum()),
        "static_candidate_varying_dimensions": int((~invariant).sum()),
        "formal_static_route_eta_squared": {"median": float(np.median(eta)), "p90": float(np.quantile(eta, 0.9)), "max": float(eta.max())},
        "interpretation": "Static context is runtime-safe but may encode route/domain appearance; route-balanced strict OOF remains mandatory.",
        "candidate_mapping": "exact float32 candidate union/order verified for all fixed+three-round caches",
        "source_hashes": {"legacy": legacy_hash, "fresh": fresh_hash, "new": new_hash, "base_table": sha256_file(SOURCE)},
        "output": {"path": str(OUTPUT), "sha256": sha256_file(OUTPUT)},
        "dwm_ltwm_scorer_rerun": False,
        "targets_or_thresholds_changed": False,
        "sealed_validation_included": False,
        "reserve_accessed": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    AUDIT.write_text(canonical_json(audit) + "\n", encoding="utf-8")
    print(canonical_json({"status": audit["status"], "output": str(OUTPUT), "sha256": audit["output"]["sha256"], "feature_dim": 771, "invariant_static_dims": audit["static_candidate_invariant_dimensions"], "route_eta2": audit["formal_static_route_eta_squared"]}))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(canonical_json({"status": "fail_closed", "error_type": type(error).__name__, "error": str(error), "sealed_validation_opened": False}), file=sys.stderr)
        raise SystemExit(2) from error
