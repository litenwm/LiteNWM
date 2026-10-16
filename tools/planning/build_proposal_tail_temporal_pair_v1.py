#!/usr/bin/env python3
"""Build label-free candidate-vs-formal-v70 temporal pair features.

The input is a completed proposal-tail sidecar.  The tool is purely offline:
it never calls the planner, scorer, DWM/LTWM, label builder, or reserve.  It
verifies the sidecar hashes and candidate ordering before producing features.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from planning.proposal_tail_sidecar_v1 import (  # noqa: E402
    action_structure_features,
    array_sha256,
    canonical_json,
    candidate_hashes,
    reject_forbidden_path,
    sha256_file,
)


SCHEMA = "scand_proposal_tail_temporal_pair_v1"
SUMMARY_NAMES = ("mean", "std", "min", "max")
LATENT_BLOCK_NAMES = (
    "goal_nmse",
    "goal_cosine",
    "current_nmse",
    "goal_progress",
    "step_energy",
    "step_goal_direction_cosine",
    "goal_norm_ratio",
)
HORIZON_NAMES = ("h1", "h2", "h3", "h4")
POOL_PATH = {
    -1: "fixed_pool/candidates_and_scores.npz",
    0: "iteration_00/candidates_and_scores.npz",
    1: "iteration_01/candidates_and_scores.npz",
    2: "iteration_02/candidates_and_scores.npz",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sidecar-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    return parser.parse_args()


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON object required: {path}")
    return value


def load_npz(path: Path) -> dict[str, np.ndarray]:
    require(path.is_file(), f"missing NPZ: {path}")
    with np.load(path, allow_pickle=False) as payload:
        return {key: np.asarray(payload[key]) for key in payload.files}


def choose(payload: Mapping[str, np.ndarray], *names: str) -> np.ndarray | None:
    for name in names:
        if name in payload:
            return np.asarray(payload[name])
    return None


def verify_manifest_files(sidecar: Path, manifest: Mapping[str, Any]) -> dict[str, str]:
    verified = {}
    for relative, expected in manifest["files"].items():
        path = sidecar / relative
        require(path.is_file(), f"manifest-listed file missing: {relative}")
        observed = sha256_file(path)
        require(observed == expected, f"manifest hash mismatch: {relative}")
        verified[relative] = observed
    return verified


def gather_union(
    table: Mapping[str, np.ndarray],
    pools: Mapping[int, Mapping[str, np.ndarray]],
    aliases: tuple[str, ...],
    *,
    optional: bool = False,
) -> tuple[np.ndarray | None, np.ndarray]:
    rounds = table["source_round"].astype(np.int64)
    local = table["source_candidate_index"].astype(np.int64)
    states, candidates = rounds.shape
    available = np.zeros((states, candidates), dtype=bool)
    output: np.ndarray | None = None
    for state in range(states):
        for candidate in range(candidates):
            pool = pools[int(rounds[state, candidate])]
            value = choose(pool, *aliases)
            if value is None:
                continue
            index = int(local[state, candidate])
            if output is None:
                output = np.empty((states, candidates, *value.shape[2:]), dtype=value.dtype)
            output[state, candidate] = value[state, index]
            available[state, candidate] = True
    if output is None:
        if optional:
            return None, available
        raise RuntimeError(f"required union field absent: {aliases}")
    if not available.all():
        if not optional:
            missing = int((~available).sum())
            raise RuntimeError(f"required union field incomplete {aliases}: {missing} rows")
        if np.issubdtype(output.dtype, np.floating):
            output[~available] = np.nan
        elif np.issubdtype(output.dtype, np.bool_):
            output[~available] = False
        else:
            output[~available] = 0
    return output, available


def gather_seed_union(
    table: Mapping[str, np.ndarray],
    pools: Mapping[int, Mapping[str, np.ndarray]],
    aliases: tuple[str, ...],
) -> np.ndarray:
    rounds = table["source_round"].astype(np.int64)
    local = table["source_candidate_index"].astype(np.int64)
    states, candidates = rounds.shape
    output = None
    for state in range(states):
        for candidate in range(candidates):
            value = choose(pools[int(rounds[state, candidate])], *aliases)
            require(value is not None, f"required seed field absent: {aliases}")
            require(value.ndim == 3 and value.shape[1] == states, f"bad seed field shape: {aliases} {value.shape}")
            index = int(local[state, candidate])
            if output is None:
                output = np.empty((states, candidates, value.shape[0]), dtype=value.dtype)
            output[state, candidate] = value[:, state, index]
    assert output is not None
    return output


def formal_union_index(table: Mapping[str, np.ndarray], formal: Mapping[str, np.ndarray]) -> np.ndarray:
    require("archive_source_round" in formal, "formal snapshot lacks archive_source_round")
    require("archive_source_candidate" in formal, "formal snapshot lacks archive_source_candidate")
    rounds = table["source_round"].astype(np.int64)
    local = table["source_candidate_index"].astype(np.int64)
    source_round = formal["archive_source_round"].astype(np.int64)
    source_candidate = formal["archive_source_candidate"].astype(np.int64)
    require(source_round.shape == (rounds.shape[0],), "formal archive round shape mismatch")
    result = np.empty(rounds.shape[0], dtype=np.int64)
    for state in range(rounds.shape[0]):
        target_round = -1 if source_round[state] == -2 else int(source_round[state])
        target_candidate = 0 if source_round[state] == -2 else int(source_candidate[state])
        matches = np.flatnonzero((rounds[state] == target_round) & (local[state] == target_candidate))
        require(len(matches) == 1, f"formal candidate mapping is not unique for state {state}")
        result[state] = int(matches[0])
    return result


def reference(value: np.ndarray, index: np.ndarray) -> np.ndarray:
    return value[np.arange(len(value)), index]


def expand_reference(value: np.ndarray, index: np.ndarray) -> np.ndarray:
    selected = reference(value, index)
    return np.expand_dims(selected, axis=1)


def main() -> None:
    args = parse_args()
    sidecar = args.sidecar_dir.resolve()
    reject_forbidden_path(sidecar)
    reject_forbidden_path(args.output)
    reject_forbidden_path(args.audit)
    manifest_path = sidecar / "collection_manifest.json"
    manifest = read_json(manifest_path)
    require(manifest.get("schema") == "scand_proposal_tail_sidecar_v1", "wrong sidecar schema")
    require(manifest.get("status") in {"complete", "cached_serialization_dry_run"}, "incomplete sidecar")
    require(manifest.get("formal_behavior_influence") is False, "sidecar influenced formal behavior")
    require(manifest.get("labels_opened") is False, "labels were opened in sidecar contract")
    require(manifest.get("reserve_accessed") is False, "sidecar reports reserve access")
    require(int(manifest.get("candidate_count")) == 65, "candidate count is not frozen at 65")
    verified = verify_manifest_files(sidecar, manifest)

    table = load_npz(sidecar / "candidate_table.npz")
    formal = load_npz(sidecar / "formal_selection_snapshot.npz")
    require(table["source_round"].shape == (int(manifest["states"]), 65), "candidate table shape mismatch")
    require(np.array_equal(table["record_key"].astype(str), formal["record_key"].astype(str)), "formal/table record mismatch")
    pools = {source: load_npz(sidecar / relative) for source, relative in POOL_PATH.items()}
    for source, pool in pools.items():
        require(np.array_equal(pool["record_key"].astype(str), table["record_key"].astype(str)), f"record mismatch in source {source}")

    candidates, _ = gather_union(
        table,
        pools,
        ("candidates_feature_waypoint_units", "candidates"),
    )
    assert candidates is not None
    require(np.array_equal(candidates.astype(np.float32), table["candidates_feature_waypoint_units"].astype(np.float32)), "candidate order/value mismatch")
    require(np.array_equal(candidate_hashes(candidates), table["candidate_hash"].astype(str)), "candidate hash/order mismatch")

    actions, _ = gather_union(table, pools, ("actions_h4_m_rad", "actions_h4"))
    raw, _ = gather_union(table, pools, ("raw_ltwm_features",))
    scorer_aux, _ = gather_union(table, pools, ("scorer_v2_aux_features",))
    assert actions is not None and raw is not None and scorer_aux is not None
    require(actions.shape[-2:] == (4, 3), f"bad action shape: {actions.shape}")
    require(raw.shape[-1] == 112, f"bad raw LTWM shape: {raw.shape}")
    require(scorer_aux.shape[-1] == 10, f"bad scorer aux shape: {scorer_aux.shape}")
    require(np.isfinite(actions).all() and np.isfinite(raw).all() and np.isfinite(scorer_aux).all(), "core features contain NaN/Inf")

    ate_seed = gather_seed_union(table, pools, ("ltwm_ate_by_seed", "ate_by_seed"))
    rtp_seed = gather_seed_union(table, pools, ("ltwm_rtp_by_seed", "rtp_by_seed"))
    support_seed = gather_seed_union(table, pools, ("ltwm_support_by_seed", "support_by_seed"))
    require(ate_seed.shape[-1] == rtp_seed.shape[-1] == support_seed.shape[-1] == 3, "seed count changed")

    formal_index = formal_union_index(table, formal)
    action_struct = action_structure_features(actions)
    temporal = raw.reshape(*raw.shape[:2], 4, 7, 4)
    mean, std, minimum, maximum = (temporal[:, :, index] for index in range(4))
    middle = 3.0 * mean - minimum - maximum
    tolerance = 2.0e-4
    order_consistent = bool(
        (std >= -tolerance).all()
        and (minimum <= mean + tolerance).all()
        and (mean <= maximum + tolerance).all()
        and (minimum <= middle + tolerance).all()
        and (middle <= maximum + tolerance).all()
    )
    require(order_consistent, "raw LTWM mean/std/min/max contract is inconsistent")
    order_statistics = np.stack((minimum, middle, maximum), axis=2).astype(np.float32)

    output: dict[str, np.ndarray] = {
        "record_key_audit_only": table["record_key"].astype(str),
        "candidate_hash": table["candidate_hash"].astype(str),
        "formal_v70_union_index": formal_index,
        "is_formal_v70_candidate": np.arange(65)[None, :] == formal_index[:, None],
        "source_round": table["source_round"].astype(np.int8),
        "source_candidate_index": table["source_candidate_index"].astype(np.int8),
        "actions_h4_m_rad": actions.astype(np.float32),
        "action_delta_to_formal_v70": (actions - expand_reference(actions, formal_index)).astype(np.float32),
        "action_structure": action_struct,
        "action_structure_delta_to_formal_v70": (action_struct - expand_reference(action_struct, formal_index)).astype(np.float32),
        "ltwm_temporal_summary": temporal.astype(np.float32),
        "ltwm_temporal_summary_delta_to_formal_v70": (temporal - expand_reference(temporal, formal_index)).astype(np.float32),
        "ltwm_temporal_order_statistics": order_statistics,
        "ltwm_temporal_order_statistics_delta_to_formal_v70": (order_statistics - expand_reference(order_statistics, formal_index)).astype(np.float32),
        "scorer_v2_aux": scorer_aux.astype(np.float32),
        "scorer_v2_aux_delta_to_formal_v70": (scorer_aux - expand_reference(scorer_aux, formal_index)).astype(np.float32),
        "ate_by_seed": ate_seed.astype(np.float32),
        "ate_by_seed_delta_to_formal_v70": (ate_seed - expand_reference(ate_seed, formal_index)).astype(np.float32),
        "rtp_by_seed": rtp_seed.astype(np.float32),
        "rtp_by_seed_delta_to_formal_v70": (rtp_seed - expand_reference(rtp_seed, formal_index)).astype(np.float32),
        "support_by_seed": support_seed.astype(bool),
        "support_by_seed_delta_to_formal_v70": (support_seed.astype(np.int8) - expand_reference(support_seed.astype(np.int8), formal_index)).astype(np.int8),
    }

    optional_aliases = {
        "energy": ("energy",),
        "energy_rank": ("energy_rank",),
        "metric_mean": ("metric_mean", "mean"),
        "metric_std": ("metric_std", "std"),
        "metric_ucb": ("metric_ucb",),
        "dual_probability": ("dual_probability",),
        "harm_probability": ("harm_probability",),
        "rank_logit": ("rank_logit",),
        "support_score": ("support_score",),
        "support_margin": ("support_margin",),
        "valid": ("valid",),
        "safe": ("safe",),
    }
    optional_coverage = {}
    for destination, aliases in optional_aliases.items():
        value, observed = gather_union(table, pools, aliases, optional=True)
        optional_coverage[destination] = float(observed.mean())
        if value is not None:
            output[destination] = value
            output[destination + "_observed"] = observed
            if np.issubdtype(value.dtype, np.floating):
                output[destination + "_delta_to_formal_v70"] = (
                    value - expand_reference(value, formal_index)
                ).astype(np.float32)

    require(not args.output.exists(), f"refusing to overwrite: {args.output}")
    require(not args.audit.exists(), f"refusing to overwrite: {args.audit}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as handle:
        np.savez_compressed(handle, **output)
    output_sha = sha256_file(args.output)
    audit = {
        "schema": SCHEMA + "_audit",
        "status": "complete_spent_smoke_infrastructure_only"
        if manifest.get("cohort_role") == "spent_smoke"
        else "complete_prelabel_collection_features",
        "sidecar": str(sidecar),
        "sidecar_manifest_sha256": sha256_file(manifest_path),
        "verified_sidecar_files": verified,
        "states": int(actions.shape[0]),
        "candidates_per_state": int(actions.shape[1]),
        "formal_v70_reference_mapping_verified": True,
        "candidate_order_and_hash_verified": True,
        "raw_ltwm_layout": {
            "shape": [4, 7, 4],
            "axis_order": ["seed_summary", "latent_block", "horizon"],
            "seed_summary_names": list(SUMMARY_NAMES),
            "latent_block_names": list(LATENT_BLOCK_NAMES),
            "horizon_names": list(HORIZON_NAMES),
            "source_contract": "latent_trajectory_features then mean/std/min/max over three LTWM seeds",
            "true_seed_identity_available": False,
            "order_statistics_note": "min/middle/max reconstructed per scalar; not a consistent seed trajectory",
            "order_statistics_consistent": order_consistent,
        },
        "optional_field_candidate_coverage": optional_coverage,
        "labels_opened": False,
        "labels_or_truth_fields_in_output": False,
        "reserve_accessed": False,
        "planner_or_model_executed": False,
        "formal_behavior_influence": False,
        "training_allowed": manifest.get("cohort_role") != "spent_smoke",
        "performance_evidence_allowed": False,
        "output": str(args.output.resolve()),
        "output_sha256": output_sha,
        "output_array_hashes": {key: array_sha256(value) for key, value in output.items()},
    }
    args.audit.write_text(canonical_json(audit) + "\n", encoding="utf-8")
    print(canonical_json({"output": str(args.output), "output_sha256": output_sha, "audit": str(args.audit), "audit_sha256": sha256_file(args.audit)}))


if __name__ == "__main__":
    main()
