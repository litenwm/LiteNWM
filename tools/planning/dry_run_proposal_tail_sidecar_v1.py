#!/usr/bin/env python3
"""Serialize one spent SCAND window from cached v70 artifacts for smoke only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from planning.proposal_tail_sidecar_v1 import (  # noqa: E402
    FEATURE_WAYPOINT_UNIT_M,
    METRIC_NAMES,
    SCHEMA,
    array_sha256,
    candidate_hashes,
    canonical_json,
    reject_forbidden_path,
    sha256_file,
    snapshot_states,
)


EXPECTED_SELECTED_SHA256 = "95b6fcf79f8540ebb33f692571c335745d540e67998534b461629937e91b6ac0"
EXPECTED_PRELABEL_SHA256 = "7b25a44e769cfb0f7f0eb705256256cbe371c88b62c0765b06ecc6f3ddcda7fc"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--base-features", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--state-index", type=int, default=0)
    return parser.parse_args()


def write_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        np.savez_compressed(handle, **arrays)


def state_slice(key: str, value: np.ndarray, index: int, states: int) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim and array.shape[0] == states:
        return array[index : index + 1]
    if array.ndim > 1 and array.shape[1] == states and key.endswith("_by_seed"):
        return array[:, index : index + 1]
    return array


def main() -> None:
    args = parse_args()
    for path in (args.replay_dir, args.manifest, args.base_features, args.output_dir):
        reject_forbidden_path(path)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(args.output_dir)
    selected_path = args.replay_dir / "selected_actions.npz"
    prelabel_path = args.replay_dir / "prelabel_freeze.json"
    selected_sha_before = sha256_file(selected_path)
    prelabel_sha_before = sha256_file(prelabel_path)
    if selected_sha_before != EXPECTED_SELECTED_SHA256:
        raise RuntimeError("spent formal selected-actions SHA differs from audited v70")
    if prelabel_sha_before != EXPECTED_PRELABEL_SHA256:
        raise RuntimeError("spent formal prelabel SHA differs from audited v70")
    rows = [
        json.loads(line)
        for line in args.manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    states = len(rows)
    if not 0 <= args.state_index < states:
        raise IndexError(args.state_index)
    state_rows = [rows[args.state_index]]
    snapshots = snapshot_states(state_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "states.jsonl").write_text(
        canonical_json(snapshots[0]) + "\n", encoding="utf-8"
    )

    base = np.load(args.base_features, allow_pickle=False)
    fixed = base["candidates"][args.state_index : args.state_index + 1].astype(np.float32)
    fixed_arrays = {
        key: state_slice(key, base[key], args.state_index, states)
        for key in base.files
        if key in {
            "candidates",
            "actions_h4",
            "action_features",
            "raw_ltwm_features",
            "scorer_v2_aux_features",
            "support_features",
            "ate_by_seed",
            "rtp_by_seed",
            "support_by_seed",
            "majority_support",
            "record_key",
        }
    }
    write_npz(args.output_dir / "fixed_pool/candidates_and_scores.npz", **fixed_arrays)
    round_candidates = []
    for iteration in range(3):
        directory = args.replay_dir / f"iteration_{iteration:02d}"
        population = np.load(directory / "population.npz", allow_pickle=False)
        feature = np.load(directory / "cem_features.npz", allow_pickle=False)
        score = np.load(directory / "frozen_scores.npz", allow_pickle=False)
        candidate = population["candidates_feature_waypoint_units"][
            args.state_index : args.state_index + 1
        ].astype(np.float32)
        round_candidates.append(candidate[:, 1:])
        arrays = {
            "candidates_feature_waypoint_units": candidate,
            "candidate_hash": candidate_hashes(candidate),
        }
        for source in (feature, score):
            for key in source.files:
                if key in {"metadata"}:
                    continue
                arrays[key] = state_slice(key, source[key], args.state_index, states)
        write_npz(
            args.output_dir / f"iteration_{iteration:02d}/candidates_and_scores.npz",
            **arrays,
        )

    combined = np.concatenate((fixed, *round_candidates), axis=1)
    source_round = np.asarray([-1] * 17 + [value for value in range(3) for _ in range(16)], dtype=np.int8)
    source_index = np.asarray(list(range(17)) + [value for _ in range(3) for value in range(1, 17)], dtype=np.int8)
    record_key = np.asarray([snapshots[0]["record_key"]])
    write_npz(
        args.output_dir / "candidate_table.npz",
        record_key=record_key,
        candidates_feature_waypoint_units=combined,
        candidate_hash=candidate_hashes(combined),
        source_round=np.broadcast_to(source_round, combined.shape[:2]),
        source_candidate_index=np.broadcast_to(source_index, combined.shape[:2]),
    )
    selected = np.load(selected_path, allow_pickle=False)
    selected_arrays = {
        key: state_slice(key, selected[key], args.state_index, states)
        for key in selected.files
        if key in {
            "record_key",
            "fixed_pool_selected_index",
            "final_cem_selected_index",
            "final_cem_energy",
            "final_cem_valid",
            "final_cem_safe",
            "archive_energy",
            "archive_source_round",
            "archive_source_candidate",
        }
    }
    if "final_cem_selected_index" in selected_arrays:
        selected_arrays["fallback_to_incumbent"] = (
            selected_arrays["final_cem_selected_index"] == 0
        )
    write_npz(args.output_dir / "formal_selection_snapshot.npz", **selected_arrays)
    selected_sha_after = sha256_file(selected_path)
    prelabel_sha_after = sha256_file(prelabel_path)
    if selected_sha_after != selected_sha_before or prelabel_sha_after != prelabel_sha_before:
        raise RuntimeError("sidecar dry-run changed a formal artifact")

    files = {
        str(path.relative_to(args.output_dir)): sha256_file(path)
        for path in sorted(args.output_dir.rglob("*"))
        if path.is_file() and path.name != "collection_manifest.json"
    }
    manifest = {
        "schema": SCHEMA,
        "status": "cached_serialization_dry_run",
        "cohort": "scand_fresh_outer_state0_spent",
        "cohort_role": "spent_smoke",
        "evidence_role": "pipeline_smoke_only_spent_nonreserve",
        "formal_behavior_influence": False,
        "raw_draws_observable": False,
        "states": 1,
        "candidate_count": 65,
        "seed_count": 3,
        "seed_ids": [1701, 1702, 1703],
        "metric_names": list(METRIC_NAMES),
        "feature_waypoint_unit_m": FEATURE_WAYPOINT_UNIT_M,
        "support_quantile": 0.999,
        "support_threshold": 4.876645565032959,
        "physical_coordinate_names": [
            f"horizon_{step}_{axis}" for step in range(1, 5) for axis in ("dx_over_spacing", "dy_over_spacing", "dyaw_rad")
        ],
        "identity": {
            "manifest_sha256": sha256_file(args.manifest),
            "record_key_sha256": array_sha256(record_key),
        },
        "formal_baseline_audit": {
            "selected_sha256_before": selected_sha_before,
            "selected_sha256_after": selected_sha_after,
            "selected_matches_prechange_audit": True,
            "prelabel_sha256_before": prelabel_sha_before,
            "prelabel_sha256_after": prelabel_sha_after,
            "prelabel_matches_prechange_audit": True,
            "selected_energy_archive_snapshot_is_exact_slice": True,
        },
        "unavailable_fields": {
            "raw_prefilter_draws": "historical sampler rejected them before serialization",
            "sampling_rng_state": "historical cache did not serialize RNG state",
            "camera_metadata": "spent manifest has no explicit resolution/encoding/quality fields",
            "raw_timestamps": "spent manifest has frame indices but no raw timestamps",
            "seed_specific_rpe_endpoint_objective_logits": "not emitted by frozen feature/predict interfaces",
            "fixed_pool_v70_aggregate_scores": "not serialized with this fixed feature artifact; dry-run does not rerun scorer",
        },
        "files": files,
        "labels_opened": False,
        "reserve_accessed": False,
        "training_allowed": False,
        "performance_evidence_allowed": False,
    }
    manifest_path = args.output_dir / "collection_manifest.json"
    manifest_path.write_text(canonical_json(manifest) + "\n", encoding="utf-8")
    print(canonical_json({"status": "pass", "manifest": str(manifest_path), "sha256": sha256_file(manifest_path)}))


if __name__ == "__main__":
    main()
