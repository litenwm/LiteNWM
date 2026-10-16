#!/usr/bin/env python3
"""Verify that the spent proposal-tail smoke is an exact formal-artifact slice."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from planning.proposal_tail_sidecar_v1 import canonical_json, sha256_file  # noqa: E402


FORMAL = ROOT / "outputs/cem_development_v1/dualdomain_cem_tail_v70_broad_hashfold_archive/replays/scand_fresh_outer/proposal_aware_v70"
BASE = ROOT / "outputs/scand_domain_adaptation_v1/fresh_outer_v1/features/scand_fresh_outer_h4_100_seed20260720_features.npz"
SMOKE = ROOT / "outputs/proposal_tail_collection_v1/pipeline_smoke_spent_nonreserve_state0"
OUTPUT = SMOKE / "behavioral_equivalence.audit.json"
EXPECTED_SELECTED_SHA256 = "95b6fcf79f8540ebb33f692571c335745d540e67998534b461629937e91b6ac0"
EXPECTED_PRELABEL_SHA256 = "7b25a44e769cfb0f7f0eb705256256cbe371c88b62c0765b06ecc6f3ddcda7fc"


def main() -> None:
    selected_path = FORMAL / "selected_actions.npz"
    prelabel_path = FORMAL / "prelabel_freeze.json"
    selected_sha = sha256_file(selected_path)
    prelabel_sha = sha256_file(prelabel_path)
    if selected_sha != EXPECTED_SELECTED_SHA256 or prelabel_sha != EXPECTED_PRELABEL_SHA256:
        raise RuntimeError("formal v70 artifact hash differs from frozen audit")

    with np.load(selected_path, allow_pickle=False) as source, np.load(
        SMOKE / "formal_selection_snapshot.npz", allow_pickle=False
    ) as snapshot:
        exact_keys: dict[str, bool] = {}
        for key in snapshot.files:
            if key == "fallback_to_incumbent":
                continue
            if key not in source.files:
                raise KeyError(f"snapshot key missing from source: {key}")
            exact_keys[key] = bool(np.array_equal(snapshot[key], source[key][0:1]))
        fallback_expected = source["final_cem_selected_index"][0:1] == 0
        fallback_exact = bool(
            np.array_equal(snapshot["fallback_to_incumbent"], fallback_expected)
        )
    if not all(exact_keys.values()) or not fallback_exact:
        raise RuntimeError("formal selection snapshot is not an exact state-0 slice")

    with np.load(BASE, allow_pickle=False) as base:
        expected_parts = [base["candidates"][0:1].astype(np.float32)]
    round_shapes: list[list[int]] = []
    for iteration in range(3):
        path = FORMAL / f"iteration_{iteration:02d}/population.npz"
        with np.load(path, allow_pickle=False) as population:
            candidates = population["candidates_feature_waypoint_units"][0:1].astype(
                np.float32
            )
        round_shapes.append(list(candidates.shape))
        expected_parts.append(candidates[:, 1:])
    expected_candidates = np.concatenate(expected_parts, axis=1)
    with np.load(SMOKE / "candidate_table.npz", allow_pickle=False) as table:
        actual_candidates = table["candidates_feature_waypoint_units"]
        candidate_order_exact = bool(np.array_equal(actual_candidates, expected_candidates))
        candidate_shape = list(actual_candidates.shape)
        source_round = table["source_round"]
        source_index = table["source_candidate_index"]
    expected_round = np.asarray(
        [-1] * 17 + [round_id for round_id in range(3) for _ in range(16)],
        dtype=np.int8,
    )[None, :]
    expected_index = np.asarray(
        list(range(17)) + [index for _ in range(3) for index in range(1, 17)],
        dtype=np.int8,
    )[None, :]
    source_mapping_exact = bool(
        np.array_equal(source_round, expected_round)
        and np.array_equal(source_index, expected_index)
    )
    if not candidate_order_exact or not source_mapping_exact or candidate_shape != [1, 65, 4, 2]:
        raise RuntimeError("cached fixed+r1+r2+r3 candidate order differs")

    audit = {
        "schema": "proposal_tail_smoke_behavioral_equivalence_audit_v1",
        "status": "pass",
        "scope": "spent_scand_fresh_outer_state0_pipeline_smoke_only",
        "formal_selected_actions_sha256": selected_sha,
        "formal_prelabel_freeze_sha256": prelabel_sha,
        "formal_snapshot_exact_by_key": exact_keys,
        "fallback_to_incumbent_exact": fallback_exact,
        "candidate_order_fixed_plus_r1_r2_r3_exact": candidate_order_exact,
        "candidate_source_mapping_exact": source_mapping_exact,
        "candidate_shape": candidate_shape,
        "source_round_shapes": round_shapes,
        "raw_sampler_rng_state": "historically_unobservable; covered by synthetic parity regression",
        "reserve_accessed": False,
        "performance_evidence": False,
    }
    OUTPUT.write_text(canonical_json(audit) + "\n", encoding="utf-8")
    print(
        canonical_json(
            {"status": audit["status"], "output": str(OUTPUT), "sha256": sha256_file(OUTPUT)}
        )
    )


if __name__ == "__main__":
    main()
