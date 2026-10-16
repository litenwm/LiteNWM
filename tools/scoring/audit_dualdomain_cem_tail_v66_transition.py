#!/usr/bin/env python3
"""Verify that a v66 q_t -> q_{t+1} transition is exactly safe-elite driven."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from tools.evaluation.export_recon_dense_dinov3_features_v1 import load_npz  # noqa: E402
from tools.planning.run_cem_safe_elite_v55 import safe_elite_update  # noqa: E402
from tools.scoring.train_recon_dense_meta_dinov3_v1 import file_sha256  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("previous_iteration", type=Path)
    parser.add_argument("next_iteration", type=Path)
    args = parser.parse_args()
    previous_population_path = args.previous_iteration / "population.npz"
    previous_scores_path = args.previous_iteration / "scores.npz"
    next_population_path = args.next_iteration / "population.npz"
    previous = load_npz(previous_population_path)
    score = load_npz(previous_scores_path)
    following = load_npz(next_population_path)
    record = previous["record_key"].astype(str)
    if not (
        np.array_equal(record, score["record_key"].astype(str))
        and np.array_equal(record, following["record_key"].astype(str))
    ):
        raise RuntimeError("transition identity mismatch")
    mean, covariance, elite, updated, safe_count = safe_elite_update(
        previous["candidates_feature_waypoint_units"].astype(np.float32),
        score["energy"].astype(np.float32),
        score["safe"].astype(bool),
        previous_mean=previous["proposal_mean"].astype(np.float32),
        previous_covariance=previous["proposal_covariance"].astype(np.float32),
        elite_count=4,
        floor=0.015,
        scale=0.70,
    )
    stored_elite = following["elite_index_from_previous"].astype(np.int64)
    if not np.array_equal(elite, stored_elite):
        raise RuntimeError("next iteration elite binding mismatch")
    mean_error = np.abs(mean - following["proposal_mean"].astype(np.float32))
    covariance_error = np.abs(
        covariance - following["proposal_covariance"].astype(np.float32)
    )
    if mean_error.max(initial=0.0) != 0.0:
        raise RuntimeError("q mean is not the exact safe-elite update")
    if covariance_error.max(initial=0.0) != 0.0:
        raise RuntimeError("q covariance is not the exact safe-elite update")
    fallback = ~updated
    previous_mean = previous["proposal_mean"].astype(np.float32)
    previous_covariance = previous["proposal_covariance"].astype(np.float32)
    output = {
        "schema": "dualdomain_cem_tail_v66_transition_audit",
        "status": "passed",
        "states": int(len(record)),
        "updated_windows": int(updated.sum()),
        "fallback_windows": int(fallback.sum()),
        "fallback_mean_unchanged": bool(np.array_equal(mean[fallback], previous_mean[fallback])),
        "fallback_covariance_unchanged": bool(
            np.array_equal(covariance[fallback], previous_covariance[fallback])
        ),
        "safe_count_min": int(safe_count.min()),
        "safe_count_mean": float(safe_count.mean()),
        "proposal_mean_max_abs_error": float(mean_error.max(initial=0.0)),
        "proposal_covariance_max_abs_error": float(
            covariance_error.max(initial=0.0)
        ),
        "previous_population_sha256": file_sha256(previous_population_path),
        "previous_scores_sha256": file_sha256(previous_scores_path),
        "next_population_sha256": file_sha256(next_population_path),
        "targets_opened": False,
        "A_to_H_accessed": False,
        "reserve_accessed": False,
    }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
