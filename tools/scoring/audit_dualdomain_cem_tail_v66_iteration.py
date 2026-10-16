#!/usr/bin/env python3
"""Audit a frozen v66 harvest iteration without opening outcome labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from tools.evaluation.export_recon_dense_dinov3_features_v1 import load_npz  # noqa: E402
from tools.scoring.train_recon_dense_meta_dinov3_v1 import file_sha256  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("iteration_dir", type=Path)
    args = parser.parse_args()
    population_path = args.iteration_dir / "population.npz"
    feature_path = args.iteration_dir / "features.npz"
    score_path = args.iteration_dir / "scores.npz"
    population = load_npz(population_path)
    feature = load_npz(feature_path)
    score = load_npz(score_path)
    record = score["record_key"].astype(str)
    if not (
        np.array_equal(record, population["record_key"].astype(str))
        and np.array_equal(record, feature["record_key"].astype(str))
    ):
        raise RuntimeError("iteration artifact identity mismatch")
    valid = score["valid"].astype(bool)
    safe = score["safe"].astype(bool)
    energy = score["energy"].astype(np.float32)
    elite = score["elite_index_for_update"].astype(np.int64)
    updated = score["proposal_updated"].astype(bool)
    safe_count = score["safe_count"].astype(np.int64)
    if valid.shape != safe.shape or valid.shape != energy.shape:
        raise RuntimeError("score array shape mismatch")
    if np.any(safe & ~valid):
        raise RuntimeError("safe mask is not a subset of valid")
    if safe[:, 0].any():
        raise RuntimeError("incumbent must not enter the safe challenger set")
    if not np.array_equal(safe_count, safe.sum(axis=1)):
        raise RuntimeError("stored safe_count differs from safe mask")
    if not np.array_equal(updated, safe_count >= elite.shape[1]):
        raise RuntimeError("proposal update/fallback contract mismatch")
    if np.any(elite[~updated] != -1):
        raise RuntimeError("fallback window contains elite indices")
    for state in np.flatnonzero(updated):
        index = elite[state]
        if len(np.unique(index)) != len(index):
            raise RuntimeError("updated window contains duplicate elites")
        if not safe[state, index].all():
            raise RuntimeError("elite contains a non-safe candidate")
        if not np.isfinite(energy[state, index]).all():
            raise RuntimeError("elite contains non-finite energy")
    challenger_valid = valid[:, 1:]
    challenger_safe = safe[:, 1:]
    output = {
        "schema": "dualdomain_cem_tail_v66_iteration_structural_audit",
        "status": "passed",
        "iteration_dir": str(args.iteration_dir),
        "states": int(len(record)),
        "candidates_per_state": int(valid.shape[1]),
        "valid_candidates": int(challenger_valid.sum()),
        "valid_fraction": float(challenger_valid.mean()),
        "safe_candidates": int(challenger_safe.sum()),
        "safe_fraction_of_valid": float(
            challenger_safe.sum() / max(challenger_valid.sum(), 1)
        ),
        "windows_with_any_safe": int(challenger_safe.any(axis=1).sum()),
        "updated_windows": int(updated.sum()),
        "fallback_windows": int((~updated).sum()),
        "safe_count_min": int(safe_count.min()),
        "safe_count_mean": float(safe_count.mean()),
        "safe_count_p50": float(np.quantile(safe_count, 0.5)),
        "safe_count_p90": float(np.quantile(safe_count, 0.9)),
        "elite_count": int(elite.shape[1]),
        "population_sha256": file_sha256(population_path),
        "features_sha256": file_sha256(feature_path),
        "scores_sha256": file_sha256(score_path),
        "targets_opened": False,
        "A_to_H_accessed": False,
        "reserve_accessed": False,
    }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
