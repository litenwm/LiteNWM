#!/usr/bin/env python3
"""Run v20 audit with latent identity verified by frozen SHA metadata."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tools.evaluation import audit_recon_target_calibration_v20 as audit
from tools.scoring.train_recon_dense_meta_dinov3_v1 import file_sha256


def audit_bank(root: Path, letter: str):
    bank = root / f"bank_{letter}"
    feature = bank / "features" / f"recon_target_bank_{letter}_v20_features.npz"
    latent = bank / "features" / f"recon_target_bank_{letter}_v20_latents.npz"
    rollout = bank / "spatial_rollout_v17.npz"
    score = bank / "universal_scores.npz"
    source = bank / "source_labeled.npz"
    paths = (feature, latent, rollout, score, source)
    if any(not path.is_file() for path in paths):
        raise FileNotFoundError([str(path) for path in paths if not path.is_file()])
    with np.load(source, allow_pickle=False) as payload:
        source_record = payload["record_key"].astype(str)
        source_route = payload["route_group"].astype(str)
        source_trajectory = payload["trajectory"].astype(str)
        targets = payload["targets"]
    with np.load(feature, allow_pickle=False) as payload:
        feature_record = payload["record_key"].astype(str)
        feature_meta = json.loads(str(payload["metadata"]))
        feature_arrays = [payload[key] for key in (
            "static_features", "raw_ltwm_features", "actions_h4",
            "candidates", "majority_support",
        )]
    with np.load(latent, allow_pickle=False) as payload:
        history = payload["history"]
        goal = payload["goal"]
    with np.load(rollout, allow_pickle=False) as payload:
        rollout_record = payload["record_key"].astype(str)
        spatial = payload["spatial_rollout_features"]
        rollout_meta = json.loads(str(payload["metadata"]))
    with np.load(score, allow_pickle=False) as payload:
        score_record = payload["record_key"].astype(str)
        score_arrays = [payload[key] for key in (
            "mean", "std", "dual_probability", "harm_probability", "rank_logit",
        )]
        score_meta = json.loads(str(payload["metadata"]))
    identity_ok = (
        np.array_equal(source_record, feature_record)
        and np.array_equal(source_record, rollout_record)
        and np.array_equal(source_record, score_record)
    )
    shape_ok = (
        targets.shape == (100, 17, 8)
        and history.shape == (100, 4, 256, 384)
        and goal.shape == (100, 256, 384)
        and spatial.shape == (100, 17, 258)
    )
    finite_ok = all(np.isfinite(value).all() for value in (
        [targets, history, goal, spatial] + feature_arrays + score_arrays
    ))
    latent_contract_ok = feature_meta.get("latent_cache_sha256") == file_sha256(latent)
    label_contract_ok = (
        feature_meta.get("labels_in_output") is True
        and rollout_meta.get("labels_in_output") is False
        and rollout_meta.get("outcomes_accessed") is False
        and score_meta.get("outcomes_accessed") is False
        and latent_contract_ok
    )
    report = {
        "states": int(len(source_record)),
        "routes": int(len(set(source_route))),
        "trajectories": int(len(set(source_trajectory))),
        "identity_ok": identity_ok,
        "latent_contract_ok": latent_contract_ok,
        "shape_ok": shape_ok,
        "finite_ok": finite_ok,
        "label_contract_ok": label_contract_ok,
        "shapes": {
            "targets": list(targets.shape),
            "history": list(history.shape),
            "goal": list(goal.shape),
            "spatial_rollout_features": list(spatial.shape),
        },
        "sha256": {path.name: file_sha256(path) for path in paths},
    }
    if not all((identity_ok, shape_ok, finite_ok, label_contract_ok)):
        raise RuntimeError(f"bank_{letter} audit failed: {report}")
    return report, set(source_record), set(source_route)


if __name__ == "__main__":
    audit.audit_bank = audit_bank
    audit.main()
