#!/usr/bin/env python3
"""Run the v16 tail-risk protocol with v17 spatial LTWM rollout features."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tools.scoring import train_recon_cem_latent_base_risk_v16 as v16
from tools.scoring.run_recon_cem_latent_base_risk_v16 import (  # noqa: E402
    audited_base_score,
)


ROLLOUT_PATHS = (
    Path("outputs/cem_development_v1/recon/cem_spatial_rollout_v17/dev1.npz"),
    Path(
        "outputs/cem_development_v1/recon/cem_spatial_rollout_v17/"
        "confirm2.npz"
    ),
    Path(
        "outputs/cem_development_v1/recon/cem_spatial_rollout_v17/"
        "fresh_spent3.npz"
    ),
    Path(
        "outputs/cem_development_v1/recon/cem_spatial_rollout_v17/"
        "v11_replication_spent4.npz"
    ),
)


def load_rollouts() -> tuple[dict[str, np.ndarray], list[dict[str, str]]]:
    result = {}
    records = []
    for path in ROLLOUT_PATHS:
        with np.load(path, allow_pickle=False) as payload:
            key = payload["record_key"].astype(str)
            feature = payload["spatial_rollout_features"].astype(np.float32)
            metadata = json.loads(str(payload["metadata"]))
        if metadata.get("outcomes_accessed") or metadata.get("labels_in_output"):
            raise RuntimeError("v17 rollout cache is not label-free")
        for record_key, value in zip(key, feature):
            if record_key in result:
                raise RuntimeError(f"duplicate v17 record: {record_key}")
            result[record_key] = value
        records.append(
            {
                "path": str(path),
                "sha256": v16.file_sha256(path),
            }
        )
    return result, records


ROLLOUT_BY_RECORD, ROLLOUT_RECORDS = load_rollouts()
_compact_base_feature = v16.compact_base_feature


def spatial_compact_base_feature(
    base_feature_path: Path,
    base_score_path: Path,
    latent_feature: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    feature, support = _compact_base_feature(
        base_feature_path,
        base_score_path,
        latent_feature,
    )
    with np.load(base_feature_path, allow_pickle=False) as payload:
        record_key = payload["record_key"].astype(str)
    missing = [value for value in record_key if value not in ROLLOUT_BY_RECORD]
    if missing:
        raise RuntimeError(f"missing v17 rollout records: {missing[:3]}")
    rollout = np.stack(
        [ROLLOUT_BY_RECORD[value] for value in record_key]
    )[:, 1:]
    if rollout.shape != (len(feature), 16, 258):
        raise RuntimeError("v17 rollout alignment changed")
    return np.concatenate((feature, rollout), axis=2), support


def result_output_dir() -> Path:
    try:
        index = sys.argv.index("--output-dir")
    except ValueError as error:
        raise RuntimeError("--output-dir is required") from error
    return Path(sys.argv[index + 1])


if __name__ == "__main__":
    v16.resolve_base_score = audited_base_score
    v16.compact_base_feature = spatial_compact_base_feature
    v16.main()
    result_path = result_output_dir() / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["schema"] = "recon_cem_spatial_rollout_base_risk_v17"
    result["candidate_conditioned_future_latent_used"] = True
    result["spatial_rollout_dim"] = 258
    result["spatial_rollout_contract"] = (
        "three-seed mean/std of real LTWM predicted-token H1-H4 spatial "
        "alignment to the goal; no action proxy, target future, or outcome"
    )
    result["spatial_rollout_caches"] = ROLLOUT_RECORDS
    result["status"] = (
        "loco_base_safety_passed_cem_replacement_may_train"
        if result["policy_search"]["strict"]
        else "loco_base_safety_failed_stop_same_feature_tuning"
    )
    v16.write_json(result_path, result)
    print(
        json.dumps(
            {
                "result": str(result_path),
                "result_sha256": v16.file_sha256(result_path),
                "schema": result["schema"],
                "status": result["status"],
                "policy_search": result["policy_search"],
                "top40": result[
                    "noncausal_top40_feature_separability_diagnostic_only"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
