#!/usr/bin/env python3
"""Run tail-risk LOCO with candidate and candidate-minus-incumbent rollouts."""

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
from tools.scoring.run_recon_cem_spatial_rollout_base_risk_v17 import (  # noqa: E402
    ROLLOUT_BY_RECORD,
    ROLLOUT_RECORDS,
)


_compact_base_feature = v16.compact_base_feature


def spatial_delta_compact_base_feature(
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
        raise RuntimeError(f"missing v18 rollout records: {missing[:3]}")
    rollout = np.stack(
        [ROLLOUT_BY_RECORD[value] for value in record_key]
    )
    candidate = rollout[:, 1:]
    difference = candidate - rollout[:, :1]
    if candidate.shape != (len(feature), 16, 258):
        raise RuntimeError("v18 rollout alignment changed")
    return np.concatenate((feature, candidate, difference), axis=2), support


def result_output_dir() -> Path:
    try:
        index = sys.argv.index("--output-dir")
    except ValueError as error:
        raise RuntimeError("--output-dir is required") from error
    return Path(sys.argv[index + 1])


if __name__ == "__main__":
    v16.resolve_base_score = audited_base_score
    v16.compact_base_feature = spatial_delta_compact_base_feature
    v16.main()
    result_path = result_output_dir() / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["schema"] = "recon_cem_spatial_delta_base_risk_v18"
    result["candidate_conditioned_future_latent_used"] = True
    result["spatial_rollout_dim"] = 258
    result["spatial_rollout_delta_dim"] = 258
    result["spatial_rollout_contract"] = (
        "v17 candidate spatial rollout plus candidate-minus-incumbent "
        "spatial rollout residual; outcome-free and deployment-computable"
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
