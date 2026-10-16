#!/usr/bin/env python3
"""Freeze the native scorer-v2 selection before fresh-outer labels open."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from scoring.safe_promotion_v1 import greedy_fixed_pool_selection  # noqa: E402
from tools.scoring.train_recon_dense_meta_dinov3_three_arm_v1 import (  # noqa: E402
    load_features,
)
from tools.scoring.train_recon_dense_meta_dinov3_v1 import (  # noqa: E402
    file_sha256,
    write_json,
    write_npz,
)


ROOT = REPO_ROOT / "outputs/scand_domain_adaptation_v1/fresh_outer_v1"


def main() -> None:
    feature_path = (
        ROOT / "features/scand_fresh_outer_h4_100_seed20260720_features.npz"
    )
    policy_path = REPO_ROOT / "outputs/scand_domain_adaptation_v1/training_v1/result.json"
    primary_freeze_path = ROOT / "prelabel_freeze/prelabel_freeze.json"
    output_path = ROOT / "prelabel_freeze/native_v2_selected_index.npz"
    complete_path = ROOT / "prelabel_freeze/prelabel_freeze_complete.json"
    if output_path.exists() or complete_path.exists():
        raise FileExistsError(output_path)
    data = load_features(feature_path, labels_expected=False)
    if data["metadata"]["labels_accessed"] or data["metadata"]["labels_in_output"]:
        raise RuntimeError("fresh-outer feature artifact contains labels")
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    primary = json.loads(primary_freeze_path.read_text(encoding="utf-8"))
    if primary.get("labels_opened") or primary.get("future_pose_outcomes_opened"):
        raise RuntimeError("primary freeze provenance already opened labels")
    selected = greedy_fixed_pool_selection(
        data["ate_by_seed"],
        data["rtp_by_seed"],
        ate_scale=float(policy["scorer_v2"]["ate_scale"]),
        rtp_scale=float(policy["scorer_v2"]["rtp_scale"]),
        valid=data["majority_support"],
    )
    write_npz(
        output_path,
        record_key=data["record_key"].astype(str),
        family=data["family"].astype(str),
        native_v2_selected_index=selected.astype(np.int64),
    )
    complete = {
        "schema": "scand_fresh_outer_prelabel_complete_v1",
        "status": "all_fixed_pool_selections_frozen_before_pose_labels",
        "primary_freeze": str(primary_freeze_path),
        "primary_freeze_sha256": file_sha256(primary_freeze_path),
        "six_arm_selected_index": primary["selected_index"],
        "six_arm_selected_index_sha256": primary["selected_index_sha256"],
        "native_v2_selected_index": str(output_path),
        "native_v2_selected_index_sha256": file_sha256(output_path),
        "native_v2_promotions": int((selected != 0).sum()),
        "incumbent_index": 0,
        "candidate_pool_sha256": primary["candidate_pool_sha256"],
        "feature_artifact_sha256": primary["feature_artifact_sha256"],
        "training_result_sha256": primary["training_result_sha256"],
        "labels_opened": False,
        "future_pose_outcomes_opened": False,
        "historically_spent": False,
        "fixed_pool_only": True,
        "cem": False,
    }
    write_json(complete_path, complete)
    print(json.dumps(complete, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
