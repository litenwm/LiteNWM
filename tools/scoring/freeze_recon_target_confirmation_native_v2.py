#!/usr/bin/env python3
"""Freeze native scorer-v2 choices before target confirmation outcomes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from scoring.safe_promotion_v1 import greedy_fixed_pool_selection
from tools.scoring.train_recon_dense_meta_dinov3_three_arm_v1 import load_features
from tools.scoring.train_recon_dense_meta_dinov3_v1 import file_sha256, write_json, write_npz


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirmation-root", type=Path, required=True)
    parser.add_argument("--feature-name", required=True)
    parser.add_argument("--native-policy", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result_path = args.output_dir / "freeze.json"
    selected_path = args.output_dir / "selected_index.npz"
    if result_path.exists() or selected_path.exists():
        raise FileExistsError(result_path)
    if (args.confirmation_root / "source_labeled.npz").exists():
        raise RuntimeError("confirmation labels already opened")
    source_path = args.confirmation_root / "source_label_free.npz"
    feature_path = args.confirmation_root / "features" / args.feature_name
    data = load_features(feature_path, labels_expected=False)
    metadata = data["metadata"]
    if metadata.get("labels_accessed") or metadata.get("labels_in_output"):
        raise RuntimeError("feature artifact contains labels")
    with np.load(source_path, allow_pickle=False) as payload:
        record = payload["record_key"].astype(str)
        trajectory = payload["trajectory"].astype(str)
        route = payload["route_group"].astype(str)
    if not np.array_equal(data["record_key"].astype(str), record):
        raise RuntimeError("feature/source identity mismatch")
    native_policy = json.loads(args.native_policy.read_text(encoding="utf-8"))
    native = native_policy["arms"]["dinov3_scorer_v2_native"]
    selected = greedy_fixed_pool_selection(
        data["ate_by_seed"], data["rtp_by_seed"],
        ate_scale=float(native["ate_scale"]),
        rtp_scale=float(native["rtp_scale"]),
        valid=data["majority_support"],
    ).astype(np.int64)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_npz(
        selected_path, record_key=record, trajectory=trajectory,
        route_group=route, native_scorer_v2_selected_index=selected,
    )
    report = {
        "schema": "recon_target_confirmation_native_v2_selection_freeze",
        "status": "native_v2_selected_before_confirmation_outcomes",
        "states": len(record),
        "promotions_label_blind": int((selected != 0).sum()),
        "coverage_label_blind": float((selected != 0).mean()),
        "feature_contract": native["feature_contract"],
        "ate_scale": float(native["ate_scale"]),
        "rtp_scale": float(native["rtp_scale"]),
        "native_policy": str(args.native_policy),
        "native_policy_sha256": file_sha256(args.native_policy),
        "feature_sha256": file_sha256(feature_path),
        "source_label_free_sha256": file_sha256(source_path),
        "selected_index": str(selected_path),
        "selected_index_sha256": file_sha256(selected_path),
        "labels_opened": False,
        "outcomes_accessed": False,
        "cem": False,
    }
    write_json(result_path, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
