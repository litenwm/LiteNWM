#!/usr/bin/env python3
"""Freeze six-arm selections for the truly unused SCAND fresh outer H4-100."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from scoring.spacing_support_v1 import support_mask  # noqa: E402
from tools.evaluation.freeze_scand_internal_holdout_selections_v1 import (  # noqa: E402
    load_members,
    load_support,
    native_selection,
    predict_modes,
    raw_selection,
    selection_audit,
)
from tools.scoring.train_recon_dense_meta_dinov3_three_arm_v1 import (  # noqa: E402
    load_features,
)
from tools.scoring.train_recon_dense_meta_dinov3_v1 import (  # noqa: E402
    file_sha256,
    write_json,
    write_npz,
)


DEFAULT_ROOT = REPO_ROOT / "outputs/scand_domain_adaptation_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features",
        type=Path,
        default=DEFAULT_ROOT
        / "fresh_outer_v1/features/scand_fresh_outer_h4_100_seed20260720_features.npz",
    )
    parser.add_argument(
        "--candidate",
        type=Path,
        default=DEFAULT_ROOT / "fresh_outer_v1/fresh_outer_k17.npz",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_ROOT / "fresh_outer_v1/manifest.windows.jsonl",
    )
    parser.add_argument(
        "--manifest-audit",
        type=Path,
        default=DEFAULT_ROOT / "fresh_outer_v1/manifest.windows.audit.json",
    )
    parser.add_argument(
        "--excluded-universe",
        type=Path,
        default=DEFAULT_ROOT
        / "fresh_outer_v1/manifest.windows.excluded_trajectories.json",
    )
    parser.add_argument(
        "--training-result",
        type=Path,
        default=DEFAULT_ROOT / "training_v1/result.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_ROOT / "fresh_outer_v1/prelabel_freeze",
    )
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


def read_manifest(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    rows = read_manifest(args.manifest)
    if len(rows) != 100:
        raise RuntimeError("fresh outer is not H4-100")
    trajectories = {str(row["trajectory_id"]) for row in rows}
    routes = {str(row["route_group"]) for row in rows}
    bags = {str(row["bag_id"]) for row in rows}
    if (len(trajectories), len(routes), len(bags)) != (25, 25, 25):
        raise RuntimeError("fresh outer trajectory/route/bag diversity changed")
    manifest_audit = json.loads(args.manifest_audit.read_text(encoding="utf-8"))
    if manifest_audit.get("status") != "manifest_frozen_rgb_only":
        raise RuntimeError("fresh outer manifest is not frozen")
    if manifest_audit.get("labels_opened") or manifest_audit.get("pose_outcomes_opened"):
        raise RuntimeError("fresh outer manifest provenance opened labels")
    excluded = json.loads(args.excluded_universe.read_text(encoding="utf-8"))
    if trajectories.intersection(excluded["used_trajectories"]):
        raise RuntimeError("fresh outer overlaps excluded trajectory universe")
    if routes.intersection(excluded["used_routes"]):
        raise RuntimeError("fresh outer overlaps excluded route universe")

    data = load_features(args.features, labels_expected=False)
    if data["metadata"]["split"] != "scand_fresh_outer_h4_100_seed20260720":
        raise RuntimeError("fresh outer feature split changed")
    if data["metadata"]["labels_accessed"] or data["metadata"]["labels_in_output"]:
        raise RuntimeError("fresh outer features unexpectedly contain labels")
    if len(data["record_key"]) != 100:
        raise RuntimeError("fresh outer feature state count changed")
    expected_records = np.asarray(
        [f"{row['trajectory_id']}:{int(row['current_t'])}" for row in rows]
    )
    if not np.array_equal(data["record_key"].astype(str), expected_records):
        raise RuntimeError("fresh outer feature/manifest identity mismatch")
    training = json.loads(args.training_result.read_text(encoding="utf-8"))
    if training.get("status") != "calibrated_six_arm_fixed_pool_cem_false":
        raise RuntimeError("training result is not frozen/calibrated")
    if training.get("spent_strict100_outcomes_used") or training.get(
        "internal_holdout_accessed"
    ):
        raise RuntimeError("training provenance violates fresh outer")
    if training.get("cem"):
        raise RuntimeError("CEM must remain disabled")

    args.output_dir.mkdir(parents=True)
    spacing = np.full(100, 0.38, dtype=np.float32)
    arrays: dict[str, np.ndarray] = {
        "record_key": data["record_key"].astype(str),
        "family": data["family"].astype(str),
    }
    audit: dict[str, Any] = {}
    ate_scale = float(training["scorer_v2"]["ate_scale"])
    rtp_scale = float(training["scorer_v2"]["rtp_scale"])
    for regime_name in ("scand_only", "domain_balanced"):
        regime = training["regimes"][regime_name]
        members = load_members(regime["training"], args.device)
        predictions = predict_modes(members, data, args.device)
        calibration = regime["calibration"]
        support = load_support(
            Path(calibration["support_checkpoint"]),
            calibration["support_checkpoint_sha256"],
        )
        regime_audit: dict[str, Any] = {}
        for arm_name in ("support_only", "raw_only", "support_plus_raw"):
            policy = calibration["arms"][arm_name]
            if arm_name == "raw_only":
                valid = data["majority_support"].astype(bool)
                support_info = {
                    "source": "frozen_scorer_v2_majority_support",
                    "candidate_acceptance": float(valid[:, 1:].mean()),
                    "no_supported_proposal_windows": int(
                        (~valid[:, 1:].any(axis=1)).sum()
                    ),
                }
            else:
                valid, support_info = support_mask(
                    support,
                    data["actions_h4"],
                    spacing,
                    quantile=float(policy["support_quantile"]),
                )
            if arm_name == "support_only":
                selected = native_selection(
                    data, valid, policy, ate_scale, rtp_scale
                )
                meta_score = np.zeros(100, dtype=np.float32)
            else:
                selected, meta_score = raw_selection(
                    predictions, data, valid, policy
                )
            label = f"{regime_name}__{arm_name}"
            arrays[label] = selected
            arrays[f"{label}__meta_score"] = meta_score
            regime_audit[arm_name] = {
                **selection_audit(selected, valid),
                "support": support_info,
            }
        audit[regime_name] = regime_audit
        del members, predictions
        if args.device == "cuda":
            torch.cuda.empty_cache()
    selected_path = args.output_dir / "selected_index.npz"
    write_npz(selected_path, **arrays)
    checkpoint_hashes = sorted(
        {
            record["sha256"]
            for regime in training["regimes"].values()
            for records in regime["training"]["checkpoints"].values()
            for record in records
        }
    )
    freeze = {
        "schema": "scand_fresh_outer_prelabel_freeze_v1",
        "status": "fresh_outer_selected_indices_frozen_before_pose_labels",
        "states": 100,
        "trajectories": 25,
        "route_groups": 25,
        "bags": 25,
        "manifest": str(args.manifest),
        "manifest_sha256": file_sha256(args.manifest),
        "manifest_audit_sha256": file_sha256(args.manifest_audit),
        "excluded_universe_sha256": file_sha256(args.excluded_universe),
        "candidate_pool": str(args.candidate),
        "candidate_pool_sha256": file_sha256(args.candidate),
        "feature_artifact": str(args.features),
        "feature_artifact_sha256": file_sha256(args.features),
        "training_result": str(args.training_result),
        "training_result_sha256": file_sha256(args.training_result),
        "model_checkpoint_sha256": checkpoint_hashes,
        "selected_index": str(selected_path),
        "selected_index_sha256": file_sha256(selected_path),
        "selection_audit": audit,
        "historically_spent": False,
        "trajectory_overlap_with_used_universe": [],
        "route_overlap_with_used_universe": [],
        "labels_opened": False,
        "future_pose_outcomes_opened": False,
        "spent_strict100_outcomes_used": False,
        "cem_cal_accessed": False,
        "fixed_pool_only": True,
        "cem": False,
    }
    write_json(args.output_dir / "prelabel_freeze.json", freeze)
    print(json.dumps(freeze, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
