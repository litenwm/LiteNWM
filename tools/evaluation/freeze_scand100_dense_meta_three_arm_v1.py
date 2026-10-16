#!/usr/bin/env python3
"""Freeze strict SCAND100 K17 selections before opening any outcome artifact."""

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

from scoring.dense_meta_v1 import ensemble_predict, feature_by_mode  # noqa: E402
from scoring.safe_promotion_v1 import greedy_fixed_pool_selection  # noqa: E402
from tools.evaluation.evaluate_recon100_dense_meta_three_arm_v1 import (  # noqa: E402
    arm_from_policy,
    deserialize_meta,
    load_members,
    load_policy,
    predictions,
)
from tools.scoring.train_recon_dense_meta_dinov3_three_arm_v1 import (  # noqa: E402
    load_features,
    predict_branch,
)
from tools.scoring.train_recon_dense_meta_dinov3_v1 import (  # noqa: E402
    file_sha256,
    write_json,
    write_npz,
)


DEFAULT_OUTPUT = REPO_ROOT / "outputs/scand_strict_h4_100_three_arm_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features",
        type=Path,
        default=DEFAULT_OUTPUT
        / "features/dinov3_ltwm/scand_strict_h4_100_seed17_features.npz",
    )
    parser.add_argument(
        "--candidate",
        type=Path,
        default=DEFAULT_OUTPUT / "candidates/frozen_k17.npz",
    )
    parser.add_argument(
        "--candidate-audit",
        type=Path,
        default=DEFAULT_OUTPUT / "candidates/frozen_k17.audit.json",
    )
    parser.add_argument(
        "--preflight-audit",
        type=Path,
        default=DEFAULT_OUTPUT / "candidates/preflight_audit.json",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPO_ROOT
        / "data/evaluation/scand/strict_test_h4_100_seed17/manifest.json",
    )
    parser.add_argument(
        "--policy",
        type=Path,
        default=REPO_ROOT
        / "outputs/recon_dense_meta_dinov3_v1/three_arm_training/frozen_policy.json",
    )
    parser.add_argument(
        "--training-dir",
        type=Path,
        default=REPO_ROOT / "outputs/recon_dense_meta_dinov3_v1/three_arm_training",
    )
    parser.add_argument(
        "--raw-clean-policy",
        type=Path,
        default=REPO_ROOT
        / "outputs/recon_dense_meta_dinov3_v1/raw_clean_matched_training/frozen_policy.json",
    )
    parser.add_argument(
        "--raw-clean-training-dir",
        type=Path,
        default=REPO_ROOT
        / "outputs/recon_dense_meta_dinov3_v1/raw_clean_matched_training",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT / "frozen_selection")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


def load_raw_clean_policy(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != "recon_dense_meta_dinov3_raw_clean_frozen_v1":
        raise RuntimeError("raw-clean frozen policy schema mismatch")
    if value.get("status") != "frozen_recon_trained_before_scand_outcomes":
        raise RuntimeError("raw-clean policy was not frozen before SCAND outcomes")
    if value.get("scand_outcomes_opened") is not False or value.get("cem") is not False:
        raise RuntimeError("raw-clean outcome/CEM contract changed")
    arm = value["arm"]
    if arm.get("training_states") != value["matched_stacked_arm"].get("training_states"):
        raise RuntimeError("raw-clean and stacked training universes differ")
    return value, {
        "name": str(arm["name"]),
        "variant": "raw",
        "policies": arm["policies"],
        "meta_model": deserialize_meta(arm["meta"]),
    }


def raw_clean_predictions(
    data: dict[str, Any], shared_dir: Path, clean_dir: Path, device: str
) -> dict[str, Any]:
    action = ensemble_predict(
        load_members(shared_dir, "shared_action", device),
        feature_by_mode(data, "action", variant="raw"),
        device=device,
    )
    static = ensemble_predict(
        load_members(shared_dir, "shared_static", device),
        feature_by_mode(data, "static", variant="raw"),
        device=device,
    )
    latent = ensemble_predict(
        load_members(clean_dir, "raw_clean_latent", device),
        feature_by_mode(data, "latent", variant="raw"),
        device=device,
    )
    full = ensemble_predict(
        load_members(clean_dir, "raw_clean_full", device),
        feature_by_mode(data, "full", variant="raw"),
        device=device,
    )
    return {"action": action, "static": static, "latent": latent, "full": full}


def checkpoint_hashes(policy: dict[str, Any], raw_clean: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for label, rows in policy["checkpoints"].items():
        result[label] = [dict(row) for row in rows]
        for row in rows:
            if file_sha256(Path(row["path"])) != row["sha256"]:
                raise RuntimeError(f"checkpoint hash mismatch: {row['path']}")
    for label, rows in raw_clean["checkpoints"].items():
        result[label] = [dict(row) for row in rows]
        for row in rows:
            if file_sha256(Path(row["path"])) != row["sha256"]:
                raise RuntimeError(f"checkpoint hash mismatch: {row['path']}")
    return result


def main() -> None:
    args = parse_args()
    freeze_path = args.output_dir / "pre_label_freeze.json"
    if freeze_path.exists():
        raise FileExistsError(freeze_path)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    preflight = json.loads(args.preflight_audit.read_text(encoding="utf-8"))
    candidate_audit = json.loads(args.candidate_audit.read_text(encoding="utf-8"))
    if not preflight.get("all_overlap_checks_zero"):
        raise RuntimeError("preflight overlap audit did not pass")
    forbidden_true = (
        "navigation_eval_deserialized",
        "strict_test_outcomes_opened",
        "pose_outcomes_opened",
        "labels_accessed",
    )
    for document in (preflight, candidate_audit):
        for key in forbidden_true:
            if document.get(key) is True:
                raise RuntimeError(f"pre-label input reports forbidden outcome access: {key}")
        if document.get("cem") is not False or document.get("fixed_pool_only") is not True:
            raise RuntimeError("candidate/preflight fixed-pool contract changed")

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    records = list(manifest["records"])
    expected_key = np.asarray(
        [f"{row['trajectory_id']}:{int(row['curr_time'])}" for row in records]
    )
    eval_index = np.asarray([int(row["eval_index"]) for row in records], dtype=np.int64)
    route = np.asarray([str(row["route_group"]) for row in records])

    data = load_features(args.features, labels_expected=False)
    if data["metadata"].get("split") != "scand_strict_h4_100_seed17":
        raise RuntimeError("strict SCAND100 feature split name changed")
    if data["metadata"].get("cem") is not False:
        raise RuntimeError("feature artifact unexpectedly enables CEM")
    if not np.array_equal(data["record_key"].astype(str), expected_key):
        raise RuntimeError("feature/manifest record order mismatch")
    if not np.array_equal(data["family"].astype(str), np.asarray([row["trajectory_id"] for row in records])):
        raise RuntimeError("feature/manifest trajectory order mismatch")

    with np.load(args.candidate, allow_pickle=False) as frozen:
        candidate_units = frozen["candidates_waypoint_units"].astype(np.float32)
        candidate_actions = frozen["actions_m"].astype(np.float32)
        candidate_eval_index = frozen["eval_index"].astype(np.int64)
    if not np.array_equal(candidate_eval_index, eval_index):
        raise RuntimeError("candidate/manifest eval-index mismatch")
    metric_candidates = data["candidates"].astype(np.float32) * np.float32(0.25)
    if not np.allclose(metric_candidates, candidate_units * np.float32(0.38), atol=2e-7, rtol=2e-7):
        raise RuntimeError("SCAND 0.38m candidate unit bridge mismatch")
    if not np.allclose(data["actions_h4"], candidate_actions, atol=2e-7, rtol=2e-7):
        raise RuntimeError("feature/candidate action identity mismatch")

    policy = load_policy(args.policy)
    raw_clean_policy, raw_clean_arm = load_raw_clean_policy(args.raw_clean_policy)
    hashes = checkpoint_hashes(policy, raw_clean_policy)
    raw_arm = arm_from_policy(policy, "dinov3_full_meta_raw", "raw")
    stacked_arm = arm_from_policy(
        policy, "dinov3_full_meta_v2_stacked", "v2_stacked"
    )
    raw_prediction, stacked_prediction = predictions(data, args.training_dir, args.device)
    clean_prediction = raw_clean_predictions(
        data, args.training_dir, args.raw_clean_training_dir, args.device
    )
    raw_selected, _, raw_score = predict_branch(
        branch=raw_arm, predictions=raw_prediction, data=data
    )
    stacked_selected, _, stacked_score = predict_branch(
        branch=stacked_arm, predictions=stacked_prediction, data=data
    )
    clean_selected, _, clean_score = predict_branch(
        branch=raw_clean_arm, predictions=clean_prediction, data=data
    )
    native = policy["arms"]["dinov3_scorer_v2_native"]
    native_selected = greedy_fixed_pool_selection(
        data["ate_by_seed"],
        data["rtp_by_seed"],
        ate_scale=float(native["ate_scale"]),
        rtp_scale=float(native["rtp_scale"]),
        valid=data["majority_support"],
    )
    selections = {
        "dinov3_full_meta_raw": raw_selected,
        "dinov3_full_meta_v2_stacked": stacked_selected,
        "dinov3_scorer_v2_native": native_selected,
        "dinov3_full_meta_raw_clean": clean_selected,
    }
    for name, selected in selections.items():
        if selected.shape != (100,) or np.any((selected < 0) | (selected >= 17)):
            raise RuntimeError(f"invalid selection array: {name}")
        if not np.all(data["majority_support"][np.arange(100), selected]):
            raise RuntimeError(f"selection bypassed frozen support mask: {name}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    inference = args.output_dir / "frozen_inference.npz"
    write_npz(
        inference,
        record_key=expected_key,
        trajectory=data["family"].astype(str),
        route=route,
        eval_index=eval_index,
        **{f"{name}_selected_index": value for name, value in selections.items()},
        raw_meta_score=raw_score,
        stacked_meta_score=stacked_score,
        raw_clean_meta_score=clean_score,
    )
    audit = {
        "schema": "scand100_k17_dense_meta_three_arm_pre_label_v1",
        "status": "all_selections_frozen_before_scand_strict_test_outcomes_opened",
        "states": 100,
        "trajectories": int(len(np.unique(data["family"].astype(str)))),
        "routes": int(len(np.unique(route))),
        "candidates": 17,
        "waypoint_spacing_m": 0.38,
        "candidate_pool": "trimmed_mean75_incumbent_plus_same_frozen_nomad_x16",
        "candidate": str(args.candidate.resolve()),
        "candidate_sha256": file_sha256(args.candidate),
        "candidate_audit": str(args.candidate_audit.resolve()),
        "candidate_audit_sha256": file_sha256(args.candidate_audit),
        "preflight_audit": str(args.preflight_audit.resolve()),
        "preflight_audit_sha256": file_sha256(args.preflight_audit),
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": file_sha256(args.manifest),
        "features": str(args.features.resolve()),
        "features_sha256": file_sha256(args.features),
        "policy": str(args.policy.resolve()),
        "policy_sha256": file_sha256(args.policy),
        "raw_clean_policy": str(args.raw_clean_policy.resolve()),
        "raw_clean_policy_sha256": file_sha256(args.raw_clean_policy),
        "checkpoint_hashes": hashes,
        "inference": str(inference.resolve()),
        "inference_sha256": file_sha256(inference),
        "selection_promotions_without_labels": {
            name: int(np.count_nonzero(value)) for name, value in selections.items()
        },
        "support_audit": {
            "candidate_acceptance_fraction": float(data["support_by_seed"][:, :, 1:].mean()),
            "majority_acceptance_fraction": float(data["majority_support"][:, 1:].mean()),
            "states_with_no_supported_nomad_candidate": int(
                np.count_nonzero(data["majority_support"][:, 1:].sum(axis=1) == 0)
            ),
            "role": "frozen deterministic candidate validity mask only",
        },
        "feature_contracts": {
            "dinov3_full_meta_raw": "action + DINOv3 static + raw LTWM; no scorer-v2 prediction feature",
            "dinov3_full_meta_v2_stacked": "raw plus frozen scorer-v2 ATE/RTP/uncertainty",
            "dinov3_scorer_v2_native": "frozen scorer-v2 greedy fixed-pool baseline",
            "dinov3_full_meta_raw_clean": "matched 824-state raw-only control",
        },
        "matched_data_audit": {
            "raw_clean_training_states": raw_clean_policy["arm"]["training_states"],
            "stacked_training_states": policy["arms"]["dinov3_full_meta_v2_stacked"]["training_states"],
            "same_state_universe": raw_clean_policy["matched_stacked_arm"]["same_state_universe"],
            "same_shared_action_static_oof": raw_clean_policy["matched_stacked_arm"]["same_shared_action_static_oof"],
            "latent_full_seed_offsets_matched": raw_clean_policy["matched_stacked_arm"]["latent_full_seed_offsets_matched"],
        },
        "navigation_eval_deserialized": False,
        "pose_outcomes_opened": False,
        "labels_opened": False,
        "scand_calibration_performed": False,
        "fixed_pool_only": True,
        "cem": False,
    }
    write_json(freeze_path, audit)
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
