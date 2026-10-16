#!/usr/bin/env python3
"""Apply the frozen dense-meta three-arm policy to RECON100 K17."""

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

from scoring.dense_meta_v1 import DenseHeadConfig, GroupFirstNet, ensemble_predict, feature_by_mode, selection_metrics  # noqa: E402
from scoring.safe_promotion_v1 import greedy_fixed_pool_selection  # noqa: E402
from tools.scoring.train_recon_dense_meta_dinov3_three_arm_v1 import (  # noqa: E402
    load_features, metrics_json, predict_branch, scorer_overlap_audit,
)
from tools.scoring.train_recon_dense_meta_dinov3_v1 import file_sha256, write_json, write_npz  # noqa: E402


DEFAULT_ROOT = REPO_ROOT / "outputs/recon_dense_meta_dinov3_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("freeze", "evaluate"), required=True)
    parser.add_argument("--features", type=Path, default=DEFAULT_ROOT / "features/strict_external100_k17_features.npz")
    parser.add_argument("--labels", type=Path, default=DEFAULT_ROOT / "recon100_k17/labels.npz")
    parser.add_argument("--policy", type=Path, default=DEFAULT_ROOT / "three_arm_training/frozen_policy.json")
    parser.add_argument("--training-dir", type=Path, default=DEFAULT_ROOT / "three_arm_training")
    parser.add_argument("--scorer-train-manifest", type=Path, default=REPO_ROOT / "data/inner_manifests/recon/train-fit.windows.jsonl")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ROOT / "recon100_k17")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


def deserialize_meta(value: dict[str, Any]) -> dict[str, Any]:
    return {
        "feature_mean": np.asarray(value["feature_mean"], dtype=np.float32),
        "feature_scale": np.asarray(value["feature_scale"], dtype=np.float32),
        "coefficient": np.asarray(value["coefficient"], dtype=np.float32),
        "intercept": float(value["intercept"]),
        "regularization": float(value["regularization"]),
        "threshold": float(value["threshold"]),
    }


def load_members(training_dir: Path, label: str, device: str) -> list[tuple[torch.nn.Module, dict[str, np.ndarray]]]:
    members = []
    for fold in range(5):
        path = training_dir / "checkpoints" / f"{label}_fold{fold}.pt"
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("schema") != "recon_dense_meta_dinov3_groupfirst_member_v2":
            raise RuntimeError(f"checkpoint schema mismatch: {path}")
        config = DenseHeadConfig(**payload["config"])
        model = GroupFirstNet(int(payload["input_dim"]), config.hidden, config.dropout)
        model.load_state_dict(payload["state_dict"], strict=True)
        model.eval().to(device).requires_grad_(False)
        stats = {key: np.asarray(value, dtype=np.float32) for key, value in payload["normalization"].items()}
        members.append((model, stats))
    return members


def load_policy(path: Path) -> dict[str, Any]:
    policy = json.loads(path.read_text(encoding="utf-8"))
    if policy.get("schema") != "recon_dense_meta_dinov3_three_arm_frozen_v1":
        raise RuntimeError("three-arm frozen policy schema mismatch")
    if not policy.get("primary_raw_test3_authorized"):
        raise RuntimeError("primary raw branch was not authorized on dense-cal")
    if policy.get("cem") is not False:
        raise RuntimeError("policy unexpectedly enables CEM")
    return policy


def arm_from_policy(policy: dict[str, Any], name: str, variant: str) -> dict[str, Any]:
    value = policy["arms"][name]
    return {"name": name, "variant": variant, "policies": value["policies"],
            "meta_model": deserialize_meta(value["meta"])}


def predictions(data: dict[str, Any], training_dir: Path, device: str) -> tuple[dict[str, Any], dict[str, Any]]:
    members = {label: load_members(training_dir, label, device) for label in (
        "shared_action", "shared_static", "raw_latent", "raw_full",
        "v2_stacked_latent", "v2_stacked_full")}
    shared_action = ensemble_predict(members["shared_action"], feature_by_mode(data, "action", variant="raw"), device=device)
    shared_static = ensemble_predict(members["shared_static"], feature_by_mode(data, "static", variant="raw"), device=device)
    raw = {"action": shared_action, "static": shared_static,
           "latent": ensemble_predict(members["raw_latent"], feature_by_mode(data, "latent", variant="raw"), device=device),
           "full": ensemble_predict(members["raw_full"], feature_by_mode(data, "full", variant="raw"), device=device)}
    stacked = {"action": shared_action, "static": shared_static,
               "latent": ensemble_predict(members["v2_stacked_latent"], feature_by_mode(data, "latent", variant="v2_stacked"), device=device),
               "full": ensemble_predict(members["v2_stacked_full"], feature_by_mode(data, "full", variant="v2_stacked"), device=device)}
    return raw, stacked


def freeze(args: argparse.Namespace) -> None:
    if args.output_dir.exists() and (args.output_dir / "pre_label_freeze.json").exists():
        raise FileExistsError(args.output_dir / "pre_label_freeze.json")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    policy = load_policy(args.policy)
    data = load_features(args.features, labels_expected=False)
    if not data["metadata"]["split"].startswith("strict_external100_k17"):
        raise RuntimeError("RECON100 feature split name changed")
    _, overlap = scorer_overlap_audit(data, args.scorer_train_manifest)
    if overlap["trajectory_overlap_states"]:
        raise RuntimeError("RECON100 overlaps scorer-v2 train-fit trajectories")
    raw_arm = arm_from_policy(policy, "dinov3_full_meta_raw", "raw")
    stacked_arm = arm_from_policy(policy, "dinov3_full_meta_v2_stacked", "v2_stacked")
    raw_prediction, stacked_prediction = predictions(data, args.training_dir, args.device)
    raw_selected, _, raw_score = predict_branch(branch=raw_arm, predictions=raw_prediction, data=data)
    stacked_selected, _, stacked_score = predict_branch(branch=stacked_arm, predictions=stacked_prediction, data=data)
    native = policy["arms"]["dinov3_scorer_v2_native"]
    native_selected = greedy_fixed_pool_selection(data["ate_by_seed"], data["rtp_by_seed"],
                                                   ate_scale=float(native["ate_scale"]),
                                                   rtp_scale=float(native["rtp_scale"]),
                                                   valid=data["majority_support"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    inference = args.output_dir / "frozen_inference.npz"
    write_npz(inference, record_key=data["record_key"].astype(str), family=data["family"].astype(str),
              dinov3_full_meta_raw_selected_index=raw_selected,
              dinov3_full_meta_v2_stacked_selected_index=stacked_selected,
              dinov3_scorer_v2_native_selected_index=native_selected,
              raw_meta_score=raw_score, stacked_meta_score=stacked_score)
    audit = {
        "schema": "recon100_k17_dense_meta_three_arm_pre_label_v1",
        "status": "all_three_selections_frozen_before_recon100_labels_open",
        "states": int(len(data["family"])), "candidates": 17,
        "candidate_pool": "trimmed_mean75_incumbent_plus_same_frozen_nomad_x16",
        "policy": str(args.policy), "policy_sha256": file_sha256(args.policy),
        "features": str(args.features), "features_sha256": file_sha256(args.features),
        "inference": str(inference), "inference_sha256": file_sha256(inference),
        "scorer_v2_overlap_audit": overlap,
        "feature_contracts": {
            "dinov3_full_meta_raw": "no scorer-v2 ATE/RTP prediction feature",
            "dinov3_full_meta_v2_stacked": "raw plus frozen scorer-v2 ATE/RTP/uncertainty",
            "dinov3_scorer_v2_native": "frozen scorer-v2 greedy fixed-pool baseline",
        },
        "support_role": "deterministic candidate validity mask only; not a learned feature",
        "labels_opened": False, "fixed_pool_only": True, "cem": False,
    }
    write_json(args.output_dir / "pre_label_freeze.json", audit)
    print(json.dumps(audit, indent=2, sort_keys=True))


def evaluate(args: argparse.Namespace) -> None:
    freeze_path = args.output_dir / "pre_label_freeze.json"
    audit = json.loads(freeze_path.read_text(encoding="utf-8"))
    if audit.get("status") != "all_three_selections_frozen_before_recon100_labels_open":
        raise RuntimeError("RECON100 selections were not frozen before labels")
    inference_path = Path(audit["inference"])
    if file_sha256(inference_path) != audit["inference_sha256"]:
        raise RuntimeError("frozen inference artifact changed")
    with np.load(inference_path, allow_pickle=False) as inference:
        record_key = inference["record_key"].astype(str)
        selections = {name: inference[f"{name}_selected_index"].astype(np.int64) for name in (
            "dinov3_full_meta_raw", "dinov3_full_meta_v2_stacked", "dinov3_scorer_v2_native")}
    with np.load(args.labels, allow_pickle=False) as labels:
        if not np.array_equal(record_key, labels["record_key"].astype(str)):
            raise RuntimeError("RECON100 label order mismatch")
        targets = labels["targets"].astype(np.float32)
    metrics = {name: metrics_json(selection_metrics(selected, targets)) for name, selected in selections.items()}
    result = {**audit,
              "schema": "recon100_k17_dense_meta_three_arm_result_v1",
              "status": "completed_frozen_recon100_k17_three_arm",
              "labels": str(args.labels), "labels_sha256": file_sha256(args.labels),
              "labels_opened": True, "metrics": metrics,
              "evidence_note": "RECON100 outcomes were already spent by prior studies; this is frozen-policy developmental replay, not fresh deployment evidence",
              "naming_note": "raw is independently retrained; v2-stacked is a scorer-v2-dependent ablation",
              "deployment_accepted": False, "cem": False}
    write_json(args.output_dir / "result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


def main() -> None:
    args = parse_args()
    if args.stage == "freeze":
        freeze(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
