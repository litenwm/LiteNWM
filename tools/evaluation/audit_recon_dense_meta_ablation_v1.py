#!/usr/bin/env python3
"""Audit frozen action/static/latent/full/meta ablations without retuning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from scoring.dense_meta_v1 import MODES, selection_metrics  # noqa: E402
from scoring.safe_promotion_v1 import greedy_fixed_pool_selection  # noqa: E402
from tools.evaluation.evaluate_recon100_dense_meta_three_arm_v1 import (  # noqa: E402
    arm_from_policy, load_policy, predictions,
)
from tools.scoring.train_recon_dense_meta_dinov3_three_arm_v1 import (  # noqa: E402
    load_features, metrics_json, predict_branch, rows_without_labels,
)
from tools.scoring.train_recon_dense_meta_dinov3_v1 import (  # noqa: E402
    file_sha256, policy_selections, write_json,
)


DEFAULT_ROOT = REPO_ROOT / "outputs/recon_dense_meta_dinov3_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--policy", type=Path, default=DEFAULT_ROOT / "three_arm_training/frozen_policy.json")
    parser.add_argument("--training-dir", type=Path, default=DEFAULT_ROOT / "three_arm_training")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


def load_label_targets(path: Path, data: dict[str, object]) -> np.ndarray:
    with np.load(path, allow_pickle=False) as payload:
        if not np.array_equal(payload["record_key"].astype(str), data["record_key"].astype(str)):
            raise RuntimeError("label record order mismatch")
        if "family" in payload.files and not np.array_equal(payload["family"].astype(str), data["family"].astype(str)):
            raise RuntimeError("label family order mismatch")
        return payload["targets"].astype(np.float32)


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    policy = load_policy(args.policy)
    data = load_features(args.features, labels_expected=False)
    targets = load_label_targets(args.labels, data)
    raw_prediction, stacked_prediction = predictions(data, args.training_dir, args.device)
    arms = (
        (arm_from_policy(policy, "dinov3_full_meta_raw", "raw"), raw_prediction),
        (arm_from_policy(policy, "dinov3_full_meta_v2_stacked", "v2_stacked"), stacked_prediction),
    )
    metrics = {}
    for arm, prediction in arms:
        rows = rows_without_labels(prediction, data["majority_support"], arm["policies"])
        selections = policy_selections(rows, arm["policies"])
        meta_selected, _, _ = predict_branch(branch=arm, predictions=prediction, data=data)
        metrics[arm["name"]] = {
            **{mode: metrics_json(selection_metrics(selections[mode], targets)) for mode in MODES},
            "meta": metrics_json(selection_metrics(meta_selected, targets)),
        }
    native = policy["arms"]["dinov3_scorer_v2_native"]
    native_selected = greedy_fixed_pool_selection(
        data["ate_by_seed"], data["rtp_by_seed"],
        ate_scale=float(native["ate_scale"]), rtp_scale=float(native["rtp_scale"]),
        valid=data["majority_support"])
    metrics["dinov3_scorer_v2_native"] = metrics_json(selection_metrics(native_selected, targets))
    result = {
        "schema": "recon_dense_meta_frozen_ablation_v1",
        "split": data["metadata"]["split"],
        "status": "postfreeze_no_retuning_ablation",
        "metrics": metrics,
        "features": str(args.features), "features_sha256": file_sha256(args.features),
        "labels": str(args.labels), "labels_sha256": file_sha256(args.labels),
        "policy": str(args.policy), "policy_sha256": file_sha256(args.policy),
        "thresholds_or_models_refit": False,
        "support_is_feature": False,
        "cem": False,
    }
    write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
