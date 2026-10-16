#!/usr/bin/env python3
"""Train the matched-data raw-clean control for the frozen stacked-clean arm."""

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

from scoring.dense_meta_v1 import (  # noqa: E402
    DenseHeadConfig,
    empty_prediction,
    family_fold,
    feature_by_mode,
    predict,
)
from tools.evaluation.evaluate_recon100_dense_meta_three_arm_v1 import (  # noqa: E402
    load_members,
)
from tools.scoring.train_recon_dense_meta_dinov3_three_arm_v1 import (  # noqa: E402
    build_targets,
    fit_branch,
    load_features,
    scorer_overlap_audit,
    train_head,
)
from tools.scoring.train_recon_dense_meta_dinov3_v1 import (  # noqa: E402
    file_sha256,
    serializable_ridge,
    store_prediction,
    write_json,
)


DEFAULT_ROOT = REPO_ROOT / "outputs/recon_dense_meta_dinov3_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-features",
        type=Path,
        default=DEFAULT_ROOT / "features/dense_train_features.npz",
    )
    parser.add_argument(
        "--scorer-train-manifest",
        type=Path,
        default=REPO_ROOT / "data/inner_manifests/recon/train-fit.windows.jsonl",
    )
    parser.add_argument(
        "--shared-training-dir",
        type=Path,
        default=DEFAULT_ROOT / "three_arm_training",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--hidden", type=int, default=192)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--batch-groups", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=4.0e-4)
    parser.add_argument("--weight-decay", type=float, default=3.0e-3)
    parser.add_argument("--target-temperature", type=float, default=0.05)
    parser.add_argument("--min-promotions", type=int, default=64)
    parser.add_argument("--target-dual-precision", type=float, default=0.85)
    parser.add_argument("--target-wilson-lower", type=float, default=0.80)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


def shared_oof(
    *,
    label: str,
    mode: str,
    train: dict,
    fold_all: np.ndarray,
    training_dir: Path,
    device: str,
) -> dict[str, np.ndarray]:
    feature = feature_by_mode(train, mode, variant="raw")
    result = empty_prediction(len(feature), feature.shape[1])
    members = load_members(training_dir, label, device)
    for heldout, (model, stats) in enumerate(members):
        index = np.flatnonzero(fold_all == heldout)
        store_prediction(
            result,
            index,
            predict(model, stats, feature, index, device=device),
        )
    return result


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    config = DenseHeadConfig(
        hidden=args.hidden,
        dropout=args.dropout,
        epochs=args.epochs,
        batch_groups=args.batch_groups,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        target_temperature=args.target_temperature,
    )
    config.validate()
    train = load_features(args.train_features, labels_expected=True)
    clean, overlap = scorer_overlap_audit(train, args.scorer_train_manifest)
    universe = np.flatnonzero(clean)
    families = train["family"].astype(str)
    fold_all = np.asarray(
        [family_fold(value, args.folds) for value in families], dtype=np.int64
    )
    target = build_targets(train["targets"], config.target_temperature)
    args.output_dir.mkdir(parents=True)
    shared_action = shared_oof(
        label="shared_action",
        mode="action",
        train=train,
        fold_all=fold_all,
        training_dir=args.shared_training_dir,
        device=args.device,
    )
    shared_static = shared_oof(
        label="shared_static",
        mode="static",
        train=train,
        fold_all=fold_all,
        training_dir=args.shared_training_dir,
        device=args.device,
    )
    predictions = {
        "action": {key: value[universe] for key, value in shared_action.items()},
        "static": {key: value[universe] for key, value in shared_static.items()},
    }
    checkpoints = {}
    for label, mode, offset in (
        ("raw_clean_latent", "latent", 4000),
        ("raw_clean_full", "full", 5000),
    ):
        oof, _, checkpoints[label] = train_head(
            label=label,
            mode=mode,
            variant="raw",
            universe=universe,
            train=train,
            target=target,
            fold_all=fold_all,
            config=config,
            args=args,
            seed_offset=offset,
            train_sha=file_sha256(args.train_features),
        )
        predictions[mode] = oof
    branch = fit_branch(
        name="dinov3_full_meta_raw_clean",
        variant="raw",
        universe=universe,
        predictions=predictions,
        train=train,
        fold_all=fold_all,
        args=args,
    )
    stacked_policy_path = args.shared_training_dir / "frozen_policy.json"
    stacked_policy = json.loads(stacked_policy_path.read_text())
    stacked_states = int(
        stacked_policy["arms"]["dinov3_full_meta_v2_stacked"]["training_states"]
    )
    if stacked_states != len(universe):
        raise RuntimeError("raw-clean/stacked-clean training-state count mismatch")
    output = {
        "schema": "recon_dense_meta_dinov3_raw_clean_frozen_v1",
        "status": "frozen_recon_trained_before_scand_outcomes",
        "arm": {
            "name": branch["name"],
            "variant": "raw",
            "feature_contract": "action + DINOv3 static + raw LTWM only",
            "scorer_v2_predictions_in_features": False,
            "training_states": len(universe),
            "training_families": int(len(np.unique(families[universe]))),
            "policies": branch["policies"],
            "meta": serializable_ridge(branch["meta_model"]),
            "train_oof": branch["meta_metrics"],
        },
        "matched_stacked_arm": {
            "name": "dinov3_full_meta_v2_stacked",
            "training_states": stacked_states,
            "same_state_universe": True,
            "same_shared_action_static_oof": True,
            "latent_full_seed_offsets_matched": True,
        },
        "leakage_audit": overlap,
        "train_features": str(args.train_features.resolve()),
        "train_features_sha256": file_sha256(args.train_features),
        "shared_training_dir": str(args.shared_training_dir.resolve()),
        "stacked_policy_sha256": file_sha256(stacked_policy_path),
        "checkpoints": checkpoints,
        "scand_train_or_cal_used": False,
        "scand_outcomes_opened": False,
        "support_used_only_as_candidate_validity_mask": True,
        "fixed_pool_only": True,
        "cem": False,
    }
    write_json(args.output_dir / "frozen_policy.json", output)
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
