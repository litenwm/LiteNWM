#!/usr/bin/env python3
"""Train a compact route-disjoint risk calibrator for base+CEM choices."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from tools.scoring.train_recon_cem_combined_pool_v7 import (  # noqa: E402
    evaluate_policy,
    load_combined_cohort,
)
from tools.scoring.train_recon_cem_state_gate_v6 import (  # noqa: E402
    ModelConfig,
    predict,
    sigmoid,
    train_model,
)
from tools.scoring.train_recon_dense_meta_dinov3_v1 import (  # noqa: E402
    file_sha256,
    write_json,
    write_npz,
)
from tools.scoring.tune_recon_cem_combined_frozen_universal_v8 import (  # noqa: E402
    combined_prediction,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v4-result", type=Path, required=True)
    parser.add_argument("--base-scores", type=Path, nargs=3, required=True)
    parser.add_argument("--preflight-base-scores", type=Path, required=True)
    parser.add_argument("--preflight-cem-scores", type=Path, required=True)
    parser.add_argument("--preflight-manifest", type=Path, required=True)
    parser.add_argument("--preflight-freeze", type=Path, required=True)
    parser.add_argument("--training-result", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=260)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--seed", type=int, default=20260814)
    return parser.parse_args()


def fold_for_group(group: str, folds: int) -> int:
    digest = hashlib.sha256(f"cem-compact-v14|{group}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % folds


def normalized_rank(prediction: dict[str, np.ndarray]) -> np.ndarray:
    rank = prediction["rank_logit"].astype(np.float64)
    valid = prediction["valid"].astype(bool)
    center = np.nanmedian(
        np.where(valid, rank, np.nan), axis=1, keepdims=True
    )
    scale = np.nanstd(
        np.where(valid, rank, np.nan), axis=1, keepdims=True
    )
    return ((rank - center) / np.maximum(scale, 1.0e-3)).astype(
        np.float32
    )


def choose_candidate(
    prediction: dict[str, np.ndarray],
    candidate_mask: np.ndarray,
) -> np.ndarray:
    valid = prediction["valid"].astype(bool) & candidate_mask
    valid[:, 0] = False
    rank_z = normalized_rank(prediction)
    is_cem = np.zeros((1, rank_z.shape[1]), dtype=np.float32)
    if rank_z.shape[1] == 33:
        is_cem[:, 17:] = 1.0
    score = (
        rank_z
        + prediction["dual_probability"]
        - prediction["harm_probability"]
        - is_cem
    )
    score = np.where(valid, score, -np.inf)
    if not valid.any(axis=1).all():
        raise RuntimeError("compact gate state has no valid candidate")
    return score.argmax(axis=1).astype(np.int64)


def masked_stats(value: np.ndarray, valid: np.ndarray) -> np.ndarray:
    masked = np.where(valid[..., None], value, np.nan)
    return np.concatenate(
        (
            np.nanmean(masked, axis=1),
            np.nanstd(masked, axis=1),
            np.nanmin(masked, axis=1),
            np.nanmax(masked, axis=1),
        ),
        axis=1,
    ).astype(np.float32)


def compact_feature(
    prediction: dict[str, np.ndarray],
    candidate_mask: np.ndarray,
    candidate: np.ndarray,
) -> np.ndarray:
    row = np.arange(len(candidate))
    rank_z = normalized_rank(prediction)
    is_cem = np.zeros((1, prediction["mean"].shape[1]), dtype=np.float32)
    if is_cem.shape[1] == 33:
        is_cem[:, 17:] = 1.0
    tensor = np.concatenate(
        (
            prediction["mean"],
            np.log1p(prediction["std"]),
            prediction["dual_probability"][..., None],
            prediction["harm_probability"][..., None],
            rank_z[..., None],
            is_cem.repeat(len(row), axis=0)[..., None],
        ),
        axis=-1,
    ).astype(np.float32)
    valid = prediction["valid"].astype(bool) & candidate_mask
    chosen = tensor[row, candidate]
    mean = prediction["mean"][row, candidate]
    std = prediction["std"][row, candidate]
    ucb = np.concatenate(
        tuple(mean + z * std for z in (0.0, 1.0, 2.0)),
        axis=1,
    ).astype(np.float32)
    distribution = masked_stats(tensor, valid)
    fractions = np.stack(
        (
            valid.mean(axis=1),
            (
                valid
                & (prediction["dual_probability"] >= 0.50)
            ).mean(axis=1),
            (
                valid
                & (prediction["dual_probability"] >= 0.80)
            ).mean(axis=1),
            (
                valid
                & (prediction["harm_probability"] <= 0.05)
            ).mean(axis=1),
            (
                valid
                & (prediction["harm_probability"] <= 0.10)
            ).mean(axis=1),
        ),
        axis=1,
    ).astype(np.float32)
    return np.nan_to_num(
        np.concatenate((chosen, ucb, distribution, fractions), axis=1),
        nan=0.0,
        posinf=10.0,
        neginf=-10.0,
    ).astype(np.float32)


def accepted(
    prediction: dict[str, np.ndarray],
    mean_std: np.ndarray,
    policy: dict[str, float],
) -> np.ndarray:
    p_dual = sigmoid(prediction["dual_logit"])
    p_harm = sigmoid(prediction["harm_logit"])
    mean = prediction["mean"]
    return (
        (p_dual >= float(policy["dual_threshold"]))
        & (p_harm <= float(policy["harm_threshold"]))
        & (
            mean[:, 0]
            + float(policy["uncertainty_scale"]) * mean_std[:, 0]
            < -float(policy["margin"])
        )
        & (
            mean[:, 3]
            + float(policy["uncertainty_scale"]) * mean_std[:, 3]
            < -float(policy["margin"])
        )
    )


def main() -> None:
    args = parse_args()
    result_path = args.output_dir / "result.json"
    if result_path.exists():
        raise FileExistsError(result_path)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    preflight_freeze = json.loads(
        args.preflight_freeze.read_text(encoding="utf-8")
    )
    if (
        preflight_freeze.get("outcomes_accessed")
        or preflight_freeze.get("labels_opened")
    ):
        raise RuntimeError("preflight labels were already opened")
    preflight_rows = [
        json.loads(line)
        for line in args.preflight_manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    preflight_record = np.asarray(
        [
            f"{row['trajectory_id']}:{int(row['current_t'])}"
            for row in preflight_rows
        ]
    )
    preflight_prediction = combined_prediction(
        args.preflight_base_scores,
        args.preflight_cem_scores,
        preflight_record,
    )

    v4 = json.loads(args.v4_result.read_text(encoding="utf-8"))
    cohorts = []
    predictions = []
    for value, base_score in zip(v4["inputs"], args.base_scores):
        cohort = load_combined_cohort(
            name=value["name"],
            final_feature_path=Path(value["features"]),
            final_score_path=Path(value["scores"]),
            result_path=Path(value["result"]),
        )
        predictions.append(
            combined_prediction(
                base_score,
                Path(value["scores"]),
                cohort["record_key"],
            )
        )
        cohorts.append(cohort)
    prediction = {
        key: np.concatenate([value[key] for value in predictions])
        for key in predictions[0]
    }
    cohort_name = np.concatenate(
        [
            np.full(len(value["route"]), value["name"], dtype="<U32")
            for value in cohorts
        ]
    )
    group = np.concatenate(
        [
            np.asarray([f"{value['name']}::{route}" for route in value["route"]])
            for value in cohorts
        ]
    )
    false_mask = np.zeros((300, 33), dtype=bool)
    false_mask[:, :17] = True
    true_mask = np.ones((300, 33), dtype=bool)
    false_candidate = choose_candidate(prediction, false_mask)
    true_candidate = choose_candidate(prediction, true_mask)
    false_feature = compact_feature(
        prediction, false_mask, false_candidate
    )
    true_feature = compact_feature(
        prediction, true_mask, true_candidate
    )
    feature = np.concatenate((false_feature, true_feature))
    option_candidate = np.concatenate(
        (false_candidate, true_candidate)
    )
    option_group = np.concatenate((group, group))
    option_cohort = np.concatenate((cohort_name, cohort_name))
    combined_delta = np.concatenate(
        [value["combined_delta"] for value in cohorts]
    )
    row = np.arange(300)
    false_target = combined_delta[row, false_candidate - 1]
    true_target = combined_delta[row, true_candidate - 1]
    target = np.concatenate((false_target, true_target))
    dual = (
        (target[:, 0] < -1.0e-7) & (target[:, 3] < -1.0e-7)
    ).astype(np.float32)
    harm = (
        (target[:, 0] > 1.0e-7) | (target[:, 3] > 1.0e-7)
    ).astype(np.float32)
    fold_by_state = np.asarray(
        [fold_for_group(value, args.folds) for value in group],
        dtype=np.int64,
    )
    fold_id = np.concatenate((fold_by_state, fold_by_state))
    config = ModelConfig(
        input_dim=feature.shape[1], hidden_dim=48, dropout=0.20
    )
    member_predictions = []
    for member in range(3):
        oof = {
            "mean": np.empty((600, 4), dtype=np.float32),
            "dual_logit": np.empty(600, dtype=np.float32),
            "harm_logit": np.empty(600, dtype=np.float32),
        }
        for fold in range(args.folds):
            train_index = np.flatnonzero(fold_id != fold)
            heldout_index = np.flatnonzero(fold_id == fold)
            model, stats = train_model(
                feature=feature,
                target=target,
                dual=dual,
                harm=harm,
                groups=option_group,
                train_index=train_index,
                config=config,
                epochs=args.epochs,
                batch_states=48,
                learning_rate=5.0e-4,
                weight_decay=1.0e-3,
                seed=args.seed + member * 1009 + fold,
                device=args.device,
            )
            values = predict(
                model, stats, feature[heldout_index], device=args.device
            )
            for key in oof:
                oof[key][heldout_index] = values[key]
            del model
            if args.device == "cuda":
                torch.cuda.empty_cache()
        member_predictions.append(oof)
    oof_prediction = {
        key: np.mean(
            np.stack([value[key] for value in member_predictions]), axis=0
        ).astype(np.float32)
        for key in member_predictions[0]
    }
    oof_std = np.std(
        np.stack([value["mean"] for value in member_predictions]), axis=0
    ).astype(np.float32)

    full_models = []
    full_index = np.arange(600)
    for member in range(3):
        model, stats = train_model(
            feature=feature,
            target=target,
            dual=dual,
            harm=harm,
            groups=option_group,
            train_index=full_index,
            config=config,
            epochs=args.epochs,
            batch_states=48,
            learning_rate=5.0e-4,
            weight_decay=1.0e-3,
            seed=args.seed + 10001 + member * 1009,
            device=args.device,
        )
        full_models.append((model, stats))

    preflight_false_mask = np.zeros((100, 33), dtype=bool)
    preflight_false_mask[:, :17] = True
    preflight_true_mask = np.ones((100, 33), dtype=bool)
    preflight_false_candidate = choose_candidate(
        preflight_prediction, preflight_false_mask
    )
    preflight_true_candidate = choose_candidate(
        preflight_prediction, preflight_true_mask
    )
    preflight_feature = np.concatenate(
        (
            compact_feature(
                preflight_prediction,
                preflight_false_mask,
                preflight_false_candidate,
            ),
            compact_feature(
                preflight_prediction,
                preflight_true_mask,
                preflight_true_candidate,
            ),
        )
    )
    preflight_members = [
        predict(model, stats, preflight_feature, device=args.device)
        for model, stats in full_models
    ]
    preflight_model_prediction = {
        key: np.mean(
            np.stack([value[key] for value in preflight_members]), axis=0
        ).astype(np.float32)
        for key in preflight_members[0]
    }
    preflight_std = np.std(
        np.stack([value["mean"] for value in preflight_members]), axis=0
    ).astype(np.float32)

    feasible = []
    searched = 0
    for uncertainty_scale in (0.0, 0.5, 1.0, 1.5, 2.0):
        for dual_threshold in np.linspace(0.30, 0.90, 13):
            for harm_threshold in (
                0.05,
                0.10,
                0.15,
                0.20,
                0.30,
                0.40,
                0.50,
            ):
                for margin in (0.0, 0.005, 0.010, 0.020):
                    searched += 1
                    policy = {
                        "uncertainty_scale": uncertainty_scale,
                        "dual_threshold": float(dual_threshold),
                        "harm_threshold": harm_threshold,
                        "margin": margin,
                    }
                    false_accept = accepted(
                        {
                            key: value[:300]
                            for key, value in oof_prediction.items()
                        },
                        oof_std[:300],
                        policy,
                    )
                    true_accept = accepted(
                        {
                            key: value[300:]
                            for key, value in oof_prediction.items()
                        },
                        oof_std[300:],
                        policy,
                    )
                    false_selected = np.where(
                        false_accept, false_candidate, 0
                    )
                    true_selected = np.where(
                        true_accept, true_candidate, 0
                    )
                    evaluation = evaluate_policy(
                        false_selected=false_selected,
                        true_selected=true_selected,
                        cohorts=cohorts,
                        cohort_name=cohort_name,
                    )
                    if not evaluation["strict_feasible"]:
                        continue
                    pf_false_accept = accepted(
                        {
                            key: value[:100]
                            for key, value in preflight_model_prediction.items()
                        },
                        preflight_std[:100],
                        policy,
                    )
                    pf_true_accept = accepted(
                        {
                            key: value[100:]
                            for key, value in preflight_model_prediction.items()
                        },
                        preflight_std[100:],
                        policy,
                    )
                    pf_false_selected = np.where(
                        pf_false_accept, preflight_false_candidate, 0
                    )
                    pf_true_selected = np.where(
                        pf_true_accept, preflight_true_candidate, 0
                    )
                    false_count = int((pf_false_selected != 0).sum())
                    true_count = int((pf_true_selected != 0).sum())
                    if (
                        true_count < 40
                        or true_count < false_count
                        or not (pf_true_selected > 16).any()
                    ):
                        continue
                    feasible.append(
                        {
                            "policy": policy,
                            "development": evaluation,
                            "preflight_false_promotions": false_count,
                            "preflight_true_promotions": true_count,
                            "preflight_cem_selected": int(
                                (pf_true_selected > 16).sum()
                            ),
                        }
                    )
    if not feasible:
        raise RuntimeError(
            "compact calibrator has no spent-safe preflight-covered policy"
        )
    best = max(
        feasible,
        key=lambda value: (
            min(
                item["cem_true"]["coverage"]
                for item in value["development"]["per_cohort"].values()
            ),
            value["preflight_true_promotions"],
            value["development"]["coverage_sum"],
        ),
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_records = []
    for member, (model, stats) in enumerate(full_models):
        checkpoint = args.output_dir / f"member_{member}.pt"
        torch.save(
            {
                "schema": "recon_cem_compact_gate_v14_member",
                "config": asdict(config),
                **stats,
                "model": {
                    key: value.detach().cpu()
                    for key, value in model.state_dict().items()
                },
            },
            checkpoint,
        )
        checkpoint_records.append(
            {"path": str(checkpoint), "sha256": file_sha256(checkpoint)}
        )
    oof_path = args.output_dir / "oof_predictions.npz"
    write_npz(
        oof_path,
        cohort=option_cohort,
        route_group=option_group,
        fold_id=fold_id,
        candidate_index=option_candidate,
        mean=oof_prediction["mean"],
        mean_std=oof_std,
        dual_logit=oof_prediction["dual_logit"],
        harm_logit=oof_prediction["harm_logit"],
    )
    report = {
        "schema": "recon_cem_compact_gate_v14",
        "status": (
            "route_disjoint_oof_and_label_blind_preflight_passed_frozen"
        ),
        "evidence_role": (
            "development_only; preflight covariates spent, outcomes sealed"
        ),
        "cem": True,
        "feature_dim": int(feature.shape[1]),
        "config": {**asdict(config), "folds": args.folds, "epochs": args.epochs},
        "searched_policies": searched,
        "feasible_policies": len(feasible),
        "policy": best["policy"],
        "development": best["development"],
        "preflight": {
            "manifest": str(args.preflight_manifest),
            "manifest_sha256": file_sha256(args.preflight_manifest),
            "freeze": str(args.preflight_freeze),
            "freeze_sha256": file_sha256(args.preflight_freeze),
            "cem_false_promotions": best[
                "preflight_false_promotions"
            ],
            "cem_true_promotions": best[
                "preflight_true_promotions"
            ],
            "cem_selected": best["preflight_cem_selected"],
            "outcomes_accessed": False,
            "labels_opened": False,
            "status": "label_blind_preflight_spent_do_not_open",
        },
        "checkpoints": checkpoint_records,
        "oof_predictions": str(oof_path),
        "oof_predictions_sha256": file_sha256(oof_path),
        "candidate_rule": (
            "rank_z + p_dual - p_harm - 1.0*CEM_token"
        ),
        "online_causal": True,
        "future_route_windows_used": False,
        "source_v4_result": str(args.v4_result),
        "source_v4_result_sha256": file_sha256(args.v4_result),
        "training_result": str(args.training_result),
        "training_result_sha256": file_sha256(args.training_result),
        "next_confirmation_accessed": False,
        "formal_claim_allowed": False,
    }
    write_json(result_path, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
