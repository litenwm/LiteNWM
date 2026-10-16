#!/usr/bin/env python3
"""Train a route-disjoint selector over base K17 plus final CEM proposals."""

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

from tools.scoring.train_recon_cem_candidate_calibrator_v4 import (  # noqa: E402
    ModelConfig,
    file_sha256,
    load_cohort,
    predict,
    sigmoid,
    train_model,
    write_json,
    write_npz,
)
from tools.scoring.train_recon_cem_state_gate_v6 import (  # noqa: E402
    METRIC_NAMES,
    method_summary,
    optional,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v4-result", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=220)
    parser.add_argument("--batch-states", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=5.0e-4)
    parser.add_argument("--weight-decay", type=float, default=5.0e-4)
    parser.add_argument("--ensemble-members", type=int, default=3)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--seed", type=int, default=20260807)
    return parser.parse_args()


def fold_for_group(group: str, folds: int) -> int:
    digest = hashlib.sha256(f"cem-combined-v7|{group}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % folds


def resolve_base_feature(result_path: Path) -> Path:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    freeze_path = Path(
        result.get("selection_freeze") or result.get("freeze")
    )
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if "cem_freeze" in freeze:
        freeze = json.loads(
            Path(freeze["cem_freeze"]).read_text(encoding="utf-8")
        )
    return Path(freeze["base_features"])


def load_combined_cohort(
    *,
    name: str,
    final_feature_path: Path,
    final_score_path: Path,
    result_path: Path,
) -> dict[str, Any]:
    cohort = load_cohort(
        name=name,
        feature_path=final_feature_path,
        score_path=final_score_path,
        result_path=result_path,
    )
    base_feature_path = resolve_base_feature(result_path)
    with np.load(base_feature_path, allow_pickle=False) as base:
        base_record_key = base["record_key"].astype(str)
        base_static = base["static_features"].astype(np.float32)
        base_raw = base["raw_ltwm_features"].astype(np.float32)
        base_action = base["action_features"].astype(np.float32)
        base_support = base["majority_support"].astype(bool)
    with np.load(final_feature_path, allow_pickle=False) as final:
        final_record_key = final["record_key"].astype(str)
        final_static = final["static_features"].astype(np.float32)
        final_raw = final["raw_ltwm_features"].astype(np.float32)
        final_action = final["action_features"].astype(np.float32)
        final_support = final["majority_support"].astype(bool)
    if not (
        np.array_equal(base_record_key, cohort["record_key"])
        and np.array_equal(final_record_key, cohort["record_key"])
    ):
        raise RuntimeError("combined-pool feature identity mismatch")
    incumbent_raw = base_raw[:, :1]
    incumbent_action = base_action[:, :1]
    candidate_static = np.concatenate(
        (base_static[:, 1:], final_static[:, 1:]), axis=1
    )
    candidate_raw = np.concatenate(
        (base_raw[:, 1:], final_raw[:, 1:]), axis=1
    )
    candidate_action = np.concatenate(
        (base_action[:, 1:], final_action[:, 1:]), axis=1
    )
    candidate_support = np.concatenate(
        (base_support[:, 1:], final_support[:, 1:]), axis=1
    )
    incumbent_raw = np.repeat(incumbent_raw, 32, axis=1)
    incumbent_action = np.repeat(incumbent_action, 32, axis=1)
    candidate_kind = np.concatenate(
        (
            np.zeros((100, 16, 1), dtype=np.float32),
            np.ones((100, 16, 1), dtype=np.float32),
        ),
        axis=1,
    )
    feature = np.concatenate(
        (
            candidate_static,
            candidate_raw,
            candidate_raw - incumbent_raw,
            candidate_action,
            candidate_action - incumbent_action,
            candidate_kind,
            candidate_support[..., None].astype(np.float32),
        ),
        axis=-1,
    ).astype(np.float32)
    combined_absolute = cohort["absolute_combined"]
    delta = combined_absolute[:, 1:] - combined_absolute[:, :1]
    dual = (delta[..., 0] < -1.0e-7) & (delta[..., 3] < -1.0e-7)
    harm = (delta[..., 0] > 1.0e-7) | (delta[..., 3] > 1.0e-7)
    cohort.update(
        {
            "combined_feature": feature,
            "combined_delta": delta.astype(np.float32),
            "combined_dual": dual.astype(np.float32),
            "combined_harm": harm.astype(np.float32),
            "combined_valid": candidate_support,
            "combined_absolute": combined_absolute,
            "base_feature_path": base_feature_path,
            "baseline_metrics": json.loads(
                result_path.read_text(encoding="utf-8")
            )["methods"]["universal_d_cem_false"],
        }
    )
    return cohort


def selected_for_policy(
    prediction: dict[str, np.ndarray],
    mean_std: np.ndarray,
    valid: np.ndarray,
    candidate_mask: np.ndarray,
    policy: dict[str, float],
) -> np.ndarray:
    p_dual = sigmoid(prediction["dual_logit"])
    p_harm = sigmoid(prediction["harm_logit"])
    mean = prediction["mean"]
    upper_ate = (
        mean[..., 0]
        + float(policy["uncertainty_scale"]) * mean_std[..., 0]
    )
    upper_objective = (
        mean[..., 3]
        + float(policy["uncertainty_scale"]) * mean_std[..., 3]
    )
    allowed = (
        valid
        & candidate_mask
        & (p_dual >= float(policy["dual_threshold"]))
        & (p_harm <= float(policy["harm_threshold"]))
        & (
            p_dual - p_harm
            >= float(policy["dual_harm_gap_threshold"])
        )
        & (upper_ate < -float(policy["ate_margin"]))
        & (upper_objective < -float(policy["objective_margin"]))
    )
    utility = (
        mean[..., 0]
        + 0.25 * mean[..., 1]
        + 0.25 * np.maximum(mean[..., 2], 0.0)
        + 0.50 * mean[..., 3]
        + float(policy["safety_rank_weight"]) * (p_harm - p_dual)
    )
    utility = np.where(allowed, utility, np.inf)
    candidate = utility.argmin(axis=1)
    exists = np.isfinite(
        utility[np.arange(len(utility)), candidate]
    )
    return np.where(exists, candidate + 1, 0).astype(np.int64)


def evaluate_policy(
    *,
    false_selected: np.ndarray,
    true_selected: np.ndarray,
    cohorts: list[dict[str, Any]],
    cohort_name: np.ndarray,
) -> dict[str, Any]:
    per_cohort: dict[str, Any] = {}
    strict = True
    quality = True
    penalty = 0.0
    coverage_sum = 0.0
    cem_count = 0
    for cohort in cohorts:
        mask = cohort_name == cohort["name"]
        false = method_summary(
            false_selected[mask],
            cohort["combined_delta"],
            cohort["combined_absolute"],
        )
        true = method_summary(
            true_selected[mask],
            cohort["combined_delta"],
            cohort["combined_absolute"],
        )
        safety_checks = {
            "coverage_at_least_0.40": true["coverage"] >= 0.40,
            "dual_at_least_0.90": (
                optional(true["dual_precision"], 0.0) >= 0.90
            ),
            "wilson_at_least_0.80": (
                optional(true["dual_wilson_lower_95"], 0.0) >= 0.80
            ),
            "harm_at_most_0.10": (
                optional(true["harm_rate"], 1.0) <= 0.10
            ),
            "dual_not_below_same_selector_false": (
                optional(true["dual_precision"], 0.0)
                >= optional(false["dual_precision"], 0.0)
            ),
            "wilson_not_below_same_selector_false": (
                optional(true["dual_wilson_lower_95"], 0.0)
                >= optional(false["dual_wilson_lower_95"], 0.0)
            ),
            "harm_not_above_same_selector_false": (
                optional(true["harm_rate"], 1.0)
                <= optional(false["harm_rate"], 1.0)
            ),
            "coverage_not_below_same_selector_false": (
                true["coverage"] >= false["coverage"]
            ),
        }
        quality_checks = {
            "all_means_not_worse_than_same_selector_false": all(
                true["absolute"][name]["mean_m"]
                <= false["absolute"][name]["mean_m"] + 1.0e-12
                for name in METRIC_NAMES
            ),
            "all_p90_not_worse_than_same_selector_false": all(
                true["absolute"][name]["p90_m"]
                <= false["absolute"][name]["p90_m"] + 1.0e-12
                for name in METRIC_NAMES
            ),
            "at_least_one_mean_strictly_better": any(
                true["absolute"][name]["mean_m"]
                < false["absolute"][name]["mean_m"] - 1.0e-7
                for name in METRIC_NAMES
            ),
        }
        strict &= all(safety_checks.values())
        quality &= all(quality_checks.values())
        coverage = true["coverage"]
        dual = optional(true["dual_precision"], 0.0)
        wilson = optional(true["dual_wilson_lower_95"], 0.0)
        harm = optional(true["harm_rate"], 1.0)
        penalty += (
            12.0 * max(0.0, 0.40 - coverage)
            + 10.0 * max(0.0, 0.90 - dual)
            + 10.0 * max(0.0, 0.80 - wilson)
            + 10.0 * max(0.0, harm - 0.10)
            + (
                0.0
                if safety_checks[
                    "dual_not_below_same_selector_false"
                ]
                else 0.5
            )
            + (
                0.0
                if safety_checks[
                    "harm_not_above_same_selector_false"
                ]
                else 0.5
            )
            + (
                0.0
                if quality_checks[
                    "all_means_not_worse_than_same_selector_false"
                ]
                else 0.5
            )
            + (
                0.0
                if quality_checks[
                    "all_p90_not_worse_than_same_selector_false"
                ]
                else 0.5
            )
        )
        local_cem = int((true_selected[mask] > 16).sum())
        cem_count += local_cem
        coverage_sum += coverage
        per_cohort[cohort["name"]] = {
            "cem_selected": local_cem,
            "same_selector_cem_false": false,
            "cem_true": true,
            "safety_checks": safety_checks,
            "quality_checks": quality_checks,
        }
    return {
        "per_cohort": per_cohort,
        "strict_safety_all_cohorts": strict,
        "quality_all_cohorts": quality,
        "strict_feasible": strict and quality and cem_count > 0,
        "penalty": float(penalty),
        "coverage_sum": float(coverage_sum),
        "cem_selected_total": cem_count,
    }


def main() -> None:
    args = parse_args()
    result_path = args.output_dir / "result.json"
    if result_path.exists():
        raise FileExistsError(result_path)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    v4 = json.loads(args.v4_result.read_text(encoding="utf-8"))
    cohorts = [
        load_combined_cohort(
            name=value["name"],
            final_feature_path=Path(value["features"]),
            final_score_path=Path(value["scores"]),
            result_path=Path(value["result"]),
        )
        for value in v4["inputs"]
    ]
    feature = np.concatenate(
        [value["combined_feature"] for value in cohorts]
    )
    delta = np.concatenate(
        [value["combined_delta"] for value in cohorts]
    )
    dual = np.concatenate(
        [value["combined_dual"] for value in cohorts]
    )
    harm = np.concatenate(
        [value["combined_harm"] for value in cohorts]
    )
    valid = np.concatenate(
        [value["combined_valid"] for value in cohorts]
    )
    group = np.concatenate(
        [
            np.asarray([f"{value['name']}::{route}" for route in value["route"]])
            for value in cohorts
        ]
    )
    cohort_name = np.concatenate(
        [
            np.full(len(value["route"]), value["name"], dtype="<U32")
            for value in cohorts
        ]
    )
    fold_id = np.asarray(
        [fold_for_group(value, args.folds) for value in group],
        dtype=np.int64,
    )
    config = ModelConfig(input_dim=feature.shape[-1], hidden_dim=128)
    member_predictions = []
    fold_records = []
    for member in range(args.ensemble_members):
        oof = {
            "mean": np.empty((len(feature), 32, 4), dtype=np.float32),
            "dual_logit": np.empty((len(feature), 32), dtype=np.float32),
            "harm_logit": np.empty((len(feature), 32), dtype=np.float32),
        }
        for fold in range(args.folds):
            train_index = np.flatnonzero(fold_id != fold)
            heldout_index = np.flatnonzero(fold_id == fold)
            model, stats = train_model(
                feature=feature,
                delta=delta,
                dual=dual,
                harm=harm,
                groups=group,
                train_index=train_index,
                config=config,
                epochs=args.epochs,
                batch_states=args.batch_states,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                seed=args.seed + member * 1009 + fold,
                device=args.device,
            )
            prediction = predict(
                model, stats, feature[heldout_index], device=args.device
            )
            for key in oof:
                oof[key][heldout_index] = prediction[key]
            if member == 0:
                fold_records.append(
                    {
                        "fold": fold,
                        "train_states": int(len(train_index)),
                        "heldout_states": int(len(heldout_index)),
                        "train_routes": int(
                            len(np.unique(group[train_index]))
                        ),
                        "heldout_routes": int(
                            len(np.unique(group[heldout_index]))
                        ),
                    }
                )
            del model
            if args.device == "cuda":
                torch.cuda.empty_cache()
        member_predictions.append(oof)
    prediction = {
        key: np.mean(
            np.stack([value[key] for value in member_predictions]), axis=0
        ).astype(np.float32)
        for key in member_predictions[0]
    }
    mean_std = np.std(
        np.stack([value["mean"] for value in member_predictions]), axis=0
    ).astype(np.float32)
    false_mask = np.zeros((len(feature), 32), dtype=bool)
    false_mask[:, :16] = True
    true_mask = np.ones((len(feature), 32), dtype=bool)

    policies = []
    for safety_rank_weight in (0.0, 0.10, 0.25, 0.50, 1.0):
        for uncertainty_scale in (0.0, 0.5, 1.0, 1.5):
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
                    for gap_threshold in (-0.10, 0.0, 0.10, 0.20):
                        for margin in (0.0, 0.005, 0.010, 0.020):
                            policy = {
                                "safety_rank_weight": safety_rank_weight,
                                "uncertainty_scale": uncertainty_scale,
                                "dual_threshold": float(dual_threshold),
                                "harm_threshold": harm_threshold,
                                "dual_harm_gap_threshold": gap_threshold,
                                "ate_margin": margin,
                                "objective_margin": margin,
                            }
                            false_selected = selected_for_policy(
                                prediction,
                                mean_std,
                                valid,
                                false_mask,
                                policy,
                            )
                            true_selected = selected_for_policy(
                                prediction,
                                mean_std,
                                valid,
                                true_mask,
                                policy,
                            )
                            evaluation = evaluate_policy(
                                false_selected=false_selected,
                                true_selected=true_selected,
                                cohorts=cohorts,
                                cohort_name=cohort_name,
                            )
                            policies.append(
                                {
                                    "policy": policy,
                                    "false_selected": false_selected,
                                    "true_selected": true_selected,
                                    **evaluation,
                                }
                            )
    feasible = [value for value in policies if value["strict_feasible"]]
    if feasible:
        best = max(
            feasible,
            key=lambda value: (
                value["coverage_sum"],
                value["cem_selected_total"],
                -value["penalty"],
            ),
        )
        tier = "strict_all_three_oof_cohorts"
    else:
        best = min(
            policies,
            key=lambda value: (
                value["penalty"],
                -value["coverage_sum"],
                -value["cem_selected_total"],
            ),
        )
        tier = "best_available_oof"

    args.output_dir.mkdir(parents=True, exist_ok=False)
    oof_path = args.output_dir / "oof_predictions.npz"
    write_npz(
        oof_path,
        record_key=np.concatenate(
            [value["record_key"] for value in cohorts]
        ),
        cohort=cohort_name,
        route_group=group,
        fold_id=fold_id,
        mean=prediction["mean"],
        mean_std=mean_std,
        dual_logit=prediction["dual_logit"],
        harm_logit=prediction["harm_logit"],
        valid=valid,
        cem_false_selected_index=best["false_selected"],
        cem_true_selected_index=best["true_selected"],
    )
    checkpoint_records = []
    if feasible:
        full_index = np.arange(len(feature))
        for member in range(args.ensemble_members):
            seed = args.seed + 10001 + member * 1009
            model, stats = train_model(
                feature=feature,
                delta=delta,
                dual=dual,
                harm=harm,
                groups=group,
                train_index=full_index,
                config=config,
                epochs=args.epochs,
                batch_states=args.batch_states,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                seed=seed,
                device=args.device,
            )
            checkpoint = args.output_dir / f"member_{member}.pt"
            torch.save(
                {
                    "schema": "recon_cem_combined_pool_v7_member",
                    "config": asdict(config),
                    **stats,
                    "model": {
                        key: value.detach().cpu()
                        for key, value in model.state_dict().items()
                    },
                    "seed": seed,
                },
                checkpoint,
            )
            checkpoint_records.append(
                {
                    "path": str(checkpoint),
                    "sha256": file_sha256(checkpoint),
                    "seed": seed,
                }
            )
            del model
            if args.device == "cuda":
                torch.cuda.empty_cache()
    report = {
        "schema": "recon_cem_combined_pool_v7",
        "status": (
            "route_disjoint_oof_passed_full_ensemble_frozen"
            if feasible
            else "route_disjoint_oof_failed_no_full_checkpoint"
        ),
        "evidence_role": "development_only_three_spent_cohorts",
        "cem": True,
        "states": int(len(feature)),
        "routes": int(len(np.unique(group))),
        "candidate_training_examples": int(len(feature) * 32),
        "feature_dim": int(feature.shape[-1]),
        "feature_contract": (
            "independent raw model: DINOv3 static + raw LTWM + action "
            "features/deltas + physical support + base/CEM token; no "
            "native scorer-v2 prediction features"
        ),
        "config": {
            **asdict(config),
            "folds": args.folds,
            "epochs": args.epochs,
            "batch_states": args.batch_states,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "ensemble_members": args.ensemble_members,
            "seed": args.seed,
        },
        "folds": fold_records,
        "policy_search": {
            "tier": tier,
            "strict_feasible_policies": len(feasible),
            "searched_policies": len(policies),
            "policy": best["policy"],
            "per_cohort": best["per_cohort"],
            "strict_safety_all_cohorts": best[
                "strict_safety_all_cohorts"
            ],
            "quality_all_cohorts": best["quality_all_cohorts"],
            "strict_feasible": best["strict_feasible"],
            "penalty": best["penalty"],
            "coverage_sum": best["coverage_sum"],
            "cem_selected_total": best["cem_selected_total"],
        },
        "oof_predictions": str(oof_path),
        "oof_predictions_sha256": file_sha256(oof_path),
        "checkpoints": checkpoint_records,
        "source_v4_result": str(args.v4_result),
        "source_v4_result_sha256": file_sha256(args.v4_result),
        "next_confirmation_accessed": False,
        "formal_claim_allowed": False,
    }
    write_json(result_path, report)
    print(
        json.dumps(
            {
                "result": str(result_path),
                "result_sha256": file_sha256(result_path),
                "policy_search": report["policy_search"],
                "checkpoints": checkpoint_records,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
