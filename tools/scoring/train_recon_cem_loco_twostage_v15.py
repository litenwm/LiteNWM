#!/usr/bin/env python3
"""Train a leave-one-cohort-out, two-stage RECON base+CEM selector.

This is development-only calibration.  Every prediction used to choose the
policy is produced by models that did not train on the evaluated cohort.  At
deployment the frozen ensemble first chooses a safe K17 challenger and only
then permits a CEM candidate to replace it when the CEM risk is lower by a
fixed margin.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
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
    predict,
    sigmoid,
    train_model,
)
from tools.scoring.train_recon_cem_combined_pool_v7 import (  # noqa: E402
    evaluate_policy,
    load_combined_cohort,
)
from tools.scoring.train_recon_dense_meta_dinov3_v1 import (  # noqa: E402
    file_sha256,
    write_json,
    write_npz,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v4-result", type=Path, required=True)
    parser.add_argument("--replication-result", type=Path, required=True)
    parser.add_argument("--replication-base-features", type=Path, required=True)
    parser.add_argument("--replication-cem-features", type=Path, required=True)
    parser.add_argument("--replication-cem-scores", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=240)
    parser.add_argument("--batch-states", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=5.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-3)
    parser.add_argument("--ensemble-members", type=int, default=3)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--seed", type=int, default=20260816)
    return parser.parse_args()


def combined_feature(
    base_path: Path,
    cem_path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    with np.load(base_path, allow_pickle=False) as base:
        record = base["record_key"].astype(str)
        base_static = base["static_features"].astype(np.float32)
        base_raw = base["raw_ltwm_features"].astype(np.float32)
        base_action = base["action_features"].astype(np.float32)
        base_support = base["majority_support"].astype(bool)
    with np.load(cem_path, allow_pickle=False) as cem:
        if not np.array_equal(record, cem["record_key"].astype(str)):
            raise RuntimeError("replication base/CEM identity mismatch")
        cem_static = cem["static_features"].astype(np.float32)
        cem_raw = cem["raw_ltwm_features"].astype(np.float32)
        cem_action = cem["action_features"].astype(np.float32)
        cem_support = cem["majority_support"].astype(bool)
    candidate_static = np.concatenate(
        (base_static[:, 1:], cem_static[:, 1:]), axis=1
    )
    candidate_raw = np.concatenate(
        (base_raw[:, 1:], cem_raw[:, 1:]), axis=1
    )
    candidate_action = np.concatenate(
        (base_action[:, 1:], cem_action[:, 1:]), axis=1
    )
    incumbent_raw = np.repeat(base_raw[:, :1], 32, axis=1)
    incumbent_action = np.repeat(base_action[:, :1], 32, axis=1)
    candidate_kind = np.concatenate(
        (
            np.zeros((len(record), 16, 1), dtype=np.float32),
            np.ones((len(record), 16, 1), dtype=np.float32),
        ),
        axis=1,
    )
    valid = np.concatenate(
        (base_support[:, 1:], cem_support[:, 1:]), axis=1
    )
    feature = np.concatenate(
        (
            candidate_static,
            candidate_raw,
            candidate_raw - incumbent_raw,
            candidate_action,
            candidate_action - incumbent_action,
            candidate_kind,
            valid[..., None].astype(np.float32),
        ),
        axis=-1,
    ).astype(np.float32)
    return feature, valid


def load_replication(args: argparse.Namespace) -> dict[str, Any]:
    result = json.loads(
        args.replication_result.read_text(encoding="utf-8")
    )
    if (
        result.get("domain") != "recon"
        or not result.get("cem")
        or not result.get("confirmation_now_spent")
    ):
        raise RuntimeError("replication is not spent RECON CEM evidence")
    feature, valid = combined_feature(
        args.replication_base_features,
        args.replication_cem_features,
    )
    with np.load(
        args.replication_base_features, allow_pickle=False
    ) as payload:
        record = payload["record_key"].astype(str)
    with np.load(result["outcomes"], allow_pickle=False) as payload:
        if not np.array_equal(record, payload["record_key"].astype(str)):
            raise RuntimeError("replication feature/outcome identity mismatch")
        absolute = payload["combined_absolute_metrics"].astype(np.float32)
        route = payload["route_group"].astype(str)
        trajectory = payload["trajectory"].astype(str)
        baseline = payload[
            "original_universal_d_cem_false_selected_index"
        ].astype(np.int64)
    delta = absolute[:, 1:] - absolute[:, :1]
    return {
        "name": "v11_replication_spent4",
        "combined_feature": feature,
        "combined_delta": delta.astype(np.float32),
        "combined_dual": (
            (delta[..., 0] < -1.0e-7)
            & (delta[..., 3] < -1.0e-7)
        ).astype(np.float32),
        "combined_harm": (
            (delta[..., 0] > 1.0e-7)
            | (delta[..., 3] > 1.0e-7)
        ).astype(np.float32),
        "combined_valid": valid,
        "combined_absolute": absolute,
        "baseline": baseline,
        "record_key": record,
        "route": route,
        "trajectory": trajectory,
        "feature_path": args.replication_cem_features,
        "score_path": args.replication_cem_scores,
        "result_path": args.replication_result,
    }


def aggregate_predictions(
    members: list[dict[str, np.ndarray]],
) -> dict[str, np.ndarray]:
    means = np.stack([value["mean"] for value in members])
    dual = np.stack(
        [sigmoid(value["dual_logit"]) for value in members]
    )
    harm = np.stack(
        [sigmoid(value["harm_logit"]) for value in members]
    )
    return {
        "mean": means.mean(0).astype(np.float32),
        "mean_std": means.std(0).astype(np.float32),
        "dual_mean": dual.mean(0).astype(np.float32),
        "dual_std": dual.std(0).astype(np.float32),
        "harm_mean": harm.mean(0).astype(np.float32),
        "harm_std": harm.std(0).astype(np.float32),
    }


def candidate_risk(
    prediction: dict[str, np.ndarray],
    policy: dict[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    z = float(policy["uncertainty_scale"])
    dual_lower = (
        prediction["dual_mean"] - z * prediction["dual_std"]
    )
    harm_upper = (
        prediction["harm_mean"] + z * prediction["harm_std"]
    )
    delta_upper = (
        prediction["mean"] + z * prediction["mean_std"]
    )
    utility = (
        delta_upper[..., 0]
        + 0.25 * delta_upper[..., 1]
        + 0.25 * np.maximum(delta_upper[..., 2], 0.0)
        + 0.50 * delta_upper[..., 3]
        + float(policy["risk_weight"]) * (harm_upper - dual_lower)
    )
    allowed = (
        (dual_lower >= float(policy["dual_threshold"]))
        & (harm_upper <= float(policy["harm_threshold"]))
        & (delta_upper[..., 0] < -float(policy["margin"]))
        & (delta_upper[..., 3] < -float(policy["margin"]))
    )
    return utility.astype(np.float32), allowed


def select_two_stage(
    prediction: dict[str, np.ndarray],
    valid: np.ndarray,
    policy: dict[str, float],
    *,
    cem_enabled: bool,
) -> np.ndarray:
    utility, allowed = candidate_risk(prediction, policy)
    allowed &= valid
    base_allowed = allowed[:, :16]
    base_utility = np.where(
        base_allowed, utility[:, :16], np.inf
    )
    base_index = base_utility.argmin(axis=1)
    base_exists = np.isfinite(
        base_utility[np.arange(len(base_utility)), base_index]
    )
    selected = np.where(base_exists, base_index + 1, 0).astype(np.int64)
    if not cem_enabled:
        return selected

    cem_allowed = (
        allowed[:, 16:]
        & (
            prediction["dual_mean"][:, 16:]
            - float(policy["uncertainty_scale"])
            * prediction["dual_std"][:, 16:]
            >= float(policy["dual_threshold"])
            + float(policy["cem_dual_offset"])
        )
        & (
            prediction["harm_mean"][:, 16:]
            + float(policy["uncertainty_scale"])
            * prediction["harm_std"][:, 16:]
            <= float(policy["harm_threshold"])
            * float(policy["cem_harm_ratio"])
        )
    )
    cem_utility = np.where(
        cem_allowed, utility[:, 16:], np.inf
    )
    cem_index = cem_utility.argmin(axis=1)
    row = np.arange(len(cem_utility))
    cem_exists = np.isfinite(cem_utility[row, cem_index])
    base_value = np.where(
        base_exists, base_utility[row, base_index], np.inf
    )
    replace = (
        cem_exists
        & (
            cem_utility[row, cem_index]
            + float(policy["cem_replacement_margin"])
            < base_value
        )
    )
    return np.where(replace, cem_index + 17, selected).astype(np.int64)


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
    cohorts.append(load_replication(args))
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
    cohort_name = np.concatenate(
        [
            np.full(len(value["route"]), value["name"], dtype="<U40")
            for value in cohorts
        ]
    )
    group = np.concatenate(
        [
            np.asarray(
                [f"{value['name']}::{route}" for route in value["route"]]
            )
            for value in cohorts
        ]
    )
    config = ModelConfig(
        input_dim=feature.shape[-1],
        hidden_dim=128,
        dropout=0.20,
    )
    oof_members: dict[str, list[dict[str, np.ndarray]]] = {}
    checkpoint_records = []
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for heldout_number, heldout_name in enumerate(np.unique(cohort_name)):
        train_index = np.flatnonzero(cohort_name != heldout_name)
        heldout_index = np.flatnonzero(cohort_name == heldout_name)
        member_values = []
        for member in range(args.ensemble_members):
            seed = args.seed + heldout_number * 10007 + member * 1009
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
                seed=seed,
                device=args.device,
            )
            member_values.append(
                predict(
                    model,
                    stats,
                    feature[heldout_index],
                    device=args.device,
                )
            )
            checkpoint = (
                args.output_dir
                / f"exclude_{heldout_number}_member_{member}.pt"
            )
            torch.save(
                {
                    "schema": "recon_cem_loco_twostage_v15_member",
                    "excluded_cohort": heldout_name,
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
                    "excluded_cohort": heldout_name,
                    "seed": seed,
                }
            )
            del model
            if args.device == "cuda":
                torch.cuda.empty_cache()
        oof_members[heldout_name] = member_values

    prediction = {
        key: np.empty(
            (len(feature), 32, 4)
            if key in ("mean", "mean_std")
            else (len(feature), 32),
            dtype=np.float32,
        )
        for key in (
            "mean",
            "mean_std",
            "dual_mean",
            "dual_std",
            "harm_mean",
            "harm_std",
        )
    }
    for heldout_name, values in oof_members.items():
        index = np.flatnonzero(cohort_name == heldout_name)
        aggregated = aggregate_predictions(values)
        for key in prediction:
            prediction[key][index] = aggregated[key]

    policies: list[dict[str, Any]] = []
    for uncertainty_scale in (0.0, 0.5, 1.0, 1.5, 2.0):
        for dual_threshold in (0.50, 0.60, 0.70, 0.80, 0.85, 0.90):
            for harm_threshold in (0.05, 0.075, 0.10, 0.15, 0.20):
                for margin in (0.0, 0.0025, 0.005, 0.010):
                    for risk_weight in (0.25, 0.50, 1.0):
                        for replacement_margin in (
                            0.0,
                            0.005,
                            0.010,
                            0.020,
                        ):
                            policy = {
                                "uncertainty_scale": uncertainty_scale,
                                "dual_threshold": dual_threshold,
                                "harm_threshold": harm_threshold,
                                "margin": margin,
                                "risk_weight": risk_weight,
                                "cem_dual_offset": 0.05,
                                "cem_harm_ratio": 0.75,
                                "cem_replacement_margin": (
                                    replacement_margin
                                ),
                                "selection_order": (
                                    "safe_base_then_strictly_lower_risk_cem"
                                ),
                            }
                            false_selected = select_two_stage(
                                prediction,
                                valid,
                                policy,
                                cem_enabled=False,
                            )
                            true_selected = select_two_stage(
                                prediction,
                                valid,
                                policy,
                                cem_enabled=True,
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
                min(
                    item["cem_true"]["coverage"]
                    for item in value["per_cohort"].values()
                ),
                value["cem_selected_total"],
                value["coverage_sum"],
            ),
        )
        tier = "strict_leave_one_cohort_out_all_four"
    else:
        best = min(
            policies,
            key=lambda value: (
                value["penalty"],
                -value["coverage_sum"],
                -value["cem_selected_total"],
            ),
        )
        tier = "failed_best_available_loco"

    prediction_path = args.output_dir / "loco_predictions.npz"
    write_npz(
        prediction_path,
        record_key=np.concatenate(
            [value["record_key"] for value in cohorts]
        ),
        cohort=cohort_name,
        route_group=group,
        valid=valid,
        **prediction,
        cem_false_selected_index=best["false_selected"],
        cem_true_selected_index=best["true_selected"],
    )
    report = {
        "schema": "recon_cem_loco_twostage_v15",
        "status": (
            "loco_development_passed_frozen_ensemble"
            if feasible
            else "loco_development_failed"
        ),
        "evidence_role": "development_only_four_spent_cohorts",
        "cem": True,
        "states": int(len(feature)),
        "cohorts": list(np.unique(cohort_name)),
        "routes": int(len(np.unique(group))),
        "candidate_training_examples": int(len(feature) * 32),
        "feature_dim": int(feature.shape[-1]),
        "feature_contract": (
            "DINOv3 static + raw LTWM + action/delta + physical support "
            "+ base/CEM token; no native scorer-v2 prediction feature"
        ),
        "online_causal": True,
        "future_route_windows_used": False,
        "loco_protocol": (
            "every policy-search prediction excludes its entire cohort; "
            "deployment ensemble contains all twelve frozen exclusion models"
        ),
        "config": {
            **asdict(config),
            "epochs": args.epochs,
            "batch_states": args.batch_states,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "ensemble_members_per_exclusion": args.ensemble_members,
            "seed": args.seed,
        },
        "policy_search": {
            "tier": tier,
            "searched_policies": len(policies),
            "strict_feasible_policies": len(feasible),
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
        "loco_predictions": str(prediction_path),
        "loco_predictions_sha256": file_sha256(prediction_path),
        "checkpoints": checkpoint_records,
        "inputs": [
            {
                "name": value["name"],
                "features": str(value["feature_path"]),
                "features_sha256": file_sha256(value["feature_path"]),
                "scores": str(value["score_path"]),
                "scores_sha256": file_sha256(value["score_path"]),
                "result": str(value["result_path"]),
                "result_sha256": file_sha256(value["result_path"]),
            }
            for value in cohorts
        ],
        "next_confirmation_accessed": False,
        "formal_claim_allowed": False,
        "scand_fresh_confirmation_available": False,
    }
    write_json(result_path, report)
    print(
        json.dumps(
            {
                "result": str(result_path),
                "result_sha256": file_sha256(result_path),
                "policy_search": report["policy_search"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
