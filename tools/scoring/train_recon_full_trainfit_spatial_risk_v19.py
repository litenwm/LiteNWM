#!/usr/bin/env python3
"""Train spatial tail risk on all 5,292 RECON train-fit states."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tools.scoring.run_recon_cem_latent_base_risk_v16 import (  # noqa: E402
    audited_base_score,
)
from tools.scoring.train_recon_cem_combined_pool_v7 import (  # noqa: E402
    load_combined_cohort,
    resolve_base_feature,
)
from tools.scoring.train_recon_cem_latent_base_risk_v16 import (  # noqa: E402
    TailConfig,
    calibration,
    compact_base_feature,
    predict_tail,
    train_tail_model,
)
from tools.scoring.train_recon_cem_loco_twostage_v15 import (  # noqa: E402
    load_replication,
)
from tools.scoring.train_recon_cem_state_gate_v6 import (  # noqa: E402
    method_summary,
    optional,
)
from tools.scoring.train_recon_dense_meta_dinov3_v1 import (  # noqa: E402
    file_sha256,
    write_json,
    write_npz,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-features", type=Path, required=True)
    parser.add_argument("--train-scores", type=Path, required=True)
    parser.add_argument("--train-rollout", type=Path, required=True)
    parser.add_argument("--v4-result", type=Path, required=True)
    parser.add_argument("--replication-result", type=Path, required=True)
    parser.add_argument("--replication-base-features", type=Path, required=True)
    parser.add_argument("--replication-cem-features", type=Path, required=True)
    parser.add_argument("--replication-cem-scores", type=Path, required=True)
    parser.add_argument(
        "--heldout-rollouts", type=Path, nargs=4, required=True
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=140)
    parser.add_argument("--batch-states", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=4.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-3)
    parser.add_argument("--ensemble-members", type=int, default=3)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--seed", type=int, default=20260820)
    return parser.parse_args()


def compact_with_rollout(
    feature_path: Path,
    score_path: Path,
    rollout_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(feature_path, allow_pickle=False) as payload:
        record = payload["record_key"].astype(str)
    with np.load(rollout_path, allow_pickle=False) as payload:
        rollout_record = payload["record_key"].astype(str)
        rollout = payload["spatial_rollout_features"].astype(np.float32)
        metadata = json.loads(str(payload["metadata"]))
    if (
        not np.array_equal(record, rollout_record)
        or rollout.shape != (len(record), 17, 258)
    ):
        raise RuntimeError("v19 rollout/feature identity mismatch")
    if metadata.get("outcomes_accessed") or metadata.get("labels_in_output"):
        raise RuntimeError("v19 rollout contains forbidden labels")
    empty_state_latent = np.empty((len(record), 0), dtype=np.float32)
    feature, support = compact_base_feature(
        feature_path,
        score_path,
        empty_state_latent,
    )
    return (
        np.concatenate((feature, rollout[:, 1:]), axis=2),
        support,
        record,
    )


def main() -> None:
    args = parse_args()
    result_path = args.output_dir / "result.json"
    if result_path.exists():
        raise FileExistsError(result_path)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    train_feature, train_valid, train_record = compact_with_rollout(
        args.train_features,
        args.train_scores,
        args.train_rollout,
    )
    with np.load(args.train_features, allow_pickle=False) as payload:
        targets = payload["targets"].astype(np.float32)
        train_group = payload["family"].astype(str)
    train_delta = targets[:, 1:, :4]
    train_dual = (
        (train_delta[..., 0] < -1.0e-7)
        & (train_delta[..., 3] < -1.0e-7)
    ).astype(np.float32)
    train_harm = (
        (train_delta[..., 0] > 1.0e-7)
        | (train_delta[..., 3] > 1.0e-7)
    ).astype(np.float32)

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
    heldout_features = []
    heldout_valid_values = []
    heldout_records = []
    for cohort, rollout_path in zip(cohorts, args.heldout_rollouts):
        base_feature = resolve_base_feature(cohort["result_path"])
        base_score = audited_base_score(cohort["result_path"])
        feature, valid, record = compact_with_rollout(
            base_feature, base_score, rollout_path
        )
        if not np.array_equal(record, cohort["record_key"]):
            raise RuntimeError("v19 heldout cohort identity mismatch")
        heldout_features.append(feature)
        heldout_valid_values.append(valid)
        heldout_records.append(record)
    heldout_feature = np.concatenate(heldout_features)
    heldout_valid = np.concatenate(heldout_valid_values)
    cohort_name = np.concatenate(
        [
            np.full(len(value["route"]), value["name"], dtype="<U40")
            for value in cohorts
        ]
    )
    heldout_group = np.concatenate(
        [
            np.asarray(
                [f"{value['name']}::{route}" for route in value["route"]]
            )
            for value in cohorts
        ]
    )
    if set(train_record) & set(np.concatenate(heldout_records)):
        raise RuntimeError("v19 train/heldout record leakage")
    if set(train_group) & set(heldout_group):
        raise RuntimeError("v19 train/heldout group leakage")

    config = TailConfig(
        input_dim=train_feature.shape[-1],
        hidden_dim=192,
        dropout=0.25,
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    member_dual = []
    member_harm = []
    checkpoints = []
    train_index = np.arange(len(train_feature))
    for member in range(args.ensemble_members):
        seed = args.seed + member * 1009
        model, mean, scale = train_tail_model(
            feature=train_feature,
            dual=train_dual,
            harm=train_harm,
            group=train_group,
            train_index=train_index,
            config=config,
            args=args,
            seed=seed,
        )
        dual_value, harm_value = predict_tail(
            model,
            mean,
            scale,
            heldout_feature,
            args.device,
        )
        member_dual.append(dual_value)
        member_harm.append(harm_value)
        checkpoint = args.output_dir / f"member_{member}.pt"
        torch.save(
            {
                "schema": "recon_full_trainfit_spatial_risk_v19_member",
                "config": asdict(config),
                "feature_mean": mean,
                "feature_scale": scale,
                "model": {
                    key: value.detach().cpu()
                    for key, value in model.state_dict().items()
                },
                "seed": seed,
            },
            checkpoint,
        )
        checkpoints.append(
            {
                "path": str(checkpoint),
                "sha256": file_sha256(checkpoint),
                "seed": seed,
            }
        )
        del model
        if args.device == "cuda":
            torch.cuda.empty_cache()

    dual_stack = np.stack(member_dual)
    harm_stack = np.stack(member_harm)
    p_dual = dual_stack.mean(0)
    p_harm = harm_stack.mean(0)
    p_dual_std = dual_stack.std(0)
    p_harm_std = harm_stack.std(0)
    heldout_dual = np.concatenate(
        [value["combined_dual"][:, :16] for value in cohorts]
    )
    heldout_harm = np.concatenate(
        [value["combined_harm"][:, :16] for value in cohorts]
    )
    calibration_by_cohort = {}
    for cohort in cohorts:
        mask = cohort_name == cohort["name"]
        calibration_by_cohort[cohort["name"]] = {
            "dual": calibration(p_dual[mask], heldout_dual[mask]),
            "harm": calibration(p_harm[mask], heldout_harm[mask]),
        }

    policies = []
    for z in (0.0, 0.5, 1.0, 1.5, 2.0):
        dual_lower = p_dual - z * p_dual_std
        harm_upper = p_harm + z * p_harm_std
        for risk_weight in (0.5, 1.0, 2.0, 4.0):
            score = np.where(
                heldout_valid,
                dual_lower - risk_weight * harm_upper,
                -np.inf,
            )
            candidate = score.argmax(axis=1)
            row = np.arange(len(candidate))
            for dual_threshold in (
                0.50,
                0.60,
                0.70,
                0.75,
                0.80,
                0.85,
                0.90,
                0.95,
            ):
                for harm_threshold in (
                    0.025,
                    0.05,
                    0.075,
                    0.10,
                    0.15,
                    0.20,
                    0.30,
                ):
                    promote = (
                        (dual_lower[row, candidate] >= dual_threshold)
                        & (harm_upper[row, candidate] <= harm_threshold)
                    )
                    selected = np.where(
                        promote, candidate + 1, 0
                    ).astype(np.int64)
                    per_cohort = {}
                    strict = True
                    for cohort in cohorts:
                        mask = cohort_name == cohort["name"]
                        metrics = method_summary(
                            selected[mask],
                            cohort["combined_delta"],
                            cohort["combined_absolute"],
                        )
                        checks = {
                            "coverage_at_least_0.40": (
                                metrics["coverage"] >= 0.40
                            ),
                            "dual_at_least_0.90": (
                                optional(
                                    metrics["dual_precision"], 0.0
                                )
                                >= 0.90
                            ),
                            "wilson_at_least_0.80": (
                                optional(
                                    metrics["dual_wilson_lower_95"], 0.0
                                )
                                >= 0.80
                            ),
                            "harm_at_most_0.10": (
                                optional(metrics["harm_rate"], 1.0)
                                <= 0.10
                            ),
                        }
                        strict &= all(checks.values())
                        per_cohort[cohort["name"]] = {
                            "metrics": metrics,
                            "checks": checks,
                        }
                    policies.append(
                        {
                            "policy": {
                                "uncertainty_scale": z,
                                "risk_weight": risk_weight,
                                "dual_threshold": dual_threshold,
                                "harm_threshold": harm_threshold,
                            },
                            "selected": selected,
                            "per_cohort": per_cohort,
                            "strict": strict,
                            "coverage_sum": sum(
                                value["metrics"]["coverage"]
                                for value in per_cohort.values()
                            ),
                        }
                    )
    feasible = [value for value in policies if value["strict"]]
    if feasible:
        best = max(feasible, key=lambda value: value["coverage_sum"])
        tier = "strict_full_trainfit_all_four_heldout"
    else:
        covered = [
            value
            for value in policies
            if all(
                item["metrics"]["coverage"] >= 0.40
                for item in value["per_cohort"].values()
            )
        ]
        best = max(
            covered or policies,
            key=lambda value: (
                min(
                    optional(item["metrics"]["dual_precision"], 0.0)
                    for item in value["per_cohort"].values()
                ),
                -max(
                    optional(item["metrics"]["harm_rate"], 1.0)
                    for item in value["per_cohort"].values()
                ),
                value["coverage_sum"],
            ),
        )
        tier = "failed_best_available_full_trainfit"

    top40 = {}
    for cohort in cohorts:
        mask = cohort_name == cohort["name"]
        best_ordering = None
        for z in (0.0, 0.5, 1.0, 1.5, 2.0):
            dual_lower = p_dual[mask] - z * p_dual_std[mask]
            harm_upper = p_harm[mask] + z * p_harm_std[mask]
            for weight in (0.5, 1.0, 2.0, 4.0):
                score = np.where(
                    heldout_valid[mask],
                    dual_lower - weight * harm_upper,
                    -np.inf,
                )
                candidate = score.argmax(axis=1)
                row = np.arange(len(candidate))
                order = np.argsort(
                    -score[row, candidate], kind="stable"
                )[:40]
                chosen = candidate[order]
                dual_count = int(
                    heldout_dual[mask][order, chosen].sum()
                )
                harm_count = int(
                    heldout_harm[mask][order, chosen].sum()
                )
                value = {
                    "z": z,
                    "risk_weight": weight,
                    "dual_precision": dual_count / 40.0,
                    "harm_rate": harm_count / 40.0,
                }
                if best_ordering is None or (
                    value["dual_precision"],
                    -value["harm_rate"],
                ) > (
                    best_ordering["dual_precision"],
                    -best_ordering["harm_rate"],
                ):
                    best_ordering = value
        top40[cohort["name"]] = best_ordering

    prediction_path = args.output_dir / "heldout_predictions.npz"
    write_npz(
        prediction_path,
        record_key=np.concatenate(heldout_records),
        cohort=cohort_name,
        route_group=heldout_group,
        p_dual=p_dual,
        p_harm=p_harm,
        p_dual_std=p_dual_std,
        p_harm_std=p_harm_std,
        valid=heldout_valid,
        selected_index=best["selected"],
    )
    report = {
        "schema": "recon_full_trainfit_spatial_risk_v19",
        "status": (
            "heldout_base_safety_passed_cem_replacement_may_train"
            if feasible
            else "heldout_base_safety_failed"
        ),
        "evidence_role": "development_only_four_spent_heldout_cohorts",
        "train_states": int(len(train_feature)),
        "train_groups": int(len(np.unique(train_group))),
        "heldout_states": int(len(heldout_feature)),
        "record_overlap": 0,
        "group_overlap": 0,
        "feature_dim": int(train_feature.shape[-1]),
        "candidate_conditioned_future_latent_used": True,
        "spatial_rollout_contract": (
            "real three-seed LTWM predicted-token spatial alignment to goal"
        ),
        "config": {
            **asdict(config),
            "epochs": args.epochs,
            "batch_states": args.batch_states,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "ensemble_members": args.ensemble_members,
            "seed": args.seed,
        },
        "calibration_by_cohort": calibration_by_cohort,
        "policy_search": {
            "tier": tier,
            "searched_policies": len(policies),
            "strict_feasible_policies": len(feasible),
            "policy": best["policy"],
            "per_cohort": best["per_cohort"],
            "strict": best["strict"],
        },
        "noncausal_top40_feature_separability_diagnostic_only": top40,
        "checkpoints": checkpoints,
        "heldout_predictions": str(prediction_path),
        "heldout_predictions_sha256": file_sha256(prediction_path),
        "train_features": str(args.train_features),
        "train_features_sha256": file_sha256(args.train_features),
        "train_scores": str(args.train_scores),
        "train_scores_sha256": file_sha256(args.train_scores),
        "train_rollout": str(args.train_rollout),
        "train_rollout_sha256": file_sha256(args.train_rollout),
        "heldout_rollouts": [
            {"path": str(path), "sha256": file_sha256(path)}
            for path in args.heldout_rollouts
        ],
        "cem_replacement_trained": False,
        "next_confirmation_accessed": False,
        "formal_claim_allowed": False,
        "scand_fresh_confirmation_available": False,
    }
    write_json(result_path, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
