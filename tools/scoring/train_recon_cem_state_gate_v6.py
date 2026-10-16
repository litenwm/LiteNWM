#!/usr/bin/env python3
"""Train a route-disjoint state-risk gate for frozen RECON CEM proposals."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from scoring.dense_meta_v1 import wilson_lower  # noqa: E402
from tools.scoring.train_recon_cem_candidate_calibrator_v4 import (  # noqa: E402
    METRIC_NAMES,
    file_sha256,
    load_cohort,
    write_json,
    write_npz,
)


@dataclass(frozen=True)
class ModelConfig:
    input_dim: int
    hidden_dim: int = 96
    dropout: float = 0.25


class StateGate(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
        )
        self.head = nn.Linear(config.hidden_dim, 6)

    def forward(self, value: torch.Tensor) -> dict[str, torch.Tensor]:
        output = self.head(self.trunk(value))
        return {
            "mean": output[..., :4],
            "dual_logit": output[..., 4],
            "harm_logit": output[..., 5],
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v4-result", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=240)
    parser.add_argument("--batch-states", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=5.0e-4)
    parser.add_argument("--weight-decay", type=float, default=5.0e-4)
    parser.add_argument("--ensemble-members", type=int, default=3)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--seed", type=int, default=20260806)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def fold_for_group(group: str, folds: int) -> int:
    digest = hashlib.sha256(f"cem-state-gate-v6|{group}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % folds


def sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -30.0, 30.0)))


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


def state_feature_and_challenger(
    feature_path: Path,
    score_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(feature_path, allow_pickle=False) as payload:
        candidates = payload["candidates"].astype(np.float32)
        actions = payload["actions_h4"].astype(np.float32)
        static = payload["static_features"].astype(np.float32)
        raw = payload["raw_ltwm_features"].astype(np.float32)
        record_key = payload["record_key"].astype(str)
    with np.load(score_path, allow_pickle=False) as payload:
        energy = np.nan_to_num(
            payload["energy"].astype(np.float32),
            nan=10.0,
            posinf=10.0,
            neginf=-10.0,
        )
        mean = payload["mean"].astype(np.float32)
        std = payload["std"].astype(np.float32)
        p_dual = payload["dual_probability"].astype(np.float32)
        p_harm = payload["harm_probability"].astype(np.float32)
        rank = payload["rank_logit"].astype(np.float32)
        valid = payload["valid"].astype(bool)
    challenger_valid = valid[:, 1:]
    challenger_energy = np.where(
        challenger_valid, energy[:, 1:], np.inf
    )
    challenger = challenger_energy.argmin(axis=1) + 1
    if not np.all(np.isfinite(
        challenger_energy[
            np.arange(len(challenger_energy)), challenger - 1
        ]
    )):
        raise RuntimeError("a state has no valid final CEM challenger")
    row = np.arange(len(challenger))
    chosen_static = static[row, challenger]
    chosen_raw = raw[row, challenger]
    incumbent_static = static[:, 0]
    incumbent_raw = raw[:, 0]
    actions_xy = actions[..., :2] / 0.25
    chosen_action = actions_xy[row, challenger].reshape(len(row), -1)
    incumbent_action = actions_xy[:, 0].reshape(len(row), -1)
    chosen_path = candidates[row, challenger].reshape(len(row), -1)
    incumbent_path = candidates[:, 0].reshape(len(row), -1)
    score_tensor = np.concatenate(
        (
            mean,
            np.log1p(std),
            p_dual[..., None],
            p_harm[..., None],
            rank[..., None],
            energy[..., None],
        ),
        axis=-1,
    )
    chosen_score = score_tensor[row, challenger]
    incumbent_score = score_tensor[:, 0]
    challenger_stats = masked_stats(
        score_tensor[:, 1:], challenger_valid
    )
    proposal_mean = np.mean(actions_xy[:, 1:], axis=1).reshape(len(row), -1)
    proposal_std = np.std(actions_xy[:, 1:], axis=1).reshape(len(row), -1)
    state_feature = np.concatenate(
        (
            incumbent_static,
            chosen_static - incumbent_static,
            incumbent_raw,
            chosen_raw,
            chosen_raw - incumbent_raw,
            incumbent_score,
            chosen_score,
            chosen_score - incumbent_score,
            challenger_stats,
            incumbent_action,
            chosen_action,
            chosen_action - incumbent_action,
            incumbent_path,
            chosen_path,
            chosen_path - incumbent_path,
            proposal_mean,
            proposal_std,
            challenger_valid.mean(axis=1, keepdims=True),
        ),
        axis=1,
    ).astype(np.float32)
    state_feature = np.nan_to_num(
        state_feature, nan=0.0, posinf=10.0, neginf=-10.0
    )
    return state_feature, challenger.astype(np.int64), record_key


def normalization(
    feature: np.ndarray,
    target: np.ndarray,
    index: np.ndarray,
) -> dict[str, np.ndarray]:
    return {
        "feature_mean": feature[index].mean(0).astype(np.float32),
        "feature_scale": np.maximum(
            feature[index].std(0), 1.0e-4
        ).astype(np.float32),
        "target_mean": target[index].mean(0).astype(np.float32),
        "target_scale": np.maximum(
            target[index].std(0), 1.0e-4
        ).astype(np.float32),
    }


def train_model(
    *,
    feature: np.ndarray,
    target: np.ndarray,
    dual: np.ndarray,
    harm: np.ndarray,
    groups: np.ndarray,
    train_index: np.ndarray,
    config: ModelConfig,
    epochs: int,
    batch_states: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    device: str,
) -> tuple[StateGate, dict[str, np.ndarray]]:
    set_seed(seed)
    stats = normalization(feature, target, train_index)
    x = (
        (feature - stats["feature_mean"]) / stats["feature_scale"]
    ).astype(np.float32)
    y = (
        (target - stats["target_mean"]) / stats["target_scale"]
    ).astype(np.float32)
    model = StateGate(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    train_groups, counts = np.unique(groups[train_index], return_counts=True)
    inverse = {
        group: float(1.0 / count)
        for group, count in zip(train_groups, counts)
    }
    state_weight = np.asarray(
        [inverse.get(group, 0.0) for group in groups], dtype=np.float32
    )
    state_weight[train_index] *= float(
        len(train_index) / state_weight[train_index].sum()
    )
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        order = rng.permutation(train_index)
        model.train()
        for start in range(0, len(order), batch_states):
            index = order[start : start + batch_states]
            xb = torch.from_numpy(x[index]).to(device)
            yb = torch.from_numpy(y[index]).to(device)
            dualb = torch.from_numpy(dual[index]).to(device)
            harmb = torch.from_numpy(harm[index]).to(device)
            weight = torch.from_numpy(state_weight[index]).to(device)
            output = model(xb)
            regression = F.smooth_l1_loss(
                output["mean"], yb, reduction="none"
            ).mean(-1)
            dual_loss = F.binary_cross_entropy_with_logits(
                output["dual_logit"], dualb, reduction="none"
            )
            harm_loss = F.binary_cross_entropy_with_logits(
                output["harm_logit"],
                harmb,
                reduction="none",
                pos_weight=torch.tensor(3.0, device=device),
            )
            loss = (
                weight
                * (0.50 * regression + 1.25 * dual_loss + 2.0 * harm_loss)
            ).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
    model.eval()
    return model, stats


def predict(
    model: StateGate,
    stats: dict[str, np.ndarray],
    feature: np.ndarray,
    *,
    device: str,
) -> dict[str, np.ndarray]:
    x = (
        (feature - stats["feature_mean"]) / stats["feature_scale"]
    ).astype(np.float32)
    with torch.inference_mode():
        output = model(torch.from_numpy(x).to(device))
    mean = (
        output["mean"].cpu().numpy() * stats["target_scale"]
        + stats["target_mean"]
    )
    return {
        "mean": mean.astype(np.float32),
        "dual_logit": output["dual_logit"].cpu().numpy().astype(np.float32),
        "harm_logit": output["harm_logit"].cpu().numpy().astype(np.float32),
    }


def selected_for_policy(
    prediction: dict[str, np.ndarray],
    prediction_std: np.ndarray,
    challenger: np.ndarray,
    policy: dict[str, float],
) -> np.ndarray:
    p_dual = sigmoid(prediction["dual_logit"])
    p_harm = sigmoid(prediction["harm_logit"])
    mean = prediction["mean"]
    accepted = (
        (p_dual >= float(policy["dual_threshold"]))
        & (p_harm <= float(policy["harm_threshold"]))
        & (
            p_dual - p_harm
            >= float(policy["dual_harm_gap_threshold"])
        )
        & (
            mean[:, 0]
            + float(policy["uncertainty_scale"]) * prediction_std[:, 0]
            < -float(policy["ate_margin"])
        )
        & (
            mean[:, 3]
            + float(policy["uncertainty_scale"]) * prediction_std[:, 3]
            < -float(policy["objective_margin"])
        )
    )
    return np.where(accepted, challenger, 0).astype(np.int64)


def method_summary(
    selected: np.ndarray,
    delta: np.ndarray,
    absolute: np.ndarray,
) -> dict[str, Any]:
    promote = selected != 0
    row = np.arange(len(selected))
    chosen_delta = np.zeros((len(selected), 4), dtype=np.float32)
    chosen_delta[promote] = delta[
        row[promote], selected[promote] - 1
    ]
    chosen_absolute = absolute[row, selected]
    count = int(promote.sum())
    dual_count = int(
        (
            (chosen_delta[promote, 0] < -1.0e-7)
            & (chosen_delta[promote, 3] < -1.0e-7)
        ).sum()
    )
    harm_count = int(
        (
            (chosen_delta[promote, 0] > 1.0e-7)
            | (chosen_delta[promote, 3] > 1.0e-7)
        ).sum()
    )
    return {
        "promotions": count,
        "coverage": float(count / len(selected)),
        "dual_precision": (
            float(dual_count / count) if count else None
        ),
        "dual_wilson_lower_95": wilson_lower(dual_count, count),
        "harm_rate": float(harm_count / count) if count else None,
        "harm_count": harm_count,
        "absolute": {
            name: {
                "mean_m": float(chosen_absolute[:, metric].mean()),
                "p90_m": float(
                    np.quantile(chosen_absolute[:, metric], 0.9)
                ),
            }
            for metric, name in enumerate(METRIC_NAMES)
        },
    }


def optional(value: float | None, fallback: float) -> float:
    return fallback if value is None else float(value)


def main() -> None:
    args = parse_args()
    result_path = args.output_dir / "result.json"
    if result_path.exists():
        raise FileExistsError(result_path)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    v4 = json.loads(args.v4_result.read_text(encoding="utf-8"))
    cohorts = []
    for value in v4["inputs"]:
        cohort = load_cohort(
            name=value["name"],
            feature_path=Path(value["features"]),
            score_path=Path(value["scores"]),
            result_path=Path(value["result"]),
        )
        feature, challenger, record_key = state_feature_and_challenger(
            Path(value["features"]), Path(value["scores"])
        )
        if not np.array_equal(record_key, cohort["record_key"]):
            raise RuntimeError("state gate feature identity mismatch")
        row = np.arange(len(challenger))
        target = cohort["delta"][row, challenger - 1]
        cohort.update(
            {
                "state_feature": feature,
                "challenger": challenger,
                "state_target": target,
                "state_dual": (
                    (target[:, 0] < -1.0e-7)
                    & (target[:, 3] < -1.0e-7)
                ).astype(np.float32),
                "state_harm": (
                    (target[:, 0] > 1.0e-7)
                    | (target[:, 3] > 1.0e-7)
                ).astype(np.float32),
                "baseline_metrics": json.loads(
                    Path(value["result"]).read_text(encoding="utf-8")
                )["methods"]["universal_d_cem_false"],
            }
        )
        cohorts.append(cohort)
    feature = np.concatenate([value["state_feature"] for value in cohorts])
    target = np.concatenate([value["state_target"] for value in cohorts])
    dual = np.concatenate([value["state_dual"] for value in cohorts])
    harm = np.concatenate([value["state_harm"] for value in cohorts])
    challenger = np.concatenate([value["challenger"] for value in cohorts])
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
    config = ModelConfig(input_dim=feature.shape[1])
    member_predictions = []
    fold_records = []
    for member in range(args.ensemble_members):
        oof = {
            "mean": np.empty((len(feature), 4), dtype=np.float32),
            "dual_logit": np.empty(len(feature), dtype=np.float32),
            "harm_logit": np.empty(len(feature), dtype=np.float32),
        }
        for fold in range(args.folds):
            train_index = np.flatnonzero(fold_id != fold)
            heldout_index = np.flatnonzero(fold_id == fold)
            seed = args.seed + member * 1009 + fold
            model, stats = train_model(
                feature=feature,
                target=target,
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

    policies = []
    for uncertainty_scale in (0.0, 0.5, 1.0, 1.5, 2.0):
        for dual_threshold in np.linspace(0.20, 0.90, 15):
            for harm_threshold in (
                0.05,
                0.10,
                0.15,
                0.20,
                0.25,
                0.30,
                0.40,
                0.50,
                0.60,
            ):
                for gap_threshold in (-0.20, -0.10, 0.0, 0.10, 0.20):
                    for ate_margin in (0.0, 0.005, 0.010, 0.020):
                        for objective_margin in (0.0, 0.005, 0.010, 0.020):
                            policy = {
                                "uncertainty_scale": uncertainty_scale,
                                "dual_threshold": float(dual_threshold),
                                "harm_threshold": harm_threshold,
                                "dual_harm_gap_threshold": gap_threshold,
                                "ate_margin": ate_margin,
                                "objective_margin": objective_margin,
                            }
                            selected = selected_for_policy(
                                prediction,
                                mean_std,
                                challenger,
                                policy,
                            )
                            per_cohort = {}
                            strict = True
                            quality = True
                            penalty = 0.0
                            coverage_sum = 0.0
                            for cohort in cohorts:
                                mask = cohort_name == cohort["name"]
                                summary = method_summary(
                                    selected[mask],
                                    cohort["delta"],
                                    cohort["absolute_final"],
                                )
                                checks = {
                                    "coverage_at_least_0.40": (
                                        summary["coverage"] >= 0.40
                                    ),
                                    "dual_at_least_0.90": (
                                        optional(
                                            summary["dual_precision"], 0.0
                                        )
                                        >= 0.90
                                    ),
                                    "wilson_at_least_0.80": (
                                        optional(
                                            summary[
                                                "dual_wilson_lower_95"
                                            ],
                                            0.0,
                                        )
                                        >= 0.80
                                    ),
                                    "harm_at_most_0.10": (
                                        optional(
                                            summary["harm_rate"], 1.0
                                        )
                                        <= 0.10
                                    ),
                                }
                                baseline = cohort["baseline_metrics"]
                                quality_checks = {
                                    "means_not_worse": all(
                                        summary["absolute"][name]["mean_m"]
                                        <= baseline["absolute"][name]["mean_m"]
                                        + 1.0e-12
                                        for name in METRIC_NAMES
                                    ),
                                    "p90_not_worse": all(
                                        summary["absolute"][name]["p90_m"]
                                        <= baseline["absolute"][name]["p90_m"]
                                        + 1.0e-12
                                        for name in METRIC_NAMES
                                    ),
                                }
                                strict &= all(checks.values())
                                quality &= all(quality_checks.values())
                                coverage = summary["coverage"]
                                dual_value = optional(
                                    summary["dual_precision"], 0.0
                                )
                                wilson = optional(
                                    summary["dual_wilson_lower_95"], 0.0
                                )
                                harm_value = optional(
                                    summary["harm_rate"], 1.0
                                )
                                penalty += (
                                    12.0 * max(0.0, 0.40 - coverage)
                                    + 10.0
                                    * max(0.0, 0.90 - dual_value)
                                    + 10.0 * max(0.0, 0.80 - wilson)
                                    + 10.0
                                    * max(0.0, harm_value - 0.10)
                                    + (
                                        0.0
                                        if quality_checks["means_not_worse"]
                                        else 0.25
                                    )
                                    + (
                                        0.0
                                        if quality_checks["p90_not_worse"]
                                        else 0.25
                                    )
                                )
                                coverage_sum += coverage
                                per_cohort[cohort["name"]] = {
                                    "summary": summary,
                                    "safety_checks": checks,
                                    "quality_checks": quality_checks,
                                }
                            policies.append(
                                {
                                    "policy": policy,
                                    "selected": selected,
                                    "per_cohort": per_cohort,
                                    "strict_safety_all_cohorts": strict,
                                    "quality_all_cohorts": quality,
                                    "strict_feasible": strict and quality,
                                    "penalty": float(penalty),
                                    "coverage_sum": float(coverage_sum),
                                }
                            )
    feasible = [value for value in policies if value["strict_feasible"]]
    if feasible:
        best = max(
            feasible,
            key=lambda value: (
                value["coverage_sum"],
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
        challenger_index=challenger,
        selected_index=best["selected"],
    )
    report = {
        "schema": "recon_cem_state_gate_v6",
        "status": "route_disjoint_oof_development_complete",
        "evidence_role": "development_only_three_spent_cohorts",
        "cem": True,
        "states": int(len(feature)),
        "routes": int(len(np.unique(group))),
        "feature_dim": int(feature.shape[1]),
        "feature_contract": (
            "DINOv3 static context + raw LTWM incumbent/challenger/delta + "
            "frozen scorer distribution + spacing-normalized geometry"
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
        },
        "oof_predictions": str(oof_path),
        "oof_predictions_sha256": file_sha256(oof_path),
        "source_v4_result": str(args.v4_result),
        "source_v4_result_sha256": file_sha256(args.v4_result),
        "full_checkpoint_frozen": False,
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
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
