#!/usr/bin/env python3
"""Train a route-disjoint CEM candidate calibrator on three spent RECON cohorts."""

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
from tools.evaluation.evaluate_scand_fresh_outer_v1 import method_metrics  # noqa: E402
from tools.scoring.train_recon_dense_meta_dinov3_v1 import (  # noqa: E402
    file_sha256,
    write_json,
    write_npz,
)


METRIC_NAMES = ("ate", "rpe", "endpoint", "objective")


@dataclass(frozen=True)
class ModelConfig:
    input_dim: int
    hidden_dim: int = 96
    dropout: float = 0.10


class CandidateCalibrator(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.trunk = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
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
    parser.add_argument(
        "--feature-paths", type=Path, nargs=3, required=True
    )
    parser.add_argument(
        "--score-paths", type=Path, nargs=3, required=True
    )
    parser.add_argument(
        "--result-paths", type=Path, nargs=3, required=True
    )
    parser.add_argument(
        "--cohort-names",
        nargs=3,
        default=("cem_dev1", "cem_confirm2", "fresh_spent3"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--batch-states", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=8.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--seed", type=int, default=20260802)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def fold_for_group(group: str, folds: int) -> int:
    digest = hashlib.sha256(f"cem-calibrator-v4|{group}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % folds


def sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -30.0, 30.0)))


def load_baseline_from_result(result: dict[str, Any]) -> np.ndarray:
    outcome_path = Path(result["outcomes"])
    with np.load(outcome_path, allow_pickle=False) as payload:
        if "universal_d_cem_false_selected_index" in payload.files:
            return payload[
                "universal_d_cem_false_selected_index"
            ].astype(np.int64)
    freeze = json.loads(Path(result["freeze"]).read_text(encoding="utf-8"))
    with np.load(freeze["selected_index"], allow_pickle=False) as payload:
        return payload[
            "universal_d_cem_false_selected_index"
        ].astype(np.int64)


def load_cohort(
    *,
    name: str,
    feature_path: Path,
    score_path: Path,
    result_path: Path,
) -> dict[str, Any]:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if (
        result.get("domain") != "recon"
        or not result.get("cem")
        or not (
            result.get("cem_cal_now_spent")
            or result.get("confirmation_now_spent")
            or result.get("fresh_confirmation_now_spent")
        )
    ):
        raise RuntimeError(f"cohort is not spent RECON CEM evidence: {result_path}")
    with np.load(feature_path, allow_pickle=False) as payload:
        candidates = payload["candidates"].astype(np.float32)
        actions = payload["actions_h4"].astype(np.float32)
        record_key = payload["record_key"].astype(str)
    with np.load(score_path, allow_pickle=False) as payload:
        prediction = {
            "energy": payload["energy"].astype(np.float32),
            "mean": payload["mean"].astype(np.float32),
            "std": payload["std"].astype(np.float32),
            "p_dual": payload["dual_probability"].astype(np.float32),
            "p_harm": payload["harm_probability"].astype(np.float32),
            "rank": payload["rank_logit"].astype(np.float32),
            "valid": payload["valid"].astype(bool),
        }
    with np.load(result["outcomes"], allow_pickle=False) as payload:
        if not np.array_equal(payload["record_key"].astype(str), record_key):
            raise RuntimeError("CEM feature/outcome identity mismatch")
        absolute = payload["combined_absolute_metrics"].astype(np.float32)
        route = payload["route_group"].astype(str)
        trajectory = payload["trajectory"].astype(str)
    if (
        candidates.shape != (100, 17, 4, 2)
        or actions.shape != (100, 17, 4, 3)
        or absolute.shape != (100, 33, 4)
        or prediction["energy"].shape != (100, 17)
    ):
        raise RuntimeError("CEM candidate calibrator input shape changed")

    action = actions[:, 1:].copy()
    action[..., :2] /= 0.25
    path = candidates[:, 1:].astype(np.float32)
    energy = np.nan_to_num(
        prediction["energy"][:, 1:],
        nan=10.0,
        posinf=10.0,
        neginf=-10.0,
    )
    feature = np.concatenate(
        (
            prediction["mean"][:, 1:],
            np.log1p(prediction["std"][:, 1:]),
            prediction["p_dual"][:, 1:, None],
            prediction["p_harm"][:, 1:, None],
            prediction["rank"][:, 1:, None],
            energy[..., None],
            action.reshape(100, 16, 12),
            path.reshape(100, 16, 8),
            prediction["valid"][:, 1:, None].astype(np.float32),
        ),
        axis=-1,
    ).astype(np.float32)
    final_absolute = np.concatenate(
        (absolute[:, :1], absolute[:, 17:]), axis=1
    )
    delta = final_absolute[:, 1:] - final_absolute[:, :1]
    dual = (delta[..., 0] < -1.0e-7) & (delta[..., 3] < -1.0e-7)
    harm = (delta[..., 0] > 1.0e-7) | (delta[..., 3] > 1.0e-7)
    baseline = load_baseline_from_result(result)
    return {
        "name": name,
        "feature": feature,
        "delta": delta.astype(np.float32),
        "dual": dual.astype(np.float32),
        "harm": harm.astype(np.float32),
        "valid": prediction["valid"][:, 1:].astype(bool),
        "absolute_final": final_absolute,
        "absolute_combined": absolute,
        "baseline": baseline,
        "record_key": record_key,
        "route": route,
        "trajectory": trajectory,
        "feature_path": feature_path,
        "score_path": score_path,
        "result_path": result_path,
    }


def normalization(
    feature: np.ndarray, delta: np.ndarray, index: np.ndarray
) -> dict[str, np.ndarray]:
    value = feature[index].reshape(-1, feature.shape[-1])
    target = delta[index].reshape(-1, 4)
    return {
        "feature_mean": value.mean(0).astype(np.float32),
        "feature_scale": np.maximum(value.std(0), 1.0e-4).astype(np.float32),
        "target_mean": target.mean(0).astype(np.float32),
        "target_scale": np.maximum(target.std(0), 1.0e-4).astype(np.float32),
    }


def train_model(
    *,
    feature: np.ndarray,
    delta: np.ndarray,
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
) -> tuple[CandidateCalibrator, dict[str, np.ndarray]]:
    set_seed(seed)
    stats = normalization(feature, delta, train_index)
    x = (
        (feature - stats["feature_mean"]) / stats["feature_scale"]
    ).astype(np.float32)
    y = (
        (delta - stats["target_mean"]) / stats["target_scale"]
    ).astype(np.float32)
    model = CandidateCalibrator(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    train_groups, counts = np.unique(groups[train_index], return_counts=True)
    group_weight = {
        group: float(1.0 / count)
        for group, count in zip(train_groups, counts)
    }
    state_weight = np.asarray(
        [group_weight.get(group, 0.0) for group in groups], dtype=np.float32
    )
    state_weight[train_index] *= float(
        len(train_index) / state_weight[train_index].sum()
    )
    dual_positive = float(
        np.clip(
            (1.0 - dual[train_index].mean())
            / max(dual[train_index].mean(), 1.0e-4),
            0.5,
            4.0,
        )
    )
    harm_positive = float(
        np.clip(
            (1.0 - harm[train_index].mean())
            / max(harm[train_index].mean(), 1.0e-4),
            0.5,
            4.0,
        )
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
            weight = torch.from_numpy(state_weight[index]).to(device)[:, None]
            output = model(xb)
            regression = F.smooth_l1_loss(
                output["mean"], yb, reduction="none"
            ).mean(-1)
            dual_loss = F.binary_cross_entropy_with_logits(
                output["dual_logit"],
                dualb,
                reduction="none",
                pos_weight=torch.tensor(dual_positive, device=device),
            )
            harm_loss = F.binary_cross_entropy_with_logits(
                output["harm_logit"],
                harmb,
                reduction="none",
                pos_weight=torch.tensor(harm_positive, device=device),
            )
            pred_delta = (
                output["mean"]
                * torch.from_numpy(stats["target_scale"]).to(device)
                + torch.from_numpy(stats["target_mean"]).to(device)
            )
            pred_utility = (
                pred_delta[..., 0]
                + 0.25 * pred_delta[..., 1]
                + 0.25 * torch.relu(pred_delta[..., 2])
                + 0.50 * pred_delta[..., 3]
            )
            true_utility = (
                torch.from_numpy(delta[index]).to(device)[..., 0]
                + 0.25 * torch.from_numpy(delta[index]).to(device)[..., 1]
                + 0.25
                * torch.relu(
                    torch.from_numpy(delta[index]).to(device)[..., 2]
                )
                + 0.50 * torch.from_numpy(delta[index]).to(device)[..., 3]
            )
            distribution = torch.softmax(-true_utility / 0.03, dim=1)
            listwise = -(distribution * torch.log_softmax(-pred_utility, dim=1)).sum(1)
            candidate_loss = (
                regression + 1.2 * dual_loss + 1.5 * harm_loss
            )
            loss = (
                (candidate_loss * weight).mean()
                + 0.35 * (listwise * weight[:, 0]).mean()
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
    model.eval()
    return model, stats


def predict(
    model: CandidateCalibrator,
    stats: dict[str, np.ndarray],
    feature: np.ndarray,
    *,
    device: str,
) -> dict[str, np.ndarray]:
    x = (
        (feature - stats["feature_mean"]) / stats["feature_scale"]
    ).astype(np.float32)
    values: dict[str, list[np.ndarray]] = {
        "mean": [],
        "dual_logit": [],
        "harm_logit": [],
    }
    with torch.inference_mode():
        for start in range(0, len(x), 64):
            output = model(torch.from_numpy(x[start : start + 64]).to(device))
            mean = (
                output["mean"].cpu().numpy() * stats["target_scale"]
                + stats["target_mean"]
            )
            values["mean"].append(mean.astype(np.float32))
            values["dual_logit"].append(
                output["dual_logit"].cpu().numpy().astype(np.float32)
            )
            values["harm_logit"].append(
                output["harm_logit"].cpu().numpy().astype(np.float32)
            )
    return {key: np.concatenate(value) for key, value in values.items()}


def selected_for_policy(
    prediction: dict[str, np.ndarray],
    valid: np.ndarray,
    policy: dict[str, float],
) -> np.ndarray:
    p_dual = sigmoid(prediction["dual_logit"])
    p_harm = sigmoid(prediction["harm_logit"])
    mean = prediction["mean"]
    utility = (
        mean[..., 0]
        + 0.25 * mean[..., 1]
        + 0.25 * np.maximum(mean[..., 2], 0.0)
        + 0.50 * mean[..., 3]
        + 0.15 * p_harm
        - 0.05 * p_dual
    )
    allowed = (
        valid
        & (p_dual >= float(policy["dual_threshold"]))
        & (p_harm <= float(policy["harm_threshold"]))
        & (mean[..., 0] < -float(policy["ate_margin"]))
        & (mean[..., 3] < -float(policy["objective_margin"]))
    )
    score = np.where(allowed, utility, np.inf)
    candidate = score.argmin(axis=1)
    exists = np.isfinite(score[np.arange(len(score)), candidate])
    return np.where(exists, candidate + 1, 0).astype(np.int64)


def safety(selected: np.ndarray, delta: np.ndarray) -> dict[str, Any]:
    promote = selected != 0
    chosen = np.zeros((len(selected), 4), dtype=np.float32)
    chosen[promote] = delta[
        np.arange(len(selected))[promote], selected[promote] - 1
    ]
    count = int(promote.sum())
    dual = int(
        (
            (chosen[promote, 0] < -1.0e-7)
            & (chosen[promote, 3] < -1.0e-7)
        ).sum()
    )
    harm = int(
        (
            (chosen[promote, 0] > 1.0e-7)
            | (chosen[promote, 3] > 1.0e-7)
        ).sum()
    )
    return {
        "promotions": count,
        "coverage": float(count / len(selected)),
        "dual_precision": float(dual / count) if count else None,
        "dual_wilson_lower_95": wilson_lower(dual, count),
        "harm_rate": float(harm / count) if count else None,
        "harm_count": harm,
    }


def absolute_summary(
    selected: np.ndarray, absolute: np.ndarray
) -> dict[str, dict[str, float]]:
    chosen = absolute[np.arange(len(selected)), selected]
    return {
        name: {
            "mean_m": float(chosen[:, index].mean()),
            "p90_m": float(np.quantile(chosen[:, index], 0.9)),
        }
        for index, name in enumerate(METRIC_NAMES)
    }


def main() -> None:
    args = parse_args()
    result_path = args.output_dir / "result.json"
    if result_path.exists():
        raise FileExistsError(result_path)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.folds < 3:
        raise ValueError("route-disjoint OOF needs at least three folds")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cohorts = [
        load_cohort(
            name=name,
            feature_path=feature,
            score_path=score,
            result_path=result,
        )
        for name, feature, score, result in zip(
            args.cohort_names,
            args.feature_paths,
            args.score_paths,
            args.result_paths,
        )
    ]
    feature = np.concatenate([value["feature"] for value in cohorts])
    delta = np.concatenate([value["delta"] for value in cohorts])
    dual = np.concatenate([value["dual"] for value in cohorts])
    harm = np.concatenate([value["harm"] for value in cohorts])
    valid = np.concatenate([value["valid"] for value in cohorts])
    group = np.concatenate(
        [
            np.asarray([f"{value['name']}::{route}" for route in value["route"]])
            for value in cohorts
        ]
    )
    cohort_name = np.concatenate(
        [np.full(100, value["name"], dtype="<U32") for value in cohorts]
    )
    config = ModelConfig(input_dim=feature.shape[-1])
    fold_id = np.asarray(
        [fold_for_group(value, args.folds) for value in group], dtype=np.int64
    )
    oof = {
        "mean": np.empty((len(feature), 16, 4), dtype=np.float32),
        "dual_logit": np.empty((len(feature), 16), dtype=np.float32),
        "harm_logit": np.empty((len(feature), 16), dtype=np.float32),
    }
    fold_records = []
    for fold in range(args.folds):
        train_index = np.flatnonzero(fold_id != fold)
        heldout_index = np.flatnonzero(fold_id == fold)
        if not len(train_index) or not len(heldout_index):
            raise RuntimeError("empty CEM OOF fold")
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
            seed=args.seed + fold,
            device=args.device,
        )
        prediction = predict(
            model, stats, feature[heldout_index], device=args.device
        )
        for key in oof:
            oof[key][heldout_index] = prediction[key]
        fold_records.append(
            {
                "fold": fold,
                "train_states": int(len(train_index)),
                "heldout_states": int(len(heldout_index)),
                "train_routes": int(len(np.unique(group[train_index]))),
                "heldout_routes": int(len(np.unique(group[heldout_index]))),
            }
        )
        del model
        if args.device == "cuda":
            torch.cuda.empty_cache()

    policies = []
    for dual_threshold in np.linspace(0.50, 0.95, 10):
        for harm_threshold in (0.05, 0.10, 0.15, 0.20, 0.30, 0.40):
            for ate_margin in (0.0, 0.005, 0.010, 0.020):
                for objective_margin in (0.0, 0.005, 0.010, 0.020):
                    policy = {
                        "dual_threshold": float(dual_threshold),
                        "harm_threshold": float(harm_threshold),
                        "ate_margin": ate_margin,
                        "objective_margin": objective_margin,
                    }
                    selected = selected_for_policy(oof, valid, policy)
                    per_cohort = {}
                    strict = True
                    quality = True
                    penalty = 0.0
                    for value in cohorts:
                        mask = cohort_name == value["name"]
                        local_selected = selected[mask]
                        metrics = safety(local_selected, delta[mask])
                        absolute = absolute_summary(
                            local_selected, value["absolute_final"]
                        )
                        baseline = value["result_path"]
                        baseline_metrics = json.loads(
                            Path(baseline).read_text(encoding="utf-8")
                        )["methods"]["universal_d_cem_false"]
                        checks = {
                            "coverage_at_least_0.40": metrics["coverage"] >= 0.40,
                            "dual_at_least_0.90": (
                                metrics["dual_precision"] is not None
                                and metrics["dual_precision"] >= 0.90
                            ),
                            "wilson_at_least_0.80": (
                                metrics["dual_wilson_lower_95"] is not None
                                and metrics["dual_wilson_lower_95"] >= 0.80
                            ),
                            "harm_at_most_0.10": (
                                metrics["harm_rate"] is not None
                                and metrics["harm_rate"] <= 0.10
                            ),
                        }
                        quality_checks = {
                            "means_not_worse": all(
                                absolute[name]["mean_m"]
                                <= baseline_metrics["absolute"][name]["mean_m"]
                                + 1.0e-12
                                for name in METRIC_NAMES
                            ),
                            "p90_not_worse": all(
                                absolute[name]["p90_m"]
                                <= baseline_metrics["absolute"][name]["p90_m"]
                                + 1.0e-12
                                for name in METRIC_NAMES
                            ),
                        }
                        strict &= all(checks.values())
                        quality &= all(quality_checks.values())
                        penalty += (
                            5.0 * max(0.0, 0.40 - metrics["coverage"])
                            + 5.0
                            * max(
                                0.0,
                                0.90 - (metrics["dual_precision"] or 0.0),
                            )
                            + 5.0
                            * max(
                                0.0,
                                0.80
                                - (metrics["dual_wilson_lower_95"] or 0.0),
                            )
                            + 5.0
                            * max(0.0, (metrics["harm_rate"] or 1.0) - 0.10)
                        )
                        per_cohort[value["name"]] = {
                            "safety": metrics,
                            "absolute": absolute,
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
                        }
                    )
    feasible = [value for value in policies if value["strict_feasible"]]
    if feasible:
        best = max(
            feasible,
            key=lambda value: (
                sum(
                    item["safety"]["coverage"]
                    for item in value["per_cohort"].values()
                ),
                -value["penalty"],
            ),
        )
        tier = "strict_all_three_oof_cohorts"
    else:
        best = min(
            policies,
            key=lambda value: (
                value["penalty"],
                -sum(
                    item["safety"]["coverage"]
                    for item in value["per_cohort"].values()
                ),
            ),
        )
        tier = "best_available_oof"

    checkpoint_records = []
    full_index = np.arange(len(feature))
    for member_index, seed in enumerate(
        (args.seed + 101, args.seed + 202, args.seed + 303)
    ):
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
        checkpoint = args.output_dir / f"member_{member_index}.pt"
        torch.save(
            {
                "schema": "recon_cem_candidate_calibrator_v4_member",
                "config": asdict(config),
                "feature_mean": stats["feature_mean"],
                "feature_scale": stats["feature_scale"],
                "target_mean": stats["target_mean"],
                "target_scale": stats["target_scale"],
                "model": {
                    key: value.detach().cpu()
                    for key, value in model.state_dict().items()
                },
                "seed": seed,
                "training_states": len(feature),
                "training_routes": len(np.unique(group)),
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

    oof_path = args.output_dir / "oof_predictions.npz"
    write_npz(
        oof_path,
        record_key=np.concatenate([value["record_key"] for value in cohorts]),
        cohort=cohort_name,
        route_group=group,
        fold_id=fold_id,
        mean=oof["mean"],
        dual_logit=oof["dual_logit"],
        harm_logit=oof["harm_logit"],
        selected_index=best["selected"],
    )
    report = {
        "schema": "recon_cem_candidate_calibrator_v4",
        "status": "route_disjoint_oof_trained_and_full_ensemble_frozen",
        "evidence_role": "development_only_three_spent_cohorts",
        "cem": True,
        "states": len(feature),
        "routes": int(len(np.unique(group))),
        "candidate_training_examples": int(len(feature) * 16),
        "feature_dim": int(feature.shape[-1]),
        "feature_contract": (
            "frozen universal-raw predictions and uncertainty + spacing-"
            "normalized action geometry; no native scorer-v2 predictions"
        ),
        "config": {
            **asdict(config),
            "folds": args.folds,
            "epochs": args.epochs,
            "batch_states": args.batch_states,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
        },
        "folds": fold_records,
        "policy_search": {
            "tier": tier,
            "strict_feasible_policies": len(feasible),
            "searched_policies": len(policies),
            "policy": best["policy"],
            "per_cohort": best["per_cohort"],
            "strict_safety_all_cohorts": best["strict_safety_all_cohorts"],
            "quality_all_cohorts": best["quality_all_cohorts"],
            "strict_feasible": best["strict_feasible"],
            "penalty": best["penalty"],
        },
        "checkpoints": checkpoint_records,
        "oof_predictions": str(oof_path),
        "oof_predictions_sha256": file_sha256(oof_path),
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
    }
    write_json(result_path, report)
    print(
        json.dumps(
            {
                "result": str(result_path),
                "result_sha256": file_sha256(result_path),
                "policy_search": report["policy_search"],
                "next_confirmation_accessed": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
