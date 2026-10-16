#!/usr/bin/env python3
"""Train a cohort-held-out tail-risk gate with spatial DINOv3 latents.

This stage deliberately solves the safe fixed-pool base decision before any
CEM replacement is calibrated.  It consumes the already frozen DINOv3 token
caches, so no image encoding or outcome-dependent feature generation occurs.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
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

from tools.scoring.train_recon_cem_combined_pool_v7 import (  # noqa: E402
    load_combined_cohort,
    resolve_base_feature,
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
    parser.add_argument("--v4-result", type=Path, required=True)
    parser.add_argument("--replication-result", type=Path, required=True)
    parser.add_argument("--replication-base-features", type=Path, required=True)
    parser.add_argument("--replication-cem-features", type=Path, required=True)
    parser.add_argument("--replication-cem-scores", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=260)
    parser.add_argument("--batch-states", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=5.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-3)
    parser.add_argument("--ensemble-members", type=int, default=3)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--seed", type=int, default=20260817)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def latent_path_from_feature(feature_path: Path) -> Path:
    with np.load(feature_path, allow_pickle=False) as payload:
        metadata = json.loads(str(payload["metadata"]))
    path = Path(metadata["latent_cache"])
    if file_sha256(path) != metadata["latent_cache_sha256"]:
        raise RuntimeError("latent cache SHA mismatch")
    return path


def spatial_latent_feature(latent_path: Path) -> np.ndarray:
    with np.load(latent_path, allow_pickle=False) as payload:
        history = payload["history"].astype(np.float32)
        goal = payload["goal"].astype(np.float32)
    if history.shape[1:] != (4, 256, 384) or goal.shape[1:] != (256, 384):
        raise RuntimeError("DINOv3 token layout changed")
    history /= np.maximum(
        np.linalg.norm(history, axis=-1, keepdims=True), 1.0e-6
    )
    goal /= np.maximum(
        np.linalg.norm(goal, axis=-1, keepdims=True), 1.0e-6
    )
    same = (history * goal[:, None]).sum(-1).reshape(
        len(history), 4, 16, 16
    )
    same_pool = same.reshape(
        len(history), 4, 4, 4, 4, 4
    ).mean(axis=(3, 5)).reshape(len(history), -1)

    match_features = []
    current_similarity = None
    source_y, source_x = np.divmod(np.arange(256), 16)
    for frame in range(4):
        similarity = np.einsum(
            "nid,njd->nij",
            history[:, frame],
            goal,
            optimize=True,
        )
        if frame == 3:
            current_similarity = similarity
        maximum = similarity.max(axis=2)
        second = np.partition(similarity, -2, axis=2)[:, :, -2]
        target = similarity.argmax(axis=2)
        target_y, target_x = np.divmod(target, 16)
        dx = target_x - source_x[None]
        dy = target_y - source_y[None]
        match_features.append(
            np.stack(
                (
                    maximum.mean(1),
                    maximum.std(1),
                    np.quantile(maximum, 0.10, axis=1),
                    np.quantile(maximum, 0.50, axis=1),
                    np.quantile(maximum, 0.90, axis=1),
                    (maximum - second).mean(1),
                    (maximum - second).std(1),
                    dx.mean(1),
                    dx.std(1),
                    np.quantile(dx, 0.10, axis=1),
                    np.quantile(dx, 0.90, axis=1),
                    dy.mean(1),
                    dy.std(1),
                    np.quantile(dy, 0.10, axis=1),
                    np.quantile(dy, 0.90, axis=1),
                ),
                axis=1,
            )
        )
    if current_similarity is None:
        raise RuntimeError("current-goal correlation missing")
    current_grid = current_similarity.reshape(
        len(history), 16, 16, 16, 16
    )
    offset_values = []
    for dy in range(-4, 5):
        source_y_slice = slice(max(0, -dy), min(16, 16 - dy))
        target_y_slice = slice(max(0, dy), min(16, 16 + dy))
        for dx in range(-4, 5):
            source_x_slice = slice(max(0, -dx), min(16, 16 - dx))
            target_x_slice = slice(max(0, dx), min(16, 16 + dx))
            value = current_grid[
                :,
                source_y_slice,
                source_x_slice,
                target_y_slice,
                target_x_slice,
            ]
            diagonal = np.einsum("nxyxy->nxy", value, optimize=True)
            offset_values.append(diagonal.mean(axis=(1, 2)))
    offset_feature = np.stack(offset_values, axis=1)
    global_progress = np.concatenate(
        (
            same.mean(axis=(2, 3)),
            same.std(axis=(2, 3)),
            np.quantile(same, 0.10, axis=(2, 3)),
            np.quantile(same, 0.90, axis=(2, 3)),
        ),
        axis=1,
    )
    result = np.concatenate(
        (
            same_pool,
            np.concatenate(match_features, axis=1),
            offset_feature,
            global_progress,
        ),
        axis=1,
    ).astype(np.float32)
    if result.shape[1] != 221 or not np.isfinite(result).all():
        raise RuntimeError(
            f"spatial latent feature contract failed: {result.shape}"
        )
    return result


def resolve_base_score(result_path: Path) -> Path:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    freeze_path = Path(
        result.get("selection_freeze") or result.get("freeze")
    )
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if "base_scores" in freeze:
        return Path(freeze["base_scores"])
    if "cem_freeze" in freeze:
        nested = json.loads(
            Path(freeze["cem_freeze"]).read_text(encoding="utf-8")
        )
        return Path(nested["base_scores"])
    raise RuntimeError("base score path unavailable")


def compact_base_feature(
    base_feature_path: Path,
    base_score_path: Path,
    latent_feature: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    with np.load(base_feature_path, allow_pickle=False) as payload:
        static = payload["static_features"][:, 1:].astype(np.float32)
        raw = payload["raw_ltwm_features"].astype(np.float32)
        action = payload["actions_h4"].astype(np.float32)
        candidate = payload["candidates"].astype(np.float32)
        support = payload["majority_support"][:, 1:].astype(bool)
        record = payload["record_key"].astype(str)
    with np.load(base_score_path, allow_pickle=False) as payload:
        score_record = payload["record_key"].astype(str)
        if not np.array_equal(record, score_record):
            raise RuntimeError("base feature/score identity mismatch")
        universal = np.concatenate(
            (
                payload["mean"][:, 1:].astype(np.float32),
                payload["std"][:, 1:].astype(np.float32),
                payload["dual_probability"][:, 1:, None].astype(
                    np.float32
                ),
                payload["harm_probability"][:, 1:, None].astype(
                    np.float32
                ),
                payload["rank_logit"][:, 1:, None].astype(np.float32),
            ),
            axis=2,
        )
    action_candidate = action[:, 1:]
    action_delta = action_candidate - action[:, :1]
    path_candidate = candidate[:, 1:]
    path_delta = path_candidate - candidate[:, :1]
    state_latent = np.repeat(
        latent_feature[:, None], 16, axis=1
    )
    feature = np.concatenate(
        (
            static,
            raw[:, 1:],
            raw[:, 1:] - raw[:, :1],
            action_candidate.reshape(len(record), 16, -1),
            action_delta.reshape(len(record), 16, -1),
            path_candidate.reshape(len(record), 16, -1),
            path_delta.reshape(len(record), 16, -1),
            universal,
            support[..., None].astype(np.float32),
            state_latent,
        ),
        axis=2,
    ).astype(np.float32)
    return feature, support


@dataclass(frozen=True)
class TailConfig:
    input_dim: int
    hidden_dim: int = 160
    dropout: float = 0.25


class TailRiskModel(nn.Module):
    def __init__(self, config: TailConfig) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.head = nn.Linear(config.hidden_dim, 2)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.head(self.trunk(value))


def normalization(
    feature: np.ndarray, train_index: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    value = feature[train_index].reshape(-1, feature.shape[-1])
    return (
        value.mean(0).astype(np.float32),
        np.maximum(value.std(0), 1.0e-4).astype(np.float32),
    )


def train_tail_model(
    *,
    feature: np.ndarray,
    dual: np.ndarray,
    harm: np.ndarray,
    group: np.ndarray,
    train_index: np.ndarray,
    config: TailConfig,
    args: argparse.Namespace,
    seed: int,
) -> tuple[TailRiskModel, np.ndarray, np.ndarray]:
    set_seed(seed)
    mean, scale = normalization(feature, train_index)
    normalized = ((feature - mean) / scale).astype(np.float32)
    model = TailRiskModel(config).to(args.device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    unique, counts = np.unique(group[train_index], return_counts=True)
    group_weight = {value: 1.0 / count for value, count in zip(unique, counts)}
    state_weight = np.asarray(
        [group_weight.get(value, 0.0) for value in group],
        dtype=np.float32,
    )
    state_weight[train_index] *= (
        len(train_index) / state_weight[train_index].sum()
    )
    rng = np.random.default_rng(seed)
    for _ in range(args.epochs):
        order = rng.permutation(train_index)
        model.train()
        for start in range(0, len(order), args.batch_states):
            index = order[start : start + args.batch_states]
            x = torch.from_numpy(normalized[index]).to(args.device)
            dual_target = torch.from_numpy(dual[index]).to(args.device)
            harm_target = torch.from_numpy(harm[index]).to(args.device)
            weight = torch.from_numpy(state_weight[index]).to(
                args.device
            )[:, None]
            logits = model(x)
            dual_loss = F.binary_cross_entropy_with_logits(
                logits[..., 0], dual_target, reduction="none"
            )
            dual_loss *= torch.where(
                dual_target > 0.5,
                torch.ones_like(dual_target),
                torch.full_like(dual_target, 3.0),
            )
            harm_loss = F.binary_cross_entropy_with_logits(
                logits[..., 1], harm_target, reduction="none"
            )
            harm_loss *= torch.where(
                harm_target > 0.5,
                torch.full_like(harm_target, 4.0),
                torch.ones_like(harm_target),
            )
            safety_logit = logits[..., 0] - logits[..., 1]
            chosen = safety_logit.argmax(dim=1)
            row = torch.arange(len(index), device=args.device)
            hard_dual = F.binary_cross_entropy_with_logits(
                logits[row, chosen, 0],
                dual_target[row, chosen],
                reduction="none",
            )
            hard_harm = F.binary_cross_entropy_with_logits(
                logits[row, chosen, 1],
                harm_target[row, chosen],
                reduction="none",
            )
            loss = (
                ((dual_loss + 1.5 * harm_loss) * weight).mean()
                + 3.0
                * ((hard_dual + hard_harm) * weight[:, 0]).mean()
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
    model.eval()
    return model, mean, scale


def predict_tail(
    model: TailRiskModel,
    mean: np.ndarray,
    scale: np.ndarray,
    feature: np.ndarray,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    normalized = ((feature - mean) / scale).astype(np.float32)
    with torch.inference_mode():
        logits = model(torch.from_numpy(normalized).to(device))
        probability = torch.sigmoid(logits).cpu().numpy()
    return (
        probability[..., 0].astype(np.float32),
        probability[..., 1].astype(np.float32),
    )


def calibration(
    probability: np.ndarray, target: np.ndarray
) -> dict[str, float]:
    bins = np.linspace(0.0, 1.0, 11)
    ece = 0.0
    for lower, upper in zip(bins[:-1], bins[1:]):
        mask = (probability >= lower) & (
            (probability < upper) | ((upper == 1.0) & (probability <= upper))
        )
        if mask.any():
            ece += float(mask.mean()) * abs(
                float(probability[mask].mean())
                - float(target[mask].mean())
            )
    return {
        "brier": float(np.mean((probability - target) ** 2)),
        "ece10": float(ece),
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
    cohorts.append(load_replication(args))

    feature_values = []
    latent_records = []
    for cohort in cohorts:
        base_feature = resolve_base_feature(cohort["result_path"])
        base_score = resolve_base_score(cohort["result_path"])
        latent_path = latent_path_from_feature(base_feature)
        latent = spatial_latent_feature(latent_path)
        feature, support = compact_base_feature(
            base_feature, base_score, latent
        )
        if not np.array_equal(
            support, cohort["combined_valid"][:, :16]
        ):
            raise RuntimeError("base support identity mismatch")
        feature_values.append(feature)
        latent_records.append(
            {
                "cohort": cohort["name"],
                "path": str(latent_path),
                "sha256": file_sha256(latent_path),
            }
        )
    feature = np.concatenate(feature_values)
    delta = np.concatenate(
        [value["combined_delta"][:, :16] for value in cohorts]
    )
    dual = np.concatenate(
        [value["combined_dual"][:, :16] for value in cohorts]
    )
    harm = np.concatenate(
        [value["combined_harm"][:, :16] for value in cohorts]
    )
    valid = np.concatenate(
        [value["combined_valid"][:, :16] for value in cohorts]
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
    config = TailConfig(input_dim=feature.shape[-1])
    args.output_dir.mkdir(parents=True, exist_ok=False)
    p_dual = np.empty((len(feature), 16), dtype=np.float32)
    p_harm = np.empty((len(feature), 16), dtype=np.float32)
    p_dual_std = np.empty_like(p_dual)
    p_harm_std = np.empty_like(p_harm)
    checkpoints = []
    calibration_by_fold = {}
    for fold_number, heldout_name in enumerate(np.unique(cohort_name)):
        train_index = np.flatnonzero(cohort_name != heldout_name)
        heldout_index = np.flatnonzero(cohort_name == heldout_name)
        member_dual = []
        member_harm = []
        for member in range(args.ensemble_members):
            seed = args.seed + fold_number * 10007 + member * 1009
            model, mean, scale = train_tail_model(
                feature=feature,
                dual=dual,
                harm=harm,
                group=group,
                train_index=train_index,
                config=config,
                args=args,
                seed=seed,
            )
            dual_value, harm_value = predict_tail(
                model,
                mean,
                scale,
                feature[heldout_index],
                args.device,
            )
            member_dual.append(dual_value)
            member_harm.append(harm_value)
            checkpoint = (
                args.output_dir
                / f"exclude_{fold_number}_member_{member}.pt"
            )
            torch.save(
                {
                    "schema": "recon_cem_latent_base_risk_v16_member",
                    "excluded_cohort": heldout_name,
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
                    "excluded_cohort": heldout_name,
                    "seed": seed,
                }
            )
            del model
            if args.device == "cuda":
                torch.cuda.empty_cache()
        dual_stack = np.stack(member_dual)
        harm_stack = np.stack(member_harm)
        p_dual[heldout_index] = dual_stack.mean(0)
        p_harm[heldout_index] = harm_stack.mean(0)
        p_dual_std[heldout_index] = dual_stack.std(0)
        p_harm_std[heldout_index] = harm_stack.std(0)
        calibration_by_fold[heldout_name] = {
            "dual": calibration(
                p_dual[heldout_index],
                dual[heldout_index],
            ),
            "harm": calibration(
                p_harm[heldout_index],
                harm[heldout_index],
            ),
        }

    policies = []
    for z in (0.0, 0.5, 1.0, 1.5, 2.0):
        dual_lower = p_dual - z * p_dual_std
        harm_upper = p_harm + z * p_harm_std
        for risk_weight in (0.5, 1.0, 2.0, 4.0):
            score = dual_lower - risk_weight * harm_upper
            score = np.where(valid, score, -np.inf)
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
        tier = "strict_all_four_loco_base_gate"
    else:
        best = max(
            policies,
            key=lambda value: (
                min(
                    optional(
                        item["metrics"]["dual_precision"], 0.0
                    )
                    for item in value["per_cohort"].values()
                    if item["metrics"]["coverage"] >= 0.40
                )
                if all(
                    item["metrics"]["coverage"] >= 0.40
                    for item in value["per_cohort"].values()
                )
                else -1.0,
                -max(
                    optional(item["metrics"]["harm_rate"], 1.0)
                    for item in value["per_cohort"].values()
                ),
                value["coverage_sum"],
            ),
        )
        tier = "failed_best_available_loco_base_gate"

    top40 = {}
    for name in np.unique(cohort_name):
        mask = cohort_name == name
        best_ordering = None
        for z in (0.0, 0.5, 1.0, 1.5, 2.0):
            dual_lower = p_dual[mask] - z * p_dual_std[mask]
            harm_upper = p_harm[mask] + z * p_harm_std[mask]
            for weight in (0.5, 1.0, 2.0, 4.0):
                score = np.where(
                    valid[mask],
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
                    dual[mask][order, chosen].sum()
                )
                harm_count = int(
                    harm[mask][order, chosen].sum()
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
        top40[name] = best_ordering

    prediction_path = args.output_dir / "loco_predictions.npz"
    write_npz(
        prediction_path,
        record_key=np.concatenate(
            [value["record_key"] for value in cohorts]
        ),
        cohort=cohort_name,
        route_group=group,
        p_dual=p_dual,
        p_harm=p_harm,
        p_dual_std=p_dual_std,
        p_harm_std=p_harm_std,
        valid=valid,
        selected_index=best["selected"],
    )
    report = {
        "schema": "recon_cem_latent_base_risk_v16",
        "status": (
            "loco_base_safety_passed_cem_replacement_may_train"
            if feasible
            else "loco_base_safety_failed_stop_same_feature_tuning"
        ),
        "evidence_role": "development_only_four_spent_cohorts",
        "cem_samples_remain_real_and_frozen": True,
        "cem_replacement_trained": False,
        "states": int(len(feature)),
        "feature_dim": int(feature.shape[-1]),
        "spatial_latent_dim": 221,
        "latent_feature_contract": (
            "same-patch 4x4 maps across four history frames + per-frame "
            "goal-token best-match/displacement statistics + current-goal "
            "9x9 offset correlation + temporal global progression"
        ),
        "no_dinov3_reencoding": True,
        "online_causal": True,
        "future_route_windows_used_for_policy": False,
        "config": {
            **asdict(config),
            "epochs": args.epochs,
            "batch_states": args.batch_states,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "ensemble_members_per_exclusion": args.ensemble_members,
            "seed": args.seed,
        },
        "calibration_by_fold": calibration_by_fold,
        "policy_search": {
            "tier": tier,
            "searched_policies": len(policies),
            "strict_feasible_policies": len(feasible),
            "policy": best["policy"],
            "per_cohort": best["per_cohort"],
            "strict": best["strict"],
        },
        "noncausal_top40_feature_separability_diagnostic_only": top40,
        "latent_caches": latent_records,
        "checkpoints": checkpoints,
        "loco_predictions": str(prediction_path),
        "loco_predictions_sha256": file_sha256(prediction_path),
        "next_confirmation_accessed": False,
        "formal_claim_allowed": False,
        "scand_fresh_confirmation_available": False,
    }
    write_json(result_path, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
