"""DINOv3 reimplementation of the 019f70d5 dense group-first selector.

The module contains only label-safe model and selection mechanics.  Dataset
opening order, split freezing, and artifact provenance are owned by the runner.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import random
from typing import Any

import numpy as np
import torch
from torch import nn


MODES = ("action", "static", "latent", "full")


@dataclass(frozen=True)
class DenseHeadConfig:
    hidden: int = 192
    dropout: float = 0.10
    epochs: int = 150
    batch_groups: int = 64
    learning_rate: float = 4.0e-4
    weight_decay: float = 3.0e-3
    target_temperature: float = 0.05

    def validate(self) -> None:
        if min(self.hidden, self.epochs, self.batch_groups) <= 0:
            raise ValueError("hidden/epochs/batch_groups must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0,1)")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("optimizer configuration is invalid")
        if self.target_temperature <= 0:
            raise ValueError("target temperature must be positive")


class GroupFirstNet(nn.Module):
    """Per-candidate outcome heads plus set-level challenge confidence."""

    def __init__(self, input_dim: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.outcome = nn.Linear(hidden, 4)
        self.help = nn.Linear(hidden, 1)
        self.safe = nn.Linear(hidden, 1)
        self.harm = nn.Linear(hidden, 1)
        self.select = nn.Linear(hidden, 1)
        self.rank = nn.Linear(hidden, 1)
        self.challenge = nn.Sequential(
            nn.Linear(2 * hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, value: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.encoder(value)
        pooled = torch.cat((hidden.mean(dim=1), hidden.max(dim=1).values), dim=-1)
        return {
            "outcome": self.outcome(hidden),
            "help": self.help(hidden).squeeze(-1),
            "safe": self.safe(hidden).squeeze(-1),
            "harm": self.harm(hidden).squeeze(-1),
            "select": self.select(hidden).squeeze(-1),
            "rank": self.rank(hidden).squeeze(-1),
            "challenge": self.challenge(pooled).squeeze(-1),
        }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def family_fold(family: str, folds: int) -> int:
    if folds < 2:
        raise ValueError("at least two folds are required")
    digest = hashlib.sha256(("dinov3-dense-meta-v1:" + family).encode()).digest()
    return int.from_bytes(digest[:8], "big") % folds


def feature_by_mode(
    data: dict[str, np.ndarray], mode: str, *, variant: str = "raw"
) -> np.ndarray:
    action = np.asarray(data["action_features"], dtype=np.float32)
    static = np.asarray(data["static_features"], dtype=np.float32)
    raw = np.asarray(data["raw_ltwm_features"], dtype=np.float32)
    if variant == "raw":
        latent = raw
    elif variant == "v2_stacked":
        auxiliary = np.asarray(data["scorer_v2_aux_features"], dtype=np.float32)
        if auxiliary.shape[:2] != raw.shape[:2]:
            raise ValueError("raw LTWM/scorer-v2 auxiliary dimensions differ")
        latent = np.concatenate((raw, auxiliary), axis=-1)
    else:
        raise ValueError(f"unknown feature variant: {variant}")
    if action.ndim != 3 or static.shape[:2] != action.shape[:2] or latent.shape[:2] != action.shape[:2]:
        raise ValueError("dense mode feature dimensions differ")
    reference = latent[:, 1:]
    center = reference.mean(axis=1, keepdims=True)
    scale = np.maximum(reference.std(axis=1, keepdims=True), 1.0e-5)
    relative = np.clip((latent - center) / scale, -8.0, 8.0).astype(np.float32)
    if mode == "action":
        value = action
    elif mode == "static":
        value = np.concatenate((action, static), axis=-1)
    elif mode == "latent":
        value = np.concatenate((latent, relative), axis=-1)
    elif mode == "full":
        value = np.concatenate((action, static, latent, relative), axis=-1)
    else:
        raise ValueError(f"unknown mode: {mode}")
    if not np.isfinite(value).all():
        raise RuntimeError(f"{mode} features contain NaN or Inf")
    return value.astype(np.float32)


def build_targets(value: np.ndarray, temperature: float) -> dict[str, np.ndarray]:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    targets = np.asarray(value, dtype=np.float32)
    if targets.ndim != 3 or targets.shape[-1] < 4 or targets.shape[1] < 2:
        raise ValueError("expected target tensor [N,K,>=4]")
    delta = targets[..., :4]
    helpful = delta[..., 0] < -0.005
    safe = (delta[..., 0] < -0.005) & (delta[..., 3] < -0.005)
    hard = (
        (delta[..., 0] > 0.0)
        | (delta[..., 1] > 0.005)
        | (delta[..., 2] > 0.020)
        | (delta[..., 3] > 0.010)
    )
    utility = (
        delta[..., 0]
        + 0.25 * np.maximum(delta[..., 2], 0.0)
        + 0.25 * np.maximum(delta[..., 3], 0.0)
        + 0.10 * hard.astype(np.float32)
    )
    oracle_select = np.zeros_like(utility, dtype=np.float32)
    challenge = np.zeros(len(utility), dtype=np.float32)
    for row in range(len(utility)):
        eligible = np.flatnonzero(safe[row, 1:]) + 1
        if len(eligible):
            best = int(eligible[np.argmin(utility[row, eligible])])
            oracle_select[row, best] = 1.0
            challenge[row] = float(utility[row, best] < -0.005)
    shifted = -utility[:, 1:] / temperature
    shifted -= shifted.max(axis=1, keepdims=True)
    distribution = np.exp(shifted)
    distribution /= distribution.sum(axis=1, keepdims=True)
    return {
        "delta": delta.astype(np.float32),
        "helpful": helpful.astype(np.float32),
        "safe": safe.astype(np.float32),
        "hard": hard.astype(np.float32),
        "oracle_select": oracle_select,
        "challenge": challenge,
        "list_distribution": distribution.astype(np.float32),
    }


def _positive_weight(value: np.ndarray, maximum: float = 30.0) -> float:
    positive = float(value.sum())
    negative = float(value.size - positive)
    return min(negative / max(positive, 1.0), maximum)


def fit_fold(
    feature: np.ndarray,
    target: dict[str, np.ndarray],
    train_index: np.ndarray,
    config: DenseHeadConfig,
    *,
    seed: int,
    device: str,
) -> tuple[GroupFirstNet, dict[str, np.ndarray]]:
    config.validate()
    set_seed(seed)
    train_feature = feature[train_index]
    flat = train_feature.reshape(-1, train_feature.shape[-1])
    mean = flat.mean(axis=0).astype(np.float32)
    scale = np.maximum(flat.std(axis=0), 1.0e-5).astype(np.float32)
    normalized = np.clip((train_feature - mean) / scale, -8.0, 8.0).astype(np.float32)
    delta = target["delta"][train_index]
    target_mean = delta.reshape(-1, 4).mean(axis=0).astype(np.float32)
    target_scale = np.maximum(delta.reshape(-1, 4).std(axis=0), 0.01).astype(np.float32)
    model = GroupFirstNet(feature.shape[-1], config.hidden, config.dropout).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    bce = {
        name: nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(
                _positive_weight(target[name][train_index]), device=device
            )
        )
        for name in ("helpful", "safe", "hard", "oracle_select", "challenge")
    }
    rng = np.random.default_rng(seed)
    for _ in range(config.epochs):
        order = rng.permutation(len(train_index))
        model.train()
        for start in range(0, len(order), config.batch_groups):
            local = order[start : start + config.batch_groups]
            value = torch.from_numpy(normalized[local]).to(device)
            output = model(value)
            true_delta = torch.from_numpy(
                (delta[local] - target_mean) / target_scale
            ).to(device)
            loss = nn.functional.smooth_l1_loss(output["outcome"], true_delta)
            for target_name, output_name, weight in (
                ("helpful", "help", 0.2),
                ("safe", "safe", 1.7),
                ("hard", "harm", 2.2),
                ("oracle_select", "select", 1.7),
            ):
                expected = torch.from_numpy(target[target_name][train_index][local]).to(
                    device
                )
                loss = loss + weight * bce[target_name](output[output_name], expected)
            distribution = torch.from_numpy(
                target["list_distribution"][train_index][local]
            ).to(device)
            loss = loss + 1.6 * (
                -(
                    distribution
                    * torch.log_softmax(output["rank"][:, 1:], dim=1)
                )
                .sum(dim=1)
                .mean()
            )
            expected_challenge = torch.from_numpy(
                target["challenge"][train_index][local]
            ).to(device)
            loss = loss + 0.4 * bce["challenge"](
                output["challenge"], expected_challenge
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
    model.eval()
    return model, {
        "mean": mean,
        "scale": scale,
        "target_mean": target_mean,
        "target_scale": target_scale,
    }


def predict(
    model: GroupFirstNet,
    stats: dict[str, np.ndarray],
    feature: np.ndarray,
    index: np.ndarray,
    *,
    device: str,
) -> dict[str, np.ndarray]:
    normalized = np.clip(
        (feature[index] - stats["mean"]) / stats["scale"], -8.0, 8.0
    ).astype(np.float32)
    with torch.inference_mode():
        output = model(torch.from_numpy(normalized).to(device))
    decoded = (
        output["outcome"].cpu().numpy() * stats["target_scale"]
        + stats["target_mean"]
    ).astype(np.float32)
    return {
        "outcome": decoded,
        "p_help": torch.sigmoid(output["help"]).cpu().numpy().astype(np.float32),
        "p_safe": torch.sigmoid(output["safe"]).cpu().numpy().astype(np.float32),
        "p_harm": torch.sigmoid(output["harm"]).cpu().numpy().astype(np.float32),
        "p_select": torch.sigmoid(output["select"]).cpu().numpy().astype(np.float32),
        "p_rank": torch.softmax(output["rank"][:, 1:], dim=1)
        .cpu()
        .numpy()
        .astype(np.float32),
        "p_challenge": torch.sigmoid(output["challenge"])
        .cpu()
        .numpy()
        .astype(np.float32),
    }


def empty_prediction(states: int, candidates: int) -> dict[str, np.ndarray]:
    if states <= 0 or candidates < 2:
        raise ValueError("empty prediction dimensions are invalid")
    return {
        "outcome": np.empty((states, candidates, 4), dtype=np.float32),
        "p_help": np.empty((states, candidates), dtype=np.float32),
        "p_safe": np.empty((states, candidates), dtype=np.float32),
        "p_harm": np.empty((states, candidates), dtype=np.float32),
        "p_select": np.empty((states, candidates), dtype=np.float32),
        "p_rank": np.empty((states, candidates - 1), dtype=np.float32),
        "p_challenge": np.empty(states, dtype=np.float32),
    }


def ensemble_predict(
    members: list[tuple[GroupFirstNet, dict[str, np.ndarray]]],
    feature: np.ndarray,
    *,
    device: str,
) -> dict[str, np.ndarray]:
    if not members:
        raise ValueError("at least one ensemble member is required")
    index = np.arange(len(feature))
    values = [predict(model, stats, feature, index, device=device) for model, stats in members]
    return {
        key: np.mean(np.stack([value[key] for value in values]), axis=0).astype(np.float32)
        for key in values[0]
    }


def _candidate_valid(valid: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray:
    if valid is None:
        return np.ones(shape, dtype=bool)
    mask = np.asarray(valid, dtype=bool)
    if mask.shape != shape:
        raise ValueError(f"candidate valid mask must have shape {shape}")
    return mask.copy()


def group_first_rows(
    prediction: dict[str, np.ndarray],
    true_target: np.ndarray | None = None,
    *,
    valid: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    predicted = prediction["outcome"]
    states, candidates = predicted.shape[:2]
    utility = (
        predicted[:, 1:, 0]
        + 0.25 * np.maximum(predicted[:, 1:, 2], 0.0)
        + 0.25 * np.maximum(predicted[:, 1:, 3], 0.0)
    )
    allowed = _candidate_valid(valid, (states, candidates))[:, 1:]
    utility = np.where(allowed, utility, np.inf)
    local = utility.argmin(axis=1)
    exists = allowed.any(axis=1)
    challenger = np.where(exists, local + 1, 0).astype(np.int64)
    row = np.arange(states)
    p_gate_all = (
        0.35 * prediction["p_select"][:, 1:]
        + 0.25 * prediction["p_rank"]
        + 0.25 * prediction["p_safe"][:, 1:]
        + 0.15 * prediction["p_challenge"][:, None]
    )
    output = {
        "challenger": challenger,
        "p_gate": np.where(exists, p_gate_all[row, local], 0.0).astype(np.float32),
        "predicted_delta": predicted[row, challenger].astype(np.float32),
        "p_safe": prediction["p_safe"][row, challenger].astype(np.float32),
        "p_select": prediction["p_select"][row, challenger].astype(np.float32),
        "p_rank": np.where(exists, prediction["p_rank"][row, local], 0.0).astype(
            np.float32
        ),
        "p_challenge": prediction["p_challenge"].astype(np.float32),
    }
    if true_target is not None:
        output["true_delta"] = np.asarray(true_target, dtype=np.float32)[
            row, challenger, :4
        ]
    return output


def residual_rows(
    action: dict[str, np.ndarray],
    residual: dict[str, np.ndarray],
    *,
    alpha: float,
    gate: str,
    risk: str,
    true_target: np.ndarray | None = None,
    valid: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("residual alpha must be in [0,1]")
    states, candidates = action["p_safe"].shape
    blended_safe = action["p_safe"] + alpha * (
        residual["p_safe"] - action["p_safe"]
    )
    allowed = _candidate_valid(valid, (states, candidates))
    allowed[:, 0] = False
    ranked = np.where(allowed, blended_safe, -np.inf)
    challenger = ranked.argmax(axis=1).astype(np.int64)
    exists = allowed.any(axis=1)
    challenger = np.where(exists, challenger, 0)
    row = np.arange(states)
    action_not_harm = 1.0 - action["p_harm"][row, challenger]
    residual_not_harm = 1.0 - residual["p_harm"][row, challenger]
    if gate == "linear":
        p_gate = action_not_harm + alpha * (residual_not_harm - action_not_harm)
    elif gate == "min_confirm":
        p_gate = np.minimum(action_not_harm, residual_not_harm)
    elif gate == "geometric":
        p_gate = np.sqrt(
            np.clip(action_not_harm, 0.0, 1.0)
            * np.clip(residual_not_harm, 0.0, 1.0)
        )
    else:
        raise ValueError(f"unknown residual gate: {gate}")
    action_outcome = action["outcome"][row, challenger]
    residual_outcome = residual["outcome"][row, challenger]
    if risk == "action":
        predicted = action_outcome
    elif risk == "residual":
        predicted = residual_outcome
    elif risk == "both":
        predicted = np.maximum(action_outcome, residual_outcome)
    elif risk == "blend":
        predicted = action_outcome + alpha * (residual_outcome - action_outcome)
    else:
        raise ValueError(f"unknown residual risk: {risk}")
    output = {
        "challenger": challenger,
        "p_gate": np.where(exists, np.clip(p_gate, 0.0, 1.0), 0.0).astype(np.float32),
        "predicted_delta": predicted.astype(np.float32),
        "p_safe": blended_safe[row, challenger].astype(np.float32),
        "action_p_safe": action["p_safe"][row, challenger].astype(np.float32),
        "residual_p_safe": residual["p_safe"][row, challenger].astype(np.float32),
    }
    if true_target is not None:
        output["true_delta"] = np.asarray(true_target, dtype=np.float32)[
            row, challenger, :4
        ]
    return output


def promoted_mask(rows: dict[str, np.ndarray], threshold: float) -> np.ndarray:
    predicted = rows["predicted_delta"]
    return (
        (rows["challenger"] != 0)
        & (rows["p_gate"] >= threshold)
        & (predicted[:, 0] < 0.0)
        & (predicted[:, 3] < 0.0)
    )


def wilson_lower(successes: int, total: int, z: float = 1.959963984540054) -> float | None:
    if total <= 0:
        return None
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = proportion + z * z / (2.0 * total)
    radius = z * math.sqrt(
        proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)
    )
    return float((center - radius) / denominator)


def threshold_metrics(rows: dict[str, np.ndarray], threshold: float) -> dict[str, Any]:
    if "true_delta" not in rows:
        raise ValueError("threshold metrics require true deltas")
    promote = promoted_mask(rows, threshold)
    true = rows["true_delta"]
    count = int(promote.sum())
    ate = int((true[promote, 0] < -1.0e-7).sum())
    objective = int((true[promote, 3] < -1.0e-7).sum())
    dual = int(
        ((true[promote, 0] < -1.0e-7) & (true[promote, 3] < -1.0e-7)).sum()
    )
    return {
        "threshold": float(threshold),
        "promotions": count,
        "ate_successes": ate,
        "objective_successes": objective,
        "dual_successes": dual,
        "ate_precision": ate / count if count else None,
        "objective_precision": objective / count if count else None,
        "dual_precision": dual / count if count else None,
        "dual_wilson_lower_95": wilson_lower(dual, count),
        "mean_ate_delta_m": float(true[promote, 0].mean()) if count else None,
        "mean_objective_delta_m": float(true[promote, 3].mean()) if count else None,
        "net_ate_delta_per_state_m": float(true[promote, 0].sum() / len(true)),
        "net_objective_delta_per_state_m": float(true[promote, 3].sum() / len(true)),
    }


def choose_threshold(
    rows: dict[str, np.ndarray],
    *,
    min_promotions: int,
    target_dual_precision: float,
    target_wilson_lower: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    values = np.unique(np.concatenate(([0.0, 1.0], rows["p_gate"])))
    frontier = [threshold_metrics(rows, float(value)) for value in values]
    feasible = [
        value
        for value in frontier
        if value["promotions"] >= min_promotions
        and value["dual_precision"] is not None
        and value["dual_precision"] >= target_dual_precision
        and value["dual_wilson_lower_95"] is not None
        and value["dual_wilson_lower_95"] >= target_wilson_lower
    ]
    pool = feasible or [value for value in frontier if value["promotions"] > 0]
    if not pool:
        pool = [threshold_metrics(rows, 1.0)]
    best = min(
        pool,
        key=lambda value: (
            0 if value in feasible else 1,
            value["net_ate_delta_per_state_m"]
            + value["net_objective_delta_per_state_m"],
            -value["promotions"],
        ),
    )
    return {**best, "feasible_gate": bool(feasible)}, frontier


def selected_indices(rows: dict[str, np.ndarray], threshold: float) -> np.ndarray:
    promote = promoted_mask(rows, threshold)
    return np.where(promote, rows["challenger"], 0).astype(np.int64)


def selection_metrics(selected: np.ndarray, targets: np.ndarray) -> dict[str, Any]:
    index = np.asarray(selected, dtype=np.int64)
    target = np.asarray(targets, dtype=np.float32)
    row = np.arange(len(index))
    delta = target[row, index, :4]
    promote = index != 0
    count = int(promote.sum())
    ate = int((delta[promote, 0] < -1.0e-7).sum())
    objective = int((delta[promote, 3] < -1.0e-7).sum())
    dual = int(
        ((delta[promote, 0] < -1.0e-7) & (delta[promote, 3] < -1.0e-7)).sum()
    )
    return {
        "states": int(len(index)),
        "promotions": count,
        "ate_precision": ate / count if count else None,
        "objective_precision": objective / count if count else None,
        "dual_precision": dual / count if count else None,
        "dual_wilson_lower_95": wilson_lower(dual, count),
        "net_ate_delta_per_state_m": float(delta[:, 0].mean()),
        "net_rpe_delta_per_state_m": float(delta[:, 1].mean()),
        "net_endpoint_delta_per_state_m": float(delta[:, 2].mean()),
        "net_objective_delta_per_state_m": float(delta[:, 3].mean()),
        "selected_delta": delta,
    }


def build_meta_features(
    predictions: dict[str, dict[str, np.ndarray]],
    rows: dict[str, dict[str, np.ndarray]],
    data: dict[str, np.ndarray],
    *,
    variant: str = "raw",
) -> np.ndarray:
    """Build label-free state features for the static/full route decision."""
    blocks = []
    for mode in MODES:
        prediction = predictions[mode]
        selected = rows[mode]
        challengers = slice(1, None)
        blocks.extend(
            (
                selected["predicted_delta"],
                selected["p_gate"][:, None],
                selected["p_safe"][:, None],
                prediction["p_challenge"][:, None],
            )
        )
        for key in ("p_safe", "p_harm", "p_select"):
            value = prediction[key][:, challengers]
            blocks.append(
                np.stack(
                    (value.mean(1), value.std(1), value.min(1), value.max(1)), axis=-1
                )
            )
        outcome = prediction["outcome"][:, challengers]
        blocks.extend(
            (
                outcome.mean(axis=1),
                outcome.std(axis=1),
                outcome.min(axis=1),
                outcome.max(axis=1),
            )
        )
    blocks.append(
        (rows["full"]["challenger"] != rows["static"]["challenger"])[
            :, None
        ].astype(np.float32)
    )
    blocks.append(rows["full"]["predicted_delta"] - rows["static"]["predicted_delta"])
    if variant not in {"raw", "v2_stacked"}:
        raise ValueError(f"unknown feature variant: {variant}")
    keys = ["action_features", "static_features", "raw_ltwm_features"]
    if variant == "v2_stacked":
        keys.append("scorer_v2_aux_features")
    for key in keys:
        value = np.asarray(data[key], dtype=np.float32)
        norm = np.linalg.norm(value, axis=-1)
        blocks.append(
            np.stack((norm.mean(1), norm.std(1), norm.min(1), norm.max(1)), axis=-1)
        )
    result = np.concatenate(blocks, axis=-1).astype(np.float32)
    if not np.isfinite(result).all():
        raise RuntimeError("meta features contain NaN or Inf")
    return result


def fit_ridge(
    feature: np.ndarray,
    target: np.ndarray,
    *,
    regularization: float,
) -> dict[str, np.ndarray | float]:
    value = np.asarray(feature, dtype=np.float64)
    outcome = np.asarray(target, dtype=np.float64).reshape(-1)
    if value.ndim != 2 or outcome.shape != (len(value),):
        raise ValueError("ridge feature/target dimensions differ")
    mean = value.mean(axis=0)
    scale = np.maximum(value.std(axis=0), 1.0e-6)
    normalized = np.clip((value - mean) / scale, -8.0, 8.0)
    design = np.concatenate((normalized, np.ones((len(value), 1))), axis=1)
    penalty = np.eye(design.shape[1]) * float(regularization)
    penalty[-1, -1] = 0.0
    coefficient = np.linalg.solve(design.T @ design + penalty, design.T @ outcome)
    return {
        "feature_mean": mean.astype(np.float32),
        "feature_scale": scale.astype(np.float32),
        "coefficient": coefficient[:-1].astype(np.float32),
        "intercept": float(coefficient[-1]),
        "regularization": float(regularization),
    }


def predict_ridge(model: dict[str, Any], feature: np.ndarray) -> np.ndarray:
    normalized = np.clip(
        (np.asarray(feature, dtype=np.float32) - model["feature_mean"])
        / model["feature_scale"],
        -8.0,
        8.0,
    )
    return (
        normalized @ np.asarray(model["coefficient"], dtype=np.float32)
        + float(model["intercept"])
    ).astype(np.float32)


__all__ = [
    "DenseHeadConfig",
    "GroupFirstNet",
    "MODES",
    "build_meta_features",
    "build_targets",
    "choose_threshold",
    "empty_prediction",
    "ensemble_predict",
    "family_fold",
    "feature_by_mode",
    "fit_fold",
    "fit_ridge",
    "group_first_rows",
    "predict",
    "predict_ridge",
    "promoted_mask",
    "residual_rows",
    "selected_indices",
    "selection_metrics",
    "threshold_metrics",
    "wilson_lower",
]
