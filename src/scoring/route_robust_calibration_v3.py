"""Route-robust risk calibration and fixed-pool selection for universal D."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn


RISK_FEATURE_DIM = 27


class RouteRiskCalibratorV3(nn.Module):
    """Small calibration-only head for dual-helpful and harm probabilities."""

    def __init__(self, input_dim: int = RISK_FEATURE_DIM, hidden_dim: int = 48):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return self.network(feature)


@dataclass(frozen=True)
class RouteRiskPolicyV3:
    harm_rank_weight: float
    uncertainty_rank_weight: float
    z: float
    margin: float
    dual_threshold: float
    harm_threshold: float
    risk_ood_threshold: float = 1.0e9
    calibrator_disagreement_threshold: float = 1.0e9


def risk_features(
    prediction: dict[str, np.ndarray],
    member_prediction: list[dict[str, np.ndarray]],
    support_nonconformity: np.ndarray,
    robot_context: np.ndarray | None = None,
) -> np.ndarray:
    """Build candidate-level features without outcome or scorer-v2 predictions."""

    mean = np.asarray(prediction["mean"], dtype=np.float32)
    std = np.asarray(prediction["std"], dtype=np.float32)
    if mean.shape[-1] != 4 or std.shape != mean.shape:
        raise ValueError("universal mean/std must have shape [N,K,4]")
    member_mean = np.stack(
        [np.asarray(value["mean"], dtype=np.float32) for value in member_prediction]
    )
    if member_mean.shape[1:] != mean.shape:
        raise ValueError("member prediction shape changed")
    mean_disagreement = member_mean.std(axis=0)
    logits = np.stack(
        [
            np.asarray(prediction[key], dtype=np.float32)
            for key in (
                "helpful_logit",
                "dual_logit",
                "harm_logit",
                "rank_logit",
            )
        ],
        axis=-1,
    )
    member_logits = np.stack(
        [
            np.stack(
                [
                    np.asarray(value[key], dtype=np.float32)
                    for key in (
                        "helpful_logit",
                        "dual_logit",
                        "harm_logit",
                        "rank_logit",
                    )
                ],
                axis=-1,
            )
            for value in member_prediction
        ]
    )
    logit_disagreement = member_logits.std(axis=0)
    incumbent_relative = mean - mean[:, :1]
    support = np.asarray(support_nonconformity, dtype=np.float32)[..., None]
    if robot_context is None:
        robot = np.zeros((*mean.shape[:2], 2), dtype=np.float32)
    else:
        robot = np.asarray(robot_context, dtype=np.float32)
        if robot.shape == (mean.shape[0], 2):
            robot = np.broadcast_to(
                robot[:, None, :], (*mean.shape[:2], 2)
            )
        if robot.shape != (*mean.shape[:2], 2):
            raise ValueError("robot context must have shape [N,2] or [N,K,2]")
    result = np.concatenate(
        (
            mean,
            incumbent_relative,
            np.log(np.maximum(std, 1.0e-5)),
            mean_disagreement,
            logits,
            logit_disagreement,
            support,
            robot,
        ),
        axis=-1,
    ).astype(np.float32)
    if result.shape[-1] != RISK_FEATURE_DIM:
        raise RuntimeError(f"route-risk dimension changed: {result.shape}")
    if not np.isfinite(result).all():
        raise RuntimeError("route-risk features contain NaN or Inf")
    return result


def normalization(
    feature: np.ndarray, valid: np.ndarray
) -> dict[str, np.ndarray]:
    mask = np.asarray(valid, dtype=bool).copy()
    mask[:, 0] = False
    values = np.asarray(feature, dtype=np.float32)[mask]
    if not len(values):
        raise ValueError("no valid proposal for risk normalization")
    center = np.median(values, axis=0)
    scale = np.median(np.abs(values - center), axis=0) * np.float32(1.4826)
    scale = np.maximum(scale, np.float32(1.0e-4))
    return {
        "feature_center": center.astype(np.float32),
        "feature_scale": scale.astype(np.float32),
    }


def normalize(
    feature: np.ndarray, stats: dict[str, np.ndarray]
) -> np.ndarray:
    return np.clip(
        (
            np.asarray(feature, dtype=np.float32)
            - np.asarray(stats["feature_center"], dtype=np.float32)
        )
        / np.asarray(stats["feature_scale"], dtype=np.float32),
        -12.0,
        12.0,
    ).astype(np.float32)


def risk_nonconformity(
    feature: np.ndarray, stats: dict[str, np.ndarray]
) -> np.ndarray:
    # Current-candidate OOD score against a frozen calibration reference.
    standardized = np.abs(normalize(feature, stats))
    return np.quantile(standardized, 0.90, axis=-1).astype(np.float32)


def calibrated_probabilities(
    model: RouteRiskCalibratorV3,
    feature: np.ndarray,
    stats: dict[str, np.ndarray],
    *,
    device: str,
    batch_candidates: int = 8192,
) -> tuple[np.ndarray, np.ndarray]:
    flat = normalize(feature, stats).reshape(-1, feature.shape[-1])
    output = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(flat), batch_candidates):
            value = torch.from_numpy(
                flat[start : start + batch_candidates]
            ).to(device)
            output.append(torch.sigmoid(model(value)).cpu().numpy())
    probability = np.concatenate(output).reshape(*feature.shape[:2], 2)
    return (
        probability[..., 0].astype(np.float32),
        probability[..., 1].astype(np.float32),
    )


def select_fixed_pool(
    prediction: dict[str, np.ndarray],
    p_dual: np.ndarray,
    p_harm: np.ndarray,
    valid: np.ndarray,
    policy: RouteRiskPolicyV3 | dict[str, float],
    *,
    risk_nonconformity_score: np.ndarray | None = None,
    calibrator_disagreement: np.ndarray | None = None,
) -> np.ndarray:
    if isinstance(policy, dict):
        policy = RouteRiskPolicyV3(**policy)
    allowed = np.asarray(valid, dtype=bool).copy()
    allowed[:, 0] = False
    uncertainty = (
        prediction["std"][..., 0] + prediction["std"][..., 3]
    )
    rank_score = (
        p_dual
        - policy.harm_rank_weight * p_harm
        - policy.uncertainty_rank_weight * uncertainty
    )
    challenger = np.where(allowed, rank_score, -np.inf).argmax(axis=1)
    exists = allowed.any(axis=1)
    challenger = np.where(exists, challenger, 0).astype(np.int64)
    row = np.arange(len(challenger))
    if risk_nonconformity_score is None:
        risk_nonconformity_score = np.zeros_like(p_dual, dtype=np.float32)
    if calibrator_disagreement is None:
        calibrator_disagreement = np.zeros_like(p_dual, dtype=np.float32)
    ucb_ate = (
        prediction["mean"][row, challenger, 0]
        + policy.z * prediction["std"][row, challenger, 0]
    )
    ucb_objective = (
        prediction["mean"][row, challenger, 3]
        + policy.z * prediction["std"][row, challenger, 3]
    )
    promote = (
        exists
        & (p_dual[row, challenger] >= policy.dual_threshold)
        & (p_harm[row, challenger] <= policy.harm_threshold)
        & (
            risk_nonconformity_score[row, challenger]
            <= policy.risk_ood_threshold
        )
        & (
            calibrator_disagreement[row, challenger]
            <= policy.calibrator_disagreement_threshold
        )
        & (ucb_ate < -policy.margin)
        & (ucb_objective < -policy.margin)
    )
    return np.where(promote, challenger, 0).astype(np.int64)


def policy_dict(policy: RouteRiskPolicyV3) -> dict[str, float]:
    return {
        key: float(value)
        for key, value in policy.__dict__.items()
    }


__all__ = [
    "RISK_FEATURE_DIM",
    "RouteRiskCalibratorV3",
    "RouteRiskPolicyV3",
    "calibrated_probabilities",
    "normalization",
    "normalize",
    "policy_dict",
    "risk_nonconformity",
    "risk_features",
    "select_fixed_pool",
]
