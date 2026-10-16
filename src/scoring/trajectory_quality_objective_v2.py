"""Absolute, difference, and Pareto losses for trajectory scorer-v2."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from scoring.trajectory_quality_scorer_v2 import TrajectoryQualityPrediction


@dataclass(frozen=True)
class TrajectoryQualityLossConfig:
    ate_scale: float
    rtp_scale: float
    absolute_weight: float = 1.0
    pairwise_weight: float = 0.5
    dominance_weight: float = 0.25
    huber_delta: float = 1.0
    ate_dominance_epsilon: float = 0.0
    rtp_dominance_epsilon: float = 0.0
    dominance_margin: float = 0.05
    dominance_temperature: float = 0.10

    def validate(self) -> None:
        if self.ate_scale <= 0 or self.rtp_scale <= 0:
            raise ValueError("target scales must be positive")
        if min(
            self.absolute_weight,
            self.pairwise_weight,
            self.dominance_weight,
        ) < 0:
            raise ValueError("loss weights must be non-negative")
        if self.huber_delta <= 0 or self.dominance_temperature <= 0:
            raise ValueError("loss delta/temperature must be positive")
        if self.ate_dominance_epsilon < 0 or self.rtp_dominance_epsilon < 0:
            raise ValueError("dominance epsilons must be non-negative")


@dataclass(frozen=True)
class TrajectoryQualityLoss:
    total: torch.Tensor
    absolute: torch.Tensor
    pairwise: torch.Tensor
    dominance: torch.Tensor
    valid_candidates: int
    valid_pairs: int
    dominance_pairs: int


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if values.shape != mask.shape:
        raise ValueError("value and mask shapes differ")
    count = mask.sum()
    if int(count.item()) == 0:
        return values.sum() * 0.0
    return values.masked_select(mask).mean()


def trajectory_quality_loss(
    prediction: TrajectoryQualityPrediction,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    config: TrajectoryQualityLossConfig,
) -> TrajectoryQualityLoss:
    """Compute a candidate-order-invariant loss.

    ``target[...,0]`` is absolute ATE and ``target[...,1]`` is absolute RTP.
    Pairwise regression learns metric differences for every valid unordered
    pair.  The ordering loss is applied only when one target clearly improves
    both metrics, so ambiguous ATE/RTP trade-offs are never assigned a forced
    scalar ordering.
    """
    config.validate()
    if target.ndim != 3 or target.shape[-1] != 2:
        raise ValueError("expected target [B,K,2]")
    if prediction.ate.shape != target.shape[:2]:
        raise ValueError("prediction and target shapes differ")
    if prediction.rtp.shape != target.shape[:2]:
        raise ValueError("prediction and target shapes differ")
    if valid_mask.shape != target.shape[:2] or valid_mask.dtype != torch.bool:
        raise ValueError("valid_mask must be bool [B,K]")
    finite = (
        torch.isfinite(target).all(dim=-1)
        & torch.isfinite(prediction.ate)
        & torch.isfinite(prediction.rtp)
    )
    valid = valid_mask & finite

    predicted = prediction.stacked()
    scales = target.new_tensor((config.ate_scale, config.rtp_scale))
    residual = (predicted - target) / scales
    absolute_per_metric = F.huber_loss(
        residual,
        torch.zeros_like(residual),
        reduction="none",
        delta=config.huber_delta,
    )
    absolute = _masked_mean(
        absolute_per_metric,
        valid[..., None].expand_as(absolute_per_metric),
    )

    candidate_count = target.shape[1]
    pair_triangle = torch.triu(
        torch.ones(
            candidate_count,
            candidate_count,
            dtype=torch.bool,
            device=target.device,
        ),
        diagonal=1,
    )
    valid_pairs = (
        valid[:, :, None] & valid[:, None, :] & pair_triangle[None, :, :]
    )
    predicted_difference = predicted[:, :, None, :] - predicted[:, None, :, :]
    target_difference = target[:, :, None, :] - target[:, None, :, :]
    difference_residual = (predicted_difference - target_difference) / scales
    pairwise_per_metric = F.huber_loss(
        difference_residual,
        torch.zeros_like(difference_residual),
        reduction="none",
        delta=config.huber_delta,
    )
    pairwise = _masked_mean(
        pairwise_per_metric,
        valid_pairs[..., None].expand_as(pairwise_per_metric),
    )

    target_ate = target[..., 0]
    target_rtp = target[..., 1]
    # i clearly dominates j only when i is better in both metrics.
    i_dominates_j = (
        target_ate[:, :, None]
        < target_ate[:, None, :] - config.ate_dominance_epsilon
    ) & (
        target_rtp[:, :, None]
        > target_rtp[:, None, :] + config.rtp_dominance_epsilon
    )
    dominance_mask = (
        valid[:, :, None]
        & valid[:, None, :]
        & i_dominates_j
        & ~torch.eye(
            candidate_count,
            dtype=torch.bool,
            device=target.device,
        )[None]
    )
    predicted_ate_gain = (
        prediction.ate[:, None, :] - prediction.ate[:, :, None]
    ) / config.ate_scale
    predicted_rtp_gain = (
        prediction.rtp[:, :, None] - prediction.rtp[:, None, :]
    ) / config.rtp_scale
    ate_penalty = F.softplus(
        (config.dominance_margin - predicted_ate_gain)
        / config.dominance_temperature
    )
    rtp_penalty = F.softplus(
        (config.dominance_margin - predicted_rtp_gain)
        / config.dominance_temperature
    )
    dominance = _masked_mean(
        0.5 * (ate_penalty + rtp_penalty),
        dominance_mask,
    )

    total = (
        config.absolute_weight * absolute
        + config.pairwise_weight * pairwise
        + config.dominance_weight * dominance
    )
    return TrajectoryQualityLoss(
        total=total,
        absolute=absolute,
        pairwise=pairwise,
        dominance=dominance,
        valid_candidates=int(valid.sum().item()),
        valid_pairs=int(valid_pairs.sum().item()),
        dominance_pairs=int(dominance_mask.sum().item()),
    )


__all__ = [
    "TrajectoryQualityLoss",
    "TrajectoryQualityLossConfig",
    "trajectory_quality_loss",
]
