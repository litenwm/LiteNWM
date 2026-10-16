"""Losses for LiteTokenFormer-WM-v2 dense-token dynamics and selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

EPS = 1e-8


@dataclass(frozen=True)
class V2LossWeights:
    reconstruction: float = 1.0
    cosine: float = 0.05
    delta_direction: float = 0.05
    norm: float = 0.03
    action_order: float = 0.05
    action_decode: float = 0.05
    consistency: float = 0.02
    history: float = 0.03
    velocity: float = 0.02
    uncertainty: float = 0.01
    quality: float = 0.0


def nmse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (prediction - target).square().mean() / target.square().mean().clamp_min(EPS)


def cosine_distance(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (1.0 - F.cosine_similarity(prediction.flatten(-2), target.flatten(-2), dim=-1)).mean()


def delta_direction_loss(prediction: torch.Tensor, target: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
    initial = current[:, None, None].expand(prediction.shape[0], prediction.shape[1], 1, *current.shape[1:])
    pred_prev = torch.cat((initial, prediction[:, :, :-1]), dim=2)
    target_prev = torch.cat((initial, target[:, :, :-1]), dim=2)
    pred_delta = prediction - pred_prev
    target_delta = target - target_prev
    return (1.0 - F.cosine_similarity(pred_delta.flatten(-2), target_delta.flatten(-2), dim=-1)).mean()


def delta_norm_loss(prediction: torch.Tensor, target: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
    initial = current[:, None, None].expand(prediction.shape[0], prediction.shape[1], 1, *current.shape[1:])
    pred_prev = torch.cat((initial, prediction[:, :, :-1]), dim=2)
    target_prev = torch.cat((initial, target[:, :, :-1]), dim=2)
    pred_norm = torch.linalg.vector_norm((prediction - pred_prev).flatten(-2), dim=-1)
    target_norm = torch.linalg.vector_norm((target - target_prev).flatten(-2), dim=-1)
    # Symmetric relative error stays bounded when the true visual delta is
    # nearly zero, while still penalizing both norm shrink and expansion.
    step_relative = (pred_norm - target_norm).abs() / (pred_norm + target_norm).clamp_min(1e-4)
    current_expanded = current[:, None, None].expand_as(prediction)
    pred_horizon_norm = torch.linalg.vector_norm((prediction - current_expanded).flatten(-2), dim=-1)
    target_horizon_norm = torch.linalg.vector_norm((target - current_expanded).flatten(-2), dim=-1)
    horizon_relative = (pred_horizon_norm - target_horizon_norm).abs() / (
        pred_horizon_norm + target_horizon_norm
    ).clamp_min(1e-4)
    step_loss = F.smooth_l1_loss(step_relative, torch.zeros_like(step_relative))
    horizon_loss = F.smooth_l1_loss(horizon_relative, torch.zeros_like(horizon_relative))
    return 0.5 * (step_loss + horizon_loss)


def velocity_delta_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if prediction.shape[2] < 3:
        return prediction.new_zeros(())
    pred_velocity = prediction[:, :, 1:] - prediction[:, :, :-1]
    target_velocity = target[:, :, 1:] - target[:, :, :-1]
    pred_accel = pred_velocity[:, :, 1:] - pred_velocity[:, :, :-1]
    target_accel = target_velocity[:, :, 1:] - target_velocity[:, :, :-1]
    return F.smooth_l1_loss(pred_accel, target_accel)


def history_motion_loss(prediction: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
    observed_delta = history[:, -1] - history[:, -2]
    predicted_delta = prediction[:, :, 0] - history[:, None, -1]
    observed = observed_delta[:, None].expand_as(predicted_delta)
    return (1.0 - F.cosine_similarity(predicted_delta.flatten(-2), observed.flatten(-2), dim=-1)).mean()


def uncertainty_nll(prediction: torch.Tensor, target: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    token_error = (prediction - target).square().mean(dim=-1)
    return (torch.exp(-logvar) * token_error + logvar).mean()


def action_order_contrastive_loss(
    true_prediction: torch.Tensor,
    corrupted_prediction: torch.Tensor | None,
    target: torch.Tensor,
    margin: float = 0.02,
) -> torch.Tensor:
    if corrupted_prediction is None:
        return true_prediction.new_zeros(())
    true_error = (true_prediction - target).square().mean(dim=(-1, -2, -3))
    corrupt_error = (corrupted_prediction - target).square().mean(dim=(-1, -2, -3))
    return F.relu(margin + true_error - corrupt_error).mean()


def rollout_consistency_loss(prediction: torch.Tensor, prefix_prediction: torch.Tensor | None, prefix_len: int = 3) -> torch.Tensor:
    if prefix_prediction is None:
        return prediction.new_zeros(())
    return F.smooth_l1_loss(prediction[:, :, :prefix_len], prefix_prediction[:, :, :prefix_len])


def compute_world_model_v2_loss(
    output: dict[str, torch.Tensor],
    target: torch.Tensor,
    history: torch.Tensor,
    actions: torch.Tensor,
    *,
    weights: V2LossWeights | None = None,
    corrupted_output: dict[str, torch.Tensor] | None = None,
    prefix_output: dict[str, torch.Tensor] | None = None,
    quality_loss: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    weights = weights or V2LossWeights()
    prediction = output["predicted_tokens"]
    if target.ndim == 4:
        target = target[:, None]
    if actions.ndim == 3:
        actions = actions[:, None]
    target = target.expand(prediction.shape[0], prediction.shape[1], *target.shape[2:])
    current = history[:, -1]
    components = {
        "reconstruction": F.smooth_l1_loss(prediction, target) + nmse(prediction, target),
        "cosine": cosine_distance(prediction, target),
        "delta_direction": delta_direction_loss(prediction, target, current),
        "norm": delta_norm_loss(prediction, target, current),
        "action_order": action_order_contrastive_loss(
            prediction,
            corrupted_output["predicted_tokens"] if corrupted_output is not None else None,
            target,
        ),
        "action_decode": F.smooth_l1_loss(output["decoded_action"], actions),
        "consistency": rollout_consistency_loss(
            prediction,
            prefix_output["predicted_tokens"] if prefix_output is not None else None,
        ),
        "history": history_motion_loss(prediction, history),
        "velocity": velocity_delta_loss(prediction, target),
        "uncertainty": uncertainty_nll(prediction, target, output["predicted_logvar"]),
        "quality": quality_loss if quality_loss is not None else prediction.new_zeros(()),
    }
    total = (
        weights.reconstruction * components["reconstruction"]
        + weights.cosine * components["cosine"]
        + weights.delta_direction * components["delta_direction"]
        + weights.norm * components["norm"]
        + weights.action_order * components["action_order"]
        + weights.action_decode * components["action_decode"]
        + weights.consistency * components["consistency"]
        + weights.history * components["history"]
        + weights.velocity * components["velocity"]
        + weights.uncertainty * components["uncertainty"]
        + weights.quality * components["quality"]
    )
    return total, components


def pairwise_dominance_loss(scores: torch.Tensor, dominance: torch.Tensor) -> torch.Tensor:
    difference = scores[:, :, None] - scores[:, None, :]
    mask = dominance.bool()
    if not mask.any():
        return scores.new_zeros(())
    return F.softplus(-difference[mask]).mean()


def trajectory_multitask_loss(
    output: dict[str, torch.Tensor],
    metric_targets: torch.Tensor,
    success_targets: torch.Tensor,
    risk_targets: torch.Tensor,
    dominance: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    metric = F.smooth_l1_loss(output["metric_delta"], metric_targets)
    success = F.binary_cross_entropy_with_logits(output["success_logits"], success_targets)
    risk = F.binary_cross_entropy_with_logits(output["risk_logit"], risk_targets)
    ranking = pairwise_dominance_loss(output["ranking_score"], dominance)
    confidence_target = torch.exp(-torch.abs(output["metric_delta"].detach() - metric_targets).mean(dim=-1))
    calibration = F.mse_loss(output["confidence"], confidence_target)
    total = metric + 0.5 * success + 0.5 * risk + ranking + 0.1 * calibration
    return total, {"metric": metric, "success": success, "risk": risk, "ranking": ranking, "calibration": calibration}
