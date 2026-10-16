"""Pure future-latent losses that emphasize action-conditioned change."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from recon_latents import losses_v2 as base


EPS = 1.0e-8


@dataclass(frozen=True)
class ActionResidualLossWeights:
    reconstruction: float = 1.0
    residual_reconstruction: float = 0.35
    dynamic_token: float = 0.25
    cosine: float = 0.05
    delta_direction: float = 0.10
    norm: float = 0.05
    action_order: float = 0.05
    action_residual_order: float = 0.20
    action_decode: float = 0.02
    consistency: float = 0.02
    history: float = 0.03
    velocity: float = 0.05
    uncertainty: float = 0.01
    quality: float = 0.0


def horizon_residual_reconstruction_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    current: torch.Tensor,
) -> torch.Tensor:
    """Reconstruct true future change without static-scene domination."""
    current_expanded = current[:, None, None].expand_as(prediction)
    predicted_residual = prediction - current_expanded
    target_residual = target - current_expanded
    error = (predicted_residual - target_residual).square().mean(dim=(-1, -2))
    target_energy = target_residual.square().mean(dim=(-1, -2)).clamp_min(1.0e-4)
    return torch.log1p(error / target_energy).mean()


def dynamic_token_reconstruction_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    current: torch.Tensor,
) -> torch.Tensor:
    """Upweight spatial tokens whose true future latent changes most."""
    current_expanded = current[:, None, None].expand_as(target)
    token_change = (target - current_expanded).square().mean(dim=-1, keepdim=True)
    mean_change = token_change.mean(dim=-2, keepdim=True).clamp_min(EPS)
    token_weight = (token_change / mean_change).sqrt().clamp(0.25, 4.0).detach()
    error = F.smooth_l1_loss(prediction, target, reduction="none")
    expanded_weight = token_weight.expand_as(error)
    return (error * expanded_weight).sum() / expanded_weight.sum().clamp_min(EPS)


def action_residual_order_contrastive_loss(
    true_prediction: torch.Tensor,
    corrupted_prediction: torch.Tensor | None,
    target: torch.Tensor,
    current: torch.Tensor,
    margin: float = 0.05,
) -> torch.Tensor:
    """Make the executed action explain latent change better than a corrupt action."""
    if corrupted_prediction is None:
        return true_prediction.new_zeros(())
    current_expanded = current[:, None, None].expand_as(true_prediction)
    target_residual = target - current_expanded
    target_energy = target_residual.square().mean(dim=(-1, -2, -3)).clamp_min(1.0e-4)
    true_residual = true_prediction - current_expanded
    corrupt_residual = corrupted_prediction - current_expanded
    true_error = (
        (true_residual - target_residual).square().mean(dim=(-1, -2, -3))
        / target_energy
    )
    corrupt_error = (
        (corrupt_residual - target_residual).square().mean(dim=(-1, -2, -3))
        / target_energy
    )
    return F.relu(
        margin + torch.log1p(true_error) - torch.log1p(corrupt_error)
    ).mean()


def compute_action_residual_world_model_loss(
    output: dict[str, torch.Tensor],
    target: torch.Tensor,
    history: torch.Tensor,
    actions: torch.Tensor,
    *,
    weights: ActionResidualLossWeights | None = None,
    corrupted_output: dict[str, torch.Tensor] | None = None,
    prefix_output: dict[str, torch.Tensor] | None = None,
    quality_loss: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    weights = weights or ActionResidualLossWeights()
    base_weights = base.V2LossWeights(
        reconstruction=weights.reconstruction,
        cosine=weights.cosine,
        delta_direction=weights.delta_direction,
        norm=weights.norm,
        action_order=weights.action_order,
        action_decode=weights.action_decode,
        consistency=weights.consistency,
        history=weights.history,
        velocity=weights.velocity,
        uncertainty=weights.uncertainty,
        quality=weights.quality,
    )
    _, components = base.compute_world_model_v2_loss(
        output,
        target,
        history,
        actions,
        weights=base_weights,
        corrupted_output=corrupted_output,
        prefix_output=prefix_output,
        quality_loss=quality_loss,
    )
    prediction = output["predicted_tokens"]
    if target.ndim == 4:
        target = target[:, None]
    target = target.expand(prediction.shape[0], prediction.shape[1], *target.shape[2:])
    current = history[:, -1]
    components["residual_reconstruction"] = horizon_residual_reconstruction_loss(
        prediction,
        target,
        current,
    )
    components["dynamic_token"] = dynamic_token_reconstruction_loss(
        prediction,
        target,
        current,
    )
    components["action_residual_order"] = action_residual_order_contrastive_loss(
        prediction,
        (
            corrupted_output["predicted_tokens"]
            if corrupted_output is not None
            else None
        ),
        target,
        current,
    )
    total = (
        weights.reconstruction * components["reconstruction"]
        + weights.residual_reconstruction * components["residual_reconstruction"]
        + weights.dynamic_token * components["dynamic_token"]
        + weights.cosine * components["cosine"]
        + weights.delta_direction * components["delta_direction"]
        + weights.norm * components["norm"]
        + weights.action_order * components["action_order"]
        + weights.action_residual_order * components["action_residual_order"]
        + weights.action_decode * components["action_decode"]
        + weights.consistency * components["consistency"]
        + weights.history * components["history"]
        + weights.velocity * components["velocity"]
        + weights.uncertainty * components["uncertainty"]
        + weights.quality * components["quality"]
    )
    return total, components


__all__ = [
    "ActionResidualLossWeights",
    "compute_action_residual_world_model_loss",
]
