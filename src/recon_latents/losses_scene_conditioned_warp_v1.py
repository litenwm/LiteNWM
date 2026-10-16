"""True-future-latent objectives for the state-conditioned token warp."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from recon_latents import losses_v2 as base
from recon_latents.losses_action_residual_v1 import (
    dynamic_token_reconstruction_loss,
    horizon_residual_reconstruction_loss,
)


EPS = 1.0e-8


@dataclass(frozen=True)
class SceneWarpLossWeights:
    reconstruction: float = 1.0
    residual_reconstruction: float = 0.25
    dynamic_token: float = 0.25
    reference_reconstruction: float = 0.50
    action_order: float = 0.10
    cosine: float = 0.05
    delta_direction: float = 0.05
    norm: float = 0.02
    consistency: float = 0.02
    history: float = 0.02
    velocity: float = 0.03
    uncertainty: float = 0.01
    flow_smoothness: float = 0.01
    flow_magnitude: float = 0.001
    quality: float = 0.0


def reference_reconstruction_loss(
    reference: torch.Tensor,
    target: torch.Tensor,
    current: torch.Tensor,
) -> torch.Tensor:
    current_expanded = current[:, None, None].expand_as(reference)
    target_residual = target - current_expanded
    residual_energy = target_residual.square().mean(dim=(-1, -2)).clamp_min(1.0e-4)
    error = F.smooth_l1_loss(reference, target, reduction="none").mean(dim=(-1, -2))
    return torch.log1p(error / residual_energy).mean()


def flow_regularizers(offset: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    grid = offset.reshape(*offset.shape[:-2], 16, 16, 2)
    horizontal = (grid[..., :, 1:, :] - grid[..., :, :-1, :]).square().mean()
    vertical = (grid[..., 1:, :, :] - grid[..., :-1, :, :]).square().mean()
    return horizontal + vertical, offset.square().mean()


def compute_scene_warp_world_model_loss(
    output: dict[str, torch.Tensor],
    target: torch.Tensor,
    history: torch.Tensor,
    actions: torch.Tensor,
    *,
    weights: SceneWarpLossWeights | None = None,
    corrupted_output: dict[str, torch.Tensor] | None = None,
    prefix_output: dict[str, torch.Tensor] | None = None,
    quality_loss: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    weights = weights or SceneWarpLossWeights()
    base_weights = base.V2LossWeights(
        reconstruction=weights.reconstruction,
        cosine=weights.cosine,
        delta_direction=weights.delta_direction,
        norm=weights.norm,
        action_order=weights.action_order,
        action_decode=0.0,
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
    components["reference_reconstruction"] = reference_reconstruction_loss(
        output["reference_tokens"],
        target,
        current,
    )
    smoothness, magnitude = flow_regularizers(output["scene_flow_offset"])
    components["flow_smoothness"] = smoothness
    components["flow_magnitude"] = magnitude
    total = (
        weights.reconstruction * components["reconstruction"]
        + weights.residual_reconstruction * components["residual_reconstruction"]
        + weights.dynamic_token * components["dynamic_token"]
        + weights.reference_reconstruction * components["reference_reconstruction"]
        + weights.action_order * components["action_order"]
        + weights.cosine * components["cosine"]
        + weights.delta_direction * components["delta_direction"]
        + weights.norm * components["norm"]
        + weights.consistency * components["consistency"]
        + weights.history * components["history"]
        + weights.velocity * components["velocity"]
        + weights.uncertainty * components["uncertainty"]
        + weights.flow_smoothness * components["flow_smoothness"]
        + weights.flow_magnitude * components["flow_magnitude"]
        + weights.quality * components["quality"]
    )
    return total, components


__all__ = [
    "SceneWarpLossWeights",
    "compute_scene_warp_world_model_loss",
]
