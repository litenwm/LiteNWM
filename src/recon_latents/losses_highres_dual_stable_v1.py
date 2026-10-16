"""Numerically stable motion terms for the high-resolution dual-latent loss.

This module patches only the two cosine terms that are undefined at the
model's intentional zero-delta initialization.  The public loss contract and
all weights remain identical to ``losses_highres_dual_v1``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from recon_latents import losses_highres_dual_v1 as _base


def _stable_delta_terms(
    prediction: torch.Tensor,
    target: torch.Tensor,
    current: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    initial = current[:, None, None].expand(
        prediction.shape[0],
        prediction.shape[1],
        1,
        *current.shape[1:],
    )
    pred_previous = torch.cat((initial, prediction[:, :, :-1]), dim=2)
    target_previous = torch.cat((initial, target[:, :, :-1]), dim=2)
    pred_delta = prediction - pred_previous
    target_delta = target - target_previous
    pred_norm = torch.linalg.vector_norm(pred_delta, dim=-1)
    target_norm = torch.linalg.vector_norm(target_delta, dim=-1)
    valid_direction = (
        (target_norm > 1.0e-4)
        & (pred_norm > torch.maximum(0.05 * target_norm, pred_norm.new_tensor(1.0e-4)))
    )
    direction_per_token = 1.0 - F.cosine_similarity(
        pred_delta,
        target_delta,
        dim=-1,
        eps=1.0e-4,
    )
    direction = (
        (direction_per_token * valid_direction).sum()
        / valid_direction.sum().clamp_min(1)
    )
    relative = (pred_norm - target_norm).abs() / (
        pred_norm + target_norm
    ).clamp_min(1.0e-4)
    norm = F.smooth_l1_loss(relative, torch.zeros_like(relative))
    return direction, norm


def _stable_cross_latent_motion(
    semantic: torch.Tensor,
    geometry: torch.Tensor,
    semantic_current: torch.Tensor,
    geometry_current: torch.Tensor,
) -> torch.Tensor:
    semantic_energy = torch.linalg.vector_norm(
        semantic - semantic_current[:, None, None],
        dim=-1,
    )
    geometry_energy = torch.linalg.vector_norm(
        geometry - geometry_current[:, None, None],
        dim=-1,
    )
    semantic_centered = F.layer_norm(
        semantic_energy,
        (semantic_energy.shape[-1],),
    )
    geometry_centered = F.layer_norm(
        geometry_energy,
        (geometry_energy.shape[-1],),
    )
    semantic_norm = torch.linalg.vector_norm(semantic_centered, dim=-1)
    geometry_norm = torch.linalg.vector_norm(geometry_centered, dim=-1)
    valid = (semantic_norm > 1.0e-3) & (geometry_norm > 1.0e-3)
    per_rollout = 1.0 - F.cosine_similarity(
        semantic_centered,
        geometry_centered,
        dim=-1,
        eps=1.0e-4,
    )
    return (per_rollout * valid).sum() / valid.sum().clamp_min(1)


_base._delta_terms = _stable_delta_terms
_base._cross_latent_motion = _stable_cross_latent_motion

HighResDualLatentLossWeights = _base.HighResDualLatentLossWeights
compute_highres_dual_latent_loss = _base.compute_highres_dual_latent_loss

__all__ = [
    "HighResDualLatentLossWeights",
    "compute_highres_dual_latent_loss",
]
