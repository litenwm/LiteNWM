"""Stable execution wrapper for the action-residual loss."""

from __future__ import annotations

import torch

from recon_latents import losses_highres_dual_action_residual_v1 as _base


_original_motion_delta_loss = _base._motion_delta_loss


def _stable_motion_delta_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    current: torch.Tensor,
) -> torch.Tensor:
    if target.ndim == prediction.ndim and target.shape[1] == 1:
        target = target[:, 0]
    return _original_motion_delta_loss(prediction, target, current)


_base._motion_delta_loss = _stable_motion_delta_loss

ActionResidualLossWeights = _base.ActionResidualLossWeights
compute_action_residual_loss = _base.compute_action_residual_loss
per_candidate_dual_nmse = _base.per_candidate_dual_nmse

__all__ = [
    "ActionResidualLossWeights",
    "compute_action_residual_loss",
    "per_candidate_dual_nmse",
]
