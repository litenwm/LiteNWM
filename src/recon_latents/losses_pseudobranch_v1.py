"""True-future-latent loss with on-support natural action-branch supervision."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from recon_latents.losses_action_residual_v1 import (
    ActionResidualLossWeights,
    compute_action_residual_world_model_loss,
)


@dataclass(frozen=True)
class PseudoBranchLossWeights(ActionResidualLossWeights):
    branch_delta: float = 0.50
    matched_action_order: float = 0.20


def pool_h4_residual(
    prediction: torch.Tensor,
    history: torch.Tensor,
) -> torch.Tensor:
    value = prediction[:, 0, 3].float() - history[:, -1].float()
    batch = value.shape[0]
    value = value.reshape(batch, 16, 16, 32, 12).mean(dim=-1)
    value = value.reshape(batch, 4, 4, 4, 4, 32).mean(dim=(2, 4))
    return value.reshape(batch, 512)


def compute_pseudobranch_world_model_loss(
    output: dict[str, torch.Tensor],
    target: torch.Tensor,
    history: torch.Tensor,
    actions: torch.Tensor,
    *,
    weights: PseudoBranchLossWeights | None = None,
    corrupted_output: dict[str, torch.Tensor] | None = None,
    prefix_output: dict[str, torch.Tensor] | None = None,
    quality_loss: torch.Tensor | None = None,
    observed_branch_delta: torch.Tensor | None = None,
    branch_similarity_weight: torch.Tensor | None = None,
    residual_scale: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    weights = weights or PseudoBranchLossWeights()
    total, components = compute_action_residual_world_model_loss(
        output,
        target,
        history,
        actions,
        weights=weights,
        corrupted_output=corrupted_output,
        prefix_output=prefix_output,
        quality_loss=quality_loss,
    )
    zero = output["predicted_tokens"].new_zeros(())
    components["branch_delta"] = zero
    components["matched_action_order"] = zero
    if observed_branch_delta is None:
        return total, components
    if corrupted_output is None:
        raise ValueError("pseudo-branch target requires matched branch output")
    if branch_similarity_weight is None or residual_scale is None:
        raise ValueError("pseudo-branch similarity weight and residual scale required")
    correct_residual = pool_h4_residual(output["predicted_tokens"], history)
    branch_residual = pool_h4_residual(
        corrupted_output["predicted_tokens"],
        history,
    )
    scale = max(float(residual_scale), 1.0e-6)
    predicted_delta = (branch_residual - correct_residual) / scale
    target_delta = observed_branch_delta.float() / scale
    weight = branch_similarity_weight.float()
    branch_per_state = F.smooth_l1_loss(
        predicted_delta,
        target_delta,
        reduction="none",
    ).mean(dim=1)
    components["branch_delta"] = (branch_per_state * weight).sum() / weight.sum()
    target_h4 = target[:, 3].float()
    target_energy = (
        (target_h4 - history[:, -1].float())
        .square()
        .mean(dim=(-1, -2))
        .clamp_min(1.0e-4)
    )
    correct_error = (
        (output["predicted_tokens"][:, 0, 3].float() - target_h4)
        .square()
        .mean(dim=(-1, -2))
        / target_energy
    )
    branch_error = (
        (corrupted_output["predicted_tokens"][:, 0, 3].float() - target_h4)
        .square()
        .mean(dim=(-1, -2))
        / target_energy
    )
    order = F.relu(
        0.05 + torch.log1p(correct_error) - torch.log1p(branch_error)
    )
    components["matched_action_order"] = (order * weight).sum() / weight.sum()
    total = (
        total
        + weights.branch_delta * components["branch_delta"]
        + weights.matched_action_order * components["matched_action_order"]
    )
    return total, components


__all__ = [
    "PseudoBranchLossWeights",
    "compute_pseudobranch_world_model_loss",
]
