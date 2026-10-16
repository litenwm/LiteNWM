"""Action-identifiability losses without navigation-outcome supervision."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


EPS = 1.0e-8


@dataclass(frozen=True)
class ActionResidualLossWeights:
    reconstruction: float = 1.0
    motion_delta: float = 0.25
    latent_cosine: float = 0.10
    action_contrastive: float = 1.0
    action_margin: float = 0.50
    latent_action_decode: float = 0.25
    residual_energy: float = 0.001
    contrastive_temperature: float = 0.04
    ranking_margin: float = 0.02


def per_candidate_dual_nmse(
    output: dict[str, torch.Tensor],
    semantic_target: torch.Tensor,
    geometry_target: torch.Tensor,
) -> torch.Tensor:
    semantic = output["predicted_semantic_tokens"].float()
    geometry = output["predicted_geometry_tokens"].float()
    semantic_target = semantic_target[:, None].float()
    geometry_target = geometry_target[:, None].float()
    semantic_error = (
        (semantic - semantic_target).square().mean(dim=(-1, -2))
        / semantic_target.square().mean(dim=(-1, -2)).clamp_min(EPS)
    )
    geometry_error = (
        (geometry - geometry_target).square().mean(dim=(-1, -2))
        / geometry_target.square().mean(dim=(-1, -2)).clamp_min(EPS)
    )
    return 0.5 * (semantic_error + geometry_error)


def _motion_delta_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    current: torch.Tensor,
) -> torch.Tensor:
    prediction_delta = prediction.float() - current[:, None, None].float()
    target_delta = target[:, None].float() - current[:, None, None].float()
    denominator = target_delta.square().mean().clamp_min(1.0e-5)
    return F.smooth_l1_loss(prediction_delta, target_delta) / denominator.sqrt()


def compute_action_residual_loss(
    correct: dict[str, torch.Tensor],
    negatives: dict[str, torch.Tensor],
    semantic_target: torch.Tensor,
    geometry_target: torch.Tensor,
    semantic_history: torch.Tensor,
    geometry_history: torch.Tensor,
    correct_actions: torch.Tensor,
    negative_actions: torch.Tensor,
    *,
    weights: ActionResidualLossWeights | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    weights = weights or ActionResidualLossWeights()
    semantic = correct["predicted_semantic_tokens"]
    geometry = correct["predicted_geometry_tokens"]
    semantic_expanded = semantic_target[:, None]
    geometry_expanded = geometry_target[:, None]
    correct_error = per_candidate_dual_nmse(
        correct,
        semantic_target,
        geometry_target,
    ).mean(dim=-1)
    negative_error = per_candidate_dual_nmse(
        negatives,
        semantic_target,
        geometry_target,
    ).mean(dim=-1)
    logits = -torch.cat((correct_error, negative_error), dim=1)
    logits = logits / weights.contrastive_temperature
    contrastive = F.cross_entropy(
        logits,
        torch.zeros(logits.shape[0], device=logits.device, dtype=torch.long),
    )
    margin = F.relu(
        weights.ranking_margin
        + correct_error
        - negative_error
    ).mean()

    decoded = torch.cat(
        (
            correct["decoded_action_from_predicted_latent"],
            negatives["decoded_action_from_predicted_latent"],
        ),
        dim=1,
    )
    action_target = torch.cat(
        (correct_actions[:, None], negative_actions),
        dim=1,
    )
    components = {
        "reconstruction": correct_error.mean(),
        "motion_delta": 0.5
        * (
            _motion_delta_loss(
                semantic,
                semantic_expanded,
                semantic_history[:, -1],
            )
            + _motion_delta_loss(
                geometry,
                geometry_expanded,
                geometry_history[:, -1],
            )
        ),
        "latent_cosine": 0.5
        * (
            (
                1.0
                - F.cosine_similarity(
                    semantic.float(),
                    semantic_expanded.float(),
                    dim=-1,
                )
            ).mean()
            + (
                1.0
                - F.cosine_similarity(
                    geometry.float(),
                    geometry_expanded.float(),
                    dim=-1,
                )
            ).mean()
        ),
        "action_contrastive": contrastive,
        "action_margin": margin,
        "latent_action_decode": F.smooth_l1_loss(decoded, action_target),
        "residual_energy": 0.5
        * (
            correct["semantic_action_residual"].float().square().mean()
            + correct["geometry_action_residual"].float().square().mean()
            + negatives["semantic_action_residual"].float().square().mean()
            + negatives["geometry_action_residual"].float().square().mean()
        ),
    }
    total = (
        weights.reconstruction * components["reconstruction"]
        + weights.motion_delta * components["motion_delta"]
        + weights.latent_cosine * components["latent_cosine"]
        + weights.action_contrastive * components["action_contrastive"]
        + weights.action_margin * components["action_margin"]
        + weights.latent_action_decode * components["latent_action_decode"]
        + weights.residual_energy * components["residual_energy"]
    )
    return total, components


__all__ = [
    "ActionResidualLossWeights",
    "compute_action_residual_loss",
    "per_candidate_dual_nmse",
]
