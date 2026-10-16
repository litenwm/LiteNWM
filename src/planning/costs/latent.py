"""Transparent costs over frozen predicted latent trajectories."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F


EPS = 1.0e-8


def _validate(
    predicted_future: torch.Tensor,
    goal_latent: torch.Tensor,
    current_latent: torch.Tensor | None = None,
) -> None:
    if predicted_future.ndim != 5:
        raise ValueError(
            "expected predicted_future [B,K,H,256,384], got "
            f"{tuple(predicted_future.shape)}"
        )
    if goal_latent.ndim != 3:
        raise ValueError(f"expected goal_latent [B,256,384], got {goal_latent.shape}")
    if predicted_future.shape[0] != goal_latent.shape[0]:
        raise ValueError("prediction and goal batch dimensions differ")
    if predicted_future.shape[-2:] != goal_latent.shape[-2:]:
        raise ValueError("prediction and goal latent shapes differ")
    if current_latent is not None and current_latent.shape != goal_latent.shape:
        raise ValueError("current and goal latent shapes differ")


def _per_horizon_goal_nmse(
    predicted_future: torch.Tensor,
    goal_latent: torch.Tensor,
) -> torch.Tensor:
    goal = goal_latent[:, None, None]
    mse = (predicted_future - goal).square().mean(dim=(-1, -2))
    denominator = goal.square().mean(dim=(-1, -2)).clamp_min(EPS)
    return mse / denominator


def terminal_latent_distance(
    predicted_future: torch.Tensor,
    goal_latent: torch.Tensor,
) -> torch.Tensor:
    """Normalized MSE from the H4 prediction to the supplied goal latent."""
    _validate(predicted_future, goal_latent)
    return _per_horizon_goal_nmse(predicted_future, goal_latent)[..., -1]


def trajectory_latent_distance(
    predicted_future: torch.Tensor,
    goal_latent: torch.Tensor,
    horizon_weights: Sequence[float] | None = None,
) -> torch.Tensor:
    """Weighted mean goal distance across all predicted H1-H4 latents."""
    _validate(predicted_future, goal_latent)
    distances = _per_horizon_goal_nmse(predicted_future, goal_latent)
    horizon = distances.shape[-1]
    if horizon_weights is None:
        weights = torch.ones(horizon, device=distances.device, dtype=distances.dtype)
    else:
        if len(horizon_weights) != horizon:
            raise ValueError("horizon_weights length differs from rollout horizon")
        weights = torch.as_tensor(
            horizon_weights,
            device=distances.device,
            dtype=distances.dtype,
        )
        if not torch.isfinite(weights).all() or bool((weights < 0).any()):
            raise ValueError("horizon_weights must be finite and non-negative")
    weights = weights / weights.sum().clamp_min(EPS)
    return (distances * weights).sum(dim=-1)


def latent_trajectory_features(
    predicted_future: torch.Tensor,
    goal_latent: torch.Tensor,
    current_latent: torch.Tensor,
) -> tuple[torch.Tensor, tuple[str, ...]]:
    """Extract deployment-safe H1-H4 statistics for the fixed-pool readout.

    Features use only current, goal, and predicted latents. No target future or
    oracle label enters feature computation.
    """
    _validate(predicted_future, goal_latent, current_latent)
    goal = goal_latent[:, None, None]
    current = current_latent[:, None, None]
    goal_energy = goal.square().mean(dim=(-1, -2)).clamp_min(EPS)
    current_energy = current.square().mean(dim=(-1, -2)).clamp_min(EPS)
    goal_nmse = (predicted_future - goal).square().mean(dim=(-1, -2)) / goal_energy
    current_nmse = (
        (predicted_future - current).square().mean(dim=(-1, -2)) / current_energy
    )
    goal_cosine = 1.0 - F.cosine_similarity(
        predicted_future.flatten(-2),
        goal.expand_as(predicted_future).flatten(-2),
        dim=-1,
    )
    current_goal_nmse = (
        (current_latent - goal_latent).square().mean(dim=(-1, -2))
        / goal_latent.square().mean(dim=(-1, -2)).clamp_min(EPS)
    )[:, None, None]
    progress = current_goal_nmse - goal_nmse
    current_per_candidate = current.expand(
        -1, predicted_future.shape[1], -1, -1, -1
    )
    previous = torch.cat((current_per_candidate, predicted_future[:, :, :-1]), dim=2)
    step_delta = predicted_future - previous
    step_energy = step_delta.square().mean(dim=(-1, -2)) / current_energy
    goal_direction = goal - previous
    direction_cosine = F.cosine_similarity(
        step_delta.flatten(-2),
        goal_direction.expand_as(step_delta).flatten(-2),
        dim=-1,
    )
    predicted_norm = torch.linalg.vector_norm(predicted_future, dim=(-1, -2))
    goal_norm = torch.linalg.vector_norm(goal_latent, dim=(-1, -2))[:, None, None]
    norm_ratio = predicted_norm / goal_norm.clamp_min(EPS)

    feature_blocks = (
        ("goal_nmse", goal_nmse),
        ("goal_cosine", goal_cosine),
        ("current_nmse", current_nmse),
        ("goal_progress", progress),
        ("step_energy", step_energy),
        ("step_goal_direction_cosine", direction_cosine),
        ("goal_norm_ratio", norm_ratio),
    )
    values = []
    names = []
    for prefix, block in feature_blocks:
        for horizon_index in range(block.shape[-1]):
            values.append(block[..., horizon_index])
            names.append(f"h{horizon_index + 1}_{prefix}")
    features = torch.stack(values, dim=-1)
    if not torch.isfinite(features).all():
        raise RuntimeError("latent trajectory features contain NaN or Inf")
    return features, tuple(names)


__all__ = [
    "latent_trajectory_features",
    "terminal_latent_distance",
    "trajectory_latent_distance",
]
