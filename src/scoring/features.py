"""Deployment-safe features for independent trajectory scorers."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from planning.costs import latent_trajectory_features


POOL_CHANNELS = 64
LATENT_CHANNELS = 384
HORIZON = 4
ACTION_DIM = 3


@dataclass(frozen=True)
class ScorerFeatureBatch:
    terminal: torch.Tensor
    trajectory: torch.Tensor
    latent_only: torch.Tensor
    action_only: torch.Tensor
    full: torch.Tensor
    route_summary: torch.Tensor
    names: dict[str, tuple[str, ...]]


def _pool_channels(latent: torch.Tensor) -> torch.Tensor:
    if latent.shape[-1] != LATENT_CHANNELS:
        raise ValueError(f"expected latent channel size {LATENT_CHANNELS}")
    if LATENT_CHANNELS % POOL_CHANNELS:
        raise RuntimeError("latent channels are not divisible by pool channels")
    width = LATENT_CHANNELS // POOL_CHANNELS
    return latent.reshape(*latent.shape[:-1], POOL_CHANNELS, width).mean(dim=-1)


def _integrate_actions(actions: torch.Tensor) -> torch.Tensor:
    x = torch.zeros_like(actions[..., 0, 0])
    y = torch.zeros_like(x)
    yaw = torch.zeros_like(x)
    poses = []
    for step in range(actions.shape[-2]):
        dx, dy, dyaw = actions[..., step, :].unbind(dim=-1)
        cosine = torch.cos(yaw)
        sine = torch.sin(yaw)
        x = x + cosine * dx - sine * dy
        y = y + sine * dx + cosine * dy
        yaw = yaw + dyaw
        poses.append(torch.stack((x, y, yaw), dim=-1))
    return torch.stack(poses, dim=-2)


def action_trajectory_features(
    candidate_actions: torch.Tensor,
) -> tuple[torch.Tensor, tuple[str, ...]]:
    if candidate_actions.ndim != 4 or candidate_actions.shape[-1] != ACTION_DIM:
        raise ValueError("expected candidate_actions [B,K,H,3]")
    if candidate_actions.shape[-2] not in (4, 6):
        raise ValueError("expected H4 or H6 candidate actions")
    actions = candidate_actions[..., :HORIZON, :].float()
    poses = _integrate_actions(actions)
    action_norm = torch.linalg.vector_norm(actions[..., :2], dim=-1)
    yaw_abs = actions[..., 2].abs()
    first_difference = actions[..., 1:, :] - actions[..., :-1, :]
    second_difference = first_difference[..., 1:, :] - first_difference[..., :-1, :]
    path_length = action_norm.sum(dim=-1, keepdim=True)
    roughness = first_difference.square().mean(dim=(-1, -2), keepdim=False)[..., None]
    jerk = second_difference.square().mean(dim=(-1, -2), keepdim=False)[..., None]
    curvature = yaw_abs.sum(dim=-1, keepdim=True) / path_length.clamp_min(1.0e-8)
    summary = torch.cat(
        (
            path_length,
            action_norm.mean(dim=-1, keepdim=True),
            action_norm.amax(dim=-1, keepdim=True),
            yaw_abs.mean(dim=-1, keepdim=True),
            yaw_abs.amax(dim=-1, keepdim=True),
            roughness,
            jerk,
            curvature,
        ),
        dim=-1,
    )
    values = torch.cat(
        (
            actions.flatten(-2),
            poses.flatten(-2),
            action_norm,
            yaw_abs,
            summary,
        ),
        dim=-1,
    )
    names = []
    for step in range(HORIZON):
        for dimension in ("dx", "dy", "dyaw"):
            names.append(f"a{step + 1}_{dimension}")
    for step in range(HORIZON):
        for dimension in ("x", "y", "yaw"):
            names.append(f"pose{step + 1}_{dimension}")
    names.extend(f"a{step + 1}_translation_norm" for step in range(HORIZON))
    names.extend(f"a{step + 1}_abs_yaw" for step in range(HORIZON))
    names.extend(
        (
            "path_length",
            "translation_norm_mean",
            "translation_norm_max",
            "abs_yaw_mean",
            "abs_yaw_max",
            "roughness",
            "jerk",
            "curvature",
        )
    )
    if values.shape[-1] != len(names):
        raise RuntimeError("action feature names do not match values")
    return values, tuple(names)


def extract_scorer_features(
    predicted_future: torch.Tensor,
    goal_latent: torch.Tensor,
    current_latent: torch.Tensor,
    candidate_actions: torch.Tensor,
) -> ScorerFeatureBatch:
    """Extract shared candidate features without evaluation-only identity fields."""
    if predicted_future.ndim != 5 or predicted_future.shape[2:] != (
        HORIZON,
        256,
        LATENT_CHANNELS,
    ):
        raise ValueError("expected predicted future [B,K,4,256,384]")
    if goal_latent.shape != (predicted_future.shape[0], 256, LATENT_CHANNELS):
        raise ValueError("goal latent shape differs from prediction")
    if current_latent.shape != goal_latent.shape:
        raise ValueError("current latent shape differs from goal")
    if candidate_actions.shape[:2] != predicted_future.shape[:2]:
        raise ValueError("candidate action dimensions differ from prediction")

    predicted_pooled = _pool_channels(predicted_future.mean(dim=-2))
    goal_pooled = _pool_channels(goal_latent.mean(dim=-2))[:, None, None]
    current_pooled = _pool_channels(current_latent.mean(dim=-2))[:, None, None]
    current_per_candidate = current_pooled.expand(
        -1, predicted_future.shape[1], -1, -1
    )
    previous_pooled = torch.cat(
        (current_per_candidate, predicted_pooled[:, :, :-1]), dim=2
    )
    goal_difference = predicted_pooled - goal_pooled
    step_delta = predicted_pooled - previous_pooled
    scalar, scalar_names = latent_trajectory_features(
        predicted_future, goal_latent, current_latent
    )

    trajectory_blocks = (
        ("pred", predicted_pooled),
        ("goal_diff", goal_difference),
        ("step_delta", step_delta),
    )
    trajectory_values = []
    trajectory_names = []
    for prefix, block in trajectory_blocks:
        trajectory_values.append(block.flatten(-2))
        for step in range(HORIZON):
            trajectory_names.extend(
                f"h{step + 1}_{prefix}_pool{channel:02d}"
                for channel in range(POOL_CHANNELS)
            )
    trajectory_values.append(scalar)
    trajectory_names.extend(scalar_names)
    trajectory = torch.cat(trajectory_values, dim=-1)

    h4_scalar_names = tuple(name for name in scalar_names if name.startswith("h4_"))
    h4_scalar_indices = [scalar_names.index(name) for name in h4_scalar_names]
    terminal = torch.cat(
        (
            predicted_pooled[:, :, -1],
            goal_difference[:, :, -1],
            step_delta[:, :, -1],
            scalar[..., h4_scalar_indices],
        ),
        dim=-1,
    )
    terminal_names = (
        tuple(f"h4_pred_pool{channel:02d}" for channel in range(POOL_CHANNELS))
        + tuple(
            f"h4_goal_diff_pool{channel:02d}" for channel in range(POOL_CHANNELS)
        )
        + tuple(
            f"h4_step_delta_pool{channel:02d}" for channel in range(POOL_CHANNELS)
        )
        + h4_scalar_names
    )

    action, action_names = action_trajectory_features(candidate_actions)
    full = torch.cat((trajectory, action), dim=-1)
    full_names = tuple(trajectory_names) + action_names
    route_summary = (current_pooled - goal_pooled)[:, 0, 0]
    route_summary = route_summary[:, None].expand(-1, predicted_future.shape[1], -1)
    route_summary_names = tuple(
        f"current_goal_diff_pool{channel:02d}" for channel in range(POOL_CHANNELS)
    )
    names = {
        "terminal": terminal_names,
        "trajectory": tuple(trajectory_names),
        "latent_only": tuple(trajectory_names),
        "action_only": action_names,
        "full": full_names,
        "route_summary": route_summary_names,
    }
    tensors = {
        "terminal": terminal,
        "trajectory": trajectory,
        "latent_only": trajectory,
        "action_only": action,
        "full": full,
        "route_summary": route_summary,
    }
    for key, value in tensors.items():
        if value.shape[-1] != len(names[key]):
            raise RuntimeError(f"{key} feature names do not match values")
        if not torch.isfinite(value).all():
            raise RuntimeError(f"{key} features contain NaN or Inf")
    return ScorerFeatureBatch(names=names, **tensors)


__all__ = [
    "ScorerFeatureBatch",
    "action_trajectory_features",
    "extract_scorer_features",
]
