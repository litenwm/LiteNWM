"""Losses for real-future semantic/geometry dual-latent prediction."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


EPS = 1.0e-8


@dataclass(frozen=True)
class HighResDualLatentLossWeights:
    semantic_reconstruction: float = 1.0
    geometry_reconstruction: float = 1.0
    semantic_cosine: float = 0.05
    geometry_cosine: float = 0.10
    delta_direction: float = 0.05
    delta_norm: float = 0.05
    velocity: float = 0.02
    action_order: float = 0.10
    action_decode: float = 0.05
    uncertainty: float = 0.01
    cross_latent_motion: float = 0.05


def _expand_target(
    target: torch.Tensor,
    prediction: torch.Tensor,
) -> torch.Tensor:
    if target.ndim == 4:
        target = target[:, None]
    return target.expand(
        prediction.shape[0],
        prediction.shape[1],
        *target.shape[2:],
    )


def _nmse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (prediction - target).square().mean() / target.square().mean().clamp_min(
        EPS
    )


def _cosine(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (
        1.0
        - F.cosine_similarity(
            prediction,
            target,
            dim=-1,
        )
    ).mean()


def _delta_terms(
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
    direction = (
        1.0
        - F.cosine_similarity(
            pred_delta,
            target_delta,
            dim=-1,
        )
    ).mean()
    pred_norm = torch.linalg.vector_norm(pred_delta, dim=-1)
    target_norm = torch.linalg.vector_norm(target_delta, dim=-1)
    relative = (pred_norm - target_norm).abs() / (
        pred_norm + target_norm
    ).clamp_min(1.0e-4)
    norm = F.smooth_l1_loss(relative, torch.zeros_like(relative))
    return direction, norm


def _velocity(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if prediction.shape[2] < 3:
        return prediction.new_zeros(())
    pred_velocity = prediction[:, :, 1:] - prediction[:, :, :-1]
    target_velocity = target[:, :, 1:] - target[:, :, :-1]
    pred_acceleration = pred_velocity[:, :, 1:] - pred_velocity[:, :, :-1]
    target_acceleration = target_velocity[:, :, 1:] - target_velocity[:, :, :-1]
    return F.smooth_l1_loss(pred_acceleration, target_acceleration)


def _uncertainty(
    prediction: torch.Tensor,
    target: torch.Tensor,
    logvar: torch.Tensor,
) -> torch.Tensor:
    error = (prediction - target).square().mean(dim=-1)
    return (torch.exp(-logvar) * error + logvar).mean()


def _action_order(
    semantic: torch.Tensor,
    geometry: torch.Tensor,
    corrupted_semantic: torch.Tensor | None,
    corrupted_geometry: torch.Tensor | None,
    semantic_target: torch.Tensor,
    geometry_target: torch.Tensor,
    margin: float = 0.02,
) -> torch.Tensor:
    if corrupted_semantic is None or corrupted_geometry is None:
        return semantic.new_zeros(())
    correct = 0.5 * (
        (semantic - semantic_target).square().mean(dim=(-1, -2))
        + (geometry - geometry_target).square().mean(dim=(-1, -2))
    )
    corrupted = 0.5 * (
        (corrupted_semantic - semantic_target).square().mean(dim=(-1, -2))
        + (corrupted_geometry - geometry_target).square().mean(dim=(-1, -2))
    )
    return F.relu(margin + correct - corrupted).mean()


def _cross_latent_motion(
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
    semantic_energy = F.layer_norm(
        semantic_energy,
        (semantic_energy.shape[-1],),
    )
    geometry_energy = F.layer_norm(
        geometry_energy,
        (geometry_energy.shape[-1],),
    )
    return (
        1.0
        - F.cosine_similarity(
            semantic_energy,
            geometry_energy,
            dim=-1,
        )
    ).mean()


def compute_highres_dual_latent_loss(
    output: dict[str, torch.Tensor],
    semantic_target: torch.Tensor,
    geometry_target: torch.Tensor,
    semantic_history: torch.Tensor,
    geometry_history: torch.Tensor,
    actions: torch.Tensor,
    *,
    weights: HighResDualLatentLossWeights | None = None,
    corrupted_output: dict[str, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    weights = weights or HighResDualLatentLossWeights()
    semantic = output["predicted_semantic_tokens"]
    geometry = output["predicted_geometry_tokens"]
    semantic_target = _expand_target(semantic_target, semantic)
    geometry_target = _expand_target(geometry_target, geometry)
    if actions.ndim == 3:
        actions = actions[:, None]
    semantic_current = semantic_history[:, -1]
    geometry_current = geometry_history[:, -1]
    semantic_direction, semantic_norm = _delta_terms(
        semantic,
        semantic_target,
        semantic_current,
    )
    geometry_direction, geometry_norm = _delta_terms(
        geometry,
        geometry_target,
        geometry_current,
    )
    components = {
        "semantic_reconstruction": F.smooth_l1_loss(
            semantic,
            semantic_target,
        )
        + _nmse(semantic, semantic_target),
        "geometry_reconstruction": F.smooth_l1_loss(
            geometry,
            geometry_target,
        )
        + _nmse(geometry, geometry_target),
        "semantic_cosine": _cosine(semantic, semantic_target),
        "geometry_cosine": _cosine(geometry, geometry_target),
        "delta_direction": 0.5 * (
            semantic_direction + geometry_direction
        ),
        "delta_norm": 0.5 * (semantic_norm + geometry_norm),
        "velocity": 0.5 * (
            _velocity(semantic, semantic_target)
            + _velocity(geometry, geometry_target)
        ),
        "action_order": _action_order(
            semantic,
            geometry,
            (
                corrupted_output["predicted_semantic_tokens"]
                if corrupted_output is not None
                else None
            ),
            (
                corrupted_output["predicted_geometry_tokens"]
                if corrupted_output is not None
                else None
            ),
            semantic_target,
            geometry_target,
        ),
        "action_decode": F.smooth_l1_loss(
            output["decoded_action"],
            actions,
        ),
        "uncertainty": 0.5 * (
            _uncertainty(
                semantic,
                semantic_target,
                output["semantic_logvar"],
            )
            + _uncertainty(
                geometry,
                geometry_target,
                output["geometry_logvar"],
            )
        ),
        "cross_latent_motion": _cross_latent_motion(
            semantic,
            geometry,
            semantic_current,
            geometry_current,
        ),
    }
    total = sum(
        getattr(weights, name) * value
        for name, value in components.items()
    )
    return total, components

