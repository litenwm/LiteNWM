from __future__ import annotations

import torch

from recon_latents.losses_highres_dual_action_residual_stable_v1 import (
    compute_action_residual_loss,
)
from recon_latents.models_highres_dual_action_residual_v1 import (
    ActionResidualConfig,
    FrozenDualLTWMActionResidual,
)
from recon_latents.models_highres_dual_v1 import (
    HighResDualLatentWMConfig,
    HighResMultiLayerDualLatentWM,
)


def test_action_residual_is_exact_zero_at_initialization_and_has_gradients():
    config = HighResDualLatentWMConfig(
        semantic_dim=12,
        geometry_dim=8,
        grid_height=2,
        grid_width=3,
        model_dim=16,
        heads=4,
        action_layers=1,
        rollout_blocks=1,
    )
    base = HighResMultiLayerDualLatentWM(config)
    model = FrozenDualLTWMActionResidual(
        base,
        ActionResidualConfig(residual_dim=12, decoder_dim=8),
    )
    semantic_history = torch.randn(2, 4, 6, 12)
    geometry_history = torch.randn(2, 4, 6, 8)
    semantic_future = torch.randn(2, 4, 6, 12)
    geometry_future = torch.randn(2, 4, 6, 8)
    actions = torch.randn(2, 4, 3)
    negative_actions = torch.randn(2, 3, 4, 3)
    with torch.no_grad():
        reference = base(semantic_history, geometry_history, actions)
    correct = model(semantic_history, geometry_history, actions)
    negatives = model(
        semantic_history,
        geometry_history,
        negative_actions,
    )
    assert torch.equal(
        correct["predicted_semantic_tokens"],
        reference["predicted_semantic_tokens"],
    )
    assert torch.equal(
        correct["predicted_geometry_tokens"],
        reference["predicted_geometry_tokens"],
    )
    loss, components = compute_action_residual_loss(
        correct,
        negatives,
        semantic_future,
        geometry_future,
        semantic_history,
        geometry_history,
        actions,
        negative_actions,
    )
    assert torch.isfinite(loss)
    assert all(torch.isfinite(value) for value in components.values())
    loss.backward()
    residual_gradients = [
        parameter.grad
        for name, parameter in model.named_parameters()
        if not name.startswith("base.") and parameter.grad is not None
    ]
    assert residual_gradients
    assert sum(value.abs().sum() for value in residual_gradients) > 0
    assert all(
        parameter.grad is None
        for name, parameter in model.named_parameters()
        if name.startswith("base.")
    )
