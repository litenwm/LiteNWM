from __future__ import annotations

import torch

from recon_latents.losses_highres_dual_stable_v1 import (
    compute_highres_dual_latent_loss,
)
from recon_latents.models_highres_dual_v1 import (
    HighResDualLatentWMConfig,
    HighResMultiLayerDualLatentWM,
)


def test_zero_delta_initialization_has_bounded_gradient() -> None:
    torch.manual_seed(7)
    config = HighResDualLatentWMConfig(
        semantic_dim=12,
        geometry_dim=8,
        grid_height=2,
        grid_width=3,
        model_dim=16,
        heads=4,
        rollout_blocks=1,
    )
    model = HighResMultiLayerDualLatentWM(config)
    semantic_history = torch.randn(2, 4, 6, 12)
    geometry_history = torch.randn(2, 4, 6, 8)
    actions = 0.1 * torch.randn(2, 4, 3)
    semantic_future = semantic_history[:, -1, None] + 0.05 * torch.randn(2, 4, 6, 12)
    geometry_future = geometry_history[:, -1, None] + 0.05 * torch.randn(2, 4, 6, 8)
    output = model(semantic_history, geometry_history, actions)
    corrupted = model(semantic_history, geometry_history, actions.flip(1))
    loss, components = compute_highres_dual_latent_loss(
        output,
        semantic_future,
        geometry_future,
        semantic_history,
        geometry_history,
        actions,
        corrupted_output=corrupted,
    )
    loss.backward()
    gradient_norm = torch.sqrt(
        sum(
            parameter.grad.square().sum()
            for parameter in model.parameters()
            if parameter.grad is not None
        )
    )
    assert torch.isfinite(loss)
    assert torch.isfinite(gradient_norm)
    assert float(gradient_norm) < 100.0
    assert float(components["delta_direction"]) == 0.0
    assert float(components["cross_latent_motion"]) == 0.0
