from __future__ import annotations

import torch

from recon_latents.dinov3_adapter import DinoV3TokenAdapter
from recon_latents.dinov3_multilayer_dual import (
    FrozenMultiLayerDualLatentEncoder,
    MultiLayerDualLatentConfig,
    extract_rectangular_patch_tokens,
)
from recon_latents.losses_highres_dual_v1 import (
    compute_highres_dual_latent_loss,
)
from recon_latents.models_highres_dual_v1 import (
    HighResDualLatentWMConfig,
    HighResMultiLayerDualLatentWM,
)


def test_rectangular_patch_extraction_ignores_prefix() -> None:
    hidden = torch.randn(2, 5 + 30, 1024)
    patch = extract_rectangular_patch_tokens(
        hidden,
        grid_height=5,
        grid_width=6,
    )
    assert patch.shape == (2, 30, 1024)
    torch.testing.assert_close(patch, hidden[:, -30:])


def test_multilayer_encoder_is_deterministic_and_dual() -> None:
    config = MultiLayerDualLatentConfig(
        image_height=80,
        image_width=96,
        semantic_block=3,
        geometry_blocks=(0, 1, 2),
        geometry_dim=24,
    )
    first = FrozenMultiLayerDualLatentEncoder(
        DinoV3TokenAdapter(),
        config,
    )
    second = FrozenMultiLayerDualLatentEncoder(
        DinoV3TokenAdapter(),
        config,
    )
    second.semantic_norm.load_state_dict(first.semantic_norm.state_dict())
    second.semantic_projection.load_state_dict(
        first.semantic_projection.state_dict()
    )
    hidden = tuple(torch.randn(2, 5 + 30, 1024) for _ in range(5))
    output = first(hidden)
    repeated = second(hidden)
    assert output["semantic_tokens"].shape == (2, 30, 384)
    assert output["geometry_tokens"].shape == (2, 30, 24)
    torch.testing.assert_close(
        output["geometry_tokens"],
        repeated["geometry_tokens"],
    )
    assert first.geometry_projection_sha256 == second.geometry_projection_sha256
    assert not any(parameter.requires_grad for parameter in first.parameters())


def test_dual_latent_wm_forward_loss_and_gradient() -> None:
    config = HighResDualLatentWMConfig(
        semantic_dim=16,
        geometry_dim=8,
        grid_height=3,
        grid_width=4,
        model_dim=24,
        heads=4,
        action_layers=1,
        rollout_blocks=1,
    )
    model = HighResMultiLayerDualLatentWM(config)
    semantic_history = torch.randn(2, 4, 12, 16)
    geometry_history = torch.randn(2, 4, 12, 8)
    actions = 0.05 * torch.randn(2, 4, 3)
    semantic_future = torch.randn(2, 4, 12, 16)
    geometry_future = torch.randn(2, 4, 12, 8)
    output = model(semantic_history, geometry_history, actions)
    corrupted = model(
        semantic_history,
        geometry_history,
        actions.flip(1),
    )
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
    assert output["predicted_semantic_tokens"].shape == (2, 1, 4, 12, 16)
    assert output["predicted_geometry_tokens"].shape == (2, 1, 4, 12, 8)
    assert output["decoded_action"].shape == (2, 1, 4, 3)
    assert torch.isfinite(loss)
    assert set(components) == {
        "semantic_reconstruction",
        "geometry_reconstruction",
        "semantic_cosine",
        "geometry_cosine",
        "delta_direction",
        "delta_norm",
        "velocity",
        "action_order",
        "action_decode",
        "uncertainty",
        "cross_latent_motion",
    }
    assert model.geometry_delta.weight.grad is not None
    assert model.semantic_delta.weight.grad is not None

