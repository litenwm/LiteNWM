from __future__ import annotations

import torch

from recon_latents.models_highres_dual_rotation_transport_v1 import (
    RotationTransportHighResMultiLayerDualLatentWM,
)
from recon_latents.models_highres_dual_rotation_warp_v1 import (
    RotationWarpHighResDualLatentWMConfig,
)
from recon_latents.models_highres_dual_v1 import (
    HighResMultiLayerDualLatentWM,
)


def config() -> RotationWarpHighResDualLatentWMConfig:
    return RotationWarpHighResDualLatentWMConfig(
        semantic_dim=12,
        geometry_dim=8,
        grid_height=3,
        grid_width=4,
        model_dim=16,
        heads=4,
        rollout_blocks=1,
        image_width=64,
        image_height=48,
        effective_focal_px=42.6,
        principal_x_px=32.0,
        principal_y_px=24.0,
    )


def inputs() -> tuple[torch.Tensor, ...]:
    semantic = torch.randn(2, 4, 12, 12)
    geometry = torch.randn(2, 4, 12, 8)
    actions = torch.randn(2, 2, 4, 3) * 0.05
    return semantic, geometry, actions


def test_zero_transport_is_exact_baseline_forward() -> None:
    torch.manual_seed(21)
    model = RotationTransportHighResMultiLayerDualLatentWM(config()).eval()
    semantic, geometry, actions = inputs()
    expected = HighResMultiLayerDualLatentWM.forward(
        model,
        semantic,
        geometry,
        actions,
    )
    observed = model(semantic, geometry, actions)
    assert torch.equal(
        observed["predicted_semantic_tokens"],
        expected["predicted_semantic_tokens"],
    )
    assert torch.equal(
        observed["predicted_geometry_tokens"],
        expected["predicted_geometry_tokens"],
    )
    assert torch.equal(
        observed["rotation_transport_scale"],
        torch.zeros(2),
    )


def test_transport_scale_receives_reconstruction_gradient() -> None:
    torch.manual_seed(22)
    model = RotationTransportHighResMultiLayerDualLatentWM(config())
    semantic, geometry, actions = inputs()
    output = model(semantic, geometry, actions)
    loss = (
        output["predicted_semantic_tokens"].square().mean()
        + output["predicted_geometry_tokens"].square().mean()
    )
    loss.backward()
    gradient = model.rotation_transport_raw_scale.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert gradient.abs().sum() > 0


def test_only_two_extra_trainable_scalars() -> None:
    model = RotationTransportHighResMultiLayerDualLatentWM(config())
    named = dict(model.named_parameters())
    assert named["rotation_transport_raw_scale"].numel() == 2
