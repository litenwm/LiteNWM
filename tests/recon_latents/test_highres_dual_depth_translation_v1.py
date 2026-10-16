from __future__ import annotations

import torch

from recon_latents.models_highres_dual_depth_translation_v1 import (
    DepthTranslationResidualConfig,
    FrozenRotationTransportDepthTranslationResidual,
)
from recon_latents.models_highres_dual_rotation_transport_v1 import (
    RotationTransportHighResMultiLayerDualLatentWM,
)
from recon_latents.models_highres_dual_rotation_warp_v1 import (
    RotationWarpHighResDualLatentWMConfig,
)


def model() -> FrozenRotationTransportDepthTranslationResidual:
    base_config = RotationWarpHighResDualLatentWMConfig(
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
    base = RotationTransportHighResMultiLayerDualLatentWM(base_config)
    return FrozenRotationTransportDepthTranslationResidual(
        base,
        DepthTranslationResidualConfig(),
    )


def inputs() -> tuple[torch.Tensor, ...]:
    semantic = torch.randn(2, 4, 12, 12)
    geometry = torch.randn(2, 4, 12, 8)
    actions = torch.randn(2, 2, 4, 3) * 0.05
    return semantic, geometry, actions


def test_zero_translation_transport_is_exact_frozen_base() -> None:
    torch.manual_seed(31)
    value = model().eval()
    semantic, geometry, actions = inputs()
    with torch.no_grad():
        expected = value.base(semantic, geometry, actions)
    observed = value(semantic, geometry, actions)
    assert torch.equal(
        observed["predicted_semantic_tokens"],
        expected["predicted_semantic_tokens"],
    )
    assert torch.equal(
        observed["predicted_geometry_tokens"],
        expected["predicted_geometry_tokens"],
    )
    assert torch.allclose(
        observed["predicted_depth_m"],
        torch.full((2, 12), 10.0),
        atol=1.0e-5,
    )


def test_only_adapter_parameters_are_trainable() -> None:
    value = model()
    trainable = {
        name for name, parameter in value.named_parameters()
        if parameter.requires_grad
    }
    assert trainable == {
        "translation_transport_raw_scale",
        "depth_norm.weight",
        "depth_norm.bias",
        "depth_logit.weight",
        "depth_logit.bias",
    }


def test_depth_and_transport_receive_future_gradient() -> None:
    torch.manual_seed(32)
    value = model()
    semantic, geometry, actions = inputs()
    value.translation_transport_raw_scale.data.fill_(0.05)
    output = value(semantic, geometry, actions)
    loss = (
        output["predicted_semantic_tokens"].square().mean()
        + output["predicted_geometry_tokens"].square().mean()
    )
    loss.backward()
    assert value.translation_transport_raw_scale.grad is not None
    assert value.depth_logit.weight.grad is not None
    assert torch.isfinite(value.depth_logit.weight.grad).all()
