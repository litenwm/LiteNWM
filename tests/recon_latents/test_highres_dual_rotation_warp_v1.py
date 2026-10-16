from __future__ import annotations

import torch

from recon_latents.models_highres_dual_rotation_warp_v1 import (
    RotationWarpHighResDualLatentWMConfig,
    RotationWarpHighResMultiLayerDualLatentWM,
)


def small_model() -> RotationWarpHighResMultiLayerDualLatentWM:
    config = RotationWarpHighResDualLatentWMConfig(
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
    return RotationWarpHighResMultiLayerDualLatentWM(config)


def test_zero_yaw_warp_is_identity() -> None:
    torch.manual_seed(12)
    model = small_model().eval()
    reference = torch.randn(2, 12, 7)
    yaw = torch.zeros(2, 3, 4)
    warped = model._warp_reference(reference, yaw)
    expected = reference[:, None, None].expand_as(warped)
    assert torch.allclose(warped, expected, atol=2.0e-6, rtol=1.0e-6)


def test_candidate_yaw_changes_reference_and_preserves_contract() -> None:
    torch.manual_seed(13)
    model = small_model().eval()
    semantic = torch.randn(2, 4, 12, 12)
    geometry = torch.randn(2, 4, 12, 8)
    actions = torch.zeros(2, 2, 4, 3)
    actions[:, 1, :, 2] = 0.05
    output = model(semantic, geometry, actions)
    assert output["predicted_semantic_tokens"].shape == (2, 2, 4, 12, 12)
    assert output["predicted_geometry_tokens"].shape == (2, 2, 4, 12, 8)
    assert not torch.equal(
        output["warped_geometry_reference"][:, 0],
        output["warped_geometry_reference"][:, 1],
    )


def test_no_extra_trainable_parameters_over_baseline() -> None:
    model = small_model()
    trainable = sum(value.numel() for value in model.parameters())
    buffers = dict(model.named_buffers())
    assert trainable > 0
    assert "camera_yaw_axis" in buffers
    assert "output_camera_rays" in buffers
