"""Calibrated SE(3) residual adapter on a frozen strong dual-latent WM.

The wrapped rotation-transport model is initialized from an existing strong
HighResMultiLayerDualLatentWM checkpoint.  Every original parameter remains
frozen.  Only two rotation scales, two translation scales, and the small
token-depth head are trainable.  With all four scales initialized at zero,
the prediction is exactly the original strong checkpoint.
"""

from __future__ import annotations

import torch

from recon_latents.models_highres_dual_depth_translation_v1 import (
    DepthTranslationResidualConfig,
    FrozenRotationTransportDepthTranslationResidual,
)
from recon_latents.models_highres_dual_rotation_transport_v1 import (
    RotationTransportHighResMultiLayerDualLatentWM,
)


class FrozenStrongHighResCalibratedSE3Residual(
    FrozenRotationTransportDepthTranslationResidual
):
    """Train calibrated geometry while preserving all strong-base weights."""

    def __init__(
        self,
        base: RotationTransportHighResMultiLayerDualLatentWM,
        config: DepthTranslationResidualConfig | None = None,
    ) -> None:
        super().__init__(base, config)
        self.base.rotation_transport_raw_scale.requires_grad_(True)

    def forward(
        self,
        semantic_history: torch.Tensor,
        geometry_history: torch.Tensor,
        candidate_actions: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        self.base.eval()
        # Original base parameters are frozen, but the base's two zero-start
        # rotation transport scales must retain gradients.
        base_output = self.base(
            semantic_history,
            geometry_history,
            candidate_actions,
        )
        cumulative = base_output["cumulative_pose"].detach()
        depth = self._depth(
            base_output["geometry_state_tokens"].detach()
        )
        yaw = cumulative[..., 2]
        semantic_current = semantic_history[:, -1]
        geometry_current = geometry_history[:, -1]
        semantic_se3 = self._warp_se3(
            semantic_current,
            cumulative,
            depth,
        )
        geometry_se3 = self._warp_se3(
            geometry_current,
            cumulative,
            depth,
        )
        semantic_rotation = self.base._warp_reference(
            semantic_current,
            yaw,
        ).detach()
        geometry_rotation = self.base._warp_reference(
            geometry_current,
            yaw,
        ).detach()
        semantic_translation = semantic_se3 - semantic_rotation
        geometry_translation = geometry_se3 - geometry_rotation
        translation_scale = torch.tanh(
            self.translation_transport_raw_scale
        ).to(dtype=semantic_translation.dtype)
        predicted_semantic = (
            base_output["predicted_semantic_tokens"]
            + translation_scale[0] * semantic_translation
        )
        predicted_geometry = (
            base_output["predicted_geometry_tokens"]
            + translation_scale[1] * geometry_translation
        )
        current_semantic = semantic_current[:, None, None]
        current_geometry = geometry_current[:, None, None]
        return {
            **base_output,
            "predicted_semantic_tokens": predicted_semantic,
            "predicted_geometry_tokens": predicted_geometry,
            "predicted_semantic_delta": (
                predicted_semantic - current_semantic
            ),
            "predicted_geometry_delta": (
                predicted_geometry - current_geometry
            ),
            "semantic_translation_transport": semantic_translation,
            "geometry_translation_transport": geometry_translation,
            "predicted_depth_m": depth,
            "translation_transport_scale": translation_scale,
        }

    def adapter_state_dict(self) -> dict[str, torch.Tensor]:
        values = super().adapter_state_dict()
        values["base.rotation_transport_raw_scale"] = (
            self.base.rotation_transport_raw_scale.detach().clone()
        )
        return values
