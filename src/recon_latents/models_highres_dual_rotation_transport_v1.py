"""Soft rotation transport for the high-resolution dual-latent WM.

Unlike the hard rotation-warp pilot, this model preserves the baseline
prediction exactly at initialization.  It exposes the physically derived
``warped_current - current`` transport as a two-scalar residual, one for each
latent branch.  Real-future latent supervision learns how much of that
rotation-only transport is useful; no outcome gradient is involved.
"""

from __future__ import annotations

import torch
from torch import nn

from recon_latents.models_highres_dual_rotation_warp_v1 import (
    RotationWarpHighResDualLatentWMConfig,
    RotationWarpHighResMultiLayerDualLatentWM,
)
from recon_latents.models_highres_dual_v1 import (
    HighResMultiLayerDualLatentWM,
)


class RotationTransportHighResMultiLayerDualLatentWM(
    RotationWarpHighResMultiLayerDualLatentWM
):
    """Add a learned bounded amount of analytic rotation transport."""

    def __init__(
        self,
        config: RotationWarpHighResDualLatentWMConfig | None = None,
    ) -> None:
        super().__init__(config)
        # tanh keeps the transport bounded to one full analytic warp while
        # allowing the data to correct an unexpected sign mismatch.
        self.rotation_transport_raw_scale = nn.Parameter(torch.zeros(2))

    def forward(
        self,
        semantic_history: torch.Tensor,
        geometry_history: torch.Tensor,
        candidate_actions: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        # Bypass the hard-warp forward while reusing its camera-ray helpers.
        output = HighResMultiLayerDualLatentWM.forward(
            self,
            semantic_history,
            geometry_history,
            candidate_actions,
        )
        cumulative_yaw = output["cumulative_pose"][..., 2]
        current_semantic = semantic_history[:, None, None, -1]
        current_geometry = geometry_history[:, None, None, -1]
        warped_semantic = self._warp_reference(
            semantic_history[:, -1],
            cumulative_yaw,
        )
        warped_geometry = self._warp_reference(
            geometry_history[:, -1],
            cumulative_yaw,
        )
        semantic_transport = warped_semantic - current_semantic
        geometry_transport = warped_geometry - current_geometry
        scale = torch.tanh(self.rotation_transport_raw_scale).to(
            dtype=output["predicted_semantic_tokens"].dtype,
        )
        predicted_semantic = (
            output["predicted_semantic_tokens"]
            + scale[0] * semantic_transport
        )
        predicted_geometry = (
            output["predicted_geometry_tokens"]
            + scale[1] * geometry_transport
        )
        output.update(
            {
                "predicted_semantic_tokens": predicted_semantic,
                "predicted_geometry_tokens": predicted_geometry,
                "predicted_semantic_delta": (
                    predicted_semantic - current_semantic
                ),
                "predicted_geometry_delta": (
                    predicted_geometry - current_geometry
                ),
                "semantic_rotation_transport": semantic_transport,
                "geometry_rotation_transport": geometry_transport,
                "rotation_transport_scale": scale,
            }
        )
        return output
