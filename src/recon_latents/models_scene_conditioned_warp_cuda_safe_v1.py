"""Float32 CUDA-kernel boundary for the frozen scene-conditioned warp design."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from recon_latents.models_scene_conditioned_warp_v1 import (
    LiteTokenFormerConfig,
    LiteTokenFormerWMSceneWarpV1,
    SceneConditionedTokenWarp,
)


class Float32SceneConditionedTokenWarp(SceneConditionedTokenWarp):
    """Run the small flow path and grid-sampling kernel outside bf16 autocast."""

    def forward(
        self,
        current_tokens: torch.Tensor,
        action_features: torch.Tensor,
        base_theta: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if current_tokens.shape[-2:] != (256, 384):
            raise ValueError(
                f"expected current tokens [B,256,384], got {tuple(current_tokens.shape)}"
            )
        if action_features.shape[-2:] != (4, 12):
            raise ValueError(
                f"expected action features [B,K,4,12], got {tuple(action_features.shape)}"
            )
        with torch.autocast(device_type=current_tokens.device.type, enabled=False):
            current = current_tokens.float()
            action_input = action_features.float()
            theta_input = base_theta.float()
            batch, candidates, horizon = action_input.shape[:3]
            state = self.state(current)[:, None, None]
            state = state.expand(batch, candidates, horizon, 256, -1)
            action = self.action(action_input)[:, :, :, None]
            action = action.expand(batch, candidates, horizon, 256, -1)
            position = self.position(
                self.token_coordinates.to(device=current.device, dtype=torch.float32)
            )
            position = position[None, None, None].expand(
                batch,
                candidates,
                horizon,
                256,
                -1,
            )
            offset = torch.tanh(
                self.flow(torch.cat((state, action, position), dim=-1))
            )
            offset = offset * self.maximum_normalized_offset
            theta = theta_input.reshape(batch * candidates * horizon, 2, 3)
            channels = current.shape[-1]
            images = current.reshape(batch, 1, 1, 16, 16, channels)
            images = images.expand(
                batch,
                candidates,
                horizon,
                16,
                16,
                channels,
            )
            images = images.reshape(-1, 16, 16, channels).permute(0, 3, 1, 2)
            grid = F.affine_grid(theta, images.shape, align_corners=False)
            grid = grid + offset.reshape(-1, 16, 16, 2)
            warped = F.grid_sample(
                images,
                grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=False,
            )
            warped = warped.permute(0, 2, 3, 1).reshape(
                batch,
                candidates,
                horizon,
                256,
                channels,
            )
        return warped, offset


class LiteTokenFormerWMSceneWarpCUDAFloat32V1(LiteTokenFormerWMSceneWarpV1):
    """Identical scene-warp hypothesis with a float32 grid-sample boundary."""

    def __init__(self, config: LiteTokenFormerConfig | None = None) -> None:
        super().__init__(config)
        self.scene_conditioned_warp = Float32SceneConditionedTokenWarp(
            token_dimension=self.config.input_dim
        )


__all__ = [
    "Float32SceneConditionedTokenWarp",
    "LiteTokenFormerWMSceneWarpCUDAFloat32V1",
]
