"""State-conditioned spatial action warp for LiteTokenFormer future latents."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from recon_latents.models_v2 import (
    LiteTokenFormerConfig,
    LiteTokenFormerWMV2,
    build_v2_action_features,
)


class SceneConditionedTokenWarp(nn.Module):
    """Refine the global SE(2) image warp with a bounded per-token flow.

    The incumbent applies one affine transform to every spatial token.  Camera
    motion instead induces scene-dependent parallax.  This module predicts a
    small offset from the current token, its 2-D position, and the causal action
    prefix.  Its final layer is zero initialized, so warm start exactly
    reproduces the incumbent before training.
    """

    def __init__(
        self,
        token_dimension: int = 384,
        state_width: int = 48,
        action_width: int = 32,
        position_width: int = 8,
        hidden_width: int = 64,
        maximum_normalized_offset: float = 0.12,
    ) -> None:
        super().__init__()
        self.maximum_normalized_offset = float(maximum_normalized_offset)
        self.state = nn.Sequential(
            nn.LayerNorm(token_dimension),
            nn.Linear(token_dimension, state_width),
            nn.SiLU(),
        )
        self.action = nn.Sequential(
            nn.Linear(12, action_width),
            nn.SiLU(),
        )
        self.position = nn.Sequential(
            nn.Linear(2, position_width),
            nn.SiLU(),
        )
        self.flow = nn.Sequential(
            nn.Linear(state_width + action_width + position_width, hidden_width),
            nn.SiLU(),
            nn.Linear(hidden_width, hidden_width // 2),
            nn.SiLU(),
            nn.Linear(hidden_width // 2, 2),
        )
        nn.init.zeros_(self.flow[-1].weight)
        nn.init.zeros_(self.flow[-1].bias)
        coordinate = torch.linspace(-1.0 + 1.0 / 16.0, 1.0 - 1.0 / 16.0, 16)
        yy, xx = torch.meshgrid(coordinate, coordinate, indexing="ij")
        self.register_buffer(
            "token_coordinates",
            torch.stack((xx, yy), dim=-1).reshape(256, 2),
            persistent=True,
        )

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
        batch, candidates, horizon = action_features.shape[:3]
        state = self.state(current_tokens)[:, None, None]
        state = state.expand(batch, candidates, horizon, 256, -1)
        action = self.action(action_features)[:, :, :, None]
        action = action.expand(batch, candidates, horizon, 256, -1)
        position = self.position(
            self.token_coordinates.to(dtype=current_tokens.dtype)
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
        theta = base_theta.reshape(batch * candidates * horizon, 2, 3)
        channels = current_tokens.shape[-1]
        images = current_tokens.reshape(batch, 1, 1, 16, 16, channels)
        images = images.expand(batch, candidates, horizon, 16, 16, channels)
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


class LiteTokenFormerWMSceneWarpV1(LiteTokenFormerWMV2):
    """Frozen incumbent LTWM plus one trainable scene-conditioned flow warp."""

    def __init__(self, config: LiteTokenFormerConfig | None = None) -> None:
        super().__init__(config)
        self.scene_conditioned_warp = SceneConditionedTokenWarp(
            token_dimension=self.config.input_dim
        )
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        for parameter in self.scene_conditioned_warp.parameters():
            parameter.requires_grad_(True)

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        result = super().load_state_dict(state_dict, strict=False, assign=assign)
        missing = [
            name
            for name in result.missing_keys
            if not name.startswith("scene_conditioned_warp.")
        ]
        if strict and (missing or result.unexpected_keys):
            raise RuntimeError(
                "scene-warp checkpoint mismatch: "
                f"missing={missing}, unexpected={result.unexpected_keys}"
            )
        return result

    def forward(
        self,
        history_tokens: torch.Tensor,
        candidate_actions: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if candidate_actions.ndim == 3:
            local_actions = candidate_actions[:, None]
        else:
            local_actions = candidate_actions
        output = super().forward(history_tokens, local_actions)
        action_features, _ = build_v2_action_features(local_actions)
        reference, offset = self.scene_conditioned_warp(
            history_tokens[:, -1],
            action_features,
            output["warp_theta"],
        )
        output["global_affine_reference_tokens"] = output["reference_tokens"]
        output["reference_tokens"] = reference
        output["scene_flow_offset"] = offset
        output["predicted_tokens"] = reference + output["predicted_delta_tokens"]
        return output


__all__ = [
    "LiteTokenFormerWMSceneWarpV1",
    "SceneConditionedTokenWarp",
]
