"""Calibration-conditioned extension of the high-resolution dual-latent WM."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from recon_latents.camera_calibration_v1 import CAMERA_FEATURE_DIM
from recon_latents.models_highres_dual_v1 import (
    HighResDualLatentWMConfig,
    HighResMultiLayerDualLatentWM,
)


@dataclass(frozen=True)
class CalibratedHighResDualLatentWMConfig(HighResDualLatentWMConfig):
    camera_feature_dim: int = CAMERA_FEATURE_DIM


class CalibratedHighResMultiLayerDualLatentWM(
    HighResMultiLayerDualLatentWM
):
    """Use physical calibration only when an intrinsics-valid flag is present."""

    def __init__(
        self,
        config: CalibratedHighResDualLatentWMConfig | None = None,
    ) -> None:
        config = config or CalibratedHighResDualLatentWMConfig()
        super().__init__(config)
        self.config = config
        self.camera_encoder = nn.Sequential(
            nn.LayerNorm(config.camera_feature_dim),
            nn.Linear(config.camera_feature_dim, config.model_dim),
            nn.SiLU(),
            nn.Linear(config.model_dim, config.model_dim),
        )
        self.semantic_camera = nn.Linear(
            config.model_dim,
            config.model_dim,
            bias=False,
        )
        self.geometry_camera = nn.Linear(
            config.model_dim,
            config.model_dim,
            bias=False,
        )
        self.spatial_camera = nn.Linear(
            config.model_dim,
            config.model_dim,
            bias=False,
        )
        self.camera_scale = nn.Parameter(torch.full((3,), 0.05))

    def forward(
        self,
        semantic_history: torch.Tensor,
        geometry_history: torch.Tensor,
        candidate_actions: torch.Tensor,
        camera_features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        cfg = self.config
        if candidate_actions.ndim == 3:
            candidate_actions = candidate_actions[:, None]
        if candidate_actions.shape[-2:] != (cfg.horizon, 3):
            raise ValueError("candidate actions do not match configured horizon")
        if camera_features.shape != (
            semantic_history.shape[0],
            cfg.camera_feature_dim,
        ):
            raise ValueError(
                f"expected camera features [B,{cfg.camera_feature_dim}], got "
                f"{tuple(camera_features.shape)}"
            )
        if semantic_history.shape[-3:] != (
            cfg.history_frames,
            cfg.num_tokens,
            cfg.semantic_dim,
        ):
            raise ValueError("semantic history does not match config")
        if geometry_history.shape[-3:] != (
            cfg.history_frames,
            cfg.num_tokens,
            cfg.geometry_dim,
        ):
            raise ValueError("geometry history does not match config")

        batch, candidates = candidate_actions.shape[:2]
        semantic_state = self.semantic_history(semantic_history)
        geometry_state = self.geometry_history(geometry_history)
        action, cumulative = self.action_encoder(candidate_actions)
        spatial_action = self._spatial_action(cumulative)
        horizon = self.horizon_embedding[None, None, :, None]

        # Feature 32 is the explicit physical-intrinsics validity flag.  When
        # absent, every calibration residual is exactly zero regardless of MLP
        # biases or preprocessing-only metadata.
        valid = camera_features[:, 32:33].to(
            dtype=semantic_state.dtype,
        )
        camera = self.camera_encoder(
            camera_features.to(dtype=semantic_state.dtype)
        ) * valid
        semantic_camera = (
            self.camera_scale[0]
            * self.semantic_camera(camera)[:, None, None, None]
        )
        geometry_camera = (
            self.camera_scale[1]
            * self.geometry_camera(camera)[:, None, None, None]
        )
        spatial_camera = (
            self.camera_scale[2]
            * self.spatial_camera(camera)[:, None, None, None]
        )
        semantic = (
            semantic_state[:, None, None]
            + horizon
            + self.semantic_action_projection(action)[:, :, :, None]
            + 0.25 * spatial_action
            + semantic_camera
        )
        geometry = (
            geometry_state[:, None, None]
            + horizon
            + self.geometry_action_projection(action)[:, :, :, None]
            + spatial_action
            + geometry_camera
            + spatial_camera
        )
        semantic = semantic.reshape(
            batch * candidates,
            cfg.horizon,
            cfg.num_tokens,
            cfg.model_dim,
        )
        geometry = geometry.reshape_as(semantic)
        action_flat = action.reshape(
            batch * candidates,
            cfg.horizon,
            cfg.model_dim,
        )
        for block in self.blocks:
            semantic, geometry = block(semantic, geometry, action_flat)

        semantic_hidden = self.semantic_norm(semantic).reshape(
            batch,
            candidates,
            cfg.horizon,
            cfg.num_tokens,
            cfg.model_dim,
        )
        geometry_hidden = self.geometry_norm(geometry).reshape_as(
            semantic_hidden
        )
        semantic_delta = self.semantic_delta(semantic_hidden)
        geometry_delta = self.geometry_delta(geometry_hidden)
        predicted_semantic = (
            semantic_history[:, None, None, -1] + semantic_delta
        )
        predicted_geometry = (
            geometry_history[:, None, None, -1] + geometry_delta
        )
        decoded_action = self.action_decode(
            torch.cat(
                (
                    semantic_hidden.mean(dim=3),
                    geometry_hidden.mean(dim=3),
                ),
                dim=-1,
            )
        )
        return {
            "predicted_semantic_tokens": predicted_semantic,
            "predicted_geometry_tokens": predicted_geometry,
            "predicted_semantic_delta": semantic_delta,
            "predicted_geometry_delta": geometry_delta,
            "semantic_logvar": self.semantic_logvar(
                semantic_hidden
            ).squeeze(-1).clamp(-8.0, 8.0),
            "geometry_logvar": self.geometry_logvar(
                geometry_hidden
            ).squeeze(-1).clamp(-8.0, 8.0),
            "decoded_action": decoded_action,
            "action_prefix_tokens": action,
            "cumulative_pose": cumulative,
            "semantic_state_tokens": semantic_state,
            "geometry_state_tokens": geometry_state,
            "camera_conditioning_active": valid.squeeze(-1).bool(),
        }
