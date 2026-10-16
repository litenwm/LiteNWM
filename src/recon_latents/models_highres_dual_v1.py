"""Action-conditioned high-resolution semantic/geometry dual-latent WM."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from recon_latents.models_v2 import (
    ContinuousActionPrefixEncoder,
    integrate_se2_prefix,
)


@dataclass(frozen=True)
class HighResDualLatentWMConfig:
    semantic_dim: int = 384
    geometry_dim: int = 192
    grid_height: int = 21
    grid_width: int = 28
    history_frames: int = 4
    horizon: int = 4
    model_dim: int = 128
    heads: int = 4
    action_layers: int = 2
    rollout_blocks: int = 3
    mlp_ratio: int = 2

    @property
    def num_tokens(self) -> int:
        return self.grid_height * self.grid_width


class RectangularSpatialMix(nn.Module):
    def __init__(self, dim: int, grid_height: int, grid_width: int) -> None:
        super().__init__()
        self.dim = dim
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.norm = nn.LayerNorm(dim)
        self.depthwise = nn.Conv2d(
            dim,
            dim,
            kernel_size=3,
            padding=1,
            groups=dim,
        )
        self.pointwise = nn.Conv2d(dim, dim, kernel_size=1)
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        leading = x.shape[:-2]
        tokens, dim = x.shape[-2:]
        expected = self.grid_height * self.grid_width
        if tokens != expected or dim != self.dim:
            raise ValueError(
                f"expected [...,{expected},{self.dim}], got {tuple(x.shape)}"
            )
        image = self.norm(x).reshape(
            -1,
            self.grid_height,
            self.grid_width,
            dim,
        )
        image = image.permute(0, 3, 1, 2)
        mixed = self.pointwise(F.silu(self.depthwise(image)))
        mixed = mixed.permute(0, 2, 3, 1).reshape(*leading, tokens, dim)
        return x + self.scale * mixed


class BranchHistoryEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        config: HighResDualLatentWMConfig,
    ) -> None:
        super().__init__()
        self.config = config
        self.input_projection = nn.Linear(input_dim, config.model_dim)
        self.frame_embedding = nn.Parameter(
            torch.randn(config.history_frames, config.model_dim) * 0.02
        )
        self.temporal_gru = nn.GRU(
            config.model_dim,
            config.model_dim,
            batch_first=True,
        )
        self.spatial = RectangularSpatialMix(
            config.model_dim,
            config.grid_height,
            config.grid_width,
        )
        self.norm = nn.LayerNorm(config.model_dim)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        cfg = self.config
        if history.shape[-3:-1] != (cfg.history_frames, cfg.num_tokens):
            raise ValueError(
                f"expected history [...,{cfg.history_frames},"
                f"{cfg.num_tokens},C], got {tuple(history.shape)}"
            )
        batch = history.shape[0]
        x = self.input_projection(history)
        x = x + self.frame_embedding[None, :, None]
        temporal = x.permute(0, 2, 1, 3).reshape(
            batch * cfg.num_tokens,
            cfg.history_frames,
            cfg.model_dim,
        )
        temporal, _ = self.temporal_gru(temporal)
        current = temporal[:, -1].reshape(
            batch,
            cfg.num_tokens,
            cfg.model_dim,
        )
        return self.norm(self.spatial(current))


class DualRolloutBlock(nn.Module):
    def __init__(self, config: HighResDualLatentWMConfig) -> None:
        super().__init__()
        cfg = config
        self.config = cfg
        self.semantic_temporal_norm = nn.LayerNorm(cfg.model_dim)
        self.geometry_temporal_norm = nn.LayerNorm(cfg.model_dim)
        self.semantic_temporal = nn.MultiheadAttention(
            cfg.model_dim,
            cfg.heads,
            batch_first=True,
        )
        self.geometry_temporal = nn.MultiheadAttention(
            cfg.model_dim,
            cfg.heads,
            batch_first=True,
        )
        self.semantic_spatial = RectangularSpatialMix(
            cfg.model_dim,
            cfg.grid_height,
            cfg.grid_width,
        )
        self.geometry_spatial = RectangularSpatialMix(
            cfg.model_dim,
            cfg.grid_height,
            cfg.grid_width,
        )
        self.semantic_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cfg.model_dim, 3 * cfg.model_dim),
        )
        self.geometry_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cfg.model_dim, 3 * cfg.model_dim),
        )
        hidden_dim = cfg.mlp_ratio * cfg.model_dim
        self.semantic_mlp = nn.Sequential(
            nn.LayerNorm(cfg.model_dim),
            nn.Linear(cfg.model_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, cfg.model_dim),
        )
        self.geometry_mlp = nn.Sequential(
            nn.LayerNorm(cfg.model_dim),
            nn.Linear(cfg.model_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, cfg.model_dim),
        )
        self.semantic_from_geometry = nn.Linear(
            cfg.model_dim,
            cfg.model_dim,
            bias=False,
        )
        self.geometry_from_semantic = nn.Linear(
            cfg.model_dim,
            cfg.model_dim,
            bias=False,
        )
        self.cross_scale = nn.Parameter(torch.full((2,), 0.05))
        self.residual_scale = nn.Parameter(torch.full((4,), 0.1))

    def _temporal(
        self,
        x: torch.Tensor,
        norm: nn.LayerNorm,
        attention: nn.MultiheadAttention,
    ) -> torch.Tensor:
        batch_candidates, horizon, tokens, dim = x.shape
        sequence = norm(x).permute(0, 2, 1, 3).reshape(
            batch_candidates * tokens,
            horizon,
            dim,
        )
        causal = torch.triu(
            torch.ones(
                horizon,
                horizon,
                device=x.device,
                dtype=torch.bool,
            ),
            diagonal=1,
        )
        sequence = attention(
            sequence,
            sequence,
            sequence,
            attn_mask=causal,
            need_weights=False,
        )[0]
        return sequence.reshape(
            batch_candidates,
            tokens,
            horizon,
            dim,
        ).permute(0, 2, 1, 3)

    @staticmethod
    def _modulated_mlp(
        x: torch.Tensor,
        action: torch.Tensor,
        modulation: nn.Module,
        mlp: nn.Module,
    ) -> torch.Tensor:
        shift, scale, gate = modulation(action).chunk(3, dim=-1)
        conditioned = x * (1.0 + scale[:, :, None]) + shift[:, :, None]
        return torch.sigmoid(gate[:, :, None]) * mlp(conditioned)

    def forward(
        self,
        semantic: torch.Tensor,
        geometry: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        semantic = semantic + self.residual_scale[0] * self._temporal(
            semantic,
            self.semantic_temporal_norm,
            self.semantic_temporal,
        )
        geometry = geometry + self.residual_scale[1] * self._temporal(
            geometry,
            self.geometry_temporal_norm,
            self.geometry_temporal,
        )
        semantic = self.semantic_spatial(semantic)
        geometry = self.geometry_spatial(geometry)
        semantic = semantic + self.residual_scale[2] * self._modulated_mlp(
            semantic,
            action,
            self.semantic_modulation,
            self.semantic_mlp,
        )
        geometry = geometry + self.residual_scale[3] * self._modulated_mlp(
            geometry,
            action,
            self.geometry_modulation,
            self.geometry_mlp,
        )
        semantic_before = semantic
        geometry_before = geometry
        semantic = semantic + self.cross_scale[0] * self.semantic_from_geometry(
            geometry_before
        )
        geometry = geometry + self.cross_scale[1] * self.geometry_from_semantic(
            semantic_before
        )
        return semantic, geometry


class HighResMultiLayerDualLatentWM(nn.Module):
    """Predict real future semantic and geometry latents under an action prefix."""

    def __init__(
        self,
        config: HighResDualLatentWMConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config or HighResDualLatentWMConfig()
        cfg = self.config
        if cfg.history_frames < 2 or cfg.horizon < 1:
            raise ValueError("history must be >=2 and horizon must be positive")
        if cfg.model_dim % cfg.heads:
            raise ValueError("model dim must be divisible by heads")
        self.semantic_history = BranchHistoryEncoder(cfg.semantic_dim, cfg)
        self.geometry_history = BranchHistoryEncoder(cfg.geometry_dim, cfg)
        self.action_encoder = ContinuousActionPrefixEncoder(
            cfg.model_dim,
            cfg.heads,
            cfg.action_layers,
            horizon=cfg.horizon,
        )
        self.horizon_embedding = nn.Parameter(
            torch.randn(cfg.horizon, cfg.model_dim) * 0.02
        )
        self.semantic_action_projection = nn.Linear(
            cfg.model_dim,
            cfg.model_dim,
        )
        self.geometry_action_projection = nn.Linear(
            cfg.model_dim,
            cfg.model_dim,
        )
        self.spatial_action_mlp = nn.Sequential(
            nn.Linear(10, cfg.model_dim),
            nn.SiLU(),
            nn.Linear(cfg.model_dim, cfg.model_dim),
        )
        self.blocks = nn.ModuleList(
            [DualRolloutBlock(cfg) for _ in range(cfg.rollout_blocks)]
        )
        self.semantic_norm = nn.LayerNorm(cfg.model_dim)
        self.geometry_norm = nn.LayerNorm(cfg.model_dim)
        self.semantic_delta = nn.Linear(cfg.model_dim, cfg.semantic_dim)
        self.geometry_delta = nn.Linear(cfg.model_dim, cfg.geometry_dim)
        self.semantic_logvar = nn.Linear(cfg.model_dim, 1)
        self.geometry_logvar = nn.Linear(cfg.model_dim, 1)
        self.action_decode = nn.Sequential(
            nn.LayerNorm(2 * cfg.model_dim),
            nn.Linear(2 * cfg.model_dim, 3),
        )
        nn.init.zeros_(self.semantic_delta.weight)
        nn.init.zeros_(self.semantic_delta.bias)
        nn.init.zeros_(self.geometry_delta.weight)
        nn.init.zeros_(self.geometry_delta.bias)
        self.register_buffer(
            "normalized_coordinates",
            self._coordinates(cfg.grid_height, cfg.grid_width),
            persistent=True,
        )

    @staticmethod
    def _coordinates(height: int, width: int) -> torch.Tensor:
        y = torch.linspace(-1.0, 1.0, height)
        x = torch.linspace(-1.0, 1.0, width)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        return torch.stack((xx, yy), dim=-1).reshape(height * width, 2)

    def _spatial_action(
        self,
        cumulative: torch.Tensor,
    ) -> torch.Tensor:
        batch, candidates, horizon = cumulative.shape[:3]
        coordinates = self.normalized_coordinates.to(
            device=cumulative.device,
            dtype=cumulative.dtype,
        )
        coordinates = coordinates.view(1, 1, 1, -1, 2).expand(
            batch,
            candidates,
            horizon,
            -1,
            -1,
        )
        pose = cumulative[:, :, :, None].expand(
            batch,
            candidates,
            horizon,
            self.config.num_tokens,
            3,
        )
        radius = torch.linalg.vector_norm(coordinates, dim=-1, keepdim=True)
        interaction = coordinates * pose[..., :2]
        yaw_interaction = coordinates * pose[..., 2:3]
        features = torch.cat(
            (coordinates, pose, radius, interaction, yaw_interaction),
            dim=-1,
        )
        return self.spatial_action_mlp(features)

    def forward(
        self,
        semantic_history: torch.Tensor,
        geometry_history: torch.Tensor,
        candidate_actions: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        cfg = self.config
        if candidate_actions.ndim == 3:
            candidate_actions = candidate_actions[:, None]
        if candidate_actions.shape[-2:] != (cfg.horizon, 3):
            raise ValueError(
                f"expected candidate actions [B,K,{cfg.horizon},3], got "
                f"{tuple(candidate_actions.shape)}"
            )
        if semantic_history.shape[-3:] != (
            cfg.history_frames,
            cfg.num_tokens,
            cfg.semantic_dim,
        ):
            raise ValueError(
                "semantic history does not match configured dual-latent contract"
            )
        if geometry_history.shape[-3:] != (
            cfg.history_frames,
            cfg.num_tokens,
            cfg.geometry_dim,
        ):
            raise ValueError(
                "geometry history does not match configured dual-latent contract"
            )

        batch, candidates = candidate_actions.shape[:2]
        semantic_state = self.semantic_history(semantic_history)
        geometry_state = self.geometry_history(geometry_history)
        action, cumulative = self.action_encoder(candidate_actions)
        spatial_action = self._spatial_action(cumulative)
        horizon = self.horizon_embedding[None, None, :, None]

        semantic = (
            semantic_state[:, None, None]
            + horizon
            + self.semantic_action_projection(action)[:, :, :, None]
            + 0.25 * spatial_action
        )
        geometry = (
            geometry_state[:, None, None]
            + horizon
            + self.geometry_action_projection(action)[:, :, :, None]
            + spatial_action
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
        semantic_reference = semantic_history[:, None, None, -1]
        geometry_reference = geometry_history[:, None, None, -1]
        predicted_semantic = semantic_reference + semantic_delta
        predicted_geometry = geometry_reference + geometry_delta
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
        }

