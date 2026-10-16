"""Frozen dual-LTWM plus a small action-identifiable latent residual."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn

from recon_latents.models_highres_dual_v1 import (
    HighResMultiLayerDualLatentWM,
)


@dataclass(frozen=True)
class ActionResidualConfig:
    residual_dim: int = 96
    decoder_dim: int = 32
    residual_scale: float = 0.10


class _BranchResidual(nn.Module):
    def __init__(
        self,
        *,
        state_dim: int,
        output_dim: int,
        residual_dim: int,
    ) -> None:
        super().__init__()
        self.state_projection = nn.Linear(state_dim, residual_dim)
        self.action_projection = nn.Linear(state_dim, residual_dim)
        self.spatial_projection = nn.Sequential(
            nn.Linear(10, residual_dim),
            nn.SiLU(),
            nn.Linear(residual_dim, residual_dim),
        )
        self.trunk = nn.Sequential(
            nn.LayerNorm(residual_dim),
            nn.Linear(residual_dim, 2 * residual_dim),
            nn.SiLU(),
            nn.Linear(2 * residual_dim, residual_dim),
            nn.SiLU(),
        )
        self.output = nn.Linear(residual_dim, output_dim)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        spatial_action: torch.Tensor,
    ) -> torch.Tensor:
        hidden = (
            self.state_projection(state)[:, None, None]
            + self.action_projection(action)[:, :, :, None]
            + self.spatial_projection(spatial_action)
        )
        return self.output(self.trunk(hidden))


class LatentActionDecoder(nn.Module):
    """Decode each action from the actual predicted latent displacement."""

    def __init__(
        self,
        semantic_dim: int,
        geometry_dim: int,
        decoder_dim: int,
        coordinates: torch.Tensor,
    ) -> None:
        super().__init__()
        self.token_projection = nn.Sequential(
            nn.LayerNorm(semantic_dim + geometry_dim),
            nn.Linear(semantic_dim + geometry_dim, decoder_dim),
            nn.SiLU(),
        )
        self.decoder = nn.Sequential(
            nn.LayerNorm(3 * decoder_dim),
            nn.Linear(3 * decoder_dim, 2 * decoder_dim),
            nn.SiLU(),
            nn.Linear(2 * decoder_dim, 3),
        )
        self.register_buffer(
            "coordinates",
            coordinates.detach().clone(),
            persistent=True,
        )

    def forward(
        self,
        semantic_delta: torch.Tensor,
        geometry_delta: torch.Tensor,
    ) -> torch.Tensor:
        tokens = self.token_projection(
            torch.cat((semantic_delta, geometry_delta), dim=-1)
        )
        coordinates = self.coordinates.to(
            device=tokens.device,
            dtype=tokens.dtype,
        )
        mean = tokens.mean(dim=-2)
        x_moment = (
            tokens * coordinates[:, 0].view(1, 1, 1, -1, 1)
        ).mean(dim=-2)
        y_moment = (
            tokens * coordinates[:, 1].view(1, 1, 1, -1, 1)
        ).mean(dim=-2)
        return self.decoder(torch.cat((mean, x_moment, y_moment), dim=-1))


class FrozenDualLTWMActionResidual(nn.Module):
    """Preserve the frozen future predictor and learn only action effects."""

    def __init__(
        self,
        base: HighResMultiLayerDualLatentWM,
        config: ActionResidualConfig | None = None,
    ) -> None:
        super().__init__()
        self.base = base
        self.config = config or ActionResidualConfig()
        base_config = base.config
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.semantic_residual = _BranchResidual(
            state_dim=base_config.model_dim,
            output_dim=base_config.semantic_dim,
            residual_dim=self.config.residual_dim,
        )
        self.geometry_residual = _BranchResidual(
            state_dim=base_config.model_dim,
            output_dim=base_config.geometry_dim,
            residual_dim=self.config.residual_dim,
        )
        self.action_decoder = LatentActionDecoder(
            base_config.semantic_dim,
            base_config.geometry_dim,
            self.config.decoder_dim,
            base.normalized_coordinates,
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.base.eval()
        return self

    def forward(
        self,
        semantic_history: torch.Tensor,
        geometry_history: torch.Tensor,
        candidate_actions: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if candidate_actions.ndim == 3:
            candidate_actions = candidate_actions[:, None]
        self.base.eval()
        with torch.no_grad():
            base_output = self.base(
                semantic_history,
                geometry_history,
                candidate_actions,
            )
        cumulative = base_output["cumulative_pose"]
        spatial_action = self.base._spatial_action(cumulative).detach()
        semantic_correction = self.config.residual_scale * self.semantic_residual(
            base_output["semantic_state_tokens"].detach(),
            base_output["action_prefix_tokens"].detach(),
            spatial_action,
        )
        geometry_correction = self.config.residual_scale * self.geometry_residual(
            base_output["geometry_state_tokens"].detach(),
            base_output["action_prefix_tokens"].detach(),
            spatial_action,
        )
        semantic_prediction = (
            base_output["predicted_semantic_tokens"].detach()
            + semantic_correction
        )
        geometry_prediction = (
            base_output["predicted_geometry_tokens"].detach()
            + geometry_correction
        )
        semantic_current = semantic_history[:, None, None, -1]
        geometry_current = geometry_history[:, None, None, -1]
        decoded_action = self.action_decoder(
            semantic_prediction - semantic_current,
            geometry_prediction - geometry_current,
        )
        return {
            **base_output,
            "predicted_semantic_tokens": semantic_prediction,
            "predicted_geometry_tokens": geometry_prediction,
            "semantic_action_residual": semantic_correction,
            "geometry_action_residual": geometry_correction,
            "decoded_action_from_predicted_latent": decoded_action,
        }

    def residual_state_dict(self) -> dict[str, torch.Tensor]:
        return {
            name: value
            for name, value in self.state_dict().items()
            if not name.startswith("base.")
        }

    def contract_dict(self) -> dict[str, object]:
        return {
            "base_config": asdict(self.base.config),
            "action_residual_config": asdict(self.config),
            "base_frozen": True,
            "predicted_object": "full_future_semantic_and_geometry_latents",
            "ate_or_objective_gradient": False,
        }
