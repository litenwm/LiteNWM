"""Depth-aware translation residual on a frozen rotation-transport LTWM.

The frozen base already models real future latents and a small calibrated yaw
transport.  This adapter predicts one bounded depth per current geometry token
and adds only the difference between a full SE(3) warp and a rotation-only
warp.  It is exactly equal to the frozen base at initialization.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
from torch import nn

from recon_latents.models_highres_dual_rotation_transport_v1 import (
    RotationTransportHighResMultiLayerDualLatentWM,
)


@dataclass(frozen=True)
class DepthTranslationResidualConfig:
    minimum_depth_m: float = 0.5
    maximum_depth_m: float = 30.0
    initial_depth_m: float = 10.0
    body_forward_axis_camera: tuple[float, float, float] = (
        0.013540377093979815,
        -0.05269835610248765,
        0.9985186735621164,
    )
    body_lateral_axis_camera: tuple[float, float, float] = (
        0.9995075688849837,
        -0.02755676943468673,
        -0.01500813779098614,
    )


class FrozenRotationTransportDepthTranslationResidual(nn.Module):
    """Learn metric parallax transport while preserving the frozen base."""

    def __init__(
        self,
        base: RotationTransportHighResMultiLayerDualLatentWM,
        config: DepthTranslationResidualConfig | None = None,
    ) -> None:
        super().__init__()
        self.base = base
        self.config = config or DepthTranslationResidualConfig()
        cfg = self.config
        if not (
            0.0 < cfg.minimum_depth_m
            < cfg.initial_depth_m
            < cfg.maximum_depth_m
        ):
            raise ValueError("depth bounds and initialization are invalid")
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        model_dim = self.base.config.model_dim
        self.depth_norm = nn.LayerNorm(model_dim)
        self.depth_logit = nn.Linear(model_dim, 1)
        nn.init.zeros_(self.depth_logit.weight)
        initial_fraction = (
            (cfg.initial_depth_m - cfg.minimum_depth_m)
            / (cfg.maximum_depth_m - cfg.minimum_depth_m)
        )
        initial_logit = math.log(
            initial_fraction / (1.0 - initial_fraction)
        )
        nn.init.constant_(self.depth_logit.bias, initial_logit)
        self.translation_transport_raw_scale = nn.Parameter(torch.zeros(2))
        forward = torch.tensor(
            cfg.body_forward_axis_camera,
            dtype=torch.float32,
        )
        lateral = torch.tensor(
            cfg.body_lateral_axis_camera,
            dtype=torch.float32,
        )
        forward = forward / torch.linalg.vector_norm(forward).clamp_min(
            1.0e-8
        )
        lateral = lateral - torch.dot(lateral, forward) * forward
        lateral = lateral / torch.linalg.vector_norm(lateral).clamp_min(
            1.0e-8
        )
        self.register_buffer(
            "body_forward_axis_camera",
            forward,
            persistent=True,
        )
        self.register_buffer(
            "body_lateral_axis_camera",
            lateral,
            persistent=True,
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.base.eval()
        return self

    def _depth(
        self,
        geometry_state: torch.Tensor,
    ) -> torch.Tensor:
        cfg = self.config
        fraction = torch.sigmoid(
            self.depth_logit(self.depth_norm(geometry_state))
        ).squeeze(-1)
        return cfg.minimum_depth_m + (
            cfg.maximum_depth_m - cfg.minimum_depth_m
        ) * fraction

    def _se3_sampling_grid(
        self,
        cumulative_pose: torch.Tensor,
        depth: torch.Tensor,
    ) -> torch.Tensor:
        cfg = self.base.config
        batch, candidates, horizon = cumulative_pose.shape[:3]
        yaw = cumulative_pose[..., 2]
        rotation = self.base._camera_rotation(yaw)
        rays = self.base.output_camera_rays.to(
            device=depth.device,
            dtype=depth.dtype,
        )
        future_points = (
            rays.view(1, 1, 1, cfg.num_tokens, 3)
            * depth[:, None, None, :, None]
        )
        translation = (
            cumulative_pose[..., 0:1]
            * self.body_forward_axis_camera.to(
                device=depth.device,
                dtype=depth.dtype,
            )
            + cumulative_pose[..., 1:2]
            * self.body_lateral_axis_camera.to(
                device=depth.device,
                dtype=depth.dtype,
            )
        )
        current_points = (
            torch.einsum(
                "...ji,...nj->...ni",
                rotation,
                future_points.expand(
                    batch,
                    candidates,
                    horizon,
                    cfg.num_tokens,
                    3,
                ),
            )
            + translation[..., None, :]
        )
        z = current_points[..., 2].clamp_min(1.0e-3)
        source_x = (
            cfg.effective_focal_px * current_points[..., 0] / z
            + cfg.principal_x_px
        )
        source_y = (
            cfg.effective_focal_px * current_points[..., 1] / z
            + cfg.principal_y_px
        )
        normalized_x = 2.0 * source_x / cfg.image_width - 1.0
        normalized_y = 2.0 * source_y / cfg.image_height - 1.0
        return torch.stack(
            (normalized_x, normalized_y),
            dim=-1,
        ).reshape(
            batch,
            candidates,
            horizon,
            cfg.grid_height,
            cfg.grid_width,
            2,
        )

    def _warp_se3(
        self,
        reference: torch.Tensor,
        cumulative_pose: torch.Tensor,
        depth: torch.Tensor,
    ) -> torch.Tensor:
        cfg = self.base.config
        batch, tokens, channels = reference.shape
        candidates, horizon = cumulative_pose.shape[1:3]
        image = reference.reshape(
            batch,
            cfg.grid_height,
            cfg.grid_width,
            channels,
        ).permute(0, 3, 1, 2)
        image = (
            image[:, None, None]
            .expand(
                batch,
                candidates,
                horizon,
                channels,
                cfg.grid_height,
                cfg.grid_width,
            )
            .reshape(
                batch * candidates * horizon,
                channels,
                cfg.grid_height,
                cfg.grid_width,
            )
        )
        grid = self._se3_sampling_grid(
            cumulative_pose,
            depth,
        ).reshape(
            batch * candidates * horizon,
            cfg.grid_height,
            cfg.grid_width,
            2,
        )
        warped = F.grid_sample(
            image,
            grid,
            mode="bilinear",
            padding_mode=cfg.warp_padding_mode,
            align_corners=False,
        )
        return warped.permute(0, 2, 3, 1).reshape(
            batch,
            candidates,
            horizon,
            tokens,
            channels,
        )

    def forward(
        self,
        semantic_history: torch.Tensor,
        geometry_history: torch.Tensor,
        candidate_actions: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        self.base.eval()
        with torch.no_grad():
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
        with torch.no_grad():
            semantic_rotation = self.base._warp_reference(
                semantic_current,
                yaw,
            )
            geometry_rotation = self.base._warp_reference(
                geometry_current,
                yaw,
            )
        semantic_transport = semantic_se3 - semantic_rotation
        geometry_transport = geometry_se3 - geometry_rotation
        scale = torch.tanh(self.translation_transport_raw_scale).to(
            dtype=semantic_transport.dtype,
        )
        predicted_semantic = (
            base_output["predicted_semantic_tokens"]
            + scale[0] * semantic_transport
        )
        predicted_geometry = (
            base_output["predicted_geometry_tokens"]
            + scale[1] * geometry_transport
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
            "semantic_translation_transport": semantic_transport,
            "geometry_translation_transport": geometry_transport,
            "predicted_depth_m": depth,
            "translation_transport_scale": scale,
        }

    def adapter_state_dict(self) -> dict[str, torch.Tensor]:
        return {
            name: value
            for name, value in self.state_dict().items()
            if not name.startswith("base.")
        }
