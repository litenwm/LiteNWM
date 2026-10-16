"""Rotation-warp prior for the high-resolution dual-latent world model.

The model keeps the existing learned rollout unchanged, but replaces its
no-change output reference with an analytic camera-rotation warp of the
current latent grid.  The neural network therefore learns only the residual
future change around a physically meaningful yaw-conditioned reference.

This is intentionally a rotation-only proxy.  It does not claim or use
camera translation extrinsics.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from recon_latents.models_highres_dual_v1 import (
    HighResDualLatentWMConfig,
    HighResMultiLayerDualLatentWM,
)


@dataclass(frozen=True)
class RotationWarpHighResDualLatentWMConfig(HighResDualLatentWMConfig):
    image_width: int = 448
    image_height: int = 336
    effective_focal_px: float = 298.47442166440477
    principal_x_px: float = 224.0
    principal_y_px: float = 168.0
    camera_yaw_axis_x: float = 0.028306853053325053
    camera_yaw_axis_y: float = 0.9982301877434978
    camera_yaw_axis_z: float = 0.05229927674259504
    warp_padding_mode: str = "border"


class RotationWarpHighResMultiLayerDualLatentWM(
    HighResMultiLayerDualLatentWM
):
    """Predict residual dual latents around an analytic rotation warp."""

    def __init__(
        self,
        config: RotationWarpHighResDualLatentWMConfig | None = None,
    ) -> None:
        config = config or RotationWarpHighResDualLatentWMConfig()
        super().__init__(config)
        self.config = config
        if config.image_width <= 0 or config.image_height <= 0:
            raise ValueError("image dimensions must be positive")
        if config.effective_focal_px <= 0:
            raise ValueError("effective focal length must be positive")
        if config.warp_padding_mode not in {"zeros", "border", "reflection"}:
            raise ValueError("unsupported grid_sample padding mode")
        axis = torch.tensor(
            (
                config.camera_yaw_axis_x,
                config.camera_yaw_axis_y,
                config.camera_yaw_axis_z,
            ),
            dtype=torch.float32,
        )
        axis = axis / torch.linalg.vector_norm(axis).clamp_min(1.0e-8)
        self.register_buffer("camera_yaw_axis", axis, persistent=True)
        self.register_buffer(
            "output_camera_rays",
            self._output_rays(config),
            persistent=True,
        )

    @staticmethod
    def _output_rays(
        config: RotationWarpHighResDualLatentWMConfig,
    ) -> torch.Tensor:
        # Token centers are mapped back to the image plane.  The grid uses
        # align_corners=False, so normalized coordinates are referenced to
        # pixel-cell edges rather than corner-center coordinates.
        cell_width = config.image_width / config.grid_width
        cell_height = config.image_height / config.grid_height
        x = (
            torch.arange(config.grid_width, dtype=torch.float32) + 0.5
        ) * cell_width
        y = (
            torch.arange(config.grid_height, dtype=torch.float32) + 0.5
        ) * cell_height
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        rays = torch.stack(
            (
                (xx - config.principal_x_px) / config.effective_focal_px,
                (yy - config.principal_y_px) / config.effective_focal_px,
                torch.ones_like(xx),
            ),
            dim=-1,
        )
        return rays.reshape(config.num_tokens, 3)

    def _camera_rotation(
        self,
        yaw: torch.Tensor,
    ) -> torch.Tensor:
        """Return first-camera to future-camera rotations for body yaw."""
        axis = self.camera_yaw_axis.to(device=yaw.device, dtype=yaw.dtype)
        x, y, z = axis.unbind()
        zero = torch.zeros((), device=yaw.device, dtype=yaw.dtype)
        skew = torch.stack(
            (
                torch.stack((zero, -z, y)),
                torch.stack((z, zero, -x)),
                torch.stack((-y, x, zero)),
            )
        )
        outer = axis[:, None] * axis[None, :]
        identity = torch.eye(3, device=yaw.device, dtype=yaw.dtype)
        cosine = torch.cos(yaw)[..., None, None]
        sine = torch.sin(yaw)[..., None, None]
        return (
            cosine * identity
            + (1.0 - cosine) * outer
            + sine * skew
        )

    def _sampling_grid(
        self,
        yaw: torch.Tensor,
    ) -> torch.Tensor:
        cfg = self.config
        rotation = self._camera_rotation(yaw)
        output_rays = self.output_camera_rays.to(
            device=yaw.device,
            dtype=yaw.dtype,
        )
        # H = K R K^-1 maps current pixels to future pixels.  grid_sample
        # needs the inverse map, hence R^T applied to each future ray.
        source_rays = torch.einsum(
            "...ji,nj->...ni",
            rotation,
            output_rays,
        )
        depth = source_rays[..., 2].clamp_min(1.0e-4)
        source_x = (
            cfg.effective_focal_px * source_rays[..., 0] / depth
            + cfg.principal_x_px
        )
        source_y = (
            cfg.effective_focal_px * source_rays[..., 1] / depth
            + cfg.principal_y_px
        )
        normalized_x = 2.0 * source_x / cfg.image_width - 1.0
        normalized_y = 2.0 * source_y / cfg.image_height - 1.0
        return torch.stack(
            (normalized_x, normalized_y),
            dim=-1,
        ).reshape(
            *yaw.shape,
            cfg.grid_height,
            cfg.grid_width,
            2,
        )

    def _warp_reference(
        self,
        reference: torch.Tensor,
        yaw: torch.Tensor,
    ) -> torch.Tensor:
        cfg = self.config
        if reference.shape[-2] != cfg.num_tokens:
            raise ValueError("reference token count does not match grid")
        batch, tokens, channels = reference.shape
        if yaw.shape[0] != batch:
            raise ValueError("yaw batch does not match reference")
        candidates, horizon = yaw.shape[1:3]
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
        grid = self._sampling_grid(yaw).reshape(
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
        output = super().forward(
            semantic_history,
            geometry_history,
            candidate_actions,
        )
        cumulative_yaw = output["cumulative_pose"][..., 2]
        semantic_reference = self._warp_reference(
            semantic_history[:, -1],
            cumulative_yaw,
        )
        geometry_reference = self._warp_reference(
            geometry_history[:, -1],
            cumulative_yaw,
        )
        semantic_residual = output["predicted_semantic_delta"]
        geometry_residual = output["predicted_geometry_delta"]
        predicted_semantic = semantic_reference + semantic_residual
        predicted_geometry = geometry_reference + geometry_residual
        current_semantic = semantic_history[:, None, None, -1]
        current_geometry = geometry_history[:, None, None, -1]
        output.update(
            {
                "predicted_semantic_tokens": predicted_semantic,
                "predicted_geometry_tokens": predicted_geometry,
                "predicted_semantic_residual": semantic_residual,
                "predicted_geometry_residual": geometry_residual,
                "predicted_semantic_delta": (
                    predicted_semantic - current_semantic
                ),
                "predicted_geometry_delta": (
                    predicted_geometry - current_geometry
                ),
                "warped_semantic_reference": semantic_reference,
                "warped_geometry_reference": geometry_reference,
            }
        )
        return output
