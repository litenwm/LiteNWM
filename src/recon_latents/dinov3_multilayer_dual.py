"""Frozen high-resolution, multi-layer DINOv3 dual-latent representation.

The semantic branch reuses the already-frozen DINOv3-to-LTWM adapter
projection.  The geometry branch fuses three intermediate DINOv3 blocks and
compresses them with a deterministic frozen orthogonal projection.  No
navigation outcome, ATE, objective, camera parameter, or world-model gradient
enters this representation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from recon_latents.dinov3_adapter import DinoV3TokenAdapter


@dataclass(frozen=True)
class MultiLayerDualLatentConfig:
    image_height: int = 336
    image_width: int = 448
    patch_size: int = 16
    semantic_block: int = 23
    geometry_blocks: tuple[int, ...] = (5, 11, 17)
    backbone_dim: int = 1024
    semantic_dim: int = 384
    geometry_dim: int = 192
    geometry_projection_seed: int = 2026072401

    @property
    def grid_height(self) -> int:
        return self.image_height // self.patch_size

    @property
    def grid_width(self) -> int:
        return self.image_width // self.patch_size

    @property
    def num_patches(self) -> int:
        return self.grid_height * self.grid_width


def validate_multilayer_config(config: MultiLayerDualLatentConfig) -> None:
    if config.image_height <= 0 or config.image_width <= 0:
        raise ValueError("image dimensions must be positive")
    if (
        config.image_height % config.patch_size
        or config.image_width % config.patch_size
    ):
        raise ValueError("image dimensions must be divisible by patch size")
    if not config.geometry_blocks:
        raise ValueError("at least one geometry block is required")
    if len(set(config.geometry_blocks)) != len(config.geometry_blocks):
        raise ValueError("geometry blocks must be unique")
    if min((*config.geometry_blocks, config.semantic_block)) < 0:
        raise ValueError("DINOv3 block indices must be non-negative")
    if config.semantic_dim <= 0 or config.geometry_dim <= 0:
        raise ValueError("latent dimensions must be positive")


def extract_rectangular_patch_tokens(
    hidden_state: torch.Tensor,
    *,
    grid_height: int,
    grid_width: int,
) -> torch.Tensor:
    """Return trailing patch tokens after CLS/register tokens."""

    expected = grid_height * grid_width
    if hidden_state.ndim != 3 or hidden_state.shape[1] < expected:
        raise ValueError(
            f"invalid hidden state {tuple(hidden_state.shape)} for "
            f"{grid_height}x{grid_width} patch grid"
        )
    return hidden_state[:, -expected:, :]


def hidden_state_for_block(
    hidden_states: Sequence[torch.Tensor],
    block_index: int,
) -> torch.Tensor:
    """Map zero-based transformer block index to HF hidden-state output."""

    hidden_index = block_index + 1
    if hidden_index >= len(hidden_states):
        raise ValueError(
            f"block {block_index} is absent from {len(hidden_states) - 1} "
            "transformer block outputs"
        )
    return hidden_states[hidden_index]


def _orthogonal_projection(
    input_dim: int,
    output_dim: int,
    seed: int,
) -> torch.Tensor:
    if output_dim > input_dim:
        raise ValueError("geometry output dim cannot exceed fused input dim")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    matrix = torch.randn(input_dim, output_dim, generator=generator)
    projection = torch.linalg.qr(matrix, mode="reduced").Q
    return projection.contiguous()


def tensor_sha256(value: torch.Tensor) -> str:
    payload = value.detach().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()


class FrozenMultiLayerDualLatentEncoder(nn.Module):
    """Create separate semantic and geometry targets from frozen DINOv3."""

    def __init__(
        self,
        semantic_adapter: DinoV3TokenAdapter,
        config: MultiLayerDualLatentConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config or MultiLayerDualLatentConfig()
        validate_multilayer_config(self.config)
        if semantic_adapter.config.input_dim != self.config.backbone_dim:
            raise ValueError("semantic adapter input dim does not match backbone")
        if semantic_adapter.config.output_dim != self.config.semantic_dim:
            raise ValueError("semantic adapter output dim does not match config")
        self.semantic_norm = semantic_adapter.input_norm
        self.semantic_projection = semantic_adapter.projection
        for parameter in self.parameters():
            parameter.requires_grad_(False)

        fused_dim = len(self.config.geometry_blocks) * self.config.semantic_dim
        geometry_projection = _orthogonal_projection(
            fused_dim,
            self.config.geometry_dim,
            self.config.geometry_projection_seed,
        )
        self.register_buffer(
            "geometry_projection",
            geometry_projection,
            persistent=True,
        )

    @property
    def geometry_projection_sha256(self) -> str:
        return tensor_sha256(self.geometry_projection)

    def _adapt_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.semantic_projection(self.semantic_norm(tokens))

    def forward(
        self,
        hidden_states: Sequence[torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        cfg = self.config
        semantic_native = extract_rectangular_patch_tokens(
            hidden_state_for_block(hidden_states, cfg.semantic_block),
            grid_height=cfg.grid_height,
            grid_width=cfg.grid_width,
        )
        semantic = self._adapt_tokens(semantic_native)

        geometry_levels = []
        for block_index in cfg.geometry_blocks:
            native = extract_rectangular_patch_tokens(
                hidden_state_for_block(hidden_states, block_index),
                grid_height=cfg.grid_height,
                grid_width=cfg.grid_width,
            )
            geometry_levels.append(self._adapt_tokens(native))
        fused_geometry = torch.cat(geometry_levels, dim=-1)
        fused_geometry = F.layer_norm(
            fused_geometry,
            (fused_geometry.shape[-1],),
        )
        geometry = fused_geometry @ self.geometry_projection.to(
            dtype=fused_geometry.dtype
        )
        return {
            "semantic_tokens": semantic,
            "geometry_tokens": geometry,
            "grid_shape": semantic.new_tensor(
                (cfg.grid_height, cfg.grid_width),
                dtype=torch.int64,
            ),
        }

    def contract_dict(self) -> dict[str, object]:
        return {
            **asdict(self.config),
            "grid_height": self.config.grid_height,
            "grid_width": self.config.grid_width,
            "num_patches": self.config.num_patches,
            "geometry_projection": "frozen_seeded_orthogonal_qr",
            "geometry_projection_sha256": self.geometry_projection_sha256,
            "semantic_projection": "frozen_existing_dinov3_adapter",
            "camera_conditioning": False,
            "outcome_gradients": False,
        }


def imagenet_normalize(image: torch.Tensor) -> torch.Tensor:
    """Normalize an RGB tensor in [0,1] using the DINOv3 ImageNet contract."""

    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError(f"expected RGB tensor [3,H,W], got {tuple(image.shape)}")
    mean = image.new_tensor((0.485, 0.456, 0.406))[:, None, None]
    std = image.new_tensor((0.229, 0.224, 0.225))[:, None, None]
    return (image - mean) / std

