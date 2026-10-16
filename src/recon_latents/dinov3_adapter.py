"""DINOv3 ViT-L/16 patch-token adapter for the frozen LTWM contract."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class DinoV3AdapterConfig:
    input_dim: int = 1024
    output_dim: int = 384
    input_grid: int = 14
    output_grid: int = 16


class DinoV3TokenAdapter(nn.Module):
    """Map native DINOv3 patches to the existing 16x16x384 LTWM interface."""

    def __init__(self, config: DinoV3AdapterConfig | None = None) -> None:
        super().__init__()
        self.config = config or DinoV3AdapterConfig()
        self.input_norm = nn.LayerNorm(self.config.input_dim)
        self.projection = nn.Linear(self.config.input_dim, self.config.output_dim)
        nn.init.trunc_normal_(self.projection.weight, std=0.02)
        nn.init.zeros_(self.projection.bias)

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        cfg = self.config
        if patch_tokens.ndim == 3:
            expected = cfg.input_grid * cfg.input_grid
            if patch_tokens.shape[1:] != (expected, cfg.input_dim):
                raise ValueError(
                    f"expected [B,{expected},{cfg.input_dim}], got "
                    f"{tuple(patch_tokens.shape)}"
                )
            patches = patch_tokens.reshape(
                patch_tokens.shape[0],
                cfg.input_grid,
                cfg.input_grid,
                cfg.input_dim,
            )
        elif patch_tokens.ndim == 4:
            if patch_tokens.shape[1:] != (
                cfg.input_grid,
                cfg.input_grid,
                cfg.input_dim,
            ):
                raise ValueError(
                    f"expected [B,{cfg.input_grid},{cfg.input_grid},{cfg.input_dim}], "
                    f"got {tuple(patch_tokens.shape)}"
                )
            patches = patch_tokens
        else:
            raise ValueError(f"expected rank-3 or rank-4 patches, got rank {patch_tokens.ndim}")

        projected = self.projection(self.input_norm(patches))
        channels_first = projected.permute(0, 3, 1, 2)
        resized = F.interpolate(
            channels_first,
            size=(cfg.output_grid, cfg.output_grid),
            mode="bicubic",
            align_corners=False,
        )
        return resized.permute(0, 2, 3, 1).contiguous()

    def config_dict(self) -> dict[str, int]:
        return asdict(self.config)


def extract_dinov3_patch_tokens(last_hidden_state: torch.Tensor, grid: int = 14) -> torch.Tensor:
    """Extract trailing patch tokens after CLS/register tokens."""

    expected = grid * grid
    if last_hidden_state.ndim != 3 or last_hidden_state.shape[1] < expected:
        raise ValueError(
            f"invalid DINOv3 hidden state {tuple(last_hidden_state.shape)} for {grid}x{grid}"
        )
    return last_hidden_state[:, -expected:, :]
