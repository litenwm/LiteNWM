"""LiteTokenFormer-WM-v2 models and token-level trajectory evaluators."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from recon_latents.action_prefix_v2 import ActionPrefixEncoderV2, ActionPrefixV2Config


def count_trainable_parameters(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.in_proj = nn.Linear(dim, 2 * hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value, gate = self.in_proj(x).chunk(2, dim=-1)
        return self.out_proj(value * F.silu(gate))


def integrate_se2_prefix(actions: torch.Tensor) -> torch.Tensor:
    """Integrate raw local [dx,dy,dyaw] actions to cumulative [x,y,yaw]."""
    if actions.ndim < 3 or actions.shape[-1] != 3:
        raise ValueError(f"expected actions [...,H,3], got {tuple(actions.shape)}")
    x = torch.zeros_like(actions[..., 0, 0])
    y = torch.zeros_like(x)
    yaw = torch.zeros_like(x)
    poses = []
    for step in range(actions.shape[-2]):
        dx, dy, dyaw = actions[..., step, :].unbind(dim=-1)
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)
        x = x + cos_yaw * dx - sin_yaw * dy
        y = y + sin_yaw * dx + cos_yaw * dy
        yaw = yaw + dyaw
        poses.append(torch.stack((x, y, yaw), dim=-1))
    return torch.stack(poses, dim=-2)


def build_v2_action_features(actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return 12D causal action features and cumulative SE(2) prefixes."""
    cumulative = integrate_se2_prefix(actions)
    previous = torch.cat((torch.zeros_like(actions[..., :1, :]), actions[..., :-1, :]), dim=-2)
    difference = actions - previous
    horizon = actions.shape[-2]
    step = torch.linspace(0.0, 1.0, horizon, device=actions.device, dtype=actions.dtype)
    step = step.view(*([1] * (actions.ndim - 2)), horizon, 1).expand(*actions.shape[:-1], 1)
    translation_norm = torch.linalg.vector_norm(actions[..., :2], dim=-1, keepdim=True)
    yaw_abs = actions[..., 2:3].abs()
    return torch.cat((actions, cumulative, difference, step, translation_norm, yaw_abs), dim=-1), cumulative


class ContinuousActionPrefixEncoder(nn.Module):
    def __init__(
        self,
        model_dim: int = 192,
        heads: int = 6,
        layers: int = 2,
        dropout: float = 0.0,
        ordered_action_gru: bool = False,
        ordered_action_scale_init: float = 0.05,
        horizon: int = 6,
    ) -> None:
        super().__init__()
        if horizon < 1:
            raise ValueError("horizon must be positive")
        self.model_dim = model_dim
        self.horizon = int(horizon)
        self.ordered_action_gru_enabled = bool(ordered_action_gru)
        self.input_proj = nn.Linear(12, model_dim)
        self.horizon_embed = nn.Parameter(torch.randn(self.horizon, model_dim) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=heads,
            dim_feedforward=3 * model_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(model_dim)
        if self.ordered_action_gru_enabled:
            self.ordered_gru = nn.GRU(12, model_dim, num_layers=1, batch_first=True)
            self.ordered_norm = nn.LayerNorm(model_dim)
            self.ordered_alpha = nn.Parameter(
                torch.full((self.horizon,), ordered_action_scale_init)
            )

    def forward(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if actions.shape[-2:] != (self.horizon, 3):
            raise ValueError(
                f"expected actions [...,{self.horizon},3], got {tuple(actions.shape)}"
            )
        features, cumulative = build_v2_action_features(actions)
        flat = features.reshape(-1, self.horizon, 12)
        x = self.input_proj(flat) + self.horizon_embed[None]
        causal_mask = torch.triu(
            torch.ones(
                self.horizon,
                self.horizon,
                device=x.device,
                dtype=torch.bool,
            ),
            diagonal=1,
        )
        encoded = self.norm(self.encoder(x, mask=causal_mask))
        if self.ordered_action_gru_enabled:
            ordered, _ = self.ordered_gru(flat)
            ordered = self.ordered_norm(ordered)
            encoded = self.norm(encoded + self.ordered_alpha[None, :, None] * ordered)
        return (
            encoded.reshape(*actions.shape[:-2], self.horizon, self.model_dim),
            cumulative,
        )


class SpatialMix(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.depthwise = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.pointwise = nn.Conv2d(dim, dim, kernel_size=1)
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x [B,T,256,D]
        batch, frames, tokens, dim = x.shape
        if tokens != 256:
            raise ValueError("SpatialMix expects a 16x16 token grid")
        residual = x
        image = self.norm(x).reshape(batch * frames, 16, 16, dim).permute(0, 3, 1, 2)
        image = self.pointwise(F.silu(self.depthwise(image)))
        mixed = image.permute(0, 2, 3, 1).reshape(batch, frames, tokens, dim)
        return residual + self.scale * mixed


class HistoryMotionEncoder(nn.Module):
    def __init__(self, input_dim: int = 384, model_dim: int = 192, heads: int = 6, layers: int = 2) -> None:
        super().__init__()
        self.model_dim = model_dim
        self.input_proj = nn.Linear(input_dim, model_dim)
        self.spatial_embed = nn.Parameter(torch.randn(256, model_dim) * 0.02)
        self.frame_embed = nn.Parameter(torch.randn(4, model_dim) * 0.02)
        temporal_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=heads,
            dim_feedforward=3 * model_dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(temporal_layer, num_layers=layers, enable_nested_tensor=False)
        self.spatial_mix = nn.ModuleList([SpatialMix(model_dim) for _ in range(layers)])
        self.norm = nn.LayerNorm(model_dim)

    def forward(self, history_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if history_tokens.shape[-3:] != (4, 256, 384):
            raise ValueError(f"expected history [B,4,256,384], got {tuple(history_tokens.shape)}")
        batch = history_tokens.shape[0]
        x = self.input_proj(history_tokens)
        x = x + self.frame_embed[None, :, None, :] + self.spatial_embed[None, None, :, :]
        temporal = x.permute(0, 2, 1, 3).reshape(batch * 256, 4, self.model_dim)
        temporal = self.temporal(temporal)
        x = temporal.reshape(batch, 256, 4, self.model_dim).permute(0, 2, 1, 3)
        for spatial in self.spatial_mix:
            x = spatial(x)
        x = self.norm(x)
        return x[:, -1], x.reshape(batch, 4 * 256, self.model_dim)


class SE2TokenWarp(nn.Module):
    def __init__(self, model_dim: int) -> None:
        super().__init__()
        self.theta_mlp = nn.Sequential(nn.Linear(3, 64), nn.SiLU(), nn.Linear(64, 4))
        self.bounds = (0.45, 0.12, 0.45, 0.45)
        nn.init.zeros_(self.theta_mlp[-1].weight)
        nn.init.zeros_(self.theta_mlp[-1].bias)
        self.model_dim = model_dim

    def theta(self, cumulative_pose: torch.Tensor) -> torch.Tensor:
        raw = torch.tanh(self.theta_mlp(cumulative_pose))
        bounds = raw.new_tensor(self.bounds)
        rotation, log_scale, tx, ty = (raw * bounds).unbind(dim=-1)
        scale = torch.exp(log_scale)
        cos_r = torch.cos(rotation) * scale
        sin_r = torch.sin(rotation) * scale
        theta = torch.zeros(*raw.shape[:-1], 2, 3, device=raw.device, dtype=raw.dtype)
        theta[..., 0, 0] = cos_r
        theta[..., 0, 1] = -sin_r
        theta[..., 1, 0] = sin_r
        theta[..., 1, 1] = cos_r
        theta[..., 0, 2] = tx
        theta[..., 1, 2] = ty
        return theta

    def forward(self, tokens: torch.Tensor, cumulative_pose: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # tokens [B,256,C], poses [B,K,H,3]
        batch, candidates, horizon = cumulative_pose.shape[:3]
        channels = tokens.shape[-1]
        theta = self.theta(cumulative_pose).reshape(batch * candidates * horizon, 2, 3)
        images = tokens.reshape(batch, 1, 1, 16, 16, channels).expand(batch, candidates, horizon, 16, 16, channels)
        images = images.reshape(-1, 16, 16, channels).permute(0, 3, 1, 2)
        grid = F.affine_grid(theta, images.shape, align_corners=False)
        warped = F.grid_sample(images, grid, mode="bilinear", padding_mode="border", align_corners=False)
        warped = warped.permute(0, 2, 3, 1).reshape(batch, candidates, horizon, 256, channels)
        return warped, theta.reshape(batch, candidates, horizon, 2, 3)


def window_partition(x: torch.Tensor, window: int) -> tuple[torch.Tensor, tuple[int, ...]]:
    # x [B,H,256,D]
    batch, horizon, _, dim = x.shape
    grid = x.reshape(batch, horizon, 16, 16, dim)
    blocks = grid.reshape(batch, horizon, 16 // window, window, 16 // window, window, dim)
    blocks = blocks.permute(0, 1, 2, 4, 3, 5, 6).reshape(-1, window * window, dim)
    return blocks, (batch, horizon, dim)


def window_reverse(blocks: torch.Tensor, shape: tuple[int, ...], window: int) -> torch.Tensor:
    batch, horizon, dim = shape
    grid = blocks.reshape(batch, horizon, 16 // window, 16 // window, window, window, dim)
    grid = grid.permute(0, 1, 2, 4, 3, 5, 6).reshape(batch, horizon, 256, dim)
    return grid


class FactorizedRolloutBlock(nn.Module):
    def __init__(
        self,
        dim: int = 192,
        heads: int = 6,
        window: int = 8,
        mlp_ratio: int = 3,
        action_cross_attention_floor: float = 0.0,
        action_cross_attention_enforce_floor: bool = False,
        strong_action_cross_attention: bool = False,
        strong_action_beta_init: float = 0.01,
        strong_action_beta_max: float = 0.10,
        block_causal_action_integration: bool = False,
        block_causal_beta_init: float = 0.01,
        block_causal_beta_max: float = 0.10,
        block_causal_action_v2: bool = False,
        block_causal_v2_beta_init: float = 0.01,
        block_causal_v2_beta_max: float = 0.10,
        interleaved_action_state_fusion: bool = False,
        interleaved_beta_init: float = 0.01,
        interleaved_beta_max: float = 0.10,
        horizon: int = 6,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.window = window
        self.action_cross_attention_floor = float(action_cross_attention_floor)
        self.action_cross_attention_enforce_floor = bool(action_cross_attention_enforce_floor)
        self.temporal_norm = nn.LayerNorm(dim)
        self.temporal_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.spatial_norm = nn.LayerNorm(dim)
        self.spatial_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.action_norm = nn.LayerNorm(dim)
        self.action_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.strong_action_cross = (
            CausalActionPrefixCrossAttention(
                dim=dim,
                heads=heads,
                beta_init=strong_action_beta_init,
                beta_max=strong_action_beta_max,
                horizon=horizon,
            )
            if strong_action_cross_attention
            else None
        )
        self.block_causal_action = (
            BlockCausalActionStateFusion(
                dim=dim,
                heads=heads,
                beta_init=block_causal_beta_init,
                beta_max=block_causal_beta_max,
                horizon=horizon,
            )
            if block_causal_action_integration
            else None
        )
        self.block_causal_action_v2 = (
            TimeAlignedBlockCausalActionFusion(
                dim=dim,
                heads=heads,
                beta_init=block_causal_v2_beta_init,
                beta_max=block_causal_v2_beta_max,
                horizon=horizon,
            )
            if block_causal_action_v2
            else None
        )
        self.interleaved_action_state = (
            InterleavedActionStateCausalFusion(
                dim=dim,
                heads=heads,
                beta_init=interleaved_beta_init,
                beta_max=interleaved_beta_max,
                horizon=horizon,
            )
            if interleaved_action_state_fusion
            else None
        )
        self.history_norm = nn.LayerNorm(dim)
        self.history_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 3 * dim))
        self.mlp_norm = nn.LayerNorm(dim)
        self.mlp = SwiGLU(dim, mlp_ratio * dim)
        self.scales = nn.Parameter(torch.full((5,), 0.1))

    def forward(
        self,
        x: torch.Tensor,
        action_memory: torch.Tensor,
        history_memory: torch.Tensor,
        increment_memory: torch.Tensor | None = None,
        strong_action_memory: torch.Tensor | None = None,
        block_causal_increment_memory: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # x [BK,H,256,D], action [BK,H,D], history [BK,1024,D]
        bk, horizon, tokens, dim = x.shape
        causal = torch.triu(torch.ones(horizon, horizon, device=x.device, dtype=torch.bool), diagonal=1)
        temporal = self.temporal_norm(x).permute(0, 2, 1, 3).reshape(bk * tokens, horizon, dim)
        temporal = self.temporal_attn(temporal, temporal, temporal, attn_mask=causal, need_weights=False)[0]
        x = x + self.scales[0] * temporal.reshape(bk, tokens, horizon, dim).permute(0, 2, 1, 3)
        windows, shape = window_partition(self.spatial_norm(x), self.window)
        windows = self.spatial_attn(windows, windows, windows, need_weights=False)[0]
        x = x + self.scales[1] * window_reverse(windows, shape, self.window)
        action_q = self.action_norm(x).permute(0, 2, 1, 3).reshape(bk * tokens, horizon, dim)
        action_kv = action_memory[:, None].expand(bk, tokens, horizon, dim).reshape(bk * tokens, horizon, dim)
        action_out = self.action_attn(action_q, action_kv, action_kv, attn_mask=causal, need_weights=False)[0]
        if self.action_cross_attention_enforce_floor and self.action_cross_attention_floor > 0:
            action_scale = self.action_cross_attention_floor + self.scales[2].abs()
        else:
            action_scale = self.scales[2] + self.action_cross_attention_floor
        x = x + action_scale * action_out.reshape(bk, tokens, horizon, dim).permute(0, 2, 1, 3)
        if self.strong_action_cross is not None:
            x = self.strong_action_cross(
                x,
                action_memory if strong_action_memory is None else strong_action_memory,
            )
        if self.block_causal_action is not None:
            if increment_memory is None:
                raise ValueError("block-causal action integration requires increment_memory")
            x = self.block_causal_action(x, action_memory, increment_memory)
        if self.block_causal_action_v2 is not None:
            action_prefix = action_memory if strong_action_memory is None else strong_action_memory
            action_increment = (
                increment_memory
                if block_causal_increment_memory is None
                else block_causal_increment_memory
            )
            if action_increment is None:
                raise ValueError("block-causal action v2 requires increment memory")
            x = self.block_causal_action_v2(x, action_prefix, action_increment)
        if self.interleaved_action_state is not None:
            action_prefix = action_memory if strong_action_memory is None else strong_action_memory
            action_increment = (
                increment_memory
                if block_causal_increment_memory is None
                else block_causal_increment_memory
            )
            if action_increment is None:
                raise ValueError("interleaved action-state fusion requires increment memory")
            x = self.interleaved_action_state(x, action_prefix, action_increment)
        history_q = self.history_norm(x).reshape(bk, horizon * tokens, dim)
        history_out = self.history_attn(history_q, history_memory, history_memory, need_weights=False)[0]
        x = x + self.scales[3] * history_out.reshape(bk, horizon, tokens, dim)
        shift, scale, gate = self.modulation(action_memory).chunk(3, dim=-1)
        normalized = self.mlp_norm(x)
        normalized = normalized * (1.0 + scale[:, :, None]) + shift[:, :, None]
        x = x + self.scales[4] * torch.sigmoid(gate[:, :, None]) * self.mlp(normalized)
        return x


class CausalActionPrefixCrossAttention(nn.Module):
    """Bounded late-block action-prefix injection with exact horizon causality."""

    def __init__(
        self,
        dim: int,
        heads: int,
        beta_init: float = 0.01,
        beta_max: float = 0.10,
        horizon: int = 6,
    ) -> None:
        super().__init__()
        if not 0.0 < beta_init < beta_max:
            raise ValueError("strong action beta must satisfy 0 < init < max")
        self.dim = dim
        self.beta_max = float(beta_max)
        self.query_norm = nn.LayerNorm(dim)
        self.memory_norm = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.gate = nn.Linear(2 * dim, 1)
        ratio = torch.tensor(beta_init / beta_max, dtype=torch.float32)
        self.beta_raw = nn.Parameter(
            torch.full((horizon,), float(torch.atanh(ratio)))
        )
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)
        self.capture_diagnostics = False
        self.last_diagnostics: dict[str, torch.Tensor] = {}

    def effective_beta(self) -> torch.Tensor:
        return self.beta_max * torch.tanh(self.beta_raw)

    def forward(self, x: torch.Tensor, action_memory: torch.Tensor) -> torch.Tensor:
        # x [BK,H,256,D], action_memory [BK,H,D]. Query h can see c_1..c_h only.
        bk, horizon, tokens, dim = x.shape
        query = self.query_norm(x).permute(0, 2, 1, 3).reshape(bk * tokens, horizon, dim)
        memory = self.memory_norm(action_memory)
        memory = memory[:, None].expand(bk, tokens, horizon, dim).reshape(bk * tokens, horizon, dim)
        causal = torch.triu(torch.ones(horizon, horizon, device=x.device, dtype=torch.bool), diagonal=1)
        attended, weights = self.attention(
            query,
            memory,
            memory,
            attn_mask=causal,
            need_weights=self.capture_diagnostics,
            average_attn_weights=False,
        )
        current_prefix = action_memory[:, :, None, :].expand(bk, horizon, tokens, dim)
        gate = torch.sigmoid(self.gate(torch.cat((self.query_norm(x), current_prefix), dim=-1)))
        attended_tokens = attended.reshape(bk, tokens, horizon, dim).permute(0, 2, 1, 3)
        beta = self.effective_beta().view(1, horizon, 1, 1)
        correction = beta * gate * attended_tokens
        if self.capture_diagnostics:
            with torch.no_grad():
                correction_energy = correction.float().square().mean(dim=(0, 2, 3))
                hidden_energy = x.float().square().mean(dim=(0, 2, 3)).clamp_min(1e-12)
                spatial_variance = torch.linalg.vector_norm(correction.float(), dim=-1).var(dim=2, unbiased=False).mean(0)
                attention_mean = weights.float().mean(dim=(0, 1))
                entropy = -(attention_mean.clamp_min(1e-12) * attention_mean.clamp_min(1e-12).log()).sum(-1)
                self.last_diagnostics = {
                    "beta": self.effective_beta().detach(),
                    "gate_mean": gate.float().mean(dim=(0, 2, 3)),
                    "correction_energy_fraction": correction_energy / hidden_energy,
                    "spatial_variance": spatial_variance,
                    "attention_mean": attention_mean,
                    "attention_entropy": entropy,
                }
        return x + correction


class BlockCausalActionStateFusion(nn.Module):
    """Fuse same-time, previous-step, and causal-prefix action tokens into hidden patches."""

    def __init__(
        self,
        dim: int,
        heads: int,
        beta_init: float = 0.01,
        beta_max: float = 0.10,
        horizon: int = 6,
    ) -> None:
        super().__init__()
        if not 0.0 < beta_init < beta_max:
            raise ValueError("block-causal beta must satisfy 0 < init < max")
        self.beta_max = float(beta_max)
        self.query_norm = nn.LayerNorm(dim)
        self.memory_norm = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.gate = nn.Linear(2 * dim, 1)
        ratio = torch.tensor(beta_init / beta_max, dtype=torch.float32)
        self.beta_raw = nn.Parameter(
            torch.full((horizon,), float(torch.atanh(ratio)))
        )
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)
        self.capture_diagnostics = False
        self.last_diagnostics: dict[str, torch.Tensor] = {}

    def effective_beta(self) -> torch.Tensor:
        return self.beta_max * torch.tanh(self.beta_raw)

    def forward(
        self,
        x: torch.Tensor,
        action_prefix: torch.Tensor,
        action_increment: torch.Tensor,
    ) -> torch.Tensor:
        # Memory at h is [increment_(h-1), increment_h, prefix_h]; no future action is visible.
        bk, horizon, tokens, dim = x.shape
        previous = torch.cat((torch.zeros_like(action_increment[:, :1]), action_increment[:, :-1]), dim=1)
        memory = torch.stack((previous, action_increment, action_prefix), dim=2)
        memory = self.memory_norm(memory).reshape(bk * horizon, 3, dim)
        query = self.query_norm(x).reshape(bk * horizon, tokens, dim)
        attended, weights = self.attention(
            query,
            memory,
            memory,
            need_weights=self.capture_diagnostics,
            average_attn_weights=False,
        )
        attended = attended.reshape(bk, horizon, tokens, dim)
        prefix_expanded = action_prefix[:, :, None].expand(bk, horizon, tokens, dim)
        gate = torch.sigmoid(self.gate(torch.cat((self.query_norm(x), prefix_expanded), dim=-1)))
        beta = self.effective_beta().view(1, horizon, 1, 1)
        correction = beta * gate * attended
        if self.capture_diagnostics:
            with torch.no_grad():
                correction_energy = correction.float().square().mean(dim=(0, 2, 3))
                hidden_energy = x.float().square().mean(dim=(0, 2, 3)).clamp_min(1e-12)
                spatial_variance = torch.linalg.vector_norm(correction.float(), dim=-1).var(dim=2, unbiased=False).mean(0)
                # [BK*H, heads, tokens, 3] -> [H, 3]
                attention_mean = weights.float().reshape(bk, horizon, weights.shape[1], tokens, 3).mean(dim=(0, 2, 3))
                entropy = -(attention_mean.clamp_min(1e-12) * attention_mean.clamp_min(1e-12).log()).sum(-1)
                self.last_diagnostics = {
                    "beta": self.effective_beta().detach(),
                    "gate_mean": gate.float().mean(dim=(0, 2, 3)),
                    "correction_energy_fraction": correction_energy / hidden_energy,
                    "spatial_variance": spatial_variance,
                    "attention_mean": attention_mean,
                    "attention_entropy": entropy,
                }
        return x + correction


class TimeAlignedBlockCausalActionFusion(nn.Module):
    """Attend from each visual horizon to every increment/prefix token available so far."""

    def __init__(
        self,
        dim: int,
        heads: int,
        beta_init: float = 0.01,
        beta_max: float = 0.10,
        horizon: int = 6,
    ) -> None:
        super().__init__()
        if not 0.0 < beta_init < beta_max:
            raise ValueError("block-causal v2 beta must satisfy 0 < init < max")
        self.beta_max = float(beta_max)
        self.query_norm = nn.LayerNorm(dim)
        self.memory_norm = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.gate = nn.Linear(3 * dim, 1)
        ratio = torch.tensor(beta_init / beta_max, dtype=torch.float32)
        self.beta_raw = nn.Parameter(
            torch.full((horizon,), float(torch.atanh(ratio)))
        )
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)
        self.capture_diagnostics = False
        self.last_diagnostics: dict[str, torch.Tensor] = {}

    def effective_beta(self) -> torch.Tensor:
        return self.beta_max * torch.tanh(self.beta_raw)

    def forward(
        self,
        x: torch.Tensor,
        action_prefix: torch.Tensor,
        action_increment: torch.Tensor,
    ) -> torch.Tensor:
        # Memory is [a1,c1,a2,c2,...,aH,cH]; query h cannot see later entries.
        bk, horizon, tokens, dim = x.shape
        memory = torch.stack((action_increment, action_prefix), dim=2).reshape(
            bk, 2 * horizon, dim
        )
        memory = self.memory_norm(memory)
        memory = memory[:, None].expand(bk, tokens, 2 * horizon, dim).reshape(
            bk * tokens, 2 * horizon, dim
        )
        query = self.query_norm(x).permute(0, 2, 1, 3).reshape(
            bk * tokens, horizon, dim
        )
        query_step = torch.arange(horizon, device=x.device)[:, None]
        memory_step = torch.arange(horizon, device=x.device).repeat_interleave(2)[None]
        future_mask = memory_step > query_step
        attended, weights = self.attention(
            query,
            memory,
            memory,
            attn_mask=future_mask,
            need_weights=self.capture_diagnostics,
            average_attn_weights=False,
        )
        attended = attended.reshape(bk, tokens, horizon, dim).permute(0, 2, 1, 3)
        same_increment = action_increment[:, :, None].expand(bk, horizon, tokens, dim)
        same_prefix = action_prefix[:, :, None].expand(bk, horizon, tokens, dim)
        gate = torch.sigmoid(
            self.gate(torch.cat((self.query_norm(x), same_increment, same_prefix), dim=-1))
        )
        beta = self.effective_beta().view(1, horizon, 1, 1)
        correction = beta * gate * attended
        if self.capture_diagnostics:
            with torch.no_grad():
                correction_energy = correction.float().square().mean(dim=(0, 2, 3))
                hidden_energy = x.float().square().mean(dim=(0, 2, 3)).clamp_min(1e-12)
                spatial_variance = torch.linalg.vector_norm(
                    correction.float(), dim=-1
                ).var(dim=2, unbiased=False).mean(0)
                attention_mean = weights.float().mean(dim=(0, 1))
                entropy = -(
                    attention_mean.clamp_min(1e-12)
                    * attention_mean.clamp_min(1e-12).log()
                ).sum(-1)
                self.last_diagnostics = {
                    "beta": self.effective_beta().detach(),
                    "gate_mean": gate.float().mean(dim=(0, 2, 3)),
                    "correction_energy_fraction": correction_energy / hidden_energy,
                    "spatial_variance": spatial_variance,
                    "attention_mean": attention_mean,
                    "attention_entropy": entropy,
                }
        return x + correction


class InterleavedActionStateCausalFusion(nn.Module):
    """Two-layer block-causal mixer over [action_h, visual_h] token pairs."""

    def __init__(
        self,
        dim: int,
        heads: int,
        beta_init: float = 0.01,
        beta_max: float = 0.10,
        horizon: int = 6,
    ) -> None:
        super().__init__()
        if not 0.0 < beta_init < beta_max:
            raise ValueError("interleaved beta must satisfy 0 < init < max")
        self.beta_max = float(beta_max)
        self.state_norm = nn.LayerNorm(dim)
        self.action_norm = nn.LayerNorm(2 * dim)
        self.action_fusion = nn.Linear(2 * dim, dim)
        self.type_embedding = nn.Parameter(torch.randn(2, dim) * 0.02)
        self.attention_norms = nn.ModuleList([nn.LayerNorm(dim), nn.LayerNorm(dim)])
        self.attention_layers = nn.ModuleList(
            [nn.MultiheadAttention(dim, heads, batch_first=True) for _ in range(2)]
        )
        self.attention_scales = nn.Parameter(torch.full((2,), 0.10))
        self.gate = nn.Linear(3 * dim, 1)
        ratio = torch.tensor(beta_init / beta_max, dtype=torch.float32)
        self.beta_raw = nn.Parameter(
            torch.full((horizon,), float(torch.atanh(ratio)))
        )
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)
        self.capture_diagnostics = False
        self.last_diagnostics: dict[str, torch.Tensor] = {}

    def effective_beta(self) -> torch.Tensor:
        return self.beta_max * torch.tanh(self.beta_raw)

    def forward(
        self,
        x: torch.Tensor,
        action_prefix: torch.Tensor,
        action_increment: torch.Tensor,
    ) -> torch.Tensor:
        bk, horizon, tokens, dim = x.shape
        action = self.action_fusion(
            self.action_norm(torch.cat((action_increment, action_prefix), dim=-1))
        )
        action = action[:, None].expand(bk, tokens, horizon, dim).reshape(
            bk * tokens, horizon, dim
        )
        state = self.state_norm(x).permute(0, 2, 1, 3).reshape(
            bk * tokens, horizon, dim
        )
        sequence = torch.stack((action, state), dim=2)
        sequence = sequence + self.type_embedding[None, None]
        sequence = sequence.reshape(bk * tokens, 2 * horizon, dim)
        initial_state = sequence[:, 1::2]
        token_time = torch.arange(horizon, device=x.device).repeat_interleave(2)
        future_mask = token_time[None] > token_time[:, None]
        final_weights = None
        for index, (norm, attention) in enumerate(
            zip(self.attention_norms, self.attention_layers)
        ):
            update, weights = attention(
                norm(sequence),
                norm(sequence),
                norm(sequence),
                attn_mask=future_mask,
                need_weights=self.capture_diagnostics and index == 1,
                average_attn_weights=False,
            )
            sequence = sequence + self.attention_scales[index] * update
            if self.capture_diagnostics and index == 1:
                final_weights = weights
        state_update = (sequence[:, 1::2] - initial_state).reshape(
            bk, tokens, horizon, dim
        ).permute(0, 2, 1, 3)
        same_increment = action_increment[:, :, None].expand(bk, horizon, tokens, dim)
        same_prefix = action_prefix[:, :, None].expand(bk, horizon, tokens, dim)
        gate = torch.sigmoid(
            self.gate(torch.cat((self.state_norm(x), same_increment, same_prefix), dim=-1))
        )
        beta = self.effective_beta().view(1, horizon, 1, 1)
        correction = beta * gate * state_update
        if self.capture_diagnostics:
            with torch.no_grad():
                correction_energy = correction.float().square().mean(dim=(0, 2, 3))
                hidden_energy = x.float().square().mean(dim=(0, 2, 3)).clamp_min(1e-12)
                spatial_variance = torch.linalg.vector_norm(
                    correction.float(), dim=-1
                ).var(dim=2, unbiased=False).mean(0)
                if final_weights is None:
                    raise RuntimeError("missing interleaved attention diagnostics")
                attention_mean = final_weights.float()[:, :, 1::2].mean(dim=(0, 1))
                entropy = -(
                    attention_mean.clamp_min(1e-12)
                    * attention_mean.clamp_min(1e-12).log()
                ).sum(-1)
                self.last_diagnostics = {
                    "beta": self.effective_beta().detach(),
                    "gate_mean": gate.float().mean(dim=(0, 2, 3)),
                    "correction_energy_fraction": correction_energy / hidden_energy,
                    "spatial_variance": spatial_variance,
                    "attention_mean": attention_mean,
                    "attention_entropy": entropy,
                }
        return x + correction


class CausalTokenActionGRU(nn.Module):
    """Same-width token recurrence driven by current action increment and causal prefix."""

    def __init__(
        self,
        dim: int,
        gamma_init: float = 0.01,
        gamma_max: float = 0.10,
        horizon: int = 6,
    ) -> None:
        super().__init__()
        if not 0.0 < gamma_init < gamma_max:
            raise ValueError("causal token-action gamma must satisfy 0 < init < max")
        self.dim = dim
        self.gamma_max = float(gamma_max)
        self.input_norm = nn.LayerNorm(dim)
        self.action_norm = nn.LayerNorm(dim)
        self.action_fusion = nn.Linear(2 * dim, dim)
        self.cell = nn.GRUCell(dim, dim)
        self.output_gate = nn.Linear(2 * dim, 1)
        ratio = torch.tensor(gamma_init / gamma_max, dtype=torch.float32)
        self.gamma_raw = nn.Parameter(
            torch.full((horizon,), float(torch.atanh(ratio)))
        )
        nn.init.zeros_(self.output_gate.weight)
        nn.init.zeros_(self.output_gate.bias)
        self.capture_diagnostics = False
        self.last_diagnostics: dict[str, torch.Tensor] = {}

    def effective_gamma(self) -> torch.Tensor:
        return self.gamma_max * torch.tanh(self.gamma_raw)

    def forward(
        self,
        x: torch.Tensor,
        action_prefix: torch.Tensor,
        action_increment: torch.Tensor,
    ) -> torch.Tensor:
        bk, horizon, tokens, dim = x.shape
        action = self.action_fusion(torch.cat((action_increment, action_prefix), dim=-1))
        action = self.action_norm(action)
        state = x[:, 0].reshape(bk * tokens, dim)
        outputs = []
        corrections = []
        gates = []
        for horizon_index in range(horizon):
            token = self.input_norm(x[:, horizon_index])
            action_h = action[:, horizon_index, None].expand(bk, tokens, dim)
            cell_input = token + action_h
            if horizon_index == 0:
                previous = token.reshape(bk * tokens, dim)
            else:
                previous = state
            state = self.cell(cell_input.reshape(bk * tokens, dim), previous)
            state_tokens = state.reshape(bk, tokens, dim)
            gate = torch.sigmoid(self.output_gate(torch.cat((token, action_h), dim=-1)))
            gamma = self.effective_gamma()[horizon_index]
            correction = gamma * gate * (state_tokens - token)
            output = x[:, horizon_index] + correction
            # Propagate the corrected hidden state causally to the next horizon.
            state = output.reshape(bk * tokens, dim)
            outputs.append(output)
            corrections.append(correction)
            gates.append(gate)
        result = torch.stack(outputs, dim=1)
        if self.capture_diagnostics:
            with torch.no_grad():
                correction = torch.stack(corrections, dim=1).float()
                gate = torch.stack(gates, dim=1).float()
                correction_energy = correction.square().mean(dim=(0, 2, 3))
                hidden_energy = x.float().square().mean(dim=(0, 2, 3)).clamp_min(1e-12)
                spatial_variance = torch.linalg.vector_norm(correction, dim=-1).var(dim=2, unbiased=False).mean(0)
                gate_mean = gate.mean(dim=(0, 2, 3))
                carry_update = torch.stack((1.0 - gate_mean, gate_mean), dim=-1)
                entropy = -(carry_update.clamp_min(1e-12) * carry_update.clamp_min(1e-12).log()).sum(-1)
                self.last_diagnostics = {
                    "beta": self.effective_gamma().detach(),
                    "gate_mean": gate_mean,
                    "correction_energy_fraction": correction_energy / hidden_energy,
                    "spatial_variance": spatial_variance,
                    "attention_mean": carry_update,
                    "attention_entropy": entropy,
                }
        return result


@dataclass(frozen=True)
class LiteTokenFormerConfig:
    input_dim: int = 384
    horizon: int = 6
    model_dim: int = 192
    heads: int = 6
    history_layers: int = 2
    action_layers: int = 2
    rollout_blocks: int = 6
    spatial_window: int = 8
    mlp_ratio: int = 3
    action_output_residual: bool = False
    action_residual_alpha_init: float = 0.01
    action_cross_attention_floor: float = 0.0
    action_cross_attention_enforce_floor: bool = False
    ordered_action_gru: bool = False
    ordered_action_scale_init: float = 0.05
    dino_wm_action_concat: bool = False
    strong_action_cross_attention: bool = False
    strong_action_cross_attention_start_block: int = 4
    strong_action_beta_init: float = 0.01
    strong_action_beta_max: float = 0.10
    block_causal_action_integration: bool = False
    block_causal_action_start_block: int = 4
    block_causal_beta_init: float = 0.01
    block_causal_beta_max: float = 0.10
    block_causal_action_v2: bool = False
    block_causal_action_v2_start_block: int = 4
    block_causal_v2_beta_init: float = 0.01
    block_causal_v2_beta_max: float = 0.10
    interleaved_action_state_fusion: bool = False
    interleaved_action_state_start_block: int = 4
    interleaved_beta_init: float = 0.01
    interleaved_beta_max: float = 0.10
    causal_token_action_gru: bool = False
    causal_token_action_gru_after_block: int = 3
    causal_token_action_gamma_init: float = 0.01
    causal_token_action_gamma_max: float = 0.10
    action_prefix_v2: bool = False
    action_prefix_v2_hidden_dim: int = 128
    action_prefix_v2_heads: int = 4
    action_prefix_v2_layers: int = 2
    action_prefix_v2_mlp_ratio: int = 2


class LiteTokenFormerWMV2(nn.Module):
    """Direct one-pass dense-token dynamics with 4-frame context."""

    def __init__(self, config: LiteTokenFormerConfig | None = None) -> None:
        super().__init__()
        self.config = config or LiteTokenFormerConfig()
        cfg = self.config
        if cfg.horizon < 1:
            raise ValueError("horizon must be positive")
        if cfg.action_prefix_v2 and cfg.horizon != 6:
            raise ValueError(
                "ActionPrefixEncoderV2 remains an H6-only optional module"
            )
        if cfg.action_output_residual:
            raise ValueError(
                "action_output_residual was rolled back after ACTION_RESIDUAL_DISTORTS_DYNAMICS; "
                "use strong_action_cross_attention for the bounded internal repair"
            )
        self.history_encoder = HistoryMotionEncoder(cfg.input_dim, cfg.model_dim, cfg.heads, cfg.history_layers)
        self.action_encoder = ContinuousActionPrefixEncoder(
            cfg.model_dim,
            cfg.heads,
            cfg.action_layers,
            ordered_action_gru=cfg.ordered_action_gru,
            ordered_action_scale_init=cfg.ordered_action_scale_init,
            horizon=cfg.horizon,
        )
        if cfg.action_prefix_v2:
            self.action_prefix_v2_encoder = ActionPrefixEncoderV2(
                ActionPrefixV2Config(
                    output_dim=cfg.model_dim,
                    hidden_dim=cfg.action_prefix_v2_hidden_dim,
                    heads=cfg.action_prefix_v2_heads,
                    layers=cfg.action_prefix_v2_layers,
                    mlp_ratio=cfg.action_prefix_v2_mlp_ratio,
                )
            )
        if cfg.dino_wm_action_concat:
            self.action_concat_embed = nn.Linear(12, cfg.model_dim)
            self.action_concat_fusion = nn.Linear(2 * cfg.model_dim, cfg.model_dim)
            nn.init.zeros_(self.action_concat_fusion.weight)
            nn.init.zeros_(self.action_concat_fusion.bias)
            with torch.no_grad():
                identity = torch.eye(cfg.model_dim)
                self.action_concat_fusion.weight[:, : cfg.model_dim].copy_(identity)
                self.action_concat_fusion.weight[:, cfg.model_dim :].copy_(0.05 * identity)
        self.warp = SE2TokenWarp(cfg.model_dim)
        self.horizon_embed = nn.Parameter(
            torch.randn(cfg.horizon, cfg.model_dim) * 0.02
        )
        self.blocks = nn.ModuleList(
            [
                FactorizedRolloutBlock(
                    cfg.model_dim,
                    cfg.heads,
                    cfg.spatial_window,
                    cfg.mlp_ratio,
                    cfg.action_cross_attention_floor,
                    cfg.action_cross_attention_enforce_floor,
                    cfg.strong_action_cross_attention and block_index >= cfg.strong_action_cross_attention_start_block,
                    cfg.strong_action_beta_init,
                    cfg.strong_action_beta_max,
                    cfg.block_causal_action_integration and block_index >= cfg.block_causal_action_start_block,
                    cfg.block_causal_beta_init,
                    cfg.block_causal_beta_max,
                    cfg.block_causal_action_v2 and block_index >= cfg.block_causal_action_v2_start_block,
                    cfg.block_causal_v2_beta_init,
                    cfg.block_causal_v2_beta_max,
                    cfg.interleaved_action_state_fusion
                    and block_index >= cfg.interleaved_action_state_start_block,
                    cfg.interleaved_beta_init,
                    cfg.interleaved_beta_max,
                    horizon=cfg.horizon,
                )
                for block_index in range(cfg.rollout_blocks)
            ]
        )
        self.causal_token_action_gru = (
            CausalTokenActionGRU(
                cfg.model_dim,
                cfg.causal_token_action_gamma_init,
                cfg.causal_token_action_gamma_max,
                horizon=cfg.horizon,
            )
            if cfg.causal_token_action_gru
            else None
        )
        self.output_norm = nn.LayerNorm(cfg.model_dim)
        self.delta_head = nn.Linear(cfg.model_dim, cfg.input_dim)
        self.logvar_head = nn.Linear(cfg.model_dim, 1)
        self.action_decode_head = nn.Sequential(nn.LayerNorm(cfg.model_dim), nn.Linear(cfg.model_dim, 3))
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)

    def encode_history(self, history_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.history_encoder(history_tokens)

    def forward(self, history_tokens: torch.Tensor, candidate_actions: torch.Tensor) -> dict[str, torch.Tensor]:
        if candidate_actions.ndim == 3:
            candidate_actions = candidate_actions[:, None]
        horizon = self.config.horizon
        if candidate_actions.shape[-2:] != (horizon, 3):
            raise ValueError(
                f"expected candidate actions [B,K,{horizon},3], "
                f"got {tuple(candidate_actions.shape)}"
            )
        batch, candidates = candidate_actions.shape[:2]
        motion_state, history_memory = self.encode_history(history_tokens)
        action_memory, cumulative = self.action_encoder(candidate_actions)
        action_prefix_v2_features = None
        if self.config.action_prefix_v2:
            action_prefix_v2_features = self.action_prefix_v2_encoder.forward_features(candidate_actions)
        warped_state, theta = self.warp(motion_state, cumulative)
        base_query = warped_state + self.horizon_embed[None, None, :, None]
        if self.config.dino_wm_action_concat:
            raw_action_features, _ = build_v2_action_features(candidate_actions)
            concat_action_tokens = self.action_concat_embed(raw_action_features)
            tiled_action = concat_action_tokens[:, :, :, None, :].expand(-1, -1, -1, 256, -1)
            query = self.action_concat_fusion(torch.cat((base_query, tiled_action), dim=-1))
        else:
            query = base_query + action_memory[:, :, :, None]
        flat = query.reshape(
            batch * candidates, horizon, 256, self.config.model_dim
        )
        action_flat = action_memory.reshape(
            batch * candidates, horizon, self.config.model_dim
        )
        strong_action_flat = (
            action_prefix_v2_features["action_prefix_tokens"].reshape(
                batch * candidates, horizon, self.config.model_dim
            )
            if action_prefix_v2_features is not None
            else action_flat
        )
        history_flat = history_memory[:, None].expand(batch, candidates, 1024, self.config.model_dim).reshape(
            batch * candidates, 1024, self.config.model_dim
        )
        increment_flat = (
            concat_action_tokens.reshape(
                batch * candidates, horizon, self.config.model_dim
            )
            if self.config.dino_wm_action_concat
            else action_flat
        )
        block_causal_increment_flat = (
            action_prefix_v2_features["action_increment_tokens"].reshape(
                batch * candidates, horizon, self.config.model_dim
            )
            if action_prefix_v2_features is not None
            else increment_flat
        )
        for block_index, block in enumerate(self.blocks):
            flat = block(
                flat,
                action_flat,
                history_flat,
                increment_flat,
                strong_action_memory=strong_action_flat,
                block_causal_increment_memory=block_causal_increment_flat,
            )
            if (
                self.causal_token_action_gru is not None
                and block_index == self.config.causal_token_action_gru_after_block
            ):
                flat = self.causal_token_action_gru(flat, action_flat, increment_flat)
        hidden = self.output_norm(flat).reshape(
            batch, candidates, horizon, 256, self.config.model_dim
        )
        predicted_delta = self.delta_head(hidden)
        current_raw = history_tokens[:, -1]
        warped_raw, _ = self.warp(current_raw, cumulative)
        predicted = warped_raw + predicted_delta
        logvar = self.logvar_head(hidden).squeeze(-1).clamp(-8.0, 8.0)
        decoded_action = self.action_decode_head(hidden.mean(dim=3))
        output = {
            "predicted_tokens": predicted,
            "predicted_delta_tokens": predicted_delta,
            "predicted_logvar": logvar,
            "decoded_action": decoded_action,
            "reference_tokens": warped_raw,
            "action_prefix_tokens": action_memory,
            "cumulative_pose": cumulative,
            "warp_theta": theta,
            "motion_state_tokens": motion_state,
        }
        strong_diagnostics = []
        for block_index, block in enumerate(self.blocks):
            if block.strong_action_cross is not None and block.strong_action_cross.last_diagnostics:
                strong_diagnostics.append({"block_index": block_index, **block.strong_action_cross.last_diagnostics})
        if strong_diagnostics:
            output["strong_action_cross_attention_diagnostics"] = strong_diagnostics
        block_causal_diagnostics = []
        for block_index, block in enumerate(self.blocks):
            if block.block_causal_action is not None and block.block_causal_action.last_diagnostics:
                block_causal_diagnostics.append({"block_index": block_index, **block.block_causal_action.last_diagnostics})
        if block_causal_diagnostics:
            output["block_causal_action_diagnostics"] = block_causal_diagnostics
        block_causal_v2_diagnostics = []
        for block_index, block in enumerate(self.blocks):
            if (
                block.block_causal_action_v2 is not None
                and block.block_causal_action_v2.last_diagnostics
            ):
                block_causal_v2_diagnostics.append(
                    {"block_index": block_index, **block.block_causal_action_v2.last_diagnostics}
                )
        if block_causal_v2_diagnostics:
            output["block_causal_action_v2_diagnostics"] = block_causal_v2_diagnostics
        interleaved_diagnostics = []
        for block_index, block in enumerate(self.blocks):
            if (
                block.interleaved_action_state is not None
                and block.interleaved_action_state.last_diagnostics
            ):
                interleaved_diagnostics.append(
                    {"block_index": block_index, **block.interleaved_action_state.last_diagnostics}
                )
        if interleaved_diagnostics:
            output["interleaved_action_state_diagnostics"] = interleaved_diagnostics
        if self.causal_token_action_gru is not None and self.causal_token_action_gru.last_diagnostics:
            output["causal_token_action_gru_diagnostics"] = self.causal_token_action_gru.last_diagnostics
        if self.config.dino_wm_action_concat:
            output["concat_action_tokens"] = concat_action_tokens
        if action_prefix_v2_features is not None:
            output["legacy_action_prefix_tokens"] = action_memory
            output["action_prefix_tokens"] = action_prefix_v2_features["action_prefix_tokens"]
            output["action_prefix_v2_fusion_weights"] = action_prefix_v2_features["fusion_weights"]
            output["action_prefix_v2_increment_tokens"] = action_prefix_v2_features["increment_tokens"]
            output["action_prefix_v2_prefix_state_tokens"] = action_prefix_v2_features["prefix_state_tokens"]
        return output

    def capture_strong_action_attention_diagnostics(self, enabled: bool = True) -> None:
        for block in self.blocks:
            if block.strong_action_cross is not None:
                block.strong_action_cross.capture_diagnostics = enabled
                if not enabled:
                    block.strong_action_cross.last_diagnostics = {}

    def capture_block_causal_action_diagnostics(self, enabled: bool = True) -> None:
        for block in self.blocks:
            if block.block_causal_action is not None:
                block.block_causal_action.capture_diagnostics = enabled
                if not enabled:
                    block.block_causal_action.last_diagnostics = {}
            if block.block_causal_action_v2 is not None:
                block.block_causal_action_v2.capture_diagnostics = enabled
                if not enabled:
                    block.block_causal_action_v2.last_diagnostics = {}
            if block.interleaved_action_state is not None:
                block.interleaved_action_state.capture_diagnostics = enabled
                if not enabled:
                    block.interleaved_action_state.last_diagnostics = {}

    def capture_causal_token_action_gru_diagnostics(self, enabled: bool = True) -> None:
        if self.causal_token_action_gru is not None:
            self.causal_token_action_gru.capture_diagnostics = enabled
            if not enabled:
                self.causal_token_action_gru.last_diagnostics = {}


class MultiTaskTrajectoryHeads(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.metric_head = nn.Linear(dim, 3)
        self.success_head = nn.Linear(dim, 3)
        self.risk_head = nn.Linear(dim, 1)
        self.rank_head = nn.Linear(dim, 1)
        self.confidence_head = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "metric_delta": self.metric_head(x),
            "success_logits": self.success_head(x),
            "risk_logit": self.risk_head(x).squeeze(-1),
            "ranking_score": self.rank_head(x).squeeze(-1),
            "confidence": torch.sigmoid(self.confidence_head(x).squeeze(-1)),
        }


class GoalConditionedTokenTrajectoryEvaluatorV2(nn.Module):
    """No-source token-level goal-conditioned H6 trajectory evaluator."""

    def __init__(self, token_dim: int = 384, model_dim: int = 192, heads: int = 6, temporal_layers: int = 2) -> None:
        super().__init__()
        self.token_dim = token_dim
        self.model_dim = model_dim
        self.pred_proj = nn.Linear(token_dim, model_dim)
        self.goal_proj = nn.Linear(token_dim, model_dim)
        self.current_proj = nn.Linear(token_dim, model_dim)
        self.action_proj = nn.Linear(3, model_dim)
        self.uncertainty_proj = nn.Linear(1, model_dim)
        self.goal_cross = nn.MultiheadAttention(model_dim, heads, batch_first=True)
        self.pool_score = nn.Linear(model_dim, 1)
        self.horizon_embed = nn.Parameter(torch.randn(6, model_dim) * 0.02)
        layer = nn.TransformerEncoderLayer(
            model_dim, heads, dim_feedforward=3 * model_dim, dropout=0.05,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(layer, temporal_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(model_dim)
        self.heads = MultiTaskTrajectoryHeads(model_dim)

    def forward(
        self,
        predicted_tokens: torch.Tensor,
        goal_tokens: torch.Tensor,
        current_tokens: torch.Tensor,
        candidate_actions: torch.Tensor,
        uncertainty: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        # predicted [B,K,6,256,C], goal/current [B,256,C], uncertainty [B,K,6,256]
        batch, candidates, horizon, tokens, _ = predicted_tokens.shape
        pred = self.pred_proj(predicted_tokens).reshape(batch * candidates * horizon, tokens, self.model_dim)
        goal = self.goal_proj(goal_tokens)[:, None, None].expand(batch, candidates, horizon, tokens, self.model_dim)
        goal = goal.reshape(batch * candidates * horizon, tokens, self.model_dim)
        current = self.current_proj(current_tokens)[:, None, None].expand(batch, candidates, horizon, tokens, self.model_dim)
        current = current.reshape(batch * candidates * horizon, tokens, self.model_dim)
        interacted = pred - current
        interacted = interacted + self.goal_cross(pred, goal, goal, need_weights=False)[0]
        unc = self.uncertainty_proj(uncertainty[..., None]).reshape(batch * candidates * horizon, tokens, self.model_dim)
        interacted = interacted + unc
        weights = torch.softmax(self.pool_score(interacted).squeeze(-1), dim=-1)
        step = (weights[..., None] * interacted).sum(dim=1).reshape(batch * candidates, horizon, self.model_dim)
        actions = self.action_proj(candidate_actions.reshape(batch * candidates, horizon, 3))
        step = step + actions + self.horizon_embed[None]
        temporal = self.temporal(step)
        pooled = self.norm(temporal.mean(dim=1)).reshape(batch, candidates, self.model_dim)
        return self.heads(pooled)


class FrozenV1TokenScorer(nn.Module):
    """Bounded scorer for projected frozen-v1 rollout tokens."""

    def __init__(self, variant: str, token_dim: int = 64, model_dim: int = 192, heads: int = 6) -> None:
        super().__init__()
        if variant not in {"token_difference_gru", "goal_cross_attention_gru", "goal_cross_attention_tiny_transformer"}:
            raise ValueError(f"unknown variant: {variant}")
        self.variant = variant
        self.token_dim = token_dim
        self.model_dim = model_dim
        self.pred_proj = nn.Linear(token_dim, model_dim)
        self.goal_proj = nn.Linear(token_dim, model_dim)
        self.current_proj = nn.Linear(token_dim, model_dim)
        self.action_proj = nn.Linear(3, model_dim)
        self.uncertainty_proj = nn.Linear(1, model_dim)
        self.cross = nn.MultiheadAttention(model_dim, heads, batch_first=True)
        self.pool_score = nn.Linear(model_dim, 1)
        self.horizon_embed = nn.Parameter(torch.randn(6, model_dim) * 0.02)
        if variant.endswith("gru"):
            self.temporal: nn.Module = nn.GRU(model_dim, model_dim, num_layers=2, batch_first=True, dropout=0.05)
        else:
            layer = nn.TransformerEncoderLayer(
                model_dim, heads, dim_feedforward=3 * model_dim, dropout=0.05,
                activation="gelu", batch_first=True, norm_first=True,
            )
            self.temporal = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(model_dim)
        self.heads = MultiTaskTrajectoryHeads(model_dim)

    def forward(
        self,
        rollout_tokens: torch.Tensor,
        goal_tokens: torch.Tensor,
        current_tokens: torch.Tensor,
        actions: torch.Tensor,
        uncertainty: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch, candidates, horizon, tokens, _ = rollout_tokens.shape
        pred = self.pred_proj(rollout_tokens).reshape(batch * candidates * horizon, tokens, self.model_dim)
        goal = self.goal_proj(goal_tokens)[:, None, None].expand(batch, candidates, horizon, tokens, self.model_dim)
        goal = goal.reshape(batch * candidates * horizon, tokens, self.model_dim)
        current = self.current_proj(current_tokens)[:, None, None].expand(batch, candidates, horizon, tokens, self.model_dim)
        current = current.reshape(batch * candidates * horizon, tokens, self.model_dim)
        if self.variant == "token_difference_gru":
            interacted = pred - goal + 0.5 * (pred - current)
        else:
            interacted = pred - current + self.cross(pred, goal, goal, need_weights=False)[0]
        unc = self.uncertainty_proj(uncertainty[..., None]).reshape(batch * candidates * horizon, tokens, self.model_dim)
        interacted = interacted + unc
        weights = torch.softmax(self.pool_score(interacted).squeeze(-1), dim=-1)
        step = (weights[..., None] * interacted).sum(dim=1).reshape(batch * candidates, horizon, self.model_dim)
        step = step + self.action_proj(actions.reshape(batch * candidates, horizon, 3)) + self.horizon_embed[None]
        if isinstance(self.temporal, nn.GRU):
            temporal, _ = self.temporal(step)
        else:
            temporal = self.temporal(step)
        pooled = self.norm(temporal.mean(dim=1)).reshape(batch, candidates, self.model_dim)
        return self.heads(pooled)


class ActionOnlyTrajectoryControl(nn.Module):
    def __init__(self, hidden_dim: int = 192) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(18, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim), nn.SiLU()
        )
        self.heads = MultiTaskTrajectoryHeads(hidden_dim)

    def forward(self, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.heads(self.net(actions.flatten(-2)))
