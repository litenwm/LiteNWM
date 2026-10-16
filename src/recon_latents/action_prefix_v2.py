"""Causal continuous-action prefix representation for H1-H6 world-model conditioning."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


HORIZON = 6
ACTION_DIM = 3
INCREMENT_FEATURE_DIM = 14
PREFIX_FEATURE_DIM = 7


def integrate_se2_prefix_v2(actions: torch.Tensor) -> torch.Tensor:
    """Integrate local [dx, dy, dyaw] increments into causal SE(2) prefixes."""
    if actions.shape[-2:] != (HORIZON, ACTION_DIM):
        raise ValueError(f"expected actions [...,6,3], got {tuple(actions.shape)}")
    x = torch.zeros_like(actions[..., 0, 0])
    y = torch.zeros_like(x)
    yaw = torch.zeros_like(x)
    poses = []
    for step in range(HORIZON):
        dx, dy, dyaw = actions[..., step, :].unbind(dim=-1)
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)
        x = x + cos_yaw * dx - sin_yaw * dy
        y = y + sin_yaw * dx + cos_yaw * dy
        yaw = yaw + dyaw
        poses.append(torch.stack((x, y, yaw), dim=-1))
    return torch.stack(poses, dim=-2)


def build_action_prefix_v2_features(actions: torch.Tensor) -> dict[str, torch.Tensor]:
    """Build separate increment and cumulative prefix-state feature streams."""
    if actions.shape[-2:] != (HORIZON, ACTION_DIM):
        raise ValueError(f"expected actions [...,6,3], got {tuple(actions.shape)}")
    previous = torch.cat((torch.zeros_like(actions[..., :1, :]), actions[..., :-1, :]), dim=-2)
    difference = actions - previous
    translation_norm = torch.linalg.vector_norm(actions[..., :2], dim=-1, keepdim=True)
    yaw_abs = actions[..., 2:3].abs()
    curvature = (actions[..., 2:3] / translation_norm.clamp_min(1e-3)).clamp(-20.0, 20.0)
    yaw_sign_change = ((actions[..., 2:3] * previous[..., 2:3]) < 0).to(actions.dtype)
    acceleration_proxy = torch.linalg.vector_norm(difference[..., :2], dim=-1, keepdim=True)
    adjacent_dot = (actions[..., :2] * previous[..., :2]).sum(dim=-1, keepdim=True)
    adjacent_cross = (
        previous[..., 0:1] * actions[..., 1:2] - previous[..., 1:2] * actions[..., 0:1]
    )
    yaw_translation_interaction = actions[..., 2:3] * translation_norm
    increment = torch.cat(
        (
            actions,
            difference,
            translation_norm,
            yaw_abs,
            curvature,
            yaw_sign_change,
            acceleration_proxy,
            adjacent_dot,
            adjacent_cross,
            yaw_translation_interaction,
        ),
        dim=-1,
    )

    cumulative = integrate_se2_prefix_v2(actions)
    path_length = torch.cumsum(translation_norm, dim=-2)
    cumulative_abs_yaw = torch.cumsum(yaw_abs, dim=-2)
    endpoint_displacement = torch.linalg.vector_norm(cumulative[..., :2], dim=-1, keepdim=True)
    remaining = torch.arange(HORIZON - 1, -1, -1, device=actions.device, dtype=actions.dtype)
    remaining = remaining.view(*([1] * (actions.ndim - 2)), HORIZON, 1)
    remaining = remaining.expand(*actions.shape[:-2], HORIZON, 1) / float(HORIZON - 1)
    prefix = torch.cat(
        (cumulative, path_length, cumulative_abs_yaw, endpoint_displacement, remaining), dim=-1
    )
    if increment.shape[-1] != INCREMENT_FEATURE_DIM or prefix.shape[-1] != PREFIX_FEATURE_DIM:
        raise RuntimeError("unexpected ActionPrefixEncoder-v2 feature width")
    return {
        "increment_features": increment,
        "prefix_features": prefix,
        "cumulative_pose": cumulative,
        "adjacent_action": difference,
    }


def _alibi_slopes(heads: int) -> torch.Tensor:
    """Return monotonically decreasing ALiBi slopes for a small head count."""
    if heads < 1:
        raise ValueError("heads must be positive")
    power = 2 ** math.floor(math.log2(heads))
    base = 2 ** (-(2 ** -(math.log2(power) - 3)))
    slopes = [base ** (index + 1) for index in range(power)]
    if power != heads:
        extra_base = 2 ** (-(2 ** -(math.log2(2 * power) - 3)))
        slopes.extend(extra_base ** (2 * index + 1) for index in range(heads - power))
    return torch.tensor(slopes, dtype=torch.float32)


class CausalALiBiAttention(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError("attention dim must divide heads")
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("slopes", _alibi_slopes(heads), persistent=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, horizon, dim = x.shape
        qkv = self.qkv(x).reshape(batch, horizon, 3, self.heads, self.head_dim)
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(dim=0)
        scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.head_dim)
        positions = torch.arange(horizon, device=x.device)
        distance = positions[:, None] - positions[None, :]
        causal = distance >= 0
        bias = -self.slopes.to(dtype=x.dtype)[:, None, None] * distance.clamp_min(0).to(x.dtype)
        scores = scores + bias[None]
        scores = scores.masked_fill(~causal[None, None], torch.finfo(scores.dtype).min)
        attention = self.dropout(torch.softmax(scores.float(), dim=-1).to(scores.dtype))
        output = torch.matmul(attention, value).transpose(1, 2).reshape(batch, horizon, dim)
        return self.out(output)


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.input = nn.Linear(dim, 2 * hidden_dim)
        self.output = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value, gate = self.input(x).chunk(2, dim=-1)
        return self.output(value * F.silu(gate))


class CausalActionBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: int = 2, dropout: float = 0.0) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(dim)
        self.attention = CausalALiBiAttention(dim, heads, dropout)
        self.mlp_norm = nn.LayerNorm(dim)
        self.mlp = SwiGLU(dim, mlp_ratio * dim)
        self.attention_scale = nn.Parameter(torch.tensor(0.1))
        self.mlp_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention_scale * self.attention(self.attention_norm(x))
        return x + self.mlp_scale * self.mlp(self.mlp_norm(x))


@dataclass(frozen=True)
class ActionPrefixV2Config:
    output_dim: int = 192
    hidden_dim: int = 128
    heads: int = 4
    layers: int = 2
    mlp_ratio: int = 2
    dropout: float = 0.0


class ActionPrefixEncoderV2(nn.Module):
    """Dual-stream, order-aware, exactly causal H6 action-prefix encoder."""

    def __init__(self, config: ActionPrefixV2Config | None = None) -> None:
        super().__init__()
        self.config = config or ActionPrefixV2Config()
        cfg = self.config
        self.increment_norm = nn.LayerNorm(INCREMENT_FEATURE_DIM)
        self.prefix_norm = nn.LayerNorm(PREFIX_FEATURE_DIM)
        self.increment_stream = nn.Sequential(
            nn.Linear(INCREMENT_FEATURE_DIM, cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
        )
        self.prefix_stream = nn.Sequential(
            nn.Linear(PREFIX_FEATURE_DIM, cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
        )
        self.horizon_embedding = nn.Parameter(torch.randn(HORIZON, cfg.hidden_dim) * 0.02)
        self.fusion_gate = nn.Linear(3 * cfg.hidden_dim, 3)
        nn.init.zeros_(self.fusion_gate.weight)
        nn.init.zeros_(self.fusion_gate.bias)
        self.blocks = nn.ModuleList(
            [
                CausalActionBlock(cfg.hidden_dim, cfg.heads, cfg.mlp_ratio, cfg.dropout)
                for _ in range(cfg.layers)
            ]
        )
        self.output_norm = nn.LayerNorm(cfg.hidden_dim)
        self.output_projection = nn.Linear(cfg.hidden_dim, cfg.output_dim)

    def forward_features(self, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        features = build_action_prefix_v2_features(actions)
        leading = actions.shape[:-2]
        increment = self.increment_stream(
            self.increment_norm(features["increment_features"]).reshape(-1, HORIZON, INCREMENT_FEATURE_DIM)
        )
        prefix = self.prefix_stream(
            self.prefix_norm(features["prefix_features"]).reshape(-1, HORIZON, PREFIX_FEATURE_DIM)
        )
        temporal = self.horizon_embedding[None].expand(increment.shape[0], -1, -1)
        fusion_weights = torch.softmax(self.fusion_gate(torch.cat((increment, prefix, temporal), dim=-1)), dim=-1)
        x = (
            fusion_weights[..., 0:1] * increment
            + fusion_weights[..., 1:2] * prefix
            + fusion_weights[..., 2:3] * temporal
        )
        for block in self.blocks:
            x = block(x)
        hidden = self.output_norm(x)
        output = self.output_projection(hidden)
        increment_output = self.output_projection(self.output_norm(increment))
        return {
            "action_prefix_tokens": output.reshape(*leading, HORIZON, self.config.output_dim),
            "action_increment_tokens": increment_output.reshape(
                *leading, HORIZON, self.config.output_dim
            ),
            "hidden_tokens": hidden.reshape(*leading, HORIZON, self.config.hidden_dim),
            "increment_tokens": increment.reshape(*leading, HORIZON, self.config.hidden_dim),
            "prefix_state_tokens": prefix.reshape(*leading, HORIZON, self.config.hidden_dim),
            "fusion_weights": fusion_weights.reshape(*leading, HORIZON, 3),
            **features,
        }

    def forward(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        output = self.forward_features(actions)
        return output["action_prefix_tokens"], output["cumulative_pose"]

    def step_embedding(self, action_prefix: torch.Tensor) -> torch.Tensor:
        """Encode one legal causal prefix without accepting future actions."""
        if action_prefix.ndim < 3 or action_prefix.shape[-1] != ACTION_DIM:
            raise ValueError(f"expected action prefix [...,h,3], got {tuple(action_prefix.shape)}")
        prefix_length = action_prefix.shape[-2]
        if not 1 <= prefix_length <= HORIZON:
            raise ValueError(f"prefix length must be in [1,{HORIZON}], got {prefix_length}")
        padded = F.pad(action_prefix, (0, 0, 0, HORIZON - prefix_length))
        return self.forward_features(padded)["action_prefix_tokens"][..., prefix_length - 1, :]


class ActionPrefixPretrainingModel(nn.Module):
    """Standalone auxiliary heads; only ``encoder`` transfers to visual dynamics."""

    def __init__(self, config: ActionPrefixV2Config | None = None) -> None:
        super().__init__()
        self.encoder = ActionPrefixEncoderV2(config)
        hidden_dim = self.encoder.config.hidden_dim
        self.action_head = nn.Linear(hidden_dim, ACTION_DIM)
        self.prefix_state_head = nn.Linear(hidden_dim, PREFIX_FEATURE_DIM)
        self.adjacent_action_head = nn.Linear(hidden_dim, ACTION_DIM)
        self.sequence_head = nn.Linear(hidden_dim, HORIZON * ACTION_DIM)

    def forward(self, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        output = self.encoder.forward_features(actions)
        hidden = output["hidden_tokens"]
        output.update(
            {
                "reconstructed_action": self.action_head(hidden),
                "reconstructed_prefix_state": self.prefix_state_head(hidden),
                "reconstructed_adjacent_action": self.adjacent_action_head(hidden),
                "reconstructed_sequence": self.sequence_head(hidden[..., -1, :]).reshape(
                    *actions.shape[:-2], HORIZON, ACTION_DIM
                ),
            }
        )
        return output


def endpoint_matched_order_negative(actions: torch.Tensor) -> torch.Tensor:
    """Choose the closest-endpoint non-identity order from bounded deterministic permutations."""
    if actions.ndim != 3 or actions.shape[-2:] != (HORIZON, ACTION_DIM):
        raise ValueError(f"expected [B,6,3], got {tuple(actions.shape)}")
    permutations = actions.new_tensor(
        [
            [1, 0, 2, 3, 4, 5],
            [0, 2, 1, 3, 4, 5],
            [0, 1, 3, 2, 4, 5],
            [0, 1, 2, 4, 3, 5],
            [0, 1, 2, 3, 5, 4],
            [2, 5, 0, 4, 1, 3],
            [5, 4, 3, 2, 1, 0],
        ],
        dtype=torch.long,
    )
    candidates = actions[:, permutations]
    batch, variants = candidates.shape[:2]
    target_endpoint = integrate_se2_prefix_v2(actions)[:, -1]
    candidate_endpoint = integrate_se2_prefix_v2(candidates.reshape(batch * variants, HORIZON, ACTION_DIM))
    candidate_endpoint = candidate_endpoint[:, -1].reshape(batch, variants, ACTION_DIM)
    delta = candidate_endpoint - target_endpoint[:, None]
    delta[..., 2] = torch.atan2(torch.sin(delta[..., 2]), torch.cos(delta[..., 2]))
    distance = torch.sqrt(delta[..., 0].square() + delta[..., 1].square() + 0.25 * delta[..., 2].square())
    selected = distance.argmin(dim=1)
    return candidates[torch.arange(batch, device=actions.device), selected]


def action_path_divergence(actions: torch.Tensor, counter: torch.Tensor) -> torch.Tensor:
    """Continuous physical divergence used to weight order negatives."""
    pose = integrate_se2_prefix_v2(actions)
    other = integrate_se2_prefix_v2(counter)
    delta = other - pose
    delta[..., 2] = torch.atan2(torch.sin(delta[..., 2]), torch.cos(delta[..., 2]))
    path = torch.sqrt(delta[..., 0].square() + delta[..., 1].square() + 0.25 * delta[..., 2].square())
    translation = torch.linalg.vector_norm(actions[..., :2], dim=-1).clamp_min(1e-3)
    other_translation = torch.linalg.vector_norm(counter[..., :2], dim=-1).clamp_min(1e-3)
    curvature = (actions[..., 2] / translation).clamp(-20.0, 20.0)
    other_curvature = (counter[..., 2] / other_translation).clamp(-20.0, 20.0)
    curvature_gap = torch.sqrt((curvature - other_curvature).square().mean(dim=-1))
    return path[:, -1] + path.mean(dim=-1) + 0.25 * curvature_gap


def count_parameters(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())
