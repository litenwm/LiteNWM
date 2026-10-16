"""Candidate-independent absolute ATE/RTP scorer for H4 trajectory planning.

The scorer intentionally has no candidate identity, family, route, dataset, or
optimizer-iteration inputs.  Every candidate is processed by the same MLP, so
changing K or permuting candidates cannot change another candidate's score.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

import torch
import torch.nn.functional as F

from scoring.features import (
    LATENT_CHANNELS,
    POOL_CHANNELS,
    action_trajectory_features,
    extract_scorer_features,
)


HORIZON = 4
ACTION_DIM = 3
OUTPUT_DIM = 2


@dataclass(frozen=True)
class TrajectoryQualityScorerConfig:
    """Architecture-only configuration.

    ``use_predicted_future=False`` is the matched M0 control.  It uses exactly
    the same current/goal/action information as M1 but cannot inspect a world
    model prediction.
    """

    hidden_dim: int = 640
    hidden_layers: int = 3
    dropout: float = 0.10
    use_predicted_future: bool = True

    def validate(self) -> None:
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if self.hidden_layers <= 0:
            raise ValueError("hidden_layers must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")


@dataclass(frozen=True)
class TrajectoryQualityPrediction:
    """Absolute scorer outputs; ATE is lower-better and RTP is higher-better."""

    ate: torch.Tensor
    rtp: torch.Tensor

    def stacked(self) -> torch.Tensor:
        return torch.stack((self.ate, self.rtp), dim=-1)


@dataclass(frozen=True)
class TrajectoryQualityEnsemble:
    ate_mean: torch.Tensor
    ate_std: torch.Tensor
    rtp_mean: torch.Tensor
    rtp_std: torch.Tensor


def _pool_current_goal(
    current_latent: torch.Tensor,
    goal_latent: torch.Tensor,
    candidate_count: int,
) -> torch.Tensor:
    if current_latent.ndim != 3 or goal_latent.shape != current_latent.shape:
        raise ValueError("expected matching current/goal latents [B,256,384]")
    if tuple(current_latent.shape[1:]) != (256, LATENT_CHANNELS):
        raise ValueError("expected current/goal latent shape [B,256,384]")
    if candidate_count <= 0:
        raise ValueError("candidate_count must be positive")
    width = LATENT_CHANNELS // POOL_CHANNELS
    current = current_latent.mean(dim=-2).reshape(
        current_latent.shape[0], POOL_CHANNELS, width
    ).mean(dim=-1)
    goal = goal_latent.mean(dim=-2).reshape(
        goal_latent.shape[0], POOL_CHANNELS, width
    ).mean(dim=-1)
    route = torch.cat((current, goal, current - goal), dim=-1)
    return route[:, None].expand(-1, candidate_count, -1)


class TrajectoryQualityScorerV2(torch.nn.Module):
    """Shared per-candidate MLP predicting absolute H4 ATE and RTP.

    The scorer does not perform attention, pooling, centering, or normalization
    across K.  Train-fitted feature normalization is stored as frozen buffers.
    """

    def __init__(
        self,
        config: TrajectoryQualityScorerConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config or TrajectoryQualityScorerConfig()
        self.config.validate()
        self.action_feature_dim = 40
        self.route_feature_dim = 3 * POOL_CHANNELS
        # M0 keeps a zeroed block of the same width, making it a capacity-
        # matched control without granting access to predicted futures.
        self.future_feature_dim = 3 * HORIZON * POOL_CHANNELS + 7 * HORIZON
        self.input_dim = (
            self.action_feature_dim
            + self.route_feature_dim
            + self.future_feature_dim
        )

        layers: list[torch.nn.Module] = []
        input_dim = self.input_dim
        for _ in range(self.config.hidden_layers):
            layers.extend(
                (
                    torch.nn.Linear(input_dim, self.config.hidden_dim),
                    torch.nn.LayerNorm(self.config.hidden_dim),
                    torch.nn.SiLU(),
                    torch.nn.Dropout(self.config.dropout),
                )
            )
            input_dim = self.config.hidden_dim
        self.backbone = torch.nn.Sequential(*layers)
        self.head = torch.nn.Linear(input_dim, OUTPUT_DIM)
        self.register_buffer("feature_mean", torch.zeros(self.input_dim))
        self.register_buffer("feature_scale", torch.ones(self.input_dim))
        self.register_buffer(
            "normalization_fitted", torch.tensor(False, dtype=torch.bool)
        )

    @staticmethod
    def _validate_actions(candidate_actions: torch.Tensor) -> None:
        if candidate_actions.ndim != 4:
            raise ValueError("expected candidate_actions [B,K,4,3]")
        if tuple(candidate_actions.shape[-2:]) != (HORIZON, ACTION_DIM):
            raise ValueError("scorer-v2 requires exactly H4 actions")
        if candidate_actions.shape[1] <= 0:
            raise ValueError("candidate count K must be positive")
        if not torch.isfinite(candidate_actions).all():
            raise ValueError("candidate_actions contains NaN or Inf")

    def extract_features(
        self,
        *,
        current_latent: torch.Tensor,
        goal_latent: torch.Tensor,
        candidate_actions: torch.Tensor,
        predicted_future: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return independent features shaped ``[B,K,F]``."""
        self._validate_actions(candidate_actions)
        batch, candidate_count = candidate_actions.shape[:2]
        if current_latent.shape[0] != batch or goal_latent.shape[0] != batch:
            raise ValueError("latent and candidate batch dimensions differ")
        route = _pool_current_goal(current_latent, goal_latent, candidate_count)
        action, _ = action_trajectory_features(candidate_actions)

        if self.config.use_predicted_future:
            if predicted_future is None:
                raise ValueError("M1 scorer requires predicted_future")
            expected = (batch, candidate_count, HORIZON, 256, LATENT_CHANNELS)
            if tuple(predicted_future.shape) != expected:
                raise ValueError(
                    f"expected predicted_future {expected}, got "
                    f"{tuple(predicted_future.shape)}"
                )
            safe = extract_scorer_features(
                predicted_future,
                goal_latent,
                current_latent,
                candidate_actions,
            )
            values = torch.cat((safe.trajectory, action, route), dim=-1)
        else:
            # Do not even inspect predicted_future in the matched no-future arm.
            no_future = action.new_zeros(
                batch,
                candidate_count,
                self.future_feature_dim,
            )
            values = torch.cat((no_future, action, route), dim=-1)

        if values.shape[-1] != self.input_dim:
            raise RuntimeError(
                f"feature dimension {values.shape[-1]} != expected {self.input_dim}"
            )
        if not torch.isfinite(values).all():
            raise RuntimeError("scorer-v2 features contain NaN or Inf")
        return values

    @torch.no_grad()
    def set_normalization(
        self,
        mean: torch.Tensor,
        scale: torch.Tensor,
    ) -> None:
        """Install normalization statistics fitted on the training split only."""
        mean_tensor = torch.as_tensor(
            mean, dtype=self.feature_mean.dtype, device=self.feature_mean.device
        )
        scale_tensor = torch.as_tensor(
            scale, dtype=self.feature_scale.dtype, device=self.feature_scale.device
        )
        if mean_tensor.shape != self.feature_mean.shape:
            raise ValueError("normalization mean has the wrong shape")
        if scale_tensor.shape != self.feature_scale.shape:
            raise ValueError("normalization scale has the wrong shape")
        if not torch.isfinite(mean_tensor).all():
            raise ValueError("normalization mean contains NaN or Inf")
        if not torch.isfinite(scale_tensor).all() or bool((scale_tensor <= 0).any()):
            raise ValueError("normalization scale must be finite and positive")
        self.feature_mean.copy_(mean_tensor)
        self.feature_scale.copy_(scale_tensor.clamp_min(1.0e-6))
        self.normalization_fitted.fill_(True)

    def forward(
        self,
        *,
        current_latent: torch.Tensor,
        goal_latent: torch.Tensor,
        candidate_actions: torch.Tensor,
        predicted_future: torch.Tensor | None = None,
    ) -> TrajectoryQualityPrediction:
        features = self.extract_features(
            current_latent=current_latent,
            goal_latent=goal_latent,
            candidate_actions=candidate_actions,
            predicted_future=predicted_future,
        )
        standardized = (features - self.feature_mean) / self.feature_scale
        raw = self.head(self.backbone(standardized))
        # ATE is a distance. RTP=(d_start-d_final)/d_start is upper-bounded by
        # one but may be negative when a trajectory moves away from the goal.
        ate = F.softplus(raw[..., 0])
        rtp = 1.0 - F.softplus(-raw[..., 1])
        if not torch.isfinite(ate).all() or not torch.isfinite(rtp).all():
            raise RuntimeError("scorer-v2 produced NaN or Inf")
        return TrajectoryQualityPrediction(ate=ate, rtp=rtp)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def artifact_metadata(self) -> dict[str, Any]:
        return {
            "kind": "candidate_independent_absolute_ate_rtp_scorer_v2",
            "config": asdict(self.config),
            "input_dim": self.input_dim,
            "parameter_count": self.parameter_count(),
            "output": {
                "ate": "absolute_nonnegative_lower_is_better",
                "rtp": "absolute_unclipped_upper_bounded_by_one_higher_is_better",
            },
            "candidate_identity_inputs": False,
            "cross_candidate_operations": False,
            "candidate0_anchor": False,
            "world_model_gradients": False,
            "normalization_fitted": bool(self.normalization_fitted.item()),
        }


def aggregate_scorer_ensemble(
    predictions: Sequence[TrajectoryQualityPrediction],
) -> TrajectoryQualityEnsemble:
    """Aggregate independently trained scorer seeds without changing ranking."""
    values = tuple(predictions)
    if not values:
        raise ValueError("at least one scorer prediction is required")
    reference_shape = values[0].ate.shape
    if any(
        value.ate.shape != reference_shape or value.rtp.shape != reference_shape
        for value in values
    ):
        raise ValueError("ensemble prediction shapes differ")
    ate = torch.stack([value.ate for value in values], dim=0)
    rtp = torch.stack([value.rtp for value in values], dim=0)
    return TrajectoryQualityEnsemble(
        ate_mean=ate.mean(dim=0),
        ate_std=ate.std(dim=0, unbiased=False),
        rtp_mean=rtp.mean(dim=0),
        rtp_std=rtp.std(dim=0, unbiased=False),
    )


__all__ = [
    "TrajectoryQualityEnsemble",
    "TrajectoryQualityPrediction",
    "TrajectoryQualityScorerConfig",
    "TrajectoryQualityScorerV2",
    "aggregate_scorer_ensemble",
]
