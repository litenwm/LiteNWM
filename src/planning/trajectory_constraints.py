"""Train-fitted action support and smoothness constraints for H4 search."""

from __future__ import annotations

from dataclasses import dataclass

import torch


HORIZON = 4
ACTION_DIM = 3
EPS = 1.0e-8


@dataclass(frozen=True)
class ActionSupportStats:
    """Frozen statistics fitted on the training split only."""

    lower: torch.Tensor
    upper: torch.Tensor
    mean: torch.Tensor | None = None
    covariance_inverse: torch.Tensor | None = None
    mahalanobis_squared_max: float | None = None
    support_actions: torch.Tensor | None = None
    knn_k: int | None = None
    knn_distance_max: float | None = None
    roughness_max: float | None = None
    jerk_max: float | None = None
    curvature_max: float | None = None
    parent_trust_radius: float | None = None
    tail_support_actions: torch.Tensor | None = None
    tail_normalization_lower: torch.Tensor | None = None
    tail_normalization_upper: torch.Tensor | None = None
    tail_knn_distance_max: float | None = None
    tail_min_forward_m: float = 0.0
    tail_min_step_length_m: float = 0.0
    tail_max_step_length_m: float | None = None
    tail_max_abs_dyaw_rad: float | None = None
    tail_max_abs_yaw_sum_rad: float | None = None
    tail_yaw_sign_tolerance_rad: float = 0.0
    duplicate_tolerance: float = 1.0e-6

    def validate(self) -> None:
        lower = torch.as_tensor(self.lower)
        upper = torch.as_tensor(self.upper)
        if lower.shape not in ((ACTION_DIM,), (HORIZON, ACTION_DIM)):
            raise ValueError("lower must have shape [3] or [4,3]")
        if upper.shape != lower.shape:
            raise ValueError("upper shape differs from lower")
        if not torch.isfinite(lower).all() or not torch.isfinite(upper).all():
            raise ValueError("action bounds contain NaN or Inf")
        if bool((upper <= lower).any()):
            raise ValueError("every upper action bound must exceed lower")
        if (self.mean is None) != (self.covariance_inverse is None):
            raise ValueError("mean and covariance_inverse must be supplied together")
        if self.mean is not None:
            mean = torch.as_tensor(self.mean)
            covariance_inverse = torch.as_tensor(self.covariance_inverse)
            if mean.shape != (HORIZON * ACTION_DIM,):
                raise ValueError("support mean must have shape [12]")
            if covariance_inverse.shape != (
                HORIZON * ACTION_DIM,
                HORIZON * ACTION_DIM,
            ):
                raise ValueError("covariance_inverse must have shape [12,12]")
            if not torch.isfinite(mean).all() or not torch.isfinite(
                covariance_inverse
            ).all():
                raise ValueError("support statistics contain NaN or Inf")
            if self.mahalanobis_squared_max is None:
                raise ValueError("Mahalanobis threshold is required with covariance")
        if self.support_actions is not None:
            support_actions = torch.as_tensor(self.support_actions)
            if support_actions.ndim != 2 or support_actions.shape[1] != (
                HORIZON * ACTION_DIM
            ):
                raise ValueError("support_actions must have shape [N,12]")
            if support_actions.shape[0] < 1 or not torch.isfinite(
                support_actions
            ).all():
                raise ValueError("support_actions is empty or non-finite")
            if self.knn_k is None or self.knn_distance_max is None:
                raise ValueError("KNN k/distance are required with support_actions")
            if not 0 < self.knn_k <= support_actions.shape[0]:
                raise ValueError("knn_k is invalid")
            if self.knn_distance_max <= 0:
                raise ValueError("knn_distance_max must be positive")
        elif self.knn_k is not None or self.knn_distance_max is not None:
            raise ValueError("KNN settings require support_actions")
        tail_items = (
            self.tail_support_actions,
            self.tail_normalization_lower,
            self.tail_normalization_upper,
            self.tail_knn_distance_max,
            self.tail_max_step_length_m,
            self.tail_max_abs_dyaw_rad,
            self.tail_max_abs_yaw_sum_rad,
        )
        if any(item is not None for item in tail_items):
            if any(item is None for item in tail_items):
                raise ValueError(
                    "tail support requires templates, normalization bounds, "
                    "distance, step-length, and yaw limits"
                )
            tail_support_actions = torch.as_tensor(self.tail_support_actions)
            if (
                tail_support_actions.ndim != 2
                or tail_support_actions.shape[1] != HORIZON * ACTION_DIM
                or tail_support_actions.shape[0] < 2
                or not torch.isfinite(tail_support_actions).all()
            ):
                raise ValueError(
                    "tail_support_actions must be finite [N,12] with N >= 2"
                )
            tail_lower = torch.as_tensor(self.tail_normalization_lower)
            tail_upper = torch.as_tensor(self.tail_normalization_upper)
            if tail_lower.shape not in ((ACTION_DIM,), (HORIZON, ACTION_DIM)):
                raise ValueError(
                    "tail_normalization_lower must have shape [3] or [4,3]"
                )
            if tail_upper.shape != tail_lower.shape:
                raise ValueError("tail normalization bound shapes differ")
            if (
                not torch.isfinite(tail_lower).all()
                or not torch.isfinite(tail_upper).all()
                or bool((tail_upper <= tail_lower).any())
            ):
                raise ValueError("tail normalization bounds are invalid")
            for name in (
                "tail_knn_distance_max",
                "tail_max_step_length_m",
                "tail_max_abs_dyaw_rad",
                "tail_max_abs_yaw_sum_rad",
            ):
                if float(getattr(self, name)) <= 0:
                    raise ValueError(f"{name} must be positive")
        if self.tail_min_forward_m < 0:
            raise ValueError("tail_min_forward_m must be non-negative")
        if self.tail_min_step_length_m < 0:
            raise ValueError("tail_min_step_length_m must be non-negative")
        if self.tail_yaw_sign_tolerance_rad < 0:
            raise ValueError(
                "tail_yaw_sign_tolerance_rad must be non-negative"
            )
        for name in (
            "mahalanobis_squared_max",
            "roughness_max",
            "jerk_max",
            "curvature_max",
            "parent_trust_radius",
        ):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive when set")
        if self.duplicate_tolerance < 0:
            raise ValueError("duplicate_tolerance must be non-negative")


@dataclass(frozen=True)
class TrajectoryConstraintResult:
    valid: torch.Tensor
    finite: torch.Tensor
    in_bounds: torch.Tensor
    support: torch.Tensor
    parent_trust: torch.Tensor
    smooth: torch.Tensor
    unique: torch.Tensor
    tail_support: torch.Tensor
    tail_physical: torch.Tensor
    tail_same_turn: torch.Tensor
    roughness: torch.Tensor
    jerk: torch.Tensor
    curvature: torch.Tensor
    mahalanobis_squared: torch.Tensor
    nearest_parent_distance: torch.Tensor
    knn_distance: torch.Tensor
    tail_knn_distance: torch.Tensor


def _expanded_bounds(
    stats: ActionSupportStats,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    lower = torch.as_tensor(stats.lower, device=device, dtype=dtype)
    upper = torch.as_tensor(stats.upper, device=device, dtype=dtype)
    if lower.shape == (ACTION_DIM,):
        lower = lower[None].expand(HORIZON, -1)
        upper = upper[None].expand(HORIZON, -1)
    return lower, upper


def evaluate_trajectory_constraints(
    candidate_actions: torch.Tensor,
    stats: ActionSupportStats,
    *,
    parent_actions: torch.Tensor | None = None,
) -> TrajectoryConstraintResult:
    """Evaluate constraints using no pose, outcome, or candidate identity fields."""
    stats.validate()
    if candidate_actions.ndim != 4 or tuple(candidate_actions.shape[-2:]) != (
        HORIZON,
        ACTION_DIM,
    ):
        raise ValueError("expected candidate_actions [B,K,4,3]")
    if candidate_actions.shape[1] <= 0:
        raise ValueError("candidate count K must be positive")
    batch, candidate_count = candidate_actions.shape[:2]
    lower, upper = _expanded_bounds(
        stats,
        device=candidate_actions.device,
        dtype=candidate_actions.dtype,
    )
    finite = torch.isfinite(candidate_actions).all(dim=(-1, -2))
    safe_actions = torch.nan_to_num(candidate_actions)
    in_bounds = (
        (safe_actions >= lower) & (safe_actions <= upper)
    ).all(dim=(-1, -2)) & finite

    first = safe_actions[..., 1:, :] - safe_actions[..., :-1, :]
    second = first[..., 1:, :] - first[..., :-1, :]
    roughness = first.square().sum(dim=-1).mean(dim=-1)
    jerk = second.square().sum(dim=-1).mean(dim=-1)
    path_length = torch.linalg.vector_norm(
        safe_actions[..., :2], dim=-1
    ).sum(dim=-1)
    curvature = safe_actions[..., 2].abs().sum(dim=-1) / path_length.clamp_min(EPS)
    smooth = finite.clone()
    if stats.roughness_max is not None:
        smooth &= roughness <= stats.roughness_max
    if stats.jerk_max is not None:
        smooth &= jerk <= stats.jerk_max
    if stats.curvature_max is not None:
        smooth &= curvature <= stats.curvature_max

    mahalanobis_squared = torch.zeros(
        batch,
        candidate_count,
        device=candidate_actions.device,
        dtype=candidate_actions.dtype,
    )
    support = finite.clone()
    if stats.mean is not None:
        flat = safe_actions.flatten(-2)
        mean = torch.as_tensor(
            stats.mean,
            device=candidate_actions.device,
            dtype=candidate_actions.dtype,
        )
        covariance_inverse = torch.as_tensor(
            stats.covariance_inverse,
            device=candidate_actions.device,
            dtype=candidate_actions.dtype,
        )
        centered = flat - mean
        mahalanobis_squared = torch.einsum(
            "bki,ij,bkj->bk",
            centered,
            covariance_inverse,
            centered,
        )
        support &= mahalanobis_squared <= float(stats.mahalanobis_squared_max)

    knn_distance = torch.zeros_like(mahalanobis_squared)
    if stats.support_actions is not None:
        scale = (upper - lower).flatten().clamp_min(EPS)
        normalized_candidate = safe_actions.flatten(-2) / scale
        normalized_support = torch.as_tensor(
            stats.support_actions,
            device=candidate_actions.device,
            dtype=candidate_actions.dtype,
        ) / scale
        neighbor_distance = torch.cdist(
            normalized_candidate,
            normalized_support[None].expand(batch, -1, -1),
        ) / (HORIZON * ACTION_DIM) ** 0.5
        knn_distance = torch.topk(
            neighbor_distance,
            k=int(stats.knn_k),
            dim=-1,
            largest=False,
        ).values[..., -1]
        support &= knn_distance <= float(stats.knn_distance_max)

    tail_knn_distance = torch.zeros_like(mahalanobis_squared)
    tail_physical = torch.zeros_like(finite)
    tail_same_turn = torch.zeros_like(finite)
    tail_support = torch.zeros_like(finite)
    if stats.tail_support_actions is not None:
        tail_lower = torch.as_tensor(
            stats.tail_normalization_lower,
            device=candidate_actions.device,
            dtype=candidate_actions.dtype,
        )
        tail_upper = torch.as_tensor(
            stats.tail_normalization_upper,
            device=candidate_actions.device,
            dtype=candidate_actions.dtype,
        )
        if tail_lower.shape == (ACTION_DIM,):
            tail_lower = tail_lower[None].expand(HORIZON, -1)
            tail_upper = tail_upper[None].expand(HORIZON, -1)
        tail_scale = (tail_upper - tail_lower).flatten().clamp_min(EPS)
        normalized_candidate = safe_actions.flatten(-2) / tail_scale
        normalized_tail = torch.as_tensor(
            stats.tail_support_actions,
            device=candidate_actions.device,
            dtype=candidate_actions.dtype,
        ) / tail_scale
        tail_knn_distance = torch.cdist(
            normalized_candidate,
            normalized_tail[None].expand(batch, -1, -1),
        ).amin(dim=-1) / (HORIZON * ACTION_DIM) ** 0.5

        translation = torch.linalg.vector_norm(
            safe_actions[..., :2], dim=-1
        )
        yaw = safe_actions[..., 2]
        yaw_tolerance = float(stats.tail_yaw_sign_tolerance_rad)
        tail_same_turn = (yaw >= -yaw_tolerance).all(dim=-1) | (
            yaw <= yaw_tolerance
        ).all(dim=-1)
        tail_physical = (
            finite
            & (safe_actions[..., 0] >= float(stats.tail_min_forward_m)).all(
                dim=-1
            )
            & (translation >= float(stats.tail_min_step_length_m)).all(dim=-1)
            & (translation <= float(stats.tail_max_step_length_m)).all(dim=-1)
            & (yaw.abs() <= float(stats.tail_max_abs_dyaw_rad)).all(dim=-1)
            & (
                yaw.abs().sum(dim=-1)
                <= float(stats.tail_max_abs_yaw_sum_rad)
            )
            & tail_same_turn
        )
        tail_support = (
            ~in_bounds
            & tail_physical
            & (tail_knn_distance <= float(stats.tail_knn_distance_max))
        )

    nearest_parent_distance = torch.zeros_like(mahalanobis_squared)
    parent_trust = finite.clone()
    if stats.parent_trust_radius is not None:
        if parent_actions is None:
            raise ValueError("parent_actions required by parent_trust_radius")
        if (
            parent_actions.ndim != 4
            or parent_actions.shape[0] != batch
            or tuple(parent_actions.shape[-2:]) != (HORIZON, ACTION_DIM)
        ):
            raise ValueError("expected parent_actions [B,P,4,3]")
        if not torch.isfinite(parent_actions).all():
            raise ValueError("parent_actions contains NaN or Inf")
        scale = (upper - lower).flatten().clamp_min(EPS)
        normalized_candidate = safe_actions.flatten(-2) / scale
        normalized_parent = parent_actions.flatten(-2) / scale
        nearest_parent_distance = torch.cdist(
            normalized_candidate,
            normalized_parent,
        ).amin(dim=-1) / (HORIZON * ACTION_DIM) ** 0.5
        parent_trust &= nearest_parent_distance <= stats.parent_trust_radius

    unique = torch.ones(
        batch,
        candidate_count,
        dtype=torch.bool,
        device=candidate_actions.device,
    )
    if candidate_count > 1 and stats.duplicate_tolerance > 0:
        distance = torch.cdist(
            safe_actions.flatten(-2),
            safe_actions.flatten(-2),
        )
        earlier = torch.tril(
            torch.ones(
                candidate_count,
                candidate_count,
                dtype=torch.bool,
                device=candidate_actions.device,
            ),
            diagonal=-1,
        )
        # Row i is a duplicate if it matches any earlier row j < i.
        unique &= ~(
            (distance <= stats.duplicate_tolerance) & earlier[None]
        ).any(dim=-1)

    distribution_support = (in_bounds & support) | tail_support
    valid = finite & distribution_support & parent_trust & smooth & unique
    return TrajectoryConstraintResult(
        valid=valid,
        finite=finite,
        in_bounds=in_bounds,
        support=support,
        parent_trust=parent_trust,
        smooth=smooth,
        unique=unique,
        roughness=roughness,
        jerk=jerk,
        tail_support=tail_support,
        tail_physical=tail_physical,
        tail_same_turn=tail_same_turn,
        curvature=curvature,
        mahalanobis_squared=mahalanobis_squared,
        nearest_parent_distance=nearest_parent_distance,
        knn_distance=knn_distance,
        tail_knn_distance=tail_knn_distance,
    )


__all__ = [
    "ActionSupportStats",
    "TrajectoryConstraintResult",
    "evaluate_trajectory_constraints",
]
