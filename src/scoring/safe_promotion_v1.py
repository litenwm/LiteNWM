"""Relative, uncertainty-aware promotion gate for a frozen scorer ensemble.

This module adapts the useful part of the 019f70d5 RECON scorer protocol to
the DINOv3 scorer without importing any old model weights.  The absolute
scorers remain frozen.  Their paired seed-wise candidate/incumbent gains are
used to rank one challenger, and a disjoint calibration split supplies
one-sided conformal residual bounds.  A failed gate always falls back to the
incumbent (candidate zero).

The module deliberately has no CEM/search loop.  It only selects from an
already frozen candidate pool.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


EPS = 1.0e-12


@dataclass(frozen=True)
class RelativeGainScores:
    """Paired ensemble gains relative to one incumbent.

    All gains are positive when a candidate is predicted to be better.  The
    lower confidence bounds already include the configured epistemic penalty.
    """

    ate_gain_mean: np.ndarray
    ate_gain_std: np.ndarray
    rtp_gain_mean: np.ndarray
    rtp_gain_std: np.ndarray
    ate_gain_lcb: np.ndarray
    rtp_gain_lcb: np.ndarray


@dataclass(frozen=True)
class SafePromotionPolicy:
    """Frozen calibration state for fixed-pool promotion."""

    ate_scale: float
    rtp_scale: float
    uncertainty_beta: float
    alpha: float
    ate_residual_quantile: float
    rtp_residual_quantile: float
    calibration_states: int
    incumbent_index: int = 0
    schema: str = "dinov3_relative_safe_promotion_v1"
    cem_enabled: bool = False

    def validate(self) -> None:
        if self.schema != "dinov3_relative_safe_promotion_v1":
            raise ValueError("safe-promotion policy schema mismatch")
        if self.ate_scale <= 0 or self.rtp_scale <= 0:
            raise ValueError("metric scales must be positive")
        if self.uncertainty_beta < 0:
            raise ValueError("uncertainty_beta must be non-negative")
        if not 0 < self.alpha < 1:
            raise ValueError("alpha must be in (0,1)")
        if self.ate_residual_quantile < 0 or self.rtp_residual_quantile < 0:
            raise ValueError("conformal residual quantiles must be non-negative")
        if self.calibration_states <= 0:
            raise ValueError("calibration_states must be positive")
        if self.incumbent_index < 0:
            raise ValueError("incumbent_index must be non-negative")
        if self.cem_enabled:
            raise ValueError("safe-promotion v1 must not enable CEM")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SafePromotionPolicy":
        policy = cls(**value)
        policy.validate()
        return policy


@dataclass(frozen=True)
class SafeSelection:
    challenger_index: np.ndarray
    selected_index: np.ndarray
    promoted: np.ndarray
    conservative_utility: np.ndarray
    ate_gain_lcb: np.ndarray
    rtp_gain_lcb: np.ndarray
    ate_gate_margin: np.ndarray
    rtp_gate_margin: np.ndarray


def _prediction_arrays(
    ate_by_seed: np.ndarray,
    rtp_by_seed: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    ate = np.asarray(ate_by_seed, dtype=np.float64)
    rtp = np.asarray(rtp_by_seed, dtype=np.float64)
    if ate.ndim != 3 or rtp.shape != ate.shape:
        raise ValueError("expected matching ATE/RTP predictions [S,N,K]")
    if min(ate.shape) <= 0:
        raise ValueError("prediction dimensions must be nonempty")
    if not np.isfinite(ate).all() or not np.isfinite(rtp).all():
        raise ValueError("predictions contain NaN or Inf")
    return ate, rtp


def paired_relative_gains(
    ate_by_seed: np.ndarray,
    rtp_by_seed: np.ndarray,
    *,
    incumbent_index: int = 0,
    uncertainty_beta: float = 1.0,
) -> RelativeGainScores:
    """Return paired candidate/incumbent gains and conservative LCBs."""

    ate, rtp = _prediction_arrays(ate_by_seed, rtp_by_seed)
    if not 0 <= incumbent_index < ate.shape[2]:
        raise ValueError("incumbent_index is outside the candidate pool")
    if uncertainty_beta < 0:
        raise ValueError("uncertainty_beta must be non-negative")
    ate_gain = ate[:, :, incumbent_index, None] - ate
    rtp_gain = rtp - rtp[:, :, incumbent_index, None]
    ate_mean = ate_gain.mean(axis=0)
    rtp_mean = rtp_gain.mean(axis=0)
    ate_std = ate_gain.std(axis=0, ddof=0)
    rtp_std = rtp_gain.std(axis=0, ddof=0)
    return RelativeGainScores(
        ate_gain_mean=ate_mean,
        ate_gain_std=ate_std,
        rtp_gain_mean=rtp_mean,
        rtp_gain_std=rtp_std,
        ate_gain_lcb=ate_mean - uncertainty_beta * ate_std,
        rtp_gain_lcb=rtp_mean - uncertainty_beta * rtp_std,
    )


def conservative_utility(
    gains: RelativeGainScores,
    *,
    ate_scale: float,
    rtp_scale: float,
) -> np.ndarray:
    if ate_scale <= 0 or rtp_scale <= 0:
        raise ValueError("metric scales must be positive")
    return np.minimum(
        gains.ate_gain_lcb / ate_scale,
        gains.rtp_gain_lcb / rtp_scale,
    )


def _valid_mask(shape: tuple[int, int], valid: np.ndarray | None) -> np.ndarray:
    if valid is None:
        return np.ones(shape, dtype=bool)
    result = np.asarray(valid, dtype=bool)
    if result.shape != shape:
        raise ValueError(f"valid mask must have shape {shape}")
    return result.copy()


def select_challenger(
    gains: RelativeGainScores,
    *,
    ate_scale: float,
    rtp_scale: float,
    incumbent_index: int = 0,
    valid: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Choose one non-incumbent challenger without looking at outcomes."""

    utility = conservative_utility(gains, ate_scale=ate_scale, rtp_scale=rtp_scale)
    if utility.ndim != 2 or not 0 <= incumbent_index < utility.shape[1]:
        raise ValueError("relative utility/candidate contract is invalid")
    allowed = _valid_mask(utility.shape, valid) & np.isfinite(utility)
    allowed[:, incumbent_index] = False
    ranked = np.where(allowed, utility, -np.inf)
    challenger = ranked.argmax(axis=1).astype(np.int64)
    exists = allowed.any(axis=1)
    challenger = np.where(exists, challenger, incumbent_index)
    row = np.arange(len(challenger))
    selected_utility = utility[row, challenger]
    selected_utility = np.where(exists, selected_utility, -np.inf)
    return challenger, exists, selected_utility


def greedy_fixed_pool_selection(
    ate_by_seed: np.ndarray,
    rtp_by_seed: np.ndarray,
    *,
    ate_scale: float,
    rtp_scale: float,
    incumbent_index: int = 0,
    valid: np.ndarray | None = None,
) -> np.ndarray:
    """Reproduce the current scorer-v2 greedy fixed-pool readout.

    Candidate zero receives utility zero.  A challenger is selected only when
    its mean predicted ATE and RTP gains have positive joint utility.
    """

    gains = paired_relative_gains(
        ate_by_seed,
        rtp_by_seed,
        incumbent_index=incumbent_index,
        uncertainty_beta=0.0,
    )
    utility = conservative_utility(gains, ate_scale=ate_scale, rtp_scale=rtp_scale)
    allowed = _valid_mask(utility.shape, valid) & np.isfinite(utility)
    ranked = np.where(allowed, utility, -np.inf)
    ranked[:, incumbent_index] = 0.0
    return ranked.argmax(axis=1).astype(np.int64)


def conformal_upper_quantile(residual: np.ndarray, *, alpha: float) -> float:
    """Finite-sample split-conformal upper quantile.

    The order statistic is ``ceil((n+1)*(1-alpha))`` with the usual clipping
    at ``n``.  Negative residual quantiles are clamped to zero so calibration
    can never make an epistemic lower bound less conservative.
    """

    values = np.asarray(residual, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        raise ValueError("conformal residual set is empty")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0,1)")
    rank = min(len(values), math.ceil((len(values) + 1) * (1.0 - alpha)))
    quantile = float(np.sort(values)[rank - 1])
    return max(0.0, quantile)


def calibrate_safe_promotion(
    ate_by_seed: np.ndarray,
    rtp_by_seed: np.ndarray,
    actual_ate: np.ndarray,
    actual_rtp: np.ndarray,
    *,
    ate_scale: float,
    rtp_scale: float,
    uncertainty_beta: float = 1.0,
    alpha: float = 0.05,
    incumbent_index: int = 0,
    valid: np.ndarray | None = None,
) -> SafePromotionPolicy:
    """Fit only conformal residual bounds on a disjoint calibration split."""

    ate, rtp = _prediction_arrays(ate_by_seed, rtp_by_seed)
    actual_ate_value = np.asarray(actual_ate, dtype=np.float64)
    actual_rtp_value = np.asarray(actual_rtp, dtype=np.float64)
    if actual_ate_value.shape != ate.shape[1:] or actual_rtp_value.shape != ate.shape[1:]:
        raise ValueError("actual ATE/RTP must have shape [N,K]")
    if not np.isfinite(actual_ate_value).all() or not np.isfinite(actual_rtp_value).all():
        raise ValueError("calibration outcomes contain NaN or Inf")
    gains = paired_relative_gains(
        ate,
        rtp,
        incumbent_index=incumbent_index,
        uncertainty_beta=uncertainty_beta,
    )
    challenger, exists, _ = select_challenger(
        gains,
        ate_scale=ate_scale,
        rtp_scale=rtp_scale,
        incumbent_index=incumbent_index,
        valid=valid,
    )
    if not exists.all():
        raise ValueError("every calibration state must have a valid challenger")
    row = np.arange(len(challenger))
    actual_ate_gain = (
        actual_ate_value[:, incumbent_index] - actual_ate_value[row, challenger]
    )
    actual_rtp_gain = (
        actual_rtp_value[row, challenger] - actual_rtp_value[:, incumbent_index]
    )
    predicted_ate_lcb = gains.ate_gain_lcb[row, challenger]
    predicted_rtp_lcb = gains.rtp_gain_lcb[row, challenger]
    return SafePromotionPolicy(
        ate_scale=float(ate_scale),
        rtp_scale=float(rtp_scale),
        uncertainty_beta=float(uncertainty_beta),
        alpha=float(alpha),
        ate_residual_quantile=conformal_upper_quantile(
            predicted_ate_lcb - actual_ate_gain,
            alpha=alpha,
        ),
        rtp_residual_quantile=conformal_upper_quantile(
            predicted_rtp_lcb - actual_rtp_gain,
            alpha=alpha,
        ),
        calibration_states=int(len(challenger)),
        incumbent_index=int(incumbent_index),
    )


def apply_safe_promotion(
    ate_by_seed: np.ndarray,
    rtp_by_seed: np.ndarray,
    policy: SafePromotionPolicy,
    *,
    valid: np.ndarray | None = None,
) -> SafeSelection:
    """Apply a frozen safe gate and fall back to the incumbent on failure."""

    policy.validate()
    gains = paired_relative_gains(
        ate_by_seed,
        rtp_by_seed,
        incumbent_index=policy.incumbent_index,
        uncertainty_beta=policy.uncertainty_beta,
    )
    challenger, exists, utility = select_challenger(
        gains,
        ate_scale=policy.ate_scale,
        rtp_scale=policy.rtp_scale,
        incumbent_index=policy.incumbent_index,
        valid=valid,
    )
    row = np.arange(len(challenger))
    ate_lcb = gains.ate_gain_lcb[row, challenger]
    rtp_lcb = gains.rtp_gain_lcb[row, challenger]
    ate_margin = ate_lcb - policy.ate_residual_quantile
    rtp_margin = rtp_lcb - policy.rtp_residual_quantile
    promoted = exists & (ate_margin > 0.0) & (rtp_margin > 0.0)
    selected = np.where(promoted, challenger, policy.incumbent_index).astype(np.int64)
    return SafeSelection(
        challenger_index=challenger,
        selected_index=selected,
        promoted=promoted,
        conservative_utility=utility,
        ate_gain_lcb=ate_lcb,
        rtp_gain_lcb=rtp_lcb,
        ate_gate_margin=ate_margin,
        rtp_gate_margin=rtp_margin,
    )


def wilson_interval(successes: int, trials: int, *, z: float = 1.959963984540054) -> tuple[float, float]:
    if trials < 0 or successes < 0 or successes > trials:
        raise ValueError("invalid binomial counts")
    if trials == 0:
        return (0.0, 1.0)
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    center = (proportion + z * z / (2.0 * trials)) / denominator
    radius = z * math.sqrt(
        proportion * (1.0 - proportion) / trials + z * z / (4.0 * trials * trials)
    ) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def selection_metrics(
    selected_index: np.ndarray,
    ate: np.ndarray,
    rtp: np.ndarray,
    *,
    incumbent_index: int = 0,
) -> dict[str, Any]:
    """Evaluate one frozen selection under the current mean-ATE/RTP contract."""

    selected = np.asarray(selected_index, dtype=np.int64).reshape(-1)
    ate_value = np.asarray(ate, dtype=np.float64)
    rtp_value = np.asarray(rtp, dtype=np.float64)
    if ate_value.ndim != 2 or rtp_value.shape != ate_value.shape:
        raise ValueError("ATE/RTP outcomes must have matching shape [N,K]")
    if len(selected) != ate_value.shape[0]:
        raise ValueError("selection length differs from outcome states")
    if np.any(selected < 0) or np.any(selected >= ate_value.shape[1]):
        raise ValueError("selection contains an invalid candidate index")
    row = np.arange(len(selected))
    selected_ate = ate_value[row, selected]
    selected_rtp = rtp_value[row, selected]
    incumbent_ate = ate_value[:, incumbent_index]
    incumbent_rtp = rtp_value[:, incumbent_index]
    ate_delta = selected_ate - incumbent_ate
    rtp_delta = selected_rtp - incumbent_rtp
    promoted = selected != incumbent_index
    ate_success = promoted & (ate_delta < 0.0)
    rtp_success = promoted & (rtp_delta > 0.0)
    dual_success = ate_success & rtp_success
    promotions = int(promoted.sum())
    dual_count = int(dual_success.sum())
    wilson = wilson_interval(dual_count, promotions)

    def precision(mask: np.ndarray) -> float | None:
        return None if promotions == 0 else float(mask.sum() / promotions)

    return {
        "states": int(len(selected)),
        "promotions": promotions,
        "promotion_fraction": float(promotions / len(selected)),
        "selected_ate_mean_m": float(selected_ate.mean()),
        "incumbent_ate_mean_m": float(incumbent_ate.mean()),
        "net_ate_delta_per_state_m": float(ate_delta.mean()),
        "selected_rtp_mean": float(selected_rtp.mean()),
        "incumbent_rtp_mean": float(incumbent_rtp.mean()),
        "net_rtp_delta_per_state": float(rtp_delta.mean()),
        "ate_precision": precision(ate_success),
        "rtp_precision": precision(rtp_success),
        "dual_precision": precision(dual_success),
        "dual_precision_wilson95": [float(wilson[0]), float(wilson[1])],
        "ate_delta_by_state_m": ate_delta.tolist(),
        "rtp_delta_by_state": rtp_delta.tolist(),
    }


__all__ = [
    "RelativeGainScores",
    "SafePromotionPolicy",
    "SafeSelection",
    "apply_safe_promotion",
    "calibrate_safe_promotion",
    "conformal_upper_quantile",
    "conservative_utility",
    "greedy_fixed_pool_selection",
    "paired_relative_gains",
    "select_challenger",
    "selection_metrics",
    "wilson_interval",
]
