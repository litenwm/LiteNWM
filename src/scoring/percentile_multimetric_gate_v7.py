"""Percentile-normalized, current-window multi-metric promotion gate."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class PercentileMultimetricPolicyV7:
    harm_rank_weight: float
    uncertainty_rank_weight: float
    rpe_rank_weight: float
    endpoint_rank_weight: float
    z: float
    margin_ate: float
    margin_objective: float
    rpe_ucb_cap: float
    endpoint_ucb_cap: float
    dual_quantile_threshold: float
    harm_quantile_threshold: float
    ood_quantile_threshold: float = 0.975
    disagreement_quantile_threshold: float = 0.90


def empirical_cdf(
    value: np.ndarray, sorted_reference: np.ndarray
) -> np.ndarray:
    """Map current values to a frozen calibration-reference percentile."""

    reference = np.asarray(sorted_reference, dtype=np.float32).reshape(-1)
    if not len(reference) or not np.isfinite(reference).all():
        raise ValueError("empirical CDF reference is empty or non-finite")
    if np.any(reference[1:] < reference[:-1]):
        raise ValueError("empirical CDF reference must be sorted")
    observed = np.asarray(value, dtype=np.float32)
    result = np.searchsorted(reference, observed, side="right") / len(reference)
    return result.astype(np.float32)


def select_percentile_multimetric(
    prediction: dict[str, np.ndarray],
    q_dual: np.ndarray,
    q_harm: np.ndarray,
    q_ood: np.ndarray,
    q_disagreement: np.ndarray,
    valid: np.ndarray,
    policy: PercentileMultimetricPolicyV7 | dict[str, float],
) -> np.ndarray:
    """Select one K17 challenger with no route-future or label input."""

    if isinstance(policy, dict):
        policy = PercentileMultimetricPolicyV7(**policy)
    allowed = np.asarray(valid, dtype=bool).copy()
    allowed[:, 0] = False
    mean = np.asarray(prediction["mean"], dtype=np.float32)
    std = np.asarray(prediction["std"], dtype=np.float32)
    if mean.shape[-1] != 4 or std.shape != mean.shape:
        raise ValueError("multi-metric prediction shape changed")
    ucb = mean + policy.z * std
    uncertainty = std[..., 0] + std[..., 3]
    rank = (
        np.asarray(q_dual, dtype=np.float32)
        - policy.harm_rank_weight * np.asarray(q_harm, dtype=np.float32)
        - policy.uncertainty_rank_weight * uncertainty
        - policy.rpe_rank_weight * np.maximum(ucb[..., 1], 0.0)
        - policy.endpoint_rank_weight * np.maximum(ucb[..., 2], 0.0)
    )
    challenger = np.where(allowed, rank, -np.inf).argmax(axis=1)
    exists = allowed.any(axis=1)
    challenger = np.where(exists, challenger, 0).astype(np.int64)
    row = np.arange(len(challenger))
    promote = (
        exists
        & (
            q_dual[row, challenger]
            >= policy.dual_quantile_threshold
        )
        & (
            q_harm[row, challenger]
            <= policy.harm_quantile_threshold
        )
        & (q_ood[row, challenger] <= policy.ood_quantile_threshold)
        & (
            q_disagreement[row, challenger]
            <= policy.disagreement_quantile_threshold
        )
        & (ucb[row, challenger, 0] < -policy.margin_ate)
        & (ucb[row, challenger, 3] < -policy.margin_objective)
        & (ucb[row, challenger, 1] <= policy.rpe_ucb_cap)
        & (ucb[row, challenger, 2] <= policy.endpoint_ucb_cap)
    )
    return np.where(promote, challenger, 0).astype(np.int64)


def policy_dict(
    policy: PercentileMultimetricPolicyV7,
) -> dict[str, float]:
    return {key: float(value) for key, value in asdict(policy).items()}


__all__ = [
    "PercentileMultimetricPolicyV7",
    "empirical_cdf",
    "policy_dict",
    "select_percentile_multimetric",
]
