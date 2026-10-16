"""Spacing-aware action support for RECON/SCAND fixed K17 pools.

The envelope is fitted without outcome labels.  Translational action channels
are expressed in waypoint-spacing units while yaw remains in radians, so the
same support statistic has a physical interpretation for RECON (0.25 m) and
SCAND (0.38 m).  Candidate zero is always retained as the safe fallback.
"""

from __future__ import annotations

from typing import Any

import numpy as np


EPS = 1.0e-8


def normalized_action_vector(actions_h4: np.ndarray, spacing_m: np.ndarray) -> np.ndarray:
    """Return flattened ``[dx/spacing, dy/spacing, dyaw]`` vectors."""

    actions = np.asarray(actions_h4, dtype=np.float64)
    spacing = np.asarray(spacing_m, dtype=np.float64).reshape(-1)
    if actions.ndim != 4 or actions.shape[-2:] != (4, 3):
        raise ValueError("actions_h4 must have shape [N,K,4,3]")
    if spacing.shape != (len(actions),) or not np.isfinite(spacing).all():
        raise ValueError("spacing_m must contain one finite value per state")
    if np.any(spacing <= 0.0) or not np.isfinite(actions).all():
        raise ValueError("actions and spacing must be finite and spacing positive")
    value = actions.copy()
    value[..., 0] /= spacing[:, None, None]
    value[..., 1] /= spacing[:, None, None]
    return value.reshape(actions.shape[0], actions.shape[1], 12)


def _ledoit_wolf_covariance(value: np.ndarray) -> tuple[np.ndarray, float]:
    """Small dense implementation of Ledoit-Wolf shrinkage to the identity."""

    x = np.asarray(value, dtype=np.float64)
    if x.ndim != 2 or len(x) < 2:
        raise ValueError("Ledoit-Wolf fitting needs at least two samples")
    x = x - x.mean(axis=0, keepdims=True)
    samples, features = x.shape
    empirical = x.T @ x / samples
    mu = float(np.trace(empirical) / features)
    x2 = x * x
    beta_raw = float(np.sum(x2.T @ x2) / samples - np.sum(empirical * empirical))
    beta = beta_raw / (features * samples)
    delta_matrix = empirical - mu * np.eye(features)
    delta = float(np.sum(delta_matrix * delta_matrix) / features)
    beta = min(max(beta, 0.0), max(delta, 0.0))
    shrinkage = 0.0 if delta <= EPS else beta / delta
    covariance = (1.0 - shrinkage) * empirical
    covariance.flat[:: features + 1] += shrinkage * mu
    covariance.flat[:: features + 1] += 1.0e-6
    return covariance, float(shrinkage)


def fit_spacing_envelope(
    actions_h4: np.ndarray,
    spacing_m: np.ndarray,
    *,
    physical_guard_quantile: float = 0.999,
) -> dict[str, Any]:
    """Fit the label-free support envelope on non-incumbent candidates."""

    if not 0.5 < physical_guard_quantile < 1.0:
        raise ValueError("physical_guard_quantile must be in (0.5,1)")
    vector = normalized_action_vector(actions_h4, spacing_m)
    sample = vector[:, 1:].reshape(-1, vector.shape[-1])
    mean = sample.mean(axis=0)
    covariance, shrinkage = _ledoit_wolf_covariance(sample)
    inverse = np.linalg.pinv(covariance, hermitian=True)
    robust_center = np.median(sample, axis=0)
    robust_scale = 1.4826 * np.median(np.abs(sample - robust_center), axis=0)
    fallback_scale = np.maximum(sample.std(axis=0), 1.0e-3)
    robust_scale = np.where(robust_scale > 1.0e-3, robust_scale, fallback_scale)
    coordinate_limit = np.quantile(
        np.abs(sample), physical_guard_quantile, axis=0, method="higher"
    )
    model: dict[str, Any] = {
        "schema": "domain_aware_spacing_normalized_envelope_v1",
        "mean": mean.astype(np.float32),
        "covariance": covariance.astype(np.float32),
        "covariance_inverse": inverse.astype(np.float32),
        "ledoit_wolf_shrinkage": float(shrinkage),
        "robust_center": robust_center.astype(np.float32),
        "robust_scale": robust_scale.astype(np.float32),
        "coordinate_abs_limit": coordinate_limit.astype(np.float32),
        "physical_guard_quantile": float(physical_guard_quantile),
        "fit_states": int(vector.shape[0]),
        "fit_candidates": int(len(sample)),
        "feature_dim": int(sample.shape[1]),
        "candidate_zero_always_supported": True,
    }
    score, guard = score_spacing_envelope(model, actions_h4, spacing_m)
    train_score = score[:, 1:][guard[:, 1:]]
    if not len(train_score):
        raise RuntimeError("support fit produced no physically valid candidates")
    model["fit_nonconformity"] = train_score.astype(np.float32)
    model["fit_physical_guard_acceptance"] = float(guard[:, 1:].mean())
    return model


def score_spacing_envelope(
    model: dict[str, Any], actions_h4: np.ndarray, spacing_m: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return nonconformity scores and the spacing-aware coordinate guard."""

    if model.get("schema") != "domain_aware_spacing_normalized_envelope_v1":
        raise ValueError("spacing support schema mismatch")
    vector = normalized_action_vector(actions_h4, spacing_m)
    centered = vector - np.asarray(model["mean"], dtype=np.float64)
    inverse = np.asarray(model["covariance_inverse"], dtype=np.float64)
    mahalanobis_sq = np.einsum("nkd,df,nkf->nk", centered, inverse, centered)
    mahalanobis = np.sqrt(np.maximum(mahalanobis_sq, 0.0) / vector.shape[-1])
    robust_z = np.abs(
        (vector - np.asarray(model["robust_center"], dtype=np.float64))
        / np.asarray(model["robust_scale"], dtype=np.float64)
    ).max(axis=-1)
    score = np.maximum(mahalanobis, robust_z / 3.0)
    limit = np.asarray(model["coordinate_abs_limit"], dtype=np.float64)
    guard = (np.abs(vector) <= limit).all(axis=-1)
    score[:, 0] = 0.0
    guard[:, 0] = True
    if not np.isfinite(score).all():
        raise RuntimeError("spacing support score contains NaN or Inf")
    return score.astype(np.float32), guard


def threshold_for_quantile(model: dict[str, Any], quantile: float) -> float:
    if not 0.0 < quantile < 1.0:
        raise ValueError("support quantile must be in (0,1)")
    values = np.asarray(model["fit_nonconformity"], dtype=np.float64)
    return float(np.quantile(values, quantile, method="higher"))


def support_mask(
    model: dict[str, Any],
    actions_h4: np.ndarray,
    spacing_m: np.ndarray,
    *,
    quantile: float,
) -> tuple[np.ndarray, dict[str, float | int]]:
    score, guard = score_spacing_envelope(model, actions_h4, spacing_m)
    threshold = threshold_for_quantile(model, quantile)
    valid = guard & (score <= threshold)
    valid[:, 0] = True
    proposals = valid[:, 1:]
    audit: dict[str, float | int] = {
        "quantile": float(quantile),
        "threshold": float(threshold),
        "candidate_acceptance": float(proposals.mean()),
        "window_any_proposal_acceptance": float(proposals.any(axis=1).mean()),
        "no_supported_proposal_windows": int((~proposals.any(axis=1)).sum()),
        "physical_guard_acceptance": float(guard[:, 1:].mean()),
    }
    return valid, audit


def serializable_envelope(model: dict[str, Any]) -> dict[str, Any]:
    return {
        key: (value.tolist() if isinstance(value, np.ndarray) else value)
        for key, value in model.items()
        if key != "fit_nonconformity"
    }


__all__ = [
    "fit_spacing_envelope",
    "normalized_action_vector",
    "score_spacing_envelope",
    "serializable_envelope",
    "support_mask",
    "threshold_for_quantile",
]
