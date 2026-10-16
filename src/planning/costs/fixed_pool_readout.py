"""Deterministic closed-form ridge baseline for fixed-pool utility audits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


EPS = 1.0e-12


def _group_center(values: np.ndarray, group_ids: np.ndarray) -> np.ndarray:
    centered = np.empty_like(values, dtype=np.float64)
    for group in np.unique(group_ids):
        mask = group_ids == group
        centered[mask] = values[mask] - values[mask].mean(axis=0, keepdims=True)
    return centered


@dataclass(frozen=True)
class FixedPoolRidgeReadout:
    """Evaluation-only readout with fixed alpha and no hyperparameter search.

    The readout is fit on route-disjoint candidate-train features, then frozen.
    It is not connected to the world-model graph and is not a deployable scorer.
    """

    feature_names: tuple[str, ...]
    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray
    alpha: float

    @classmethod
    def fit(
        cls,
        features: np.ndarray,
        targets: np.ndarray,
        group_ids: Sequence[str],
        feature_names: Sequence[str],
        *,
        alpha: float = 1.0,
    ) -> "FixedPoolRidgeReadout":
        x = np.asarray(features, dtype=np.float64)
        y = np.asarray(targets, dtype=np.float64).reshape(-1)
        groups = np.asarray(group_ids, dtype=str)
        if x.ndim != 2 or len(x) != len(y) or len(y) != len(groups):
            raise ValueError("invalid fixed-pool readout training shapes")
        if x.shape[1] != len(feature_names):
            raise ValueError("feature_names do not match feature columns")
        if alpha <= 0 or not np.isfinite(alpha):
            raise ValueError("alpha must be finite and positive")
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("readout training data contain NaN or Inf")
        mean = x.mean(axis=0)
        scale = x.std(axis=0)
        scale[scale < EPS] = 1.0
        standardized = (x - mean) / scale
        centered_x = _group_center(standardized, groups)
        centered_y = _group_center(y[:, None], groups)[:, 0]
        regularizer = alpha * np.eye(centered_x.shape[1], dtype=np.float64)
        weights = np.linalg.solve(
            centered_x.T @ centered_x + regularizer,
            centered_x.T @ centered_y,
        )
        return cls(tuple(feature_names), mean, scale, weights, float(alpha))

    def predict(
        self,
        features: np.ndarray,
        group_ids: Sequence[str],
    ) -> np.ndarray:
        x = np.asarray(features, dtype=np.float64)
        groups = np.asarray(group_ids, dtype=str)
        if x.ndim != 2 or x.shape[1] != len(self.feature_names):
            raise ValueError("invalid fixed-pool readout feature shape")
        if len(x) != len(groups):
            raise ValueError("features and group_ids differ in length")
        standardized = (x - self.mean) / self.scale
        centered = _group_center(standardized, groups)
        prediction = centered @ self.weights
        if not np.isfinite(prediction).all():
            raise RuntimeError("fixed-pool readout produced NaN or Inf")
        return prediction

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "fixed_pool_closed_form_ridge_v1",
            "feature_names": list(self.feature_names),
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "weights": self.weights.tolist(),
            "alpha": self.alpha,
            "world_model_gradients": False,
            "hyperparameter_search": False,
            "deployment_accepted": False,
        }


__all__ = ["FixedPoolRidgeReadout"]
