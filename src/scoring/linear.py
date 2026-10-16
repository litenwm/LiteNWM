"""Closed-form shared linear scorer with train-only group centering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


EPS = 1.0e-12


def group_center(values: np.ndarray, group_ids: np.ndarray) -> np.ndarray:
    centered = np.empty_like(values, dtype=np.float64)
    for group in np.unique(group_ids):
        mask = group_ids == group
        centered[mask] = values[mask] - values[mask].mean(axis=0, keepdims=True)
    return centered


def _ridge_weights_path(
    features: np.ndarray,
    targets: np.ndarray,
    alphas: Sequence[float],
) -> dict[float, np.ndarray]:
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    if x.shape[1] <= x.shape[0]:
        gram = x.T @ x
        eigenvalues, eigenvectors = np.linalg.eigh(gram)
        projected = eigenvectors.T @ (x.T @ y)
        return {
            float(alpha): eigenvectors
            @ (projected / (eigenvalues + float(alpha)))
            for alpha in alphas
        }
    gram = x @ x.T
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    projected = eigenvectors.T @ y
    return {
        float(alpha): x.T
        @ (
            eigenvectors
            @ (projected / (eigenvalues + float(alpha)))
        )
        for alpha in alphas
    }


@dataclass(frozen=True)
class SharedLinearScorer:
    feature_names: tuple[str, ...]
    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray
    alpha: float
    supervision: str = "SE2_GEOMETRY_PROXY"

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SharedLinearScorer":
        return cls(
            tuple(payload["feature_names"]),
            np.asarray(payload["mean"], dtype=np.float64),
            np.asarray(payload["scale"], dtype=np.float64),
            np.asarray(payload["weights"], dtype=np.float64),
            float(payload["alpha"]),
            str(payload.get("supervision", "SE2_GEOMETRY_PROXY")),
        )

    @classmethod
    def fit_path(
        cls,
        features: np.ndarray,
        targets: np.ndarray,
        group_ids: Sequence[str],
        feature_names: Sequence[str],
        alphas: Sequence[float],
    ) -> dict[float, "SharedLinearScorer"]:
        x = np.asarray(features, dtype=np.float64)
        y = np.asarray(targets, dtype=np.float64).reshape(-1)
        groups = np.asarray(group_ids, dtype=str)
        if x.ndim != 2 or len(x) != len(y) or len(y) != len(groups):
            raise ValueError("invalid linear scorer training shapes")
        if x.shape[1] != len(feature_names):
            raise ValueError("feature names do not match feature columns")
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("linear scorer training data contain NaN or Inf")
        alpha_values = tuple(float(value) for value in alphas)
        if not alpha_values or any(value <= 0 for value in alpha_values):
            raise ValueError("alphas must be non-empty and positive")
        mean = x.mean(axis=0)
        scale = x.std(axis=0)
        scale[scale < EPS] = 1.0
        centered_x = group_center((x - mean) / scale, groups)
        centered_y = group_center(y[:, None], groups)[:, 0]
        weights = _ridge_weights_path(centered_x, centered_y, alpha_values)
        return {
            alpha: cls(
                tuple(feature_names),
                mean.copy(),
                scale.copy(),
                weight,
                alpha,
            )
            for alpha, weight in weights.items()
        }

    def predict(
        self,
        features: np.ndarray,
        group_ids: Sequence[str],
    ) -> np.ndarray:
        x = np.asarray(features, dtype=np.float64)
        groups = np.asarray(group_ids, dtype=str)
        if x.ndim != 2 or x.shape[1] != len(self.feature_names):
            raise ValueError("invalid linear scorer feature shape")
        standardized = (x - self.mean) / self.scale
        prediction = group_center(standardized, groups) @ self.weights
        if not np.isfinite(prediction).all():
            raise RuntimeError("linear scorer produced NaN or Inf")
        return prediction

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "shared_group_centered_linear_ridge_v1",
            "feature_names": list(self.feature_names),
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "weights": self.weights.tolist(),
            "alpha": self.alpha,
            "supervision": self.supervision,
            "world_model_gradients": False,
            "candidate_identity_inputs": False,
        }


__all__ = ["SharedLinearScorer", "group_center"]
