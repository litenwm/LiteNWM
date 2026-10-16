"""Multimodal current-window risk calibration for universal raw scoring."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from scoring.route_robust_calibration_v3 import normalize


BASE_RISK_FEATURE_DIM = 27
UNIVERSAL_RAW_FEATURE_DIM = 815
MULTIMODAL_RISK_FEATURE_DIM = (
    BASE_RISK_FEATURE_DIM + UNIVERSAL_RAW_FEATURE_DIM
)


class RouteMultimodalCalibratorV5(nn.Module):
    """Residual safety head over raw features and frozen scorer predictions."""

    def __init__(
        self,
        input_dim: int = MULTIMODAL_RISK_FEATURE_DIM,
        hidden_dim: int = 96,
        dropout: float = 0.10,
    ):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2),
        )

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return self.network(feature)


def multimodal_risk_features(
    base_risk: np.ndarray, universal_raw: np.ndarray
) -> np.ndarray:
    """Join only current-window, deployment-visible raw and risk features."""

    risk = np.asarray(base_risk, dtype=np.float32)
    raw = np.asarray(universal_raw, dtype=np.float32)
    if risk.shape[:2] != raw.shape[:2]:
        raise ValueError("base-risk/universal identity mismatch")
    if risk.shape[-1] != BASE_RISK_FEATURE_DIM:
        raise ValueError(f"base risk dimension changed: {risk.shape}")
    if raw.shape[-1] != UNIVERSAL_RAW_FEATURE_DIM:
        raise ValueError(f"universal raw dimension changed: {raw.shape}")
    result = np.concatenate((risk, raw), axis=-1).astype(np.float32)
    if result.shape[-1] != MULTIMODAL_RISK_FEATURE_DIM:
        raise RuntimeError(f"multimodal risk dimension changed: {result.shape}")
    if not np.isfinite(result).all():
        raise RuntimeError("multimodal risk features contain NaN or Inf")
    return result


def calibrated_probabilities_v5(
    model: RouteMultimodalCalibratorV5,
    feature: np.ndarray,
    stats: dict[str, np.ndarray],
    *,
    device: str,
    batch_candidates: int = 8192,
) -> tuple[np.ndarray, np.ndarray]:
    flat = normalize(feature, stats).reshape(-1, feature.shape[-1])
    output = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(flat), batch_candidates):
            value = torch.from_numpy(
                flat[start : start + batch_candidates]
            ).to(device)
            output.append(torch.sigmoid(model(value)).cpu().numpy())
    probability = np.concatenate(output).reshape(*feature.shape[:2], 2)
    return (
        probability[..., 0].astype(np.float32),
        probability[..., 1].astype(np.float32),
    )


__all__ = [
    "BASE_RISK_FEATURE_DIM",
    "MULTIMODAL_RISK_FEATURE_DIM",
    "RouteMultimodalCalibratorV5",
    "calibrated_probabilities_v5",
    "multimodal_risk_features",
]
