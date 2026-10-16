from __future__ import annotations

import numpy as np
import torch

from scoring.route_multimodal_calibration_v5 import (
    MULTIMODAL_RISK_FEATURE_DIM,
    RouteMultimodalCalibratorV5,
    calibrated_probabilities_v5,
    multimodal_risk_features,
)


def test_multimodal_contract_and_probabilities() -> None:
    base = np.zeros((3, 17, 27), dtype=np.float32)
    raw = np.zeros((3, 17, 815), dtype=np.float32)
    feature = multimodal_risk_features(base, raw)
    assert feature.shape == (3, 17, MULTIMODAL_RISK_FEATURE_DIM)

    model = RouteMultimodalCalibratorV5(dropout=0.0)
    stats = {
        "feature_center": np.zeros(
            MULTIMODAL_RISK_FEATURE_DIM, dtype=np.float32
        ),
        "feature_scale": np.ones(
            MULTIMODAL_RISK_FEATURE_DIM, dtype=np.float32
        ),
    }
    dual, harm = calibrated_probabilities_v5(
        model, feature, stats, device="cpu", batch_candidates=13
    )
    assert dual.shape == (3, 17)
    assert harm.shape == (3, 17)
    assert np.isfinite(dual).all()
    assert np.isfinite(harm).all()
    assert ((dual >= 0.0) & (dual <= 1.0)).all()
    assert ((harm >= 0.0) & (harm <= 1.0)).all()


def test_multimodal_model_output_shape() -> None:
    model = RouteMultimodalCalibratorV5()
    output = model(torch.zeros(5, MULTIMODAL_RISK_FEATURE_DIM))
    assert output.shape == (5, 2)
