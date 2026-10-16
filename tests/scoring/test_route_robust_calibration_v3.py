from __future__ import annotations

import numpy as np
import torch

from scoring.route_robust_calibration_v3 import (
    RISK_FEATURE_DIM,
    RouteRiskCalibratorV3,
    RouteRiskPolicyV3,
    normalization,
    risk_features,
    select_fixed_pool,
)


def prediction(states: int = 2, candidates: int = 3) -> dict[str, np.ndarray]:
    return {
        "mean": np.zeros((states, candidates, 4), dtype=np.float32),
        "std": np.full((states, candidates, 4), 0.1, dtype=np.float32),
        "helpful_logit": np.zeros((states, candidates), dtype=np.float32),
        "dual_logit": np.zeros((states, candidates), dtype=np.float32),
        "harm_logit": np.zeros((states, candidates), dtype=np.float32),
        "rank_logit": np.zeros((states, candidates), dtype=np.float32),
    }


def test_risk_feature_contract_and_model_shape() -> None:
    value = prediction()
    members = [prediction(), prediction(), prediction()]
    support = np.arange(6, dtype=np.float32).reshape(2, 3)
    feature = risk_features(value, members, support)
    assert feature.shape == (2, 3, RISK_FEATURE_DIM)
    stats = normalization(feature, np.ones((2, 3), dtype=bool))
    assert stats["feature_center"].shape == (RISK_FEATURE_DIM,)
    output = RouteRiskCalibratorV3()(torch.from_numpy(feature))
    assert output.shape == (2, 3, 2)


def test_fixed_pool_gate_keeps_incumbent_when_risk_fails() -> None:
    value = prediction(states=1)
    value["mean"][0, 1, (0, 3)] = -0.2
    value["mean"][0, 2, (0, 3)] = -0.1
    p_dual = np.asarray([[0.0, 0.95, 0.80]], dtype=np.float32)
    p_harm = np.asarray([[1.0, 0.05, 0.01]], dtype=np.float32)
    valid = np.ones((1, 3), dtype=bool)
    policy = RouteRiskPolicyV3(
        harm_rank_weight=1.0,
        uncertainty_rank_weight=0.0,
        z=0.0,
        margin=0.01,
        dual_threshold=0.90,
        harm_threshold=0.10,
    )
    assert select_fixed_pool(value, p_dual, p_harm, valid, policy).tolist() == [1]
    strict = RouteRiskPolicyV3(
        **{**policy.__dict__, "dual_threshold": 0.99}
    )
    assert select_fixed_pool(value, p_dual, p_harm, valid, strict).tolist() == [0]
