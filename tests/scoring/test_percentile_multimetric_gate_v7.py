from __future__ import annotations

import numpy as np

from scoring.percentile_multimetric_gate_v7 import (
    PercentileMultimetricPolicyV7,
    empirical_cdf,
    select_percentile_multimetric,
)


def test_empirical_cdf_uses_frozen_reference() -> None:
    reference = np.asarray([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
    value = np.asarray([-1.0, 0.0, 1.5, 4.0], dtype=np.float32)
    assert np.allclose(
        empirical_cdf(value, reference),
        np.asarray([0.0, 0.25, 0.50, 1.0], dtype=np.float32),
    )


def test_multimetric_gate_rejects_bad_rpe_tail() -> None:
    mean = np.zeros((2, 17, 4), dtype=np.float32)
    std = np.full_like(mean, 0.001)
    mean[:, 1, (0, 3)] = -0.02
    mean[:, 1, 1] = np.asarray([0.001, 0.02], dtype=np.float32)
    prediction = {"mean": mean, "std": std}
    q_dual = np.zeros((2, 17), dtype=np.float32)
    q_dual[:, 1] = 0.95
    q_harm = np.ones((2, 17), dtype=np.float32)
    q_harm[:, 1] = 0.05
    q_ood = np.zeros((2, 17), dtype=np.float32)
    q_disagreement = np.zeros((2, 17), dtype=np.float32)
    valid = np.ones((2, 17), dtype=bool)
    policy = PercentileMultimetricPolicyV7(
        harm_rank_weight=1.0,
        uncertainty_rank_weight=0.0,
        rpe_rank_weight=0.0,
        endpoint_rank_weight=0.0,
        z=0.0,
        margin_ate=0.0,
        margin_objective=0.0,
        rpe_ucb_cap=0.005,
        endpoint_ucb_cap=0.01,
        dual_quantile_threshold=0.8,
        harm_quantile_threshold=0.2,
    )
    selected = select_percentile_multimetric(
        prediction,
        q_dual,
        q_harm,
        q_ood,
        q_disagreement,
        valid,
        policy,
    )
    assert selected.tolist() == [1, 0]
