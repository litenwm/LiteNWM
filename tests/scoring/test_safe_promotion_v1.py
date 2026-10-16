from __future__ import annotations

import numpy as np
import pytest

from scoring.safe_promotion_v1 import (
    SafePromotionPolicy,
    apply_safe_promotion,
    calibrate_safe_promotion,
    conformal_upper_quantile,
    greedy_fixed_pool_selection,
    paired_relative_gains,
    selection_metrics,
)


def test_paired_relative_gain_has_exact_zero_incumbent_uncertainty() -> None:
    ate = np.asarray(
        [
            [[1.0, 0.7, 0.8]],
            [[2.0, 1.6, 1.9]],
            [[3.0, 2.5, 2.8]],
        ]
    )
    rtp = np.asarray(
        [
            [[0.1, 0.3, 0.2]],
            [[0.0, 0.3, 0.1]],
            [[-0.1, 0.3, 0.0]],
        ]
    )
    gains = paired_relative_gains(ate, rtp, uncertainty_beta=1.0)
    np.testing.assert_array_equal(gains.ate_gain_lcb[:, 0], 0.0)
    np.testing.assert_array_equal(gains.rtp_gain_lcb[:, 0], 0.0)
    assert gains.ate_gain_lcb[0, 1] > gains.ate_gain_lcb[0, 2]
    assert gains.rtp_gain_lcb[0, 1] > gains.rtp_gain_lcb[0, 2]


def test_greedy_selection_falls_back_when_joint_gain_is_not_positive() -> None:
    ate = np.asarray([[[1.0, 0.5, 1.2], [1.0, 0.6, 1.1]]])
    rtp = np.asarray([[[0.5, 0.4, 0.8], [0.5, 0.7, 0.4]]])
    selected = greedy_fixed_pool_selection(
        ate,
        rtp,
        ate_scale=1.0,
        rtp_scale=1.0,
    )
    np.testing.assert_array_equal(selected, np.asarray([0, 1]))


def test_conformal_quantile_uses_finite_sample_order_statistic() -> None:
    # n=4, alpha=.25 => ceil(5*.75)=4, so the maximum is selected.
    assert conformal_upper_quantile(np.asarray([-2.0, 1.0, 3.0, 2.0]), alpha=0.25) == 3.0
    assert conformal_upper_quantile(np.asarray([-3.0, -2.0]), alpha=0.5) == 0.0


def test_safe_gate_calibrates_without_test_outcomes_and_falls_back() -> None:
    # Calibration predicts candidate 1 has gain 1.0, while its realized gain
    # is 0.8.  The conformal correction is therefore 0.2 for both metrics.
    cal_ate = np.asarray(
        [
            [[1.0, 0.0], [2.0, 1.0]],
            [[1.0, 0.0], [2.0, 1.0]],
        ]
    )
    cal_rtp = np.asarray(
        [
            [[0.0, 1.0], [0.0, 1.0]],
            [[0.0, 1.0], [0.0, 1.0]],
        ]
    )
    actual_ate = np.asarray([[1.0, 0.2], [2.0, 1.2]])
    actual_rtp = np.asarray([[0.0, 0.8], [0.1, 0.9]])
    policy = calibrate_safe_promotion(
        cal_ate,
        cal_rtp,
        actual_ate,
        actual_rtp,
        ate_scale=1.0,
        rtp_scale=1.0,
        uncertainty_beta=0.0,
        alpha=0.5,
    )
    assert policy.ate_residual_quantile == pytest.approx(0.2)
    assert policy.rtp_residual_quantile == pytest.approx(0.2)
    assert policy.cem_enabled is False

    test_ate = np.asarray([[[1.0, 0.5], [1.0, 0.9]]])
    test_rtp = np.asarray([[[0.0, 0.6], [0.0, 0.1]]])
    selection = apply_safe_promotion(test_ate, test_rtp, policy)
    np.testing.assert_array_equal(selection.selected_index, np.asarray([1, 0]))
    np.testing.assert_array_equal(selection.promoted, np.asarray([True, False]))


def test_policy_rejects_cem_and_metrics_use_only_promotions() -> None:
    with pytest.raises(ValueError, match="must not enable CEM"):
        SafePromotionPolicy(
            ate_scale=1.0,
            rtp_scale=1.0,
            uncertainty_beta=1.0,
            alpha=0.05,
            ate_residual_quantile=0.0,
            rtp_residual_quantile=0.0,
            calibration_states=3,
            cem_enabled=True,
        ).validate()

    metrics = selection_metrics(
        np.asarray([1, 0, 2]),
        np.asarray([[1.0, 0.5, 2.0], [1.0, 0.0, 2.0], [1.0, 2.0, 0.5]]),
        np.asarray([[0.0, 0.5, -1.0], [0.0, 1.0, -1.0], [0.0, 1.0, -0.5]]),
    )
    assert metrics["promotions"] == 2
    assert metrics["ate_precision"] == 1.0
    assert metrics["rtp_precision"] == 0.5
    assert metrics["dual_precision"] == 0.5

