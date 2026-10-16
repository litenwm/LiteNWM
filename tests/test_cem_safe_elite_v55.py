from __future__ import annotations

import numpy as np

from tools.planning.run_cem_safe_elite_v55 import (
    candidate_safety_mask,
    safe_elite_update,
)


def test_candidate_safety_mask_applies_every_frozen_threshold() -> None:
    mean = np.zeros((1, 5, 4), dtype=np.float32)
    std = np.zeros_like(mean)
    mean[:, :, 0] = -0.02
    mean[:, :, 3] = -0.02
    prediction = {"mean": mean, "std": std}
    p_dual = np.full((1, 5), 0.95, dtype=np.float32)
    p_harm = np.full((1, 5), 0.10, dtype=np.float32)
    valid = np.ones((1, 5), dtype=bool)
    policy = {
        "z": 0.5,
        "margin_ate": 0.01,
        "margin_objective": 0.01,
        "dual_threshold": 0.90,
        "harm_threshold": 0.20,
    }

    p_dual[0, 2] = 0.89
    p_harm[0, 3] = 0.21
    valid[0, 4] = False
    safe = candidate_safety_mask(prediction, p_dual, p_harm, valid, policy)

    assert safe.tolist() == [[False, True, False, False, False]]


def test_safe_elite_update_keeps_previous_q_when_fewer_than_four_safe() -> None:
    candidates = np.arange(2 * 17 * 4 * 2, dtype=np.float32).reshape(2, 17, 4, 2)
    energy = np.tile(np.arange(17, dtype=np.float32), (2, 1))
    safe = np.zeros((2, 17), dtype=bool)
    safe[0, [1, 2, 3]] = True
    safe[1, [4, 5, 6, 7]] = True
    previous_mean = np.full((2, 8), -7.0, dtype=np.float32)
    previous_covariance = np.stack([np.eye(8, dtype=np.float32)] * 2)

    mean, covariance, elite, updated, safe_count = safe_elite_update(
        candidates,
        energy,
        safe,
        previous_mean=previous_mean,
        previous_covariance=previous_covariance,
        elite_count=4,
        floor=0.015,
        scale=0.70,
    )

    assert safe_count.tolist() == [3, 4]
    assert updated.tolist() == [False, True]
    assert elite[0].tolist() == [-1, -1, -1, -1]
    assert elite[1].tolist() == [4, 5, 6, 7]
    np.testing.assert_array_equal(mean[0], previous_mean[0])
    np.testing.assert_array_equal(covariance[0], previous_covariance[0])
    assert np.isfinite(mean[1]).all()
    assert np.isfinite(covariance[1]).all()
