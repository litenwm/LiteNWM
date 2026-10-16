from __future__ import annotations

import math
import unittest

import numpy as np

from scoring.nwm_trajectory_metrics import (
    integrate_local_actions,
    nwm_ate_rpe,
)


class NwmTrajectoryMetricsTest(unittest.TestCase):
    def test_nwm_metrics_are_zero_for_identical_trajectories(self) -> None:
        trajectory = np.asarray(
            ((0.25, 0.0), (0.50, 0.0), (0.75, 0.1), (1.00, 0.2))
        )
        ate, rpe = nwm_ate_rpe(
            trajectory,
            trajectory,
            waypoint_spacing_m=0.25,
        )
        self.assertAlmostEqual(ate, 0.0)
        self.assertAlmostEqual(rpe, 0.0)

    def test_nwm_ate_is_ape_rmse_and_rpe_uses_adjacent_waypoints(self) -> None:
        reference = np.zeros((4, 2), dtype=np.float64)
        predicted = np.asarray(
            ((1.0, 0.0), (1.0, 0.0), (3.0, 0.0), (3.0, 0.0))
        )
        ate, rpe = nwm_ate_rpe(
            predicted,
            reference,
            waypoint_spacing_m=2.0,
        )
        self.assertAlmostEqual(ate, math.sqrt(5.0) / 2.0)
        self.assertAlmostEqual(rpe, math.sqrt(4.0 / 3.0) / 2.0)

    def test_nwm_rpe_ignores_constant_translation_but_ate_does_not(self) -> None:
        reference = np.asarray(
            ((0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0))
        )
        predicted = reference + np.asarray((2.0, -1.0))
        ate, rpe = nwm_ate_rpe(
            predicted,
            reference,
            waypoint_spacing_m=1.0,
        )
        self.assertAlmostEqual(ate, math.sqrt(5.0))
        self.assertAlmostEqual(rpe, 0.0)

    def test_local_action_integration_respects_heading(self) -> None:
        actions = np.asarray(
            (
                (1.0, 0.0, math.pi / 2.0),
                (1.0, 0.0, 0.0),
                (1.0, 0.0, -math.pi / 2.0),
                (1.0, 0.0, 0.0),
            )
        )
        positions = integrate_local_actions(actions)
        np.testing.assert_allclose(
            positions,
            np.asarray(((1.0, 0.0), (1.0, 1.0), (1.0, 2.0), (2.0, 2.0))),
            atol=1.0e-12,
        )


if __name__ == "__main__":
    unittest.main()
