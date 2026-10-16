"""Tests for train-fitted H4 trajectory support constraints."""

from __future__ import annotations

import unittest

import torch

from planning.trajectory_constraints import (
    ActionSupportStats,
    evaluate_trajectory_constraints,
)


class TrajectoryConstraintTests(unittest.TestCase):
    def test_bounds_support_smoothness_trust_and_duplicates(self) -> None:
        stats = ActionSupportStats(
            lower=torch.tensor([-1.0, -1.0, -1.0]),
            upper=torch.tensor([1.0, 1.0, 1.0]),
            mean=torch.zeros(12),
            covariance_inverse=torch.eye(12),
            mahalanobis_squared_max=6.0,
            support_actions=torch.zeros(2, 12),
            knn_k=1,
            knn_distance_max=0.4,
            roughness_max=0.5,
            jerk_max=1.0,
            curvature_max=3.0,
            parent_trust_radius=0.4,
            duplicate_tolerance=1.0e-8,
        )
        parents = torch.zeros(1, 1, 4, 3)
        actions = torch.zeros(1, 5, 4, 3)
        actions[0, 1, :, 0] = 0.2
        actions[0, 2, :, 0] = 2.0  # bounds/support/trust failure
        actions[0, 3, :, 0] = torch.tensor([0.0, 0.9, -0.9, 0.9])
        actions[0, 4] = actions[0, 1]  # duplicate of candidate 1
        result = evaluate_trajectory_constraints(
            actions,
            stats,
            parent_actions=parents,
        )
        self.assertEqual(
            result.valid.tolist(),
            [[True, True, False, False, False]],
        )
        self.assertFalse(bool(result.in_bounds[0, 2]))
        self.assertFalse(bool(result.smooth[0, 3]))
        self.assertFalse(bool(result.unique[0, 4]))
        self.assertGreater(float(result.knn_distance[0, 2]), 0.4)

    def test_nan_is_rejected_without_propagating_diagnostics(self) -> None:
        stats = ActionSupportStats(
            lower=torch.full((3,), -1.0),
            upper=torch.full((3,), 1.0),
        )
        actions = torch.zeros(1, 1, 4, 3)
        actions[0, 0, 1, 0] = torch.nan
        result = evaluate_trajectory_constraints(actions, stats)
        self.assertFalse(bool(result.valid[0, 0]))
        self.assertTrue(torch.isfinite(result.roughness).all())
        self.assertTrue(torch.isfinite(result.curvature).all())

    def test_training_like_coherent_turn_can_use_tail_support(self) -> None:
        template = torch.tensor(
            [
                [0.069081, 0.018285, 0.258755],
                [0.063194, 0.026743, 0.400331],
                [0.076872, 0.040637, 0.486287],
                [0.125453, 0.046932, 0.357980],
            ]
        )
        second_template = template.clone()
        second_template[:, 2] *= 0.9
        lower = torch.tensor(
            [
                [0.075776, -0.052476, -0.175197],
                [0.065973, -0.098059, -0.354264],
                [0.058270, -0.080949, -0.308709],
                [0.052328, -0.073595, -0.281782],
            ]
        )
        upper = torch.tensor(
            [
                [0.502526, 0.058764, 0.195555],
                [0.500860, 0.101684, 0.350438],
                [0.500888, 0.087743, 0.329792],
                [0.500640, 0.083019, 0.317716],
            ]
        )
        stats = ActionSupportStats(
            lower=lower,
            upper=upper,
            tail_support_actions=torch.stack(
                (template.flatten(), second_template.flatten())
            ),
            tail_normalization_lower=lower,
            tail_normalization_upper=upper,
            tail_knn_distance_max=0.126,
            tail_min_forward_m=0.04,
            tail_min_step_length_m=0.05,
            tail_max_step_length_m=0.35,
            tail_max_abs_dyaw_rad=0.75,
            tail_max_abs_yaw_sum_rad=1.70,
            tail_yaw_sign_tolerance_rad=0.03,
        )
        current = torch.tensor(
            [
                [0.083267, 0.019204, 0.226672],
                [0.074857, 0.027698, 0.354390],
                [0.083928, 0.049710, 0.534736],
                [0.121140, 0.053760, 0.417673],
            ]
        )[None, None]
        result = evaluate_trajectory_constraints(current, stats)
        self.assertFalse(bool(result.in_bounds[0, 0]))
        self.assertTrue(bool(result.tail_physical[0, 0]))
        self.assertTrue(bool(result.tail_support[0, 0]))
        self.assertTrue(bool(result.valid[0, 0]))
        self.assertLess(float(result.tail_knn_distance[0, 0]), 0.126)

    def test_tail_support_rejects_short_or_alternating_turns(self) -> None:
        template = torch.tensor(
            [
                [0.08, 0.02, 0.23],
                [0.08, 0.03, 0.35],
                [0.09, 0.05, 0.53],
                [0.12, 0.05, 0.42],
            ]
        )
        stats = ActionSupportStats(
            lower=torch.full((4, 3), -0.1),
            upper=torch.full((4, 3), 0.1),
            tail_support_actions=torch.stack(
                (template.flatten(), (template * 0.95).flatten())
            ),
            tail_normalization_lower=torch.full((4, 3), -0.5),
            tail_normalization_upper=torch.full((4, 3), 0.5),
            tail_knn_distance_max=1.0,
            tail_min_forward_m=0.04,
            tail_min_step_length_m=0.05,
            tail_max_step_length_m=0.35,
            tail_max_abs_dyaw_rad=0.75,
            tail_max_abs_yaw_sum_rad=1.70,
            tail_yaw_sign_tolerance_rad=0.03,
        )
        actions = template.repeat(2, 1, 1)
        actions[0, 2, :2] = torch.tensor([0.005, 0.0])
        actions[1, :, 2] = torch.tensor([0.23, -0.35, 0.53, -0.42])
        result = evaluate_trajectory_constraints(actions[None], stats)
        self.assertEqual(result.tail_physical.tolist(), [[False, False]])
        self.assertEqual(result.tail_support.tolist(), [[False, False]])
        self.assertEqual(result.valid.tolist(), [[False, False]])


if __name__ == "__main__":
    unittest.main()
