"""Deterministic tests for multi-modal archive-preserving H4 CEM."""

from __future__ import annotations

import unittest

import torch

from planning.cem_sampler import (
    CEMConfig,
    MultiModalCEM,
    conservative_joint_cost,
    paired_conservative_joint_cost,
    conservative_replacement_mask,
)
from scoring.trajectory_quality_scorer_v2 import TrajectoryQualityEnsemble


def ensemble(
    ate: torch.Tensor,
    rtp: torch.Tensor,
    std: float = 0.0,
):
    return TrajectoryQualityEnsemble(
        ate_mean=ate,
        ate_std=torch.full_like(ate, std),
        rtp_mean=rtp,
        rtp_std=torch.full_like(rtp, std),
    )


class CEMSamplerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = CEMConfig(
            modes=4,
            population_size=64,
            elites_per_mode=4,
            iterations=4,
            std_floor=0.05,
            std_ceiling=0.8,
            seed=31,
        )

    def test_kmedoids_keeps_distinct_modes_and_sampling_is_deterministic(self) -> None:
        centers = torch.tensor([-1.0, -0.3, 0.4, 1.1])
        parents = torch.zeros(1, 16, 4, 3)
        for mode, center in enumerate(centers):
            parents[0, mode * 4 : (mode + 1) * 4, :, 0] = (
                center + torch.linspace(-0.03, 0.03, 4)[:, None]
            )
        cem = MultiModalCEM(self.config)
        state = cem.initialize(parents)
        sorted_means = state.means[0, :, :, 0].mean(dim=-1).sort().values
        torch.testing.assert_close(sorted_means, centers, atol=0.04, rtol=0.0)
        first = cem.sample(state)
        second = cem.sample(state)
        torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)
        self.assertGreater(float(first.max()), float(state.means.max()))

    def test_invalid_mode_keeps_state_and_std_never_collapses(self) -> None:
        parents = torch.randn(1, 8, 4, 3) * 0.1
        cem = MultiModalCEM(self.config)
        state = cem.initialize(parents)
        samples = cem.sample(state)
        costs = samples.square().sum(dim=(-1, -2))
        valid = torch.ones_like(costs, dtype=torch.bool)
        valid[:, 2] = False
        updated = cem.update(state, samples, costs, valid)
        torch.testing.assert_close(updated.means[:, 2], state.means[:, 2])
        torch.testing.assert_close(updated.stds[:, 2], state.stds[:, 2])
        self.assertTrue(bool((updated.stds >= self.config.std_floor).all()))

    def test_refine_preserves_parent_archive_and_budget(self) -> None:
        parents = torch.linspace(0.6, 1.4, 16).reshape(1, 16, 1, 1)
        parents = parents.expand(-1, -1, 4, 3).clone()

        def cost_function(actions: torch.Tensor):
            costs = actions.square().mean(dim=(-1, -2))
            return costs, torch.isfinite(costs)

        cem = MultiModalCEM(self.config)
        result = cem.refine(
            parents,
            cost_function,
            constraint_function=lambda values: torch.ones(
                values.shape[:2],
                dtype=torch.bool,
                device=values.device,
            ),
            fixed_winner_indices=torch.tensor([0]),
        )
        self.assertEqual(result.archive_actions.shape[1], 16 + 4 * 64)
        self.assertEqual(int(result.archive_is_parent.sum()), 16)
        torch.testing.assert_close(result.archive_actions[:, :16], parents)
        self.assertEqual(result.generated_evaluations, 256)
        self.assertLessEqual(
            float(result.selected_costs[0]),
            float(cost_function(parents)[0].min()),
        )

    def test_all_generated_invalid_falls_back_to_best_fixed_parent(self) -> None:
        parents = torch.randn(1, 8, 4, 3)

        def cost_function(actions: torch.Tensor):
            costs = actions.square().sum(dim=(-1, -2))
            return costs, torch.ones_like(costs, dtype=torch.bool)

        fixed_winner = int(cost_function(parents)[0].argmin())
        cem = MultiModalCEM(self.config)
        result = cem.refine(
            parents,
            cost_function,
            constraint_function=lambda values: torch.zeros(
                values.shape[:2],
                dtype=torch.bool,
                device=values.device,
            ),
            fixed_winner_indices=torch.tensor([fixed_winner]),
        )
        self.assertLess(result.selected_archive_indices.item(), parents.shape[1])
        torch.testing.assert_close(
            result.selected_actions[0],
            parents[0, fixed_winner],
        )

    def test_conservative_cost_requires_both_metric_improvements(self) -> None:
        candidate = ensemble(
            torch.tensor([[0.8, 0.7, 1.2]]),
            torch.tensor([[0.7, 0.4, 0.8]]),
            std=0.01,
        )
        reference = ensemble(
            torch.tensor([[1.0, 1.0, 1.0]]),
            torch.tensor([[0.5, 0.5, 0.5]]),
            std=0.01,
        )
        cost = conservative_joint_cost(
            candidate,
            reference,
            ate_scale=0.2,
            rtp_scale=0.2,
            uncertainty_beta=1.0,
        )
        replace = conservative_replacement_mask(
            cost,
            ate_margin=0.01,
            rtp_margin=0.01,
        )
        self.assertEqual(replace.tolist(), [[True, False, False]])
        self.assertLess(float(cost.cost[0, 0]), 0.0)
        self.assertGreater(float(cost.cost[0, 1]), 0.0)
        self.assertGreater(float(cost.cost[0, 2]), 0.0)

    def test_paired_reference_has_exact_zero_cost(self) -> None:
        ate = torch.tensor(
            [[[1.0, 0.8]], [[1.2, 0.9]], [[0.9, 0.7]]]
        )
        rtp = torch.tensor(
            [[[0.4, 0.6]], [[0.5, 0.7]], [[0.3, 0.5]]]
        )
        cost = paired_conservative_joint_cost(
            ate,
            rtp,
            ate[..., :1],
            rtp[..., :1],
            ate_scale=0.2,
            rtp_scale=0.2,
        )
        self.assertEqual(float(cost.cost[0, 0]), 0.0)
        self.assertEqual(float(cost.lcb_ate_gain[0, 0]), 0.0)
        self.assertEqual(float(cost.lcb_rtp_gain[0, 0]), 0.0)
        self.assertLess(float(cost.cost[0, 1]), 0.0)


if __name__ == "__main__":
    unittest.main()
