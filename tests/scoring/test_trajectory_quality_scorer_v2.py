"""Leakage-boundary and invariance tests for trajectory scorer-v2."""

from __future__ import annotations

import inspect
import unittest

import torch

from scoring.trajectory_quality_objective_v2 import (
    TrajectoryQualityLossConfig,
    trajectory_quality_loss,
)
from scoring.trajectory_quality_scorer_v2 import (
    TrajectoryQualityPrediction,
    TrajectoryQualityScorerConfig,
    TrajectoryQualityScorerV2,
)


def scorer(*, future: bool) -> TrajectoryQualityScorerV2:
    torch.manual_seed(19)
    return TrajectoryQualityScorerV2(
        TrajectoryQualityScorerConfig(
            hidden_dim=32,
            hidden_layers=1,
            dropout=0.0,
            use_predicted_future=future,
        )
    ).eval()


class TrajectoryQualityScorerV2Tests(unittest.TestCase):
    def setUp(self) -> None:
        generator = torch.Generator().manual_seed(23)
        self.current = torch.randn(1, 256, 384, generator=generator)
        self.goal = torch.randn(1, 256, 384, generator=generator)
        self.actions = torch.randn(1, 4, 4, 3, generator=generator)
        self.future = torch.randn(1, 4, 4, 256, 384, generator=generator)

    def test_public_forward_signature_has_no_identity_or_label_inputs(self) -> None:
        parameters = set(inspect.signature(TrajectoryQualityScorerV2.forward).parameters)
        allowed = {
            "self",
            "current_latent",
            "goal_latent",
            "candidate_actions",
            "predicted_future",
        }
        self.assertEqual(parameters, allowed)
        forbidden = {
            "candidate_index",
            "candidate_family",
            "route_id",
            "dataset_id",
            "ate",
            "rtp",
            "pose",
            "iteration",
        }
        self.assertTrue(parameters.isdisjoint(forbidden))

    def test_variable_k_and_absolute_output_ranges(self) -> None:
        model = scorer(future=False)
        for candidate_count in (1, 16, 64):
            actions = self.actions[:, :1].expand(-1, candidate_count, -1, -1)
            output = model(
                current_latent=self.current,
                goal_latent=self.goal,
                candidate_actions=actions,
            )
            self.assertEqual(output.ate.shape, (1, candidate_count))
            self.assertTrue(bool((output.ate >= 0).all()))
            self.assertTrue(bool((output.rtp <= 1).all()))
            self.assertTrue(torch.isfinite(output.stacked()).all())

    def test_m0_and_m1_are_parameter_matched(self) -> None:
        self.assertEqual(
            scorer(future=False).parameter_count(),
            scorer(future=True).parameter_count(),
        )

    def test_candidate_permutation_equivariance_and_no_candidate0_anchor(self) -> None:
        model = scorer(future=True)
        permutation = torch.tensor([2, 0, 3, 1])
        inverse = torch.argsort(permutation)
        baseline = model(
            current_latent=self.current,
            goal_latent=self.goal,
            candidate_actions=self.actions,
            predicted_future=self.future,
        ).stacked()
        permuted = model(
            current_latent=self.current,
            goal_latent=self.goal,
            candidate_actions=self.actions[:, permutation],
            predicted_future=self.future[:, permutation],
        ).stacked()
        torch.testing.assert_close(baseline, permuted[:, inverse])

        alone = model(
            current_latent=self.current,
            goal_latent=self.goal,
            candidate_actions=self.actions[:, 2:3],
            predicted_future=self.future[:, 2:3],
        ).stacked()
        torch.testing.assert_close(alone[:, 0], baseline[:, 2])

    def test_m0_is_invariant_to_future_intervention(self) -> None:
        model = scorer(future=False)
        first = model(
            current_latent=self.current,
            goal_latent=self.goal,
            candidate_actions=self.actions,
            predicted_future=self.future,
        ).stacked()
        intervened = model(
            current_latent=self.current,
            goal_latent=self.goal,
            candidate_actions=self.actions,
            predicted_future=self.future * -100.0 + 17.0,
        ).stacked()
        torch.testing.assert_close(first, intervened, rtol=0.0, atol=0.0)

    def test_loss_is_permutation_invariant_and_has_no_reference_anchor(self) -> None:
        prediction = TrajectoryQualityPrediction(
            ate=torch.tensor([[0.4, 0.2, 0.7, 0.3]]),
            rtp=torch.tensor([[0.5, 0.8, 0.2, 0.6]]),
        )
        target = torch.tensor(
            [[[0.5, 0.4], [0.2, 0.9], [0.8, 0.1], [0.3, 0.7]]]
        )
        valid = torch.ones(1, 4, dtype=torch.bool)
        config = TrajectoryQualityLossConfig(ate_scale=0.5, rtp_scale=0.25)
        baseline = trajectory_quality_loss(prediction, target, valid, config)

        permutation = torch.tensor([3, 1, 0, 2])
        changed = trajectory_quality_loss(
            TrajectoryQualityPrediction(
                ate=prediction.ate[:, permutation],
                rtp=prediction.rtp[:, permutation],
            ),
            target[:, permutation],
            valid[:, permutation],
            config,
        )
        torch.testing.assert_close(baseline.total, changed.total)
        torch.testing.assert_close(baseline.absolute, changed.absolute)
        torch.testing.assert_close(baseline.pairwise, changed.pairwise)
        torch.testing.assert_close(baseline.dominance, changed.dominance)
        self.assertEqual(baseline.valid_pairs, 6)

    def test_dominance_loss_prefers_correct_pareto_direction(self) -> None:
        target = torch.tensor([[[0.1, 0.9], [0.9, 0.1]]])
        valid = torch.ones(1, 2, dtype=torch.bool)
        config = TrajectoryQualityLossConfig(
            ate_scale=1.0,
            rtp_scale=1.0,
            absolute_weight=0.0,
            pairwise_weight=0.0,
            dominance_weight=1.0,
        )
        correct = trajectory_quality_loss(
            TrajectoryQualityPrediction(
                ate=torch.tensor([[0.1, 0.9]]),
                rtp=torch.tensor([[0.9, 0.1]]),
            ),
            target,
            valid,
            config,
        )
        reversed_prediction = trajectory_quality_loss(
            TrajectoryQualityPrediction(
                ate=torch.tensor([[0.9, 0.1]]),
                rtp=torch.tensor([[0.1, 0.9]]),
            ),
            target,
            valid,
            config,
        )
        self.assertLess(correct.dominance, reversed_prediction.dominance)
        self.assertEqual(correct.dominance_pairs, 1)


if __name__ == "__main__":
    unittest.main()
