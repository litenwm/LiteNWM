from __future__ import annotations

import torch

from tools.evaluation.export_recon_spatial_rollout_features_v17 import (
    spatial_rollout_feature,
)


def test_spatial_rollout_feature_shape_and_finite() -> None:
    generator = torch.Generator().manual_seed(17)
    goal = torch.randn(1, 256, 384, generator=generator)
    future = torch.randn(1, 2, 4, 256, 384, generator=generator)

    value = spatial_rollout_feature(future, goal)

    assert value.shape == (1, 2, 129)
    assert torch.isfinite(value).all()


def test_spatial_rollout_feature_is_candidate_conditioned() -> None:
    generator = torch.Generator().manual_seed(18)
    goal = torch.randn(1, 256, 384, generator=generator)
    future = goal[:, None, None].repeat(1, 2, 4, 1, 1)
    future[:, 1] = torch.roll(future[:, 1], shifts=1, dims=-2)

    value = spatial_rollout_feature(future, goal)

    assert not torch.allclose(value[:, 0], value[:, 1])
