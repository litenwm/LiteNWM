from __future__ import annotations

import pytest
import torch

from recon_latents.losses_v2 import compute_world_model_v2_loss
from recon_latents.models_v2 import LiteTokenFormerConfig, LiteTokenFormerWMV2


def test_ltwm_four_step_forward_and_loss() -> None:
    config = LiteTokenFormerConfig(
        horizon=4,
        model_dim=24,
        heads=4,
        history_layers=1,
        action_layers=1,
        rollout_blocks=1,
        spatial_window=8,
        mlp_ratio=2,
    )
    model = LiteTokenFormerWMV2(config).eval()
    history = torch.randn(1, 4, 256, 384)
    actions = 0.01 * torch.randn(1, 4, 3)
    target = torch.randn(1, 4, 256, 384)
    with torch.no_grad():
        output = model(history, actions)
        loss, _ = compute_world_model_v2_loss(output, target, history, actions)
    assert output["predicted_tokens"].shape == (1, 1, 4, 256, 384)
    assert output["decoded_action"].shape == (1, 1, 4, 3)
    assert torch.isfinite(loss)


def test_ltwm_rejects_actions_outside_configured_horizon() -> None:
    model = LiteTokenFormerWMV2(
        LiteTokenFormerConfig(
            horizon=4,
            model_dim=24,
            heads=4,
            history_layers=1,
            action_layers=1,
            rollout_blocks=1,
        )
    )
    with pytest.raises(ValueError, match="\\[B,K,4,3\\]"):
        model(torch.randn(1, 4, 256, 384), torch.randn(1, 6, 3))


def test_default_horizon_remains_six_for_legacy_checkpoints() -> None:
    assert LiteTokenFormerConfig().horizon == 6
