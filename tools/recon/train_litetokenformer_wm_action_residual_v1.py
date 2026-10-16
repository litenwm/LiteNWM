#!/usr/bin/env python3
"""Warm-start one LTWM using only action-sensitive true future-latent losses."""

from __future__ import annotations

from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(Path(__file__).resolve().parent)]

from recon_latents.losses_action_residual_v1 import (  # noqa: E402
    ActionResidualLossWeights,
    compute_action_residual_world_model_loss,
)
import train_litetokenformer_wm_v2_4ctx as trainer  # noqa: E402


_ORIGINAL_PARSE_ARGS = trainer.parse_args


def parse_args():
    args = _ORIGINAL_PARSE_ARGS()
    if args.objective_revision != "base":
        raise ValueError("action-residual entry requires --objective-revision base")
    args.objective_revision = "pure_future_latent_action_residual_v1"
    return args


def step_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    history: torch.Tensor,
) -> dict[str, list[float]]:
    pred = prediction[:, 0].float()
    target = target.float()
    current = history[:, -1].float()
    error = (pred - target).square().mean(dim=(-1, -2))
    denominator = target.square().mean(dim=(-1, -2)).clamp_min(1.0e-8)
    nmse = (error / denominator).mean(0)
    cosine = (
        1.0
        - torch.nn.functional.cosine_similarity(
            pred.flatten(-2),
            target.flatten(-2),
            dim=-1,
        )
    ).mean(0)
    predicted_residual = pred - current[:, None]
    target_residual = target - current[:, None]
    residual_error = (
        (predicted_residual - target_residual).square().mean(dim=(-1, -2))
    )
    residual_energy = target_residual.square().mean(dim=(-1, -2)).clamp_min(1.0e-4)
    residual_nmse = (residual_error / residual_energy).mean(0)
    residual_cosine = (
        1.0
        - torch.nn.functional.cosine_similarity(
            predicted_residual.flatten(-2),
            target_residual.flatten(-2),
            dim=-1,
        )
    ).mean(0)
    norm_ratio = (
        torch.linalg.vector_norm(predicted_residual.flatten(-2), dim=-1)
        / torch.linalg.vector_norm(target_residual.flatten(-2), dim=-1).clamp_min(1.0e-8)
    ).mean(0)
    return {
        "nmse": nmse.tolist(),
        "cosine": cosine.tolist(),
        "residual_nmse": residual_nmse.tolist(),
        "residual_cosine": residual_cosine.tolist(),
        "norm_ratio": norm_ratio.tolist(),
    }


trainer.parse_args = parse_args
trainer.V2LossWeights = ActionResidualLossWeights
trainer.compute_world_model_v2_loss = compute_action_residual_world_model_loss
trainer.step_metrics = step_metrics


if __name__ == "__main__":
    trainer.main()
