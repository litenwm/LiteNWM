#!/usr/bin/env python3
"""Resume the frozen reranker after mechanical export/training shape fixes."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch


_ORIGINAL_TRANSPOSE = torch.Tensor.transpose


def _transpose_or_permute(
    value: torch.Tensor,
    *dimensions: int,
) -> torch.Tensor:
    if len(dimensions) > 2:
        return value.permute(*dimensions)
    return _ORIGINAL_TRANSPOSE(value, *dimensions)


torch.Tensor.transpose = _transpose_or_permute

from tools.evaluation import run_scand_top10_latent_reranker_v1 as runner  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
EXPORTED = (
    ROOT
    / "outputs/cem_development_v1/scand_top10_latent_reranker_v1"
    / "rich_ltwm_runtime_features.npz"
)


def _reuse_export(
    contract,
    data,
    output,
    device,
    state_batch,
    candidate_chunk,
):
    if not EXPORTED.is_file():
        raise FileNotFoundError(EXPORTED)
    os.link(EXPORTED, output)
    return {
        "path": str(output),
        "sha256": runner.sha256_file(output),
        "runtime_seconds": 358.39631806899706,
        "labels_included": False,
        "reused_after_mechanical_training_shape_fix": True,
    }


def _stream_normalizer(
    rich,
    actions,
    data,
    states,
    candidates,
    route,
    batch_size,
):
    weight = runner.coarse.state_weights(route, states)
    total = 0.0
    total_sum = None
    total_square = None
    token_dimension = 0
    for start in range(0, len(states), batch_size):
        end = min(start + batch_size, len(states))
        token = runner.token_batch(
            rich,
            actions,
            data,
            states[start:end],
            candidates[start:end],
        ).astype(np.float64)
        token_dimension = token.shape[-1]
        local_weight = weight[start:end, None, None, None, None]
        normalizer = token.shape[1] * token.shape[2] * token.shape[3]
        local_sum = (token * local_weight).sum(axis=(0, 1, 2, 3))
        local_square = ((token * token) * local_weight).sum(axis=(0, 1, 2, 3))
        local_total = float(weight[start:end].sum() * normalizer)
        total_sum = local_sum if total_sum is None else total_sum + local_sum
        total_square = (
            local_square if total_square is None else total_square + local_square
        )
        total += local_total
    if total_sum is None or total_square is None:
        raise RuntimeError("empty normalization set")
    mean = total_sum / total
    variance = np.maximum(total_square / total - mean * mean, 0.0)
    std = np.maximum(np.sqrt(variance), 1.0e-5)
    return mean.astype(np.float32), std.astype(np.float32), token_dimension


runner.export_rich_features = _reuse_export
runner.stream_normalizer = _stream_normalizer


if __name__ == "__main__":
    runner.main()
