#!/usr/bin/env python3
"""Mechanical entrypoint for the frozen top-10 latent reranker experiment."""

from __future__ import annotations

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

from tools.evaluation.run_scand_top10_latent_reranker_v1 import main  # noqa: E402


if __name__ == "__main__":
    main()
