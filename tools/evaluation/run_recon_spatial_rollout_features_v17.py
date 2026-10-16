#!/usr/bin/env python3
"""Run v17 with spatial statistics forced to float32 outside autocast."""

from __future__ import annotations

from pathlib import Path
import sys

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tools.evaluation import export_recon_spatial_rollout_features_v17 as v17


_spatial_rollout_feature = v17.spatial_rollout_feature


def float32_spatial_rollout_feature(
    future: torch.Tensor,
    goal: torch.Tensor,
) -> torch.Tensor:
    with torch.autocast(
        device_type=future.device.type,
        enabled=False,
    ):
        return _spatial_rollout_feature(
            future.float(),
            goal.float(),
        )


if __name__ == "__main__":
    v17.spatial_rollout_feature = float32_spatial_rollout_feature
    v17.main()
