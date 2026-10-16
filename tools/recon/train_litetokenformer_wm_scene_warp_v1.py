#!/usr/bin/env python3
"""Train one zero-initialized scene-conditioned LTWM token warp."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(Path(__file__).resolve().parent)]

from recon_latents.losses_scene_conditioned_warp_v1 import (  # noqa: E402
    SceneWarpLossWeights,
    compute_scene_warp_world_model_loss,
)
from recon_latents.models_scene_conditioned_warp_v1 import (  # noqa: E402
    LiteTokenFormerWMSceneWarpV1,
)
import train_litetokenformer_wm_action_residual_v1 as residual_entry  # noqa: E402


trainer = residual_entry.trainer
_ORIGINAL_PARSE_ARGS = residual_entry._ORIGINAL_PARSE_ARGS


def parse_args():
    args = _ORIGINAL_PARSE_ARGS()
    if args.objective_revision != "base":
        raise ValueError("scene-warp entry requires --objective-revision base")
    args.objective_revision = "pure_future_latent_scene_conditioned_warp_v1"
    if args.train_scope != "all":
        raise ValueError("scene-warp entry uses model-defined frozen-base scope")
    return args


trainer.parse_args = parse_args
trainer.LiteTokenFormerWMV2 = LiteTokenFormerWMSceneWarpV1
trainer.V2LossWeights = SceneWarpLossWeights
trainer.compute_world_model_v2_loss = compute_scene_warp_world_model_loss
trainer.step_metrics = residual_entry.step_metrics


if __name__ == "__main__":
    trainer.main()
