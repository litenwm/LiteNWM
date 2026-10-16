#!/usr/bin/env python3
"""Paired evaluator for the float32-kernel scene-warp LTWM."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "tools/recon")]

from recon_latents.models_scene_conditioned_warp_cuda_safe_v1 import (  # noqa: E402
    LiteTokenFormerWMSceneWarpCUDAFloat32V1,
)
from tools.evaluation import evaluate_ltwm_action_residual_v1 as evaluation  # noqa: E402


evaluation.LiteTokenFormerWMV2 = LiteTokenFormerWMSceneWarpCUDAFloat32V1
evaluation.false = False
_ORIGINAL_DUMPS = json.dumps


def _dumps(value, *args, **kwargs):
    kwargs.setdefault(
        "default",
        lambda item: item.item() if isinstance(item, np.generic) else str(item),
    )
    return _ORIGINAL_DUMPS(value, *args, **kwargs)


evaluation.json.dumps = _dumps


if __name__ == "__main__":
    evaluation.main()
    output = Path(sys.argv[sys.argv.index("--output") + 1]).resolve()
    report = json.loads(output.read_text(encoding="utf-8"))
    report["schema"] = "ltwm_scene_conditioned_token_warp_evaluation_v1"
    report["architecture"] = (
        "frozen incumbent LTWM plus state-conditioned per-token flow; "
        "flow and grid-sample kernel executed in float32"
    )
    report["inputs"]["scene_warp_evaluator"] = {
        "path": str(Path(__file__).resolve()),
        "sha256": evaluation.sha256_file(Path(__file__).resolve()),
    }
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "decision": report["decision"],
                "checks": report["checks"],
                "output": str(output),
                "output_sha256": evaluation.sha256_file(output),
            },
            sort_keys=True,
        )
    )
