#!/usr/bin/env python3
"""Shape-contract fix entry for the metric-depth forward-splat pilot."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time
from typing import Any

from PIL import Image
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(Path(__file__).parent)]

import run_metric_depth_forward_splat_pilot_v1 as pilot  # noqa: E402


def infer_depth_fixed(
    rows: list[pilot.WindowRef],
    model_dir: Path,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    from transformers import (
        DepthProForDepthEstimation,
        DepthProImageProcessorFast,
    )

    processor = DepthProImageProcessorFast.from_pretrained(
        model_dir,
        local_files_only=True,
    )
    model = DepthProForDepthEstimation.from_pretrained(
        model_dir,
        local_files_only=True,
        torch_dtype=torch.float16,
    ).to(device)
    model.eval()
    depth_rows = []
    focal_rows = []
    started = time.perf_counter()
    with torch.inference_mode():
        for index, row in enumerate(rows):
            with Image.open(row.image_path) as image:
                image = image.convert("RGB")
                original_width, original_height = image.size
                inputs = processor(
                    images=image,
                    return_tensors="pt",
                ).to(device)
            with torch.autocast("cuda", dtype=torch.float16):
                output = model(**inputs)
            processed = processor.post_process_depth_estimation(
                output,
                target_sizes=[(original_height, original_width)],
            )[0]
            depth = (
                processed["predicted_depth"]
                .float()
                .squeeze()[None, None]
            )
            depth = F.interpolate(
                depth,
                size=(pilot.GRID_HEIGHT, pilot.GRID_WIDTH),
                mode="area",
            )[0, 0].reshape(-1)
            focal = float(processed["focal_length"])
            focal *= pilot.IMAGE_WIDTH / float(original_width)
            depth_rows.append(depth.cpu())
            focal_rows.append(focal)
            if (index + 1) % 16 == 0 or index + 1 == len(rows):
                print(
                    json.dumps(
                        {
                            "stage": "depth",
                            "images": index + 1,
                            "total": len(rows),
                            "elapsed_seconds": time.perf_counter() - started,
                        }
                    ),
                    flush=True,
                )
    del model
    torch.cuda.empty_cache()
    depth = torch.stack(depth_rows)
    focal = torch.tensor(focal_rows, dtype=torch.float32)
    summary = {
        "images": len(rows),
        "depth_m": {
            "q10": float(torch.quantile(depth, 0.10)),
            "q50": float(torch.quantile(depth, 0.50)),
            "q90": float(torch.quantile(depth, 0.90)),
            "minimum": float(depth.min()),
            "maximum": float(depth.max()),
        },
        "focal_448_px": {
            "q10": float(torch.quantile(focal, 0.10)),
            "q50": float(torch.quantile(focal, 0.50)),
            "q90": float(torch.quantile(focal, 0.90)),
            "selfcalibration_reference": pilot.SELF_CAL_FOCAL_448_PX,
            "median_relative_difference": float(
                abs(
                    float(torch.median(focal))
                    - pilot.SELF_CAL_FOCAL_448_PX
                )
                / pilot.SELF_CAL_FOCAL_448_PX
            ),
        },
        "runtime_seconds": time.perf_counter() - started,
    }
    return depth, focal, summary


if __name__ == "__main__":
    pilot.infer_depth = infer_depth_fixed
    pilot.__file__ = str(Path(__file__).resolve())
    pilot.main()
