#!/usr/bin/env python3
"""Evaluate a native six-image GNM/ViNT checkpoint with four unique images.

The underlying model is built with its native ``context_size=5``.  Evaluation
selects only ``t-3..t`` and left-pads the observation tensor by repeating the
earliest available image, producing ``[t-3, t-3, t-3, t-2, t-1, t]``.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


BASE_RUNNER = Path(__file__).with_name("run_visualnav_offline_eval.py")
spec = importlib.util.spec_from_file_location("visualnav_eval_base", BASE_RUNNER)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot load base evaluator: {BASE_RUNNER}")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)

native_load_model = base.load_visualnav_model
native_transform_images = base.transform_images


def load_native_model_then_limit_context(model_path, config, device, strict_load=True):
    if config["model_type"] not in {"gnm", "vint"}:
        raise ValueError("The strict-four-frame wrapper only supports GNM and ViNT")
    native_context_size = int(config["context_size"])
    if native_context_size != 5:
        raise ValueError(f"Expected a native six-image checkpoint, got context_size={native_context_size}")
    model, report = native_load_model(model_path, dict(config), device, strict_load)
    config["context_size"] = 3
    report["native_context_size"] = native_context_size
    report["evaluation_unique_context_images"] = 4
    report["context_padding"] = "left_repeat_earliest_twice"
    return model, report


def left_pad_four_images(pil_imgs, image_size, center_crop=False):
    images = list(pil_imgs)
    if len(images) == 4:
        images = [images[0], images[0], *images]
    return native_transform_images(images, image_size, center_crop)


def output_dir_from_argv() -> Path:
    try:
        index = sys.argv.index("--output-dir")
        return Path(sys.argv[index + 1])
    except (ValueError, IndexError) as exc:
        raise RuntimeError("--output-dir is required") from exc


base.load_visualnav_model = load_native_model_then_limit_context
base.transform_images = left_pad_four_images
base.main()

summary_path = output_dir_from_argv() / "summary.json"
summary = json.loads(summary_path.read_text())
summary["observation_protocol"] = {
    "unique_images": 4,
    "source_times": ["t-3", "t-2", "t-1", "t"],
    "model_input_times": ["t-3", "t-3", "t-3", "t-2", "t-1", "t"],
    "padding": "repeat earliest available image twice on the left",
}
summary_path.write_text(json.dumps(summary, indent=2) + "\n")
