#!/usr/bin/env python3
"""Run NoMaD on an already captured four-frame snapshot.

This process reads image files and writes candidate trajectories only.  It
does not initialize Unitree channels, create ROS publishers, or issue motion
commands.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from v70_latency_protocol import StageTimer, write_json_atomic  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame-dir", required=True, type=Path)
    parser.add_argument("--goal-image", required=True, type=Path)
    parser.add_argument("--candidates", type=int, default=17)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/home/unitree/v70_nomad_snapshot_candidates.npy"),
    )
    parser.add_argument(
        "--latency-output",
        type=Path,
        help="Optional exact stage-timing JSON for the Orin benchmark.",
    )
    parser.add_argument(
        "--metadata-output",
        type=Path,
        help=(
            "Optional JSON containing the NoMaD temporal-distance estimate "
            "and immutable snapshot hashes."
        ),
    )
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    timer = StageTimer("nomad")
    backend_started_ns = time.perf_counter_ns()

    print("===== NoMaD saved-snapshot shadow =====")
    print("No camera channel is initialized.")
    print("No ROS publisher is created.")
    print("No SportClient command is issued.")
    print()

    from planning.nomad_backend import (
        NoMaDSampler,
        add_external_paths,
        load_config,
        load_nomad_model,
        transform_images,
    )

    assets_path = ROOT / "configs/local_assets.json"
    assets = json.loads(assets_path.read_text(encoding="utf-8"))

    def resolve(value: str) -> Path:
        return (assets_path.parent / value).resolve()

    nomad_assets = assets["nomad"]
    config_path = resolve(nomad_assets["config"])
    checkpoint_path = resolve(nomad_assets["checkpoint"])
    visualnav_root = resolve(nomad_assets["visualnav_root"])
    diffusion_policy_root = resolve(
        nomad_assets["diffusion_policy_root"]
    )

    for path in (
        config_path,
        checkpoint_path,
        visualnav_root,
        diffusion_policy_root,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    frame_dir = args.frame_dir.expanduser().resolve()
    goal_path = args.goal_image.expanduser().resolve()
    if not frame_dir.is_dir():
        raise FileNotFoundError(frame_dir)
    if not goal_path.is_file():
        raise FileNotFoundError(goal_path)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    add_external_paths(
        visualnav_root=visualnav_root,
        diffusion_policy_root=diffusion_policy_root,
    )
    config = load_config(config_path)
    timer.add_ms(
        "asset_and_backend_setup_ms",
        (time.perf_counter_ns() - backend_started_ns) / 1.0e6,
    )

    observation_count = int(config["context_size"]) + 1
    frame_paths = [
        frame_dir / f"history_{index}.jpg"
        for index in range(observation_count)
    ]
    missing = [str(path) for path in frame_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"snapshot is missing observation frames: {missing}"
        )

    preprocess_started_ns = time.perf_counter_ns()
    observation_images: list[Image.Image] = []
    for index, path in enumerate(frame_paths):
        with Image.open(path) as image:
            observation = image.convert("RGB").copy()
        observation_images.append(observation)
        print(
            f"frame {index}: {observation.width}x{observation.height}, "
            f"sha256={file_sha256(path)}"
        )

    with Image.open(goal_path) as image:
        goal_image = image.convert("RGB").copy()

    observation_tensor = transform_images(
        observation_images,
        config["image_size"],
    )
    goal_tensor = transform_images(
        [goal_image],
        config["image_size"],
    )
    timer.add_ms(
        "decode_and_transform_5_images_ms",
        (time.perf_counter_ns() - preprocess_started_ns) / 1.0e6,
    )

    print("goal sha256:", file_sha256(goal_path))
    print("observation tensor:", tuple(observation_tensor.shape))
    print("goal tensor:", tuple(goal_tensor.shape))
    print("Loading NoMaD...")

    model_load_started_ns = time.perf_counter_ns()
    model, load_report = load_nomad_model(
        checkpoint_path=checkpoint_path,
        config=config,
        device=device,
    )
    sampler = NoMaDSampler(model, config, device)

    torch.cuda.synchronize(device)
    timer.add_ms(
        "model_cuda_load_ms",
        (time.perf_counter_ns() - model_load_started_ns) / 1.0e6,
    )
    print("NoMaD load report:")
    print(json.dumps(load_report, indent=2))

    torch.cuda.synchronize()
    started = time.monotonic()
    candidates, temporal_distance = sampler.sample_with_distance(
        observation_tensor,
        goal_tensor,
        candidates=args.candidates,
        seed=args.seed,
    )
    torch.cuda.synchronize()
    elapsed = time.monotonic() - started

    timer.add_ms("candidate_generation_ms", elapsed * 1000.0)
    postprocess_started_ns = time.perf_counter_ns()
    candidates = np.asarray(candidates, dtype=np.float32)
    if candidates.ndim != 3 or candidates.shape[-1] != 2:
        raise RuntimeError(
            f"unexpected NoMaD candidate shape: {candidates.shape}"
        )
    if not np.isfinite(candidates).all():
        raise RuntimeError("NoMaD candidates contain NaN/Inf")

    output_path = args.output.expanduser().resolve()
    timer.add_ms(
        "candidate_validation_ms",
        (time.perf_counter_ns() - postprocess_started_ns) / 1.0e6,
    )
    if output_path.suffix != ".npy":
        raise ValueError("--output must end in .npy")
    save_started_ns = time.perf_counter_ns()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, candidates)
    if args.metadata_output is not None:
        write_json_atomic(
            args.metadata_output,
            {
                "schema": "v70_nomad_snapshot_metadata_v1",
                "goal_sha256": file_sha256(goal_path),
                "frame_sha256": [
                    file_sha256(path) for path in frame_paths
                ],
                "temporal_distance": float(temporal_distance),
                "candidates": int(args.candidates),
                "candidate_shape": list(candidates.shape),
                "seed": int(args.seed),
            },
        )
    timer.add_ms(
        "artifact_write_ms",
        (time.perf_counter_ns() - save_started_ns) / 1.0e6,
    )

    print()
    print("NoMaD saved-snapshot shadow: PASS")
    timer.details.update(
        {
            "candidates": int(args.candidates),
            "observation_images": int(observation_count),
            "candidate_shape": list(candidates.shape),
            "seed": int(args.seed),
            "temporal_distance": float(temporal_distance),
            "generation_scope": (
                "NoMaD vision/distance forward + diffusion candidate sampling"
            ),
            "model_load_excluded_from_candidate_generation_ms": True,
        }
    )
    if args.latency_output is not None:
        write_json_atomic(args.latency_output, timer.payload())

    print("candidate shape:", candidates.shape)
    print("temporal distance:", round(float(temporal_distance), 4))
    print("inference seconds:", round(elapsed, 3))
    print("saved:", output_path)
    if args.metadata_output is not None:
        print("metadata:", args.metadata_output)
    if args.latency_output is not None:
        print("latency:", args.latency_output)
    print("No action was executed.")


if __name__ == "__main__":
    main()
