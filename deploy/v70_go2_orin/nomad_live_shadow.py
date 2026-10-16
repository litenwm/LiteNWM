#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--goal-image", required=True)
    p.add_argument("--interface", default="enP8p1s0")
    p.add_argument("--candidates", type=int, default=17)
    p.add_argument("--seed", type=int, default=20260730)
    p.add_argument("--frame-interval", type=float, default=0.20)
    p.add_argument(
        "--output",
        default="/home/unitree/v70_nomad_live_candidates.npy",
    )
    return p.parse_args()


def main():
    args = parse_args()

    # 这里只读取相机和运行模型，完全不创建 ROS publisher。
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    from unitree_sdk2py.go2.video.video_client import VideoClient

    from src.planning.nomad_backend import (
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

    goal_path = Path(args.goal_image).expanduser().resolve()
    if not goal_path.exists():
        raise FileNotFoundError(goal_path)

    device = torch.device("cuda:0")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")

    torch.cuda.set_device(0)

    add_external_paths(
        visualnav_root=visualnav_root,
        diffusion_policy_root=diffusion_policy_root,
    )

    config = load_config(config_path)

    print("Loading NoMaD...")
    model, load_report = load_nomad_model(
        checkpoint_path=checkpoint_path,
        config=config,
        device=device,
    )
    sampler = NoMaDSampler(model, config, device)

    print("NoMaD load report:")
    print(json.dumps(load_report, indent=2))

    ChannelFactoryInitialize(0, args.interface)

    client = VideoClient()
    client.SetTimeout(5.0)
    client.Init()

    # NoMaD context_size=3 表示当前帧加前三帧，共4帧。
    observation_count = int(config["context_size"]) + 1
    observation_images: list[Image.Image] = []

    print(f"Capturing {observation_count} frames from {args.interface}...")

    for index in range(observation_count):
        code, data = client.GetImageSample()
        if code != 0:
            raise RuntimeError(
                f"GetImageSample failed at frame {index}: code={code}"
            )

        raw = bytes(data)
        if not raw:
            raise RuntimeError(f"empty camera payload at frame {index}")

        bgr = cv2.imdecode(
            np.frombuffer(raw, dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )
        if bgr is None:
            raise RuntimeError(f"JPEG decode failed at frame {index}")

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        observation_images.append(Image.fromarray(rgb))

        print(
            f"frame {index}: "
            f"{rgb.shape[1]}x{rgb.shape[0]}, bytes={len(raw)}"
        )

        if index + 1 < observation_count:
            time.sleep(max(0.0, args.frame_interval))

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

    print("observation tensor:", tuple(observation_tensor.shape))
    print("goal tensor:", tuple(goal_tensor.shape))

    torch.cuda.synchronize()
    start = time.monotonic()

    candidates = sampler.sample(
        observation_tensor,
        goal_tensor,
        candidates=args.candidates,
        seed=args.seed,
    )

    torch.cuda.synchronize()
    elapsed = time.monotonic() - start

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, candidates)

    print()
    print("NoMaD live shadow: PASS")
    print("candidate shape:", candidates.shape)
    print("inference seconds:", round(elapsed, 3))
    print("saved:", output_path)
    print()
    print("First five H4 candidate trajectories:")

    for index in range(min(5, len(candidates))):
        print(f"candidate {index}:")
        print(np.array2string(
            candidates[index, :4],
            precision=4,
            suppress_small=True,
        ))


if __name__ == "__main__":
    main()
