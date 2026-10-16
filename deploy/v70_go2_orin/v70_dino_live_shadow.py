#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from planning.checkpoint_io import load_checkpoint
from recon_latents.dinov3_adapter import (  # noqa: E402
    DinoV3AdapterConfig,
    DinoV3TokenAdapter,
    extract_dinov3_patch_tokens,
)
from tools.evaluation.export_recon_dense_dinov3_features_v1 import (  # noqa: E402
    pool_static_features,
)
from v70_latency_protocol import StageTimer, write_json_atomic  # noqa: E402

HISTORY = 4
TOKEN_COUNT = 256
LATENT_DIM = 384


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--goal-image", required=True, type=Path)
    parser.add_argument("--interface", default="enP8p1s0")
    parser.add_argument("--frame-interval", type=float, default=0.20)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument(
        "--adapter-checkpoint",
        type=Path,
        default=(
            ROOT
            / "checkpoints/project/dinov3_adapter/"
            "adapter_pilot_v1_final.pt"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/home/unitree/v70_dino_live_shadow.npz"),
    )
    parser.add_argument(
        "--frame-output-dir",
        type=Path,
        default=Path("/home/unitree/v70_dino_live_frames"),
    )
    parser.add_argument(
        "--latency-output",
        type=Path,
        help="Optional exact stage-timing JSON for the Orin benchmark.",
    )
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def iter_string_values(value: Any):
    if isinstance(value, dict):
        for child in value.values():
            yield from iter_string_values(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from iter_string_values(child)
    elif isinstance(value, str):
        yield value


def is_dinov3_model_dir(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / "config.json").is_file()
        and (path / "model.safetensors").is_file()
    )


def discover_model_dir(explicit: Path | None) -> Path:
    if explicit is not None:
        result = explicit.expanduser().resolve()
        if not is_dinov3_model_dir(result):
            raise FileNotFoundError(
                f"invalid DINOv3 model directory: {result}"
            )
        return result

    candidates: list[Path] = [
        ROOT / "external/dinov3-vitl16-pretrain-lvd1689m",
        ROOT / "models/facebook/dinov3-vitl16-pretrain-lvd1689m",
        Path(
            "/home/unitree/models/facebook/"
            "dinov3-vitl16-pretrain-lvd1689m"
        ),
    ]

    assets_path = ROOT / "configs/local_assets.json"
    if assets_path.is_file():
        assets = json.loads(assets_path.read_text(encoding="utf-8"))
        for value in iter_string_values(assets):
            if "dinov3" not in value.lower():
                continue

            raw = Path(value).expanduser()
            if not raw.is_absolute():
                raw = assets_path.parent / raw
            raw = raw.resolve()

            candidates.append(raw)
            if raw.is_file():
                candidates.append(raw.parent)

    for candidate in candidates:
        if is_dinov3_model_dir(candidate):
            return candidate

    # 最后只在当前 release 内查找，不访问网络。
    for weight in ROOT.rglob("model.safetensors"):
        parent = weight.parent
        if "dinov3" in str(parent).lower() and is_dinov3_model_dir(parent):
            return parent.resolve()

    raise FileNotFoundError(
        "没有自动找到 DINOv3 本地模型目录。"
        "请用 --model-dir 指定包含 config.json 和 "
        "model.safetensors 的目录。"
    )


def preprocess_pil(image: Image.Image) -> torch.Tensor:
    image = image.convert("RGB").resize(
        (224, 224),
        Image.Resampling.BICUBIC,
    )

    array = np.asarray(image, dtype=np.float32) / 255.0
    array = (
        array
        - np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
    ) / np.asarray((0.229, 0.224, 0.225), dtype=np.float32)

    return torch.from_numpy(
        np.ascontiguousarray(array.transpose(2, 0, 1))
    )


def capture_images(
    interface: str,
    interval: float,
    output_dir: Path,
) -> list[Image.Image]:
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    from unitree_sdk2py.go2.video.video_client import VideoClient

    ChannelFactoryInitialize(0, interface)

    client = VideoClient()
    client.SetTimeout(5.0)
    client.Init()

    output_dir.mkdir(parents=True, exist_ok=True)
    images: list[Image.Image] = []

    print(f"Capturing {HISTORY} frames from {interface}...")

    for index in range(HISTORY):
        code, data = client.GetImageSample()
        if code != 0:
            raise RuntimeError(
                f"GetImageSample failed at frame {index}: code={code}"
            )

        raw = bytes(data)
        if not raw:
            raise RuntimeError(f"empty camera frame {index}")

        bgr = cv2.imdecode(
            np.frombuffer(raw, dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )
        if bgr is None:
            raise RuntimeError(f"JPEG decode failed at frame {index}")

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        images.append(image)

        image.save(output_dir / f"history_{index}.jpg", quality=95)

        print(
            f"frame {index}: "
            f"{rgb.shape[1]}x{rgb.shape[0]}, "
            f"bytes={len(raw)}"
        )

        if index + 1 < HISTORY:
            time.sleep(max(0.0, interval))

    return images


def load_backbone(model_dir: Path, device: torch.device):
    from transformers import AutoModel

    try:
        backbone = AutoModel.from_pretrained(
            model_dir,
            local_files_only=True,
            dtype=torch.float32,
        )
    except TypeError:
        # 兼容旧版 transformers。
        backbone = AutoModel.from_pretrained(
            model_dir,
            local_files_only=True,
            torch_dtype=torch.float32,
        )

    return (
        backbone.eval()
        .to(device)
        .requires_grad_(False)
    )


def main() -> None:
    args = parse_args()

    timer = StageTimer("dinov3")
    print("===== V70 DINOv3 live shadow =====")
    print("No ROS publisher is created.")
    print("No SportClient command is issued.")
    print()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")

    asset_started_ns = time.perf_counter_ns()
    goal_path = args.goal_image.expanduser().resolve()
    if not goal_path.is_file():
        raise FileNotFoundError(goal_path)

    adapter_path = args.adapter_checkpoint.expanduser().resolve()
    if not adapter_path.is_file():
        raise FileNotFoundError(adapter_path)

    model_dir = discover_model_dir(args.model_dir)
    weight_path = model_dir / "model.safetensors"

    print("DINOv3 model:", model_dir)
    print("adapter:", adapter_path)
    print("goal image:", goal_path)

    adapter_payload = load_checkpoint(
        adapter_path,
        map_location="cpu",
        weights_only=False,
    )

    expected_hash = adapter_payload.get("dinov3_weight_sha256")
    if expected_hash:
        print("Checking DINOv3 weight SHA256...")
        actual_hash = file_sha256(weight_path)
        if actual_hash != expected_hash:
            raise RuntimeError(
                "adapter/DINOv3 weight mismatch:\n"
                f"expected={expected_hash}\n"
                f"actual={actual_hash}"
            )
        print("DINOv3 weight binding: PASS")

    adapter = DinoV3TokenAdapter(
        DinoV3AdapterConfig(**adapter_payload["config"])
    )
    adapter.load_state_dict(adapter_payload["model"], strict=True)
    timer.add_ms(
        "asset_validation_and_adapter_cpu_load_ms",
        (time.perf_counter_ns() - asset_started_ns) / 1.0e6,
    )

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    capture_started_ns = time.perf_counter_ns()
    history_images = capture_images(
        interface=args.interface,
        interval=args.frame_interval,
        output_dir=args.frame_output_dir,
    )

    timer.add_ms(
        "camera_capture_4_frames_ms",
        (time.perf_counter_ns() - capture_started_ns) / 1.0e6,
    )
    preprocess_started_ns = time.perf_counter_ns()
    with Image.open(goal_path) as image:
        goal_image = image.convert("RGB").copy()

    args.frame_output_dir.mkdir(parents=True, exist_ok=True)
    goal_image.save(args.frame_output_dir / "goal.jpg", quality=95)

    image_batch = torch.stack(
        [preprocess_pil(image) for image in history_images]
        + [preprocess_pil(goal_image)]
    )

    if image_batch.shape != (5, 3, 224, 224):
        raise RuntimeError(
            f"preprocess shape changed: {tuple(image_batch.shape)}"
        )

    print("Loading DINOv3 backbone...")
    timer.add_ms(
        "decode_goal_and_preprocess_5_images_ms",
        (time.perf_counter_ns() - preprocess_started_ns) / 1.0e6,
    )
    model_load_started_ns = time.perf_counter_ns()
    backbone = load_backbone(model_dir, device)
    adapter = (
        adapter.eval()
        .to(device)
        .requires_grad_(False)
    )

    image_batch = image_batch.to(device)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    timer.add_ms(
        "backbone_and_adapter_cuda_load_ms",
        (time.perf_counter_ns() - model_load_started_ns) / 1.0e6,
    )
    torch.cuda.synchronize(device)
    started = time.monotonic()

    with torch.inference_mode():
        hidden = backbone(
            pixel_values=image_batch
        ).last_hidden_state

        patch_tokens = extract_dinov3_patch_tokens(hidden)
        adapted = adapter(patch_tokens).float()

        # Adapter may preserve the 16x16 spatial token grid:
        # [B,16,16,384] -> [B,256,384].
        if adapted.ndim == 4:
            if adapted.shape[1:3] == (16, 16) and adapted.shape[-1] == LATENT_DIM:
                adapted = adapted.reshape(
                    adapted.shape[0],
                    TOKEN_COUNT,
                    LATENT_DIM,
                )
            elif adapted.shape[1] == LATENT_DIM and adapted.shape[2:] == (16, 16):
                adapted = adapted.permute(0, 2, 3, 1).contiguous().reshape(
                    adapted.shape[0],
                    TOKEN_COUNT,
                    LATENT_DIM,
                )
            else:
                raise RuntimeError(
                    f"unsupported adapter grid shape: {tuple(adapted.shape)}"
                )

    torch.cuda.synchronize(device)
    elapsed = time.monotonic() - started
    timer.add_ms("encode_5_images_ms", elapsed * 1000.0)
    postprocess_started_ns = time.perf_counter_ns()

    if adapted.shape != (5, TOKEN_COUNT, LATENT_DIM):
        raise RuntimeError(
            f"adapted latent shape changed: {tuple(adapted.shape)}"
        )

    history = adapted[:HISTORY].unsqueeze(0)
    goal = adapted[HISTORY].unsqueeze(0)

    static = pool_static_features(history, goal)
    current = history[:, -1]
    goal_energy = goal.square().mean(dim=(-1, -2)).clamp_min(1.0e-8)
    current_goal_nmse = (
        (current - goal).square().mean(dim=(-1, -2)) / goal_energy
    )
    current_goal_cosine_distance = 1.0 - torch.cosine_similarity(
        current.flatten(1),
        goal.flatten(1),
        dim=-1,
    )
    current_goal_norm_ratio = (
        torch.linalg.vector_norm(current, dim=(-1, -2))
        / torch.linalg.vector_norm(goal, dim=(-1, -2)).clamp_min(1.0e-8)
    )

    if history.shape != (1, 4, 256, 384):
        raise RuntimeError(f"history shape changed: {history.shape}")
    if goal.shape != (1, 256, 384):
        raise RuntimeError(f"goal shape changed: {goal.shape}")
    if static.shape != (1, 210):
        raise RuntimeError(f"static shape changed: {static.shape}")

    arrays = (
        history,
        goal,
        static,
        current_goal_nmse,
        current_goal_cosine_distance,
        current_goal_norm_ratio,
    )
    if not all(bool(torch.isfinite(value).all()) for value in arrays):
        raise RuntimeError("DINOv3 output contains NaN or Inf")
    torch.cuda.synchronize(device)
    timer.add_ms(
        "latent_pool_and_validation_ms",
        (time.perf_counter_ns() - postprocess_started_ns) / 1.0e6,
    )


    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_started_ns = time.perf_counter_ns()
    np.savez_compressed(
        args.output,
        history_latent=history.cpu().numpy().astype(np.float16),
        goal_latent=goal.cpu().numpy().astype(np.float16),
        static_features=static.cpu().numpy().astype(np.float32),
        current_goal_nmse=(
            current_goal_nmse.cpu().numpy().astype(np.float32)
        ),
        current_goal_cosine_distance=(
            current_goal_cosine_distance.cpu().numpy().astype(np.float32)
        ),
        current_goal_norm_ratio=(
            current_goal_norm_ratio.cpu().numpy().astype(np.float32)
        ),
        image_shape=np.asarray(image_batch.shape, dtype=np.int64),
        inference_seconds=np.asarray(elapsed, dtype=np.float64),
    )

    timer.add_ms(
        "artifact_write_ms",
        (time.perf_counter_ns() - save_started_ns) / 1.0e6,
    )

    peak_gib = torch.cuda.max_memory_allocated(device) / (1024 ** 3)

    print()
    timer.details.update(
        {
            "batch_images": 5,
            "history_images": HISTORY,
            "input_shape": list(image_batch.shape),
            "encoding_scope": (
                "DINOv3 backbone + patch-token extraction + trained adapter"
            ),
            "model_load_excluded_from_encode_5_images_ms": True,
            "camera_interval_seconds": float(args.frame_interval),
            "current_goal_nmse": float(current_goal_nmse.item()),
            "current_goal_cosine_distance": float(
                current_goal_cosine_distance.item()
            ),
            "current_goal_norm_ratio": float(
                current_goal_norm_ratio.item()
            ),
            "peak_cuda_allocated_gib": float(peak_gib),
        }
    )
    if args.latency_output is not None:
        write_json_atomic(args.latency_output, timer.payload())

    print("===== RESULT =====")
    print("input images:", tuple(image_batch.shape))
    print("backbone hidden:", tuple(hidden.shape))
    print("patch tokens:", tuple(patch_tokens.shape))
    print("adapted tokens:", tuple(adapted.shape))
    print("history latent:", tuple(history.shape))
    print("goal latent:", tuple(goal.shape))
    print("current-goal NMSE:", round(float(current_goal_nmse.item()), 5))
    print(
        "current-goal cosine distance:",
        round(float(current_goal_cosine_distance.item()), 5),
    )
    print("current-goal norm ratio:", round(float(current_goal_norm_ratio.item()), 5))
    print("static features:", tuple(static.shape))
    print("finite: True")
    print("inference seconds:", round(elapsed, 3))
    print("peak CUDA allocated GiB:", round(peak_gib, 3))
    print("saved:", args.output)
    print("frames:", args.frame_output_dir)
    if args.latency_output is not None:
        print("latency:", args.latency_output)
    print()
    print("V70 DINO LIVE SHADOW: PASS")


if __name__ == "__main__":
    main()
