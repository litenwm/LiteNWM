#!/usr/bin/env python3
"""Paired raw-resolution versus legacy-320 dual-latent pilot.

This bounded development experiment uses three local SCAND ROS bags.  The
same sampled frames and pose-derived actions are encoded through:

* legacy320: raw -> 4:3 crop -> 320x240 JPEG -> 448x336
* raw448:    raw -> 4:3 crop -> 448x336 JPEG

Frozen DINOv3 representations and identically initialized LTWMs are compared
on one held-route bag.  No navigation outcomes, CEM tables, or sealed data are
loaded.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict, dataclass
import hashlib
import io
import json
import math
from pathlib import Path
import random
import sys
import time
from typing import Any

import numpy as np
from PIL import Image, ImageOps
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(Path(__file__).resolve().parent)]

from process_scand_bags_rosbags import (  # noqa: E402
    IMAGE_TOPICS,
    ODOM_TOPICS,
    choose_topic,
    forward_segments,
    quaternion_yaw,
)
from recon_latents.actions import derive_continuous_actions  # noqa: E402
from recon_latents.dinov3_adapter import (  # noqa: E402
    DinoV3AdapterConfig,
    DinoV3TokenAdapter,
)
from recon_latents.dinov3_multilayer_dual import (  # noqa: E402
    FrozenMultiLayerDualLatentEncoder,
    MultiLayerDualLatentConfig,
    imagenet_normalize,
)
from recon_latents.losses_highres_dual_action_residual_stable_v1 import (  # noqa: E402
    per_candidate_dual_nmse,
)
from recon_latents.losses_highres_dual_stable_v1 import (  # noqa: E402
    HighResDualLatentLossWeights,
    compute_highres_dual_latent_loss,
)
from recon_latents.models_highres_dual_v1 import (  # noqa: E402
    HighResDualLatentWMConfig,
    HighResMultiLayerDualLatentWM,
)
from train_highres_multilayer_dual_ltwm_v1 import corrupt_actions  # noqa: E402


DEFAULT_OUTPUT = (
    ROOT / "outputs/highres_dual_ltwm_v1/raw_resolution_paired_pilot_v1"
)
MODEL_DIR = Path("/home/rainbow/models/facebook/dinov3-vitl16-pretrain-lvd1689m")
ADAPTER = ROOT / "checkpoints/project/dinov3_adapter/adapter_pilot_v1_final.pt"


@dataclass(frozen=True)
class BagSpec:
    trajectory_id: str
    path: Path
    split: str


DEFAULT_BAGS = (
    BagSpec(
        "rawpilot_B_Spot_RLM_RLM_Wed_Nov_10_44",
        Path("/mnt/d/VEDIOS/B_Spot_RLM_RLM_Wed_Nov_10_44.bag"),
        "train",
    ),
    BagSpec(
        "rawpilot_A_Spot_NHB_Ahg_Wed_Nov_10_55",
        Path("/mnt/d/VEDIOS/A_Spot_NHB_Ahg_Wed_Nov_10_55.bag"),
        "train",
    ),
    BagSpec(
        "rawpilot_A_Spot_Thompson_Butler_Sat_Nov_13_103",
        Path("/mnt/d/VEDIOS/A_Spot_Thompson_Butler_Sat_Nov_13_103.bag"),
        "val",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sample-rate", type=float, default=4.0)
    parser.add_argument("--encode-batch-size", type=int, default=4)
    parser.add_argument("--train-steps", type=int, default=500)
    parser.add_argument("--train-batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026072403)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--amp", choices=("bf16", "fp16", "off"), default="bf16")
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(canonical_json(value) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_npz_atomic(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def amp_context(amp: str):
    if amp == "off":
        return nullcontext()
    dtype = torch.bfloat16 if amp == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def center_crop_4_3(image: Image.Image) -> Image.Image:
    width, height = image.size
    target_width = int(round(height * 4.0 / 3.0))
    if target_width < width:
        left = (width - target_width) // 2
        return image.crop((left, 0, left + target_width, height))
    if target_width > width:
        target_height = int(round(width * 3.0 / 4.0))
        top = max(0, (height - target_height) // 2)
        return image.crop((0, top, width, top + target_height))
    return image


def jpeg_roundtrip(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    resized = image.resize(size, Image.Resampling.BICUBIC)
    payload = io.BytesIO()
    resized.save(payload, format="JPEG", quality=95)
    payload.seek(0)
    with Image.open(payload) as decoded:
        return decoded.convert("RGB").copy()


def paired_views(image: Image.Image) -> tuple[Image.Image, Image.Image]:
    cropped = center_crop_4_3(image.convert("RGB"))
    legacy = jpeg_roundtrip(cropped, (320, 240))
    legacy = ImageOps.fit(
        legacy,
        (448, 336),
        method=Image.Resampling.BICUBIC,
        centering=(0.5, 0.5),
    )
    raw = jpeg_roundtrip(cropped, (448, 336))
    return legacy, raw


def read_longest_forward_segment(
    bag_path: Path,
    sample_rate: float,
) -> tuple[list[tuple[Image.Image, np.ndarray, float, int]], dict[str, Any]]:
    from rosbags.highlevel import AnyReader
    from rosbags.typesys import Stores, get_typestore

    typestore = get_typestore(Stores.ROS1_NOETIC)
    with AnyReader([bag_path], default_typestore=typestore) as reader:
        image_topic = choose_topic(reader.connections, IMAGE_TOPICS)
        odom_topic = choose_topic(reader.connections, ODOM_TOPICS)
        selected = [
            connection
            for connection in reader.connections
            if connection.topic in {image_topic, odom_topic}
        ]
        latest_image: Image.Image | None = None
        latest_position: np.ndarray | None = None
        latest_yaw: float | None = None
        next_sample_ns: int | None = None
        samples: list[tuple[Image.Image, np.ndarray, float, int]] = []
        source_sizes: dict[str, int] = {}
        for connection, timestamp, rawdata in reader.messages(connections=selected):
            message = reader.deserialize(rawdata, connection.msgtype)
            if connection.topic == image_topic:
                with Image.open(io.BytesIO(bytes(message.data))) as decoded:
                    latest_image = decoded.convert("RGB").copy()
                key = f"{latest_image.width}x{latest_image.height}"
                source_sizes[key] = source_sizes.get(key, 0) + 1
            else:
                position = message.pose.pose.position
                latest_position = np.asarray(
                    [float(position.x), float(position.y)],
                    dtype=np.float32,
                )
                latest_yaw = quaternion_yaw(message.pose.pose.orientation)
            if next_sample_ns is None:
                next_sample_ns = int(timestamp)
            if (
                int(timestamp) >= next_sample_ns
                and latest_image is not None
                and latest_position is not None
                and latest_yaw is not None
            ):
                samples.append(
                    (
                        latest_image.copy(),
                        latest_position.copy(),
                        float(latest_yaw),
                        int(timestamp),
                    )
                )
                next_sample_ns = int(timestamp + round(1e9 / sample_rate))
    segments = forward_segments(samples, 1)
    require(bool(segments), f"no forward segment in {bag_path}")
    ranked = sorted(
        enumerate(segments),
        key=lambda item: (-len(item[1]), item[0]),
    )
    selected_index, segment = ranked[0]
    require(len(segment) >= 8, f"segment has no LTWM window: {bag_path}")
    return segment, {
        "image_topic": image_topic,
        "odom_topic": odom_topic,
        "sample_count": len(samples),
        "forward_segment_count": len(segments),
        "selected_segment_index": selected_index,
        "selected_segment_frames": len(segment),
        "discarded_segment_lengths": [
            len(value)
            for index, value in enumerate(segments)
            if index != selected_index
        ],
        "source_resolution_message_counts": source_sizes,
    }


def image_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1)))
    return imagenet_normalize(tensor)


def load_representation(
    device: torch.device,
) -> tuple[Any, FrozenMultiLayerDualLatentEncoder, dict[str, Any]]:
    adapter_payload = torch.load(ADAPTER, map_location="cpu", weights_only=False)
    adapter = DinoV3TokenAdapter(
        DinoV3AdapterConfig(**adapter_payload["config"])
    )
    adapter.load_state_dict(adapter_payload["model"], strict=True)
    config = MultiLayerDualLatentConfig()
    representation = FrozenMultiLayerDualLatentEncoder(adapter, config)
    from transformers import AutoModel

    backbone = AutoModel.from_pretrained(
        MODEL_DIR,
        local_files_only=True,
        dtype=torch.float32,
    ).eval().to(device).requires_grad_(False)
    representation = representation.eval().to(device).requires_grad_(False)
    lineage = {
        "dinov3_model": str(MODEL_DIR),
        "dinov3_sha256": sha256_file(MODEL_DIR / "model.safetensors"),
        "adapter": str(ADAPTER),
        "adapter_sha256": sha256_file(ADAPTER),
        "representation": representation.contract_dict(),
    }
    return backbone, representation, lineage


def encode_view(
    images: list[Image.Image],
    backbone: Any,
    representation: FrozenMultiLayerDualLatentEncoder,
    *,
    batch_size: int,
    device: torch.device,
    amp: str,
) -> tuple[np.ndarray, np.ndarray]:
    semantic_rows = []
    geometry_rows = []
    config = representation.config
    for start in range(0, len(images), batch_size):
        batch = torch.stack(
            [image_tensor(image) for image in images[start : start + batch_size]]
        ).to(device)
        with torch.inference_mode(), amp_context(amp):
            hidden = backbone(
                pixel_values=batch,
                output_hidden_states=True,
            ).hidden_states
            encoded = representation(hidden)
        semantic_rows.append(
            encoded["semantic_tokens"]
            .reshape(
                -1,
                config.grid_height,
                config.grid_width,
                config.semantic_dim,
            )
            .float()
            .cpu()
            .numpy()
            .astype(np.float16)
        )
        geometry_rows.append(
            encoded["geometry_tokens"]
            .reshape(
                -1,
                config.grid_height,
                config.grid_width,
                config.geometry_dim,
            )
            .float()
            .cpu()
            .numpy()
            .astype(np.float16)
        )
    return np.concatenate(semantic_rows), np.concatenate(geometry_rows)


def prepare_latents(
    output: Path,
    specs: tuple[BagSpec, ...],
    *,
    backbone: Any,
    representation: FrozenMultiLayerDualLatentEncoder,
    sample_rate: float,
    batch_size: int,
    device: torch.device,
    amp: str,
) -> tuple[dict[str, dict[str, Path]], list[dict[str, Any]]]:
    paths: dict[str, dict[str, Path]] = {"legacy320": {}, "raw448": {}}
    rows = []
    for spec in specs:
        started = time.perf_counter()
        segment, audit = read_longest_forward_segment(spec.path, sample_rate)
        legacy_images = []
        raw_images = []
        for image, _, _, _ in segment:
            legacy, raw = paired_views(image)
            legacy_images.append(legacy)
            raw_images.append(raw)
        positions = np.stack([value[1] for value in segment]).astype(np.float32)
        yaw = np.asarray([value[2] for value in segment], dtype=np.float32)
        timestamps = np.asarray([value[3] for value in segment], dtype=np.int64)
        actions = derive_continuous_actions(positions, yaw)
        mode_images = {
            "legacy320": legacy_images,
            "raw448": raw_images,
        }
        for mode, images in mode_images.items():
            semantic, geometry = encode_view(
                images,
                backbone,
                representation,
                batch_size=batch_size,
                device=device,
                amp=amp,
            )
            shard = output / "latents" / mode / f"{spec.trajectory_id}.npz"
            write_npz_atomic(
                shard,
                semantic_latents=semantic,
                geometry_latents=geometry,
                actions_continuous=actions,
                positions=positions,
                yaw=yaw,
                timestamps_ns=timestamps,
                trajectory_id=np.asarray(spec.trajectory_id),
                split=np.asarray(spec.split),
            )
            paths[mode][spec.trajectory_id] = shard
        rows.append(
            {
                "trajectory_id": spec.trajectory_id,
                "split": spec.split,
                "bag": str(spec.path),
                "bag_sha256": sha256_file(spec.path),
                "frames": len(segment),
                "windows": len(segment) - 7,
                **audit,
                "runtime_seconds": time.perf_counter() - started,
            }
        )
        print(
            canonical_json(
                {
                    "stage": "encoded_paired_bag",
                    "trajectory": spec.trajectory_id,
                    "split": spec.split,
                    "frames": len(segment),
                    "windows": len(segment) - 7,
                    "runtime_seconds": rows[-1]["runtime_seconds"],
                }
            ),
            flush=True,
        )
    return paths, rows


@dataclass(frozen=True)
class WindowRef:
    trajectory_id: str
    t: int


def window_refs(
    paths: dict[str, Path],
    split_by_trajectory: dict[str, str],
    split: str,
) -> tuple[list[WindowRef], dict[str, list[int]]]:
    refs = []
    by_trajectory: dict[str, list[int]] = {}
    for trajectory_id in sorted(paths):
        if split_by_trajectory[trajectory_id] != split:
            continue
        with np.load(paths[trajectory_id], allow_pickle=False) as shard:
            frames = len(shard["semantic_latents"])
        indices = []
        for t in range(3, frames - 4):
            indices.append(len(refs))
            refs.append(WindowRef(trajectory_id, t))
        by_trajectory[trajectory_id] = indices
    return refs, by_trajectory


class ShardCache:
    def __init__(self, paths: dict[str, Path]) -> None:
        self.values = {}
        for trajectory_id, path in paths.items():
            with np.load(path, allow_pickle=False) as shard:
                self.values[trajectory_id] = {
                    "semantic": shard["semantic_latents"].astype(
                        np.float32, copy=True
                    ),
                    "geometry": shard["geometry_latents"].astype(
                        np.float32, copy=True
                    ),
                    "actions": shard["actions_continuous"].astype(
                        np.float32, copy=True
                    ),
                }

    def batch(
        self,
        refs: list[WindowRef],
    ) -> tuple[torch.Tensor, ...]:
        semantic_history = []
        geometry_history = []
        semantic_future = []
        geometry_future = []
        actions = []
        for ref in refs:
            value = self.values[ref.trajectory_id]
            t = ref.t
            semantic_history.append(
                value["semantic"][t - 3 : t + 1].reshape(4, 588, 384)
            )
            geometry_history.append(
                value["geometry"][t - 3 : t + 1].reshape(4, 588, 192)
            )
            semantic_future.append(
                value["semantic"][t + 1 : t + 5].reshape(4, 588, 384)
            )
            geometry_future.append(
                value["geometry"][t + 1 : t + 5].reshape(4, 588, 192)
            )
            actions.append(value["actions"][t : t + 4])
        return tuple(
            torch.from_numpy(np.stack(rows))
            for rows in (
                semantic_history,
                geometry_history,
                semantic_future,
                geometry_future,
                actions,
            )
        )


def route_balanced_schedule(
    refs_by_trajectory: dict[str, list[int]],
    *,
    steps: int,
    batch_size: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    trajectories = sorted(refs_by_trajectory)
    require(bool(trajectories), "empty training trajectories")
    schedule = np.empty((steps, batch_size), dtype=np.int64)
    for step in range(steps):
        for column in range(batch_size):
            trajectory = trajectories[(step * batch_size + column) % len(trajectories)]
            choices = refs_by_trajectory[trajectory]
            schedule[step, column] = choices[int(rng.integers(0, len(choices)))]
    return schedule


def train_model(
    cache: ShardCache,
    refs: list[WindowRef],
    schedule: np.ndarray,
    *,
    seed: int,
    device: torch.device,
    amp: str,
) -> tuple[HighResMultiLayerDualLatentWM, dict[str, Any]]:
    set_seed(seed)
    config = HighResDualLatentWMConfig()
    model = HighResMultiLayerDualLatentWM(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=2.0e-4,
        weight_decay=1.0e-4,
    )
    rows = []
    started = time.perf_counter()
    model.train()
    for step, indices in enumerate(schedule, start=1):
        batch = cache.batch([refs[int(index)] for index in indices])
        semantic_history, geometry_history, semantic_future, geometry_future, actions = (
            value.to(device, non_blocking=True) for value in batch
        )
        optimizer.zero_grad(set_to_none=True)
        with amp_context(amp):
            output = model(semantic_history, geometry_history, actions)
            corrupted_output = None
            if (step - 1) % 4 == 0:
                corrupted_output = model(
                    semantic_history,
                    geometry_history,
                    corrupt_actions(actions, (step - 1) // 4),
                )
            loss, components = compute_highres_dual_latent_loss(
                output,
                semantic_future,
                geometry_future,
                semantic_history,
                geometry_history,
                actions,
                weights=HighResDualLatentLossWeights(),
                corrupted_output=corrupted_output,
            )
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        rows.append(
            {
                "loss": float(loss.detach()),
                "gradient_norm": float(gradient_norm),
                **{
                    key: float(value.detach())
                    for key, value in components.items()
                },
            }
        )
        if step % 100 == 0 or step == len(schedule):
            print(
                canonical_json(
                    {
                        "stage": "train",
                        "step": step,
                        "steps": len(schedule),
                        "loss_last_100": float(
                            np.mean([row["loss"] for row in rows[-100:]])
                        ),
                        "elapsed_seconds": time.perf_counter() - started,
                    }
                ),
                flush=True,
            )
    return model, {
        "config": asdict(config),
        "steps": len(schedule),
        "batch_size": int(schedule.shape[1]),
        "learning_rate": 2.0e-4,
        "weight_decay": 1.0e-4,
        "loss_weights": asdict(HighResDualLatentLossWeights()),
        "mean_loss": float(np.mean([row["loss"] for row in rows])),
        "last_100_loss": float(
            np.mean([row["loss"] for row in rows[-100:]])
        ),
        "mean_gradient_norm": float(
            np.mean([row["gradient_norm"] for row in rows])
        ),
        "parameter_count": int(sum(value.numel() for value in model.parameters())),
        "runtime_seconds": time.perf_counter() - started,
    }


def h4_nmse(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    return (
        (prediction.float() - target.float()).square().mean(dim=(-1, -2))
        / target.float().square().mean(dim=(-1, -2)).clamp_min(1.0e-8)
    )


def validation_negatives(
    actions: torch.Tensor,
    action_bank: torch.Tensor,
    start: int,
) -> torch.Tensor:
    batch = len(actions)
    observed = []
    for offset in range(1, 8):
        index = (
            torch.arange(start, start + batch, device=actions.device) + offset
        ) % len(action_bank)
        observed.append(action_bank[index])
    reverse = actions.flip(1)
    yaw_flip = actions.clone()
    yaw_flip[..., 2].neg_()
    lateral_flip = actions.clone()
    lateral_flip[..., 1].neg_()
    magnitude = actions.clone()
    magnitude[..., :2] *= 0.5
    magnitude[..., 2] *= 1.5
    return torch.cat(
        (
            torch.stack(observed, dim=1),
            torch.stack((reverse, yaw_flip, lateral_flip, magnitude), dim=1),
        ),
        dim=1,
    )


@torch.no_grad()
def evaluate_model(
    model: HighResMultiLayerDualLatentWM,
    cache: ShardCache,
    refs: list[WindowRef],
    *,
    batch_size: int,
    device: torch.device,
    amp: str,
) -> dict[str, Any]:
    model.eval()
    all_actions = []
    for start in range(0, len(refs), batch_size):
        all_actions.append(cache.batch(refs[start : start + batch_size])[-1])
    action_bank = torch.cat(all_actions).to(device)
    semantic_h4 = []
    geometry_h4 = []
    nochange_semantic_h4 = []
    nochange_geometry_h4 = []
    correct_errors = []
    negative_errors = []
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    for start in range(0, len(refs), batch_size):
        selected = refs[start : start + batch_size]
        batch = cache.batch(selected)
        semantic_history, geometry_history, semantic_future, geometry_future, actions = (
            value.to(device, non_blocking=True) for value in batch
        )
        negatives = validation_negatives(actions, action_bank, start)
        with amp_context(amp):
            correct = model(semantic_history, geometry_history, actions)
            negative = model(semantic_history, geometry_history, negatives)
        semantic_prediction = correct["predicted_semantic_tokens"][:, 0, -1]
        geometry_prediction = correct["predicted_geometry_tokens"][:, 0, -1]
        semantic_target = semantic_future[:, -1]
        geometry_target = geometry_future[:, -1]
        semantic_h4.append(h4_nmse(semantic_prediction, semantic_target).cpu())
        geometry_h4.append(h4_nmse(geometry_prediction, geometry_target).cpu())
        nochange_semantic_h4.append(
            h4_nmse(semantic_history[:, -1], semantic_target).cpu()
        )
        nochange_geometry_h4.append(
            h4_nmse(geometry_history[:, -1], geometry_target).cpu()
        )
        correct_errors.append(
            per_candidate_dual_nmse(
                correct,
                semantic_future,
                geometry_future,
            )
            .mean(dim=-1)[:, 0]
            .cpu()
        )
        negative_errors.append(
            per_candidate_dual_nmse(
                negative,
                semantic_future,
                geometry_future,
            )
            .mean(dim=-1)
            .cpu()
        )
    semantic = torch.cat(semantic_h4).numpy()
    geometry = torch.cat(geometry_h4).numpy()
    nochange_semantic = torch.cat(nochange_semantic_h4).numpy()
    nochange_geometry = torch.cat(nochange_geometry_h4).numpy()
    correct = torch.cat(correct_errors).numpy()
    negative = torch.cat(negative_errors).numpy()
    rank = 1 + (negative < correct[:, None]).sum(axis=1)
    gap = float(negative.mean() - correct.mean())
    return {
        "windows": len(refs),
        "candidate_count_including_true": 12,
        "semantic_h4_nmse": float(semantic.mean()),
        "geometry_h4_nmse": float(geometry.mean()),
        "dual_h4_nmse": float(0.5 * (semantic.mean() + geometry.mean())),
        "nochange_semantic_h4_nmse": float(nochange_semantic.mean()),
        "nochange_geometry_h4_nmse": float(nochange_geometry.mean()),
        "nochange_dual_h4_nmse": float(
            0.5 * (nochange_semantic.mean() + nochange_geometry.mean())
        ),
        "correct_action_dual_nmse": float(correct.mean()),
        "negative_action_dual_nmse": float(negative.mean()),
        "negative_action_gap": gap,
        "negative_gap_fraction_of_true_error": float(
            gap / max(float(correct.mean()), 1.0e-8)
        ),
        "pairwise_action_win_rate": float(
            (correct[:, None] < negative).mean()
        ),
        "retrieval_top1": float(np.mean(rank == 1)),
        "mean_reciprocal_rank": float(np.mean(1.0 / rank)),
        "mean_rank": float(rank.mean()),
        "random_top1": 1.0 / 12.0,
        "runtime_seconds": time.perf_counter() - started,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
    }


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    result_path = output / "result.json"
    require(not result_path.exists(), f"refusing to overwrite {result_path}")
    require(torch.cuda.is_available(), "CUDA unavailable")
    require(args.train_steps > 0, "train steps must be positive")
    require(args.sample_rate > 0, "sample rate must be positive")
    for spec in DEFAULT_BAGS:
        require(spec.path.is_file(), f"missing bag: {spec.path}")
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    device = torch.device(args.device)
    set_seed(args.seed)
    backbone, representation, lineage = load_representation(device)
    paths, bag_rows = prepare_latents(
        output,
        DEFAULT_BAGS,
        backbone=backbone,
        representation=representation,
        sample_rate=args.sample_rate,
        batch_size=args.encode_batch_size,
        device=device,
        amp=args.amp,
    )
    del backbone, representation
    torch.cuda.empty_cache()
    split_by_trajectory = {
        spec.trajectory_id: spec.split for spec in DEFAULT_BAGS
    }
    train_refs, train_by_trajectory = window_refs(
        paths["legacy320"],
        split_by_trajectory,
        "train",
    )
    val_refs, _ = window_refs(
        paths["legacy320"],
        split_by_trajectory,
        "val",
    )
    require(len(train_refs) > 0 and len(val_refs) > 0, "empty train/val")
    schedule = route_balanced_schedule(
        train_by_trajectory,
        steps=args.train_steps,
        batch_size=args.train_batch_size,
        seed=args.seed + 1,
    )
    mode_results = {}
    checkpoints = {}
    for mode_index, mode in enumerate(("legacy320", "raw448")):
        cache = ShardCache(paths[mode])
        model, training = train_model(
            cache,
            train_refs,
            schedule,
            seed=args.seed,
            device=device,
            amp=args.amp,
        )
        evaluation = evaluate_model(
            model,
            cache,
            val_refs,
            batch_size=args.eval_batch_size,
            device=device,
            amp=args.amp,
        )
        checkpoint = output / f"{mode}_pilot_final.pt"
        torch.save(
            {
                "model": model.state_dict(),
                "config": training["config"],
                "training": training,
                "evaluation": evaluation,
                "mode": mode,
                "seed": args.seed,
                "outcome_gradients": False,
                "sealed_validation_opened": False,
                "scientific_status": "paired_resolution_development_pilot",
            },
            checkpoint,
        )
        checkpoints[mode] = {
            "path": str(checkpoint),
            "sha256": sha256_file(checkpoint),
        }
        mode_results[mode] = {
            "training": training,
            "held_route_evaluation": evaluation,
        }
        print(
            canonical_json(
                {
                    "stage": "mode_complete",
                    "mode": mode,
                    "evaluation": evaluation,
                }
            ),
            flush=True,
        )
        del model, cache
        torch.cuda.empty_cache()
    low = mode_results["legacy320"]["held_route_evaluation"]
    raw = mode_results["raw448"]["held_route_evaluation"]
    comparisons = {
        "dual_h4_nmse_change_raw_minus_legacy": (
            raw["dual_h4_nmse"] - low["dual_h4_nmse"]
        ),
        "dual_h4_nmse_relative_change": (
            raw["dual_h4_nmse"] / low["dual_h4_nmse"] - 1.0
        ),
        "negative_gap_fraction_change": (
            raw["negative_gap_fraction_of_true_error"]
            - low["negative_gap_fraction_of_true_error"]
        ),
        "pairwise_retrieval_change": (
            raw["pairwise_action_win_rate"] - low["pairwise_action_win_rate"]
        ),
        "top1_retrieval_change": (
            raw["retrieval_top1"] - low["retrieval_top1"]
        ),
    }
    checks = {
        "future_prediction_improved": (
            raw["dual_h4_nmse"] < low["dual_h4_nmse"]
        ),
        "negative_gap_fraction_improved": (
            raw["negative_gap_fraction_of_true_error"]
            > low["negative_gap_fraction_of_true_error"]
        ),
        "pairwise_action_retrieval_improved": (
            raw["pairwise_action_win_rate"] > low["pairwise_action_win_rate"]
        ),
        "top1_action_retrieval_improved": (
            raw["retrieval_top1"] > low["retrieval_top1"]
        ),
    }
    go = all(checks.values())
    result = {
        "status": "GO_RAW_RESOLUTION" if go else "NO_GO_RAW_RESOLUTION",
        "interpretation": (
            "raw detail improved held-route future prediction and every "
            "predeclared action-discriminability metric"
            if go
            else "raw detail did not jointly improve future prediction and "
            "action discriminability; do not scale this branch"
        ),
        "protocol": {
            "comparison": {
                "legacy320": (
                    "raw JPEG -> center crop 4:3 -> 320x240 bicubic -> "
                    "JPEG95 -> 448x336 bicubic"
                ),
                "raw448": (
                    "raw JPEG -> center crop 4:3 -> 448x336 bicubic -> JPEG95"
                ),
            },
            "train_trajectories": [
                spec.trajectory_id for spec in DEFAULT_BAGS if spec.split == "train"
            ],
            "held_route_trajectories": [
                spec.trajectory_id for spec in DEFAULT_BAGS if spec.split == "val"
            ],
            "sample_rate_hz": args.sample_rate,
            "train_windows": len(train_refs),
            "held_route_windows": len(val_refs),
            "same_frames_actions_schedule_architecture_seed": True,
            "dino_frozen": True,
            "outcome_labels_loaded": False,
            "ate_or_objective_gradients": False,
            "sealed_validation_opened": False,
            "formal_v70_modified": False,
            "frozen_highres_dual_modified": False,
        },
        "lineage": {
            **lineage,
            "runner": str(Path(__file__).resolve()),
            "runner_sha256": sha256_file(Path(__file__).resolve()),
            "bags": bag_rows,
        },
        "mode_results": mode_results,
        "comparisons": comparisons,
        "checks": checks,
        "checkpoints": checkpoints,
        "runtime_seconds": time.perf_counter() - started,
    }
    write_json_atomic(result_path, result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "result": str(result_path),
                "result_sha256": sha256_file(result_path),
                "checks": checks,
                "comparisons": comparisons,
                "runtime_seconds": result["runtime_seconds"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
