#!/usr/bin/env python3
"""Encode strict train/val trajectories into high-resolution dual DINOv3 latents."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image, ImageOps
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from encode_dinov3_adapted_resumable import (  # noqa: E402
    load_trajectory,
    write_shard_atomic,
    write_text_atomic,
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
from recon_latents.split_lock import (  # noqa: E402
    file_sha256,
    load_split_lock,
    validate_parent_split_reuse,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--strict-splits", type=Path, required=True)
    parser.add_argument("--split-lock", type=Path, required=True)
    parser.add_argument("--dataset", choices=("recon", "scand"), required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--adapter-checkpoint", type=Path, required=True)
    parser.add_argument("--adapter-parent-split-lock", type=Path, default=None)
    parser.add_argument("--include-splits", default="train,val")
    parser.add_argument("--image-height", type=int, default=336)
    parser.add_argument("--image-width", type=int, default=448)
    parser.add_argument("--semantic-block", type=int, default=23)
    parser.add_argument("--geometry-blocks", default="5,11,17")
    parser.add_argument("--geometry-dim", type=int, default=192)
    parser.add_argument("--geometry-projection-seed", type=int, default=2026072401)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--amp", choices=("bf16", "fp16", "off"), default="bf16")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def preprocess(
    path: Path,
    *,
    image_height: int,
    image_width: int,
) -> torch.Tensor:
    with Image.open(path) as image:
        image = ImageOps.fit(
            image.convert("RGB"),
            (image_width, image_height),
            method=Image.Resampling.BICUBIC,
            centering=(0.5, 0.5),
        )
        array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1)))
    return imagenet_normalize(tensor)


def autocast_context(device: torch.device, amp: str):
    if device.type != "cuda" or amp == "off":
        return nullcontext()
    dtype = torch.bfloat16 if amp == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def validate_existing_shard(
    path: Path,
    *,
    trajectory_id: str,
    frames: int,
    config: MultiLayerDualLatentConfig,
    model_sha256: str,
    adapter_sha256: str,
    split_lock_sha256: str,
    geometry_projection_sha256: str,
) -> None:
    with np.load(path, allow_pickle=False) as shard:
        required = {
            "semantic_latents",
            "geometry_latents",
            "actions_continuous",
            "positions",
            "yaw",
            "trajectory_id",
            "dinov3_weight_sha256",
            "adapter_checkpoint_sha256",
            "split_lock_sha256",
            "geometry_projection_sha256",
        }
        if missing := required.difference(shard.files):
            raise RuntimeError(f"resume shard {path} missing arrays: {sorted(missing)}")
        scalar_contract = {
            "trajectory_id": trajectory_id,
            "dinov3_weight_sha256": model_sha256,
            "adapter_checkpoint_sha256": adapter_sha256,
            "split_lock_sha256": split_lock_sha256,
            "geometry_projection_sha256": geometry_projection_sha256,
        }
        for key, expected in scalar_contract.items():
            if str(shard[key].item()) != expected:
                raise RuntimeError(f"resume shard {key} mismatch: {path}")
        semantic_shape = (
            frames,
            config.grid_height,
            config.grid_width,
            config.semantic_dim,
        )
        geometry_shape = (
            frames,
            config.grid_height,
            config.grid_width,
            config.geometry_dim,
        )
        if shard["semantic_latents"].shape != semantic_shape:
            raise RuntimeError(f"resume semantic shape mismatch: {path}")
        if shard["geometry_latents"].shape != geometry_shape:
            raise RuntimeError(f"resume geometry shape mismatch: {path}")
        if (
            shard["semantic_latents"].dtype != np.float16
            or shard["geometry_latents"].dtype != np.float16
        ):
            raise RuntimeError(f"resume latent dtype mismatch: {path}")
        if shard["actions_continuous"].shape != (frames - 1, 3):
            raise RuntimeError(f"resume action shape mismatch: {path}")


def main() -> None:
    args = parse_args()
    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive")
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.device == "cpu" and args.amp != "off":
        raise ValueError("CPU encoding requires --amp off")
    include_splits = tuple(
        value.strip() for value in args.include_splits.split(",") if value.strip()
    )
    if not include_splits or any(value not in {"train", "val"} for value in include_splits):
        raise ValueError("only strict train,val may be encoded; test remains sealed")
    geometry_blocks = tuple(
        int(value.strip())
        for value in args.geometry_blocks.split(",")
        if value.strip()
    )
    representation_config = MultiLayerDualLatentConfig(
        image_height=args.image_height,
        image_width=args.image_width,
        semantic_block=args.semantic_block,
        geometry_blocks=geometry_blocks,
        geometry_dim=args.geometry_dim,
        geometry_projection_seed=args.geometry_projection_seed,
    )

    lock, lock_sha = load_split_lock(args.split_lock)
    strict_splits = json.loads(args.strict_splits.read_text(encoding="utf-8"))
    locked_splits = lock[args.dataset]["splits"]
    if any(
        set(map(str, strict_splits[split])) != set(map(str, locked_splits[split]))
        for split in ("train", "val", "test")
    ):
        raise ValueError("strict split file does not match active split lock")
    selected_splits = {
        split: sorted(map(str, strict_splits[split])) if split in include_splits else []
        for split in ("train", "val", "test")
    }
    split_for_id = {
        trajectory_id: split
        for split, values in selected_splits.items()
        for trajectory_id in values
    }

    adapter_payload = torch.load(
        args.adapter_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    adapter_lock_sha = adapter_payload.get("split_lock_sha256")
    adapter_lineage = None
    if adapter_lock_sha != lock_sha:
        if args.adapter_parent_split_lock is None:
            raise ValueError(
                "adapter split-lock mismatch; pass --adapter-parent-split-lock"
            )
        adapter_lineage = validate_parent_split_reuse(
            lock,
            parent_split_lock=args.adapter_parent_split_lock,
            artifact_split_lock_sha256=adapter_lock_sha,
            unchanged_datasets=("recon",),
            preserved_datasets=("scand",),
        )
    model_sha = file_sha256(args.model_dir / "model.safetensors")
    adapter_sha = file_sha256(args.adapter_checkpoint)
    if adapter_payload.get("dinov3_weight_sha256") != model_sha:
        raise ValueError("adapter checkpoint DINOv3 weight mismatch")
    adapter = DinoV3TokenAdapter(DinoV3AdapterConfig(**adapter_payload["config"]))
    adapter.load_state_dict(adapter_payload["model"], strict=True)
    representation = FrozenMultiLayerDualLatentEncoder(
        adapter,
        representation_config,
    )

    from transformers import AutoModel

    device = torch.device(args.device)
    backbone = AutoModel.from_pretrained(
        args.model_dir,
        local_files_only=True,
        dtype=torch.float32,
    ).eval().to(device).requires_grad_(False)
    representation = representation.eval().to(device).requires_grad_(False)

    shard_dir = args.output_root / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    trajectories: list[dict[str, object]] = []
    total_frames = 0
    reused_shards = 0
    started = time.perf_counter()
    for trajectory_id in sorted(split_for_id):
        image_paths, positions, yaw = load_trajectory(
            args.trajectory_root,
            trajectory_id,
        )
        shard_path = shard_dir / f"{trajectory_id}.npz"
        reused = False
        if shard_path.exists():
            if args.resume:
                validate_existing_shard(
                    shard_path,
                    trajectory_id=trajectory_id,
                    frames=len(image_paths),
                    config=representation_config,
                    model_sha256=model_sha,
                    adapter_sha256=adapter_sha,
                    split_lock_sha256=lock_sha,
                    geometry_projection_sha256=representation.geometry_projection_sha256,
                )
                reused = True
                reused_shards += 1
            elif not args.overwrite:
                raise FileExistsError(f"refusing to overwrite {shard_path}")
        if not reused:
            semantic_batches: list[np.ndarray] = []
            geometry_batches: list[np.ndarray] = []
            for start in range(0, len(image_paths), args.batch_size):
                images = torch.stack(
                    [
                        preprocess(
                            path,
                            image_height=representation_config.image_height,
                            image_width=representation_config.image_width,
                        )
                        for path in image_paths[start : start + args.batch_size]
                    ]
                ).to(device)
                with torch.inference_mode(), autocast_context(device, args.amp):
                    hidden_states = backbone(
                        pixel_values=images,
                        output_hidden_states=True,
                    ).hidden_states
                    encoded = representation(hidden_states)
                semantic_batches.append(
                    encoded["semantic_tokens"]
                    .reshape(
                        -1,
                        representation_config.grid_height,
                        representation_config.grid_width,
                        representation_config.semantic_dim,
                    )
                    .float()
                    .cpu()
                    .numpy()
                    .astype(np.float16)
                )
                geometry_batches.append(
                    encoded["geometry_tokens"]
                    .reshape(
                        -1,
                        representation_config.grid_height,
                        representation_config.grid_width,
                        representation_config.geometry_dim,
                    )
                    .float()
                    .cpu()
                    .numpy()
                    .astype(np.float16)
                )
            semantic_latents = np.concatenate(semantic_batches, axis=0)
            geometry_latents = np.concatenate(geometry_batches, axis=0)
            actions = derive_continuous_actions(positions, yaw)
            write_shard_atomic(
                shard_path,
                semantic_latents=semantic_latents,
                geometry_latents=geometry_latents,
                actions_continuous=actions,
                positions=positions,
                yaw=yaw,
                image_paths=np.asarray([str(path) for path in image_paths]),
                trajectory_id=np.asarray(trajectory_id),
                dinov3_weight_sha256=np.asarray(model_sha),
                adapter_checkpoint_sha256=np.asarray(adapter_sha),
                split_lock_sha256=np.asarray(lock_sha),
                geometry_projection_sha256=np.asarray(
                    representation.geometry_projection_sha256
                ),
            )
        trajectories.append(
            {
                "trajectory_id": trajectory_id,
                "split": split_for_id[trajectory_id],
                "shard_file": shard_path.name,
                "num_frames": len(image_paths),
                "num_transitions": len(image_paths) - 1,
                "semantic_latents_shape": [
                    len(image_paths),
                    representation_config.grid_height,
                    representation_config.grid_width,
                    representation_config.semantic_dim,
                ],
                "geometry_latents_shape": [
                    len(image_paths),
                    representation_config.grid_height,
                    representation_config.grid_width,
                    representation_config.geometry_dim,
                ],
                "source_dir": str(args.trajectory_root / trajectory_id),
                "resume_reused": reused,
            }
        )
        total_frames += len(image_paths)
        print(
            json.dumps(
                {
                    "trajectory": trajectory_id,
                    "frames": len(image_paths),
                    "split": split_for_id[trajectory_id],
                    "status": "resume_reused" if reused else "encoded",
                }
            ),
            flush=True,
        )

    metadata = {
        "format": "dinov3_vitl16_highres_multilayer_dual_npz_v1",
        "model_name": "facebook/dinov3-vitl16-pretrain-lvd1689m",
        "model_path": str(args.model_dir),
        "model_sha256": model_sha,
        "adapter_checkpoint": str(args.adapter_checkpoint),
        "adapter_checkpoint_sha256": adapter_sha,
        "adapter_split_lineage": adapter_lineage
        or {"status": "PASS", "mode": "active_split_lock", "split_lock_sha256": lock_sha},
        "split_lock_sha256": lock_sha,
        "split_lock_dataset": args.dataset,
        "included_splits": list(include_splits),
        "num_trajectories": len(trajectories),
        "num_frames": total_frames,
        "reused_shards": reused_shards,
        "atomic_shard_writes": True,
        "amp": args.amp,
        "preprocessing": {
            "resize": "PIL.ImageOps.fit_bicubic_center",
            "image_height": representation_config.image_height,
            "image_width": representation_config.image_width,
            "normalization": "ImageNet mean/std",
        },
        "representation": representation.contract_dict(),
        "dtype": {
            "semantic_latents": "float16",
            "geometry_latents": "float16",
            "actions_continuous": "float32",
        },
        "action_alignment": (
            "actions_continuous[t] maps semantic/geometry latent[t] to latent[t+1]"
        ),
        "sealed_test_policy": "test split omitted and prohibited by encoder",
        "trajectories": trajectories,
    }
    write_text_atomic(
        args.output_root / "splits.json",
        json.dumps(selected_splits, indent=2),
    )
    write_text_atomic(
        args.output_root / "metadata.json",
        json.dumps(metadata, indent=2),
    )
    write_text_atomic(
        args.output_root / "trajectories.jsonl",
        "".join(json.dumps(value) + "\n" for value in trajectories),
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "output_root": str(args.output_root),
                "trajectories": len(trajectories),
                "frames": total_frames,
                "reused_shards": reused_shards,
                "runtime_seconds": time.perf_counter() - started,
                "metadata_sha256": file_sha256(args.output_root / "metadata.json"),
                "representation": asdict(representation_config),
                "geometry_projection_sha256": (
                    representation.geometry_projection_sha256
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
