#!/usr/bin/env python3
"""Resume-safe DINOv3+adapter encoding for strict trajectory splits."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from recon_latents.actions import derive_continuous_actions  # noqa: E402
from recon_latents.dinov3_adapter import (  # noqa: E402
    DinoV3AdapterConfig,
    DinoV3TokenAdapter,
    extract_dinov3_patch_tokens,
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
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--partition-count", type=int, default=1)
    parser.add_argument("--partition-index", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def preprocess(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        image = image.convert("RGB").resize((224, 224), Image.Resampling.BICUBIC)
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = (
        array - np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
    ) / np.asarray((0.229, 0.224, 0.225), dtype=np.float32)
    return torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1)))


def write_text_atomic(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def write_shard_atomic(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
    temporary.replace(path)


def load_pose_pickle(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        try:
            value = pickle.load(handle)
        except ModuleNotFoundError as error:
            if not str(error.name).startswith("numpy._core"):
                raise

            class NumpyCompatUnpickler(pickle.Unpickler):
                def find_class(self, module: str, name: str) -> Any:
                    return super().find_class(
                        module.replace("numpy._core", "numpy.core"), name
                    )

            handle.seek(0)
            value = NumpyCompatUnpickler(handle).load()
    if not isinstance(value, dict) or not {"position", "yaw"}.issubset(value):
        raise ValueError(f"invalid trajectory pickle: {path}")
    return value


def load_trajectory(
    root: Path,
    trajectory_id: str,
) -> tuple[list[Path], np.ndarray, np.ndarray]:
    resolved_root = root.resolve(strict=True)
    trajectory_path = (resolved_root / trajectory_id).resolve(strict=True)
    try:
        trajectory_path.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError(f"trajectory path escapes root: {trajectory_path}") from error
    images = sorted(trajectory_path.glob("*.jpg"), key=lambda path: int(path.stem))
    payload = load_pose_pickle(trajectory_path / "traj_data.pkl")
    positions = np.asarray(payload["position"], dtype=np.float32)
    yaw = np.asarray(payload["yaw"], dtype=np.float32).reshape(-1)
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError(f"positions must have shape [T,2]: {trajectory_id}")
    if len(images) != len(positions) or len(yaw) != len(positions):
        raise ValueError(f"image/position/yaw length mismatch: {trajectory_id}")
    if len(images) < 2 or not np.isfinite(positions).all() or not np.isfinite(yaw).all():
        raise ValueError(f"invalid trajectory values: {trajectory_id}")
    return images, positions, yaw


def validate_existing_shard(
    path: Path,
    *,
    trajectory_id: str,
    frames: int,
    model_sha256: str,
    adapter_sha256: str,
    split_lock_sha256: str,
) -> None:
    with np.load(path, allow_pickle=False) as shard:
        required = {
            "patch_latents",
            "actions_continuous",
            "positions",
            "yaw",
            "trajectory_id",
            "dinov3_weight_sha256",
            "adapter_checkpoint_sha256",
            "split_lock_sha256",
        }
        if missing := required.difference(shard.files):
            raise RuntimeError(f"resume shard {path} missing arrays: {sorted(missing)}")
        scalar_contract = {
            "trajectory_id": trajectory_id,
            "dinov3_weight_sha256": model_sha256,
            "adapter_checkpoint_sha256": adapter_sha256,
            "split_lock_sha256": split_lock_sha256,
        }
        for key, expected in scalar_contract.items():
            if str(shard[key].item()) != expected:
                raise RuntimeError(f"resume shard {key} mismatch: {path}")
        if shard["patch_latents"].shape != (frames, 16, 16, 384):
            raise RuntimeError(f"resume shard latent shape mismatch: {path}")
        if shard["patch_latents"].dtype != np.float16:
            raise RuntimeError(f"resume shard latent dtype mismatch: {path}")
        if shard["actions_continuous"].shape != (frames - 1, 3):
            raise RuntimeError(f"resume shard action shape mismatch: {path}")
        if shard["actions_continuous"].dtype != np.float32:
            raise RuntimeError(f"resume shard action dtype mismatch: {path}")
        if shard["positions"].shape != (frames, 2) or shard["yaw"].shape != (frames,):
            raise RuntimeError(f"resume shard pose shape mismatch: {path}")


def main() -> None:
    args = parse_args()
    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive")
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    include_splits = tuple(
        value.strip() for value in args.include_splits.split(",") if value.strip()
    )
    if not include_splits or any(
        value not in {"train", "val", "test"} for value in include_splits
    ):
        raise ValueError("include-splits must contain train, val, or test")
    lock, lock_sha = load_split_lock(args.split_lock)
    strict_splits = json.loads(args.strict_splits.read_text(encoding="utf-8"))
    locked_splits = lock[args.dataset]["splits"]
    if any(
        set(map(str, strict_splits[split])) != set(map(str, locked_splits[split]))
        for split in ("train", "val", "test")
    ):
        raise ValueError("strict split file does not match the active split lock")
    selected_splits = {
        split: sorted(map(str, strict_splits[split])) if split in include_splits else []
        for split in ("train", "val", "test")
    }
    split_for_id = {
        trajectory_id: split
        for split, values in selected_splits.items()
        for trajectory_id in values
    }
    if args.partition_count < 1:
        raise ValueError("--partition-count must be >= 1")
    if not 0 <= args.partition_index < args.partition_count:
        raise ValueError("--partition-index must be in [0, partition-count)")
    if args.partition_count > 1:
        ordered_ids = sorted(split_for_id)
        split_for_id = {
            trajectory_id: split_for_id[trajectory_id]
            for offset, trajectory_id in enumerate(ordered_ids)
            if offset % args.partition_count == args.partition_index
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

    from transformers import AutoModel

    device = torch.device(args.device)
    backbone = AutoModel.from_pretrained(
        args.model_dir,
        local_files_only=True,
        dtype=torch.float32,
    ).eval().to(device).requires_grad_(False)
    adapter = adapter.eval().to(device).requires_grad_(False)

    shard_dir = args.output_root / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    trajectories = []
    total_frames = 0
    reused_shards = 0
    for trajectory_id in sorted(split_for_id):
        image_paths, positions, yaw = load_trajectory(args.trajectory_root, trajectory_id)
        shard_path = shard_dir / f"{trajectory_id}.npz"
        reused = False
        if shard_path.exists():
            if args.resume:
                validate_existing_shard(
                    shard_path,
                    trajectory_id=trajectory_id,
                    frames=len(image_paths),
                    model_sha256=model_sha,
                    adapter_sha256=adapter_sha,
                    split_lock_sha256=lock_sha,
                )
                reused = True
                reused_shards += 1
            elif not args.overwrite:
                raise FileExistsError(f"refusing to overwrite {shard_path}")
        if not reused:
            batches = []
            for start in range(0, len(image_paths), args.batch_size):
                images = torch.stack(
                    [preprocess(path) for path in image_paths[start : start + args.batch_size]]
                ).to(device)
                with torch.inference_mode():
                    hidden = backbone(pixel_values=images).last_hidden_state
                    adapted = adapter(extract_dinov3_patch_tokens(hidden))
                batches.append(adapted.cpu().numpy().astype(np.float16))
            patch_latents = np.concatenate(batches, axis=0)
            actions = derive_continuous_actions(positions, yaw)
            write_shard_atomic(
                shard_path,
                patch_latents=patch_latents,
                actions_continuous=actions,
                actions_atomic_onehot=np.zeros((len(actions), 0), dtype=np.float32),
                positions=positions,
                yaw=yaw,
                image_paths=np.asarray([str(path) for path in image_paths]),
                trajectory_id=np.asarray(trajectory_id),
                dinov3_weight_sha256=np.asarray(model_sha),
                adapter_checkpoint_sha256=np.asarray(adapter_sha),
                split_lock_sha256=np.asarray(lock_sha),
            )
        trajectories.append(
            {
                "trajectory_id": trajectory_id,
                "split": split_for_id[trajectory_id],
                "shard_file": shard_path.name,
                "num_frames": len(image_paths),
                "num_transitions": len(image_paths) - 1,
                "patch_latents_shape": [len(image_paths), 16, 16, 384],
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
        "format": "dinov3_vitl16_ltwm_adapter_npz_v2_resumable",
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
        "encoding_partition": {
            "count": args.partition_count,
            "index": args.partition_index,
        },
        "num_trajectories": len(trajectories),
        "num_frames": total_frames,
        "reused_shards": reused_shards,
        "atomic_shard_writes": True,
        "per_shard_provenance": True,
        "image_size": 224,
        "native_patch_grid_shape": [14, 14],
        "native_latent_dim": 1024,
        "patch_grid_shape": [16, 16],
        "latent_dim": 384,
        "dtype": {"patch_latents": "float16", "actions_continuous": "float32"},
        "action_alignment": "actions_continuous[t] maps patch_latents[t] to patch_latents[t+1]",
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
                "metadata_sha256": file_sha256(args.output_root / "metadata.json"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
