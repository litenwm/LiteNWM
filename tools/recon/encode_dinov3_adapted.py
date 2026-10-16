#!/usr/bin/env python3
"""Encode strict trajectory splits into adapted DINOv3 LTWM latent shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
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
from recon_latents.io import load_recon_trajectories  # noqa: E402
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
    parser.add_argument(
        "--allow-parent-test-target-encoding",
        action="store_true",
        help=(
            "Permit frozen, per-trajectory encoding of a derived target-development "
            "split only when every selected trajectory belonged to the adapter "
            "parent test split. This does not permit those trajectories in "
            "a training view; downstream views remain bound to their own split lock."
        ),
    )
    parser.add_argument("--include-splits", default="train,val")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--max-trajectories", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def preprocess(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        image = image.convert("RGB").resize((224, 224), Image.Resampling.BICUBIC)
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = (
        array - np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
    ) / np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
    return torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1)))


def main() -> None:
    args = parse_args()
    include_splits = tuple(
        item.strip() for item in args.include_splits.split(",") if item.strip()
    )
    if any(item not in {"train", "val", "test"} for item in include_splits):
        raise ValueError("--include-splits must contain train, val, or test")
    lock, lock_sha = load_split_lock(args.split_lock)
    splits_all = json.loads(args.strict_splits.read_text(encoding="utf-8"))
    locked_splits = lock[args.dataset]["splits"]
    if any(
        set(map(str, splits_all[split])) != set(map(str, locked_splits[split]))
        for split in ("train", "val", "test")
    ):
        raise ValueError(
            f"{args.dataset} strict split file does not match the active split lock"
        )
    selected_ids = {
        trajectory_id
        for split in include_splits
        for trajectory_id in map(str, splits_all[split])
    }
    selected_splits = {
        split: sorted(set(map(str, splits_all[split])) & selected_ids)
        for split in ("train", "val", "test")
    }
    split_for_id = {
        trajectory_id: split
        for split, values in selected_splits.items()
        for trajectory_id in values
    }
    records, skipped = load_recon_trajectories(
        args.trajectory_root,
        args.max_trajectories,
        allow_missing_atomic_labels=True,
    )
    records = [record for record in records if record.trajectory_id in selected_ids]
    found = {record.trajectory_id for record in records}
    if found != selected_ids:
        raise ValueError(
            f"strict split/trajectory root mismatch: missing={sorted(selected_ids-found)[:10]} "
            f"unexpected={sorted(found-selected_ids)[:10]}"
        )

    adapter_payload = torch.load(
        args.adapter_checkpoint, map_location="cpu", weights_only=False
    )
    adapter_lock_sha = adapter_payload.get("split_lock_sha256")
    adapter_lineage = None
    if adapter_lock_sha != lock_sha:
        if args.adapter_parent_split_lock is None:
            raise ValueError(
                "adapter checkpoint split lock mismatch; a verified expansion "
                "requires --adapter-parent-split-lock"
            )
        if args.allow_parent_test_target_encoding:
            parent_lock, parent_lock_sha = load_split_lock(
                args.adapter_parent_split_lock
            )
            declared_parent_sha = lock.get("lineage", {}).get(
                "parent_split_lock_sha256"
            )
            if declared_parent_sha != parent_lock_sha:
                raise ValueError(
                    "target-development split does not declare the supplied "
                    "adapter parent lock"
                )
            if adapter_lock_sha != parent_lock_sha:
                raise ValueError(
                    "adapter checkpoint is not bound to the supplied parent lock"
                )
            parent_test = set(
                map(str, parent_lock[args.dataset]["splits"].get("test", ()))
            )
            outside_parent_test = sorted(selected_ids - parent_test)
            if outside_parent_test:
                raise ValueError(
                    "target-development encoding may use only parent-test "
                    f"trajectories; outside={outside_parent_test[:10]}"
                )
            adapter_lineage = {
                "status": "PASS",
                "mode": "frozen_parent_adapter_parent_test_target_encoding",
                "parent_split_lock": str(args.adapter_parent_split_lock),
                "parent_split_lock_sha256": parent_lock_sha,
                "adapter_split_lock_sha256": adapter_lock_sha,
                "selected_trajectory_count": len(selected_ids),
                "selected_ids_all_parent_test": True,
                "causal_contract": (
                    "frozen backbone/adapter applied independently per image; "
                    "no target-bank statistics are fitted"
                ),
            }
        else:
            adapter_lineage = validate_parent_split_reuse(
                lock,
                parent_split_lock=args.adapter_parent_split_lock,
                artifact_split_lock_sha256=adapter_lock_sha,
                unchanged_datasets=("recon",),
                preserved_datasets=("scand",),
            )
    model_hash = file_sha256(args.model_dir / "model.safetensors")
    if adapter_payload.get("dinov3_weight_sha256") != model_hash:
        raise ValueError("adapter checkpoint DINOv3 weight mismatch")
    adapter = DinoV3TokenAdapter(
        DinoV3AdapterConfig(**adapter_payload["config"])
    )
    adapter.load_state_dict(adapter_payload["model"], strict=True)

    from transformers import AutoModel

    device = torch.device(args.device)
    backbone = AutoModel.from_pretrained(
        args.model_dir,
        local_files_only=True,
        dtype=torch.float32,
    ).eval().to(device)
    adapter = adapter.eval().to(device)
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    for parameter in adapter.parameters():
        parameter.requires_grad_(False)

    shard_dir = args.output_root / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    trajectories = []
    total_frames = 0
    for record in records:
        shard_path = shard_dir / f"{record.trajectory_id}.npz"
        if shard_path.exists() and not args.overwrite:
            raise FileExistsError(f"refusing to overwrite {shard_path}")
        batches = []
        for start in range(0, len(record.image_paths), args.batch_size):
            images = torch.stack(
                [preprocess(path) for path in record.image_paths[start : start + args.batch_size]]
            ).to(device)
            with torch.inference_mode():
                hidden = backbone(pixel_values=images).last_hidden_state
                adapted = adapter(extract_dinov3_patch_tokens(hidden))
            batches.append(adapted.cpu().numpy().astype(np.float16))
        patch_latents = np.concatenate(batches, axis=0)
        actions = derive_continuous_actions(record.positions, record.yaw)
        np.savez(
            shard_path,
            patch_latents=patch_latents,
            actions_continuous=actions,
            actions_atomic_onehot=np.zeros((len(actions), 0), dtype=np.float32),
            positions=record.positions.astype(np.float32),
            yaw=record.yaw.astype(np.float32),
            image_paths=np.asarray([str(path) for path in record.image_paths], dtype=str),
            trajectory_id=np.asarray(record.trajectory_id),
        )
        trajectories.append(
            {
                "trajectory_id": record.trajectory_id,
                "split": split_for_id[record.trajectory_id],
                "shard_file": shard_path.name,
                "num_frames": len(record.image_paths),
                "num_transitions": len(record.image_paths) - 1,
                "patch_latents_shape": list(patch_latents.shape),
                "source_dir": str(record.path),
            }
        )
        total_frames += len(record.image_paths)
        print(
            json.dumps(
                {
                    "trajectory": record.trajectory_id,
                    "frames": len(record.image_paths),
                    "split": split_for_id[record.trajectory_id],
                }
            ),
            flush=True,
        )

    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "splits.json").write_text(
        json.dumps(selected_splits, indent=2), encoding="utf-8"
    )
    metadata = {
        "format": "dinov3_vitl16_ltwm_adapter_npz_v1",
        "model_name": "dinov3-vitl16-pretrain-lvd1689m",
        "model_path": str(args.model_dir),
        "model_sha256": model_hash,
        "adapter_checkpoint": str(args.adapter_checkpoint),
        "adapter_checkpoint_sha256": file_sha256(args.adapter_checkpoint),
        "adapter_split_lineage": adapter_lineage or {
            "status": "PASS",
            "mode": "active_split_lock",
            "split_lock_sha256": lock_sha,
        },
        "split_lock_sha256": lock_sha,
        "split_lock_dataset": args.dataset,
        "device": str(device),
        "image_size": 224,
        "native_patch_size": 16,
        "native_patch_grid_shape": [14, 14],
        "native_latent_dim": 1024,
        "patch_grid_shape": [16, 16],
        "latent_dim": 384,
        "dtype": {"patch_latents": "float16", "actions_continuous": "float32"},
        "num_trajectories": len(trajectories),
        "num_frames": total_frames,
        "action_alignment": (
            "actions_continuous[t] is paired with transition "
            "patch_latents[t] -> patch_latents[t+1]"
        ),
        "preprocessing": {
            "resize": "RGB bicubic 224x224",
            "normalization": "ImageNet mean/std",
            "adapter": "LayerNorm 1024 -> Linear 384 -> bicubic 14x14 to 16x16",
            "layout": "NHWC",
        },
        "included_splits": list(include_splits),
        "skipped_trajectories": skipped,
        "trajectories": trajectories,
    }
    (args.output_root / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    with (args.output_root / "trajectories.jsonl").open("w", encoding="utf-8") as handle:
        for item in trajectories:
            handle.write(json.dumps(item) + "\n")
    print(
        json.dumps(
            {
                "status": "PASS",
                "output_root": str(args.output_root),
                "trajectories": len(trajectories),
                "frames": total_frames,
                "metadata_sha256": file_sha256(args.output_root / "metadata.json"),
                "splits_sha256": file_sha256(args.output_root / "splits.json"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
