#!/usr/bin/env python3
"""Fail-closed preflight validation for LTWM DINOv3 Recon+SCAND training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from recon_latents.split_lock import (  # noqa: E402
    SPLITS,
    assert_disjoint_splits,
    file_sha256,
    load_split_lock,
    recon_route_id,
    validate_latent_root,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-lock", type=Path, required=True)
    parser.add_argument("--dino-model-dir", type=Path, required=True)
    parser.add_argument("--recon-latent-root", type=Path, default=None)
    parser.add_argument("--scand-latent-root", type=Path, default=None)
    parser.add_argument("--adapter-checkpoint", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--require-finalized-scand", action="store_true")
    parser.add_argument("--history-frames", type=int, choices=(4,), default=4)
    parser.add_argument("--future-frames", type=int, choices=(4,), default=4)
    return parser.parse_args()


def validate_adapted_latent_root(
    root: Path,
    expected_splits: dict[str, list[str]],
    *,
    dataset: str,
    split_lock_sha256: str,
    model_sha256: str,
    adapter_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    result = validate_latent_root(root, expected_splits)
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    expected = {
        "format": "dinov3_vitl16_ltwm_adapter_npz_v1",
        "model_sha256": model_sha256,
        "adapter_checkpoint_sha256": adapter_sha256,
        "split_lock_sha256": split_lock_sha256,
        "split_lock_dataset": dataset,
    }
    actual = {key: metadata.get(key) for key in expected}
    if actual != expected:
        raise ValueError(
            f"{dataset} adapted latent provenance mismatch: {actual} != {expected}"
        )
    result["provenance"] = actual
    return result, metadata


def main() -> None:
    args = parse_args()
    checks: dict[str, Any] = {}
    lock, lock_sha = load_split_lock(args.split_lock)
    checks["split_lock"] = {"status": "PASS", "sha256": lock_sha}
    expected_protocol = {
        "history_frames": args.history_frames,
        "future_frames": args.future_frames,
        "minimum_window_frames": args.history_frames + args.future_frames,
    }
    actual_protocol = {
        key: lock["policy"].get(key) for key in expected_protocol
    }
    if actual_protocol != expected_protocol:
        raise ValueError(
            f"split-lock temporal protocol mismatch: {actual_protocol} != {expected_protocol}"
        )
    checks["temporal_protocol"] = {"status": "PASS", **expected_protocol}

    recon_splits = lock["recon"]["splits"]
    recon_groups = {
        trajectory_id: recon_route_id(trajectory_id)
        for split in SPLITS
        for trajectory_id in recon_splits[split]
    }
    checks["recon_split_disjoint"] = {
        "status": "PASS",
        **assert_disjoint_splits(recon_splits, group_for_id=recon_groups),
    }

    scand_splits = lock["scand"]["splits"]
    if args.require_finalized_scand and not any(scand_splits.values()):
        raise ValueError("SCAND split lock is not finalized with processed trajectories")
    if any(scand_splits.values()):
        scand_groups = {
            trajectory_id: entry["route_group"]
            for entry in lock["scand"]["bags"]
            for trajectory_id in entry["derived_trajectory_ids"]
        }
        checks["scand_split_disjoint"] = {
            "status": "PASS",
            **assert_disjoint_splits(scand_splits, group_for_id=scand_groups),
        }
        conversion_audit = lock["scand"].get("conversion_audit")
        if not conversion_audit:
            raise ValueError("finalized SCAND split lock has no conversion audit")
        manifest_path = Path(conversion_audit["manifest_path"])
        actual_manifest_hash = file_sha256(manifest_path)
        if actual_manifest_hash != conversion_audit["manifest_sha256"]:
            raise ValueError("SCAND conversion manifest changed after split finalization")
        checks["scand_conversion_manifest"] = {
            "status": "PASS",
            "path": str(manifest_path),
            "sha256": actual_manifest_hash,
            "converter": conversion_audit["converter"],
            "bag_count": conversion_audit["bag_count"],
        }
    else:
        checks["scand_split_disjoint"] = {
            "status": "PENDING",
            "reason": "SCAND conversion has not been finalized",
        }

    expected_model_hash = (
        "dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179"
    )
    model_path = args.dino_model_dir / "model.safetensors"
    actual_model_hash = file_sha256(model_path)
    if actual_model_hash != expected_model_hash:
        raise ValueError(f"unexpected DINOv3 weight hash: {actual_model_hash}")
    checks["dinov3_weight"] = {
        "status": "PASS",
        "path": str(model_path),
        "sha256": actual_model_hash,
    }

    adapter_hash = None
    if args.adapter_checkpoint is not None:
        import torch

        payload = torch.load(args.adapter_checkpoint, map_location="cpu", weights_only=False)
        if payload.get("split_lock_sha256") != lock_sha:
            raise ValueError("adapter checkpoint was not fit under this split lock")
        if payload.get("dinov3_weight_sha256") != actual_model_hash:
            raise ValueError("adapter checkpoint DINOv3 weight hash mismatch")
        adapter_hash = file_sha256(args.adapter_checkpoint)
        checks["adapter_checkpoint"] = {
            "status": "PASS",
            "path": str(args.adapter_checkpoint),
            "sha256": adapter_hash,
        }
    if (
        args.recon_latent_root is not None or args.scand_latent_root is not None
    ) and adapter_hash is None:
        raise ValueError("adapted latent validation requires --adapter-checkpoint")

    if args.recon_latent_root is not None:
        recon_training_view = {**recon_splits, "test": []}
        checks["recon_latents"], _ = validate_adapted_latent_root(
            args.recon_latent_root,
            recon_training_view,
            dataset="recon",
            split_lock_sha256=lock_sha,
            model_sha256=actual_model_hash,
            adapter_sha256=adapter_hash,
        )
    if args.scand_latent_root is not None:
        if not any(scand_splits.values()):
            raise ValueError("cannot validate SCAND latents before split finalization")
        scand_training_view = {**scand_splits, "test": []}
        checks["scand_latents"], metadata = validate_adapted_latent_root(
            args.scand_latent_root,
            scand_training_view,
            dataset="scand",
            split_lock_sha256=lock_sha,
            model_sha256=actual_model_hash,
            adapter_sha256=adapter_hash,
        )
        frame_counts = {
            str(item["trajectory_id"]): int(item["num_frames"])
            for item in metadata["trajectories"]
        }
        eligible = {
            split: sum(
                max(
                    0,
                    frame_counts[trajectory_id]
                    - args.history_frames
                    - args.future_frames
                    + 1,
                )
                for trajectory_id in scand_training_view[split]
            )
            for split in SPLITS
        }
        if eligible["train"] <= 0 or eligible["val"] <= 0:
            raise ValueError(f"SCAND has no eligible 4-context/4-future windows: {eligible}")
        checks["scand_h4_windows"] = {
            "status": "PASS",
            "history_frames": args.history_frames,
            "future_frames": args.future_frames,
            "eligible_windows": eligible,
        }

    result = {
        "status": (
            "PASS"
            if all(item["status"] == "PASS" for item in checks.values())
            else "PENDING"
        ),
        "checks": checks,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
