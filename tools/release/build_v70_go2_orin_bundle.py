#!/usr/bin/env python3
"""Build a self-contained v70 model/runtime bundle for Go2 Jetson Orin."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "artifacts/v70_go2_orin_20260723"
ARCHIVE_ROOT = "dinov3_ltwm_v70_go2_orin_20260723"
FINAL_RELEASE = (
    ROOT
    / "artifacts/v70_release_20260723_final/"
    "dinov3-ltwm-scorer-v2-mainline-v70"
)
DINO_ROOT = Path("/home/rainbow/models/facebook/dinov3-vitl16-pretrain-lvd1689m")
VISUALNAV_ROOT = Path("/home/rainbow/visualnav-transformer")
DIFFUSION_ROOT = Path("/home/rainbow/diffusion_policy")

EXPECTED = {
    DINO_ROOT / "model.safetensors":
        "dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179",
    VISUALNAV_ROOT / "deployment/model_weights/nomad.pth":
        "70f79b8262527e20e56ced64a3e3d7ef91855bc9e7c3fa348d78edcb83c6a333",
    VISUALNAV_ROOT / "train/config/nomad.yaml":
        "eb1aa594abd62a97a8ece0c735a4d2274a703d0fffaa3a97b0f880a950bba463",
}

VISUALNAV_FILES = (
    "LICENSE",
    "train/vint_train/__init__.py",
    "train/vint_train/models/__init__.py",
    "train/vint_train/models/nomad/__init__.py",
    "train/vint_train/models/nomad/nomad.py",
    "train/vint_train/models/nomad/nomad_vint.py",
    "train/vint_train/models/vint/__init__.py",
    "train/vint_train/models/vint/self_attention.py",
)
DIFFUSION_FILES = (
    "LICENSE",
    "diffusion_policy/model/diffusion/conditional_unet1d.py",
    "diffusion_policy/model/diffusion/conv1d_components.py",
    "diffusion_policy/model/diffusion/positional_embedding.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_file(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def iter_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file())


def local_assets_config() -> dict[str, Any]:
    return {
        "schema": "dinov3_ltwm_local_assets_v1",
        "dinov3": {
            "model_dir": "../external/dinov3-vitl16-pretrain-lvd1689m",
            "model_id": "facebook/dinov3-vitl16-pretrain-lvd1689m",
            "revision": "dd0a398fa8e84f2a37179332f6c561d20276300b",
            "model_safetensors_sha256":
                "dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179",
        },
        "adapter": {
            "checkpoint":
                "../checkpoints/project/dinov3_adapter/adapter_pilot_v1_final.pt",
            "sha256":
                "bc0ec2cac457556bfb362b46b20584a02d8f954d8b7c17ed36b4c8646ec4ad5a",
        },
        "nomad": {
            "visualnav_root": "../external/visualnav-transformer",
            "diffusion_policy_root": "../external/diffusion_policy",
            "checkpoint": "../external/nomad/nomad.pth",
            "checkpoint_sha256":
                "70f79b8262527e20e56ced64a3e3d7ef91855bc9e7c3fa348d78edcb83c6a333",
            "config": "../external/nomad/nomad.yaml",
            "config_sha256":
                "eb1aa594abd62a97a8ece0c735a4d2274a703d0fffaa3a97b0f880a950bba463",
        },
        "datasets": {
            "recon": {
                "trajectory_root": "../runtime_inputs/recon",
                "strict_splits": "../data/split_lock/recon_strict_splits.json",
            },
            "scand": {
                "trajectory_root": "../runtime_inputs/scand",
                "strict_splits": "../data/split_lock/scand_strict_splits.json",
            },
        },
        "split_lock": {
            "path": "../data/split_lock/split_lock.json",
            "sha256":
                "60e5c5af433b52046987b59aa7779c60c82a811a6bf6898c6d618cf7e1539b96",
        },
    }


def build(output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    for path, expected in EXPECTED.items():
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f"hash mismatch: {path}: {actual} != {expected}")
    if not FINAL_RELEASE.is_dir():
        raise FileNotFoundError(FINAL_RELEASE)

    staging = output / ARCHIVE_ROOT
    output.mkdir(parents=True)
    shutil.copytree(FINAL_RELEASE, staging)

    for name in ("config.json", "configuration.json", "preprocessor_config.json",
                 "model.safetensors"):
        copy_file(
            DINO_ROOT / name,
            staging / "external/dinov3-vitl16-pretrain-lvd1689m" / name,
        )
    copy_file(
        VISUALNAV_ROOT / "deployment/model_weights/nomad.pth",
        staging / "external/nomad/nomad.pth",
    )
    copy_file(
        VISUALNAV_ROOT / "train/config/nomad.yaml",
        staging / "external/nomad/nomad.yaml",
    )
    for relative in VISUALNAV_FILES:
        copy_file(
            VISUALNAV_ROOT / relative,
            staging / "external/visualnav-transformer" / relative,
        )
    copy_file(
        ROOT / ".gitignore",
        staging / "external/visualnav-transformer/deployment/src/.keep",
    )
    for relative in DIFFUSION_FILES:
        copy_file(
            DIFFUSION_ROOT / relative,
            staging / "external/diffusion_policy" / relative,
        )

    shutil.copytree(
        ROOT / "deploy/v70_go2_orin",
        staging / "deploy/v70_go2_orin",
    )
    copy_file(
        Path(__file__),
        staging / "tools/release/build_v70_go2_orin_bundle.py",
    )
    write_json(staging / "configs/local_assets.json", local_assets_config())
    (staging / "runtime_inputs/recon").mkdir(parents=True)
    (staging / "runtime_inputs/scand").mkdir(parents=True)
    (staging / "runtime_inputs/recon/.keep").write_text("", encoding="utf-8")
    (staging / "runtime_inputs/scand/.keep").write_text("", encoding="utf-8")

    payload_files = [
        path for path in iter_files(staging)
        if path.name != "BUNDLE_MANIFEST.json"
    ]
    file_entries = [
        {
            "relative_path": str(path.relative_to(staging)),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in payload_files
    ]
    manifest = {
        "schema": "v70_go2_orin_bundle_manifest_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "archive_root": ARCHIVE_ROOT,
        "v70_version": "v70_broad_hashfold_archive",
        "purpose": "Go2 Jetson Orin model/runtime asset transfer",
        "files": file_entries,
        "payload_bytes_excluding_manifest": sum(
            entry["bytes"] for entry in file_entries
        ),
        "includes": [
            "DINOv3 ViT-L/16",
            "project adapter",
            "LTWM 3-member ensemble",
            "scorer-v2 3-member ensemble",
            "v70 Arm-D 3-member deployment ensemble",
            "support envelope",
            "NoMaD checkpoint/config/minimal inference source",
        ],
        "excludes": [
            "raw RECON/SCAND data",
            "training feature caches",
            "OOF checkpoints",
            "reserve inputs/outcomes",
        ],
        "production_safe_claim_allowed": False,
    }
    write_json(staging / "BUNDLE_MANIFEST.json", manifest)

    archive = output / f"{ARCHIVE_ROOT}.tar.gz"
    with tarfile.open(archive, "w:gz", compresslevel=1) as handle:
        handle.add(staging, arcname=ARCHIVE_ROOT, recursive=True)
    with tarfile.open(archive, "r:gz") as handle:
        members = handle.getmembers()
        if not members or any(
            Path(member.name).parts[0] != ARCHIVE_ROOT for member in members
        ):
            raise RuntimeError("archive root/member validation failed")

    archive_hash = sha256(archive)
    checksum = output / f"{archive.name}.sha256"
    checksum.write_text(f"{archive_hash}  {archive.name}\n", encoding="utf-8")
    result = {
        "schema": "v70_go2_orin_bundle_result_v1",
        "status": "PASS",
        "staging": str(staging),
        "archive": str(archive),
        "archive_bytes": archive.stat().st_size,
        "archive_sha256": archive_hash,
        "checksum_file": str(checksum),
        "files": len(iter_files(staging)),
        "payload_bytes": sum(path.stat().st_size for path in iter_files(staging)),
        "archive_member_check": "PASS",
        "production_safe_claim_allowed": False,
    }
    write_json(output / "result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    build(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

