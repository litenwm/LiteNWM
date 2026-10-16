#!/usr/bin/env python3
"""Verify the extracted tri-domain V70 runtime and zero-motion interlocks."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


CRITICAL_HASHES = {
    "external/dinov3-vitl16-pretrain-lvd1689m/model.safetensors":
        "dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179",
    "external/nomad/nomad.pth":
        "70f79b8262527e20e56ced64a3e3d7ef91855bc9e7c3fa348d78edcb83c6a333",
    "external/nomad/nomad.yaml":
        "eb1aa594abd62a97a8ece0c735a4d2274a703d0fffaa3a97b0f880a950bba463",
    "checkpoints/project/dinov3_adapter/adapter_pilot_v1_final.pt":
        "bc0ec2cac457556bfb362b46b20584a02d8f954d8b7c17ed36b4c8646ec4ad5a",
    "checkpoints/project/ltwm_expansion/seed1701/best.pt":
        "8970c387cd642027913c1cb01d5e2d2a2bfd6b58c94d1432e62880d3a434ac66",
    "checkpoints/project/ltwm_expansion/seed1702/best.pt":
        "6b4674d54b2da00a2825d12e6439e8bd87fb4ed6220bc6568ec3ea9350447338",
    "checkpoints/project/ltwm_expansion/seed1703/best.pt":
        "03bf6ad4dd023ec2828870b2ff1a110bf41936625ecc43c460bbf07c3ba62327",
    "checkpoints/project/scorer_v2/seed1701/best.pt":
        "e7cef7ce68e97ab74db2a07474b0e0880eff06c3d4f9e2803be3a861461930ef",
    "checkpoints/project/scorer_v2/seed1702/best.pt":
        "ee8f65aae6c117e76cc3e59ecc9f2b831e28ccc3625ca4db6bcf681ce0328f09",
    "checkpoints/project/scorer_v2/seed1703/best.pt":
        "49be8e8c5022ffcc9b8fc0a6298320711070f80c77909b27f799744a7ed14754",
    (
        "runtime/v70_tridomain/selector/checkpoints/"
        "proposal_aware_full_seed0.pt"
    ): "242688fd252fd6ea650f4211afa4dc7bdb4075c6e8e3d10399b3255a67bd9daa",
    (
        "runtime/v70_tridomain/selector/checkpoints/"
        "proposal_aware_full_seed1.pt"
    ): "7e9c9fb62ff85e97946c3406c458c8b2cbf5155a3f77f3169c08073f45947726",
    (
        "runtime/v70_tridomain/selector/checkpoints/"
        "proposal_aware_full_seed2.pt"
    ): "47a65c3df352a8f1cbedbce676f51f0d9769fcde1a6e7b4b6238a6a2697c3c43",
    (
        "runtime/v70_tridomain/selector/support/"
        "tridomain_spacing_envelopes.npz"
    ): "da86c3f8286da60ac96fd0c3a9cd612ecdc3ed4198a82345868fa24592a6bfd9",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bundle-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    parser.add_argument("--all-files", action="store_true")
    parser.add_argument("--imports", action="store_true")
    parser.add_argument(
        "--stop-file",
        type=Path,
        default=Path("/tmp/go2_nav_stop"),
    )
    parser.add_argument(
        "--arm-file",
        type=Path,
        default=Path("/run/go2-safety/ARMED"),
    )
    parser.add_argument(
        "--require-zero-motion-interlock",
        action="store_true",
    )
    return parser.parse_args()


def verify_imports(root: Path) -> dict[str, object]:
    sys.path[:0] = [
        str(root),
        str(root / "src"),
        str(root / "external/visualnav-transformer/train"),
        str(root / "external/diffusion_policy"),
    ]
    import diffusers
    import numpy
    import scipy
    import torch
    import torchvision
    import transformers

    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else None
        ),
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "transformers": transformers.__version__,
        "diffusers": diffusers.__version__,
    }


def main() -> int:
    args = parse_args()
    root = args.bundle_root.resolve(strict=True)
    failures: list[str] = []
    checked = 0

    for relative, expected in CRITICAL_HASHES.items():
        path = root / relative
        if not path.is_file():
            failures.append(f"missing: {relative}")
            continue
        checked += 1
        actual = sha256(path)
        if actual != expected:
            failures.append(
                f"sha256 mismatch: {relative}: {actual}"
            )

    if args.all_files:
        manifest = json.loads(
            (root / "BUNDLE_MANIFEST.json").read_text(
                encoding="utf-8"
            )
        )
        for entry in manifest["files"]:
            relative = entry["relative_path"]
            path = root / relative
            if not path.is_file():
                failures.append(f"missing: {relative}")
                continue
            checked += 1
            if sha256(path) != entry["sha256"]:
                failures.append(f"sha256 mismatch: {relative}")

    stop_present = args.stop_file.is_file()
    arm_absent = not args.arm_file.exists()
    if args.require_zero_motion_interlock:
        if not stop_present:
            failures.append(
                f"zero-motion stop file missing: {args.stop_file}"
            )
        if not arm_absent:
            failures.append(
                f"motion arm file must be absent: {args.arm_file}"
            )

    imports = None
    if args.imports:
        try:
            imports = verify_imports(root)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"import check failed: {exc!r}")

    result = {
        "schema": "v70_tridomain_go2_orin_verification_v1",
        "status": "PASS" if not failures else "FAIL",
        "bundle_root": str(root),
        "files_checked": checked,
        "zero_motion_interlock": {
            "stop_file": str(args.stop_file),
            "stop_present": stop_present,
            "arm_file": str(args.arm_file),
            "arm_absent": arm_absent,
        },
        "imports": imports,
        "failures": failures,
        "production_safe_claim_allowed": False,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
