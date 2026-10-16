#!/usr/bin/env python3
"""Verify the extracted v70 Go2/Orin runtime bundle and its Python environment."""

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
        "889fbecc54575e8d2e356c3fae8a04ce56c05bf47e8737949154143b94558661",
    "checkpoints/project/ltwm_expansion/seed1702/best.pt":
        "1a0767f0e184719218e3bcd4414fafd378da5c9e30fb0f6048973945af2bd859",
    "checkpoints/project/ltwm_expansion/seed1703/best.pt":
        "7ee8641ed7251f39bf8e7741c775a440ab1d505d3af66355f20894f85f6c0434",
    "checkpoints/project/scorer_v2/seed1701/best.pt":
        "39affafbb340940d77e5497126e8e1ad6167d37f35bd73fd76063ebe08313c9e",
    "checkpoints/project/scorer_v2/seed1702/best.pt":
        "a449bbc5962e5be7a8a7389724c99606152cd0af4738786b1c9d3c28eb9eea4b",
    "checkpoints/project/scorer_v2/seed1703/best.pt":
        "8ccc1e26aaec15caadde4c402282f781b4127666b392dcf895fec0c9d3696291",
    (
        "outputs/cem_development_v1/dualdomain_cem_tail_v69_scand_route_crossfit/"
        "training/checkpoints/arm_D_full_seed0.pt"
    ): "7b93d4e1866fc2ab4fc3bdbe3f55ab61a08b065f56ca254ff8c61e1ff5a65cd0",
    (
        "outputs/cem_development_v1/dualdomain_cem_tail_v69_scand_route_crossfit/"
        "training/checkpoints/arm_D_full_seed1.pt"
    ): "bd9beca5034ac92a7835460d5492e529c80ae420383c4daba499db44ce24e121",
    (
        "outputs/cem_development_v1/dualdomain_cem_tail_v69_scand_route_crossfit/"
        "training/checkpoints/arm_D_full_seed2.pt"
    ): "804bc5a27e2812bf84f119fa283d8495c3c230953179e3ede1889d06e809c0eb",
    (
        "outputs/universal_dualdomain_raw_v2/training_v2/support/"
        "dualdomain_spacing_envelopes.npz"
    ): "a5ea88bdd620991dd80ad8c53d54dc607452ca3bec5074ed1aa14710f0eb1495",
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
    parser.add_argument(
        "--all-files",
        action="store_true",
        help="Verify every entry in BUNDLE_MANIFEST.json, not only runtime assets.",
    )
    parser.add_argument(
        "--imports",
        action="store_true",
        help="Also verify the Orin Python/CUDA import environment.",
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
    import efficientnet_pytorch
    import einops
    import numpy
    import safetensors
    import torch
    import torchvision
    import transformers
    import yaml
    from diffusion_policy.model.diffusion.conditional_unet1d import (
        ConditionalUnet1D,
    )
    from vint_train.models.nomad.nomad import NoMaD
    from vint_train.models.nomad.nomad_vint import NoMaD_ViNT

    assert ConditionalUnet1D and NoMaD and NoMaD_ViNT
    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "numpy": numpy.__version__,
        "transformers": transformers.__version__,
        "diffusers": diffusers.__version__,
        "safetensors": safetensors.__version__,
        "einops": einops.__version__,
        "efficientnet_pytorch": getattr(
            efficientnet_pytorch, "__version__", "installed"
        ),
        "pyyaml": yaml.__version__,
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
        actual = sha256(path)
        checked += 1
        if actual != expected:
            failures.append(f"sha256 mismatch: {relative}: {actual}")

    if args.all_files:
        manifest_path = root / "BUNDLE_MANIFEST.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for entry in manifest["files"]:
            relative = entry["relative_path"]
            path = root / relative
            if not path.is_file():
                failures.append(f"missing: {relative}")
                continue
            actual = sha256(path)
            checked += 1
            if actual != entry["sha256"]:
                failures.append(f"sha256 mismatch: {relative}: {actual}")

    imports = None
    if args.imports:
        try:
            imports = verify_imports(root)
        except Exception as error:  # noqa: BLE001 - report the exact environment error.
            failures.append(f"import check failed: {error!r}")

    result = {
        "schema": "v70_go2_orin_bundle_verification_v1",
        "status": "PASS" if not failures else "FAIL",
        "bundle_root": str(root),
        "files_checked": checked,
        "imports": imports,
        "failures": failures,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())

