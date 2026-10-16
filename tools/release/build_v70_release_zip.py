#!/usr/bin/env python3
"""Build the self-auditing v70 reproduction ZIP with project-owned B-F weights."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import zipfile
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "artifacts/v70_release_20260723"
ARCHIVE_ROOT = "dinov3-ltwm-scorer-v2-mainline-v70"

CODE_TREES = ("src", "tools", "configs", "tests", "docs", "data_splits")
ROOT_FILES = (
    "README.md",
    "AUDIT.md",
    "EXTERNAL_DEPENDENCIES.md",
    "pipeline_contract.json",
    "requirements.txt",
    ".gitignore",
    ".gitattributes",
)
DATA_TREES = ("data/split_lock",)

PROJECT_ASSETS = {
    "B_dinov3_adapter": {
        "checkpoints/project/dinov3_adapter/adapter_pilot_v1_final.pt":
        "bc0ec2cac457556bfb362b46b20584a02d8f954d8b7c17ed36b4c8646ec4ad5a"
    },
    "C_ltwm_ensemble": {
        "checkpoints/project/ltwm_expansion/seed1701/best.pt":
        "889fbecc54575e8d2e356c3fae8a04ce56c05bf47e8737949154143b94558661",
        "checkpoints/project/ltwm_expansion/seed1702/best.pt":
        "1a0767f0e184719218e3bcd4414fafd378da5c9e30fb0f6048973945af2bd859",
        "checkpoints/project/ltwm_expansion/seed1703/best.pt":
        "7ee8641ed7251f39bf8e7741c775a440ab1d505d3af66355f20894f85f6c0434",
    },
    "D_scorer_v2_ensemble": {
        "checkpoints/project/scorer_v2/seed1701/best.pt":
        "39affafbb340940d77e5497126e8e1ad6167d37f35bd73fd76063ebe08313c9e",
        "checkpoints/project/scorer_v2/seed1702/best.pt":
        "a449bbc5962e5be7a8a7389724c99606152cd0af4738786b1c9d3c28eb9eea4b",
        "checkpoints/project/scorer_v2/seed1703/best.pt":
        "8ccc1e26aaec15caadde4c402282f781b4127666b392dcf895fec0c9d3696291",
    },
    "E_v70_arm_d_deployment": {
        (
            "outputs/cem_development_v1/"
            "dualdomain_cem_tail_v69_scand_route_crossfit/training/"
            "checkpoints/arm_D_full_seed0.pt"
        ): "7b93d4e1866fc2ab4fc3bdbe3f55ab61a08b065f56ca254ff8c61e1ff5a65cd0",
        (
            "outputs/cem_development_v1/"
            "dualdomain_cem_tail_v69_scand_route_crossfit/training/"
            "checkpoints/arm_D_full_seed1.pt"
        ): "bd9beca5034ac92a7835460d5492e529c80ae420383c4daba499db44ce24e121",
        (
            "outputs/cem_development_v1/"
            "dualdomain_cem_tail_v69_scand_route_crossfit/training/"
            "checkpoints/arm_D_full_seed2.pt"
        ): "804bc5a27e2812bf84f119fa283d8495c3c230953179e3ede1889d06e809c0eb",
    },
    "F_support_envelope": {
        (
            "outputs/universal_dualdomain_raw_v2/training_v2/support/"
            "dualdomain_spacing_envelopes.npz"
        ): "a5ea88bdd620991dd80ad8c53d54dc607452ca3bec5074ed1aa14710f0eb1495"
    },
}

V70_PROVENANCE_FILES = (
    (
        "outputs/cem_development_v1/"
        "dualdomain_cem_tail_v70_broad_hashfold_archive_preregistration.json"
    ),
    (
        "outputs/cem_development_v1/"
        "dualdomain_cem_tail_v70_broad_hashfold_archive/result.json"
    ),
    (
        "outputs/cem_development_v1/"
        "dualdomain_cem_tail_v70_broad_hashfold_archive/aggregate_result.json"
    ),
    (
        "outputs/cem_development_v1/"
        "dualdomain_cem_tail_v70_broad_hashfold_archive/calibration_frontier.json"
    ),
    (
        "outputs/cem_development_v1/"
        "dualdomain_cem_tail_v70_broad_hashfold_archive/"
        "calibration_archive_selection.npz"
    ),
    (
        "outputs/cem_development_v1/"
        "dualdomain_cem_tail_v69_scand_route_crossfit/training/result.json"
    ),
    (
        "outputs/cem_development_v1/"
        "dualdomain_cem_tail_v69_scand_route_crossfit/training/"
        "calibration_selected_index.npz"
    ),
    "outputs/universal_dualdomain_raw_v2/training_v2/result.json",
)

SEALED_MANIFEST = (
    ROOT
    / (
        "outputs/cem_development_v1/"
        "dualdomain_cem_tail_v70_broad_hashfold_archive/"
        "reserve_confirmation/sealed_manifest.json"
    )
)

EXTERNAL_ASSETS = {
    "dinov3": {
        "included": False,
        "model_id": "facebook/dinov3-vitl16-pretrain-lvd1689m",
        "revision": "dd0a398fa8e84f2a37179332f6c561d20276300b",
        "model_safetensors_sha256":
        "dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179",
    },
    "nomad": {
        "included": False,
        "checkpoint_sha256":
        "70f79b8262527e20e56ced64a3e3d7ef91855bc9e7c3fa348d78edcb83c6a333",
        "config_sha256":
        "eb1aa594abd62a97a8ece0c735a4d2274a703d0fffaa3a97b0f880a950bba463",
    },
    "visualnav_transformer": {"included": False},
    "diffusion_policy": {
        "included": False,
        "local_git_commit": "5ba07ac6661db573af695b419a7947ecb704690f",
    },
    "raw_recon_scand_images": {"included": False},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_file(relative: str, staging: Path) -> None:
    source = ROOT / relative
    if not source.is_file():
        raise FileNotFoundError(source)
    destination = staging / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def ignored(_directory: str, names: list[str]) -> set[str]:
    blocked = {
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        "local_assets.json",
    }
    return {
        name
        for name in names
        if name in blocked
        or name.endswith((".pyc", ".pyo", ".orig", ".rej"))
    }


def copy_tree(relative: str, staging: Path) -> None:
    source = ROOT / relative
    if not source.is_dir():
        raise FileNotFoundError(source)
    destination = staging / relative
    shutil.copytree(source, destination, ignore=ignored)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_release_docs(staging: Path, contract: dict[str, Any]) -> None:
    readme = """# Frozen v70 reproduction release

This ZIP preserves the repository-relative layout and contains:

- source, tools, configs, tests, and documentation;
- frozen RECON and SCAND H4-100 indices;
- exact label-free v70 training/calibration identities and overlap checker;
- project-owned runtime assets B-F;
- the frozen v70 CEM/policy contract and compact provenance.

It deliberately excludes DINOv3, the external NoMaD/visualnav-transformer
repository, diffusion_policy, raw RECON/SCAND images, historical H-P
training caches, OOF checkpoints/predictions, and all reserve inputs/outcomes.

Copy `configs/local_assets.example.json` to `configs/local_assets.json` and
replace only machine-local paths. External checkpoint hashes must match.

Before using a new test manifest:

    python tools/evaluation/check_v70_training_overlap.py \
      --dataset scand --candidate path/to/manifest.json

Use `data_splits/recon/server_full_nwm_unseen_h4_100/navigation_eval.pkl` and
`data_splits/scand/project_strict_test_h4_100_seed17/navigation_eval.pkl` for
the two included H4-100 evaluations.

`RELEASE_MANIFEST.json`, `FILE_MANIFEST.json`, and `SHA256SUMS.txt` audit the
release. This is a research reproduction package, not a Go2 production-safety
certification.
"""
    (staging / "README_V70_RELEASE.md").write_text(readme, encoding="utf-8")
    write_json(staging / "V70_RUNTIME_CONTRACT.json", contract)


def iter_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file())


def build(output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing release: {output}")
    staging = output / ARCHIVE_ROOT
    staging.mkdir(parents=True)

    for tree in CODE_TREES + DATA_TREES:
        copy_tree(tree, staging)
    for name in ROOT_FILES:
        if (ROOT / name).is_file():
            copy_file(name, staging)

    verified_assets: dict[str, list[dict[str, Any]]] = {}
    for group, values in PROJECT_ASSETS.items():
        verified_assets[group] = []
        for relative, expected_sha in values.items():
            source = ROOT / relative
            actual_sha = sha256(source)
            if actual_sha != expected_sha:
                raise RuntimeError(
                    f"asset hash mismatch: {relative}: {actual_sha} != {expected_sha}"
                )
            copy_file(relative, staging)
            verified_assets[group].append(
                {
                    "relative_path": relative,
                    "bytes": source.stat().st_size,
                    "sha256": actual_sha,
                }
            )

    for relative in V70_PROVENANCE_FILES:
        copy_file(relative, staging)

    sealed = json.loads(SEALED_MANIFEST.read_text(encoding="utf-8"))
    runtime_contract = {
        "schema": "v70_runtime_contract_release_v1",
        "version": sealed["model_contract"]["version"],
        "model_contract": sealed["model_contract"],
        "cem_contract": sealed["cem_contract"],
        "scripts": sealed["scripts"],
        "production_safe_claim_allowed":
        sealed["production_safe_claim_allowed"],
        "reserve_inputs_included": False,
        "reserve_outcomes_included": False,
    }
    write_release_docs(staging, runtime_contract)

    release_manifest = {
        "schema": "v70_reproduction_release_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "archive_root": ARCHIVE_ROOT,
        "v70_version": "v70_broad_hashfold_archive",
        "project_assets": verified_assets,
        "external_assets": EXTERNAL_ASSETS,
        "included_evaluation_splits": {
            "recon":
            "data_splits/recon/server_full_nwm_unseen_h4_100",
            "scand":
            "data_splits/scand/project_strict_test_h4_100_seed17",
        },
        "training_identity_manifest":
        "data_splits/v70_training_identity/manifest.json",
        "reserve_inputs_included": False,
        "reserve_outcomes_included": False,
        "raw_images_included": False,
        "historical_oof_assets_H_to_P_included": False,
    }
    write_json(staging / "RELEASE_MANIFEST.json", release_manifest)

    payload_files = [
        path
        for path in iter_files(staging)
        if path.name not in {"FILE_MANIFEST.json", "SHA256SUMS.txt"}
    ]
    file_manifest = [
        {
            "relative_path": str(path.relative_to(staging)),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in payload_files
    ]
    write_json(staging / "FILE_MANIFEST.json", file_manifest)
    (staging / "SHA256SUMS.txt").write_text(
        "".join(
            f"{entry['sha256']}  {entry['relative_path']}\n"
            for entry in file_manifest
        ),
        encoding="utf-8",
    )

    archive = output / f"{ARCHIVE_ROOT}.zip"
    with zipfile.ZipFile(
        archive,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
        allowZip64=True,
    ) as handle:
        for path in iter_files(staging):
            relative = path.relative_to(output)
            compression = (
                zipfile.ZIP_STORED
                if path.suffix.lower() in {".pt", ".npz", ".pkl"}
                else zipfile.ZIP_DEFLATED
            )
            handle.write(path, arcname=str(relative), compress_type=compression)

    with zipfile.ZipFile(archive, "r") as handle:
        corrupt = handle.testzip()
        if corrupt is not None:
            raise RuntimeError(f"ZIP CRC failure: {corrupt}")
        zip_names = set(handle.namelist())
    staged_names = {
        str(Path(ARCHIVE_ROOT) / path.relative_to(staging))
        for path in iter_files(staging)
    }
    if zip_names != staged_names:
        raise RuntimeError("ZIP member set differs from staged release")

    result = {
        "schema": "v70_release_zip_result_v1",
        "status": "pass",
        "release_directory": str(staging),
        "archive": str(archive),
        "archive_bytes": archive.stat().st_size,
        "archive_sha256": sha256(archive),
        "files": len(staged_names),
        "payload_bytes": sum(
            path.stat().st_size for path in iter_files(staging)
        ),
        "zip_crc_check": "pass",
        "member_set_check": "pass",
        "reserve_inputs_included": False,
        "reserve_outcomes_included": False,
    }
    write_json(output / "result.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(build(args.output), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
