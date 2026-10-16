#!/usr/bin/env python3
"""Build a code/split-only v70 bundle and inventory optional large assets.

The first-stage bundle deliberately excludes model weights, feature caches,
raw datasets, and reserve outcomes.  Those assets are inventoried with roles,
sizes, and hashes so they can be added explicitly in a second stage.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "artifacts/v70_bundle_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def copy_file(source: Path, destination: Path) -> None:
    source.resolve(strict=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def copy_tree(source: Path, destination: Path) -> None:
    source.resolve(strict=True)
    shutil.copytree(
        source,
        destination,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
    )


def directory_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "bytes": None, "files": None}
    files = [value for value in path.rglob("*") if value.is_file()]
    return {
        "exists": True,
        "bytes": sum(value.stat().st_size for value in files),
        "files": len(files),
    }


def file_asset(path: Path, *, group: str, role: str) -> dict[str, Any]:
    record: dict[str, Any] = {
        "group": group,
        "role": role,
        "source_path": str(path),
        "included_in_stage1": False,
        "kind": "file",
        "exists": path.is_file(),
    }
    if path.is_file():
        record.update(bytes=path.stat().st_size, sha256=sha256(path))
    else:
        record.update(bytes=None, sha256=None)
    return record


def directory_asset(path: Path, *, group: str, role: str) -> dict[str, Any]:
    return {
        "group": group,
        "role": role,
        "source_path": str(path),
        "included_in_stage1": False,
        "kind": "directory",
        **directory_summary(path),
    }


def weight_assets() -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    dino_root = Path(
        "/home/rainbow/models/facebook/dinov3-vitl16-pretrain-lvd1689m"
    )
    for name in (
        "model.safetensors",
        "config.json",
        "configuration.json",
        "preprocessor_config.json",
    ):
        values.append(
            file_asset(
                dino_root / name,
                group="A_vision_backbone",
                role="frozen DINOv3 ViT-L/16 backbone and configuration",
            )
        )
    values.append(
        file_asset(
            ROOT
            / "checkpoints/project/dinov3_adapter/adapter_pilot_v1_final.pt",
            group="B_dinov3_adapter",
            role="frozen DINOv3-to-LTWM token adapter",
        )
    )
    for seed in (1701, 1702, 1703):
        values.append(
            file_asset(
                ROOT
                / f"checkpoints/project/ltwm_expansion/seed{seed}/best.pt",
                group="C_ltwm_ensemble",
                role=f"frozen LTWM rollout member seed {seed}",
            )
        )
        values.append(
            file_asset(
                ROOT / f"checkpoints/project/scorer_v2/seed{seed}/best.pt",
                group="D_scorer_v2_ensemble",
                role=f"frozen scorer-v2 member seed {seed}",
            )
        )
    for seed in range(3):
        values.append(
            file_asset(
                ROOT
                / (
                    "outputs/cem_development_v1/"
                    "dualdomain_cem_tail_v69_scand_route_crossfit/training/"
                    f"checkpoints/arm_D_full_seed{seed}.pt"
                ),
                group="E_v70_arm_d_deployment",
                role=f"v70 deployment Arm-D ensemble member {seed}",
            )
        )
    values.append(
        file_asset(
            ROOT
            / (
                "outputs/universal_dualdomain_raw_v2/training_v2/support/"
                "dualdomain_spacing_envelopes.npz"
            ),
            group="F_support_envelope",
            role="physical/support-domain envelope used by the frozen runner",
        )
    )
    values.append(
        file_asset(
            Path(
                "/home/rainbow/visualnav-transformer/deployment/"
                "model_weights/nomad.pth"
            ),
            group="G_nomad_candidate_generator",
            role="NoMaD checkpoint for live K17 candidate generation",
        )
    )
    return values


def optional_reproduction_assets() -> list[dict[str, Any]]:
    return [
        directory_asset(
            ROOT
            / (
                "outputs/cem_development_v1/"
                "dualdomain_cem_tail_v70_broad_hashfold_archive/oof_checkpoints"
            ),
            group="H_v70_calibration_oof",
            role="15 broad-hashfold OOF checkpoints; not needed for deployment",
        ),
        directory_asset(
            ROOT
            / (
                "outputs/cem_development_v1/"
                "dualdomain_cem_tail_v69_scand_route_crossfit/"
                "crossfit_checkpoints"
            ),
            group="I_v69_crossfit",
            role="v69 route-crossfit checkpoints; historical training reproduction",
        ),
        directory_asset(
            ROOT
            / "outputs/cem_development_v1/dualdomain_cem_tail_v66/harvest",
            group="J_proposal_harvest",
            role="proposal-aware fixed/CEM candidate harvest used in training",
        ),
        directory_asset(
            ROOT / "outputs/universal_dualdomain_raw_v2/features",
            group="K_recon_training_features",
            role="cached universal RECON training/calibration features",
        ),
        directory_asset(
            ROOT / "outputs/universal_dualdomain_raw_v2/sources",
            group="L_recon_training_sources",
            role="RECON route/source identity tables",
        ),
        directory_asset(
            ROOT / "outputs/scand_domain_adaptation_v1/features",
            group="M_scand_training_features",
            role="cached SCAND training/calibration features",
        ),
        directory_asset(
            ROOT / "outputs/scand_domain_adaptation_v1/sources",
            group="N_scand_training_sources",
            role="SCAND route/source identity tables",
        ),
        directory_asset(
            Path("/home/rainbow/nwm/data/recon"),
            group="O_raw_recon",
            role="raw/preprocessed RECON trajectories; not needed with frozen features",
        ),
        directory_asset(
            Path("/home/rainbow/nwm/data/scand"),
            group="P_raw_scand",
            role="raw/preprocessed SCAND trajectories; not needed with frozen features",
        ),
        directory_asset(
            Path("/home/rainbow/visualnav-transformer"),
            group="Q_nomad_codebase",
            role="external NoMaD codebase for live candidate generation",
        ),
    ]


def copy_code(bundle: Path) -> None:
    for name in ("src", "tools", "configs", "tests", "docs"):
        copy_tree(ROOT / name, bundle / "repository" / name)
    for name in (
        "README.md",
        "AUDIT.md",
        "EXTERNAL_DEPENDENCIES.md",
        "pipeline_contract.json",
        "requirements.txt",
        ".gitignore",
        ".gitattributes",
    ):
        source = ROOT / name
        if source.is_file():
            copy_file(source, bundle / "repository" / name)


def copy_v70_provenance(bundle: Path) -> None:
    relative_files = (
        "outputs/cem_development_v1/"
        "dualdomain_cem_tail_v70_broad_hashfold_archive_preregistration.json",
        "outputs/cem_development_v1/"
        "dualdomain_cem_tail_v70_broad_hashfold_archive/result.json",
        "outputs/cem_development_v1/"
        "dualdomain_cem_tail_v70_broad_hashfold_archive/aggregate_result.json",
        "outputs/cem_development_v1/"
        "dualdomain_cem_tail_v70_broad_hashfold_archive/calibration_frontier.json",
        "outputs/cem_development_v1/"
        "dualdomain_cem_tail_v70_broad_hashfold_archive/"
        "reserve_confirmation/sealed_manifest.json",
        "outputs/cem_development_v1/"
        "dualdomain_cem_tail_v69_scand_route_crossfit_preregistration.json",
        "outputs/cem_development_v1/"
        "dualdomain_cem_tail_v69_scand_route_crossfit/training/result.json",
        "outputs/universal_dualdomain_raw_v2/training_v2/result.json",
        "outputs/cem_development_v1/"
        "dualdomain_cem_tail_v66_preregistration.json",
        "outputs/cem_development_v1/"
        "dualdomain_cem_tail_v66_candidate_level_calibration_correction.json",
    )
    for relative in relative_files:
        source = ROOT / relative
        if source.is_file():
            copy_file(source, bundle / "provenance" / relative)


def copy_splits(bundle: Path) -> None:
    copy_tree(
        ROOT / "data/inner_manifests",
        bundle / "splits/inner_manifests",
    )
    external = {
        "recon_official100": Path(
            "/home/rainbow/rrnwm/data_splits/recon/"
            "server_full_nwm_unseen_h4_100"
        ),
        "scand_official100": Path(
            "/home/rainbow/rrnwm/data_splits/scand/"
            "official_nwm_test_h4_100_balanced_seed17"
        ),
    }
    for name, source in external.items():
        if source.is_dir():
            copy_tree(source, bundle / "splits/evaluation" / name)
    development = {
        **{
            f"recon_bank_{name}": ROOT
            / (
                "outputs/cem_development_v1/recon_target_calibration_v20/"
                f"bank_{name}/manifest.windows.jsonl"
            )
            for name in "abcdef"
        },
        "recon_bank_g": ROOT
        / (
            "outputs/cem_development_v1/recon_cem_boundary_v60/"
            "bank_g_v60/manifest.windows.jsonl"
        ),
        "recon_bank_h": ROOT
        / (
            "outputs/cem_development_v1/recon_cem_boundary_v60/"
            "bank_h_v60/manifest.windows.jsonl"
        ),
    }
    for name, source in development.items():
        if source.is_file():
            copy_file(
                source,
                bundle / "splits/development_banks" / f"{name}.windows.jsonl",
            )


def iter_files(root: Path) -> Iterable[Path]:
    return sorted(value for value in root.rglob("*") if value.is_file())


def write_bundle_readme(bundle: Path, inventory: dict[str, Any]) -> None:
    deployment_groups = ", ".join(
        group
        for group in (
            "A_vision_backbone",
            "B_dinov3_adapter",
            "C_ltwm_ensemble",
            "D_scorer_v2_ensemble",
            "E_v70_arm_d_deployment",
            "F_support_envelope",
        )
    )
    text = f"""# v70 first-stage bundle

This archive contains the current repository code closure, frozen v70
configuration/provenance, and data-split manifests. It intentionally contains
no model weights, raw datasets, feature caches, proposal harvests, or reserve
outcomes.

Minimum runtime weight groups: {deployment_groups}.

Add `G_nomad_candidate_generator` and `Q_nomad_codebase` only when the package
must generate the initial NoMaD x16/K17 pool itself. They are unnecessary when
K17 candidates are supplied by the caller.

Groups H–N are training/calibration reproduction assets and are not needed for
frozen deployment inference. Groups O–P are raw/preprocessed datasets.

The one-time reserve outcomes are excluded. The sealed manifest is included
only as provenance. See `ASSET_INVENTORY.json` for exact local paths, sizes,
and hashes.

Generated at: {inventory["created_at_utc"]}
"""
    (bundle / "README_BUNDLE.md").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing bundle: {output}")
    bundle = output / "v70_code_and_splits_v1"
    bundle.mkdir(parents=True)
    copy_code(bundle)
    copy_v70_provenance(bundle)
    copy_splits(bundle)

    assets = weight_assets() + optional_reproduction_assets()
    inventory = {
        "schema": "v70_bundle_asset_inventory_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "code_and_splits_only_waiting_for_user_asset_selection",
        "v70_version": "v70_broad_hashfold_archive",
        "reserve_outcomes_included": False,
        "assets": assets,
    }
    atomic_json(bundle / "ASSET_INVENTORY.json", inventory)
    write_bundle_readme(bundle, inventory)

    file_manifest = [
        {
            "relative_path": str(path.relative_to(bundle)),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in iter_files(bundle)
    ]
    atomic_json(bundle / "FILE_MANIFEST.json", file_manifest)

    archive = output / "v70_code_and_splits_v1.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        handle.add(bundle, arcname=bundle.name)
    summary = {
        "schema": "v70_bundle_stage1_result_v1",
        "status": "completed_waiting_for_asset_selection",
        "bundle_directory": str(bundle),
        "bundle_files": len(file_manifest) + 1,
        "archive": str(archive),
        "archive_bytes": archive.stat().st_size,
        "archive_sha256": sha256(archive),
        "asset_inventory": str(bundle / "ASSET_INVENTORY.json"),
        "file_manifest": str(bundle / "FILE_MANIFEST.json"),
        "reserve_outcomes_included": False,
    }
    atomic_json(output / "result.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
