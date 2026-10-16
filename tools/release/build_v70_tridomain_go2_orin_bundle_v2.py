#!/usr/bin/env python3
"""Build the sealed tri-domain V70 zero-motion bundle for Go2 Orin."""

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
DEFAULT_OUTPUT = ROOT / "artifacts/v70_tridomain_go2_orin_20260729_v2"
ARCHIVE_ROOT = "dinov3_ltwm_v70_tridomain_go2_orin_20260729_v2"
BASE_BUNDLE = (
    ROOT
    / "artifacts/v70_go2_orin_20260723/"
    "dinov3_ltwm_v70_go2_orin_20260723"
)
EXPORT = ROOT / "server_exports/v70_tridomain_latest_20260729/weights"
SCORER_ROOT = (
    ROOT
    / "outputs/local_old_full100_clean_median_h4_nwm_s_v70_v1/"
    "feature_scorer_download"
)

EXPECTED: dict[Path, str] = {
    EXPORT / "ltwm/seed1701/best.pt":
        "8970c387cd642027913c1cb01d5e2d2a2bfd6b58c94d1432e62880d3a434ac66",
    EXPORT / "ltwm/seed1702/best.pt":
        "6b4674d54b2da00a2825d12e6439e8bd87fb4ed6220bc6568ec3ea9350447338",
    EXPORT / "ltwm/seed1703/best.pt":
        "03bf6ad4dd023ec2828870b2ff1a110bf41936625ecc43c460bbf07c3ba62327",
    SCORER_ROOT / "seed1701_best.pt":
        "e7cef7ce68e97ab74db2a07474b0e0880eff06c3d4f9e2803be3a861461930ef",
    SCORER_ROOT / "seed1702_best.pt":
        "ee8f65aae6c117e76cc3e59ecc9f2b831e28ccc3625ca4db6bcf681ce0328f09",
    SCORER_ROOT / "seed1703_best.pt":
        "49be8e8c5022ffcc9b8fc0a6298320711070f80c77909b27f799744a7ed14754",
    (
        EXPORT
        / "selector_final/checkpoints/proposal_aware_full_seed0.pt"
    ): "242688fd252fd6ea650f4211afa4dc7bdb4075c6e8e3d10399b3255a67bd9daa",
    (
        EXPORT
        / "selector_final/checkpoints/proposal_aware_full_seed1.pt"
    ): "7e9c9fb62ff85e97946c3406c458c8b2cbf5155a3f77f3169c08073f45947726",
    (
        EXPORT
        / "selector_final/checkpoints/proposal_aware_full_seed2.pt"
    ): "47a65c3df352a8f1cbedbce676f51f0d9769fcde1a6e7b4b6238a6a2697c3c43",
    (
        EXPORT
        / "selector_final/support/tridomain_spacing_envelopes.npz"
    ): "da86c3f8286da60ac96fd0c3a9cd612ecdc3ed4198a82345868fa24592a6bfd9",
}

COMMON_EXPECTED = {
    "external/dinov3-vitl16-pretrain-lvd1689m/model.safetensors":
        "dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179",
    "external/nomad/nomad.pth":
        "70f79b8262527e20e56ced64a3e3d7ef91855bc9e7c3fa348d78edcb83c6a333",
    "external/nomad/nomad.yaml":
        "eb1aa594abd62a97a8ece0c735a4d2274a703d0fffaa3a97b0f880a950bba463",
    "checkpoints/project/dinov3_adapter/adapter_pilot_v1_final.pt":
        "bc0ec2cac457556bfb362b46b20584a02d8f954d8b7c17ed36b4c8646ec4ad5a",
}


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


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def iter_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file())


def remove_if_exists(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists() or path.is_symlink():
        path.unlink()


def verify_sources() -> None:
    if not BASE_BUNDLE.is_dir():
        raise FileNotFoundError(BASE_BUNDLE)
    for path, expected in EXPECTED.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(
                f"source SHA mismatch: {path}: {actual} != {expected}"
            )
    for relative, expected in COMMON_EXPECTED.items():
        path = BASE_BUNDLE / relative
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(
                f"base SHA mismatch: {relative}: {actual} != {expected}"
            )


def install_model_weights(staging: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    ltwm_root = staging / "checkpoints/project/ltwm_expansion"
    scorer_root = staging / "checkpoints/project/scorer_v2"
    remove_if_exists(ltwm_root)
    remove_if_exists(scorer_root)

    for seed in (1701, 1702, 1703):
        ltwm = EXPORT / f"ltwm/seed{seed}/best.pt"
        scorer = SCORER_ROOT / f"seed{seed}_best.pt"
        ltwm_dest = ltwm_root / f"seed{seed}/best.pt"
        scorer_dest = scorer_root / f"seed{seed}/best.pt"
        copy_file(ltwm, ltwm_dest)
        copy_file(scorer, scorer_dest)
        hashes[str(ltwm_dest.relative_to(staging))] = sha256(
            ltwm_dest
        )
        hashes[str(scorer_dest.relative_to(staging))] = sha256(
            scorer_dest
        )

    return hashes


def install_selector_runtime(staging: Path) -> dict[str, str]:
    runtime = staging / "runtime/v70_tridomain"
    remove_if_exists(runtime)
    checkpoints = runtime / "selector/checkpoints"
    support_dir = runtime / "selector/support"

    selector_hashes: dict[str, str] = {}
    for index in range(3):
        name = f"proposal_aware_full_seed{index}.pt"
        source = EXPORT / "selector_final/checkpoints" / name
        destination = checkpoints / name
        copy_file(source, destination)
        selector_hashes[str(destination.relative_to(staging))] = (
            sha256(destination)
        )

    support_source = (
        EXPORT
        / "selector_final/support/tridomain_spacing_envelopes.npz"
    )
    support = support_dir / "tridomain_spacing_envelopes.npz"
    copy_file(support_source, support)
    selector_hashes[str(support.relative_to(staging))] = sha256(
        support
    )

    source_result = json.loads(
        (EXPORT / "selector_final/result.json").read_text(
            encoding="utf-8"
        )
    )
    result = dict(source_result)
    localized_records = []
    for index, record in enumerate(
        source_result["checkpoints"]["full"]
    ):
        value = dict(record)
        name = f"proposal_aware_full_seed{index}.pt"
        destination = checkpoints / name
        if sha256(destination) != value["sha256"]:
            raise RuntimeError(
                f"selector record SHA mismatch: {name}"
            )
        value["path"] = (
            f"runtime/v70_tridomain/selector/checkpoints/{name}"
        )
        localized_records.append(value)
    result["checkpoints"] = {"full": localized_records}
    result["support"] = (
        "runtime/v70_tridomain/selector/support/"
        "tridomain_spacing_envelopes.npz"
    )

    localized_world_models = {}
    for seed in (1701, 1702, 1703):
        value = dict(source_result["world_models"][str(seed)])
        value["path"] = (
            f"checkpoints/project/ltwm_expansion/seed{seed}/best.pt"
        )
        if value["sha256"] != sha256(
            staging / value["path"]
        ):
            raise RuntimeError(
                f"result world-model SHA mismatch: {seed}"
            )
        localized_world_models[str(seed)] = value
    result["world_models"] = localized_world_models
    result["deployment_runtime"] = {
        "schema": "v70_tridomain_go2_orin_runtime_v1",
        "live_domain": "recon",
        "training_domains": ["recon", "scand", "sacson"],
        "motion_output_enabled": False,
        "abstention_action": [0.0, 0.0, 0.0],
    }
    result_path = runtime / "selector/result.json"
    write_json(result_path, result)

    source_seal = json.loads(
        (EXPORT / "selector_final/seal_record.json").read_text(
            encoding="utf-8"
        )
    )
    seal = dict(source_seal)
    seal["result"] = "runtime/v70_tridomain/selector/result.json"
    seal["result_sha256"] = sha256(result_path)
    seal["checkpoints"] = result["checkpoints"]
    seal["world_models"] = result["world_models"]
    seal["support"] = result["support"]
    seal["support_sha256"] = result["support_sha256"]
    seal_path = runtime / "selector/seal_record.json"
    write_json(seal_path, seal)

    for name in (
        "fair_eval_protocol_preregistration_v1.json",
        "validation_holdout_result.json",
    ):
        source = EXPORT / "selector_final" / name
        if source.is_file():
            copy_file(
                source,
                runtime / "selector" / name,
            )

    selector_hashes[str(result_path.relative_to(staging))] = (
        sha256(result_path)
    )
    selector_hashes[str(seal_path.relative_to(staging))] = sha256(
        seal_path
    )
    return selector_hashes


def install_current_runtime_code(staging: Path) -> None:
    remove_if_exists(staging / "deploy/v70_go2_orin")
    shutil.copytree(
        ROOT / "deploy/v70_go2_orin",
        staging / "deploy/v70_go2_orin",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    copy_file(
        ROOT
        / "tools/planning/run_tridomain_cem_tail_v70_runtime.py",
        staging
        / "tools/planning/run_tridomain_cem_tail_v70_runtime.py",
    )
    copy_file(
        ROOT / "src/planning/nomad_action_adapter.py",
        staging / "src/planning/nomad_action_adapter.py",
    )
    copy_file(
        ROOT / "src/scoring/universal_dualdomain_raw_v2.py",
        staging / "src/scoring/universal_dualdomain_raw_v2.py",
    )
    copy_file(
        ROOT / "tests/planning/test_v70_go2_runtime_contract.py",
        staging
        / "tests/planning/test_v70_go2_runtime_contract.py",
    )
    copy_file(
        ROOT / "tests/planning/test_safe_bridge_core.py",
        staging / "tests/planning/test_safe_bridge_core.py",
    )
    copy_file(
        Path(__file__),
        staging
        / "tools/release/build_v70_tridomain_go2_orin_bundle_v2.py",
    )


def remove_stale_dualdomain_runtime(staging: Path) -> None:
    stale = (
        "outputs/cem_development_v1/"
        "dualdomain_cem_tail_v69_scand_route_crossfit",
        "outputs/cem_development_v1/"
        "dualdomain_cem_tail_v70_broad_hashfold_archive",
        "outputs/universal_dualdomain_raw_v2/training_v2/support",
    )
    for relative in stale:
        remove_if_exists(staging / relative)
    for relative in (
        "BUNDLE_MANIFEST.json",
        "FILE_MANIFEST.json",
        "RELEASE_MANIFEST.json",
        "SHA256SUMS.txt",
        "V70_RUNTIME_CONTRACT.json",
        "README.md",
    ):
        remove_if_exists(staging / relative)


def write_runtime_docs(
    staging: Path,
    model_hashes: dict[str, str],
    selector_hashes: dict[str, str],
) -> None:
    contract = {
        "schema": "v70_tridomain_go2_orin_runtime_contract_v1",
        "bundle_version": "v70_tridomain_20260729",
        "training_domains": ["recon", "scand", "sacson"],
        "live_domain": "recon",
        "selector_seal_status": "RELAXED_FEASIBLE_V1",
        "strict_domain_feasible": {
            "recon": True,
            "scand": True,
            "sacson": False,
        },
        "relaxed_domain_feasible_v1": {
            "recon": True,
            "scand": True,
            "sacson": True,
        },
        "zero_motion_default": True,
        "required_stop_file": "/tmp/go2_nav_stop",
        "required_arm_file_for_nonzero": "/run/go2-safety/ARMED",
        "candidate_zero_is_executable_fallback": False,
        "no_safe_challenger_behavior": "ABSTAIN_ZERO",
        "production_safe_claim_allowed": False,
        "model_sha256": {
            **model_hashes,
            **selector_hashes,
        },
    }
    write_json(staging / "V70_RUNTIME_CONTRACT.json", contract)
    (staging / "README.md").write_text(
        "# V70 tri-domain Go2 Orin bundle\n\n"
        "This bundle replaces the LTWM, trajectory scorer, and selector "
        "ensembles with the sealed 2026-07-29 three-domain artifacts.\n\n"
        "The live Go2 feature contract uses domain `recon`; the selector was "
        "trained jointly on `recon`, `scand`, and `sacson`. Candidate zero is "
        "never converted into a motion fallback. If no finite safe challenger "
        "exists, the saved executable action is exactly `[0, 0, 0]`.\n\n"
        "This is a zero-motion/shadow deployment. It does not authorize a "
        "production-safety claim or removal of the stop interlock.\n",
        encoding="utf-8",
    )
    for domain in ("recon", "scand", "sacson"):
        directory = staging / f"runtime_inputs/{domain}"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / ".keep").touch()


def write_checksums_and_manifest(staging: Path) -> dict[str, Any]:
    checksum_path = staging / "SHA256SUMS.txt"
    payload_before_checksums = [
        path
        for path in iter_files(staging)
        if path.name not in {"BUNDLE_MANIFEST.json", "SHA256SUMS.txt"}
    ]
    checksum_path.write_text(
        "".join(
            f"{sha256(path)}  {path.relative_to(staging)}\n"
            for path in payload_before_checksums
        ),
        encoding="utf-8",
    )

    payload_files = [
        path
        for path in iter_files(staging)
        if path.name != "BUNDLE_MANIFEST.json"
    ]
    entries = [
        {
            "relative_path": str(path.relative_to(staging)),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in payload_files
    ]
    manifest = {
        "schema": "v70_tridomain_go2_orin_bundle_manifest_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "archive_root": ARCHIVE_ROOT,
        "v70_version": "v70_tridomain_20260729",
        "training_domains": ["recon", "scand", "sacson"],
        "live_domain": "recon",
        "files": entries,
        "payload_bytes_excluding_manifest": sum(
            entry["bytes"] for entry in entries
        ),
        "zero_motion_default": True,
        "production_safe_claim_allowed": False,
    }
    write_json(staging / "BUNDLE_MANIFEST.json", manifest)
    return manifest


def build(output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(
            f"refusing to overwrite existing output: {output}"
        )
    verify_sources()

    staging = output / ARCHIVE_ROOT
    output.mkdir(parents=True)
    shutil.copytree(BASE_BUNDLE, staging)
    remove_stale_dualdomain_runtime(staging)
    model_hashes = install_model_weights(staging)
    selector_hashes = install_selector_runtime(staging)
    install_current_runtime_code(staging)
    write_runtime_docs(staging, model_hashes, selector_hashes)
    manifest = write_checksums_and_manifest(staging)

    archive = output / f"{ARCHIVE_ROOT}.tar.gz"
    with tarfile.open(archive, "w:gz", compresslevel=1) as handle:
        handle.add(
            staging,
            arcname=ARCHIVE_ROOT,
            recursive=True,
        )
    with tarfile.open(archive, "r:gz") as handle:
        members = handle.getmembers()
        if not members or any(
            Path(member.name).parts[0] != ARCHIVE_ROOT
            for member in members
        ):
            raise RuntimeError("archive root/member validation failed")

    archive_hash = sha256(archive)
    checksum = output / f"{archive.name}.sha256"
    checksum.write_text(
        f"{archive_hash}  {archive.name}\n",
        encoding="utf-8",
    )
    result = {
        "schema": "v70_tridomain_go2_orin_bundle_result_v1",
        "status": "PASS",
        "staging": str(staging),
        "archive": str(archive),
        "archive_bytes": archive.stat().st_size,
        "archive_sha256": archive_hash,
        "checksum_file": str(checksum),
        "files": len(iter_files(staging)),
        "payload_bytes": sum(
            path.stat().st_size for path in iter_files(staging)
        ),
        "manifest_files": len(manifest["files"]),
        "archive_member_check": "PASS",
        "zero_motion_default": True,
        "production_safe_claim_allowed": False,
    }
    write_json(output / "result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    args = parser.parse_args()
    build(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
