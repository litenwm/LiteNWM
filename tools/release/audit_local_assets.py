#!/usr/bin/env python3
"""Audit local external assets without opening images or trajectory pickles."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from scoring.trajectory_artifact_contract_v2 import (  # noqa: E402
    file_sha256,
    write_json_atomic,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/local_assets.json",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--allow-missing-dinov3", action="store_true")
    parser.add_argument(
        "--allow-missing-dataset-trajectories",
        action="store_true",
    )
    return parser.parse_args()


def resolve_path(config_path: Path, raw: str) -> Path:
    value = Path(raw).expanduser()
    return value if value.is_absolute() else (config_path.parent / value).resolve()


def file_check(path: Path, expected_sha256: str) -> dict[str, Any]:
    exists = path.is_file()
    actual = file_sha256(path) if exists else None
    return {
        "path": str(path),
        "exists": exists,
        "expected_sha256": expected_sha256,
        "actual_sha256": actual,
        "sha256_match": exists and actual == expected_sha256,
    }


def dataset_check(config_path: Path, value: dict[str, Any]) -> dict[str, Any]:
    root = resolve_path(config_path, str(value["trajectory_root"]))
    split_path = resolve_path(config_path, str(value["strict_splits"]))
    splits = json.loads(split_path.read_text(encoding="utf-8"))
    expected_by_split = {
        split: set(map(str, splits[split])) for split in ("train", "val", "test")
    }
    raw_ids = {path.name for path in root.iterdir() if path.is_dir()}
    coverage = {
        split: {
            "expected": len(ids),
            "present": len(ids & raw_ids),
            "missing": len(ids - raw_ids),
            "missing_examples": sorted(ids - raw_ids)[:20],
        }
        for split, ids in expected_by_split.items()
    }
    expected_all = set().union(*expected_by_split.values())
    return {
        "trajectory_root": str(root),
        "strict_splits": str(split_path),
        "raw_trajectory_directories": len(raw_ids),
        "strict_trajectory_count": len(expected_all),
        "strict_present_count": len(expected_all & raw_ids),
        "strict_missing_count": len(expected_all - raw_ids),
        "raw_outside_strict_lock": len(raw_ids - expected_all),
        "coverage_by_split": coverage,
    }


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve(strict=True)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema") != "dinov3_ltwm_local_assets_v1":
        raise ValueError("local asset config schema mismatch")

    dinov3_dir = resolve_path(config_path, config["dinov3"]["model_dir"])
    checks = {
        "dinov3_model": file_check(
            dinov3_dir / "model.safetensors",
            config["dinov3"]["model_safetensors_sha256"],
        ),
        "dinov3_config_exists": (dinov3_dir / "config.json").is_file(),
        "dinov3_preprocessor_exists": (
            dinov3_dir / "preprocessor_config.json"
        ).is_file(),
        "adapter": file_check(
            resolve_path(config_path, config["adapter"]["checkpoint"]),
            config["adapter"]["sha256"],
        ),
        "nomad_checkpoint": file_check(
            resolve_path(config_path, config["nomad"]["checkpoint"]),
            config["nomad"]["checkpoint_sha256"],
        ),
        "nomad_config": file_check(
            resolve_path(config_path, config["nomad"]["config"]),
            config["nomad"]["config_sha256"],
        ),
        "visualnav_root_exists": resolve_path(
            config_path, config["nomad"]["visualnav_root"]
        ).is_dir(),
        "diffusion_policy_root_exists": resolve_path(
            config_path, config["nomad"]["diffusion_policy_root"]
        ).is_dir(),
        "split_lock": file_check(
            resolve_path(config_path, config["split_lock"]["path"]),
            config["split_lock"]["sha256"],
        ),
    }
    datasets = {
        name: dataset_check(config_path, value)
        for name, value in config["datasets"].items()
    }
    asset_files_pass = all(
        check["sha256_match"]
        for key, check in checks.items()
        if isinstance(check, dict) and key != "dinov3_model"
    ) and all(
        bool(value)
        for key, value in checks.items()
        if isinstance(value, bool) and not key.startswith("dinov3_")
    )
    dinov3_pass = (
        checks["dinov3_model"]["sha256_match"]
        and checks["dinov3_config_exists"]
        and checks["dinov3_preprocessor_exists"]
    )
    datasets_pass = all(
        value["strict_missing_count"] == 0 for value in datasets.values()
    )
    status = "PASS"
    if not asset_files_pass:
        status = "FAIL"
    if not dinov3_pass and not args.allow_missing_dinov3:
        status = "FAIL"
    if not datasets_pass and not args.allow_missing_dataset_trajectories:
        status = "FAIL"
    disk_anchor = next(
        resolve_path(config_path, value["trajectory_root"])
        for value in config["datasets"].values()
    )
    disk = shutil.disk_usage(disk_anchor)
    report = {
        "schema": "dinov3_ltwm_local_asset_audit_v1",
        "status": status,
        "config": str(config_path),
        "config_sha256": file_sha256(config_path),
        "checks": checks,
        "datasets": datasets,
        "policy": {
            "allow_missing_dinov3": args.allow_missing_dinov3,
            "allow_missing_dataset_trajectories": args.allow_missing_dataset_trajectories,
        },
        "disk": {
            "anchor": str(disk_anchor),
            "total_bytes": disk.total,
            "used_bytes": disk.used,
            "free_bytes": disk.free,
        },
    }
    if args.output is not None:
        write_json_atomic(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
