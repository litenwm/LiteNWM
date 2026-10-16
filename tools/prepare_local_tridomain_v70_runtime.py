#!/usr/bin/env python3
"""Relocate the sealed tri-domain V70 artifacts without changing weights."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import torch


REPO = Path("/home/rainbow/1/dinov3-ltwm-scorer-v2-mainline")
EXPORT = REPO / "server_exports/v70_tridomain_latest_20260729/weights"
EXPERIMENT = (
    REPO / "outputs/local_old_full100_clean_median_h4_nwm_s_v70_v1"
)
RUNTIME = EXPERIMENT / "v70_runtime"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def link(target: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    destination.symlink_to(target.resolve(strict=True))


def main() -> None:
    if RUNTIME.exists():
        raise FileExistsError(RUNTIME)
    RUNTIME.mkdir(parents=True)

    checkpoint_root = RUNTIME / "feature_checkpoint_root"
    scorer_download = EXPERIMENT / "feature_scorer_download"
    expected_world_model_sha: dict[int, str] = {}
    for seed in (1701, 1702, 1703):
        world_model = EXPORT / f"ltwm/seed{seed}/best.pt"
        scorer = scorer_download / f"seed{seed}_best.pt"
        world_model_sha = sha256(world_model)
        payload = torch.load(scorer, map_location="cpu", weights_only=False)
        if int(payload.get("seed", -1)) != seed:
            raise RuntimeError(f"scorer seed mismatch: {scorer}")
        if payload.get("world_model_checkpoint_sha256") != world_model_sha:
            raise RuntimeError(f"scorer/LTWM SHA mismatch: seed={seed}")
        expected_world_model_sha[seed] = world_model_sha
        link(
            world_model,
            checkpoint_root / f"ltwm_expansion/seed{seed}/best.pt",
        )
        link(
            scorer,
            checkpoint_root / f"scorer_v2/seed{seed}/best.pt",
        )

    original_result = json.loads(
        (EXPORT / "selector_final/result.json").read_text(encoding="utf-8")
    )
    result = dict(original_result)
    checkpoints = []
    for index, record in enumerate(original_result["checkpoints"]["full"]):
        path = (
            EXPORT
            / f"selector_final/checkpoints/proposal_aware_full_seed{index}.pt"
        ).resolve(strict=True)
        value = dict(record)
        value["path"] = str(path)
        if sha256(path) != value["sha256"]:
            raise RuntimeError(f"selector SHA mismatch: {path}")
        checkpoints.append(value)
    result["checkpoints"] = {"full": checkpoints}
    support = (
        EXPORT / "selector_final/support/tridomain_spacing_envelopes.npz"
    ).resolve(strict=True)
    if sha256(support) != result["support_sha256"]:
        raise RuntimeError("support SHA mismatch")
    result["support"] = str(support)
    world_models = {}
    for seed in (1701, 1702, 1703):
        value = dict(original_result["world_models"][str(seed)])
        path = (EXPORT / f"ltwm/seed{seed}/best.pt").resolve(strict=True)
        value["path"] = str(path)
        if sha256(path) != value["sha256"]:
            raise RuntimeError(f"world-model SHA mismatch: {seed}")
        world_models[str(seed)] = value
    result["world_models"] = world_models
    result_path = RUNTIME / "result.json"
    write_json(result_path, result)

    original_seal = json.loads(
        (EXPORT / "selector_final/seal_record.json").read_text(encoding="utf-8")
    )
    seal = dict(original_seal)
    seal["result"] = str(result_path.resolve())
    seal["result_sha256"] = sha256(result_path)
    seal["checkpoints"] = result["checkpoints"]
    seal["world_models"] = result["world_models"]
    seal_path = RUNTIME / "seal_record.json"
    write_json(seal_path, seal)

    prereg = json.loads(
        (
            EXPORT
            / "selector_final/fair_eval_protocol_preregistration_v1.json"
        ).read_text(encoding="utf-8")
    )
    prereg["frozen_result"] = str(result_path.resolve())
    prereg["frozen_result_sha256"] = sha256(result_path)
    prereg["seal_record"] = str(seal_path.resolve())
    prereg["seal_record_sha256"] = sha256(seal_path)
    prereg_path = RUNTIME / "fair_eval_protocol_preregistration_v1.json"
    write_json(prereg_path, prereg)

    audit = {
        "status": "LOCAL_TRIDOMAIN_V70_RUNTIME_READY",
        "result": str(result_path),
        "result_sha256": sha256(result_path),
        "seal_record": str(seal_path),
        "seal_record_sha256": sha256(seal_path),
        "protocol_preregistration": str(prereg_path),
        "protocol_preregistration_sha256": sha256(prereg_path),
        "feature_checkpoint_root": str(checkpoint_root),
        "world_model_sha256": expected_world_model_sha,
        "weights_changed": False,
        "only_paths_relocated": True,
    }
    write_json(RUNTIME / "AUDIT.json", audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
