#!/usr/bin/env python3
"""Fail-closed audit of adapter, LTWM, and scorer-v2 checkpoint lineage."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


SPLIT_SHA256 = "60e5c5af433b52046987b59aa7779c60c82a811a6bf6898c6d618cf7e1539b96"
PARENT_SPLIT_SHA256 = (
    "32763b89772e5e499a0ee5248ae05ff5c9c9509a8fdc3f47af402ce20cbf933b"
)
DINO_SHA256 = "dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179"
SEEDS = (1701, 1702, 1703)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-root", type=Path, default=Path(__file__).parents[2])
    args = parser.parse_args()
    root = args.release_root.resolve()
    failures: list[str] = []
    audited: list[dict[str, object]] = []

    adapter_path = (
        root / "checkpoints/project/dinov3_adapter/adapter_pilot_v1_final.pt"
    )
    adapter = torch.load(adapter_path, map_location="cpu", weights_only=False)
    require(
        adapter.get("dinov3_weight_sha256") == DINO_SHA256,
        "adapter does not reference the frozen DINOv3 weight",
        failures,
    )
    require(
        adapter.get("split_lock_sha256") == PARENT_SPLIT_SHA256,
        "adapter parent split-lock mismatch",
        failures,
    )
    require(
        adapter.get("config")
        == {"input_dim": 1024, "output_dim": 384, "input_grid": 14, "output_grid": 16},
        "adapter shape contract mismatch",
        failures,
    )
    audited.append(
        {
            "artifact": "dinov3_adapter",
            "sha256": sha256_file(adapter_path),
            "external_dinov3_weight_sha256": DINO_SHA256,
            "status": "PASS" if not failures else "CHECK_FAILURES",
        }
    )

    for seed in SEEDS:
        ltwm_path = root / f"checkpoints/project/ltwm_expansion/seed{seed}/best.pt"
        scorer_path = root / f"checkpoints/project/scorer_v2/seed{seed}/best.pt"
        parent_path = root / f"checkpoints/project/ltwm_parent/seed{seed}/best.pt"
        ltwm_sha = sha256_file(ltwm_path)
        scorer_sha = sha256_file(scorer_path)
        ltwm = torch.load(ltwm_path, map_location="cpu", weights_only=False)
        scorer = torch.load(scorer_path, map_location="cpu", weights_only=False)
        parent = torch.load(parent_path, map_location="cpu", weights_only=False)

        require(
            ltwm.get("split_lock_sha256") == SPLIT_SHA256,
            f"seed {seed} LTWM split-lock mismatch",
            failures,
        )
        require(
            ltwm.get("temporal_protocol")
            == {"history_frames": 4, "future_frames": 4},
            f"seed {seed} LTWM temporal contract mismatch",
            failures,
        )
        require(
            ltwm.get("config", {}).get("horizon") == 4,
            f"seed {seed} LTWM config horizon mismatch",
            failures,
        )
        require(
            ltwm.get("parameter_count") == 8_001_960,
            f"seed {seed} LTWM parameter count mismatch",
            failures,
        )
        require(
            str(ltwm.get("args", {}).get("warm_start_checkpoint", "")).endswith(
                f"ltwm_dinov3_h4_recon_exposure_matched_scand_3to1_seed{seed}/best.pt"
            ),
            f"seed {seed} LTWM warm-start lineage mismatch",
            failures,
        )
        require(
            parent.get("split_lock_sha256") == PARENT_SPLIT_SHA256,
            f"seed {seed} parent LTWM split-lock mismatch",
            failures,
        )
        require(
            scorer.get("schema") == "trajectory_quality_scorer_v2_checkpoint",
            f"seed {seed} scorer schema mismatch",
            failures,
        )
        require(
            scorer.get("split_lock_sha256") == SPLIT_SHA256,
            f"seed {seed} scorer split-lock mismatch",
            failures,
        )
        require(
            scorer.get("world_model_checkpoint_sha256") == ltwm_sha,
            f"seed {seed} scorer is not paired with its LTWM",
            failures,
        )
        require(
            scorer.get("parameter_count") == 1_484_162,
            f"seed {seed} scorer parameter count mismatch",
            failures,
        )
        require(
            scorer.get("outer_val_accessed") is False
            and scorer.get("test_accessed") is False,
            f"seed {seed} scorer claims outer validation/test access",
            failures,
        )
        require(
            scorer.get("candidate_identity_inputs") is False
            and scorer.get("cross_candidate_model_operations") is False,
            f"seed {seed} scorer violates clean independent-candidate contract",
            failures,
        )
        audited.append(
            {
                "artifact": f"seed{seed}",
                "ltwm_sha256": ltwm_sha,
                "scorer_sha256": scorer_sha,
                "scorer_epoch": scorer.get("epoch"),
            }
        )

    result = {
        "release_id": "dinov3_ltwm_scorer_v2_20260719",
        "audited": audited,
        "failures": failures,
        "decision": "PASS" if not failures else "FAIL",
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
