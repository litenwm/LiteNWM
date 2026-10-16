#!/usr/bin/env python3
"""Bind the paired, outcome-free v70 reserve replays before opening labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from tools.evaluation.export_recon_dense_dinov3_features_v1 import (  # noqa: E402
    load_npz,
    write_json_atomic,
)
from tools.scoring.train_recon_dense_meta_dinov3_v1 import file_sha256  # noqa: E402


EXPECTED_STATUS = "candidate_level_safe_elite_actions_frozen_before_outcomes"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seal", type=Path, required=True)
    parser.add_argument("--original-replay-dir", type=Path, required=True)
    parser.add_argument("--v70-replay-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_freeze(directory: Path, role: str) -> tuple[Path, dict[str, Any], dict[str, np.ndarray]]:
    freeze_path = directory / "prelabel_freeze.json"
    freeze = read_json(freeze_path)
    if freeze.get("schema") != "dualdomain_cem_tail_v70_archive_prelabel_replay":
        raise RuntimeError(f"unexpected replay schema: {freeze_path}")
    if freeze.get("status") != EXPECTED_STATUS:
        raise RuntimeError(f"replay is not frozen: {freeze_path}")
    if freeze.get("model", {}).get("model_role") != role:
        raise RuntimeError(f"unexpected model role: {freeze_path}")
    forbidden = (
        freeze.get("labels_opened")
        or freeze.get("future_pose_outcomes_opened")
        or freeze.get("outcomes_used_by_cem")
        or freeze.get("reserve_accessed")
    )
    if forbidden:
        raise RuntimeError(f"replay reports forbidden outcome access: {freeze_path}")
    if not freeze.get("candidate_level_safety_elite"):
        raise RuntimeError(f"candidate-level safe elite is disabled: {freeze_path}")
    if not freeze.get("cross_round_safe_archive"):
        raise RuntimeError(f"cross-round safe archive is disabled: {freeze_path}")
    selected_path = Path(freeze["selected_actions"])
    if file_sha256(selected_path) != freeze["selected_actions_sha256"]:
        raise RuntimeError(f"selected action hash changed: {selected_path}")
    return freeze_path, freeze, load_npz(selected_path)


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    seal = read_json(args.seal)
    if seal.get("status") != "v70_sealed_before_reserve_outcomes":
        raise RuntimeError("v70 reserve seal is not active")
    expected = seal["expected_outputs"]
    if args.original_replay_dir.resolve() != Path(expected["original_replay_dir"]).resolve():
        raise RuntimeError("original replay directory differs from sealed path")
    if args.v70_replay_dir.resolve() != Path(expected["v70_replay_dir"]).resolve():
        raise RuntimeError("v70 replay directory differs from sealed path")
    if args.output.resolve() != Path(expected["preopen_pair_audit"]).resolve():
        raise RuntimeError("pair-audit output differs from sealed path")

    original_path, original, original_arrays = load_freeze(
        args.original_replay_dir, "original_arm_d"
    )
    v70_path, v70, v70_arrays = load_freeze(
        args.v70_replay_dir, "proposal_aware_v66"
    )
    for key in (
        "domain",
        "cohort",
        "states",
        "manifest_sha256",
        "candidate_pool_sha256",
        "base_features_sha256",
        "latent_cache_sha256",
        "support_checkpoint_sha256",
        "universal_valid_contract",
        "safe_shortfall_rule",
        "config",
    ):
        if original.get(key) != v70.get(key):
            raise RuntimeError(f"paired replay mismatch for {key}")
    if original["domain"] != "recon" or original["cohort"] != "recon_final_reserve":
        raise RuntimeError("sealed reserve cohort identity changed")
    sealed_inputs = seal["reserve_inputs"]
    expected_input_hashes = {
        "manifest_sha256": sealed_inputs["manifest"]["sha256"],
        "candidate_pool_sha256": sealed_inputs["candidate_pool"]["sha256"],
        "base_features_sha256": sealed_inputs["base_features"]["sha256"],
        "latent_cache_sha256": sealed_inputs["latent_cache"]["sha256"],
    }
    for key, value in expected_input_hashes.items():
        if original.get(key) != value:
            raise RuntimeError(f"replay input differs from seal for {key}")

    expected_config = seal["cem_contract"]
    for key, value in expected_config.items():
        if original["config"].get(key) != value:
            raise RuntimeError(f"replay CEM config differs from seal for {key}")
    if original["policy"] != seal["model_contract"]["recon_policy"]:
        raise RuntimeError("original RECON policy differs from seal")
    if v70["policy"] != seal["model_contract"]["recon_policy"]:
        raise RuntimeError("v70 RECON policy differs from seal")

    if set(original_arrays) != set(v70_arrays):
        raise RuntimeError("paired selected-action schemas differ")
    differing_arrays = [
        key
        for key in original_arrays
        if not np.array_equal(original_arrays[key], v70_arrays[key], equal_nan=True)
    ]
    if differing_arrays:
        raise RuntimeError(
            "RECON original/v70 output is not bit-exact: " + ", ".join(differing_arrays)
        )
    original_selected = Path(original["selected_actions"])
    v70_selected = Path(v70["selected_actions"])
    if file_sha256(original_selected) != file_sha256(v70_selected):
        raise RuntimeError("RECON selected-action artifact hashes are not bit-exact")

    audit = {
        "schema": "dualdomain_cem_tail_v70_reserve_preopen_pair_audit",
        "status": "paired_actions_frozen_and_bit_exact_before_reserve_outcomes",
        "seal": str(args.seal.resolve()),
        "seal_sha256": file_sha256(args.seal),
        "original_replay_freeze": str(original_path.resolve()),
        "original_replay_freeze_sha256": file_sha256(original_path),
        "v70_replay_freeze": str(v70_path.resolve()),
        "v70_replay_freeze_sha256": file_sha256(v70_path),
        "original_selected_actions": str(original_selected.resolve()),
        "original_selected_actions_sha256": file_sha256(original_selected),
        "v70_selected_actions": str(v70_selected.resolve()),
        "v70_selected_actions_sha256": file_sha256(v70_selected),
        "all_selected_action_arrays_bit_exact": True,
        "states": int(original["states"]),
        "labels_opened": False,
        "future_pose_outcomes_opened": False,
        "outcomes_used_by_cem": False,
        "reserve_outcomes_accessed": False,
        "reserve_spent": False,
        "retuning_allowed_after_open": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(args.output, audit)
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
