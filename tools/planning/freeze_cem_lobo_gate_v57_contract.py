#!/usr/bin/env python3
"""Freeze the v57 LCB gate, proposals, and SHA chain before E/F outcomes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from tools.evaluation.export_recon_dense_dinov3_features_v1 import (  # noqa: E402
    write_json_atomic,
)
from tools.scoring.train_recon_dense_meta_dinov3_v1 import (  # noqa: E402
    file_sha256,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def rows(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    freeze_path = args.run_dir / "prelabel_freeze.json"
    binding_path = args.run_dir / "v57_runtime_binding.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    calibration_path = Path(binding["calibration"])
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    if (
        freeze.get("labels_opened")
        or freeze.get("future_pose_outcomes_opened")
        or freeze.get("outcomes_used_by_cem")
        or freeze.get("state_gate_used")
        or freeze.get("reserve_accessed")
        or binding.get("E_F_outcomes_accessed")
        or binding.get("reserve_accessed")
        or calibration.get("E_F_outcomes_accessed")
    ):
        raise RuntimeError("v57 pre-outcome contract invalid")
    if file_sha256(calibration_path) != binding["calibration_sha256"]:
        raise RuntimeError("v57 calibration changed")
    config = calibration["frozen_gate"]
    iteration = []
    for index in range(3):
        directory = args.run_dir / f"iteration_{index:02d}"
        score_path = directory / "frozen_scores.npz"
        with np.load(score_path, allow_pickle=False) as payload:
            mean = payload["mean"].astype(np.float32)
            std = payload["std"].astype(np.float32)
            p_dual = payload["dual_probability"].astype(np.float32)
            p_harm = payload["harm_probability"].astype(np.float32)
            valid = payload["valid"].astype(bool)
            safe = payload["safe"].astype(bool)
            energy = payload["energy"].astype(np.float32)
            elite = payload["elite_index_for_update"].astype(np.int64)
            updated = payload["proposal_updated"].astype(bool)
            safe_count = payload["safe_count"].astype(np.int64)
        benefit = -0.5 * (mean[..., 0] + mean[..., 3]) - float(
            config["beta"]
        ) * np.sqrt(std[..., 0] ** 2 + std[..., 3] ** 2)
        expected_safe = (
            valid
            & (benefit >= float(config["benefit_threshold"]))
            & (p_dual >= float(config["dual_threshold"]))
            & (p_harm <= float(config["harm_threshold"]))
        )
        expected_safe[:, 0] = False
        expected_energy = -benefit + 0.25 * p_harm - 0.10 * p_dual
        expected_energy = np.where(valid, expected_energy, np.inf).astype(np.float32)
        expected_energy[:, 0] = np.inf
        if not np.array_equal(safe, expected_safe):
            raise RuntimeError(f"iteration {index} gate mismatch")
        if not np.allclose(energy, expected_energy, rtol=1.0e-6, atol=1.0e-7):
            raise RuntimeError(f"iteration {index} LCB energy mismatch")
        if not np.array_equal(safe_count, safe.sum(axis=1)):
            raise RuntimeError(f"iteration {index} safe count mismatch")
        if not np.array_equal(updated, safe_count >= 4):
            raise RuntimeError(f"iteration {index} update mismatch")
        for state in range(len(updated)):
            if updated[state] and not safe[state, elite[state]].all():
                raise RuntimeError(f"iteration {index} unsafe elite")
            if not updated[state] and not np.all(elite[state] == -1):
                raise RuntimeError(f"iteration {index} unsafe elite padding")
        audit = freeze["iteration_audit"][index]
        iteration.append(
            {
                "iteration": index,
                "population": 16,
                "valid_candidates": int(valid[:, 1:].sum()),
                "safe_candidates": int(safe[:, 1:].sum()),
                "elite_candidates": int(updated.sum() * 4),
                "updated_windows": int(updated.sum()),
                "fallback_windows": int((~updated).sum()),
                "support_acceptance": audit["support_acceptance"],
                "ood_or_invalid_candidates": audit[
                    "ood_or_invalid_candidates"
                ],
                "predicted_scorer_utility_mean": audit[
                    "elite_scorer_utility_mean"
                ],
                "score_sha256": file_sha256(score_path),
            }
        )
    manifest = Path(freeze["manifest"])
    manifest_rows = rows(manifest)
    routes = sorted({str(row["route_id"]) for row in manifest_rows})
    contract = {
        "schema": "cem_lobo_gate_v57_preutility_contract",
        "status": "E_or_F_gate_proposals_and_selection_frozen_before_outcomes",
        "evidence_role": "spent_development_one_time_E_F_evaluation",
        "candidate_gate": config,
        "formula": calibration["formula"],
        "calibration": str(calibration_path),
        "calibration_sha256": file_sha256(calibration_path),
        "scorer_checkpoints": calibration["scorer_checkpoints"],
        "population": 16,
        "elite": 4,
        "shortfall_rule": "safe_count below 4 leaves q unchanged; no unsafe padding",
        "iterations": iteration,
        "paired_comparison": {
            "manifest": str(manifest),
            "manifest_sha256": file_sha256(manifest),
            "routes": routes,
            "route_count": len(routes),
            "seed": freeze["config"]["seed"],
            "cem_false_pool": freeze["candidate_pool"],
            "cem_false_pool_sha256": freeze["candidate_pool_sha256"],
        },
        "prelabel_freeze": str(freeze_path),
        "prelabel_freeze_sha256": file_sha256(freeze_path),
        "selected_actions": freeze["selected_actions"],
        "selected_actions_sha256": freeze["selected_actions_sha256"],
        "labels_opened": False,
        "outcomes_accessed": False,
        "E_F_outcomes_accessed": False,
        "state_gate_used": False,
        "reserve_accessed": False,
        "production_safe_claim_allowed": False,
    }
    write_json_atomic(args.output, contract)
    print(json.dumps(contract, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
