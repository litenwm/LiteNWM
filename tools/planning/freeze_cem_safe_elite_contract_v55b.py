#!/usr/bin/env python3
"""Freeze and verify the v55 candidate-safe contract before true utility."""

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
from tools.planning.run_cem_development_v1 import array_sha256  # noqa: E402
from tools.scoring.train_recon_dense_meta_dinov3_v1 import (  # noqa: E402
    file_sha256,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    freeze = json.loads(args.freeze.read_text(encoding="utf-8"))
    invalid = (
        freeze.get("schema") != "universal_raw_d_cem_safe_elite_v55"
        or freeze.get("status")
        != "cem_true_safe_elite_actions_and_selections_frozen_before_outcomes"
        or freeze.get("cem") is not True
        or freeze.get("labels_opened")
        or freeze.get("future_pose_outcomes_opened")
        or freeze.get("outcomes_used_by_cem")
        or freeze.get("state_gate_used")
        or freeze.get("reserve_accessed")
    )
    if invalid:
        raise RuntimeError("invalid pre-utility freeze")
    config = freeze["config"]
    if (
        config["iterations"] != 3
        or config["population_challengers"] != 16
        or config["elite_count"] != 4
        or config["elite_fraction"] != 0.25
        or config["proposal_scale"] != 0.70
        or freeze["support_quantile"] != 0.999
    ):
        raise RuntimeError("CEM budget changed")

    manifest = Path(freeze["manifest"])
    if file_sha256(manifest) != freeze["manifest_sha256"]:
        raise RuntimeError("manifest SHA changed")
    rows = read_rows(manifest)
    route_key = "route_id" if freeze["domain"] == "recon" else "route_group"
    routes = sorted({str(row[route_key]) for row in rows})
    trajectories = sorted({str(row["trajectory_id"]) for row in rows})

    training_path = Path(freeze["training_result"])
    if file_sha256(training_path) != freeze["training_result_sha256"]:
        raise RuntimeError("training result SHA changed")
    training = json.loads(training_path.read_text(encoding="utf-8"))
    checkpoints = []
    for checkpoint in training["arms"]["D"]["checkpoints"]["full"]:
        checkpoint_path = Path(checkpoint["path"])
        checkpoint_sha = file_sha256(checkpoint_path)
        if checkpoint_sha != checkpoint["sha256"]:
            raise RuntimeError(f"checkpoint SHA changed: {checkpoint_path}")
        checkpoints.append(
            {
                "path": str(checkpoint_path),
                "sha256": checkpoint_sha,
                "seed": int(checkpoint["seed"]),
                "member_role": str(checkpoint["member_role"]),
            }
        )

    policy = {key: float(value) for key, value in freeze["policy"].items()}
    iterations = []
    for audit in freeze["iteration_audit"]:
        iteration = int(audit["iteration"])
        directory = args.freeze.parent / f"iteration_{iteration:02d}"
        population_path = directory / "population.npz"
        score_path = directory / "frozen_scores.npz"
        if file_sha256(population_path) != audit["population_sha256"]:
            raise RuntimeError(f"iteration {iteration} population SHA changed")
        if file_sha256(score_path) != audit["frozen_scores_sha256"]:
            raise RuntimeError(f"iteration {iteration} scores SHA changed")
        with np.load(population_path, allow_pickle=False) as payload:
            proposal_mean = payload["proposal_mean"].astype(np.float32)
            proposal_covariance = payload["proposal_covariance"].astype(np.float32)
        with np.load(score_path, allow_pickle=False) as payload:
            mean = payload["mean"].astype(np.float32)
            std = payload["std"].astype(np.float32)
            p_dual = payload["dual_probability"].astype(np.float32)
            p_harm = payload["harm_probability"].astype(np.float32)
            valid = payload["valid"].astype(bool)
            safe = payload["safe"].astype(bool)
            elite = payload["elite_index_for_update"].astype(np.int64)
            updated = payload["proposal_updated"].astype(bool)
            safe_count = payload["safe_count"].astype(np.int64)
        ucb = mean + policy["z"] * std
        recomputed = (
            valid
            & (ucb[..., 0] < -policy["margin_ate"])
            & (ucb[..., 3] < -policy["margin_objective"])
            & (p_dual >= policy["dual_threshold"])
            & (p_harm <= policy["harm_threshold"])
        )
        recomputed[:, 0] = False
        if not np.array_equal(safe, recomputed):
            raise RuntimeError(f"iteration {iteration} safe mask changed")
        if not np.array_equal(safe_count, safe.sum(axis=1)):
            raise RuntimeError(f"iteration {iteration} safe count changed")
        if not np.array_equal(updated, safe_count >= 4):
            raise RuntimeError(f"iteration {iteration} fallback rule changed")
        for state in range(len(updated)):
            if updated[state] and not safe[state, elite[state]].all():
                raise RuntimeError(f"iteration {iteration} unsafe elite")
            if not updated[state] and not np.all(elite[state] == -1):
                raise RuntimeError(f"iteration {iteration} padded unsafe elite")
        iterations.append(
            {
                "iteration": iteration,
                "population_sha256": file_sha256(population_path),
                "scores_sha256": file_sha256(score_path),
                "proposal_mean_sha256": array_sha256(proposal_mean),
                "proposal_covariance_sha256": array_sha256(proposal_covariance),
                "valid_candidates": int(valid[:, 1:].sum()),
                "safe_candidates": int(safe[:, 1:].sum()),
                "elite_candidates": int(updated.sum() * 4),
                "updated_windows": int(updated.sum()),
                "fallback_windows": int((~updated).sum()),
            }
        )

    contract = {
        "schema": "cem_safe_elite_v55_preutility_contract",
        "status": "safe_gate_proposals_and_paired_comparison_frozen_before_true_utility",
        "evidence_role": "A_to_T_spent_development_only",
        "blind_confirmation": False,
        "production_safe_claim_allowed": False,
        "candidate_safe_source": "frozen universal-D ensemble plus frozen support envelope",
        "candidate_safe_formula": (
            "valid AND mean_ATE+z*std_ATE < -margin_ate AND "
            "mean_Objective+z*std_Objective < -margin_objective AND "
            "p_dual >= dual_threshold AND p_harm <= harm_threshold"
        ),
        "thresholds": {
            **policy,
            "support_quantile": float(freeze["support_quantile"]),
            "support_threshold": float(freeze["support_threshold"]),
        },
        "scorer_training_result": str(training_path),
        "scorer_training_result_sha256": file_sha256(training_path),
        "scorer_checkpoints": checkpoints,
        "support_checkpoint": freeze["support_checkpoint"],
        "support_checkpoint_sha256": freeze["support_checkpoint_sha256"],
        "population": 16,
        "elite": 4,
        "shortfall_rule": (
            "safe_count < 4 means no q update; retain the previous proposal and "
            "never fill elites with unsafe candidates"
        ),
        "paired_comparison": {
            "cohort_manifest": str(manifest),
            "cohort_manifest_sha256": file_sha256(manifest),
            "states": len(rows),
            "routes": routes,
            "route_count": len(routes),
            "trajectories": trajectories,
            "trajectory_count": len(trajectories),
            "cem_false_pool": freeze["candidate_pool"],
            "cem_false_pool_sha256": freeze["candidate_pool_sha256"],
            "cem_true_seed": int(config["seed"]),
            "shared_incumbent": True,
            "shared_frozen_scorer_and_gate": True,
            "initial_proposal": iterations[0],
        },
        "iterations": iterations,
        "prelabel_freeze": str(args.freeze),
        "prelabel_freeze_sha256": file_sha256(args.freeze),
        "selected_actions": freeze["selected_actions"],
        "selected_actions_sha256": freeze["selected_actions_sha256"],
        "state_gate_used": False,
        "labels_opened": False,
        "outcomes_accessed": False,
        "reserve_accessed": False,
    }
    write_json_atomic(args.output, contract)
    print(json.dumps(contract, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
