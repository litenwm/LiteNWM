#!/usr/bin/env python3
"""Audit and freeze the v55 candidate-safe CEM contract before outcomes."""

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
    freeze = json.loads(args.freeze.read_text(encoding="utf-8"))
    if (
        freeze.get("schema") != "universal_raw_d_cem_safe_elite_v55"
        or freeze.get("status")
        != "cem_true_safe_elite_actions_and_selections_frozen_before_outcomes"
        or freeze.get("cem") is not True
        or freeze.get("labels_opened")
        or freeze.get("future_pose_outcomes_opened")
        or freeze.get("outcomes_used_by_cem")
        or freeze.get("state_gate_used")
        or freeze.get("reserve_accessed")
    ):
        raise RuntimeError("v55 pre-utility freeze contract is invalid")
    config = freeze["config"]
    if (
        config["iterations"] != 3
        or config["population_challengers"] != 16
        or config["elite_count"] != 4
        or abs(config["elite_fraction"] - 0.25) > 1.0e-12
        or abs(config["proposal_scale"] - 0.70) > 1.0e-12
        or abs(freeze["support_quantile"] - 0.999) > 1.0e-12
    ):
        raise RuntimeError("frozen v55 budget changed")

    manifest = Path(freeze["manifest"])
    if file_sha256(manifest) != freeze["manifest_sha256"]:
        raise RuntimeError("manifest SHA changed")
    manifest_rows = rows(manifest)
    route_key = "route_id" if freeze["domain"] == "recon" else "route_group"
    routes = sorted({str(row[route_key]) for row in manifest_rows})
    trajectories = sorted({str(row["trajectory_id"]) for row in manifest_rows})

    training_path = Path(freeze["training_result"])
    if file_sha256(training_path) != freeze["training_result_sha256"]:
        raise RuntimeError("training-result SHA changed")
    training = json.loads(training_path.read_text(encoding="utf-8"))
    checkpoint_records = [
        {"path": str(path), "sha256": file_sha256(Path(path))}
        for path in training["arms"]["D"]["checkpoints"]["full"]
    ]

    policy = freeze["policy"]
    threshold_records = {
        "z": float(policy["z"]),
        "margin_ate": float(policy["margin_ate"]),
        "margin_objective": float(policy["margin_objective"]),
        "dual_threshold": float(policy["dual_threshold"]),
        "harm_threshold": float(policy["harm_threshold"]),
        "support_quantile": float(freeze["support_quantile"]),
        "support_threshold": float(freeze["support_threshold"]),
    }

    iteration_contracts = []
    first_population = None
    for audit in freeze["iteration_audit"]:
        iteration = int(audit["iteration"])
        directory = args.freeze.parent / f"iteration_{iteration:02d}"
        population_path = directory / "population.npz"
        scores_path = directory / "frozen_scores.npz"
        if file_sha256(population_path) != audit["population_sha256"]:
            raise RuntimeError(f"iteration {iteration} population SHA changed")
        if file_sha256(scores_path) != audit["frozen_scores_sha256"]:
            raise RuntimeError(f"iteration {iteration} score SHA changed")
        with np.load(population_path, allow_pickle=False) as population:
            proposal_mean = population["proposal_mean"].astype(np.float32)
            proposal_covariance = population["proposal_covariance"].astype(
                np.float32
            )
        with np.load(scores_path, allow_pickle=False) as score:
            mean = score["mean"].astype(np.float32)
            std = score["std"].astype(np.float32)
            p_dual = score["dual_probability"].astype(np.float32)
            p_harm = score["harm_probability"].astype(np.float32)
            valid = score["valid"].astype(bool)
            safe = score["safe"].astype(bool)
            elite = score["elite_index_for_update"].astype(np.int64)
            updated = score["proposal_updated"].astype(bool)
            safe_count = score["safe_count"].astype(np.int64)
        ucb = mean + float(policy["z"]) * std
        recomputed = (
            valid
            & (ucb[..., 0] < -float(policy["margin_ate"]))
            & (ucb[..., 3] < -float(policy["margin_objective"]))
            & (p_dual >= float(policy["dual_threshold"]))
            & (p_harm <= float(policy["harm_threshold"]))
        )
        recomputed[:, 0] = False
        if not np.array_equal(safe, recomputed):
            raise RuntimeError(f"iteration {iteration} safe formula mismatch")
        if not np.array_equal(safe_count, safe.sum(axis=1)):
            raise RuntimeError(f"iteration {iteration} safe count mismatch")
        if not np.array_equal(updated, safe_count >= 4):
            raise RuntimeError(f"iteration {iteration} fallback rule mismatch")
        for state in range(len(updated)):
            if updated[state]:
                if len(elite[state]) != 4 or not safe[state, elite[state]].all():
                    raise RuntimeError(
                        f"iteration {iteration} unsafe candidate entered elite"
                    )
            elif not np.all(elite[state] == -1):
                raise RuntimeError(
                    f"iteration {iteration} shortfall was padded with an elite"
                )
        proposal_record = {
            "iteration": iteration,
            "population_sha256": file_sha256(population_path),
            "frozen_scores_sha256": file_sha256(scores_path),
            "proposal_mean_array_sha256": array_sha256(proposal_mean),
            "proposal_covariance_array_sha256": array_sha256(
                proposal_covariance
            ),
            "valid_candidates": int(valid[:, 1:].sum()),
            "safe_candidates": int(safe[:, 1:].sum()),
            "elite_candidates": int(updated.sum() * 4),
            "updated_windows": int(updated.sum()),
            "safe_shortfall_fallback_windows": int((~updated).sum()),
        }
        iteration_contracts.append(proposal_record)
        if first_population is None:
            first_population = proposal_record

    contract = {
        "schema": "cem_safe_elite_v55_preutility_contract",
        "status": "candidate_gate_and_paired_cem_comparison_frozen_before_true_utility",
        "evidence_role": "spent_development_only_after_outcome_evaluation",
        "blind_confirmation": False,
        "production_safe_claim_allowed": False,
        "cem": True,
        "state_gate_used": False,
        "candidate_safe_formula": (
            "valid AND (mean_ATE + z*std_ATE < -margin_ate) AND "
            "(mean_Objective + z*std_Objective < -margin_objective) AND "
            "(p_dual >= dual_threshold) AND (p_harm <= harm_threshold)"
        ),
        "thresholds": threshold_records,
        "scorer_training_result": str(training_path),
        "scorer_training_result_sha256": file_sha256(training_path),
        "scorer_checkpoints": checkpoint_records,
        "support_checkpoint": freeze["support_checkpoint"],
        "support_checkpoint_sha256": freeze["support_checkpoint_sha256"],
        "shortfall_rule": (
            "If safe_count < 4, do not update q and retain the previous proposal; "
            "never pad elites with unsafe candidates."
        ),
        "paired_comparison": {
            "cohort_manifest": str(manifest),
            "cohort_manifest_sha256": file_sha256(manifest),
            "states": len(manifest_rows),
            "routes": routes,
            "route_count": len(routes),
            "trajectories": trajectories,
            "trajectory_count": len(trajectories),
            "cem_false_pool": freeze["candidate_pool"],
            "cem_false_pool_sha256": freeze["candidate_pool_sha256"],
            "cem_true_seed": config["seed"],
            "shared_incumbent": True,
            "shared_scorer_and_candidate_safe_gate": True,
            "initial_proposal": first_population,
        },
        "iterations": iteration_contracts,
        "prelabel_freeze": str(args.freeze),
        "prelabel_freeze_sha256": file_sha256(args.freeze),
        "selected_actions": freeze["selected_actions"],
        "selected_actions_sha256": freeze["selected_actions_sha256"],
        "labels_opened": False,
        "outcomes_accessed": False,
        "reserve_accessed": False,
    }
    write_json_atomic(args.output, contract)
    print(json.dumps(contract, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
