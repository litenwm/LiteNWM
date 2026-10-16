#!/usr/bin/env python3
"""Final automatic acceptance audit for B/C safe-elite add-only CEM."""

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
from tools.planning.run_cem_safe_elite_v55 import (  # noqa: E402
    proposal_summary,
    safe_elite_update,
)
from tools.scoring.train_recon_dense_meta_dinov3_v1 import (  # noqa: E402
    file_sha256,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank-root", type=Path, action="append", required=True)
    parser.add_argument("--pooled-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(args.bank_root) != 2:
        raise ValueError("exactly two B/C bank roots are required")
    if args.output.exists():
        raise FileExistsError(args.output)
    pooled = json.loads(args.pooled_result.read_text(encoding="utf-8"))
    if (
        pooled.get("status") != "promising_development_result_frozen"
        or not pooled["promising_development_checks"][
            "promising_development_result"
        ]
        or pooled.get("production_safe_claim_allowed")
        or pooled.get("reserve_accessed")
        or pooled.get("state_gate_used")
    ):
        raise RuntimeError("pooled promising-development contract is invalid")

    banks = {}
    route_sets = []
    for bank_root in args.bank_root:
        cem_root = bank_root / "cem_run_v55_safe_elite"
        hybrid_root = bank_root / "cem_safe_addonly_v56"
        cem_freeze_path = cem_root / "prelabel_freeze.json"
        contract_path = cem_root / "preutility_contract.json"
        hybrid_freeze_path = hybrid_root / "freeze.json"
        selected_path = hybrid_root / "selected_index.npz"
        result_path = hybrid_root / "result.json"
        cem_freeze = json.loads(cem_freeze_path.read_text(encoding="utf-8"))
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        hybrid_freeze = json.loads(
            hybrid_freeze_path.read_text(encoding="utf-8")
        )
        result = json.loads(result_path.read_text(encoding="utf-8"))
        bank = bank_root.name
        if (
            cem_freeze["config"]["population_challengers"] != 16
            or cem_freeze["config"]["elite_count"] != 4
            or cem_freeze.get("state_gate_used")
            or cem_freeze.get("reserve_accessed")
            or contract.get("outcomes_accessed")
            or contract.get("labels_opened")
            or hybrid_freeze.get("fixed_pool_action_replacement_allowed")
            or hybrid_freeze.get("reserve_accessed")
            or result.get("production_safe_claim_allowed")
            or result.get("reserve_accessed")
        ):
            raise RuntimeError(f"{bank}: frozen contract changed")
        if file_sha256(cem_freeze_path) != contract["prelabel_freeze_sha256"]:
            raise RuntimeError(f"{bank}: pre-utility freeze SHA mismatch")
        if file_sha256(contract_path) != hybrid_freeze["cem_contract_sha256"]:
            raise RuntimeError(f"{bank}: contract SHA mismatch")
        if file_sha256(hybrid_freeze_path) != result["freeze_sha256"]:
            raise RuntimeError(f"{bank}: hybrid freeze SHA mismatch")
        if file_sha256(selected_path) != hybrid_freeze["selected_index_sha256"]:
            raise RuntimeError(f"{bank}: selected-index SHA mismatch")

        with np.load(selected_path, allow_pickle=False) as payload:
            fixed = payload["safe_cem_false_selected_index"].astype(np.int64)
            hybrid = payload["hybrid_combined_selected_index"].astype(np.int64)
            added = payload["cem_added_window"].astype(bool)
        fixed_mask = fixed != 0
        if not np.array_equal(hybrid[fixed_mask], fixed[fixed_mask]):
            raise RuntimeError(f"{bank}: hybrid replaced fixed-safe action")
        if not np.array_equal(added, (~fixed_mask) & (hybrid != 0)):
            raise RuntimeError(f"{bank}: CEM marginal attribution changed")

        route_set = set(contract["paired_comparison"]["routes"])
        route_sets.append(route_set)
        iteration_reports = []
        previous_true = None
        for iteration in range(3):
            directory = cem_root / f"iteration_{iteration:02d}"
            population_path = directory / "population.npz"
            score_path = directory / "frozen_scores.npz"
            with np.load(population_path, allow_pickle=False) as payload:
                candidates = payload[
                    "candidates_feature_waypoint_units"
                ].astype(np.float32)
                proposal_mean = payload["proposal_mean"].astype(np.float32)
                proposal_covariance = payload["proposal_covariance"].astype(
                    np.float32
                )
            with np.load(score_path, allow_pickle=False) as payload:
                energy = payload["energy"].astype(np.float32)
                valid = payload["valid"].astype(bool)
                safe = payload["safe"].astype(bool)
                elite = payload["elite_index_for_update"].astype(np.int64)
                updated = payload["proposal_updated"].astype(bool)
                safe_count = payload["safe_count"].astype(np.int64)
            if candidates.shape != (100, 17, 4, 2):
                raise RuntimeError(f"{bank}/{iteration}: population is not K17")
            if not np.array_equal(safe_count, safe.sum(axis=1)):
                raise RuntimeError(f"{bank}/{iteration}: safe count mismatch")
            if not np.array_equal(updated, safe_count >= 4):
                raise RuntimeError(f"{bank}/{iteration}: update/fallback mismatch")
            for state in range(100):
                if updated[state] and not safe[state, elite[state]].all():
                    raise RuntimeError(f"{bank}/{iteration}: unsafe elite")
                if not updated[state] and not np.all(elite[state] == -1):
                    raise RuntimeError(f"{bank}/{iteration}: padded unsafe elite")

            next_mean, next_covariance, recomputed_elite, recomputed_updated, _ = (
                safe_elite_update(
                    candidates,
                    energy,
                    safe,
                    previous_mean=proposal_mean,
                    previous_covariance=proposal_covariance,
                    elite_count=4,
                    floor=float(
                        cem_freeze["config"][
                            "covariance_floor_waypoint_units"
                        ]
                    ),
                    scale=float(cem_freeze["config"]["proposal_scale"]),
                )
            )
            if not np.array_equal(recomputed_elite, elite):
                raise RuntimeError(f"{bank}/{iteration}: elite identity mismatch")
            if not np.array_equal(recomputed_updated, updated):
                raise RuntimeError(f"{bank}/{iteration}: update identity mismatch")
            fallback = ~updated
            if not np.array_equal(next_mean[fallback], proposal_mean[fallback]):
                raise RuntimeError(f"{bank}/{iteration}: fallback mean changed")
            if not np.array_equal(
                next_covariance[fallback], proposal_covariance[fallback]
            ):
                raise RuntimeError(f"{bank}/{iteration}: fallback covariance changed")
            if iteration < 2:
                with np.load(
                    cem_root / f"iteration_{iteration + 1:02d}" / "population.npz",
                    allow_pickle=False,
                ) as next_payload:
                    actual_next_mean = next_payload["proposal_mean"].astype(np.float32)
                    actual_next_covariance = next_payload[
                        "proposal_covariance"
                    ].astype(np.float32)
                if not np.array_equal(next_mean, actual_next_mean):
                    raise RuntimeError(f"{bank}/{iteration}: next mean not applied")
                if not np.array_equal(next_covariance, actual_next_covariance):
                    raise RuntimeError(
                        f"{bank}/{iteration}: next covariance not applied"
                    )

            audit = cem_freeze["iteration_audit"][iteration]
            utility = result["iteration_elite_predicted_vs_true"][iteration]
            true_change = (
                None
                if previous_true is None
                else float(utility["true_utility_mean"] - previous_true)
            )
            previous_true = float(utility["true_utility_mean"])
            iteration_reports.append(
                {
                    "iteration": iteration,
                    "population": 16,
                    "valid_candidates": int(valid[:, 1:].sum()),
                    "safe_candidates": int(safe[:, 1:].sum()),
                    "elite_candidates": int(updated.sum() * 4),
                    "updated_windows": int(updated.sum()),
                    "fallback_windows_safe_below_4": int((~updated).sum()),
                    "safe_elite_only_verified": True,
                    "fallback_proposal_exactly_unchanged_verified": True,
                    "proposal_before": proposal_summary(
                        proposal_mean, proposal_covariance
                    ),
                    "proposal_after": proposal_summary(
                        next_mean, next_covariance
                    ),
                    "support_acceptance": audit["support_acceptance"],
                    "support_fallback_samples": audit["fallback_samples"],
                    "ood_or_invalid_candidates": audit[
                        "ood_or_invalid_candidates"
                    ],
                    "predicted_scorer_utility_mean": utility[
                        "predicted_scorer_utility_mean"
                    ],
                    "true_utility_mean": utility["true_utility_mean"],
                    "true_utility_change_from_previous_iteration": true_change,
                    "elite_dual_rate": utility["elite_dual_rate"],
                    "elite_harm_rate": utility["elite_harm_rate"],
                }
            )
        banks[bank] = {
            "routes": len(route_set),
            "population_and_safe_elite_audit": iteration_reports,
            "fixed_safe_action_not_replaced_verified": True,
            "selected_index_sha256": file_sha256(selected_path),
            "preutility_freeze_sha256": file_sha256(cem_freeze_path),
            "preutility_contract_sha256": file_sha256(contract_path),
        }

    if route_sets[0] & route_sets[1]:
        raise RuntimeError("B/C route overlap")
    audit = {
        "schema": "recon_cem_safe_addonly_v56_final_acceptance_audit",
        "status": "all_automatic_acceptance_checks_passed",
        "evidence_role": "promising_development_not_production_safe",
        "banks": banks,
        "route_overlap": 0,
        "pooled_result": str(args.pooled_result),
        "pooled_result_sha256": file_sha256(args.pooled_result),
        "promising_development": True,
        "production_safe": False,
        "state_gate_used": False,
        "reserve_accessed": False,
        "calibration_substudy_complete": True,
    }
    write_json_atomic(args.output, audit)
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
