#!/usr/bin/env python3
"""Outcome-aware spent-development audit of the CEM abstain funnel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from tools.evaluation.build_recon100_k17_source_v1 import metric_labels  # noqa: E402
from tools.evaluation.build_recon_inner_k17_source_v2 import (  # noqa: E402
    local_ground_truth,
)
from tools.scoring.train_recon_dense_meta_dinov3_v1 import write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank-root", type=Path, action="append", required=True)
    parser.add_argument("--trajectory-root", type=Path, required=True)
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
    if len(args.bank_root) != 4:
        raise ValueError("expected spent banks A/D/B/C")
    if args.output.exists():
        raise FileExistsError(args.output)
    bank_reports = {}
    route_sets = []
    pooled_flags = []
    pooled_iteration = {iteration: [] for iteration in range(3)}
    for bank_root in args.bank_root:
        cem_root = bank_root / "cem_run_v55_safe_elite"
        freeze = json.loads(
            (cem_root / "prelabel_freeze.json").read_text(encoding="utf-8")
        )
        if freeze.get("state_gate_used") or freeze.get("reserve_accessed"):
            raise RuntimeError("invalid spent-development CEM contract")
        rows = read_rows(Path(freeze["manifest"]))
        ground_truth = np.stack(
            [
                local_ground_truth(
                    args.trajectory_root
                    / str(row["trajectory_id"])
                    / "traj_data.pkl",
                    int(row["current_t"]),
                )
                for row in rows
            ]
        ).astype(np.float32)
        routes = {str(row["route_id"]) for row in rows}
        route_sets.append(routes)
        with np.load(cem_root / "selected_actions.npz", allow_pickle=False) as payload:
            fixed = payload["universal_d_cem_false_selected_index"].astype(
                np.int64
            )
            final_local = payload["universal_d_cem_true_selected_index"].astype(
                np.int64
            )
        abstain = fixed == 0
        oracle_any = np.zeros(100, dtype=bool)
        valid_oracle_any = np.zeros(100, dtype=bool)
        safe_oracle_any = np.zeros(100, dtype=bool)
        update_any = np.zeros(100, dtype=bool)
        iteration_reports = []
        final_dual = np.zeros(100, dtype=bool)
        final_harm = np.zeros(100, dtype=bool)
        for iteration in range(3):
            directory = cem_root / f"iteration_{iteration:02d}"
            with np.load(directory / "population.npz", allow_pickle=False) as payload:
                candidates = payload[
                    "candidates_feature_waypoint_units"
                ].astype(np.float32)
            with np.load(directory / "frozen_scores.npz", allow_pickle=False) as payload:
                valid = payload["valid"].astype(bool)
                safe = payload["safe"].astype(bool)
                energy = payload["energy"].astype(np.float32)
                elite = payload["elite_index_for_update"].astype(np.int64)
                updated = payload["proposal_updated"].astype(bool)
            absolute = metric_labels(candidates, ground_truth, 0.25, 0.4, 0.6)[
                "values"
            ]
            delta = absolute - absolute[:, :1]
            dual = (delta[..., 0] < -1.0e-7) & (delta[..., 3] < -1.0e-7)
            harm = (delta[..., 0] > 1.0e-7) | (delta[..., 3] > 1.0e-7)
            helpful = dual & ~harm
            helpful[:, 0] = False
            oracle_any |= helpful.any(axis=1)
            valid_oracle_any |= (helpful & valid).any(axis=1)
            safe_oracle_any |= (helpful & valid & safe).any(axis=1)
            update_any |= updated
            elite_state = np.repeat(np.flatnonzero(updated), 4)
            elite_candidate = elite[updated].reshape(-1)
            elite_delta = delta[elite_state, elite_candidate]
            predicted = -energy[elite_state, elite_candidate]
            true_utility = -0.5 * (elite_delta[:, 0] + elite_delta[:, 3])
            elite_dual = (elite_delta[:, 0] < -1.0e-7) & (
                elite_delta[:, 3] < -1.0e-7
            )
            elite_harm = (elite_delta[:, 0] > 1.0e-7) | (
                elite_delta[:, 3] > 1.0e-7
            )
            correlation = (
                float(np.corrcoef(predicted, true_utility)[0, 1])
                if predicted.std() > 0 and true_utility.std() > 0
                else None
            )
            iteration_report = {
                "iteration": iteration,
                "sampled_candidates": int(abstain.sum() * 16),
                "abstain_states_with_true_oracle": int(
                    (abstain & helpful.any(axis=1)).sum()
                ),
                "abstain_states_with_valid_true_oracle": int(
                    (abstain & (helpful & valid).any(axis=1)).sum()
                ),
                "abstain_states_with_safe_true_oracle": int(
                    (abstain & (helpful & valid & safe).any(axis=1)).sum()
                ),
                "updated_abstain_states": int((abstain & updated).sum()),
                "elite_candidates": int(len(elite_state)),
                "predicted_utility_mean": float(predicted.mean()),
                "true_utility_mean": float(true_utility.mean()),
                "predicted_true_correlation": correlation,
                "elite_dual": float(elite_dual.mean()),
                "elite_harm": float(elite_harm.mean()),
            }
            iteration_reports.append(iteration_report)
            pooled_iteration[iteration].append(
                {
                    "predicted": predicted,
                    "true": true_utility,
                    "dual": elite_dual,
                    "harm": elite_harm,
                }
            )
            if iteration == 2:
                row_index = np.arange(100)
                selected_delta = delta[row_index, final_local]
                final_dual = (selected_delta[:, 0] < -1.0e-7) & (
                    selected_delta[:, 3] < -1.0e-7
                )
                final_harm = (selected_delta[:, 0] > 1.0e-7) | (
                    selected_delta[:, 3] > 1.0e-7
                )
        opportunity = abstain & oracle_any
        valid_stage = opportunity & valid_oracle_any
        safe_stage = valid_stage & safe_oracle_any
        update_stage = safe_stage & update_any
        final_safe_stage = update_stage & (final_local != 0)
        addonly = abstain & (final_local != 0)
        report = {
            "states": 100,
            "routes": len(routes),
            "fixed_pool_abstain": int(abstain.sum()),
            "funnel": {
                "sampled_true_dual_harmfree_opportunity": int(opportunity.sum()),
                "opportunity_support_valid": int(valid_stage.sum()),
                "opportunity_candidate_gate_safe": int(safe_stage.sum()),
                "opportunity_safe4_q_updated": int(update_stage.sum()),
                "opportunity_final_safe_candidate": int(final_safe_stage.sum()),
                "addonly_promotions_all": int(addonly.sum()),
                "addonly_true_dual": int((addonly & final_dual).sum()),
                "addonly_true_harm": int((addonly & final_harm).sum()),
            },
            "iteration": iteration_reports,
            "oracle_addonly_coverage_upper_bound": float(opportunity.mean()),
        }
        bank_reports[bank_root.name] = report
        pooled_flags.append(
            {
                "abstain": abstain,
                "opportunity": opportunity,
                "valid": valid_stage,
                "safe": safe_stage,
                "update": update_stage,
                "final": final_safe_stage,
                "add": addonly,
                "add_dual": addonly & final_dual,
                "add_harm": addonly & final_harm,
            }
        )
    for index, first in enumerate(route_sets):
        for second in route_sets[index + 1 :]:
            if first & second:
                raise RuntimeError("spent banks are not route-disjoint")

    pooled_funnel = {
        key: int(sum(item[key].sum() for item in pooled_flags))
        for key in (
            "abstain",
            "opportunity",
            "valid",
            "safe",
            "update",
            "final",
            "add",
            "add_dual",
            "add_harm",
        )
    }
    pooled_optimizer = []
    for iteration in range(3):
        predicted = np.concatenate(
            [item["predicted"] for item in pooled_iteration[iteration]]
        )
        true = np.concatenate([item["true"] for item in pooled_iteration[iteration]])
        dual = np.concatenate([item["dual"] for item in pooled_iteration[iteration]])
        harm = np.concatenate([item["harm"] for item in pooled_iteration[iteration]])
        pooled_optimizer.append(
            {
                "iteration": iteration,
                "elite_candidates": len(predicted),
                "predicted_utility_mean": float(predicted.mean()),
                "true_utility_mean": float(true.mean()),
                "predicted_true_correlation": float(
                    np.corrcoef(predicted, true)[0, 1]
                ),
                "elite_dual": float(dual.mean()),
                "elite_harm": float(harm.mean()),
            }
        )
    predicted_gain = (
        pooled_optimizer[-1]["predicted_utility_mean"]
        - pooled_optimizer[0]["predicted_utility_mean"]
    )
    true_gain = (
        pooled_optimizer[-1]["true_utility_mean"]
        - pooled_optimizer[0]["true_utility_mean"]
    )
    losses = {
        "proposal_no_oracle": pooled_funnel["abstain"]
        - pooled_funnel["opportunity"],
        "support_rejected_oracle": pooled_funnel["opportunity"]
        - pooled_funnel["valid"],
        "gate_rejected_valid_oracle": pooled_funnel["valid"]
        - pooled_funnel["safe"],
        "safe4_update_loss": pooled_funnel["safe"]
        - pooled_funnel["update"],
        "final_selector_loss_after_update": pooled_funnel["update"]
        - pooled_funnel["final"],
    }
    dominant = max(losses, key=losses.get)
    result = {
        "schema": "recon_cem_abstain_funnel_v57",
        "evidence_role": "outcome_aware_spent_development_diagnostic_only",
        "banks": bank_reports,
        "states": 400,
        "routes": sum(len(value) for value in route_sets),
        "route_overlap": 0,
        "pooled_funnel": pooled_funnel,
        "funnel_losses": losses,
        "dominant_observed_loss": dominant,
        "optimizer_exploitation": {
            "iteration": pooled_optimizer,
            "predicted_utility_gain_iteration0_to_2": float(predicted_gain),
            "true_utility_gain_iteration0_to_2": float(true_gain),
            "gain_ratio_predicted_over_true": (
                float(predicted_gain / true_gain) if true_gain > 1.0e-9 else None
            ),
            "confirmed": bool(predicted_gain > 0.20 and true_gain < 0.01),
        },
        "oracle_opportunity_at_least_20_per_200": bool(
            pooled_funnel["opportunity"] >= 40
        ),
        "reserve_accessed": False,
        "next_action": (
            "calibrate candidate safety/risk energy once with route-held-out spent "
            "banks, then freeze and run E/F"
        ),
    }
    write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
