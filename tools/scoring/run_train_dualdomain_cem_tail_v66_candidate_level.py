#!/usr/bin/env python3
"""Train v66 with exact universal-valid, route-OOF, candidate-level CEM calibration."""

from __future__ import annotations

import numpy as np

from tools.planning.run_cem_safe_elite_v55 import (
    candidate_safety_mask,
    select_lowest_energy_safe,
)
from tools.scoring import run_train_dualdomain_cem_tail_v66_universal_valid as universal
from tools.scoring.train_universal_dualdomain_raw_v2 import (
    policy_feasible,
    policy_grid,
    safety_metrics,
)


implementation = universal.implementation


def candidate_energy(prediction, p_dual, p_harm, valid, policy):
    """Exact frozen CEM energy, using already temperature-scaled probabilities."""
    z = float(policy["z"])
    ucb = prediction["mean"] + z * prediction["std"]
    rank = prediction["rank_logit"].astype(np.float64)
    rank_scale = np.maximum(rank[:, 1:].std(axis=1, keepdims=True), 1.0e-3)
    rank_z = (rank - rank[:, :1]) / rank_scale
    energy = (
        ucb[..., 0]
        + 0.25 * ucb[..., 1]
        + 0.25 * np.maximum(ucb[..., 2], 0.0)
        + 0.50 * ucb[..., 3]
        + 0.15 * p_harm
        - 0.05 * p_dual
        - 0.03 * rank_z
    )
    energy = np.where(valid, energy, np.inf)
    energy[:, 0] = np.inf
    return energy.astype(np.float32)


def choose_candidate_level_cem_frontier(domains, calibration, *, shared_policy):
    """Calibrate the policy against the same safe-then-energy CEM decision."""
    if shared_policy:
        raise RuntimeError("v66 preregisters per-domain candidate-level policies")
    results = {}
    for name, value in domains.items():
        frontier = []
        for policy in policy_grid():
            energy = candidate_energy(
                value["prediction"],
                value["p_dual"],
                value["p_harm"],
                value["valid"],
                policy,
            )
            safe = candidate_safety_mask(
                value["prediction"],
                value["p_dual"],
                value["p_harm"],
                value["valid"],
                policy,
            )
            selected = select_lowest_energy_safe(energy, safe)
            metrics = safety_metrics(selected, value["targets"])
            feasible = policy_feasible({name: metrics}, calibration)
            frontier.append(
                {
                    "policy": policy,
                    "metrics": metrics,
                    "feasible": feasible,
                    "selected": selected,
                }
            )
        feasible_values = [entry for entry in frontier if entry["feasible"]]
        pool = feasible_values or frontier
        best = min(
            pool,
            key=lambda entry: (
                -entry["metrics"]["coverage"] if feasible_values else 0.0,
                entry["metrics"]["net_delta_per_state_m"]["objective"],
                -(entry["metrics"]["dual_precision"] or 0.0),
            ),
        )
        results[name] = {
            "feasible": bool(feasible_values),
            "policy": best["policy"],
            "metrics": best["metrics"],
            "selected": best["selected"],
            "frontier": [
                {key: entry[key] for key in ("policy", "metrics", "feasible")}
                for entry in frontier
            ],
        }
    return {"shared": False, "domains": results}


implementation.choose_frontier = choose_candidate_level_cem_frontier


if __name__ == "__main__":
    implementation.main()
