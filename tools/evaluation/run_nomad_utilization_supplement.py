#!/usr/bin/env python3
"""Audit NoMaD-x16 utilization by NWM-S and v70 on frozen official-100 data."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--recon-nwm", type=Path, required=True)
    parser.add_argument("--scand-nwm", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260726)
    return parser.parse_args()


def bootstrap(
    delta: np.ndarray,
    route: np.ndarray,
    *,
    draws: int,
    seed: int,
) -> dict[str, float | int]:
    unique = np.asarray(sorted(np.unique(route).tolist()))
    grouped = [delta[route == value] for value in unique]
    rng = np.random.default_rng(seed)
    sampled = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        choice = rng.integers(0, len(unique), size=len(unique))
        sampled[draw] = np.concatenate(
            [grouped[index] for index in choice]
        ).mean()
    return {
        "routes": int(len(unique)),
        "draws": int(draws),
        "mean": float(delta.mean()),
        "lcb95": float(np.quantile(sampled, 0.025)),
        "ucb95": float(np.quantile(sampled, 0.975)),
        "probability_positive": float((sampled > 0.0).mean()),
    }


def load_nwm(path: Path, record: np.ndarray) -> np.ndarray:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    observed = np.asarray(
        [f"{row['traj_name']}:{int(row['curr_time'])}" for row in rows]
    )
    if len(rows) != len(record) or not np.array_equal(observed, record):
        raise RuntimeError(f"NWM-S identity mismatch: {path}")
    return np.asarray(
        [int(row["nwm_selected_index"]) + 1 for row in rows],
        dtype=np.int64,
    )


def evaluate(
    domain: str,
    benchmark_root: Path,
    nwm_path: Path,
    *,
    draws: int,
    seed: int,
) -> tuple[dict, dict[str, np.ndarray]]:
    directory = benchmark_root / domain
    feature_path = (
        directory / "features" / f"official100_{domain}_features.npz"
    )
    with np.load(feature_path, allow_pickle=False) as archive:
        candidates = archive["candidates"].astype(np.float64) * 0.25
        auxiliary = archive["scorer_v2_aux_features"].astype(np.float64)
        record = archive["record_key"].astype(str)
    with np.load(
        directory / "v70_replay" / "selected_actions.npz",
        allow_pickle=False,
    ) as archive:
        safe = archive["base_safe"].astype(bool)
        v70_index = archive["fixed_pool_selected_index"].astype(np.int64)
    with np.load(
        directory / "selected_metrics.npz",
        allow_pickle=False,
    ) as archive:
        ground_truth = archive["ground_truth_local_m"].astype(np.float64)
    nwm_index = load_nwm(nwm_path, record)
    rows = np.arange(len(record))
    error = candidates - ground_truth[:, None]
    squared = np.sum(error * error, axis=-1)
    ate = np.sqrt(np.mean(squared, axis=-1))
    endpoint = np.sqrt(squared[:, :, -1])
    objective = 0.4 * ate + 0.6 * endpoint
    raw_energy = 0.4 * auxiliary[..., 6] + 0.6 * auxiliary[..., 8]
    project_index = np.argmin(raw_energy[:, 1:], axis=1) + 1
    oracle_index = np.argmin(objective[:, 1:], axis=1) + 1
    route = np.asarray(
        [
            f"{domain}::{json.loads(line)['route_group']}"
            for line in (
                directory / "manifest.windows.jsonl"
            ).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    )
    nwm_ate = ate[rows, nwm_index]
    nwm_objective = objective[rows, nwm_index]
    project_ate = ate[rows, project_index]
    project_objective = objective[rows, project_index]
    oracle_objective = objective[rows, oracle_index]
    nwm_top_safe = safe[rows, nwm_index]
    any_safe = safe[:, 1:].any(axis=1)
    opportunity = (~nwm_top_safe) & any_safe
    rescued = opportunity & (v70_index != 0)
    ate_gain = ate[:, 0] - ate[rows, v70_index]
    objective_gain = objective[:, 0] - objective[rows, v70_index]
    dual = rescued & (ate_gain > 0.0) & (objective_gain > 0.0)
    harm = rescued & ((ate_gain < 0.0) | (objective_gain < 0.0))
    rescued_count = int(rescued.sum())
    opportunity_count = int(opportunity.sum())
    result = {
        "states": int(len(record)),
        "nwm_s": {
            "ate_mean_m": float(nwm_ate.mean()),
            "objective_mean_m": float(nwm_objective.mean()),
            "nomad_oracle_regret_mean_m": float(
                (nwm_objective - oracle_objective).mean()
            ),
            "nomad_oracle_hit_rate": float(
                (nwm_index == oracle_index).mean()
            ),
        },
        "project_raw_nomad_x16": {
            "ate_mean_m": float(project_ate.mean()),
            "objective_mean_m": float(project_objective.mean()),
            "nomad_oracle_regret_mean_m": float(
                (project_objective - oracle_objective).mean()
            ),
            "nomad_oracle_hit_rate": float(
                (project_index == oracle_index).mean()
            ),
        },
        "project_raw_advantage_over_nwm_s": {
            "ate_m": bootstrap(
                nwm_ate - project_ate,
                route,
                draws=draws,
                seed=seed,
            ),
            "objective_m": bootstrap(
                nwm_objective - project_objective,
                route,
                draws=draws,
                seed=seed + 1,
            ),
        },
        "safe_candidate_rescue": {
            "nwm_top_safe_rate": float(nwm_top_safe.mean()),
            "any_safe_nomad_challenger_rate": float(any_safe.mean()),
            "nwm_top_unsafe_but_other_safe_exists_rate": float(
                opportunity.mean()
            ),
            "opportunity_states": opportunity_count,
            "rescued_states": rescued_count,
            "v70_rescue_fraction_given_opportunity": (
                float(rescued_count / opportunity_count)
                if opportunity_count
                else None
            ),
            "rescued_dual_precision": (
                float(dual.sum() / rescued_count) if rescued_count else None
            ),
            "rescued_harm_rate": (
                float(harm.sum() / rescued_count) if rescued_count else None
            ),
            "rescued_ate_gain_mean_m": (
                float(ate_gain[rescued].mean()) if rescued_count else None
            ),
            "rescued_objective_gain_mean_m": (
                float(objective_gain[rescued].mean())
                if rescued_count
                else None
            ),
        },
        "same_selected_index_rate": float(
            (nwm_index == project_index).mean()
        ),
    }
    return result, {
        "ate": nwm_ate - project_ate,
        "objective": nwm_objective - project_objective,
        "route": route,
    }


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output}")
    nwm_paths = {"recon": args.recon_nwm, "scand": args.scand_nwm}
    domains = {}
    arrays = {}
    for index, domain in enumerate(("recon", "scand")):
        domains[domain], arrays[domain] = evaluate(
            domain,
            args.benchmark_root,
            nwm_paths[domain],
            draws=args.draws,
            seed=args.seed + 10 * index,
        )
    pooled_route = np.concatenate([arrays[key]["route"] for key in arrays])
    result = {
        "schema": "nomad_utilization_supplement_v1",
        "status": "complete_exploratory",
        "positive_direction": "positive means project raw scorer is better",
        "domains": domains,
        "pooled200": {
            "project_raw_advantage_over_nwm_s": {
                metric: bootstrap(
                    np.concatenate([arrays[key][metric] for key in arrays]),
                    pooled_route,
                    draws=args.draws,
                    seed=args.seed + 100 + index,
                )
                for index, metric in enumerate(("ate", "objective"))
            }
        },
        "interpretation": (
            "The raw project scalar score is a negative control. Attribute the "
            "deployable v70 advantage to calibrated multi-head safety selection, "
            "fallback, and CEM/archive rather than raw argmin."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result["pooled200"], sort_keys=True))


if __name__ == "__main__":
    main()
