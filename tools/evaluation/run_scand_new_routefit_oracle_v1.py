#!/usr/bin/env python3
"""Read-only proposal funnel/oracle diagnostic for a frozen 65-candidate sidecar."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from planning.proposal_tail_sidecar_v1 import canonical_json, sha256_file  # noqa: E402
from tools.planning import build_proposal_tail_temporal_pair_v1 as pair  # noqa: E402


TOLERANCE = 1.0e-9


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=10000)
    return parser.parse_args()


def load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as value:
        return {key: np.asarray(value[key]) for key in value.files}


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> list[float | None]:
    if total == 0:
        return [None, None]
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    radius = z * np.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denominator
    return [float(center - radius), float(center + radius)]


def summary(metrics: np.ndarray) -> dict:
    names = ("ate", "rpe", "endpoint", "objective")
    return {
        "mean_m": {name: float(metrics[:, index].mean()) for index, name in enumerate(names)},
        "p90_m": {name: float(np.quantile(metrics[:, index], 0.9)) for index, name in enumerate(names)},
    }


def selection_summary(true_metrics: np.ndarray, selected: np.ndarray, reference: np.ndarray) -> dict:
    row = np.arange(len(selected))
    chosen = true_metrics[row, selected]
    base = true_metrics[row, reference]
    gains = base - chosen
    replaced = selected != reference
    dual = (gains[:, 0] > TOLERANCE) & (gains[:, 3] > TOLERANCE) & replaced
    harm = ((gains[:, 0] < -TOLERANCE) | (gains[:, 3] < -TOLERANCE)) & replaced
    count = int(replaced.sum())
    return {
        **summary(chosen),
        "replacements": count,
        "coverage": float(replaced.mean()),
        "objective_gain_vs_reference_m": float(gains[:, 3].mean()),
        "ate_gain_vs_reference_m": float(gains[:, 0].mean()),
        "dual": {"count": int(dual.sum()), "rate": None if count == 0 else float(dual.sum() / count), "wilson95": wilson(int(dual.sum()), count)},
        "harm": {"count": int(harm.sum()), "rate": None if count == 0 else float(harm.sum() / count), "wilson95": wilson(int(harm.sum()), count)},
    }


def oracle(true_metrics: np.ndarray, eligible: np.ndarray, reference: np.ndarray) -> np.ndarray:
    mask = eligible.copy()
    mask[np.arange(len(reference)), reference] = True
    objective = np.where(mask, true_metrics[..., 3], np.inf)
    return np.argmin(objective, axis=1)


def route_cluster_ci(delta: np.ndarray, routes: np.ndarray, repetitions: int) -> list[float]:
    unique = np.unique(routes)
    local = [delta[routes == route] for route in unique]
    rng = np.random.default_rng(2026072301)
    values = np.empty(repetitions, dtype=np.float64)
    for index in range(repetitions):
        chosen = rng.integers(0, len(unique), size=len(unique))
        values[index] = np.concatenate([local[item] for item in chosen]).mean()
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def main() -> None:
    args = parse_args()
    prereg = json.loads(args.preregistration.read_text(encoding="utf-8"))
    if prereg.get("status") != "frozen_before_candidate_generation_and_outcomes":
        raise RuntimeError("pilot preregistration is not frozen")
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest = json.loads((args.sidecar / "collection_manifest.json").read_text(encoding="utf-8"))
    table = load(args.sidecar / "candidate_table.npz")
    formal = load(args.sidecar / "formal_selection_snapshot.npz")
    labels = load(args.labels)
    if not np.array_equal(table["record_key"].astype(str), labels["record_key"].astype(str)):
        raise RuntimeError("label/table record mismatch")
    if not np.array_equal(table["candidate_hash"].astype(str), labels["candidate_hash"].astype(str)):
        raise RuntimeError("label/table candidate mismatch")
    pools = {source: load(args.sidecar / relative) for source, relative in pair.POOL_PATH.items()}
    candidates, _ = pair.gather_union(table, pools, ("candidates_feature_waypoint_units", "candidates"))
    if not np.array_equal(candidates.astype(np.float32), table["candidates_feature_waypoint_units"].astype(np.float32)):
        raise RuntimeError("candidate union mismatch")
    physical, _ = pair.gather_union(table, pools, ("physical_mask",))
    support, _ = pair.gather_union(table, pools, ("support_mask",))
    valid, _ = pair.gather_union(table, pools, ("valid",))
    safe, _ = pair.gather_union(table, pools, ("safe",))
    energy, _ = pair.gather_union(table, pools, ("energy",))
    true_metrics = labels["true_metrics_m"].astype(np.float64)
    formal_index = pair.formal_union_index(table, formal)
    rounds = table["source_round"].astype(np.int64)
    local = table["source_candidate_index"].astype(np.int64)
    fixed_index = np.empty(len(rounds), dtype=np.int64)
    incumbent_index = np.empty(len(rounds), dtype=np.int64)
    for state in range(len(rounds)):
        fixed_match = np.flatnonzero((rounds[state] == -1) & (local[state] == int(formal["fixed_pool_selected_index"][state])))
        incumbent_match = np.flatnonzero((rounds[state] == -1) & (local[state] == 0))
        if len(fixed_match) != 1 or len(incumbent_match) != 1:
            raise RuntimeError(f"fixed/incumbent mapping failure at state {state}")
        fixed_index[state] = fixed_match[0]
        incumbent_index[state] = incumbent_match[0]

    states = [json.loads(line) for line in (args.sidecar / "states.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    routes = np.asarray([row["route"] for row in states])
    row = np.arange(len(routes))
    formal_metrics = true_metrics[row, formal_index]
    gains_vs_formal = formal_metrics[:, None] - true_metrics
    dual_vs_formal = (gains_vs_formal[..., 0] > TOLERANCE) & (gains_vs_formal[..., 3] > TOLERANCE)
    harm_vs_formal = (gains_vs_formal[..., 0] < -TOLERANCE) | (gains_vs_formal[..., 3] < -TOLERANCE)
    masks = {
        "full": np.ones_like(valid, dtype=bool),
        "physical": physical.astype(bool),
        "physical_support": physical.astype(bool) & support.astype(bool),
        "runner_valid": valid.astype(bool),
        "old_safety": safe.astype(bool),
    }
    methods = {
        "incumbent": selection_summary(true_metrics, incumbent_index, incumbent_index),
        "v70_noncem_fixed": selection_summary(true_metrics, fixed_index, incumbent_index),
        "formal_v70_cem": selection_summary(true_metrics, formal_index, fixed_index),
    }
    opportunity = {}
    oracle_index = {}
    for name, mask in masks.items():
        selected = oracle(true_metrics, mask, formal_index)
        oracle_index[name] = selected
        methods[f"oracle_{name}"] = selection_summary(true_metrics, selected, formal_index)
        candidates_dual = mask & dual_vs_formal
        opportunity[name] = {
            "eligible_candidate_count": int(mask.sum()),
            "states_with_any_eligible": int(mask.any(axis=1).sum()),
            "dual_candidate_count_vs_formal_v70": int(candidates_dual.sum()),
            "states_with_dual_opportunity_vs_formal_v70": int(candidates_dual.any(axis=1).sum()),
            "dual_candidate_purity": float(candidates_dual.sum() / mask.sum()),
            "harm_candidate_count_vs_formal_v70": int((mask & harm_vs_formal).sum()),
            "harm_candidate_rate": float((mask & harm_vs_formal).sum() / mask.sum()),
        }

    route_results = {}
    for route in sorted(np.unique(routes)):
        state = np.flatnonzero(routes == route)
        route_results[route] = {
            "states": int(len(state)),
            "formal_v70_objective_mean_m": float(formal_metrics[state, 3].mean()),
            "physical_support_oracle_gain_m": float((formal_metrics[state, 3] - true_metrics[state, oracle_index["physical_support"][state], 3]).mean()),
            "physical_support_dual_opportunity_states": int((masks["physical_support"][state] & dual_vs_formal[state]).any(axis=1).sum()),
            "old_safety_dual_opportunity_states": int((masks["old_safety"][state] & dual_vs_formal[state]).any(axis=1).sum()),
        }
    oracle_delta = formal_metrics[:, 3] - true_metrics[row, oracle_index["physical_support"], 3]
    result = {
        "schema": "scand_new_routefit_oracle_diagnostic_v1",
        "status": "development_oracle_complete",
        "evidence_scope": "four-route feasibility diagnostic; not independent validation",
        "states": len(routes),
        "routes": int(len(np.unique(routes))),
        "candidates_per_state": int(true_metrics.shape[1]),
        "metric_order": [str(value) for value in labels["metric_names"]],
        "unit": "meter; lower is better",
        "methods": methods,
        "funnel": opportunity,
        "route_results": route_results,
        "physical_support_oracle_objective_gain_route_cluster_bootstrap95_m": route_cluster_ci(oracle_delta, routes, args.bootstrap),
        "actual_formal_energy_finite": int(np.isfinite(energy).sum()),
        "inputs": {
            "preregistration_sha256": sha256_file(args.preregistration),
            "collection_manifest_sha256": sha256_file(args.sidecar / "collection_manifest.json"),
            "candidate_table_sha256": sha256_file(args.sidecar / "candidate_table.npz"),
            "formal_snapshot_sha256": sha256_file(args.sidecar / "formal_selection_snapshot.npz"),
            "labels_sha256": sha256_file(args.labels),
        },
        "reserve_accessed": False,
        "validation_labels_opened": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(canonical_json(result) + "\n", encoding="utf-8")
    print(canonical_json({"output": str(args.output), "output_sha256": sha256_file(args.output), "status": result["status"]}))


if __name__ == "__main__":
    main()
