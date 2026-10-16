#!/usr/bin/env python3
"""Route-balanced LOBO calibration of the single v57 CEM candidate gate."""

from __future__ import annotations

import argparse
import itertools
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
from tools.scoring.train_recon_dense_meta_dinov3_v1 import (  # noqa: E402
    file_sha256,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank-root", type=Path, action="append", required=True)
    parser.add_argument("--trajectory-root", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_bank(bank_root: Path, trajectory_root: Path) -> dict[str, object]:
    cem_root = bank_root / "cem_run_v55_safe_elite"
    freeze = json.loads(
        (cem_root / "prelabel_freeze.json").read_text(encoding="utf-8")
    )
    rows = read_rows(Path(freeze["manifest"]))
    ground_truth = np.stack(
        [
            local_ground_truth(
                trajectory_root / str(row["trajectory_id"]) / "traj_data.pkl",
                int(row["current_t"]),
            )
            for row in rows
        ]
    ).astype(np.float32)
    route = np.asarray([str(row["route_id"]) for row in rows])
    with np.load(cem_root / "selected_actions.npz", allow_pickle=False) as payload:
        fixed = payload["universal_d_cem_false_selected_index"].astype(np.int64)
    iterations = []
    for iteration in range(3):
        directory = cem_root / f"iteration_{iteration:02d}"
        with np.load(directory / "population.npz", allow_pickle=False) as payload:
            candidates = payload["candidates_feature_waypoint_units"].astype(
                np.float32
            )
        with np.load(directory / "frozen_scores.npz", allow_pickle=False) as payload:
            mean = payload["mean"].astype(np.float32)
            std = payload["std"].astype(np.float32)
            p_dual = payload["dual_probability"].astype(np.float32)
            p_harm = payload["harm_probability"].astype(np.float32)
            valid = payload["valid"].astype(bool)
        absolute = metric_labels(candidates, ground_truth, 0.25, 0.4, 0.6)[
            "values"
        ]
        delta = absolute - absolute[:, :1]
        dual = (delta[..., 0] < -1.0e-7) & (delta[..., 3] < -1.0e-7)
        harm = (delta[..., 0] > 1.0e-7) | (delta[..., 3] > 1.0e-7)
        iterations.append(
            {
                "mean": mean,
                "std": std,
                "p_dual": p_dual,
                "p_harm": p_harm,
                "valid": valid,
                "dual": dual,
                "harm": harm,
            }
        )
    return {
        "name": bank_root.name,
        "route": route,
        "abstain": fixed == 0,
        "iterations": iterations,
    }


def evaluate(
    bank: dict[str, object],
    config: dict[str, float],
) -> dict[str, float | int]:
    route = np.asarray(bank["route"])
    abstain = np.asarray(bank["abstain"], dtype=bool)
    update_by_route = {value: [] for value in np.unique(route)}
    dual_by_route = {value: [] for value in np.unique(route)}
    harm_by_route = {value: [] for value in np.unique(route)}
    accepted = 0
    updated_total = 0
    for data in bank["iterations"]:
        mean = data["mean"]
        std = data["std"]
        p_dual = data["p_dual"]
        p_harm = data["p_harm"]
        valid = data["valid"].copy()
        valid[:, 0] = False
        benefit_lcb = -0.5 * (mean[..., 0] + mean[..., 3]) - config[
            "beta"
        ] * np.sqrt(std[..., 0] ** 2 + std[..., 3] ** 2)
        safe = (
            valid
            & (benefit_lcb >= config["benefit_threshold"])
            & (p_dual >= config["dual_threshold"])
            & (p_harm <= config["harm_threshold"])
        )
        energy = -benefit_lcb + 0.25 * p_harm - 0.10 * p_dual
        for state in range(len(abstain)):
            if not abstain[state]:
                continue
            index = np.flatnonzero(safe[state])
            update = len(index) >= 4
            update_by_route[route[state]].append(float(update))
            if not update:
                continue
            order = index[
                np.argsort(energy[state, index], kind="stable")[:4]
            ]
            dual_by_route[route[state]].extend(
                data["dual"][state, order].astype(float).tolist()
            )
            harm_by_route[route[state]].extend(
                data["harm"][state, order].astype(float).tolist()
            )
            accepted += 4
            updated_total += 1
    route_coverage = [
        float(np.mean(value)) for value in update_by_route.values() if value
    ]
    route_dual = [
        float(np.mean(value)) for value in dual_by_route.values() if value
    ]
    route_harm = [
        float(np.mean(value)) for value in harm_by_route.values() if value
    ]
    return {
        "route_balanced_update_coverage": float(np.mean(route_coverage)),
        "route_balanced_elite_dual": (
            float(np.mean(route_dual)) if route_dual else 0.0
        ),
        "route_balanced_elite_harm": (
            float(np.mean(route_harm)) if route_harm else 1.0
        ),
        "updated_state_iterations": int(updated_total),
        "elite_candidates": int(accepted),
        "routes_with_elites": int(len(route_dual)),
    }


def config_key(config: dict[str, float]) -> str:
    return (
        f"b{config['beta']:.3f}_u{config['benefit_threshold']:.3f}_"
        f"d{config['dual_threshold']:.3f}_h{config['harm_threshold']:.3f}"
    )


def choose(
    configs: list[dict[str, float]],
    metrics: dict[str, dict[str, dict[str, float | int]]],
    banks: list[str],
) -> tuple[dict[str, float], dict[str, object]]:
    feasible = []
    for config in configs:
        key = config_key(config)
        values = [metrics[name][key] for name in banks]
        min_coverage = min(
            float(value["route_balanced_update_coverage"]) for value in values
        )
        min_dual = min(
            float(value["route_balanced_elite_dual"]) for value in values
        )
        max_harm = max(
            float(value["route_balanced_elite_harm"]) for value in values
        )
        if min_dual >= 0.85 and max_harm <= 0.15:
            score = (
                min_coverage,
                min_dual,
                -max_harm,
                config["beta"],
                config["benefit_threshold"],
                config["dual_threshold"],
                -config["harm_threshold"],
            )
            feasible.append((score, config, min_coverage, min_dual, max_harm))
    if not feasible:
        raise RuntimeError(f"no feasible v57 gate for banks {banks}")
    feasible.sort(key=lambda value: value[0], reverse=True)
    _, selected, coverage, dual, harm = feasible[0]
    return selected, {
        "feasible_configs": len(feasible),
        "minimum_training_bank_update_coverage": coverage,
        "minimum_training_bank_dual": dual,
        "maximum_training_bank_harm": harm,
    }


def main() -> None:
    args = parse_args()
    if len(args.bank_root) != 4:
        raise ValueError("expected A/D/B/C calibration banks")
    if args.output.exists():
        raise FileExistsError(args.output)
    prereg = json.loads(args.preregistration.read_text(encoding="utf-8"))
    grid = prereg["grid"]
    configs = [
        {
            "beta": float(beta),
            "benefit_threshold": float(benefit),
            "dual_threshold": float(dual),
            "harm_threshold": float(harm),
        }
        for beta, benefit, dual, harm in itertools.product(
            grid["beta"],
            grid["benefit_threshold"],
            grid["dual_threshold"],
            grid["harm_threshold"],
        )
    ]
    banks = {
        bank_root.name: load_bank(bank_root, args.trajectory_root)
        for bank_root in args.bank_root
    }
    bank_names = list(banks)
    metrics = {
        name: {config_key(config): evaluate(bank, config) for config in configs}
        for name, bank in banks.items()
    }
    folds = {}
    heldout_pass = []
    for heldout in bank_names:
        train = [name for name in bank_names if name != heldout]
        selected, selection = choose(configs, metrics, train)
        heldout_metric = metrics[heldout][config_key(selected)]
        passed = bool(
            heldout_metric["route_balanced_elite_dual"] >= 0.85
            and heldout_metric["route_balanced_elite_harm"] <= 0.15
        )
        heldout_pass.append(passed)
        folds[heldout] = {
            "training_banks": train,
            "selected": selected,
            "selection": selection,
            "heldout": heldout_metric,
            "heldout_safety_pass": passed,
        }
    global_config, global_selection = choose(configs, metrics, bank_names)
    global_metrics = {
        name: values[config_key(global_config)] for name, values in metrics.items()
    }

    training = json.loads(
        Path(
            json.loads(
                (
                    args.bank_root[0]
                    / "cem_run_v55_safe_elite/prelabel_freeze.json"
                ).read_text(encoding="utf-8")
            )["training_result"]
        ).read_text(encoding="utf-8")
    )
    checkpoints = []
    for checkpoint in training["arms"]["D"]["checkpoints"]["full"]:
        path = Path(checkpoint["path"])
        if file_sha256(path) != checkpoint["sha256"]:
            raise RuntimeError(f"checkpoint SHA changed: {path}")
        checkpoints.append(checkpoint)
    result = {
        "schema": "recon_cem_candidate_gate_lobo_v57",
        "status": "single_v57_gate_frozen_before_E_F_outcomes",
        "calibration_banks": bank_names,
        "route_overlap": 0,
        "configs_evaluated": len(configs),
        "folds": folds,
        "all_LOBO_heldout_safety_pass": all(heldout_pass),
        "frozen_gate": global_config,
        "frozen_gate_selection": global_selection,
        "frozen_gate_calibration_metrics": global_metrics,
        "formula": prereg["candidate_gate_family"],
        "scorer_checkpoints": checkpoints,
        "preregistration": str(args.preregistration),
        "preregistration_sha256": file_sha256(args.preregistration),
        "E_F_outcomes_accessed": False,
        "reserve_accessed": False,
        "state_gate_used": False,
        "production_safe_claim_allowed": False,
    }
    write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
