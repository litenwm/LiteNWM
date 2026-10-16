#!/usr/bin/env python3
"""Amended pooled-safe LOBO calibration for the one shared v57 gate."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from tools.scoring.calibrate_cem_candidate_gate_lobo_v57 import (  # noqa: E402
    config_key,
    evaluate,
    load_bank,
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
    parser.add_argument("--amendment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def choose(
    configs: list[dict[str, float]],
    metrics: dict[str, dict[str, dict[str, float | int]]],
    banks: list[str],
) -> tuple[dict[str, float], dict[str, object]]:
    feasible = []
    for config in configs:
        values = [metrics[name][config_key(config)] for name in banks]
        coverage = [
            float(value["route_balanced_update_coverage"]) for value in values
        ]
        dual = [float(value["route_balanced_elite_dual"]) for value in values]
        harm = [float(value["route_balanced_elite_harm"]) for value in values]
        pooled_dual = sum(dual) / len(dual)
        pooled_harm = sum(harm) / len(harm)
        if (
            pooled_dual >= 0.85
            and pooled_harm <= 0.15
            and min(dual) >= 0.75
            and max(harm) <= 0.25
        ):
            score = (
                min(coverage),
                pooled_dual,
                -pooled_harm,
                config["beta"],
                config["benefit_threshold"],
                config["dual_threshold"],
                -config["harm_threshold"],
            )
            feasible.append(
                (
                    score,
                    config,
                    min(coverage),
                    pooled_dual,
                    pooled_harm,
                    min(dual),
                    max(harm),
                )
            )
    if not feasible:
        raise RuntimeError(f"no amended feasible v57 gate for {banks}")
    feasible.sort(key=lambda value: value[0], reverse=True)
    _, selected, coverage, dual, harm, min_dual, max_harm = feasible[0]
    return selected, {
        "feasible_configs": len(feasible),
        "minimum_training_bank_update_coverage": coverage,
        "pooled_training_dual": dual,
        "pooled_training_harm": harm,
        "minimum_training_bank_dual": min_dual,
        "maximum_training_bank_harm": max_harm,
    }


def main() -> None:
    args = parse_args()
    if len(args.bank_root) != 4:
        raise ValueError("expected A/D/B/C")
    if args.output.exists():
        raise FileExistsError(args.output)
    prereg = json.loads(args.preregistration.read_text(encoding="utf-8"))
    amendment = json.loads(args.amendment.read_text(encoding="utf-8"))
    if amendment["timing"]["E_F_outcomes_accessed"]:
        raise RuntimeError("E/F outcome contract was already opened")
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
        path.name: load_bank(path, args.trajectory_root) for path in args.bank_root
    }
    names = list(banks)
    metrics = {
        name: {config_key(config): evaluate(bank, config) for config in configs}
        for name, bank in banks.items()
    }
    folds = {}
    heldout_catastrophic_pass = []
    for heldout in names:
        train = [name for name in names if name != heldout]
        selected, selection = choose(configs, metrics, train)
        heldout_metric = metrics[heldout][config_key(selected)]
        safe = bool(
            heldout_metric["route_balanced_elite_dual"] >= 0.75
            and heldout_metric["route_balanced_elite_harm"] <= 0.25
        )
        heldout_catastrophic_pass.append(safe)
        folds[heldout] = {
            "training_banks": train,
            "selected": selected,
            "selection": selection,
            "heldout": heldout_metric,
            "heldout_catastrophic_guard_pass": safe,
        }
    global_config, global_selection = choose(configs, metrics, names)
    global_metrics = {
        name: value[config_key(global_config)] for name, value in metrics.items()
    }
    first_freeze = json.loads(
        (
            args.bank_root[0] / "cem_run_v55_safe_elite/prelabel_freeze.json"
        ).read_text(encoding="utf-8")
    )
    training_path = Path(first_freeze["training_result"])
    training = json.loads(training_path.read_text(encoding="utf-8"))
    checkpoints = []
    for checkpoint in training["arms"]["D"]["checkpoints"]["full"]:
        path = Path(checkpoint["path"])
        if file_sha256(path) != checkpoint["sha256"]:
            raise RuntimeError(f"checkpoint SHA changed: {path}")
        checkpoints.append(checkpoint)
    result = {
        "schema": "recon_cem_candidate_gate_lobo_v57_amended",
        "status": "one_shared_gate_frozen_before_E_F_outcomes",
        "configs_evaluated": len(configs),
        "calibration_banks": names,
        "folds": folds,
        "all_LOBO_heldout_catastrophic_guard_pass": all(
            heldout_catastrophic_pass
        ),
        "frozen_gate": global_config,
        "frozen_gate_selection": global_selection,
        "frozen_gate_calibration_metrics": global_metrics,
        "formula": prereg["candidate_gate_family"],
        "single_shared_cross_cohort_rule": True,
        "cohort_specific_thresholds": False,
        "scorer_training_result": str(training_path),
        "scorer_training_result_sha256": file_sha256(training_path),
        "scorer_checkpoints": checkpoints,
        "preregistration": str(args.preregistration),
        "preregistration_sha256": file_sha256(args.preregistration),
        "amendment": str(args.amendment),
        "amendment_sha256": file_sha256(args.amendment),
        "E_F_outcomes_accessed": False,
        "reserve_accessed": False,
        "state_gate_used": False,
        "production_safe_claim_allowed": False,
    }
    write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
