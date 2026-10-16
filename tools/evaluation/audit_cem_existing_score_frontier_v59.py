#!/usr/bin/env python3
"""Audit the shared online policy frontier over spent A-F score archives."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scoring.spacing_support_v1 import threshold_for_quantile  # noqa: E402
from scoring.universal_dualdomain_raw_v2 import (  # noqa: E402
    build_universal_features,
)
from tools.evaluation.build_recon100_k17_source_v1 import metric_labels  # noqa: E402
from tools.evaluation.build_recon_inner_k17_source_v2 import (  # noqa: E402
    local_ground_truth,
)
from tools.evaluation.evaluate_scand_fresh_outer_v1 import (  # noqa: E402
    method_metrics,
)
from tools.planning.run_cem_development_v1 import domain_policy  # noqa: E402
from tools.scoring.finetune_universal_dualdomain_safe_rank_v2 import (  # noqa: E402
    load_envelopes,
)
from tools.scoring.train_recon_dense_meta_dinov3_three_arm_v1 import (  # noqa: E402
    load_features,
)
from tools.scoring.train_recon_dense_meta_dinov3_v1 import (  # noqa: E402
    file_sha256,
    write_json,
)
from tools.scoring.train_universal_dualdomain_raw_v2 import (  # noqa: E402
    ensemble_predict,
    sigmoid,
)


NAMES = ("ate", "rpe", "endpoint", "objective")
BANKS = ("bank_a", "bank_b", "bank_c", "bank_d", "bank_e", "bank_f")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--trajectory-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cpu")
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261059)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_rows(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def raw_base_scores(
    *, base_features: Path, training_result: Path, device: str
) -> dict[str, np.ndarray]:
    base = load_features(base_features, labels_expected=False)
    training = read_json(training_result)
    if training.get("cem") is not False:
        raise RuntimeError("base scorer is not frozen CEM=false")
    envelopes = load_envelopes(Path(str(training["support"])))
    envelope = envelopes["recon"]
    threshold = threshold_for_quantile(envelope, 0.999)
    feature, support_score, physical_guard = build_universal_features(
        base, spacing_m=0.25, support_envelope=envelope
    )
    valid = (
        physical_guard
        & (support_score <= threshold)
        & np.asarray(base["majority_support"], dtype=bool)
    )
    valid[:, 0] = True
    prediction = ensemble_predict(
        training["arms"]["D"]["checkpoints"]["full"],
        feature,
        np.zeros(len(base["record_key"]), dtype=np.int64),
        device=device,
    )
    _, dual_temperature, harm_temperature = domain_policy(training, "recon")
    return {
        "mean": prediction["mean"].astype(np.float32),
        "std": prediction["std"].astype(np.float32),
        "p_dual": sigmoid(
            prediction["dual_logit"] / dual_temperature
        ).astype(np.float32),
        "p_harm": sigmoid(
            prediction["harm_logit"] / harm_temperature
        ).astype(np.float32),
        "valid": valid,
        "candidates": base["candidates"].astype(np.float32),
        "record_key": base["record_key"].astype(str),
    }


def direct_metrics(selected: np.ndarray, absolute: np.ndarray) -> dict[str, object]:
    row = np.arange(len(selected))
    chosen = absolute[row, selected]
    incumbent = absolute[:, 0]
    delta = chosen - incumbent
    promoted = selected != 0
    promotion_delta = delta[promoted]
    dual = (
        (promotion_delta[:, 0] < -1.0e-7)
        & (promotion_delta[:, 3] < -1.0e-7)
        if promoted.any()
        else np.asarray([], dtype=bool)
    )
    harm = (
        (promotion_delta[:, 0] > 1.0e-7)
        | (promotion_delta[:, 3] > 1.0e-7)
        if promoted.any()
        else np.asarray([], dtype=bool)
    )
    return {
        "states": len(selected),
        "promotions": int(promoted.sum()),
        "coverage": float(promoted.mean()),
        "dual": float(dual.mean()) if len(dual) else None,
        "harm": float(harm.mean()) if len(harm) else None,
        "absolute": {
            name: {
                "mean_m": float(chosen[:, index].mean()),
                "p90_m": float(np.quantile(chosen[:, index], 0.9)),
            }
            for index, name in enumerate(NAMES)
        },
    }


def comparison(
    combined: dict[str, object], fixed: dict[str, object]
) -> dict[str, object]:
    quality_delta = {
        name: {
            summary: float(
                combined["absolute"][name][summary]
                - fixed["absolute"][name][summary]
            )
            for summary in ("mean_m", "p90_m")
        }
        for name in NAMES
    }
    return {
        "coverage_gain": float(combined["coverage"] - fixed["coverage"]),
        "quality_delta": quality_delta,
        "max_quality_degradation_m": max(
            value
            for metric in quality_delta.values()
            for value in metric.values()
        ),
        "all_quality_nondegraded": all(
            value <= 1.0e-12
            for metric in quality_delta.values()
            for value in metric.values()
        ),
    }


def select_rule(
    bank: dict[str, object], config: dict[str, object]
) -> tuple[np.ndarray, np.ndarray]:
    mean = bank["mean"]
    std = bank["std"]
    p_dual = bank["p_dual"]
    p_harm = bank["p_harm"]
    valid = bank["valid"]
    benefit = -0.5 * (mean[..., 0] + mean[..., 3]) - float(
        config["beta"]
    ) * np.sqrt(std[..., 0] ** 2 + std[..., 3] ** 2)
    energy = -benefit + 0.25 * p_harm - 0.10 * p_dual
    safe = (
        valid
        & (benefit >= float(config["benefit_threshold"]))
        & (p_dual >= float(config["dual_threshold"]))
        & (p_harm <= float(config["harm_threshold"]))
    )
    safe[:, 0] = False
    energy = np.where(safe, energy, np.inf)
    fixed_energy = energy[:, :17]
    cem_energy = energy[:, 17:]
    fixed_local = fixed_energy.argmin(axis=1).astype(np.int64)
    fixed_any = np.isfinite(fixed_energy).any(axis=1)
    fixed_selected = np.where(fixed_any, fixed_local, 0).astype(np.int64)
    cem_local = cem_energy.argmin(axis=1).astype(np.int64)
    cem_any = np.isfinite(cem_energy).any(axis=1)
    cem_global = 17 + cem_local
    row = np.arange(len(fixed_selected))
    fixed_best = fixed_energy[row, fixed_local]
    cem_best = cem_energy[row, cem_local]
    choose_cem = cem_any & (
        (~fixed_any)
        | (
            cem_best + float(config["dominance_margin"])
            < fixed_best
        )
    )
    combined = fixed_selected.copy()
    combined[choose_cem] = cem_global[choose_cem]
    return fixed_selected, combined


def pooled_eval(
    bank_data: dict[str, object],
    selections: dict[str, tuple[np.ndarray, np.ndarray]],
    subset: tuple[str, ...],
) -> dict[str, object]:
    absolute = np.concatenate([bank_data[bank]["absolute"] for bank in subset])
    fixed = np.concatenate([selections[bank][0] for bank in subset])
    combined = np.concatenate([selections[bank][1] for bank in subset])
    fixed_metrics = direct_metrics(fixed, absolute)
    combined_metrics = direct_metrics(combined, absolute)
    return {
        "fixed": fixed_metrics,
        "combined": combined_metrics,
        "relative": comparison(combined_metrics, fixed_metrics),
    }


def strict_pass(report: dict[str, object], requirement: dict[str, object]) -> bool:
    combined = report["combined"]
    relative = report["relative"]
    return bool(
        combined["coverage"] >= float(requirement["combined_coverage_min"])
        and relative["coverage_gain"]
        >= float(requirement["coverage_absolute_gain_vs_same_rule_cem_false_min"])
        and combined["dual"] >= float(requirement["dual_min"])
        and combined["harm"] <= float(requirement["harm_max"])
        and relative["all_quality_nondegraded"]
    )


def guard_pass(report: dict[str, object], guard: dict[str, object]) -> bool:
    combined = report["combined"]
    relative = report["relative"]
    return bool(
        combined["coverage"] >= float(guard["combined_coverage_min"])
        and relative["coverage_gain"]
        >= float(guard["coverage_gain_vs_same_rule_cem_false_min"])
        and combined["dual"] >= float(guard["dual_min"])
        and combined["harm"] <= float(guard["harm_max"])
        and relative["max_quality_degradation_m"]
        <= float(guard["maximum_any_mean_or_p90_degradation_m"])
    )


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    prereg = read_json(args.preregistration)
    if (
        prereg.get("status")
        != "grid_and_success_rule_frozen_before_A_F_frontier_audit"
        or prereg.get("G_H_outcomes_accessed")
        or prereg.get("reserve_accessed")
        or prereg["candidate_archive"]["new_model_allowed"]
        or prereg["candidate_archive"]["new_sampling_allowed"]
    ):
        raise RuntimeError("invalid v59 preregistration")
    configs = []
    for beta, profile, margin in itertools.product(
        prereg["grid"]["beta"],
        prereg["grid"]["gate_profiles"],
        prereg["grid"]["dominance_margin"],
    ):
        configs.append(
            {
                "id": len(configs),
                "beta": beta,
                "gate_profile": profile["name"],
                "benefit_threshold": profile["benefit_threshold"],
                "dual_threshold": profile["dual_threshold"],
                "harm_threshold": profile["harm_threshold"],
                "dominance_margin": margin,
            }
        )
    if len(configs) != int(prereg["grid"]["total_rules"]):
        raise RuntimeError("v59 grid size changed")

    bank_data = {}
    route_sets = {}
    artifact_sha = {}
    for bank in BANKS:
        run_name = (
            "cem_run_v57_lobo_gate"
            if bank in ("bank_e", "bank_f")
            else "cem_run_v55_safe_elite"
        )
        run_dir = args.root / bank / run_name
        freeze_path = run_dir / "prelabel_freeze.json"
        freeze = read_json(freeze_path)
        if freeze.get("state_gate_used") or freeze.get("reserve_accessed"):
            raise RuntimeError(f"{bank} archive contract invalid")
        manifest_path = Path(str(freeze["manifest"]))
        rows = read_rows(manifest_path)
        if len(rows) != 100:
            raise RuntimeError(f"{bank} expected 100 states")
        expected = np.asarray(
            [f"{row['trajectory_id']}:{int(row['current_t'])}" for row in rows]
        )
        routes = {str(row["route_id"]) for row in rows}
        if len(routes) != 25:
            raise RuntimeError(f"{bank} expected 25 routes")
        route_sets[bank] = routes
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
        base_path = Path(str(freeze["base_features"]))
        training_path = Path(str(freeze["training_result"]))
        base = raw_base_scores(
            base_features=base_path,
            training_result=training_path,
            device=args.device,
        )
        if not np.array_equal(base["record_key"], expected):
            raise RuntimeError(f"{bank} base identity mismatch")
        candidate_parts = [base["candidates"]]
        mean_parts = [base["mean"]]
        std_parts = [base["std"]]
        dual_parts = [base["p_dual"]]
        harm_parts = [base["p_harm"]]
        valid_parts = [base["valid"]]
        iteration_sha = []
        for iteration in range(3):
            directory = run_dir / f"iteration_{iteration:02d}"
            population_path = directory / "population.npz"
            scores_path = directory / "frozen_scores.npz"
            with np.load(population_path, allow_pickle=False) as payload:
                if not np.array_equal(payload["record_key"].astype(str), expected):
                    raise RuntimeError(f"{bank} iteration identity mismatch")
                candidate_parts.append(
                    payload["candidates_feature_waypoint_units"][:, 1:].astype(
                        np.float32
                    )
                )
            with np.load(scores_path, allow_pickle=False) as payload:
                mean_parts.append(payload["mean"][:, 1:].astype(np.float32))
                std_parts.append(payload["std"][:, 1:].astype(np.float32))
                dual_parts.append(
                    payload["dual_probability"][:, 1:].astype(np.float32)
                )
                harm_parts.append(
                    payload["harm_probability"][:, 1:].astype(np.float32)
                )
                valid_parts.append(payload["valid"][:, 1:].astype(bool))
            iteration_sha.append(
                {
                    "iteration": iteration,
                    "population_sha256": file_sha256(population_path),
                    "score_sha256": file_sha256(scores_path),
                }
            )
        candidates = np.concatenate(candidate_parts, axis=1)
        if candidates.shape != (100, 65, 4, 2):
            raise RuntimeError(f"{bank} candidate archive shape mismatch")
        absolute = metric_labels(candidates, ground_truth, 0.25, 0.4, 0.6)[
            "values"
        ]
        bank_data[bank] = {
            "absolute": absolute,
            "mean": np.concatenate(mean_parts, axis=1),
            "std": np.concatenate(std_parts, axis=1),
            "p_dual": np.concatenate(dual_parts, axis=1),
            "p_harm": np.concatenate(harm_parts, axis=1),
            "valid": np.concatenate(valid_parts, axis=1),
            "trajectory": np.asarray([str(row["trajectory_id"]) for row in rows]),
            "route": np.asarray([str(row["route_id"]) for row in rows]),
        }
        artifact_sha[bank] = {
            "run": str(run_dir),
            "prelabel_freeze_sha256": file_sha256(freeze_path),
            "manifest_sha256": file_sha256(manifest_path),
            "base_features_sha256": file_sha256(base_path),
            "iterations": iteration_sha,
        }
    for index, first_bank in enumerate(BANKS):
        for second_bank in BANKS[index + 1 :]:
            if route_sets[first_bank] & route_sets[second_bank]:
                raise RuntimeError(f"route overlap: {first_bank}/{second_bank}")

    requirement = prereg["primary_pooled_heldout_requirements"]
    guard = prereg["single_cohort_catastrophe_guard"]
    frontier = []
    config_selections = {}
    globally_feasible = []
    for config in configs:
        selections = {
            bank: select_rule(bank_data[bank], config) for bank in BANKS
        }
        config_selections[config["id"]] = selections
        pooled = pooled_eval(bank_data, selections, BANKS)
        per_bank = {
            bank: pooled_eval(bank_data, selections, (bank,)) for bank in BANKS
        }
        lobo_train = {}
        lobo_train_all_strict = True
        for heldout in BANKS:
            fit_banks = tuple(bank for bank in BANKS if bank != heldout)
            fit_report = pooled_eval(bank_data, selections, fit_banks)
            passed = strict_pass(fit_report, requirement) and all(
                guard_pass(per_bank[bank], guard) for bank in fit_banks
            )
            lobo_train[heldout] = {
                "strict_and_guards_pass": passed,
                "coverage_gain": fit_report["relative"]["coverage_gain"],
                "dual": fit_report["combined"]["dual"],
                "harm": fit_report["combined"]["harm"],
            }
            lobo_train_all_strict &= passed
        bank_guards = {
            bank: guard_pass(per_bank[bank], guard) for bank in BANKS
        }
        feasible = bool(
            strict_pass(pooled, requirement)
            and all(bank_guards.values())
            and lobo_train_all_strict
        )
        if feasible:
            globally_feasible.append(config["id"])
        worst_safety_slack = min(
            min(
                per_bank[bank]["combined"]["dual"]
                - float(guard["dual_min"]),
                float(guard["harm_max"])
                - per_bank[bank]["combined"]["harm"],
            )
            for bank in BANKS
        )
        entry = {
            "config": config,
            "feasible_single_rule": feasible,
            "pooled": pooled,
            "per_bank_guard_pass": bank_guards,
            "per_bank": {
                bank: {
                    "fixed_coverage": per_bank[bank]["fixed"]["coverage"],
                    "combined_coverage": per_bank[bank]["combined"]["coverage"],
                    "coverage_gain": per_bank[bank]["relative"]["coverage_gain"],
                    "dual": per_bank[bank]["combined"]["dual"],
                    "harm": per_bank[bank]["combined"]["harm"],
                    "max_quality_degradation_m": per_bank[bank]["relative"][
                        "max_quality_degradation_m"
                    ],
                }
                for bank in BANKS
            },
            "lobo_train": lobo_train,
            "lobo_train_all_strict": lobo_train_all_strict,
            "worst_bank_safety_slack": worst_safety_slack,
            "worst_bank_coverage_gain": min(
                per_bank[bank]["relative"]["coverage_gain"] for bank in BANKS
            ),
        }
        frontier.append(entry)

    selected_entry = None
    selected_bootstrap = None
    if globally_feasible:
        selected_entry = max(
            (frontier[index] for index in globally_feasible),
            key=lambda item: (
                item["worst_bank_safety_slack"],
                item["worst_bank_coverage_gain"],
                item["pooled"]["relative"]["coverage_gain"],
                item["pooled"]["combined"]["dual"],
                -item["pooled"]["combined"]["harm"],
            ),
        )
        config_id = selected_entry["config"]["id"]
        selections = config_selections[config_id]
        absolute = np.concatenate([bank_data[bank]["absolute"] for bank in BANKS])
        trajectory = np.concatenate(
            [bank_data[bank]["trajectory"] for bank in BANKS]
        )
        route = np.concatenate([bank_data[bank]["route"] for bank in BANKS])
        fixed = np.concatenate([selections[bank][0] for bank in BANKS])
        combined = np.concatenate([selections[bank][1] for bank in BANKS])
        selected_bootstrap = {
            "fixed": method_metrics(
                fixed,
                absolute,
                trajectory,
                route,
                replicates=args.bootstrap_replicates,
                seed=args.bootstrap_seed,
            ),
            "combined": method_metrics(
                combined,
                absolute,
                trajectory,
                route,
                replicates=args.bootstrap_replicates,
                seed=args.bootstrap_seed + 1,
            ),
        }

    safe_quality = [
        entry
        for entry in frontier
        if entry["pooled"]["combined"]["dual"] >= 0.85
        and entry["pooled"]["combined"]["harm"] <= 0.15
        and entry["pooled"]["relative"]["all_quality_nondegraded"]
    ]
    coverage_eligible = [
        entry
        for entry in frontier
        if entry["pooled"]["relative"]["coverage_gain"] >= 0.10
    ]
    result = {
        "schema": "recon_cem_existing_score_frontier_v59_result",
        "status": (
            "single_shared_rule_feasible_freeze_before_G_H_required"
            if selected_entry is not None
            else "existing_scorer_scores_not_separable_no_G_H_run"
        ),
        "evidence_role": "spent_A_to_F_route_disjoint_LOBO_development",
        "preregistration": str(args.preregistration),
        "preregistration_sha256": file_sha256(args.preregistration),
        "banks": list(BANKS),
        "states": 600,
        "routes": 150,
        "route_overlap": 0,
        "candidate_count_per_state": 65,
        "grid_rules": len(configs),
        "feasible_single_rules": len(globally_feasible),
        "selected_rule": selected_entry,
        "selected_rule_route_clustered_bootstrap": selected_bootstrap,
        "frontier": frontier,
        "frontier_summary": {
            "max_coverage_gain_any_rule": max(
                entry["pooled"]["relative"]["coverage_gain"]
                for entry in frontier
            ),
            "max_coverage_gain_with_pooled_safety_and_quality": (
                max(
                    entry["pooled"]["relative"]["coverage_gain"]
                    for entry in safe_quality
                )
                if safe_quality
                else None
            ),
            "max_dual_among_coverage_gain_eligible": (
                max(entry["pooled"]["combined"]["dual"] for entry in coverage_eligible)
                if coverage_eligible
                else None
            ),
            "min_harm_among_coverage_gain_eligible": (
                min(entry["pooled"]["combined"]["harm"] for entry in coverage_eligible)
                if coverage_eligible
                else None
            ),
            "coverage_gain_eligible_rules": len(coverage_eligible),
            "pooled_safety_quality_rules": len(safe_quality),
        },
        "artifact_sha256": artifact_sha,
        "new_model_trained": False,
        "new_candidates_sampled": False,
        "new_budget_used": False,
        "online_per_state_shared_rule_only": True,
        "cohort_specific_thresholds": False,
        "G_H_outcomes_accessed": False,
        "G_H_run_allowed": selected_entry is not None,
        "post_frontier_threshold_expansion_allowed": False,
        "reserve_accessed": False,
        "production_safe_claim_allowed": False,
    }
    write_json(args.output, result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "feasible_single_rules": result["feasible_single_rules"],
                "selected_rule": result["selected_rule"],
                "frontier_summary": result["frontier_summary"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
