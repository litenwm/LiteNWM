#!/usr/bin/env python3
"""Read-only cumulative CEM iteration budget curve over spent A-H archives."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from tools.evaluation.audit_cem_existing_score_frontier_v59 import (  # noqa: E402
    comparison,
    direct_metrics,
    read_rows,
)
from tools.evaluation.build_recon100_k17_source_v1 import metric_labels  # noqa: E402
from tools.evaluation.build_recon_inner_k17_source_v2 import local_ground_truth  # noqa: E402
from tools.evaluation.evaluate_recon_cem_boundary_gh_v60 import (  # noqa: E402
    route_bootstrap_comparison,
    route_counts,
    route_method_ci,
)
from tools.evaluation.evaluate_scand_fresh_outer_v1 import method_metrics  # noqa: E402
from tools.scoring.train_recon_dense_meta_dinov3_v1 import write_json, write_npz  # noqa: E402


NAMES = ("ate", "rpe", "endpoint", "objective")
AF_BANKS = ("bank_a", "bank_b", "bank_c", "bank_d", "bank_e", "bank_f")
GH_BANKS = ("bank_g_v60", "bank_h_v60")
BANKS = AF_BANKS + GH_BANKS


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def incremental_metrics(
    added: np.ndarray,
    selected: np.ndarray,
    absolute: np.ndarray,
) -> dict[str, Any]:
    indices = np.flatnonzero(added)
    if not len(indices):
        return {
            "n": 0,
            "dual": None,
            "harm": None,
            "quality": None,
            "all_four_mean_and_p90_nondegraded_vs_incumbent": False,
        }
    chosen = absolute[indices, selected[indices]]
    incumbent = absolute[indices, 0]
    delta = chosen - incumbent
    dual = (delta[:, 0] < -1.0e-7) & (delta[:, 3] < -1.0e-7)
    harm = (delta[:, 0] > 1.0e-7) | (delta[:, 3] > 1.0e-7)
    quality = {
        name: {
            "absolute_mean_m": float(chosen[:, metric].mean()),
            "absolute_p90_m": float(np.quantile(chosen[:, metric], 0.90)),
            "delta_vs_incumbent_mean_m": float(delta[:, metric].mean()),
            "delta_vs_incumbent_p90_m": float(np.quantile(delta[:, metric], 0.90)),
        }
        for metric, name in enumerate(NAMES)
    }
    nondegraded = all(
        quality[name][summary] <= 1.0e-12
        for name in NAMES
        for summary in (
            "delta_vs_incumbent_mean_m",
            "delta_vs_incumbent_p90_m",
        )
    )
    return {
        "n": int(len(indices)),
        "dual": float(dual.mean()),
        "harm": float(harm.mean()),
        "dual_successes": int(dual.sum()),
        "harms": int(harm.sum()),
        "quality": quality,
        "all_four_mean_and_p90_nondegraded_vs_incumbent": nondegraded,
    }


def marginal_route_ci(
    added: np.ndarray,
    selected: np.ndarray,
    absolute: np.ndarray,
    route: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    selected_value = absolute[np.arange(len(selected)), selected]
    delta = selected_value - absolute[:, 0]
    dual = added & (delta[:, 0] < -1.0e-7) & (delta[:, 3] < -1.0e-7)
    harm = added & ((delta[:, 0] > 1.0e-7) | (delta[:, 3] > 1.0e-7))
    unique = sorted(np.unique(route.astype(str)).tolist())
    counts = []
    quality_sum = []
    for name in unique:
        mask = route == name
        counts.append(
            [
                int(mask.sum()),
                int((added & mask).sum()),
                int((dual & mask).sum()),
                int((harm & mask).sum()),
            ]
        )
        quality_sum.append(delta[added & mask].sum(axis=0))
    counts_array = np.asarray(counts, dtype=np.int64)
    quality_array = np.asarray(quality_sum, dtype=np.float64)
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, len(unique), size=(replicates, len(unique)))
    sampled_count = counts_array[draw].sum(axis=1)
    sampled_quality = quality_array[draw].sum(axis=1)
    coverage = sampled_count[:, 1] / sampled_count[:, 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        dual_precision = sampled_count[:, 2] / sampled_count[:, 1]
        harm_rate = sampled_count[:, 3] / sampled_count[:, 1]
        quality_mean = sampled_quality / sampled_count[:, 1:2]

    def interval(values: np.ndarray) -> list[float] | None:
        finite = values[np.isfinite(values)]
        if not len(finite):
            return None
        return [float(np.quantile(finite, 0.025)), float(np.quantile(finite, 0.975))]

    return {
        "coverage_gain_ci95": interval(coverage),
        "dual_ci95": interval(dual_precision),
        "harm_ci95": interval(harm_rate),
        "mean_delta_ci95_m": {
            name: interval(quality_mean[:, metric])
            for metric, name in enumerate(NAMES)
        },
    }


def load_fixed_authority(path: Path) -> dict[str, dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=False) as payload:
        record = payload["record_key"].astype(str)
        bank = payload["bank"].astype(str)
        route = payload["route"].astype(str)
        fixed = payload["fixed_selected_index"].astype(np.int64)
    result = {}
    for name in BANKS:
        mask = bank == name
        result[name] = {
            "record_key": record[mask],
            "route": route[mask],
            "fixed": fixed[mask],
        }
    return result


def load_bank(
    bank: str,
    *,
    af_root: Path,
    gh_root: Path,
    trajectory_root: Path,
    authority: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if bank in AF_BANKS:
        run_name = (
            "cem_run_v57_lobo_gate"
            if bank in ("bank_e", "bank_f")
            else "cem_run_v55_safe_elite"
        )
        run = af_root / bank / run_name
    else:
        run = gh_root / bank / "cem_run_v60"
    freeze_path = run / "prelabel_freeze.json"
    freeze = read_json(freeze_path)
    manifest_path = Path(str(freeze["manifest"]))
    rows = read_rows(manifest_path)
    record = np.asarray(
        [f"{row['trajectory_id']}:{int(row['current_t'])}" for row in rows]
    )
    trajectory = np.asarray([str(row["trajectory_id"]) for row in rows])
    route = np.asarray([str(row["route_id"]) for row in rows])
    fixed = authority["fixed"]
    if (
        not np.array_equal(record, authority["record_key"])
        or not np.array_equal(route, authority["route"])
        or freeze.get("reserve_accessed")
        or freeze.get("state_gate_used")
    ):
        raise RuntimeError(f"{bank}: v61 fixed authority/archive identity mismatch")

    selected_actions_path = run / "selected_actions.npz"
    with np.load(selected_actions_path, allow_pickle=False) as selected_actions:
        if not np.array_equal(selected_actions["record_key"].astype(str), record):
            raise RuntimeError(f"{bank}: selected-actions identity mismatch")
        base_candidates = selected_actions[
            "base_k17_feature_waypoint_units"
        ].astype(np.float32)
    candidate_parts = [base_candidates]
    mean_parts = []
    std_parts = []
    dual_parts = []
    harm_parts = []
    valid_parts = []
    updated_parts = []
    artifacts = []
    for iteration in range(3):
        directory = run / f"iteration_{iteration:02d}"
        population_path = directory / "population.npz"
        score_path = directory / "frozen_scores.npz"
        with np.load(population_path, allow_pickle=False) as population:
            if not np.array_equal(population["record_key"].astype(str), record):
                raise RuntimeError(f"{bank}: iteration identity mismatch")
            candidate_parts.append(
                population["candidates_feature_waypoint_units"][:, 1:].astype(
                    np.float32
                )
            )
        with np.load(score_path, allow_pickle=False) as score:
            mean_parts.append(score["mean"][:, 1:].astype(np.float32))
            std_parts.append(score["std"][:, 1:].astype(np.float32))
            dual_parts.append(score["dual_probability"][:, 1:].astype(np.float32))
            harm_parts.append(score["harm_probability"][:, 1:].astype(np.float32))
            valid_parts.append(score["valid"][:, 1:].astype(bool))
            updated_parts.append(score["proposal_updated"].astype(bool))
        artifacts.append(
            {
                "iteration": iteration,
                "population_sha256": sha256(population_path),
                "frozen_scores_sha256": sha256(score_path),
            }
        )
    candidates = np.concatenate(candidate_parts, axis=1)
    mean = np.concatenate(mean_parts, axis=1)
    std = np.concatenate(std_parts, axis=1)
    p_dual = np.concatenate(dual_parts, axis=1)
    p_harm = np.concatenate(harm_parts, axis=1)
    valid = np.concatenate(valid_parts, axis=1)
    updated = np.stack(updated_parts, axis=1)
    expected_states = 100 if bank in AF_BANKS else 44
    if (
        candidates.shape != (expected_states, 65, 4, 2)
        or mean.shape[:2] != (expected_states, 48)
        or np.any(fixed < 0)
        or np.any(fixed >= 17)
    ):
        raise RuntimeError(f"{bank}: cached K65/score contract changed")

    benefit = -0.5 * (mean[..., 0] + mean[..., 3]) - 1.5 * np.sqrt(
        std[..., 0] ** 2 + std[..., 3] ** 2
    )
    energy = -benefit + 0.25 * p_harm - 0.10 * p_dual
    safe = valid & (benefit >= 0.0) & (p_dual >= 0.75) & (p_harm <= 0.25)
    energy = np.where(safe, energy, np.inf).astype(np.float32)

    if bank in AF_BANKS:
        ground_truth = np.stack(
            [
                local_ground_truth(
                    trajectory_root / str(row["trajectory_id"]) / "traj_data.pkl",
                    int(row["current_t"]),
                )
                for row in rows
            ]
        ).astype(np.float32)
        absolute = metric_labels(candidates, ground_truth, 0.25, 0.4, 0.6)[
            "values"
        ].astype(np.float32)
        outcome_artifact = "spent trajectory replay"
        outcome_sha = None
    else:
        primary = read_json(gh_root / "evaluation_v60/result.json")
        outcome_path = Path(primary["banks"][bank]["outcome_artifact"])
        if sha256(outcome_path) != primary["banks"][bank]["outcome_artifact_sha256"]:
            raise RuntimeError(f"{bank}: spent outcome artifact changed")
        with np.load(outcome_path, allow_pickle=False) as outcome:
            if not np.array_equal(outcome["record_key"].astype(str), record):
                raise RuntimeError(f"{bank}: outcome identity mismatch")
            absolute = outcome["absolute_metrics"].astype(np.float32)
        outcome_artifact = str(outcome_path)
        outcome_sha = sha256(outcome_path)

    return {
        "record_key": record,
        "trajectory": trajectory,
        "route": route,
        "fixed": fixed,
        "candidates": candidates,
        "absolute": absolute,
        "energy": energy,
        "safe": safe,
        "p_dual": p_dual,
        "p_harm": p_harm,
        "proposal_updated": updated,
    }, {
        "run": str(run),
        "prelabel_freeze_sha256": sha256(freeze_path),
        "manifest_sha256": sha256(manifest_path),
        "selected_actions_sha256": sha256(selected_actions_path),
        "iterations": artifacts,
        "outcome_artifact": outcome_artifact,
        "outcome_artifact_sha256": outcome_sha,
    }


def select_cumulative(data: dict[str, np.ndarray], iterations: int) -> np.ndarray:
    fixed = data["fixed"]
    selected = fixed.copy()
    stop = iterations * 16
    masked = data["energy"][:, :stop]
    any_safe = np.isfinite(masked).any(axis=1)
    local = masked.argmin(axis=1).astype(np.int64)
    add = (fixed == 0) & any_safe
    selected[add] = 17 + local[add]
    return selected


def bank_report(
    data: dict[str, np.ndarray],
    selected: np.ndarray,
    previous: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    fixed = data["fixed"]
    absolute = data["absolute"]
    trajectory = data["trajectory"]
    route = data["route"]
    metrics = method_metrics(
        selected,
        absolute,
        trajectory,
        route,
        replicates=replicates,
        seed=seed,
    )
    metrics["route_clustered_safety_ci95"] = route_method_ci(
        selected,
        absolute,
        route,
        replicates=replicates,
        seed=seed + 1,
    )
    fixed_direct = direct_metrics(fixed, absolute)
    selected_direct = direct_metrics(selected, absolute)
    relative = comparison(selected_direct, fixed_direct)
    incremental = (previous == 0) & (selected != 0)
    changed = (previous != 0) & (selected != previous)
    return {
        "metrics": metrics,
        "relative_to_fixed": relative,
        "incremental_first_additions": incremental_metrics(
            incremental, selected, absolute
        ),
        "incremental_route_clustered_ci95": marginal_route_ci(
            incremental,
            selected,
            absolute,
            route,
            replicates=replicates,
            seed=seed + 2,
        ),
        "selection_changes_from_previous_archive": int(changed.sum()),
        "route_counts": route_counts(fixed, selected, absolute, route),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--v61-selected", type=Path, required=True)
    parser.add_argument("--af-root", type=Path, required=True)
    parser.add_argument("--gh-root", type=Path, required=True)
    parser.add_argument("--trajectory-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261068)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(args.output_dir)
    if args.bootstrap_replicates < 1000:
        raise ValueError("route bootstrap requires at least 1000 replicates")
    prereg = read_json(args.preregistration)
    if (
        prereg.get("status")
        != "read_only_budget_curve_protocol_frozen_before_spent_outcome_summary"
        or not prereg.get("one_time_only")
        or prereg.get("reserve_accessed")
        or not all(prereg["prohibitions"].values())
    ):
        raise RuntimeError("v62 preregistration contract failed")
    authority = load_fixed_authority(args.v61_selected)

    reserve_path = args.af_root / "final_confirmation_reserve/selection_v26/freeze.json"
    reserve = read_json(reserve_path)
    reserve_sha_before = sha256(reserve_path)
    gh_spent_path = args.gh_root / "evaluation_v60/outcome_open_and_spent_audit.json"
    gh_spent = read_json(gh_spent_path)
    if (
        reserve.get("labels_opened")
        or reserve.get("outcomes_accessed")
        or gh_spent.get("spent_route_count") != 22
        or gh_spent.get("reserve_accessed")
    ):
        raise RuntimeError("reserve/spent-route contract failed")

    bank_data = {}
    artifacts = {}
    route_sets = {}
    for bank in BANKS:
        bank_data[bank], artifacts[bank] = load_bank(
            bank,
            af_root=args.af_root,
            gh_root=args.gh_root,
            trajectory_root=args.trajectory_root,
            authority=authority[bank],
        )
        route_sets[bank] = set(bank_data[bank]["route"].astype(str).tolist())
    for index, first in enumerate(BANKS):
        for second in BANKS[index + 1 :]:
            if route_sets[first] & route_sets[second]:
                raise RuntimeError(f"route overlap: {first}/{second}")

    curve_selected: dict[int, dict[str, np.ndarray]] = {}
    for iterations in (1, 2, 3):
        curve_selected[iterations] = {
            bank: select_cumulative(bank_data[bank], iterations) for bank in BANKS
        }

    pooled_absolute = np.concatenate([bank_data[bank]["absolute"] for bank in BANKS])
    pooled_trajectory = np.concatenate(
        [bank_data[bank]["trajectory"] for bank in BANKS]
    )
    pooled_route = np.concatenate([bank_data[bank]["route"] for bank in BANKS])
    pooled_record = np.concatenate(
        [bank_data[bank]["record_key"] for bank in BANKS]
    )
    pooled_bank = np.concatenate(
        [np.full(len(bank_data[bank]["record_key"]), bank) for bank in BANKS]
    )
    pooled_fixed = np.concatenate([bank_data[bank]["fixed"] for bank in BANKS])

    curve = {}
    first_iteration = np.full(len(pooled_fixed), -1, dtype=np.int8)
    previous_pooled = pooled_fixed.copy()
    selected_arrays = {
        "record_key": pooled_record,
        "bank": pooled_bank,
        "route": pooled_route,
        "fixed_selected_index": pooled_fixed,
    }
    for iterations in (1, 2, 3):
        pooled_selected = np.concatenate(
            [curve_selected[iterations][bank] for bank in BANKS]
        )
        selected_arrays[f"cumulative_{iterations}_selected_index"] = pooled_selected
        newly_added = (previous_pooled == 0) & (pooled_selected != 0)
        first_iteration[newly_added] = iterations - 1
        metrics = method_metrics(
            pooled_selected,
            pooled_absolute,
            pooled_trajectory,
            pooled_route,
            replicates=args.bootstrap_replicates,
            seed=args.bootstrap_seed + iterations * 100,
        )
        metrics["route_clustered_safety_ci95"] = route_method_ci(
            pooled_selected,
            pooled_absolute,
            pooled_route,
            replicates=args.bootstrap_replicates,
            seed=args.bootstrap_seed + iterations * 100 + 1,
        )
        fixed_direct = direct_metrics(pooled_fixed, pooled_absolute)
        selected_direct = direct_metrics(pooled_selected, pooled_absolute)
        relative = comparison(selected_direct, fixed_direct)
        per_bank = {}
        offset = 0
        for bank_number, bank in enumerate(BANKS):
            length = len(bank_data[bank]["fixed"])
            per_bank[bank] = bank_report(
                bank_data[bank],
                curve_selected[iterations][bank],
                (
                    bank_data[bank]["fixed"]
                    if iterations == 1
                    else curve_selected[iterations - 1][bank]
                ),
                replicates=args.bootstrap_replicates,
                seed=args.bootstrap_seed
                + iterations * 1000
                + bank_number * 10,
            )
            offset += length
        curve[f"cumulative_iterations_{iterations}"] = {
            "archive_challengers_per_state": iterations * 16,
            "metrics": metrics,
            "relative_to_fixed": relative,
            "incremental_first_additions": incremental_metrics(
                newly_added, pooled_selected, pooled_absolute
            ),
            "incremental_route_clustered_ci95": marginal_route_ci(
                newly_added,
                pooled_selected,
                pooled_absolute,
                pooled_route,
                replicates=args.bootstrap_replicates,
                seed=args.bootstrap_seed + iterations * 100 + 2,
            ),
            "selection_changes_from_previous_archive": int(
                ((previous_pooled != 0) & (pooled_selected != previous_pooled)).sum()
            ),
            "per_bank": per_bank,
            "selected_minus_fixed_route_clustered": route_bootstrap_comparison(
                pooled_fixed,
                pooled_selected,
                pooled_absolute,
                pooled_route,
                replicates=args.bootstrap_replicates,
                seed=args.bootstrap_seed + iterations * 100 + 3,
            ),
        }
        previous_pooled = pooled_selected
    selected_arrays["first_added_iteration"] = first_iteration

    third = curve["cumulative_iterations_3"]["incremental_first_additions"]
    third_bank_direction = 0
    for bank in BANKS:
        value = curve["cumulative_iterations_3"]["per_bank"][bank][
            "incremental_first_additions"
        ]
        if (
            value["n"] > 0
            and value["dual"] is not None
            and value["dual"] >= 0.85
            and value["harm"] <= 0.15
            and value["all_four_mean_and_p90_nondegraded_vs_incumbent"]
        ):
            third_bank_direction += 1
    checks = {
        "iteration2_new_states_at_least_2": third["n"] >= 2,
        "iteration2_marginal_dual_at_least_0.85": (
            third["dual"] is not None and third["dual"] >= 0.85
        ),
        "iteration2_marginal_harm_at_most_0.15": (
            third["harm"] is not None and third["harm"] <= 0.15
        ),
        "iteration2_all_four_mean_and_p90_nondegraded": third[
            "all_four_mean_and_p90_nondegraded_vs_incumbent"
        ],
        "iteration2_direction_in_at_least_two_banks": third_bank_direction >= 2,
    }
    four_by_sixteen_supported = all(checks.values())
    decision = {
        "four_by_sixteen_direct_development_evidence": four_by_sixteen_supported,
        "checks": checks,
        "iteration2_banks_with_safe_nondegraded_new_additions": third_bank_direction,
        "recommendation": (
            "4x16 is the only directly supported next budget, conditional on stationarity of iteration-2 marginal discovery and safety; no success claim"
            if four_by_sixteen_supported
            else "budget increase is not a legal unblock because the late-iteration marginal is saturated or unsafe"
        ),
        "stationarity_assumption": (
            "A fourth proposal update would preserve the iteration-2 marginal new-state discovery rate, support validity, dual precision, harm rate, and quality direction."
        ),
        "three_by_thirty_two_direct_evidence": False,
        "three_by_thirty_two_reason": (
            "The archive varies iteration count at fixed population=16 and contains no paired population=32 run."
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected_path = args.output_dir / "curve_selected_indices.npz"
    write_npz(selected_path, **selected_arrays)
    reserve_sha_after = sha256(reserve_path)
    if reserve_sha_after != reserve_sha_before:
        raise RuntimeError("reserve changed during read-only budget audit")
    result = {
        "schema": "recon_cem_budget_curve_v62_result",
        "status": "read_only_one_time_budget_curve_complete",
        "evidence_role": "spent_A_to_H_budget_development_diagnostic_only",
        "preregistration": str(args.preregistration),
        "preregistration_sha256": sha256(args.preregistration),
        "v61_fixed_authority": str(args.v61_selected),
        "v61_fixed_authority_sha256": sha256(args.v61_selected),
        "states": int(len(pooled_fixed)),
        "banks": list(BANKS),
        "routes": int(len(np.unique(pooled_route))),
        "route_overlap": 0,
        "curve": curve,
        "first_added_iteration_counts": {
            f"iteration_{iteration}": int((first_iteration == iteration).sum())
            for iteration in range(3)
        },
        "budget_decision": decision,
        "artifacts": artifacts,
        "curve_selected_indices": str(selected_path),
        "curve_selected_indices_sha256": sha256(selected_path),
        "new_candidates_generated": False,
        "new_CEM_run": False,
        "model_trained": False,
        "new_features_extracted": False,
        "thresholds_tuned": False,
        "outcome_dependent_selection": False,
        "all_outcomes_were_already_spent": True,
        "GH_22_routes_spent_audit": str(gh_spent_path),
        "GH_22_routes_spent_audit_sha256": sha256(gh_spent_path),
        "GH_22_routes_spent": True,
        "reserve": str(reserve_path),
        "reserve_sha256_before": reserve_sha_before,
        "reserve_sha256_after": reserve_sha_after,
        "reserve_labels_opened": False,
        "reserve_outcomes_accessed": False,
        "reserve_accessed": False,
        "production_safe_claim_allowed": False,
        "one_time_audit_complete": True,
        "bootstrap_replicates": args.bootstrap_replicates,
        "bootstrap_seed": args.bootstrap_seed,
    }
    result_path = args.output_dir / "result.json"
    write_json(result_path, result)
    print(
        json.dumps(
            {
                "first_added_iteration_counts": result[
                    "first_added_iteration_counts"
                ],
                "curve": {
                    name: {
                        "promotions": value["metrics"]["promotions"],
                        "coverage": value["metrics"]["promotion_fraction"],
                        "coverage_gain": value["relative_to_fixed"][
                            "coverage_gain"
                        ],
                        "dual": value["metrics"]["dual_precision"],
                        "harm": value["metrics"]["harm_rate_per_promotion"],
                        "incremental": value["incremental_first_additions"],
                        "selection_changes": value[
                            "selection_changes_from_previous_archive"
                        ],
                    }
                    for name, value in curve.items()
                },
                "budget_decision": decision,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
