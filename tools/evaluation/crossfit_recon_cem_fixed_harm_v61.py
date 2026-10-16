#!/usr/bin/env python3
"""Evaluate the frozen v61 fixed-harm replacement/veto family on spent A-H."""

from __future__ import annotations

import argparse
import hashlib
import itertools
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
    raw_base_scores,
    read_rows,
)
from tools.evaluation.build_recon100_k17_source_v1 import metric_labels  # noqa: E402
from tools.evaluation.build_recon_inner_k17_source_v2 import local_ground_truth  # noqa: E402
from tools.evaluation.evaluate_recon_cem_boundary_gh_v60 import (  # noqa: E402
    route_bootstrap_comparison,
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


def config_grid(prereg: dict[str, Any]) -> list[dict[str, Any]]:
    grid = prereg["grid"]
    configs = []
    for cosine, margin, dual, harm, veto in itertools.product(
        grid["endpoint_cosine_min"],
        grid["benefit_lcb_advantage_min"],
        grid["p_dual_advantage_min"],
        grid["p_harm_advantage_min"],
        grid["veto_mode"],
    ):
        configs.append(
            {
                "id": len(configs),
                "endpoint_cosine_min": float(cosine),
                "benefit_lcb_advantage_min": float(margin),
                "p_dual_advantage_min": float(dual),
                "p_harm_advantage_min": float(harm),
                "veto_mode": str(veto),
            }
        )
    if len(configs) != int(grid["total_rules"]):
        raise RuntimeError("frozen v61 grid cardinality changed")
    return configs


def base_policy(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    mean = data["mean"]
    std = data["std"]
    p_dual = data["p_dual"]
    p_harm = data["p_harm"]
    valid = data["valid"]
    benefit = -0.5 * (mean[..., 0] + mean[..., 3]) - 1.5 * np.sqrt(
        std[..., 0] ** 2 + std[..., 3] ** 2
    )
    energy = -benefit + 0.25 * p_harm - 0.10 * p_dual
    safe = (
        valid
        & (benefit >= 0.0)
        & (p_dual >= 0.75)
        & (p_harm <= 0.25)
    )
    safe[:, 0] = False
    energy = np.where(valid, energy, np.inf).astype(np.float32)
    energy[:, 0] = np.inf

    fixed_masked = np.where(safe[:, :17], energy[:, :17], np.inf)
    fixed_local = fixed_masked.argmin(axis=1).astype(np.int64)
    fixed_any = np.isfinite(fixed_masked).any(axis=1)
    fixed = np.where(fixed_any, fixed_local, 0).astype(np.int64)
    cem_masked = np.where(safe[:, 17:], energy[:, 17:], np.inf)
    cem_local = cem_masked.argmin(axis=1).astype(np.int64)
    cem_any = np.isfinite(cem_masked).any(axis=1)
    cem = np.where(cem_any, cem_local + 17, 0).astype(np.int64)
    addonly = fixed.copy()
    add = (~fixed_any) & cem_any
    addonly[add] = cem[add]
    return {
        "benefit": benefit.astype(np.float32),
        "energy": energy,
        "safe": safe,
        "fixed": fixed,
        "cem_best": cem,
        "addonly": addonly,
        "added": add,
    }


def endpoint_consistent(first: np.ndarray, second: np.ndarray, cosine: float) -> bool:
    first_end = np.asarray(first[-1], dtype=np.float64)
    second_end = np.asarray(second[-1], dtype=np.float64)
    denominator = float(np.linalg.norm(first_end) * np.linalg.norm(second_end))
    if denominator <= 1.0e-12:
        return False
    observed = float(np.dot(first_end, second_end) / denominator)
    first_lateral = float(first_end[1])
    second_lateral = float(second_end[1])
    lateral_consistent = (
        abs(first_lateral) < 0.05
        or abs(second_lateral) < 0.05
        or np.sign(first_lateral) == np.sign(second_lateral)
    )
    return observed >= cosine and lateral_consistent


def select_rule(
    data: dict[str, np.ndarray],
    policy: dict[str, np.ndarray],
    config: dict[str, Any],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    candidates = data["candidates"]
    p_dual = data["p_dual"]
    p_harm = data["p_harm"]
    proposal_updated = data["proposal_updated"]
    benefit = policy["benefit"]
    energy = policy["energy"]
    safe = policy["safe"]
    fixed = policy["fixed"]
    selected = policy["addonly"].copy()
    replaced = np.zeros(len(fixed), dtype=bool)
    vetoed = np.zeros(len(fixed), dtype=bool)
    reliable_pair = np.zeros(len(fixed), dtype=bool)
    reliable_candidate = np.full(len(fixed), -1, dtype=np.int64)
    reliable_iterations = np.zeros(len(fixed), dtype=np.int64)
    reliable_updated_iterations = np.zeros(len(fixed), dtype=np.int64)

    for state in np.flatnonzero(fixed != 0):
        per_iteration: list[tuple[int, int] | None] = []
        for iteration in range(3):
            start = 17 + 16 * iteration
            stop = start + 16
            local_safe = safe[state, start:stop]
            if not local_safe.any():
                per_iteration.append(None)
                continue
            masked = np.where(local_safe, energy[state, start:stop], np.inf)
            per_iteration.append((iteration, start + int(masked.argmin())))
        members: set[int] = set()
        member_iterations: set[int] = set()
        for first_position in range(3):
            first = per_iteration[first_position]
            if first is None:
                continue
            for second_position in range(first_position + 1, 3):
                second = per_iteration[second_position]
                if second is None:
                    continue
                if endpoint_consistent(
                    candidates[state, first[1]],
                    candidates[state, second[1]],
                    float(config["endpoint_cosine_min"]),
                ):
                    members.update((first[1], second[1]))
                    member_iterations.update((first[0], second[0]))
        reliable_iterations[state] = len(member_iterations)
        reliable_updated_iterations[state] = sum(
            bool(proposal_updated[state, iteration]) for iteration in member_iterations
        )
        if len(member_iterations) < 2:
            members.clear()
        fixed_index = int(fixed[state])
        eligible = [
            index
            for index in sorted(members)
            if benefit[state, index] - benefit[state, fixed_index]
            >= float(config["benefit_lcb_advantage_min"])
            and p_dual[state, index] - p_dual[state, fixed_index]
            >= float(config["p_dual_advantage_min"])
            and p_harm[state, fixed_index] - p_harm[state, index]
            >= float(config["p_harm_advantage_min"])
        ]
        if eligible:
            chosen = min(eligible, key=lambda index: (energy[state, index], index))
            selected[state] = chosen
            replaced[state] = True
            reliable_pair[state] = True
            reliable_candidate[state] = chosen
            continue
        high_risk = bool(
            p_dual[state, fixed_index] < 0.80
            or p_harm[state, fixed_index] > 0.20
            or benefit[state, fixed_index] < 0.01
        )
        if config["veto_mode"] == "risk_no_reliable_abstain" and high_risk:
            selected[state] = 0
            vetoed[state] = True

    if not np.array_equal(
        selected[(fixed == 0) & policy["added"]],
        policy["addonly"][(fixed == 0) & policy["added"]],
    ):
        raise RuntimeError("v61 rule changed an existing add-only CEM identity")
    return selected, {
        "replaced": replaced,
        "vetoed": vetoed,
        "reliable_pair": reliable_pair,
        "reliable_candidate": reliable_candidate,
        "reliable_iterations": reliable_iterations,
        "reliable_updated_iterations": reliable_updated_iterations,
    }


def load_bank(
    bank: str,
    *,
    af_root: Path,
    gh_root: Path,
    trajectory_root: Path,
    device: str,
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
    if freeze.get("state_gate_used") or freeze.get("reserve_accessed"):
        raise RuntimeError(f"{bank}: invalid CEM archive")
    manifest_path = Path(str(freeze["manifest"]))
    manifest = read_rows(manifest_path)
    expected = np.asarray(
        [f"{row['trajectory_id']}:{int(row['current_t'])}" for row in manifest]
    )
    routes = np.asarray([str(row["route_id"]) for row in manifest])
    trajectories = np.asarray([str(row["trajectory_id"]) for row in manifest])
    expected_states = 100 if bank in AF_BANKS else 44
    expected_routes = 25 if bank in AF_BANKS else 11
    if len(manifest) != expected_states or len(np.unique(routes)) != expected_routes:
        raise RuntimeError(f"{bank}: state/route contract changed")
    base_path = Path(str(freeze["base_features"]))
    training_path = Path(str(freeze["training_result"]))
    base = raw_base_scores(
        base_features=base_path, training_result=training_path, device=device
    )
    if not np.array_equal(base["record_key"], expected):
        raise RuntimeError(f"{bank}: base identity mismatch")

    candidate_parts = [base["candidates"]]
    mean_parts = [base["mean"]]
    std_parts = [base["std"]]
    dual_parts = [base["p_dual"]]
    harm_parts = [base["p_harm"]]
    valid_parts = [base["valid"]]
    proposal_updated = []
    iteration_artifacts = []
    for iteration in range(3):
        directory = run / f"iteration_{iteration:02d}"
        population_path = directory / "population.npz"
        score_path = directory / "frozen_scores.npz"
        with np.load(population_path, allow_pickle=False) as population:
            if not np.array_equal(population["record_key"].astype(str), expected):
                raise RuntimeError(f"{bank}: iteration record mismatch")
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
            proposal_updated.append(score["proposal_updated"].astype(bool))
        iteration_artifacts.append(
            {
                "iteration": iteration,
                "population_sha256": sha256(population_path),
                "frozen_scores_sha256": sha256(score_path),
            }
        )
    candidates = np.concatenate(candidate_parts, axis=1)
    if candidates.shape != (expected_states, 65, 4, 2):
        raise RuntimeError(f"{bank}: candidate archive is not K65")

    if bank in AF_BANKS:
        ground_truth = np.stack(
            [
                local_ground_truth(
                    trajectory_root / str(row["trajectory_id"]) / "traj_data.pkl",
                    int(row["current_t"]),
                )
                for row in manifest
            ]
        ).astype(np.float32)
        absolute = metric_labels(candidates, ground_truth, 0.25, 0.4, 0.6)[
            "values"
        ].astype(np.float32)
        outcome_artifact = "spent raw trajectory outcome replay"
        outcome_sha = None
    else:
        primary = read_json(gh_root / "evaluation_v60/result.json")
        outcome_path = Path(primary["banks"][bank]["outcome_artifact"])
        if sha256(outcome_path) != primary["banks"][bank]["outcome_artifact_sha256"]:
            raise RuntimeError(f"{bank}: spent outcome artifact changed")
        with np.load(outcome_path, allow_pickle=False) as outcome:
            if not np.array_equal(outcome["record_key"].astype(str), expected):
                raise RuntimeError(f"{bank}: outcome identity mismatch")
            absolute = outcome["absolute_metrics"].astype(np.float32)
        archive_path = gh_root / bank / "combined_archive_preoutcome_v60.npz"
        with np.load(archive_path, allow_pickle=False) as archive:
            if not np.allclose(
                archive["candidates_waypoint_units"].astype(np.float32),
                candidates,
                rtol=0.0,
                atol=1.0e-7,
            ):
                raise RuntimeError(f"{bank}: K65 archive mismatch")
        outcome_artifact = str(outcome_path)
        outcome_sha = sha256(outcome_path)

    data = {
        "record_key": expected,
        "route": routes,
        "trajectory": trajectories,
        "candidates": candidates,
        "mean": np.concatenate(mean_parts, axis=1),
        "std": np.concatenate(std_parts, axis=1),
        "p_dual": np.concatenate(dual_parts, axis=1),
        "p_harm": np.concatenate(harm_parts, axis=1),
        "valid": np.concatenate(valid_parts, axis=1),
        "proposal_updated": np.stack(proposal_updated, axis=1),
        "absolute": absolute,
    }
    audit = {
        "states": expected_states,
        "routes": expected_routes,
        "run": str(run),
        "prelabel_freeze_sha256": sha256(freeze_path),
        "manifest_sha256": sha256(manifest_path),
        "base_features_sha256": sha256(base_path),
        "training_result_sha256": sha256(training_path),
        "iterations": iteration_artifacts,
        "outcome_artifact": outcome_artifact,
        "outcome_artifact_sha256": outcome_sha,
    }
    return data, audit


def pooled_report(
    bank_data: dict[str, dict[str, np.ndarray]],
    selected: dict[str, np.ndarray],
    banks: tuple[str, ...],
) -> dict[str, Any]:
    absolute = np.concatenate([bank_data[bank]["absolute"] for bank in banks])
    fixed = np.concatenate([bank_data[bank]["policy"]["fixed"] for bank in banks])
    choice = np.concatenate([selected[bank] for bank in banks])
    fixed_metrics = direct_metrics(fixed, absolute)
    choice_metrics = direct_metrics(choice, absolute)
    return {
        "fixed": fixed_metrics,
        "selected": choice_metrics,
        "relative": comparison(choice_metrics, fixed_metrics),
    }


def catastrophe_pass(report: dict[str, Any], guard: dict[str, Any]) -> bool:
    return bool(
        report["relative"]["coverage_gain"] >= float(guard["coverage_gain_min"])
        and report["selected"]["dual"] >= float(guard["dual_min"])
        and report["selected"]["harm"] <= float(guard["harm_max"])
        and report["relative"]["max_quality_degradation_m"]
        <= float(guard["maximum_any_mean_or_p90_degradation_m"])
    )


def strict_pass(report: dict[str, Any]) -> bool:
    return bool(
        report["relative"]["coverage_gain"] >= 0.10
        and report["selected"]["dual"] >= 0.85
        and report["selected"]["harm"] <= 0.15
        and report["relative"]["all_quality_nondegraded"]
    )


def normalized_violation(
    report: dict[str, Any], guard_failures: int
) -> float:
    return float(
        max(0.0, 0.10 - report["relative"]["coverage_gain"]) / 0.10
        + max(0.0, 0.85 - report["selected"]["dual"]) / 0.15
        + max(0.0, report["selected"]["harm"] - 0.15) / 0.15
        + max(0.0, report["relative"]["max_quality_degradation_m"]) / 0.01
        + guard_failures
    )


def consistent_direction(report: dict[str, Any]) -> bool:
    fixed = report["fixed"]
    selected = report["selected"]
    return bool(
        report["relative"]["coverage_gain"] > 0.0
        and selected["dual"] >= fixed["dual"] - 1.0e-12
        and selected["harm"] <= fixed["harm"] + 1.0e-12
        and report["relative"]["all_quality_nondegraded"]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--af-root", type=Path, required=True)
    parser.add_argument("--gh-root", type=Path, required=True)
    parser.add_argument("--trajectory-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261067)
    args = parser.parse_args()
    result_path = args.output_dir / "result.json"
    selected_path = args.output_dir / "crossfit_selected_index.npz"
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(args.output_dir)
    prereg = read_json(args.preregistration)
    if (
        prereg.get("status")
        != "bounded_rule_family_and_stop_condition_frozen_before_A_H_summary"
        or prereg.get("outcomes_summarized_after_this_freeze")
        or prereg.get("reserve_accessed")
        or prereg["grid"]["total_rules"] != 32
        or not all(prereg["prohibitions"].values())
    ):
        raise RuntimeError("v61 preregistration is not frozen")
    configs = config_grid(prereg)

    reserve_path = (
        args.af_root / "final_confirmation_reserve/selection_v26/freeze.json"
    )
    reserve = read_json(reserve_path)
    reserve_sha_before = sha256(reserve_path)
    if reserve.get("labels_opened") or reserve.get("outcomes_accessed"):
        raise RuntimeError("sealed reserve was accessed")

    bank_data: dict[str, dict[str, np.ndarray]] = {}
    artifacts: dict[str, Any] = {}
    route_sets: dict[str, set[str]] = {}
    for bank in BANKS:
        data, audit = load_bank(
            bank,
            af_root=args.af_root,
            gh_root=args.gh_root,
            trajectory_root=args.trajectory_root,
            device=args.device,
        )
        policy = base_policy(data)
        data["policy"] = policy
        bank_data[bank] = data
        artifacts[bank] = audit
        route_sets[bank] = set(data["route"].astype(str).tolist())
    for index, first in enumerate(BANKS):
        for second in BANKS[index + 1 :]:
            if route_sets[first] & route_sets[second]:
                raise RuntimeError(f"route overlap: {first}/{second}")

    # Identity check against the previously archived v59/v60 selections.
    v59 = read_json(
        ROOT / "outputs/cem_development_v1/recon_cem_frontier_v59/result.json"
    )
    entry29 = v59["frontier"][29]
    af_fixed_count = sum(
        int((bank_data[bank]["policy"]["fixed"] != 0).sum()) for bank in AF_BANKS
    )
    af_add_count = sum(
        int((bank_data[bank]["policy"]["addonly"] != 0).sum()) for bank in AF_BANKS
    )
    if (
        af_fixed_count != entry29["pooled"]["fixed"]["promotions"]
        or af_add_count != entry29["pooled"]["combined"]["promotions"]
    ):
        raise RuntimeError("A-F v60 boundary identity replay changed")
    secondary = read_json(
        args.gh_root / "evaluation_v60/addonly_secondary_decomposition.json"
    )
    gh_fixed_count = sum(
        int((bank_data[bank]["policy"]["fixed"] != 0).sum()) for bank in GH_BANKS
    )
    gh_add_count = sum(
        int((bank_data[bank]["policy"]["addonly"] != 0).sum()) for bank in GH_BANKS
    )
    if (
        gh_fixed_count
        != secondary["pooled"]["methods"]["cem_false_fixed"]["promotions"]
        or gh_add_count
        != secondary["pooled"]["methods"]["cem_addonly_secondary"]["promotions"]
    ):
        raise RuntimeError("G/H add-only identity replay changed")

    selections: dict[int, dict[str, np.ndarray]] = {}
    diagnostics: dict[int, dict[str, dict[str, np.ndarray]]] = {}
    for config in configs:
        selection = {}
        diagnostic = {}
        for bank in BANKS:
            selection[bank], diagnostic[bank] = select_rule(
                bank_data[bank], bank_data[bank]["policy"], config
            )
        selections[config["id"]] = selection
        diagnostics[config["id"]] = diagnostic

    guard = prereg["inner_selection"]["catastrophe_guard"]
    outer_folds: dict[str, Any] = {}
    outer_selected: dict[str, np.ndarray] = {}
    outer_config_id: dict[str, int] = {}
    for heldout in BANKS:
        inner = tuple(bank for bank in BANKS if bank != heldout)
        candidates = []
        for config in configs:
            config_id = config["id"]
            report = pooled_report(bank_data, selections[config_id], inner)
            per_bank = {
                bank: pooled_report(bank_data, selections[config_id], (bank,))
                for bank in inner
            }
            guard_failures = sum(
                not catastrophe_pass(per_bank[bank], guard) for bank in inner
            )
            eligible = strict_pass(report) and guard_failures == 0
            safety_slack = min(
                min(
                    per_bank[bank]["selected"]["dual"] - float(guard["dual_min"]),
                    float(guard["harm_max"])
                    - per_bank[bank]["selected"]["harm"],
                    per_bank[bank]["relative"]["coverage_gain"]
                    - float(guard["coverage_gain_min"]),
                )
                for bank in inner
            )
            candidates.append(
                {
                    "config": config,
                    "inner": report,
                    "guard_failures": guard_failures,
                    "eligible": eligible,
                    "safety_slack": safety_slack,
                    "violation": normalized_violation(report, guard_failures),
                }
            )
        eligible = [entry for entry in candidates if entry["eligible"]]
        if eligible:
            chosen = max(
                eligible,
                key=lambda entry: (
                    entry["safety_slack"],
                    entry["inner"]["selected"]["dual"],
                    -entry["inner"]["selected"]["harm"],
                    entry["inner"]["relative"]["coverage_gain"],
                    -entry["config"]["id"],
                ),
            )
            selection_mode = "primary_eligible"
        else:
            chosen = min(
                candidates,
                key=lambda entry: (
                    entry["violation"],
                    -entry["safety_slack"],
                    -entry["inner"]["relative"]["coverage_gain"],
                    entry["config"]["id"],
                ),
            )
            selection_mode = "frozen_minimum_violation_fallback"
        config_id = int(chosen["config"]["id"])
        outer_config_id[heldout] = config_id
        outer_selected[heldout] = selections[config_id][heldout]
        heldout_report = pooled_report(
            bank_data, {heldout: outer_selected[heldout]}, (heldout,)
        )
        outer_folds[heldout] = {
            "states": int(len(outer_selected[heldout])),
            "routes": int(len(route_sets[heldout])),
            "selected_config": chosen["config"],
            "inner_selection_mode": selection_mode,
            "inner_primary_eligible_rules": len(eligible),
            "inner_selected_metrics": chosen["inner"],
            "inner_selected_guard_failures": chosen["guard_failures"],
            "inner_selected_normalized_violation": chosen["violation"],
            "heldout": heldout_report,
            "heldout_consistent_direction": consistent_direction(heldout_report),
            "heldout_catastrophe_guard_pass": catastrophe_pass(
                heldout_report, guard
            ),
            "replacement_count": int(
                diagnostics[config_id][heldout]["replaced"].sum()
            ),
            "veto_count": int(diagnostics[config_id][heldout]["vetoed"].sum()),
        }

    absolute = np.concatenate([bank_data[bank]["absolute"] for bank in BANKS])
    trajectory = np.concatenate([bank_data[bank]["trajectory"] for bank in BANKS])
    route = np.concatenate([bank_data[bank]["route"] for bank in BANKS])
    record = np.concatenate([bank_data[bank]["record_key"] for bank in BANKS])
    bank_name = np.concatenate(
        [np.full(len(bank_data[bank]["record_key"]), bank) for bank in BANKS]
    )
    fixed = np.concatenate([bank_data[bank]["policy"]["fixed"] for bank in BANKS])
    base_addonly = np.concatenate(
        [bank_data[bank]["policy"]["addonly"] for bank in BANKS]
    )
    selected = np.concatenate([outer_selected[bank] for bank in BANKS])
    fixed_metrics = method_metrics(
        fixed,
        absolute,
        trajectory,
        route,
        replicates=args.bootstrap_replicates,
        seed=args.bootstrap_seed,
    )
    addonly_metrics = method_metrics(
        base_addonly,
        absolute,
        trajectory,
        route,
        replicates=args.bootstrap_replicates,
        seed=args.bootstrap_seed + 10,
    )
    selected_metrics = method_metrics(
        selected,
        absolute,
        trajectory,
        route,
        replicates=args.bootstrap_replicates,
        seed=args.bootstrap_seed + 20,
    )
    direct_fixed = direct_metrics(fixed, absolute)
    direct_addonly = direct_metrics(base_addonly, absolute)
    direct_selected = direct_metrics(selected, absolute)
    selected_relative = comparison(direct_selected, direct_fixed)
    base_addonly_relative = comparison(direct_addonly, direct_fixed)
    route_comparison = route_bootstrap_comparison(
        fixed,
        selected,
        absolute,
        route,
        replicates=args.bootstrap_replicates,
        seed=args.bootstrap_seed + 30,
    )
    selected_metrics["route_clustered_safety_ci95"] = route_method_ci(
        selected,
        absolute,
        route,
        replicates=args.bootstrap_replicates,
        seed=args.bootstrap_seed + 40,
    )
    consistent_banks = sum(
        bool(value["heldout_consistent_direction"])
        for value in outer_folds.values()
    )
    catastrophic_banks = sum(
        not bool(value["heldout_catastrophe_guard_pass"])
        for value in outer_folds.values()
    )
    final_checks = {
        "coverage_gain_at_least_0.10": selected_relative["coverage_gain"] >= 0.10,
        "dual_at_least_0.85": direct_selected["dual"] >= 0.85,
        "harm_at_most_0.15": direct_selected["harm"] <= 0.15,
        "all_four_mean_and_p90_nondegraded": selected_relative[
            "all_quality_nondegraded"
        ],
        "at_least_two_heldout_banks_consistent_direction": consistent_banks >= 2,
        "no_catastrophic_heldout_fold": catastrophic_banks == 0,
    }
    final_pass = all(final_checks.values())

    # Outcome-aware shared-rule frontier is diagnostic only; it cannot select
    # the cross-fitted outer policy.
    frontier = []
    for config in configs:
        config_id = int(config["id"])
        report = pooled_report(bank_data, selections[config_id], BANKS)
        per_bank = {
            bank: pooled_report(bank_data, selections[config_id], (bank,))
            for bank in BANKS
        }
        guards = {
            bank: catastrophe_pass(per_bank[bank], guard) for bank in BANKS
        }
        frontier.append(
            {
                "config": config,
                "pooled": report,
                "all_bank_guards_pass": all(guards.values()),
                "bank_guard_pass": guards,
                "strict_pooled_pass": strict_pass(report),
                "replacement_count": int(
                    sum(diagnostics[config_id][bank]["replaced"].sum() for bank in BANKS)
                ),
                "veto_count": int(
                    sum(diagnostics[config_id][bank]["vetoed"].sum() for bank in BANKS)
                ),
            }
        )
    safe_quality = [
        entry
        for entry in frontier
        if entry["pooled"]["selected"]["dual"] >= 0.85
        and entry["pooled"]["selected"]["harm"] <= 0.15
        and entry["pooled"]["relative"]["all_quality_nondegraded"]
    ]
    coverage_eligible = [
        entry
        for entry in frontier
        if entry["pooled"]["relative"]["coverage_gain"] >= 0.10
    ]
    best_violation = min(
        frontier,
        key=lambda entry: (
            normalized_violation(
                entry["pooled"],
                sum(not value for value in entry["bank_guard_pass"].values()),
            ),
            -entry["pooled"]["relative"]["coverage_gain"],
            entry["config"]["id"],
        ),
    )

    fixed_promote = fixed != 0
    selected_changed = selected != fixed
    fixed_delta = absolute[np.arange(len(fixed)), fixed] - absolute[:, 0]
    fixed_harm = fixed_promote & (
        (fixed_delta[:, 0] > 1.0e-7) | (fixed_delta[:, 3] > 1.0e-7)
    )
    fixed_dual = fixed_promote & (
        (fixed_delta[:, 0] < -1.0e-7) & (fixed_delta[:, 3] < -1.0e-7)
    )
    signal_attribution = {
        "fixed_promotions": int(fixed_promote.sum()),
        "fixed_harms": int(fixed_harm.sum()),
        "fixed_dual_successes": int(fixed_dual.sum()),
        "fixed_harms_changed_by_crossfit": int((fixed_harm & selected_changed).sum()),
        "fixed_dual_successes_changed_by_crossfit": int(
            (fixed_dual & selected_changed).sum()
        ),
        "total_fixed_actions_changed": int((fixed_promote & selected_changed).sum()),
        "crossfit_replacements": int(
            sum(
                diagnostics[outer_config_id[bank]][bank]["replaced"].sum()
                for bank in BANKS
            )
        ),
        "crossfit_vetoes": int(
            sum(
                diagnostics[outer_config_id[bank]][bank]["vetoed"].sum()
                for bank in BANKS
            )
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_npz(
        selected_path,
        record_key=record,
        bank=bank_name,
        route=route,
        fixed_selected_index=fixed,
        frozen_addonly_selected_index=base_addonly,
        crossfit_selected_index=selected,
        outer_config_id=np.concatenate(
            [
                np.full(len(bank_data[bank]["record_key"]), outer_config_id[bank])
                for bank in BANKS
            ]
        ).astype(np.int64),
    )
    reserve_sha_after = sha256(reserve_path)
    if reserve_sha_after != reserve_sha_before:
        raise RuntimeError("reserve changed during v61 audit")
    structural_added = int((base_addonly != 0).sum() - (fixed != 0).sum())
    structural_coverage_ceiling = structural_added / len(fixed)
    report = {
        "schema": "recon_cem_fixed_harm_veto_v61_crossfit_result",
        "status": (
            "cross_fitted_development_result_passed"
            if final_pass
            else "existing_cached_signals_not_separable_stop_bounded_hypothesis"
        ),
        "evidence_role": "spent_A_to_H_bank_outer_cross_fitted_development_only",
        "preregistration": str(args.preregistration),
        "preregistration_sha256": sha256(args.preregistration),
        "banks": list(BANKS),
        "states": int(len(fixed)),
        "routes": int(len(np.unique(route))),
        "route_overlap": 0,
        "grid_rules": len(configs),
        "frozen_fixed_promotions": int((fixed != 0).sum()),
        "frozen_addonly_promotions": int((base_addonly != 0).sum()),
        "frozen_addonly_net_added": structural_added,
        "structural_coverage_gain_ceiling_for_replacement_or_veto_only_family": structural_coverage_ceiling,
        "structural_coverage_observation": (
            "Replacement preserves coverage and veto can only reduce it; with existing "
            "add-only identities this family cannot exceed the frozen net-add count."
        ),
        "outer_folds": outer_folds,
        "cross_fitted_pooled": {
            "fixed": fixed_metrics,
            "frozen_addonly": addonly_metrics,
            "selected": selected_metrics,
            "selected_relative_to_fixed": selected_relative,
            "frozen_addonly_relative_to_fixed": base_addonly_relative,
            "selected_minus_fixed_route_clustered": route_comparison,
            "consistent_heldout_banks": consistent_banks,
            "catastrophic_heldout_banks": catastrophic_banks,
            "acceptance_checks": final_checks,
            "accepted": final_pass,
        },
        "signal_attribution": signal_attribution,
        "attainable_frontier": frontier,
        "frontier_summary": {
            "rules_with_strict_pooled_pass": sum(
                bool(entry["strict_pooled_pass"]) for entry in frontier
            ),
            "rules_with_coverage_gain_at_least_0.10": len(coverage_eligible),
            "rules_with_pooled_safety_and_quality": len(safe_quality),
            "maximum_coverage_gain": max(
                entry["pooled"]["relative"]["coverage_gain"] for entry in frontier
            ),
            "maximum_coverage_gain_with_safety_and_quality": (
                max(
                    entry["pooled"]["relative"]["coverage_gain"]
                    for entry in safe_quality
                )
                if safe_quality
                else None
            ),
            "maximum_dual_at_maximum_coverage": max(
                entry["pooled"]["selected"]["dual"]
                for entry in frontier
                if abs(
                    entry["pooled"]["relative"]["coverage_gain"]
                    - max(
                        item["pooled"]["relative"]["coverage_gain"]
                        for item in frontier
                    )
                )
                < 1.0e-12
            ),
            "minimum_harm_at_maximum_coverage": min(
                entry["pooled"]["selected"]["harm"]
                for entry in frontier
                if abs(
                    entry["pooled"]["relative"]["coverage_gain"]
                    - max(
                        item["pooled"]["relative"]["coverage_gain"]
                        for item in frontier
                    )
                )
                < 1.0e-12
            ),
            "minimum_violation_shared_rule": best_violation,
        },
        "artifacts": artifacts,
        "crossfit_selected_index": str(selected_path),
        "crossfit_selected_index_sha256": sha256(selected_path),
        "new_model_trained": False,
        "new_feature_extracted": False,
        "new_candidates_sampled": False,
        "new_CEM_run": False,
        "outcomes_used_to_train_network": False,
        "cohort_specific_thresholds": False,
        "post_result_tuning_allowed": False,
        "bounded_hypothesis_terminal": not final_pass,
        "reserve": str(reserve_path),
        "reserve_sha256_before": reserve_sha_before,
        "reserve_sha256_after": reserve_sha_after,
        "reserve_accessed": False,
        "production_safe_claim_allowed": False,
        "bootstrap_replicates": args.bootstrap_replicates,
        "bootstrap_seed": args.bootstrap_seed,
    }
    write_json(result_path, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "structural_coverage_ceiling": structural_coverage_ceiling,
                "cross_fitted_pooled": {
                    "fixed": direct_fixed,
                    "frozen_addonly": direct_addonly,
                    "selected": direct_selected,
                    "relative": selected_relative,
                    "checks": final_checks,
                    "consistent_heldout_banks": consistent_banks,
                    "catastrophic_heldout_banks": catastrophic_banks,
                },
                "signal_attribution": signal_attribution,
                "frontier_summary": report["frontier_summary"],
                "outer_config_ids": outer_config_id,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
