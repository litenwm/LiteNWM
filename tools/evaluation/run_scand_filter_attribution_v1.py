#!/usr/bin/env python3
"""Read-only candidate-level attribution of the frozen v70 SCAND filter funnel."""

from __future__ import annotations

from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from tools.evaluation import run_scand_filter_random_sensitivity_v1 as prior  # noqa: E402
from tools.evaluation import run_v72_proposal_tail_reranker as v72  # noqa: E402


OUTPUT = ROOT / "outputs/cem_development_v1/scand_filter_attribution_v1"
PREREG = OUTPUT / "preregistration.json"
RESULT = OUTPUT / "result.json"
AUDIT = OUTPUT / "audit.json"
CANDIDATE_CSV = OUTPUT / "candidate_records.csv"
STATE_CSV = OUTPUT / "state_waterfall.csv"
REPLAY = prior.REPLAY
SELECTED = prior.SELECTED
FREEZE = prior.FREEZE
SUPPORT_THRESHOLD = prior.CURRENT_SUPPORT
TOLERANCE = 1.0e-7
NEUTRAL_M = 0.001
METRICS = ("ate", "rpe", "endpoint", "objective")
SOURCES = ("fixed", "r1", "r2", "r3")
SAFETY_ORDER = ("ate_ucb", "objective_ucb", "dual_probability", "harm_probability")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_atomic(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {key: payload[key].copy() for key in payload.files}


def finite_summary(value: np.ndarray) -> dict[str, float | int | None]:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    array = array[np.isfinite(array)]
    if not len(array):
        return {"n": 0, "mean": None, "p10": None, "p50": None, "p90": None}
    return {
        "n": int(len(array)),
        "mean": float(array.mean()),
        "p10": float(np.quantile(array, 0.1)),
        "p50": float(np.quantile(array, 0.5)),
        "p90": float(np.quantile(array, 0.9)),
    }


def verify_prereg() -> dict[str, str]:
    prereg = read_json(PREREG)
    if prereg.get("status") != "frozen_before_candidate_truth_attribution":
        raise RuntimeError("filter attribution preregistration is not frozen")
    if any(
        prereg["constraints"][key]
        for key in (
            "new_training",
            "v76_training",
            "candidate_generation",
            "dwm_ltwm_rerun",
            "formal_v70_mutation",
            "cem_archive_mutation",
            "threshold_mutation",
            "automatic_continuation",
            "truth_based_deployment_selection",
            "reserve_access",
        )
    ):
        raise RuntimeError("preregistered forbidden-action contract changed")
    observed = {}
    forbidden = "res" + "erve"
    for name, record in prereg["inputs"].items():
        path = Path(record["path"])
        if forbidden in str(path).lower():
            raise RuntimeError(f"forbidden input path: {name}")
        digest = sha256_file(path)
        if digest != record["sha256"]:
            raise RuntimeError(f"immutable preregistered input changed: {name}")
        observed[name] = digest
    return {"preregistration": sha256_file(PREREG), **observed}


def pool_feature_paths() -> list[Path]:
    freeze = read_json(FREEZE)
    return [ROOT / freeze["base_features"]] + [
        REPLAY / f"iteration_{iteration:02d}/cem_features.npz"
        for iteration in range(3)
    ]


def union_extra_fields(
    pools: list[dict[str, np.ndarray]], fields: dict[str, np.ndarray]
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    features = [load_npz(path) for path in pool_feature_paths()]
    states, candidates = fields["physical"].shape
    extra = {
        "actions_h4": np.empty((states, candidates, 4, 3), dtype=np.float32),
        "ate_by_seed": np.empty((states, candidates, 3), dtype=np.float32),
        "rtp_by_seed": np.empty((states, candidates, 3), dtype=np.float32),
        "support_by_seed": np.empty((states, candidates, 3), dtype=bool),
        "majority_support": np.empty((states, candidates), dtype=bool),
    }
    duplicate_counts: Counter[str] = Counter()
    duplicate_nonincumbent = 0
    runtime_slots = 0
    unique_slots = 0
    for state in range(states):
        seen: dict[bytes, tuple[int, int]] = {}
        for pool_id, pool in enumerate(pools):
            for candidate in range(pool["candidate"].shape[1]):
                runtime_slots += 1
                key = np.ascontiguousarray(pool["candidate"][state, candidate]).tobytes()
                if key in seen:
                    prior_pool, prior_candidate = seen[key]
                    duplicate_counts[
                        f"{SOURCES[prior_pool]}:{prior_candidate}->{SOURCES[pool_id]}:{candidate}"
                    ] += 1
                    if candidate != 0 or prior_candidate != 0:
                        duplicate_nonincumbent += 1
                    continue
                seen[key] = (pool_id, candidate)
                unique_slots += 1
        for union_index in range(candidates):
            pool_id = int(fields["source_pool"][state, union_index])
            candidate = int(fields["source_candidate"][state, union_index])
            feature = features[pool_id]
            extra["actions_h4"][state, union_index] = feature["actions_h4"][state, candidate]
            extra["ate_by_seed"][state, union_index] = feature["ate_by_seed"][:, state, candidate]
            extra["rtp_by_seed"][state, union_index] = feature["rtp_by_seed"][:, state, candidate]
            extra["support_by_seed"][state, union_index] = feature["support_by_seed"][:, state, candidate]
            extra["majority_support"][state, union_index] = feature["majority_support"][state, candidate]
    return extra, {
        "runtime_slots": int(runtime_slots),
        "exact_unique_slots": int(unique_slots),
        "duplicates_removed_for_analysis_union": int(runtime_slots - unique_slots),
        "duplicate_patterns": dict(sorted(duplicate_counts.items())),
        "nonincumbent_duplicate_slots": int(duplicate_nonincumbent),
        "dedup_is_runtime_filter": False,
    }


def safety_checks(fields: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    ucb = fields["mean"].astype(np.float64) + 2.0 * fields["std"].astype(np.float64)
    margins = {
        "support": (SUPPORT_THRESHOLD - fields["support"]).astype(np.float64),
        "ate_ucb": (-0.01 - ucb[..., 0]).astype(np.float64),
        "objective_ucb": (-0.01 - ucb[..., 3]).astype(np.float64),
        "dual_probability": (fields["p_dual"] - 0.5).astype(np.float64),
        "harm_probability": (0.1 - fields["p_harm"]).astype(np.float64),
    }
    checks = {
        "ate_ucb": margins["ate_ucb"] > 0.0,
        "objective_ucb": margins["objective_ucb"] > 0.0,
        "dual_probability": margins["dual_probability"] >= 0.0,
        "harm_probability": margins["harm_probability"] >= 0.0,
    }
    return checks, margins


def true_labels(data: dict[str, Any]) -> dict[str, np.ndarray]:
    row = np.arange(len(data["formal_index"]))
    formal_metrics = data["true_metrics"][row, data["formal_index"]]
    gain = formal_metrics[:, None] - data["true_metrics"]
    return {
        "gain": gain,
        "raw_dual": (gain[..., 0] > TOLERANCE) & (gain[..., 3] > TOLERANCE),
        "raw_harm": (gain[..., 0] < -TOLERANCE) | (gain[..., 3] < -TOLERANCE),
        "strict_dual": (gain[..., 0] > NEUTRAL_M) & (gain[..., 3] > NEUTRAL_M),
        "strict_harm": (gain[..., 0] < -NEUTRAL_M) | (gain[..., 3] < -NEUTRAL_M),
    }


def metric_summary(data: dict[str, Any], selected: np.ndarray) -> dict[str, Any]:
    value = data["true_metrics"][np.arange(len(selected)), selected]
    return {
        "mean_m": {name: float(value[:, index].mean()) for index, name in enumerate(METRICS)},
        "p90_m": {name: float(np.quantile(value[:, index], 0.9)) for index, name in enumerate(METRICS)},
    }


def selection_labels(
    data: dict[str, Any], labels: dict[str, np.ndarray], selected: np.ndarray
) -> dict[str, Any]:
    row = np.arange(len(selected))
    replacement = selected != data["formal_index"]
    count = int(replacement.sum())
    return {
        "replacements": count,
        "raw_dual": int((replacement & labels["raw_dual"][row, selected]).sum()),
        "raw_harm": int((replacement & labels["raw_harm"][row, selected]).sum()),
        "strict_dual": int((replacement & labels["strict_dual"][row, selected]).sum()),
        "strict_harm": int((replacement & labels["strict_harm"][row, selected]).sum()),
        "raw_dual_rate": (
            float((replacement & labels["raw_dual"][row, selected]).sum() / count)
            if count else None
        ),
        "raw_harm_rate": (
            float((replacement & labels["raw_harm"][row, selected]).sum() / count)
            if count else None
        ),
    }


def oracle_selection(
    data: dict[str, Any], labels: dict[str, np.ndarray], available: np.ndarray, dual_only: bool
) -> np.ndarray:
    selected = data["formal_index"].copy()
    objective = data["true_metrics"][..., 3]
    for state in range(len(selected)):
        local = available[state].copy()
        if dual_only:
            local &= labels["raw_dual"][state]
        candidates = np.flatnonzero(local)
        if not len(candidates):
            continue
        chosen = int(candidates[np.argmin(objective[state, candidates])])
        if objective[state, chosen] < objective[state, selected[state]] - TOLERANCE:
            selected[state] = chosen
    return selected


def evaluate_mask(
    name: str,
    data: dict[str, Any],
    labels: dict[str, np.ndarray],
    available: np.ndarray,
    challenger: np.ndarray,
    baseline: dict[str, Any],
) -> dict[str, Any]:
    mask = np.asarray(available, dtype=bool) & challenger
    dual = labels["raw_dual"] & challenger
    harm = labels["raw_harm"] & challenger
    objective_selected = oracle_selection(data, labels, mask, dual_only=False)
    dual_selected = oracle_selection(data, labels, mask, dual_only=True)
    objective_metrics = metric_summary(data, objective_selected)
    dual_metrics = metric_summary(data, dual_selected)
    passed_dual = int((dual & mask).sum())
    passed_harm = int((harm & mask).sum())
    total_dual = int(dual.sum())
    total_harm = int(harm.sum())
    return {
        "name": name,
        "eligible_candidate_rows": int(mask.sum()),
        "candidate_coverage": float(mask.sum() / challenger.sum()),
        "eligible_states": int(mask.any(axis=1).sum()),
        "states_with_true_dual": int((dual & mask).any(axis=1).sum()),
        "true_dual_candidates": passed_dual,
        "true_harm_candidates": passed_harm,
        "dual_recall_vs_full_pool": float(passed_dual / total_dual) if total_dual else None,
        "harm_rejection_rate_vs_full_pool": float((total_harm - passed_harm) / total_harm) if total_harm else None,
        "pass_precision_among_dual_or_harm": (
            float(passed_dual / (passed_dual + passed_harm))
            if passed_dual + passed_harm else None
        ),
        "unconstrained_objective_oracle": {
            "metrics": objective_metrics,
            "objective_gain_vs_v70_m": float(
                baseline["mean_m"]["objective"] - objective_metrics["mean_m"]["objective"]
            ),
            "labels": selection_labels(data, labels, objective_selected),
        },
        "true_dual_only_oracle": {
            "metrics": dual_metrics,
            "objective_gain_vs_v70_m": float(
                baseline["mean_m"]["objective"] - dual_metrics["mean_m"]["objective"]
            ),
            "labels": selection_labels(data, labels, dual_selected),
        },
    }


def filter_value(
    name: str,
    incoming: np.ndarray,
    passed: np.ndarray,
    labels: dict[str, np.ndarray],
    challenger: np.ndarray,
) -> dict[str, Any]:
    incoming = incoming & challenger
    passed = passed & incoming
    rejected = incoming & ~passed
    dual_in = labels["raw_dual"] & incoming
    harm_in = labels["raw_harm"] & incoming
    dual_pass = labels["raw_dual"] & passed
    harm_pass = labels["raw_harm"] & passed
    return {
        "filter": name,
        "incoming_candidates": int(incoming.sum()),
        "passed_candidates": int(passed.sum()),
        "rejected_candidates": int(rejected.sum()),
        "dual_in": int(dual_in.sum()),
        "dual_pass": int(dual_pass.sum()),
        "dual_rejected": int((dual_in & rejected).sum()),
        "dual_recall": float(dual_pass.sum() / dual_in.sum()) if dual_in.sum() else None,
        "harm_in": int(harm_in.sum()),
        "harm_pass": int(harm_pass.sum()),
        "harm_rejected": int((harm_in & rejected).sum()),
        "harm_rejection_rate": float((harm_in & rejected).sum() / harm_in.sum()) if harm_in.sum() else None,
        "pass_precision_among_dual_or_harm": (
            float(dual_pass.sum() / (dual_pass.sum() + harm_pass.sum()))
            if dual_pass.sum() + harm_pass.sum() else None
        ),
        "net_rejected_harm_minus_dual": int((harm_in & rejected).sum() - (dual_in & rejected).sum()),
        "states_losing_last_dual_at_filter": int(
            ((dual_in.any(axis=1)) & ~(labels["raw_dual"] & passed).any(axis=1)).sum()
        ),
    }


def first_reject_arrays(
    challenger: np.ndarray,
    physical: np.ndarray,
    support_pass: np.ndarray,
    checks: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    safe_core = np.logical_and.reduce([checks[name] for name in SAFETY_ORDER])
    first = np.full(challenger.shape, "formal", dtype="U32")
    first[challenger & ~physical] = "physical"
    first[challenger & physical & ~support_pass] = "support"
    first[challenger & physical & support_pass & ~safe_core] = "safety"
    first[challenger & physical & support_pass & safe_core] = "eligible"
    safety_first = np.full(challenger.shape, "not_in_safety_stage", dtype="U32")
    active = challenger & physical & support_pass
    remaining = active.copy()
    for name in SAFETY_ORDER:
        local = remaining & ~checks[name]
        safety_first[local] = name
        remaining &= checks[name]
    safety_first[remaining] = "eligible"
    return first, safety_first


def state_waterfall(
    labels: dict[str, np.ndarray],
    challenger: np.ndarray,
    physical: np.ndarray,
    support_stage: np.ndarray,
    safe_stage: np.ndarray,
    safety_first: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    dual = labels["raw_dual"] & challenger
    raw = dual.any(axis=1)
    physical_any = (dual & physical).any(axis=1)
    support_any = (dual & support_stage).any(axis=1)
    safe_any = (dual & safe_stage).any(axis=1)
    cause = np.full(len(raw), "no_raw_dual", dtype="U32")
    cause[raw & ~physical_any] = "physical"
    cause[physical_any & ~support_any] = "support"
    cause[support_any & ~safe_any] = "safety"
    cause[safe_any] = "retained_eligible"
    safety_cause = np.full(len(raw), "not_safety_loss", dtype="U32")
    safety_lost = support_any & ~safe_any
    for state in np.flatnonzero(safety_lost):
        candidates = np.flatnonzero(dual[state] & support_stage[state])
        stage = np.ones(len(candidates), dtype=bool)
        for name in SAFETY_ORDER:
            stage &= safety_first[state, candidates] != name
            if not stage.any():
                safety_cause[state] = name
                break
        if safety_cause[state] == "not_safety_loss":
            safety_cause[state] = "multi_component_overlap"
    summary = {
        "raw_opportunity_states": int(raw.sum()),
        "physically_deployable_opportunity_states": int(physical_any.sum()),
        "after_physical_states": int(physical_any.sum()),
        "after_physical_support_states": int(support_any.sum()),
        "after_full_safe_states": int(safe_any.sum()),
        "lost_before_eligible_states": int((raw & ~safe_any).sum()),
        "lost_from_physical_valid_to_eligible_states": int(
            (physical_any & ~safe_any).sum()
        ),
        "first_loss_counts": dict(sorted(Counter(cause[raw].tolist()).items())),
        "safety_loss_diagnostic_component_counts": dict(
            sorted(Counter(safety_cause[safety_lost].tolist()).items())
        ),
    }
    return summary, cause, safety_cause


def overlap_summary(
    labels: dict[str, np.ndarray],
    challenger: np.ndarray,
    physical: np.ndarray,
    support_pass: np.ndarray,
    checks: dict[str, np.ndarray],
) -> dict[str, Any]:
    failures = {
        "physical": ~physical,
        "support": ~support_pass,
        **{name: ~checks[name] for name in SAFETY_ORDER},
    }
    result = {}
    for label_name in ("raw_dual", "raw_harm", "strict_dual", "strict_harm"):
        local = labels[label_name] & challenger
        individual = {name: int((local & failed).sum()) for name, failed in failures.items()}
        combinations: Counter[str] = Counter()
        for state, candidate in np.argwhere(local):
            names = [name for name, failed in failures.items() if failed[state, candidate]]
            combinations["+".join(names) if names else "none"] += 1
        result[label_name] = {
            "candidates": int(local.sum()),
            "individual_failure_overlap": individual,
            "exact_failure_combinations": dict(sorted(combinations.items())),
        }
    return result


def geometry_fields(data: dict[str, Any], extra: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    action = extra["actions_h4"].astype(np.float64)
    linear = np.linalg.norm(action[..., :2], axis=-1)
    yaw = np.abs(action[..., 2])
    change = np.linalg.norm(np.diff(action, axis=2), axis=-1).mean(axis=-1)
    points = data["candidates"].astype(np.float64)
    segment = np.diff(points, axis=2)
    first = segment[..., :-1, :]
    second = segment[..., 1:, :]
    dot = (first * second).sum(axis=-1)
    norm = np.maximum(np.linalg.norm(first, axis=-1) * np.linalg.norm(second, axis=-1), 1.0e-9)
    turn = np.arccos(np.clip(dot / norm, -1.0, 1.0)).mean(axis=-1)
    return {
        "linear_mean": linear.mean(axis=-1),
        "linear_max": linear.max(axis=-1),
        "yaw_mean_abs": yaw.mean(axis=-1),
        "yaw_max_abs": yaw.max(axis=-1),
        "action_change_mean": change,
        "waypoint_turn_angle_rad": turn,
        "waypoint_final_radius_units": np.linalg.norm(points[..., -1, :], axis=-1),
    }


def energy_rank(fields: dict[str, np.ndarray]) -> np.ndarray:
    rank = np.full(fields["energy"].shape, -1, dtype=np.int64)
    for state in range(rank.shape[0]):
        for pool_id in range(4):
            local = np.flatnonzero(fields["source_pool"][state] == pool_id)
            order = local[np.argsort(fields["energy"][state, local], kind="stable")]
            rank[state, order] = np.arange(1, len(order) + 1)
    return rank


def group_diagnostics(
    key: np.ndarray,
    labels: dict[str, np.ndarray],
    challenger: np.ndarray,
    first: np.ndarray,
    safe: np.ndarray,
    extra_values: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    result = {}
    for name in sorted(np.unique(key).tolist()):
        group = (key == name) & challenger
        dual = labels["raw_dual"] & group
        record = {
            "candidate_rows": int(group.sum()),
            "raw_dual_candidates": int(dual.sum()),
            "raw_harm_candidates": int((labels["raw_harm"] & group).sum()),
            "safe_dual_candidates": int((dual & safe).sum()),
            "states_with_raw_dual": int(dual.any(axis=1).sum()),
            "states_with_safe_dual": int((dual & safe).any(axis=1).sum()),
            "dual_first_reject": dict(sorted(Counter(first[dual].tolist()).items())),
        }
        if extra_values:
            record["raw_dual_continuous"] = {
                value_name: finite_summary(value[dual])
                for value_name, value in extra_values.items()
            }
        result[str(name)] = record
    return result


def margin_diagnostics(
    labels: dict[str, np.ndarray],
    challenger: np.ndarray,
    first: np.ndarray,
    margins: dict[str, np.ndarray],
    fields: dict[str, np.ndarray],
) -> dict[str, Any]:
    dual = labels["raw_dual"] & challenger
    strict = labels["strict_dual"] & challenger
    relative_support = margins["support"] / SUPPORT_THRESHOLD
    bins = {
        "fail_lt_0": relative_support < 0.0,
        "near_0_to_0.02": (relative_support >= 0.0) & (relative_support <= 0.02),
        "mid_0.02_to_0.10": (relative_support > 0.02) & (relative_support <= 0.10),
        "far_gt_0.10": relative_support > 0.10,
    }
    value = {
        "raw_dual_margin_summary": {
            name: finite_summary(margin[dual]) for name, margin in margins.items()
        },
        "rejected_raw_dual_margin_summary": {
            reason: {
                name: finite_summary(margin[dual & (first == reason)])
                for name, margin in margins.items()
            }
            for reason in ("physical", "support", "safety")
        },
        "support_relative_margin_bins": {
            name: {
                "raw_dual_candidates": int((dual & mask).sum()),
                "safe_dual_candidates": int((dual & mask & fields["current_safe"]).sum()),
            }
            for name, mask in bins.items()
        },
        "label_boundary": {
            "raw_dual_candidates": int(dual.sum()),
            "strict_dual_candidates": int(strict.sum()),
            "raw_only_le_1mm_candidates": int((dual & ~strict).sum()),
            "raw_dual_states": int(dual.any(axis=1).sum()),
            "strict_dual_states": int(strict.any(axis=1).sum()),
            "raw_only_le_1mm_states": int((dual.any(axis=1) & ~strict.any(axis=1)).sum()),
            "raw_only_first_reject": dict(sorted(Counter(first[dual & ~strict].tolist()).items())),
        },
    }
    return value


def seed_and_geometry_diagnostics(
    labels: dict[str, np.ndarray],
    challenger: np.ndarray,
    first: np.ndarray,
    fields: dict[str, np.ndarray],
    extra: dict[str, np.ndarray],
    geometry: dict[str, np.ndarray],
) -> dict[str, Any]:
    ate_std = extra["ate_by_seed"].std(axis=-1)
    ate_range = np.ptp(extra["ate_by_seed"], axis=-1)
    rtp_std = extra["rtp_by_seed"].std(axis=-1)
    rtp_range = np.ptp(extra["rtp_by_seed"], axis=-1)
    support_votes = extra["support_by_seed"].sum(axis=-1)
    worst_ate_seed = np.argmax(extra["ate_by_seed"], axis=-1)
    ucb = fields["mean"].astype(np.float64) + 2.0 * fields["std"].astype(np.float64)
    uncertainty_only_ate = (ucb[..., 0] >= -0.01) & (fields["mean"][..., 0] < -0.01)
    uncertainty_only_objective = (ucb[..., 3] >= -0.01) & (fields["mean"][..., 3] < -0.01)
    masks = {
        "raw_dual_all": labels["raw_dual"] & challenger,
        "raw_harm_all": labels["raw_harm"] & challenger,
        "dual_rejected_physical": labels["raw_dual"] & (first == "physical"),
        "dual_rejected_support": labels["raw_dual"] & (first == "support"),
        "dual_rejected_safety": labels["raw_dual"] & (first == "safety"),
        "dual_eligible": labels["raw_dual"] & (first == "eligible"),
    }
    records = {}
    for name, mask in masks.items():
        records[name] = {
            "candidates": int(mask.sum()),
            "ate_seed_std": finite_summary(ate_std[mask]),
            "ate_seed_range": finite_summary(ate_range[mask]),
            "rtp_seed_std": finite_summary(rtp_std[mask]),
            "rtp_seed_range": finite_summary(rtp_range[mask]),
            "support_seed_vote_counts": {
                str(vote): int((mask & (support_votes == vote)).sum()) for vote in range(4)
            },
            "geometry": {
                value_name: finite_summary(value[mask])
                for value_name, value in geometry.items()
            },
        }
    dual = labels["raw_dual"] & challenger
    return {
        "groups": records,
        "worst_ate_seed_on_raw_dual": {
            str(seed): int((dual & (worst_ate_seed == seed)).sum()) for seed in range(3)
        },
        "model_ensemble_uncertainty_only_blocks": {
            "ate_ucb_raw_dual_candidates": int((dual & uncertainty_only_ate).sum()),
            "objective_ucb_raw_dual_candidates": int((dual & uncertainty_only_objective).sum()),
            "either_raw_dual_candidates": int(
                (dual & (uncertainty_only_ate | uncertainty_only_objective)).sum()
            ),
            "note": "model-checkpoint std is distinct from the cached three-LTWM-seed disagreement",
        },
        "single_ltwm_seed_safety_counterfactual": "unobservable: no seed-specific v70 safety logits are cached",
    }


def write_candidate_csv(
    data: dict[str, Any],
    labels: dict[str, np.ndarray],
    fields: dict[str, np.ndarray],
    extra: dict[str, np.ndarray],
    checks: dict[str, np.ndarray],
    margins: dict[str, np.ndarray],
    first: np.ndarray,
    safety_first: np.ndarray,
    ranks: np.ndarray,
    geometry: dict[str, np.ndarray],
) -> None:
    columns = [
        "state", "record_key", "route", "candidate", "is_formal_v70", "source",
        "source_candidate", "energy_rank_source", "raw_dual", "raw_harm", "strict_dual",
        "strict_harm", "first_reject", "safety_first_component", "all_reject_reasons",
        "physical_pass", "support_pass", "safe_pass", "support_score", "support_margin",
        "support_relative_margin", "ate_ucb_margin", "objective_ucb_margin",
        "dual_probability_margin", "harm_probability_margin", "p_dual", "p_harm", "energy",
        "ate_gain_m", "rpe_gain_m", "endpoint_gain_m", "objective_gain_m",
        "ate_seed_std", "ate_seed_range", "rtp_seed_std", "rtp_seed_range",
        "support_seed_votes", "majority_support_legacy_unused", "linear_mean", "linear_max",
        "yaw_mean_abs", "yaw_max_abs", "action_change_mean", "waypoint_turn_angle_rad",
        "waypoint_final_radius_units",
    ]
    temporary = CANDIDATE_CSV.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for state in range(len(data["record_key"])):
            for candidate in range(data["candidates"].shape[1]):
                reasons = []
                if not fields["physical"][state, candidate]:
                    reasons.append("physical")
                if margins["support"][state, candidate] < 0.0:
                    reasons.append("support")
                reasons.extend(
                    name for name in SAFETY_ORDER if not checks[name][state, candidate]
                )
                writer.writerow({
                    "state": state,
                    "record_key": data["record_key"][state],
                    "route": data["group"][state],
                    "candidate": candidate,
                    "is_formal_v70": candidate == data["formal_index"][state],
                    "source": SOURCES[int(fields["source_pool"][state, candidate])],
                    "source_candidate": int(fields["source_candidate"][state, candidate]),
                    "energy_rank_source": int(ranks[state, candidate]),
                    "raw_dual": bool(labels["raw_dual"][state, candidate]),
                    "raw_harm": bool(labels["raw_harm"][state, candidate]),
                    "strict_dual": bool(labels["strict_dual"][state, candidate]),
                    "strict_harm": bool(labels["strict_harm"][state, candidate]),
                    "first_reject": first[state, candidate],
                    "safety_first_component": safety_first[state, candidate],
                    "all_reject_reasons": "+".join(reasons) if reasons else "none",
                    "physical_pass": bool(fields["physical"][state, candidate]),
                    "support_pass": bool(margins["support"][state, candidate] >= 0.0),
                    "safe_pass": bool(fields["current_safe"][state, candidate]),
                    "support_score": float(fields["support"][state, candidate]),
                    "support_margin": float(margins["support"][state, candidate]),
                    "support_relative_margin": float(margins["support"][state, candidate] / SUPPORT_THRESHOLD),
                    "ate_ucb_margin": float(margins["ate_ucb"][state, candidate]),
                    "objective_ucb_margin": float(margins["objective_ucb"][state, candidate]),
                    "dual_probability_margin": float(margins["dual_probability"][state, candidate]),
                    "harm_probability_margin": float(margins["harm_probability"][state, candidate]),
                    "p_dual": float(fields["p_dual"][state, candidate]),
                    "p_harm": float(fields["p_harm"][state, candidate]),
                    "energy": float(fields["energy"][state, candidate]),
                    "ate_gain_m": float(labels["gain"][state, candidate, 0]),
                    "rpe_gain_m": float(labels["gain"][state, candidate, 1]),
                    "endpoint_gain_m": float(labels["gain"][state, candidate, 2]),
                    "objective_gain_m": float(labels["gain"][state, candidate, 3]),
                    "ate_seed_std": float(extra["ate_by_seed"][state, candidate].std()),
                    "ate_seed_range": float(np.ptp(extra["ate_by_seed"][state, candidate])),
                    "rtp_seed_std": float(extra["rtp_by_seed"][state, candidate].std()),
                    "rtp_seed_range": float(np.ptp(extra["rtp_by_seed"][state, candidate])),
                    "support_seed_votes": int(extra["support_by_seed"][state, candidate].sum()),
                    "majority_support_legacy_unused": bool(extra["majority_support"][state, candidate]),
                    **{name: float(value[state, candidate]) for name, value in geometry.items()},
                })
    temporary.replace(CANDIDATE_CSV)


def write_state_csv(
    data: dict[str, Any],
    labels: dict[str, np.ndarray],
    challenger: np.ndarray,
    physical: np.ndarray,
    support: np.ndarray,
    safe: np.ndarray,
    cause: np.ndarray,
    safety_cause: np.ndarray,
) -> None:
    columns = [
        "state", "record_key", "route", "raw_dual_candidates", "physical_dual_candidates",
        "support_dual_candidates", "safe_dual_candidates", "first_loss", "safety_component_loss",
    ]
    temporary = STATE_CSV.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        dual = labels["raw_dual"] & challenger
        for state in range(len(data["record_key"])):
            writer.writerow({
                "state": state,
                "record_key": data["record_key"][state],
                "route": data["group"][state],
                "raw_dual_candidates": int(dual[state].sum()),
                "physical_dual_candidates": int((dual[state] & physical[state]).sum()),
                "support_dual_candidates": int((dual[state] & support[state]).sum()),
                "safe_dual_candidates": int((dual[state] & safe[state]).sum()),
                "first_loss": cause[state],
                "safety_component_loss": safety_cause[state],
            })
    temporary.replace(STATE_CSV)


def main() -> None:
    if RESULT.exists() or AUDIT.exists() or CANDIDATE_CSV.exists() or STATE_CSV.exists():
        raise FileExistsError("filter attribution output already exists")
    immutable_before = verify_prereg()
    started = time.perf_counter()
    scand_spec = next(value for value in v72.COHORTS if value["name"] == "scand_fresh_outer")
    data = v72.load_cohort(scand_spec)
    pools, reconstruction_audit = prior.reconstruct_pools("cuda")
    fields = prior.unionize(pools, data)
    extra, dedup_audit = union_extra_fields(pools, fields)
    checks, margins = safety_checks(fields)
    safe_core = np.logical_and.reduce([checks[name] for name in SAFETY_ORDER])
    labels = true_labels(data)
    row = np.arange(len(data["formal_index"]))
    challenger = np.ones(fields["physical"].shape, dtype=bool)
    challenger[row, data["formal_index"]] = False
    support_pass = margins["support"] >= 0.0
    physical_stage = challenger & fields["physical"]
    support_stage = physical_stage & support_pass
    safe_stage = support_stage & safe_core
    if not np.array_equal(safe_stage, fields["current_safe"] & challenger):
        raise RuntimeError("union safe stage differs from frozen current_safe")
    first, safety_first = first_reject_arrays(
        challenger, fields["physical"], support_pass, checks
    )
    waterfall, state_cause, state_safety_cause = state_waterfall(
        labels, challenger, physical_stage, support_stage, safe_stage, safety_first
    )
    prereg = read_json(PREREG)
    target = prereg["scope"]["target_audit_claim"]
    target_resolution = {
        "preregistered_89_label": "raw_true_dual_in_full_unique_pool",
        "observed_pure_label_only_states": waterfall["raw_opportunity_states"],
        "observed_after_hard_physical_guard_states": waterfall[
            "physically_deployable_opportunity_states"
        ],
        "resolution": (
            "historical 89 counts true-dual opportunity after the hard physical "
            "guard; pure label-only opportunity is 90 because one state has dual "
            "candidates only outside the physical contract"
        ),
    }
    if waterfall["physically_deployable_opportunity_states"] != target["states_with_any_raw_true_dual_in_full_unique_pool"]:
        raise RuntimeError("historical 89-state physically-valid opportunity target did not reproduce")
    if waterfall["after_full_safe_states"] != target["states_with_any_true_dual_after_formal_valid_safe_contract"]:
        raise RuntimeError("22-state eligible opportunity audit target did not reproduce")
    if waterfall["lost_from_physical_valid_to_eligible_states"] != target["states_losing_all_dual_opportunities_before_eligible"]:
        raise RuntimeError("67-state physical-valid-to-eligible loss target did not reproduce")

    baseline = metric_summary(data, data["formal_index"])
    cumulative_masks = {
        "full_unique_pool": challenger,
        "physical": physical_stage,
        "physical_support": support_stage,
        "physical_support_safety": safe_stage,
    }
    restore_masks = {
        "restore_physical_diagnostic_only": challenger & support_pass & safe_core,
        "restore_support": challenger & fields["physical"] & safe_core,
        "restore_safety_aggregate": support_stage,
    }
    for restored in SAFETY_ORDER:
        other = np.logical_and.reduce(
            [checks[name] for name in SAFETY_ORDER if name != restored]
        )
        restore_masks[f"restore_{restored}_only"] = support_stage & other
    counterfactuals = {
        name: evaluate_mask(name, data, labels, mask, challenger, baseline)
        for name, mask in {**cumulative_masks, **restore_masks}.items()
    }

    filter_values = {
        "physical": filter_value(
            "physical_hard", challenger, physical_stage, labels, challenger
        ),
        "support": filter_value(
            "spacing_support_q0.999", physical_stage, support_stage, labels, challenger
        ),
        "safety_aggregate": filter_value(
            "candidate_safety_aggregate", support_stage, safe_stage, labels, challenger
        ),
        "safety_components_standalone_on_physical_support": {
            name: filter_value(name, support_stage, support_stage & checks[name], labels, challenger)
            for name in SAFETY_ORDER
        },
    }
    overlap = overlap_summary(labels, challenger, fields["physical"], support_pass, checks)
    ranks = energy_rank(fields)
    geometry = geometry_fields(data, extra)
    source_key = np.asarray(SOURCES, dtype="U8")[fields["source_pool"]]
    route_key = np.broadcast_to(data["group"][:, None], challenger.shape)
    rank_bin = np.full(ranks.shape, "rank_9_plus", dtype="U16")
    rank_bin[(ranks >= 1) & (ranks <= 4)] = "rank_1_4"
    rank_bin[(ranks >= 5) & (ranks <= 8)] = "rank_5_8"
    rank_bin[ranks < 0] = "unobservable"
    stratification = {
        "by_route": group_diagnostics(route_key, labels, challenger, first, safe_stage),
        "by_source": group_diagnostics(source_key, labels, challenger, first, safe_stage),
        "by_energy_rank_bin": group_diagnostics(rank_bin, labels, challenger, first, safe_stage),
    }
    route_lost_counts = Counter(data["group"][state_cause != "no_raw_dual"].tolist())
    route_lost_only = Counter(
        data["group"][(state_cause == "support") | (state_cause == "safety")].tolist()
    )
    stratification["route_state_loss"] = {
        "opportunity_states_by_route": dict(sorted(route_lost_counts.items())),
        "lost_states_by_route": dict(sorted(route_lost_only.items())),
        "maximum_single_route_share_of_67_lost_states": (
            float(max(route_lost_only.values()) / sum(route_lost_only.values()))
            if route_lost_only else None
        ),
    }
    margin_analysis = margin_diagnostics(labels, challenger, first, margins, fields)
    seed_geometry = seed_and_geometry_diagnostics(
        labels, challenger, first, fields, extra, geometry
    )

    selected = load_npz(SELECTED)
    archive_energy = selected["archive_energy"].astype(np.float64)
    safe_dual = labels["raw_dual"] & safe_stage
    energy_delta = fields["energy"].astype(np.float64) - archive_energy[:, None]
    archive = {
        "finite_archive_states": int(np.isfinite(archive_energy).sum()),
        "archive_source_counts": read_json(FREEZE)["archive_source_counts"],
        "states_with_safe_true_dual_not_selected_by_energy_archive": int(safe_dual.any(axis=1).sum()),
        "safe_true_dual_candidates": int(safe_dual.sum()),
        "safe_true_dual_energy_minus_archive_energy": finite_summary(energy_delta[safe_dual]),
        "safe_true_dual_with_lower_energy_than_archive": int((safe_dual & (energy_delta < -1.0e-7)).sum()),
        "interpretation": (
            "archive is a frozen-energy selection stage, not an eligibility filter; "
            "safe true-dual opportunities can remain but lose to the formal action by scorer energy"
        ),
    }
    proposal_updates = {
        "fixed": {
            "updated_states_reconstructed_safe_count_ge_4": int((pools[0]["current_safe"].sum(axis=1) >= 4).sum())
        }
    }
    for iteration in range(3):
        score = load_npz(REPLAY / f"iteration_{iteration:02d}/frozen_scores.npz")
        proposal_updates[f"r{iteration + 1}"] = {
            "updated_states": int(score["proposal_updated"].sum()),
            "safe_count_ge_4_exact": bool(np.array_equal(score["proposal_updated"], score["safe_count"] >= 4)),
            "raw_dual_candidates_from_round": int(
                (labels["raw_dual"] & challenger & (fields["source_pool"] == iteration + 1)).sum()
            ),
            "safe_dual_candidates_from_round": int(
                (safe_dual & (fields["source_pool"] == iteration + 1)).sum()
            ),
        }
    proposal_updates["counterfactual_new_candidates"] = (
        "unobservable without rerunning the forbidden proposal/LTWM pipeline"
    )

    legacy_majority = extra["majority_support"]
    implementation_audit = {
        "cache_reconstruction": reconstruction_audit,
        "dedup": dedup_audit,
        "shapes": {
            "candidates": list(data["candidates"].shape),
            "true_metrics": list(data["true_metrics"].shape),
            "physical": list(fields["physical"].shape),
            "support": list(fields["support"].shape),
            "model_mean_std": [list(fields["mean"].shape), list(fields["std"].shape)],
            "ltwm_seed_fields": [list(extra["ate_by_seed"].shape), list(extra["rtp_by_seed"].shape)],
        },
        "thresholds_and_direction": {
            "physical": "boolean guard must be true",
            "support": f"score <= {SUPPORT_THRESHOLD}",
            "ate_ucb": "mean + 2*std < -0.01 m",
            "objective_ucb": "mean + 2*std < -0.01 m",
            "dual_probability": ">= 0.5",
            "harm_probability": "<= 0.1",
        },
        "units": {
            "candidate_feature_waypoint_units_to_m": 0.25,
            "scand_spacing_m": 0.38,
            "native_to_feature_waypoint_bridge": 0.38 / 0.25,
            "metric_output": "m",
        },
        "legacy_majority_support_used": False,
        "challenger_rows_that_would_fail_legacy_majority_support": int((challenger & ~legacy_majority).sum()),
        "current_valid_exact_all_pools": all(
            record["current_valid_exact"] for record in reconstruction_audit["pool_audit"]
        ),
        "current_safe_exact_all_pools": all(
            record["current_safe_exact"] for record in reconstruction_audit["pool_audit"]
        ),
        "safe_union_exact": bool(np.array_equal(safe_stage, fields["current_safe"] & challenger)),
        "route_leakage": "not applicable: no fitting, threshold selection, or truth-based ranking",
    }

    write_candidate_csv(
        data, labels, fields, extra, checks, margins, first, safety_first, ranks, geometry
    )
    write_state_csv(
        data, labels, challenger, physical_stage, support_stage, safe_stage,
        state_cause, state_safety_cause,
    )
    elapsed = time.perf_counter() - started
    immutable_after = verify_prereg()
    if immutable_after != immutable_before:
        raise RuntimeError("immutable inputs changed during filter attribution")
    primary_counts = waterfall["first_loss_counts"]
    primary = max(
        ((name, count) for name, count in primary_counts.items() if name != "retained_eligible"),
        key=lambda item: item[1],
    )
    result = {
        "schema": "scand_candidate_filter_attribution_result_v1",
        "status": "complete_wait_for_user_decision",
        "evidence_role": "spent_scand_offline_filter_attribution_only",
        "elapsed_sec": elapsed,
        "baseline_formal_v70": baseline,
        "waterfall": waterfall,
        "target_89_67_definition_resolution": target_resolution,
        "candidate_first_reject_counts": {
            label_name: dict(sorted(Counter(first[labels[label_name] & challenger].tolist()).items()))
            for label_name in ("raw_dual", "raw_harm", "strict_dual", "strict_harm")
        },
        "failure_overlap": overlap,
        "filter_value": filter_values,
        "counterfactual_oracles": counterfactuals,
        "archive_and_round_guard": {"archive": archive, "proposal_update_guard": proposal_updates},
        "stratification": stratification,
        "margin_and_label_boundary": margin_analysis,
        "seed_disagreement_and_geometry": seed_geometry,
        "implementation_audit": implementation_audit,
        "primary_bottleneck": {
            "state_first_loss_stage": primary[0],
            "lost_opportunity_states": int(primary[1]),
            "classification": (
                "reasonable_hard_constraint" if primary[0] == "physical" else
                "statistical_or_heuristic_filter_requires_independent_recalibration_evidence"
            ),
            "threshold_change_authorized": False,
        },
        "observability": {
            "upstream_sampler_rejected_draw_truth": "unobservable",
            "counterfactual_q_update_new_proposals": "unobservable",
            "seed_specific_safety_logits": "unobservable",
            "existing_65_candidate_truth_and_all_cached_masks": "observable",
        },
        "next_step_assessment": {
            "automatic_continuation": False,
            "decision_rule": (
                "recommend only a preregistered independent-cohort recalibration/feature test "
                "if a soft statistical filter is primary; never relax the hard physical guard"
            ),
        },
        "audit": {
            "immutable_before": immutable_before,
            "immutable_after": immutable_after,
            "reserve_accessed": False,
            "v76_trained": False,
            "dwm_ltwm_rerun": False,
            "candidate_generated": False,
            "v70_or_threshold_mutated": False,
            "truth_used_for_deployment_ranking": False,
            "automatic_continuation": False,
        },
    }
    write_json_atomic(RESULT, result)
    audit = {
        "schema": "scand_candidate_filter_attribution_audit_v1",
        "status": "complete_wait_for_user_decision",
        "preregistration_sha256": sha256_file(PREREG),
        "result_sha256": sha256_file(RESULT),
        "candidate_records_sha256": sha256_file(CANDIDATE_CSV),
        "state_waterfall_sha256": sha256_file(STATE_CSV),
        "runner_sha256": sha256_file(Path(__file__)),
        "immutable_inputs_unchanged": True,
        "reserve_accessed": False,
        "v76_trained": False,
        "dwm_ltwm_rerun": False,
        "formal_v70_mutated": False,
        "threshold_mutated": False,
        "route_leakage": "not applicable; no model fit or threshold tuning",
        "automatic_continuation": False,
    }
    write_json_atomic(AUDIT, audit)
    print(json.dumps({
        "elapsed_sec": elapsed,
        "waterfall": waterfall,
        "target_89_67_definition_resolution": target_resolution,
        "primary_bottleneck": result["primary_bottleneck"],
        "result_sha256": audit["result_sha256"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
