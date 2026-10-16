#!/usr/bin/env python3
"""Pure-offline SCAND filter-relaxation and randomness sensitivity diagnostic."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from tools.evaluation import run_v72_proposal_tail_reranker as v72  # noqa: E402
from tools.planning import run_dualdomain_cem_tail_v66_replay as v66  # noqa: E402
from tools.planning import run_dualdomain_cem_tail_v70_archive_replay as v70  # noqa: E402


OUTPUT = ROOT / "outputs/cem_development_v1/scand_filter_random_sensitivity_v1"
PREREG = OUTPUT / "preregistration.json"
RESULT = OUTPUT / "result.json"
AUDIT = OUTPUT / "audit.json"
PARETO = OUTPUT / "pareto.csv"
V70_ROOT = v72.V70_ROOT
REPLAY = V70_ROOT / "replays/scand_fresh_outer/proposal_aware_v70"
FREEZE = REPLAY / "prelabel_freeze.json"
SELECTED = REPLAY / "selected_actions.npz"
CURRENT_SUPPORT = 4.876645565032959
METRICS = ("ate", "rpe", "endpoint", "objective")
RAW_TOLERANCE = 1.0e-7
NEUTRAL_M = 0.001


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_prereg() -> dict[str, str]:
    prereg = read_json(PREREG)
    if prereg.get("status") != "frozen_before_variant_outcome_computation":
        raise RuntimeError("preregistration is not frozen")
    if not all(prereg["forbidden"].values()):
        raise RuntimeError("forbidden-action contract is incomplete")
    paths = {
        "formal_v70_result_sha256": V70_ROOT / "result.json",
        "scand_prelabel_freeze_sha256": FREEZE,
        "selected_actions_sha256": SELECTED,
        "base_features_sha256": ROOT / read_json(FREEZE)["base_features"],
        "spent_scand_outcomes_sha256": ROOT
        / "outputs/scand_domain_adaptation_v1/fresh_outer_v1/outcomes/fresh_outer_outcomes.npz",
    }
    observed = {key: sha256(path) for key, path in paths.items()}
    for key, digest in observed.items():
        if digest != prereg["immutable_inputs"][key]:
            raise RuntimeError(f"immutable input changed: {key}")
    return {"preregistration_sha256": sha256(PREREG), **observed}


def raw_energy(
    mean: np.ndarray,
    std: np.ndarray,
    p_dual: np.ndarray,
    p_harm: np.ndarray,
    rank_logit: np.ndarray,
) -> np.ndarray:
    ucb = mean.astype(np.float64) + 2.0 * std.astype(np.float64)
    rank = rank_logit.astype(np.float64)
    scale = np.maximum(rank[:, 1:].std(axis=1, keepdims=True), 1.0e-3)
    rank_z = (rank - rank[:, :1]) / scale
    value = (
        ucb[..., 0]
        + 0.25 * ucb[..., 1]
        + 0.25 * np.maximum(ucb[..., 2], 0.0)
        + 0.50 * ucb[..., 3]
        + 0.15 * p_harm
        - 0.05 * p_dual
        - 0.03 * rank_z
    ).astype(np.float32)
    value[:, 0] = np.inf
    return value


def safety_components(
    mean: np.ndarray,
    std: np.ndarray,
    p_dual: np.ndarray,
    p_harm: np.ndarray,
    risk_multiplier: float,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, float]]:
    r = float(risk_multiplier)
    policy = {
        "z": 2.0 / r,
        "margin_ate": 0.01 / r,
        "margin_objective": 0.01 / r,
        "dual_threshold": 0.5 / r,
        "harm_threshold": min(1.0, 0.1 * r),
    }
    ucb = mean.astype(np.float64) + policy["z"] * std.astype(np.float64)
    checks = {
        "ate_ucb": ucb[..., 0] < -policy["margin_ate"],
        "objective_ucb": ucb[..., 3] < -policy["margin_objective"],
        "dual_probability": p_dual >= policy["dual_threshold"],
        "harm_probability": p_harm <= policy["harm_threshold"],
    }
    mask = np.logical_and.reduce(tuple(checks.values()))
    return mask, checks, policy


def reconstruct_pools(device: str) -> tuple[list[dict[str, np.ndarray]], dict[str, Any]]:
    freeze = read_json(FREEZE)
    selected = v72.load_npz(SELECTED)
    support_path = Path(freeze["support_checkpoint"])
    envelope = v66.load_envelopes(support_path)["scand"]
    checkpoints, policy, dual_temperature, harm_temperature, _ = v70.frozen_model_contract(
        role="proposal_aware_v66",
        domain="scand",
        base_result_path=V70_ROOT / "result.json",
        v66_result_path=V70_ROOT / "result.json",
    )
    if policy != freeze["policy"]:
        raise RuntimeError("frozen policy mismatch")

    feature_paths = [ROOT / freeze["base_features"]] + [
        REPLAY / f"iteration_{iteration:02d}/cem_features.npz" for iteration in range(3)
    ]
    score_paths: list[Path | None] = [None] + [
        REPLAY / f"iteration_{iteration:02d}/frozen_scores.npz" for iteration in range(3)
    ]
    pools: list[dict[str, np.ndarray]] = []
    pool_audit = []
    for pool_id, (feature_path, score_path) in enumerate(zip(feature_paths, score_paths)):
        feature_source = v66.load_features(feature_path, labels_expected=False)
        feature, support, physical = v66.build_universal_features(
            feature_source, spacing_m=0.38, support_envelope=envelope
        )
        scorer_aux = np.asarray(feature_source["scorer_v2_aux_features"], dtype=np.float32)
        scorer_proxy = (0.4 * scorer_aux[..., 6] + 0.6 * scorer_aux[..., 8]).astype(np.float32)
        current_valid = physical & (support <= CURRENT_SUPPORT)
        current_valid[:, 0] = True
        if score_path is None:
            prediction = v70.compatible_ensemble_predict(
                checkpoints,
                feature,
                np.ones(len(feature), dtype=np.int64),
                device=device,
            )
            _, p_dual, p_harm = v66.conservative_energy(
                prediction,
                policy=policy,
                dual_temperature=dual_temperature,
                harm_temperature=harm_temperature,
                valid=np.ones_like(current_valid, dtype=bool),
            )
            mean = prediction["mean"].astype(np.float32)
            std = prediction["std"].astype(np.float32)
            rank_logit = prediction["rank_logit"].astype(np.float32)
            energy = raw_energy(mean, std, p_dual, p_harm, rank_logit)
            score_safe = selected["base_safe"].astype(bool)
            expected_valid = selected["base_valid"].astype(bool)
        else:
            score = v72.load_npz(score_path)
            mean = score["mean"].astype(np.float32)
            std = score["std"].astype(np.float32)
            p_dual = score["dual_probability"].astype(np.float32)
            p_harm = score["harm_probability"].astype(np.float32)
            rank_logit = score["rank_logit"].astype(np.float32)
            energy = raw_energy(mean, std, p_dual, p_harm, rank_logit)
            score_safe = score["safe"].astype(bool)
            expected_valid = score["valid"].astype(bool)
        core, _, _ = safety_components(mean, std, p_dual, p_harm, 1.0)
        reconstructed_safe = current_valid & core
        reconstructed_safe[:, 0] = False
        valid_equal = bool(np.array_equal(current_valid, expected_valid))
        safe_equal = bool(np.array_equal(reconstructed_safe, score_safe))
        if not valid_equal or not safe_equal:
            raise RuntimeError(f"pool {pool_id} current mask reconstruction mismatch")
        if pool_id == 0:
            masked = np.where(reconstructed_safe, energy, np.inf)
            reconstructed_selected = np.where(
                reconstructed_safe.any(axis=1), masked.argmin(axis=1), 0
            ).astype(np.int64)
            fixed_selected_equal = bool(
                np.array_equal(reconstructed_selected, selected["fixed_pool_selected_index"])
            )
            if not fixed_selected_equal:
                raise RuntimeError("fixed K17 frozen scorer reconstruction did not reproduce selection")
        else:
            fixed_selected_equal = None
        pools.append(
            {
                "candidate": feature_source["candidates"].astype(np.float32),
                "physical": physical.astype(bool),
                "support": support.astype(np.float32),
                "mean": mean,
                "std": std,
                "p_dual": p_dual,
                "p_harm": p_harm,
                "rank_logit": rank_logit,
                "energy": energy,
                "scorer_proxy": scorer_proxy,
                "current_valid": current_valid,
                "current_safe": reconstructed_safe,
            }
        )
        pool_audit.append(
            {
                "pool": "fixed" if pool_id == 0 else f"r{pool_id}",
                "feature_path": str(feature_path),
                "feature_sha256": sha256(feature_path),
                "score_path": str(score_path) if score_path is not None else None,
                "score_sha256": sha256(score_path) if score_path is not None else None,
                "current_valid_exact": valid_equal,
                "current_safe_exact": safe_equal,
                "fixed_selection_exact": fixed_selected_equal,
            }
        )
    return pools, {
        "continuous_support_available_all_pools": True,
        "physical_guard_available_all_pools": True,
        "continuous_safety_components_cached_cem_rounds": True,
        "continuous_safety_components_reconstructed_fixed_k17": True,
        "fixed_k17_reconstruction_used_cached_features_and_frozen_v70_checkpoint_only": True,
        "dwm_ltwm_rerun": False,
        "pool_audit": pool_audit,
    }


def unionize(
    pools: list[dict[str, np.ndarray]], data: dict[str, Any]
) -> dict[str, np.ndarray]:
    states, candidates = data["candidates"].shape[:2]
    fields = {
        "physical": np.zeros((states, candidates), dtype=bool),
        "support": np.full((states, candidates), np.nan, dtype=np.float32),
        "mean": np.full((states, candidates, 4), np.nan, dtype=np.float32),
        "std": np.full((states, candidates, 4), np.nan, dtype=np.float32),
        "p_dual": np.full((states, candidates), np.nan, dtype=np.float32),
        "p_harm": np.full((states, candidates), np.nan, dtype=np.float32),
        "energy": np.full((states, candidates), np.nan, dtype=np.float32),
        "scorer_proxy": np.full((states, candidates), np.nan, dtype=np.float32),
        "current_valid": np.zeros((states, candidates), dtype=bool),
        "current_safe": np.zeros((states, candidates), dtype=bool),
        "source_pool": np.full((states, candidates), -1, dtype=np.int64),
        "source_candidate": np.full((states, candidates), -1, dtype=np.int64),
    }
    for state in range(states):
        seen: dict[bytes, int] = {}
        cursor = 0
        for pool_id, pool in enumerate(pools):
            for candidate in range(pool["candidate"].shape[1]):
                key = np.ascontiguousarray(pool["candidate"][state, candidate]).tobytes()
                if key in seen:
                    continue
                seen[key] = cursor
                if not np.array_equal(
                    pool["candidate"][state, candidate], data["candidates"][state, cursor]
                ):
                    raise RuntimeError("union candidate ordering differs from frozen v72 union")
                for name in (
                    "physical",
                    "support",
                    "mean",
                    "std",
                    "p_dual",
                    "p_harm",
                    "energy",
                    "scorer_proxy",
                    "current_valid",
                    "current_safe",
                ):
                    fields[name][state, cursor] = pool[name][state, candidate]
                fields["source_pool"][state, cursor] = pool_id
                fields["source_candidate"][state, cursor] = candidate
                cursor += 1
        if cursor != candidates:
            raise RuntimeError(f"state {state} union count {cursor} != {candidates}")
    if not np.isfinite(fields["support"]).all():
        raise RuntimeError("non-finite reconstructed support")
    if not np.array_equal(fields["current_valid"], data["valid"]):
        raise RuntimeError("union current valid mismatch")
    if not np.array_equal(fields["current_safe"], data["safe"]):
        raise RuntimeError("union current safe mismatch")
    return fields


def make_eligibility(
    fields: dict[str, np.ndarray],
    formal: np.ndarray,
    support_multiplier: float,
    safety_multiplier: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    safety, checks, policy = safety_components(
        fields["mean"],
        fields["std"],
        fields["p_dual"],
        fields["p_harm"],
        safety_multiplier,
    )
    support = fields["support"] <= CURRENT_SUPPORT * support_multiplier
    eligible = fields["physical"] & support & safety
    eligible[np.arange(len(eligible)), formal] = False
    return eligible, {
        "support_multiplier": support_multiplier,
        "support_threshold": CURRENT_SUPPORT * support_multiplier,
        "safety_risk_multiplier": safety_multiplier,
        "safety_policy": policy,
        "reason_pass_counts": {
            "physical_guard": int(fields["physical"].sum()),
            "support": int(support.sum()),
            **{name: int(value.sum()) for name, value in checks.items()},
        },
    }


def select_ranked(
    rank: np.ndarray, eligible: np.ndarray, formal: np.ndarray
) -> np.ndarray:
    masked = np.where(eligible & np.isfinite(rank), rank, np.inf)
    chosen = masked.argmin(axis=1).astype(np.int64)
    return np.where(np.isfinite(masked).any(axis=1), chosen, formal).astype(np.int64)


def select_incremental(
    rank: np.ndarray,
    eligible: np.ndarray,
    current_eligible: np.ndarray,
    formal: np.ndarray,
    formal_rank: np.ndarray,
) -> np.ndarray:
    """Keep formal v70 unless a newly admitted candidate has a better frozen rank."""
    newly_admitted = eligible & ~current_eligible
    masked = np.where(newly_admitted & np.isfinite(rank), rank, np.inf)
    chosen = masked.argmin(axis=1).astype(np.int64)
    row = np.arange(len(formal))
    replace = np.isfinite(masked[row, chosen]) & (masked[row, chosen] < formal_rank)
    return np.where(replace, chosen, formal).astype(np.int64)


def label_summary(data: dict[str, Any], selected: np.ndarray) -> dict[str, Any]:
    row = np.arange(len(selected))
    formal = data["formal_index"]
    replacement = selected != formal
    gain = data["true_metrics"][row, formal] - data["true_metrics"][row, selected]
    raw_dual = replacement & (gain[:, 0] > RAW_TOLERANCE) & (gain[:, 3] > RAW_TOLERANCE)
    raw_harm = replacement & ((gain[:, 0] < -RAW_TOLERANCE) | (gain[:, 3] < -RAW_TOLERANCE))
    neutral_dual = replacement & (gain[:, 0] > NEUTRAL_M) & (gain[:, 3] > NEUTRAL_M)
    neutral_harm = replacement & ((gain[:, 0] < -NEUTRAL_M) | (gain[:, 3] < -NEUTRAL_M))
    count = int(replacement.sum())
    return {
        "replacements": count,
        "raw_zero_mm": {
            "dual_count": int(raw_dual.sum()),
            "harm_count": int(raw_harm.sum()),
            "dual_rate": float(raw_dual.sum() / count) if count else None,
            "harm_rate": float(raw_harm.sum() / count) if count else None,
        },
        "neutral_band_1mm": {
            "strict_dual_count": int(neutral_dual.sum()),
            "strict_harm_count": int(neutral_harm.sum()),
            "strict_dual_rate_all_replacements": float(neutral_dual.sum() / count) if count else None,
            "strict_harm_rate_all_replacements": float(neutral_harm.sum() / count) if count else None,
            "ambiguous_count": int((replacement & ~neutral_dual & ~neutral_harm).sum()),
            "ambiguous_rate": float((replacement & ~neutral_dual & ~neutral_harm).sum() / count) if count else None,
        },
    }


def route_summary(
    data: dict[str, Any], selected: np.ndarray, eligible: np.ndarray
) -> dict[str, Any]:
    group = data["group"].astype(str)
    row = np.arange(len(selected))
    formal = data["formal_index"]
    base_objective = data["true_metrics"][row, formal, 3]
    selected_objective = data["true_metrics"][row, selected, 3]
    records = {}
    contributions = []
    for route in sorted(np.unique(group)):
        local = np.flatnonzero(group == route)
        delta = float(selected_objective[local].mean() - base_objective[local].mean())
        improvement_sum = float(base_objective[local].sum() - selected_objective[local].sum())
        local_labels = label_summary(
            {**data, "formal_index": formal[local], "true_metrics": data["true_metrics"][local]},
            selected[local],
        )
        records[route] = {
            "states": int(len(local)),
            "eligible_states": int(eligible[local].any(axis=1).sum()),
            "eligible_candidate_rows": int(eligible[local].sum()),
            "objective_delta_selected_minus_v70_m": delta,
            "objective_improvement_sum_m": improvement_sum,
            "replacements": local_labels["replacements"],
            "raw_dual_rate": local_labels["raw_zero_mm"]["dual_rate"],
            "raw_harm_rate": local_labels["raw_zero_mm"]["harm_rate"],
        }
        contributions.append(max(improvement_sum, 0.0))
    contribution = np.asarray(contributions, dtype=np.float64)
    total = float(contribution.sum())
    deltas = np.asarray(
        [value["objective_delta_selected_minus_v70_m"] for value in records.values()]
    )
    return {
        "records": records,
        "improved_routes": int((deltas < -RAW_TOLERANCE).sum()),
        "worsened_routes": int((deltas > RAW_TOLERANCE).sum()),
        "unchanged_routes": int((np.abs(deltas) <= RAW_TOLERANCE).sum()),
        "worst_route_objective_delta_m": float(deltas.max()),
        "best_route_objective_delta_m": float(deltas.min()),
        "maximum_single_route_share_of_total_positive_gain": float(contribution.max() / total) if total > 0 else None,
    }


def metric_summary(data: dict[str, Any], selected: np.ndarray) -> dict[str, Any]:
    value = data["true_metrics"][np.arange(len(selected)), selected]
    return {
        "mean_m": {name: float(value[:, index].mean()) for index, name in enumerate(METRICS)},
        "p90_m": {name: float(np.quantile(value[:, index], 0.9)) for index, name in enumerate(METRICS)},
    }


def oracle_selection(data: dict[str, Any], eligible: np.ndarray) -> np.ndarray:
    formal = data["formal_index"].copy()
    objective = data["true_metrics"][..., 3]
    for state in range(len(formal)):
        available = np.flatnonzero(eligible[state])
        if len(available):
            candidate = available[np.argmin(objective[state, available])]
            if objective[state, candidate] < objective[state, formal[state]] - RAW_TOLERANCE:
                formal[state] = candidate
    return formal


def evaluate_variant(
    name: str,
    ranker: str,
    rank: np.ndarray,
    data: dict[str, Any],
    eligible: np.ndarray,
    current_eligible: np.ndarray,
    formal_rank: np.ndarray,
    filter_audit: dict[str, Any],
    baseline: dict[str, Any],
) -> dict[str, Any]:
    selected = select_incremental(
        rank,
        eligible,
        current_eligible,
        data["formal_index"],
        formal_rank,
    )
    oracle = oracle_selection(data, eligible)
    metrics = metric_summary(data, selected)
    labels = label_summary(data, selected)
    routes = route_summary(data, selected, eligible)
    dual_opportunity = eligible & data["dual_vs_formal"].astype(bool)
    result = {
        "variant": name,
        "ranker": ranker,
        "filter": filter_audit,
        "eligibility": {
            "states": int(len(eligible)),
            "states_with_eligible_challenger": int(eligible.any(axis=1).sum()),
            "eligible_candidate_rows": int(eligible.sum()),
            "candidate_coverage_all_nonformal": float(eligible.sum() / (len(eligible) * (eligible.shape[1] - 1))),
            "states_with_raw_dual_opportunity": int(dual_opportunity.any(axis=1).sum()),
        },
        "oracle": {
            "objective_improvement_vs_v70_m": float(
                baseline["mean_m"]["objective"]
                - metric_summary(data, oracle)["mean_m"]["objective"]
            ),
            "selected_metrics": metric_summary(data, oracle),
            "labels": label_summary(data, oracle),
        },
        "selected_metrics": metrics,
        "objective_improvement_vs_v70_m": float(
            baseline["mean_m"]["objective"] - metrics["mean_m"]["objective"]
        ),
        "objective_p90_delta_vs_v70_m": float(
            metrics["p90_m"]["objective"] - baseline["p90_m"]["objective"]
        ),
        "labels": labels,
        "route": routes,
        "selected_index": selected,
    }
    return result


def quantiles(value: np.ndarray) -> dict[str, float]:
    array = np.asarray(value, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "p10": float(np.quantile(array, 0.1)),
        "p50": float(np.quantile(array, 0.5)),
        "p90": float(np.quantile(array, 0.9)),
    }


def monte_carlo(
    name: str,
    data: dict[str, Any],
    eligible: np.ndarray,
    energy: np.ndarray,
    baseline: dict[str, Any],
    rng: np.random.Generator,
) -> dict[str, Any]:
    formal = data["formal_index"]
    row = np.arange(len(formal))
    ordered = []
    for state in range(len(formal)):
        available = np.flatnonzero(eligible[state] & np.isfinite(energy[state]))
        ordered.append(available[np.argsort(energy[state, available], kind="stable")])
    deterministic = select_ranked(energy, eligible, formal)
    result = {
        "candidate_set": name,
        "eligible_states": int(eligible.any(axis=1).sum()),
        "eligible_candidate_rows": int(eligible.sum()),
        "deterministic_top1": {
            "objective_improvement_vs_v70_m": float(
                baseline["mean_m"]["objective"]
                - metric_summary(data, deterministic)["mean_m"]["objective"]
            ),
            "labels": label_summary(data, deterministic),
            "route": route_summary(data, deterministic, eligible),
        },
        "uniform_top_k": {},
    }
    group = data["group"].astype(str)
    routes = sorted(np.unique(group))
    baseline_objective = data["true_metrics"][row, formal, 3]
    for k in (2, 4, 8):
        objective_gain = []
        raw_harm = []
        neutral_harm = []
        worst_route = []
        worst_route_name = []
        replacements = []
        for _ in range(1000):
            selected = formal.copy()
            for state, candidates in enumerate(ordered):
                width = min(k, len(candidates))
                if width:
                    selected[state] = candidates[int(rng.integers(width))]
            metric = metric_summary(data, selected)
            labels = label_summary(data, selected)
            objective_gain.append(
                baseline["mean_m"]["objective"] - metric["mean_m"]["objective"]
            )
            raw_harm.append(labels["raw_zero_mm"]["harm_rate"] or 0.0)
            neutral_harm.append(labels["neutral_band_1mm"]["strict_harm_rate_all_replacements"] or 0.0)
            replacements.append(labels["replacements"])
            selected_objective = data["true_metrics"][row, selected, 3]
            route_delta = np.asarray(
                [
                    selected_objective[group == route].mean()
                    - baseline_objective[group == route].mean()
                    for route in routes
                ]
            )
            worst = int(np.argmax(route_delta))
            worst_route.append(float(route_delta[worst]))
            worst_route_name.append(routes[worst])
        counts = {
            route: int(sum(value == route for value in worst_route_name)) for route in routes
        }
        result["uniform_top_k"][str(k)] = {
            "trials": 1000,
            "seed": 20260722,
            "objective_improvement_vs_v70_m": quantiles(np.asarray(objective_gain)),
            "raw_harm_rate": quantiles(np.asarray(raw_harm)),
            "neutral_band_1mm_strict_harm_rate": quantiles(np.asarray(neutral_harm)),
            "worst_route_objective_delta_m": quantiles(np.asarray(worst_route)),
            "replacement_count": quantiles(np.asarray(replacements)),
            "most_frequent_worst_route": max(counts, key=counts.get),
            "most_frequent_worst_route_trials": int(max(counts.values())),
        }
    return result


def serializable_variant(value: dict[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if key != "selected_index"}


def decision(variants: list[dict[str, Any]], baseline: dict[str, Any]) -> dict[str, Any]:
    current = next(value for value in variants if value["variant"] == "current" and value["ranker"] == "v70_energy")
    support = [
        value
        for value in variants
        if value["variant"].startswith("support_only_") and value["ranker"] == "v70_energy"
    ]
    checks = []
    for value in support:
        labels = value["labels"]["raw_zero_mm"]
        route = value["route"]
        passed = (
            value["objective_improvement_vs_v70_m"] >= 0.001
            and labels["dual_rate"] is not None
            and labels["dual_rate"] >= 0.85
            and labels["harm_rate"] <= 0.15
            and value["selected_metrics"]["p90_m"]["objective"]
            <= baseline["p90_m"]["objective"] + 0.001
            and route["improved_routes"] >= 5
            and route["maximum_single_route_share_of_total_positive_gain"] is not None
            and route["maximum_single_route_share_of_total_positive_gain"] <= 0.40
            and value["labels"]["replacements"] > 0
        )
        checks.append({"variant": value["variant"], "promising": bool(passed)})
    oracle_increase = max(
        value["oracle"]["objective_improvement_vs_v70_m"]
        - current["oracle"]["objective_improvement_vs_v70_m"]
        for value in support
    )
    if any(value["promising"] for value in checks):
        verdict = "promising"
    elif oracle_increase >= 0.001:
        verdict = "diagnostic_only"
    else:
        verdict = "reject"
    return {
        "verdict": verdict,
        "support_only_checks": checks,
        "maximum_support_only_oracle_gain_increase_vs_current_m": float(oracle_increase),
        "neutral_band_used_for_success": False,
        "automatic_continuation": False,
    }


def write_pareto(variants: list[dict[str, Any]]) -> None:
    columns = [
        "variant",
        "ranker",
        "eligible_states",
        "eligible_candidate_rows",
        "dual_opportunity_states",
        "oracle_objective_gain_m",
        "selected_objective_gain_m",
        "selected_objective_p90_delta_m",
        "replacements",
        "raw_dual_rate",
        "raw_harm_rate",
        "neutral_ambiguous_rate",
        "improved_routes",
        "worsened_routes",
        "worst_route_objective_delta_m",
        "max_single_route_positive_gain_share",
    ]
    with PARETO.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for value in variants:
            writer.writerow(
                {
                    "variant": value["variant"],
                    "ranker": value["ranker"],
                    "eligible_states": value["eligibility"]["states_with_eligible_challenger"],
                    "eligible_candidate_rows": value["eligibility"]["eligible_candidate_rows"],
                    "dual_opportunity_states": value["eligibility"]["states_with_raw_dual_opportunity"],
                    "oracle_objective_gain_m": value["oracle"]["objective_improvement_vs_v70_m"],
                    "selected_objective_gain_m": value["objective_improvement_vs_v70_m"],
                    "selected_objective_p90_delta_m": value["objective_p90_delta_vs_v70_m"],
                    "replacements": value["labels"]["replacements"],
                    "raw_dual_rate": value["labels"]["raw_zero_mm"]["dual_rate"],
                    "raw_harm_rate": value["labels"]["raw_zero_mm"]["harm_rate"],
                    "neutral_ambiguous_rate": value["labels"]["neutral_band_1mm"]["ambiguous_rate"],
                    "improved_routes": value["route"]["improved_routes"],
                    "worsened_routes": value["route"]["worsened_routes"],
                    "worst_route_objective_delta_m": value["route"]["worst_route_objective_delta_m"],
                    "max_single_route_positive_gain_share": value["route"]["maximum_single_route_share_of_total_positive_gain"],
                }
            )


def main() -> None:
    if RESULT.exists() or AUDIT.exists() or PARETO.exists():
        raise FileExistsError("diagnostic output already exists")
    if not torch.cuda.is_available():
        raise RuntimeError("frozen v70 fixed-pool reconstruction requires the existing CUDA environment")
    immutable_before = verify_prereg()
    started = time.perf_counter()
    scand_spec = next(value for value in v72.COHORTS if value["name"] == "scand_fresh_outer")
    data = v72.load_cohort(scand_spec)
    pools, reconstruction_audit = reconstruct_pools("cuda")
    fields = unionize(pools, data)
    formal = data["formal_index"]
    baseline = metric_summary(data, formal)

    variant_specs = [("current", 1.0, 1.0)]
    variant_specs.extend(
        (f"support_only_{multiplier:.2f}x", multiplier, 1.0)
        for multiplier in (1.05, 1.10, 1.20, 1.50)
    )
    variant_specs.extend(
        (f"safety_only_{multiplier:.2f}x", 1.0, multiplier)
        for multiplier in (1.10, 1.25, 1.50)
    )
    variant_specs.append(("combined_support1.10x_safety1.10x", 1.10, 1.10))
    variants = []
    eligibilities = {}
    filter_audits = {}
    for name, support_multiplier, safety_multiplier in variant_specs:
        eligible, filter_audit = make_eligibility(
            fields, formal, support_multiplier, safety_multiplier
        )
        eligibilities[name] = eligible
        filter_audits[name] = filter_audit
    current_eligible = eligibilities["current"]
    selected_cache = v72.load_npz(SELECTED)
    formal_ranks = {
        "v70_energy": selected_cache["archive_energy"].astype(np.float32),
        "scorer_v2_proxy": fields["scorer_proxy"][np.arange(len(formal)), formal],
    }
    for name, _, _ in variant_specs:
        eligible = eligibilities[name]
        filter_audit = filter_audits[name]
        for ranker, rank in (
            ("v70_energy", fields["energy"]),
            ("scorer_v2_proxy", fields["scorer_proxy"]),
        ):
            variants.append(
                evaluate_variant(
                    name,
                    ranker,
                    rank,
                    data,
                    eligible,
                    current_eligible,
                    formal_ranks[ranker],
                    filter_audit,
                    baseline,
                )
            )

    physical_only = fields["physical"].copy()
    physical_only[np.arange(len(physical_only)), formal] = False
    rng = np.random.default_rng(20260722)
    randomness = {
        "current_eligible": monte_carlo(
            "current_eligible", data, current_eligible, fields["energy"], baseline, rng
        ),
        "physical_guard_only": monte_carlo(
            "physical_guard_only", data, physical_only, fields["energy"], baseline, rng
        ),
    }

    nonformal = np.ones_like(fields["physical"], dtype=bool)
    nonformal[np.arange(len(nonformal)), formal] = False
    current_safety, current_checks, _ = safety_components(
        fields["mean"], fields["std"], fields["p_dual"], fields["p_harm"], 1.0
    )
    support_pass = fields["support"] <= CURRENT_SUPPORT
    filter_attribution = {
        "nonformal_candidate_rows": int(nonformal.sum()),
        "physical_fail_rows": int((nonformal & ~fields["physical"]).sum()),
        "physical_pass_rows": int((nonformal & fields["physical"]).sum()),
        "support_fail_after_physical_rows": int(
            (nonformal & fields["physical"] & ~support_pass).sum()
        ),
        "support_pass_after_physical_rows": int(
            (nonformal & fields["physical"] & support_pass).sum()
        ),
        "safety_fail_after_physical_support_rows": int(
            (nonformal & fields["physical"] & support_pass & ~current_safety).sum()
        ),
        "current_eligible_rows": int(current_eligible.sum()),
        "overlapping_model_safety_failure_rows_after_physical_support": {
            name: int((nonformal & fields["physical"] & support_pass & ~value).sum())
            for name, value in current_checks.items()
        },
    }
    current_selected = select_incremental(
        fields["energy"],
        current_eligible,
        current_eligible,
        formal,
        formal_ranks["v70_energy"],
    )
    current_matches_formal_v70 = bool(np.array_equal(current_selected, formal))

    write_pareto(variants)
    verdict = decision(variants, baseline)
    immutable_after = verify_prereg()
    if immutable_after != immutable_before:
        raise RuntimeError("immutable inputs changed during diagnostic")
    result = {
        "schema": "scand_filter_random_sensitivity_result_v1",
        "status": "complete_wait_for_user_decision",
        "evidence_role": "spent_scand_offline_diagnostic_only",
        "baseline_formal_v70": baseline,
        "cache_reconstruction_audit": reconstruction_audit,
        "union_audit": {
            "states": int(len(data["record_key"])),
            "candidates_per_state": int(data["candidates"].shape[1]),
            "current_energy_selection_matches_formal_v70_all_states": current_matches_formal_v70,
        },
        "filter_attribution": filter_attribution,
        "variants": [serializable_variant(value) for value in variants],
        "randomness": randomness,
        "decision": verdict,
        "elapsed_sec": time.perf_counter() - started,
        "audit": {
            "immutable_before": immutable_before,
            "immutable_after": immutable_after,
            "reserve_accessed": False,
            "new_data_ingested": False,
            "v76_trained": False,
            "dwm_ltwm_rerun": False,
            "candidate_regenerated": False,
            "formal_v70_mutated": False,
            "cem_archive_mutated": False,
            "deployment_threshold_mutated": False,
            "truth_used_for_ranking": False,
            "route_leakage": "not applicable; no fit and no truth-based ranking",
            "automatic_continuation": False,
        },
    }
    write_json(RESULT, result)
    audit = {
        "schema": "scand_filter_random_sensitivity_audit_v1",
        "status": result["status"],
        "verdict": verdict["verdict"],
        "preregistration_sha256": immutable_before["preregistration_sha256"],
        "result_sha256": sha256(RESULT),
        "pareto_sha256": sha256(PARETO),
        "runner_sha256": sha256(Path(__file__)),
        "immutable_inputs_unchanged": immutable_before == immutable_after,
        **result["audit"],
    }
    write_json(AUDIT, audit)
    print(
        json.dumps(
            {
                "stage": "complete",
                "verdict": verdict,
                "filter_attribution": filter_attribution,
                "elapsed_sec": result["elapsed_sec"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
