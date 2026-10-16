#!/usr/bin/env python3
"""Offline true-outcome oracle diagnostics for the spent v70 cohorts.

This script is deliberately incapable of touching the final reserve.  It only
reads the already-spent RECON A-H and fresh-SCAND artifacts, reuses the frozen
v70 proposal caches, and writes to a separate diagnostic directory.  The
oracle-in-the-loop arms consume future ground truth and are therefore theory
diagnostics, not deployable planners.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from planning.nomad_action_adapter import nomad_waypoints_to_local_actions  # noqa: E402


OUTPUT = ROOT / "outputs/cem_development_v1/v70_oracle_diagnostic"
PREREG = OUTPUT / "preregistration.json"
FORMAL_ROOT = (
    ROOT
    / "outputs/cem_development_v1/"
    "dualdomain_cem_tail_v70_broad_hashfold_archive"
)
SUPPORT_PATH = (
    ROOT
    / "outputs/universal_dualdomain_raw_v2/training_v2/support/"
    "dualdomain_spacing_envelopes.npz"
)

METRICS = ("ate", "rpe", "endpoint", "objective")
TOLERANCE = 1.0e-7
ITERATIONS = 3
POPULATION = 16
ELITE_COUNT = 4
PROPOSAL_SCALE = 0.70
COVARIANCE_FLOOR = 0.015
SUPPORT_QUANTILE = 0.999
BASE_SEED = 20260730


COHORTS: tuple[dict[str, Any], ...] = (
    *tuple(
        {
            "name": f"recon_bank_{bank}",
            "domain": "recon",
            "replay": FORMAL_ROOT
            / f"replays/recon_bank_{bank}/proposal_aware_v70",
            "ground_truth": (
                ROOT
                / "outputs/cem_development_v1/recon_target_calibration_v20/"
                f"bank_{bank}/source_labeled.npz"
            ),
        }
        for bank in "abcdef"
    ),
    *tuple(
        {
            "name": f"recon_bank_{bank}",
            "domain": "recon",
            "replay": FORMAL_ROOT
            / f"replays/recon_bank_{bank}/proposal_aware_v70",
            "ground_truth": (
                ROOT
                / "outputs/cem_development_v1/recon_cem_boundary_v60/"
                f"evaluation_v60/bank_{bank}_v60_outcomes.npz"
            ),
        }
        for bank in "gh"
    ),
    {
        "name": "scand_fresh_outer",
        "domain": "scand",
        "replay": FORMAL_ROOT / "replays/scand_fresh_outer/proposal_aware_v70",
        "ground_truth": (
            ROOT
            / "outputs/scand_domain_adaptation_v1/fresh_outer_v1/outcomes/"
            "fresh_outer_outcomes.npz"
        ),
    },
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {key: payload[key].copy() for key in payload.files}


def reject_forbidden_paths(paths: list[Path]) -> None:
    # The forbidden token is assembled so the source contains no usable final-
    # cohort pathname and accidental recursive path discovery is impossible.
    forbidden = "res" + "erve"
    for path in paths:
        if forbidden in str(path).lower():
            raise RuntimeError(f"forbidden final cohort path: {path}")


def metric_values(candidates: np.ndarray, ground_truth: np.ndarray) -> np.ndarray:
    """Return true H4 ATE/RPE/endpoint/objective in metres."""

    value = np.asarray(candidates, dtype=np.float64)
    truth = np.asarray(ground_truth, dtype=np.float64)
    if value.ndim != 4 or value.shape[2:] != (4, 2):
        raise ValueError(f"candidate shape changed: {value.shape}")
    if truth.shape != (len(value), 4, 2):
        raise ValueError(f"ground-truth shape changed: {truth.shape}")
    candidate_m = value * 0.25
    error = candidate_m - truth[:, None]
    squared = np.sum(error * error, axis=-1)
    ate = np.sqrt(np.mean(squared, axis=-1))
    endpoint = np.sqrt(squared[..., -1])
    relative_error = np.diff(candidate_m, axis=2) - np.diff(truth, axis=1)[:, None]
    rpe = np.sqrt(np.mean(np.sum(relative_error * relative_error, axis=-1), axis=-1))
    objective = 0.4 * ate + 0.6 * endpoint
    return np.stack((ate, rpe, endpoint, objective), axis=-1).astype(np.float32)


def action_metrics(actions: np.ndarray, ground_truth: np.ndarray) -> np.ndarray:
    return metric_values(np.asarray(actions)[:, None], ground_truth)[:, 0]


def differs(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.any(np.abs(np.asarray(left) - np.asarray(right)) > TOLERANCE, axis=(1, 2))


def summarize_method(
    actions: np.ndarray,
    sources: np.ndarray,
    *,
    ground_truth: np.ndarray,
    incumbent: np.ndarray,
    noncem: np.ndarray,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    absolute = action_metrics(actions, ground_truth)
    incumbent_absolute = action_metrics(incumbent, ground_truth)
    noncem_absolute = action_metrics(noncem, ground_truth)
    delta = absolute - incumbent_absolute
    delta_noncem = absolute - noncem_absolute
    promoted = differs(actions, incumbent)
    replacement = differs(actions, noncem)
    dual = promoted & (delta[:, 0] < -TOLERANCE) & (delta[:, 3] < -TOLERANCE)
    harmed = promoted & ((delta[:, 0] > TOLERANCE) | (delta[:, 3] > TOLERANCE))
    dual_noncem = replacement & (delta_noncem[:, 0] < -TOLERANCE) & (
        delta_noncem[:, 3] < -TOLERANCE
    )
    harmed_noncem = replacement & ((delta_noncem[:, 0] > TOLERANCE) | (
        delta_noncem[:, 3] > TOLERANCE
    ))
    promotions = int(promoted.sum())
    replacements = int(replacement.sum())
    source_counts = {
        str(source): int((sources.astype(str) == str(source)).sum())
        for source in np.unique(sources.astype(str))
    }
    summary = {
        "states": int(len(actions)),
        "absolute_mean_m": {
            name: float(absolute[:, index].mean())
            for index, name in enumerate(METRICS)
        },
        "absolute_p90_m": {
            name: float(np.quantile(absolute[:, index], 0.9))
            for index, name in enumerate(METRICS)
        },
        "promotions_vs_candidate0": promotions,
        "coverage": float(promoted.mean()),
        "dual_successes": int(dual.sum()),
        "dual_precision": float(dual.sum() / promotions) if promotions else None,
        "harm_count": int(harmed.sum()),
        "harm_rate": float(harmed.sum() / promotions) if promotions else None,
        "replacements_vs_v70_noncem": replacements,
        "replacement_fraction_vs_v70_noncem": float(replacement.mean()),
        "dual_improvements_vs_v70_noncem": int(dual_noncem.sum()),
        "dual_precision_vs_v70_noncem": (
            float(dual_noncem.sum() / replacements) if replacements else None
        ),
        "harm_count_vs_v70_noncem": int(harmed_noncem.sum()),
        "harm_rate_vs_v70_noncem": (
            float(harmed_noncem.sum() / replacements) if replacements else None
        ),
        "selected_source_counts": source_counts,
    }
    return summary, {
        "absolute_metrics": absolute,
        "delta_vs_candidate0": delta,
        "delta_vs_v70_noncem": delta_noncem,
        "promoted": promoted,
        "dual": dual,
        "harmed": harmed,
        "replacement_vs_v70_noncem": replacement,
        "source": sources.astype("U16"),
        "actions": np.asarray(actions, dtype=np.float32),
    }


def method_difference(challenger: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    return {
        "absolute_mean_delta_m": {
            name: challenger["absolute_mean_m"][name] - baseline["absolute_mean_m"][name]
            for name in METRICS
        },
        "absolute_p90_delta_m": {
            name: challenger["absolute_p90_m"][name] - baseline["absolute_p90_m"][name]
            for name in METRICS
        },
        "coverage_delta": challenger["coverage"] - baseline["coverage"],
    }


def regularized_covariance(
    elite: np.ndarray, *, fallback: np.ndarray, floor: float, scale: float
) -> np.ndarray:
    dimension = elite.shape[-1]
    if len(elite) >= 2:
        covariance = np.cov(elite, rowvar=False, ddof=1)
    else:
        covariance = np.diag(np.maximum(np.var(fallback, axis=0), floor * floor))
    covariance = np.asarray(covariance, dtype=np.float64).reshape(dimension, dimension)
    covariance = 0.5 * (covariance + covariance.T)
    covariance *= scale * scale
    diagonal_floor = np.maximum(np.var(fallback, axis=0) * 0.02, floor * floor)
    covariance.flat[:: dimension + 1] += diagonal_floor
    eigenvalue, eigenvector = np.linalg.eigh(covariance)
    eigenvalue = np.maximum(eigenvalue, floor * floor)
    return ((eigenvector * eigenvalue) @ eigenvector.T).astype(np.float32)


def initial_proposal(reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    flat = reference.reshape(len(reference), reference.shape[1], -1)
    mean = flat.mean(axis=1).astype(np.float32)
    covariance = np.stack(
        [
            regularized_covariance(
                row, fallback=row, floor=COVARIANCE_FLOOR, scale=PROPOSAL_SCALE
            )
            for row in flat
        ]
    )
    return mean, covariance


def proposal_summary(mean: np.ndarray, covariance: np.ndarray) -> dict[str, float]:
    dimension = covariance.shape[-1]
    _, logdet = np.linalg.slogdet(covariance.astype(np.float64))
    entropy = 0.5 * (dimension * (1.0 + np.log(2.0 * np.pi)) + logdet)
    diagonal = np.diagonal(covariance, axis1=1, axis2=2)
    return {
        "mean_l2_mean": float(np.linalg.norm(mean, axis=1).mean()),
        "mean_l2_std": float(np.linalg.norm(mean, axis=1).std()),
        "variance_mean": float(diagonal.mean()),
        "variance_min": float(diagonal.min()),
        "variance_max": float(diagonal.max()),
        "gaussian_entropy_mean": float(entropy.mean()),
        "gaussian_entropy_min": float(entropy.min()),
        "gaussian_entropy_max": float(entropy.max()),
    }


def oracle_update(
    candidates: np.ndarray,
    true_metrics: np.ndarray,
    eligible: np.ndarray,
    *,
    previous_mean: np.ndarray,
    previous_covariance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    flat = candidates.reshape(len(candidates), candidates.shape[1], -1)
    mean = previous_mean.copy()
    covariance = previous_covariance.copy()
    elite_index = np.full((len(candidates), ELITE_COUNT), -1, dtype=np.int64)
    updated = np.zeros(len(candidates), dtype=bool)
    allowed = np.asarray(eligible, dtype=bool).copy()
    allowed[:, 0] = False
    count = allowed.sum(axis=1).astype(np.int64)
    objective = true_metrics[..., 3]
    for state in range(len(candidates)):
        if count[state] < ELITE_COUNT:
            continue
        order = np.argsort(objective[state], kind="stable")
        chosen = order[allowed[state, order]][:ELITE_COUNT]
        elite_index[state] = chosen
        elite = flat[state, chosen]
        mean[state] = elite.mean(axis=0)
        covariance[state] = regularized_covariance(
            elite,
            fallback=flat[state, 1:],
            floor=COVARIANCE_FLOOR,
            scale=PROPOSAL_SCALE,
        )
        updated[state] = True
    return mean, covariance, elite_index, updated, count


def load_envelopes(path: Path) -> dict[str, dict[str, Any]]:
    with np.load(path, allow_pickle=False) as payload:
        metadata = json.loads(str(payload["metadata"].item()))
        result = {domain: dict(value) for domain, value in metadata.items()}
        for key in payload.files:
            if key == "metadata":
                continue
            domain, name = key.split("__", maxsplit=1)
            result.setdefault(domain, {})[name] = payload[key].copy()
    return result


def normalized_action_vector(actions: np.ndarray, spacing_m: float) -> np.ndarray:
    value = np.asarray(actions, dtype=np.float64).copy()
    value[..., 0] /= spacing_m
    value[..., 1] /= spacing_m
    return value.reshape(value.shape[0], value.shape[1], 12)


def support_score(
    envelope: dict[str, Any], candidates: np.ndarray, *, spacing_m: float
) -> tuple[np.ndarray, np.ndarray]:
    actions = nomad_waypoints_to_local_actions(candidates, dataset="recon")
    vector = normalized_action_vector(actions, spacing_m)
    centered = vector - np.asarray(envelope["mean"], dtype=np.float64)
    inverse = np.asarray(envelope["covariance_inverse"], dtype=np.float64)
    mahalanobis_sq = np.einsum("nkd,df,nkf->nk", centered, inverse, centered)
    mahalanobis = np.sqrt(np.maximum(mahalanobis_sq, 0.0) / vector.shape[-1])
    robust_z = np.abs(
        (vector - np.asarray(envelope["robust_center"], dtype=np.float64))
        / np.asarray(envelope["robust_scale"], dtype=np.float64)
    ).max(axis=-1)
    score = np.maximum(mahalanobis, robust_z / 3.0)
    guard = (
        np.abs(vector) <= np.asarray(envelope["coordinate_abs_limit"], dtype=np.float64)
    ).all(axis=-1)
    score[:, 0] = 0.0
    guard[:, 0] = True
    return score.astype(np.float32), guard


def support_threshold(envelope: dict[str, Any]) -> float:
    return float(
        np.quantile(
            np.asarray(envelope["fit_nonconformity"], dtype=np.float64),
            SUPPORT_QUANTILE,
            method="higher",
        )
    )


def support_valid(
    envelope: dict[str, Any], candidates: np.ndarray, *, spacing_m: float, threshold: float
) -> np.ndarray:
    score, guard = support_score(envelope, candidates, spacing_m=spacing_m)
    valid = guard & (score <= threshold)
    valid[:, 0] = True
    return valid


def sample_population(
    *,
    incumbent: np.ndarray,
    reference: np.ndarray,
    mean: np.ndarray,
    covariance: np.ndarray,
    rng: np.random.Generator,
    envelope: dict[str, Any],
    threshold: float,
    spacing_m: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    states = len(incumbent)
    dimension = mean.shape[-1]
    result = np.empty((states, POPULATION + 1, 4, 2), dtype=np.float32)
    result[:, 0] = incumbent
    reference_flat = reference.reshape(states, reference.shape[1], dimension)
    lower = reference_flat.min(axis=1) - 0.35 * reference_flat.std(axis=1)
    upper = reference_flat.max(axis=1) + 0.35 * reference_flat.std(axis=1)
    accepted_total = attempted_total = fallback_total = 0
    for state in range(states):
        accepted: list[np.ndarray] = []
        attempts = 0
        maximum_attempts = POPULATION * 24
        while len(accepted) < POPULATION and attempts < maximum_attempts:
            draw_count = min(64, maximum_attempts - attempts)
            draw = rng.multivariate_normal(
                mean[state].astype(np.float64),
                covariance[state].astype(np.float64),
                size=draw_count,
                check_valid="raise",
            )
            draw = np.clip(draw, lower[state], upper[state]).astype(np.float32)
            paths = draw.reshape(draw_count, 4, 2)
            pool = np.concatenate((incumbent[state : state + 1], paths), axis=0)[None]
            score, guard = support_score(envelope, pool, spacing_m=spacing_m)
            valid = guard[0, 1:] & (score[0, 1:] <= threshold)
            accepted.extend(paths[valid])
            attempts += draw_count
            attempted_total += draw_count
            accepted_total += int(valid.sum())
        if len(accepted) < POPULATION:
            missing = POPULATION - len(accepted)
            fallback_total += missing
            accepted.extend(reference[state, :missing])
        result[state, 1:] = np.asarray(accepted[:POPULATION], dtype=np.float32)
    return result, {
        "attempted_samples": int(attempted_total),
        "support_accepted_samples": int(accepted_total),
        "support_acceptance": (
            float(accepted_total / attempted_total) if attempted_total else 0.0
        ),
        "fallback_samples": int(fallback_total),
    }


def source_for_formal(selected: dict[str, np.ndarray]) -> np.ndarray:
    result = np.full(len(selected["record_key"]), "incumbent", dtype="U16")
    source_round = selected["archive_source_round"].astype(np.int64)
    result[source_round == -1] = "fixed"
    for iteration in range(ITERATIONS):
        result[source_round == iteration] = f"r{iteration + 1}"
    return result


def select_existing_union(
    pools: list[np.ndarray],
    valid_masks: list[np.ndarray],
    ground_truth: np.ndarray,
    noncem_actions: np.ndarray,
    *,
    dual_safe: bool,
    require_valid: bool,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    sources = ("fixed", "r1", "r2", "r3")
    metrics = [metric_values(pool, ground_truth) for pool in pools]
    noncem_metrics = action_metrics(noncem_actions, ground_truth)
    selected = np.empty_like(noncem_actions, dtype=np.float32)
    selected_source = np.empty(len(noncem_actions), dtype="U16")
    unique_count = np.empty(len(noncem_actions), dtype=np.int64)
    eligible_count = np.empty(len(noncem_actions), dtype=np.int64)
    valid_unique_count = np.empty(len(noncem_actions), dtype=np.int64)
    for state in range(len(noncem_actions)):
        seen: dict[bytes, tuple[np.ndarray, np.ndarray, str, bool]] = {}
        for source, pool, metric, valid in zip(sources, pools, metrics, valid_masks):
            for index in range(pool.shape[1]):
                candidate = np.ascontiguousarray(pool[state, index], dtype=np.float32)
                key = candidate.tobytes()
                if key not in seen:
                    seen[key] = (
                        candidate,
                        metric[state, index],
                        source,
                        bool(valid[state, index]),
                    )
        values = list(seen.values())
        unique_count[state] = len(values)
        valid_unique_count[state] = sum(int(value[3]) for value in values)
        eligible: list[tuple[np.ndarray, np.ndarray, str, bool]] = []
        for value in values:
            if require_valid and not value[3]:
                continue
            if dual_safe and not (
                value[1][0] < noncem_metrics[state, 0] - TOLERANCE
                and value[1][3] < noncem_metrics[state, 3] - TOLERANCE
            ):
                continue
            eligible.append(value)
        eligible_count[state] = len(eligible)
        if not eligible:
            selected[state] = noncem_actions[state]
            selected_source[state] = "fixed"
            continue
        best = min(eligible, key=lambda value: float(value[1][3]))
        selected[state] = best[0]
        selected_source[state] = best[2]
    audit = {
        "raw_candidates_per_state": int(sum(pool.shape[1] for pool in pools)),
        "unique_candidates_per_state": {
            "mean": float(unique_count.mean()),
            "min": int(unique_count.min()),
            "max": int(unique_count.max()),
        },
        "valid_unique_candidates_per_state": {
            "mean": float(valid_unique_count.mean()),
            "min": int(valid_unique_count.min()),
            "max": int(valid_unique_count.max()),
        },
        "eligible_candidates_per_state": {
            "mean": float(eligible_count.mean()),
            "min": int(eligible_count.min()),
            "max": int(eligible_count.max()),
        },
        "dual_safe": bool(dual_safe),
        "support_valid_required": bool(require_valid),
    }
    return selected, selected_source, audit


def run_oracle_loop(
    *,
    base: np.ndarray,
    base_valid: np.ndarray,
    ground_truth: np.ndarray,
    noncem_actions: np.ndarray,
    domain: str,
    envelope: dict[str, Any],
    threshold: float,
    dual_safe: bool,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    spacing_m = 0.25 if domain == "recon" else 0.38
    incumbent = base[:, 0].astype(np.float32)
    reference = base[:, 1:].astype(np.float32)
    noncem_metrics = action_metrics(noncem_actions, ground_truth)
    mean, covariance = initial_proposal(reference)
    base_metrics = metric_values(base, ground_truth)
    eligible = np.asarray(base_valid, dtype=bool).copy()
    if dual_safe:
        eligible &= (
            (base_metrics[..., 0] < noncem_metrics[:, None, 0] - TOLERANCE)
            & (base_metrics[..., 3] < noncem_metrics[:, None, 3] - TOLERANCE)
        )
    before = proposal_summary(mean, covariance)
    mean, covariance, _, updated, count = oracle_update(
        base,
        base_metrics,
        eligible,
        previous_mean=mean,
        previous_covariance=covariance,
    )
    base_audit = {
        "stage": "fixed_initialization",
        "eligible_candidates": int(eligible[:, 1:].sum()),
        "eligible_count_mean": float(count.mean()),
        "eligible_count_min": int(count.min()),
        "proposal_updated_windows": int(updated.sum()),
        "shortfall_windows": int((~updated).sum()),
        "proposal_before": before,
        "proposal_after": proposal_summary(mean, covariance),
    }
    rng = np.random.default_rng(BASE_SEED + (100000 if domain == "scand" else 0))
    pools = [base]
    valids = [base_valid]
    iteration_audit = []
    for iteration in range(ITERATIONS):
        proposal_before = proposal_summary(mean, covariance)
        candidates, sampling = sample_population(
            incumbent=incumbent,
            reference=reference,
            mean=mean,
            covariance=covariance,
            rng=rng,
            envelope=envelope,
            threshold=threshold,
            spacing_m=spacing_m,
        )
        true_metrics = metric_values(candidates, ground_truth)
        valid = support_valid(
            envelope,
            candidates,
            spacing_m=spacing_m,
            threshold=threshold,
        )
        local_eligible = valid.copy()
        if dual_safe:
            local_eligible &= (
                (true_metrics[..., 0] < noncem_metrics[:, None, 0] - TOLERANCE)
                & (true_metrics[..., 3] < noncem_metrics[:, None, 3] - TOLERANCE)
            )
        mean, covariance, elite, updated, count = oracle_update(
            candidates,
            true_metrics,
            local_eligible,
            previous_mean=mean,
            previous_covariance=covariance,
        )
        pools.append(candidates)
        valids.append(valid)
        iteration_audit.append(
            {
                "round": iteration + 1,
                **sampling,
                "raw_candidates": int(candidates[:, 1:].size // 8),
                "valid_candidates": int(valid[:, 1:].sum()),
                "eligible_candidates": int(local_eligible[:, 1:].sum()),
                "eligible_count_mean": float(count.mean()),
                "eligible_count_min": int(count.min()),
                "proposal_updated_windows": int(updated.sum()),
                "shortfall_windows": int((~updated).sum()),
                "elite_candidates": int((elite >= 0).sum()),
                "proposal_before": proposal_before,
                "proposal_after": proposal_summary(mean, covariance),
            }
        )
    selected, source, archive_audit = select_existing_union(
        pools,
        valids,
        ground_truth,
        noncem_actions,
        dual_safe=dual_safe,
        require_valid=True,
    )
    return selected, source, {
        "mode": "dual_safe" if dual_safe else "unconstrained_objective",
        "offline_future_ground_truth_used_for_every_elite": True,
        "base_update": base_audit,
        "iterations": iteration_audit,
        "archive": archive_audit,
    }


def load_cohort(spec: dict[str, Any], envelopes: dict[str, dict[str, Any]]) -> dict[str, Any]:
    replay = Path(spec["replay"])
    ground_path = Path(spec["ground_truth"])
    paths = [replay, ground_path, SUPPORT_PATH, PREREG]
    reject_forbidden_paths(paths)
    selected = load_npz(replay / "selected_actions.npz")
    ground = load_npz(ground_path)
    record = selected["record_key"].astype(str)
    if not np.array_equal(record, ground["record_key"].astype(str)):
        raise RuntimeError(f"record mismatch for {spec['name']}")
    ground_truth = ground["ground_truth_local_m"].astype(np.float32)
    base = selected["base_k17_feature_waypoint_units"].astype(np.float32)
    fixed_index = selected["fixed_pool_selected_index"].astype(np.int64)
    final = selected["final_cem_feature_waypoint_units"].astype(np.float32)
    final_index = selected["final_cem_selected_index"].astype(np.int64)
    row = np.arange(len(record))
    fixed_actions = base[row, fixed_index]
    formal_actions = final[row, final_index]
    formal_source = source_for_formal(selected)
    pools = [base]
    valid_masks = [selected["base_valid"].astype(bool)]
    for iteration in range(ITERATIONS):
        directory = replay / f"iteration_{iteration:02d}"
        population = load_npz(directory / "population.npz")
        score = load_npz(directory / "frozen_scores.npz")
        if not (
            np.array_equal(population["record_key"].astype(str), record)
            and np.array_equal(score["record_key"].astype(str), record)
        ):
            raise RuntimeError(f"iteration record mismatch for {spec['name']}")
        pools.append(population["candidates_feature_waypoint_units"].astype(np.float32))
        valid_masks.append(score["valid"].astype(bool))
    envelope = envelopes[str(spec["domain"])]
    threshold = support_threshold(envelope)
    spacing_m = 0.25 if spec["domain"] == "recon" else 0.38
    recomputed_base_valid = support_valid(
        envelope, base, spacing_m=spacing_m, threshold=threshold
    )
    if not np.array_equal(recomputed_base_valid, valid_masks[0]):
        changed = int(np.logical_xor(recomputed_base_valid, valid_masks[0]).sum())
        raise RuntimeError(f"base support-valid mismatch for {spec['name']}: {changed}")
    union_objective, union_objective_source, union_objective_audit = (
        select_existing_union(
            pools,
            valid_masks,
            ground_truth,
            fixed_actions,
            dual_safe=False,
            require_valid=False,
        )
    )
    union_dual, union_dual_source, union_dual_audit = select_existing_union(
        pools,
        valid_masks,
        ground_truth,
        fixed_actions,
        dual_safe=True,
        require_valid=False,
    )
    union_valid, union_valid_source, union_valid_audit = select_existing_union(
        pools,
        valid_masks,
        ground_truth,
        fixed_actions,
        dual_safe=False,
        require_valid=True,
    )
    loop_objective, loop_objective_source, loop_objective_audit = run_oracle_loop(
        base=base,
        base_valid=valid_masks[0],
        ground_truth=ground_truth,
        noncem_actions=fixed_actions,
        domain=str(spec["domain"]),
        envelope=envelope,
        threshold=threshold,
        dual_safe=False,
    )
    loop_dual, loop_dual_source, loop_dual_audit = run_oracle_loop(
        base=base,
        base_valid=valid_masks[0],
        ground_truth=ground_truth,
        noncem_actions=fixed_actions,
        domain=str(spec["domain"]),
        envelope=envelope,
        threshold=threshold,
        dual_safe=True,
    )
    methods = {
        "v70_fixed_noncem": (fixed_actions, np.full(len(record), "fixed", dtype="U16")),
        "v70_formal_cem": (formal_actions, formal_source),
        "existing_union_objective_oracle": (union_objective, union_objective_source),
        "existing_union_dual_safe_oracle": (union_dual, union_dual_source),
        "existing_union_objective_support_valid_sensitivity": (
            union_valid,
            union_valid_source,
        ),
        "oracle_loop_objective": (loop_objective, loop_objective_source),
        "oracle_loop_dual_safe": (loop_dual, loop_dual_source),
    }
    return {
        "name": str(spec["name"]),
        "domain": str(spec["domain"]),
        "record_key": record,
        "ground_truth": ground_truth,
        "incumbent": base[:, 0],
        "noncem": fixed_actions,
        "methods": methods,
        "audit": {
            "states": int(len(record)),
            "support_threshold": threshold,
            "existing_union_objective": union_objective_audit,
            "existing_union_dual_safe": union_dual_audit,
            "existing_union_objective_support_valid_sensitivity": union_valid_audit,
            "oracle_loop_objective": loop_objective_audit,
            "oracle_loop_dual_safe": loop_dual_audit,
            "input_hashes": {
                "selected_actions": sha256_file(replay / "selected_actions.npz"),
                "ground_truth": sha256_file(ground_path),
                **{
                    f"round_{iteration + 1}_population": sha256_file(
                        replay / f"iteration_{iteration:02d}/population.npz"
                    )
                    for iteration in range(ITERATIONS)
                },
            },
        },
    }


def summarize_group(cohorts: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    method_names = tuple(cohorts[0]["methods"])
    ground_truth = np.concatenate([cohort["ground_truth"] for cohort in cohorts])
    incumbent = np.concatenate([cohort["incumbent"] for cohort in cohorts])
    noncem = np.concatenate([cohort["noncem"] for cohort in cohorts])
    summaries: dict[str, Any] = {}
    arrays: dict[str, np.ndarray] = {
        "record_key": np.concatenate([cohort["record_key"] for cohort in cohorts]),
        "domain": np.concatenate(
            [np.full(len(cohort["record_key"]), cohort["domain"], dtype="U8") for cohort in cohorts]
        ),
        "cohort": np.concatenate(
            [np.full(len(cohort["record_key"]), cohort["name"], dtype="U32") for cohort in cohorts]
        ),
    }
    for name in method_names:
        actions = np.concatenate([cohort["methods"][name][0] for cohort in cohorts])
        sources = np.concatenate([cohort["methods"][name][1] for cohort in cohorts])
        summary, local = summarize_method(
            actions,
            sources,
            ground_truth=ground_truth,
            incumbent=incumbent,
            noncem=noncem,
        )
        summaries[name] = summary
        for key, value in local.items():
            arrays[f"{name}__{key}"] = value
    baseline = summaries["v70_fixed_noncem"]
    formal = summaries["v70_formal_cem"]
    comparisons = {
        name: {
            "vs_v70_fixed_noncem": method_difference(summary, baseline),
            "vs_v70_formal_cem": method_difference(summary, formal),
        }
        for name, summary in summaries.items()
    }
    return {"methods": summaries, "comparisons": comparisons}, arrays


def decomposition(group: dict[str, Any], oracle_name: str, loop_name: str) -> dict[str, Any]:
    methods = group["methods"]
    fixed = methods["v70_fixed_noncem"]
    formal = methods["v70_formal_cem"]
    union = methods[oracle_name]
    loop = methods[loop_name]
    return {
        "realized_cem_gain_formal_minus_noncem": method_difference(formal, fixed),
        "remaining_selection_regret_union_oracle_minus_formal": method_difference(
            union, formal
        ),
        "search_distribution_gain_oracle_loop_minus_existing_union": method_difference(
            loop, union
        ),
        "positive_gap_m_objective_mean": {
            "realized_cem_gain": fixed["absolute_mean_m"]["objective"]
            - formal["absolute_mean_m"]["objective"],
            "remaining_selection_regret": formal["absolute_mean_m"]["objective"]
            - union["absolute_mean_m"]["objective"],
            "additional_search_distribution_gain": union["absolute_mean_m"]["objective"]
            - loop["absolute_mean_m"]["objective"],
        },
    }


def main() -> None:
    if not PREREG.is_file():
        raise FileNotFoundError(PREREG)
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    if prereg.get("status") != "frozen_before_oracle_computation":
        raise RuntimeError("oracle definition is not frozen")
    if (OUTPUT / "result.json").exists() or (OUTPUT / "per_state_metrics.npz").exists():
        raise FileExistsError("oracle diagnostic result already exists")
    reject_forbidden_paths([Path(spec["replay"]) for spec in COHORTS])
    envelopes = load_envelopes(SUPPORT_PATH)
    cohorts = []
    for spec in COHORTS:
        print(f"loading and evaluating {spec['name']}", flush=True)
        cohorts.append(load_cohort(spec, envelopes))
    recon = [cohort for cohort in cohorts if cohort["domain"] == "recon"]
    scand = [cohort for cohort in cohorts if cohort["domain"] == "scand"]
    pooled_summary, pooled_arrays = summarize_group(cohorts)
    recon_summary, _ = summarize_group(recon)
    scand_summary, _ = summarize_group(scand)
    if len(pooled_arrays["record_key"]) != int(prereg["states_expected"]["pooled"]):
        raise RuntimeError("pooled state count changed")
    result = {
        "schema": "v70_cem_true_outcome_oracle_diagnostic_v1",
        "status": "complete_spent_development_oracle_diagnostic",
        "preregistration": str(PREREG),
        "preregistration_sha256": sha256_file(PREREG),
        "reserve_accessed": False,
        "formal_v70_rerun": False,
        "model_or_scorer_called": False,
        "future_ground_truth_used_inside_oracle_loop": True,
        "deployment_method": False,
        "production_claim_allowed": False,
        "metric_order": list(METRICS),
        "pooled": pooled_summary,
        "recon_a_to_h": recon_summary,
        "fresh_scand": scand_summary,
        "decomposition": {
            "unconstrained_objective": decomposition(
                pooled_summary,
                "existing_union_objective_oracle",
                "oracle_loop_objective",
            ),
            "dual_safe": decomposition(
                pooled_summary,
                "existing_union_dual_safe_oracle",
                "oracle_loop_dual_safe",
            ),
        },
        "cohort_audit": {cohort["name"]: cohort["audit"] for cohort in cohorts},
        "support_checkpoint": str(SUPPORT_PATH),
        "support_checkpoint_sha256": sha256_file(SUPPORT_PATH),
        "limitations": [
            "Every oracle reads offline future ground truth and cannot be deployed online.",
            "Existing-union primary oracles intentionally include every cached unique proposal; a support-valid sensitivity is reported separately.",
            "Oracle-in-loop proposals differ from formal v70 because true outcomes, not scorer predictions, update q.",
            "All evidence cohorts were already spent development cohorts; this is not independent confirmation.",
        ],
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUTPUT / "per_state_metrics.npz", **pooled_arrays)
    (OUTPUT / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    audit = {
        "result_sha256": sha256_file(OUTPUT / "result.json"),
        "per_state_metrics_sha256": sha256_file(OUTPUT / "per_state_metrics.npz"),
        "preregistration_sha256": sha256_file(PREREG),
        "reserve_accessed": False,
        "formal_v70_rerun": False,
    }
    (OUTPUT / "audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
