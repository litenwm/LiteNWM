#!/usr/bin/env python3
"""Strict nested route-OOF SCAND-tail safety recalibration on frozen v70 candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import NormalDist
import sys
import time
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from tools.evaluation import run_scand_filter_random_sensitivity_v1 as prior  # noqa: E402
from tools.evaluation import run_v72_proposal_tail_reranker as v72  # noqa: E402


OUTPUT = ROOT / "outputs/cem_development_v1/v76_scand_tail_safety_calibration"
PREREG = OUTPUT / "preregistration.json"
RESULT = OUTPUT / "result.json"
AUDIT = OUTPUT / "audit.json"
PREDICTIONS = OUTPUT / "outer_oof_predictions.npz"
PARAMETERS = OUTPUT / "calibration_parameters.json"
MANIFEST = OUTPUT / "sha256_manifest.json"
FAMILIES = ("location_scale_platt", "source_uncertainty_conformal_platt")
PROFILES = {
    "balanced": {"q": 0.90, "dual": 0.60, "harm": 0.10},
    "conservative": {"q": 0.95, "dual": 0.75, "harm": 0.05},
}
METHODS = tuple(f"{family}__{profile}" for family in FAMILIES for profile in PROFILES)
METRICS = ("ate", "rpe", "endpoint", "objective")
SOURCES = ("fixed", "r1", "r2", "r3")
TOLERANCE = 1.0e-7
NEUTRAL_M = 0.001
SUPPORT_THRESHOLD = 4.876645565032959
Z95 = 1.96


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(jsonable(value), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def write_npz_atomic(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def verify_contract() -> tuple[dict[str, Any], dict[str, str]]:
    prereg = read_json(PREREG)
    if prereg.get("status") != "frozen_before_v76_outer_heldout_evaluation":
        raise RuntimeError("v76 preregistration is not frozen")
    if any(prereg["constraints"].values()):
        raise RuntimeError("v76 forbidden-action contract changed")
    observed: dict[str, str] = {"preregistration": sha256_file(PREREG)}
    forbidden = "res" + "erve"
    for name, record in prereg["inputs"].items():
        path = ROOT / record["path"]
        if forbidden in str(path).lower():
            raise RuntimeError(f"forbidden input path: {name}")
        digest = sha256_file(path)
        if digest != record["sha256"]:
            raise RuntimeError(f"immutable input changed: {name}")
        observed[name] = digest
    return prereg, observed


def sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -30.0, 30.0)))


def logit(probability: np.ndarray) -> np.ndarray:
    value = np.clip(np.asarray(probability, dtype=np.float64), 1.0e-5, 1.0 - 1.0e-5)
    return np.log(value / (1.0 - value))


def fit_platt(probability: np.ndarray, target: np.ndarray) -> np.ndarray:
    x = logit(probability).reshape(-1)
    y = np.asarray(target, dtype=np.float64).reshape(-1)
    center = float(x.mean())
    scale = max(float(x.std()), 1.0e-6)
    normalized = (x - center) / scale
    design = np.stack((normalized, np.ones_like(normalized)), axis=1)
    prevalence = float(np.clip(y.mean(), 1.0e-5, 1.0 - 1.0e-5))
    coefficient = np.asarray((0.0, math.log(prevalence / (1.0 - prevalence))), dtype=np.float64)
    regularization = np.diag((1.0e-3, 0.0))

    def loss(value: np.ndarray) -> float:
        linear = design @ value
        likelihood = np.logaddexp(0.0, linear) - y * linear
        return float(likelihood.sum() + 0.5 * value @ regularization @ value)

    current_loss = loss(coefficient)
    for _ in range(100):
        prediction = sigmoid(design @ coefficient)
        weight = np.maximum(prediction * (1.0 - prediction), 1.0e-6)
        gradient = design.T @ (prediction - y) + regularization @ coefficient
        hessian = (design.T * weight) @ design + regularization
        step = np.linalg.solve(hessian, gradient)
        multiplier = 1.0
        accepted = False
        while multiplier >= 1.0e-8:
            proposal = coefficient - multiplier * step
            proposal_loss = loss(proposal)
            if proposal_loss <= current_loss:
                coefficient = proposal
                current_loss = proposal_loss
                accepted = True
                break
            multiplier *= 0.5
        if not accepted or float(np.linalg.norm(multiplier * step)) < 1.0e-9:
            break
    raw_slope = coefficient[0] / scale
    raw_intercept = coefficient[1] - raw_slope * center
    result = np.asarray((raw_slope, raw_intercept), dtype=np.float64)
    if not np.isfinite(result).all() or np.max(np.abs(result)) > 1.0e3:
        raise RuntimeError("Platt solver failed numerical-convergence audit")
    return result


def apply_platt(probability: np.ndarray, coefficient: np.ndarray) -> np.ndarray:
    return sigmoid(logit(probability) * coefficient[0] + coefficient[1]).astype(np.float32)


def fit_huber_affine(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    design = np.stack((np.ones_like(x), x), axis=1)
    ridge = np.diag((1.0e-6, 1.0e-6))
    coefficient = np.linalg.solve(design.T @ design + ridge, design.T @ y)
    for _ in range(10):
        residual = y - design @ coefficient
        center = np.median(residual)
        mad = np.median(np.abs(residual - center)) / 0.67448975
        cutoff = max(1.345 * mad, 1.0e-5)
        weight = np.minimum(1.0, cutoff / np.maximum(np.abs(residual), 1.0e-12))
        coefficient = np.linalg.solve(
            (design.T * weight) @ design + ridge, (design.T * weight) @ y
        )
    return coefficient


def weighted_quantile(value: np.ndarray, weight: np.ndarray, q: float) -> float:
    value = np.asarray(value, dtype=np.float64).reshape(-1)
    weight = np.asarray(weight, dtype=np.float64).reshape(-1)
    order = np.argsort(value, kind="stable")
    value, weight = value[order], weight[order]
    cumulative = np.cumsum(weight)
    target = q * cumulative[-1]
    index = min(int(np.searchsorted(cumulative, target, side="left")), len(value) - 1)
    return float(value[index])


def fit_family(
    family: str,
    profile: str,
    fields: dict[str, np.ndarray],
    target: np.ndarray,
    dual: np.ndarray,
    harm: np.ndarray,
    state_index: np.ndarray,
    calibration_pool: np.ndarray,
    group: np.ndarray,
) -> dict[str, Any]:
    profile_spec = PROFILES[profile]
    state_mask = np.zeros(len(group), dtype=bool)
    state_mask[state_index] = True
    mask = calibration_pool & state_mask[:, None]
    route_grid = np.broadcast_to(group[:, None], mask.shape)
    source = fields["source_pool"]
    parameters: dict[str, Any] = {
        "family": family,
        "profile": profile,
        "profile_values": profile_spec,
        "train_states": int(state_mask.sum()),
        "train_routes": int(len(np.unique(group[state_mask]))),
        "train_candidates": int(mask.sum()),
        "p_dual_platt": fit_platt(fields["p_dual"][mask], dual[mask]),
        "p_harm_platt": fit_platt(fields["p_harm"][mask], harm[mask]),
    }
    if family == "location_scale_platt":
        metric_parameters: dict[str, Any] = {}
        for name, index in (("ate", 0), ("objective", 3)):
            affine = fit_huber_affine(fields["mean"][..., index][mask], target[..., index][mask])
            calibrated = affine[0] + affine[1] * fields["mean"][..., index][mask]
            residual = target[..., index][mask] - calibrated
            ratio = np.abs(residual) / np.maximum(fields["std"][..., index][mask], 1.0e-4)
            scale = float(np.clip(np.median(ratio) / 0.67448975, 0.25, 4.0))
            metric_parameters[name] = {"affine": affine, "scale": scale}
        parameters["metrics"] = metric_parameters
    elif family == "source_uncertainty_conformal_platt":
        uncertainty = np.sqrt(fields["std"][..., 0] ** 2 + fields["std"][..., 3] ** 2)
        split = float(np.median(uncertainty[mask]))
        route_values, route_counts = np.unique(route_grid[mask], return_counts=True)
        count_map = dict(zip(route_values.tolist(), route_counts.tolist()))
        weight_grid = np.zeros(mask.shape, dtype=np.float64)
        weight_grid[mask] = np.asarray(
            [1.0 / count_map[value] for value in route_grid[mask]], dtype=np.float64
        )
        q = min(1.0, profile_spec["q"] * (1.0 + 1.0 / max(int(mask.sum()), 1)))
        offsets: dict[str, Any] = {}
        for name, index in (("ate", 0), ("objective", 3)):
            residual = target[..., index] - fields["mean"][..., index]
            global_offset = weighted_quantile(residual[mask], weight_grid[mask], q)
            bins: dict[str, Any] = {}
            for source_id, source_name in enumerate(SOURCES):
                for level, uncertainty_mask in (
                    ("low", uncertainty <= split),
                    ("high", uncertainty > split),
                ):
                    local = mask & (source == source_id) & uncertainty_mask
                    routes = int(len(np.unique(route_grid[local])))
                    use_bin = int(local.sum()) >= 100 and routes >= 5
                    offset = (
                        weighted_quantile(residual[local], weight_grid[local], q)
                        if use_bin
                        else global_offset
                    )
                    bins[f"{source_name}__{level}"] = {
                        "offset": offset,
                        "candidates": int(local.sum()),
                        "routes": routes,
                        "fallback_global": not use_bin,
                    }
            offsets[name] = {"global": global_offset, "bins": bins}
        parameters["uncertainty_split"] = split
        parameters["offsets"] = offsets
    else:
        raise ValueError(family)
    return parameters


def apply_family(
    parameters: dict[str, Any], fields: dict[str, np.ndarray], state_index: np.ndarray
) -> dict[str, np.ndarray]:
    family = parameters["family"]
    profile = PROFILES[parameters["profile"]]
    shape = (len(state_index), fields["support"].shape[1])
    mean_ate = np.empty(shape, dtype=np.float32)
    mean_objective = np.empty(shape, dtype=np.float32)
    ucb_ate = np.empty(shape, dtype=np.float32)
    ucb_objective = np.empty(shape, dtype=np.float32)
    if family == "location_scale_platt":
        z = NormalDist().inv_cdf(profile["q"])
        for name, index, destination_mean, destination_ucb in (
            ("ate", 0, mean_ate, ucb_ate),
            ("objective", 3, mean_objective, ucb_objective),
        ):
            spec = parameters["metrics"][name]
            affine = np.asarray(spec["affine"], dtype=np.float64)
            destination_mean[:] = (
                affine[0] + affine[1] * fields["mean"][state_index, :, index]
            )
            destination_ucb[:] = destination_mean + z * float(spec["scale"]) * fields["std"][state_index, :, index]
    else:
        uncertainty = np.sqrt(
            fields["std"][state_index, :, 0] ** 2
            + fields["std"][state_index, :, 3] ** 2
        )
        source = fields["source_pool"][state_index]
        for name, index, destination_mean, destination_ucb in (
            ("ate", 0, mean_ate, ucb_ate),
            ("objective", 3, mean_objective, ucb_objective),
        ):
            destination_mean[:] = fields["mean"][state_index, :, index]
            destination_ucb[:] = destination_mean
            for source_id, source_name in enumerate(SOURCES):
                for level, local_uncertainty in (
                    ("low", uncertainty <= parameters["uncertainty_split"]),
                    ("high", uncertainty > parameters["uncertainty_split"]),
                ):
                    local = (source == source_id) & local_uncertainty
                    offset = parameters["offsets"][name]["bins"][f"{source_name}__{level}"]["offset"]
                    destination_ucb[local] += float(offset)
    p_dual = apply_platt(fields["p_dual"][state_index], np.asarray(parameters["p_dual_platt"]))
    p_harm = apply_platt(fields["p_harm"][state_index], np.asarray(parameters["p_harm_platt"]))
    safe = (
        (ucb_ate < -0.01)
        & (ucb_objective < -0.01)
        & (p_dual >= profile["dual"])
        & (p_harm <= profile["harm"])
    )
    return {
        "mean_ate": mean_ate,
        "mean_objective": mean_objective,
        "ucb_ate": ucb_ate,
        "ucb_objective": ucb_objective,
        "p_dual": p_dual,
        "p_harm": p_harm,
        "safe": safe,
    }


def wilson(successes: int, total: int, z: float = Z95) -> dict[str, Any]:
    if total <= 0:
        return {"successes": int(successes), "total": int(total), "rate": None, "lcb": 0.0, "ucb": 1.0}
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    radius = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denominator
    return {
        "successes": int(successes),
        "total": int(total),
        "rate": float(p),
        "lcb": float(max(0.0, center - radius)),
        "ucb": float(min(1.0, center + radius)),
    }


def select_actions(
    data: dict[str, Any],
    fields: dict[str, np.ndarray],
    safe: np.ndarray,
    pool: np.ndarray,
    formal_energy: np.ndarray,
    state_index: np.ndarray | None = None,
) -> np.ndarray:
    if state_index is None:
        state_index = np.arange(len(data["formal_index"]))
    formal = data["formal_index"][state_index]
    local_safe = safe[state_index] if safe.shape[0] == len(data["formal_index"]) else safe
    local_pool = pool[state_index]
    energy = fields["energy"][state_index]
    threshold = formal_energy[state_index]
    eligible = local_pool & local_safe & np.isfinite(energy) & (energy < threshold[:, None])
    eligible[np.arange(len(state_index)), formal] = False
    masked = np.where(eligible, energy, np.inf)
    chosen = np.argmin(masked, axis=1)
    replace = np.isfinite(masked[np.arange(len(state_index)), chosen])
    return np.where(replace, chosen, formal).astype(np.int64)


def metric_summary(value: np.ndarray) -> dict[str, Any]:
    return {
        "mean_m": {name: float(value[:, index].mean()) for index, name in enumerate(METRICS)},
        "p90_m": {name: float(np.quantile(value[:, index], 0.9)) for index, name in enumerate(METRICS)},
    }


def evaluate_selection(
    data: dict[str, Any], selected: np.ndarray, state_index: np.ndarray | None = None
) -> dict[str, Any]:
    if state_index is None:
        state_index = np.arange(len(selected))
        formal = data["formal_index"]
        true_metrics = data["true_metrics"]
    else:
        formal = data["formal_index"][state_index]
        true_metrics = data["true_metrics"][state_index]
    row = np.arange(len(selected))
    selected_metrics = true_metrics[row, selected]
    formal_metrics = true_metrics[row, formal]
    gain = formal_metrics - selected_metrics
    replacement = selected != formal
    dual = replacement & (gain[:, 0] > TOLERANCE) & (gain[:, 3] > TOLERANCE)
    harm = replacement & ((gain[:, 0] < -TOLERANCE) | (gain[:, 3] < -TOLERANCE))
    neutral_dual = replacement & (gain[:, 0] > NEUTRAL_M) & (gain[:, 3] > NEUTRAL_M)
    neutral_harm = replacement & ((gain[:, 0] < -NEUTRAL_M) | (gain[:, 3] < -NEUTRAL_M))
    count = int(replacement.sum())
    result = {
        "states": int(len(selected)),
        "metrics": metric_summary(selected_metrics),
        "formal_metrics": metric_summary(formal_metrics),
        "delta_mean_m": {
            name: float(selected_metrics[:, index].mean() - formal_metrics[:, index].mean())
            for index, name in enumerate(METRICS)
        },
        "replacements": count,
        "coverage": float(count / len(selected)),
        "strict_zero_mm": {
            "dual": wilson(int(dual.sum()), count),
            "harm": wilson(int(harm.sum()), count),
            "neither_count": int((replacement & ~dual & ~harm).sum()),
        },
        "neutral_1mm_sensitivity": {
            "dual": wilson(int(neutral_dual.sum()), count),
            "harm": wilson(int(neutral_harm.sum()), count),
            "ambiguous_count": int((replacement & ~neutral_dual & ~neutral_harm).sum()),
        },
    }
    return result


def inner_feasible(summary: dict[str, Any]) -> tuple[bool, dict[str, bool]]:
    metrics = summary["metrics"]
    formal = summary["formal_metrics"]
    checks = {
        "nonzero": summary["replacements"] > 0,
        "dual_lcb": summary["strict_zero_mm"]["dual"]["lcb"] >= 0.85,
        "harm_ucb": summary["strict_zero_mm"]["harm"]["ucb"] <= 0.15,
        "mean_ate": metrics["mean_m"]["ate"] <= formal["mean_m"]["ate"] + 1.0e-12,
        "mean_objective": metrics["mean_m"]["objective"] <= formal["mean_m"]["objective"] + 1.0e-12,
        "objective_p90": metrics["p90_m"]["objective"] <= formal["p90_m"]["objective"] + 0.001,
        "objective_gain_1mm": summary["delta_mean_m"]["objective"] <= -0.001,
    }
    return all(checks.values()), checks


def auc_score(target: np.ndarray, probability: np.ndarray) -> float | None:
    target = np.asarray(target, dtype=bool).reshape(-1)
    probability = np.asarray(probability, dtype=np.float64).reshape(-1)
    positive, negative = int(target.sum()), int((~target).sum())
    if not positive or not negative:
        return None
    order = np.argsort(probability, kind="stable")
    ranks = np.empty(len(order), dtype=np.float64)
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and probability[order[stop]] == probability[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
        start = stop
    return float((ranks[target].sum() - positive * (positive + 1) / 2.0) / (positive * negative))


def probability_metrics(target: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    probability = np.asarray(probability, dtype=np.float64).reshape(-1)
    ece = 0.0
    bins = []
    for lower in np.linspace(0.0, 0.9, 10):
        upper = lower + 0.1
        local = (probability >= lower) & ((probability < upper) if upper < 1.0 else (probability <= upper))
        if local.any():
            observed = float(target[local].mean())
            predicted = float(probability[local].mean())
            ece += float(local.mean()) * abs(observed - predicted)
            bins.append({"lower": float(lower), "upper": float(upper), "n": int(local.sum()), "predicted": predicted, "observed": observed})
    return {
        "n": int(len(target)),
        "positive_rate": float(target.mean()),
        "brier": float(np.mean((probability - target) ** 2)),
        "ece_10": float(ece),
        "auroc": auc_score(target.astype(bool), probability),
        "bins": bins,
    }


def calibration_summary(
    prediction: dict[str, np.ndarray],
    target: np.ndarray,
    dual: np.ndarray,
    harm: np.ndarray,
    pool: np.ndarray,
    group: np.ndarray,
    source: np.ndarray,
) -> dict[str, Any]:
    finite = pool & np.isfinite(prediction["ucb_ate"]) & np.isfinite(prediction["p_dual"])

    def local(mask: np.ndarray) -> dict[str, Any]:
        if not mask.any():
            return {"n": 0}
        result: dict[str, Any] = {"n": int(mask.sum())}
        for name, index in (("ate", 0), ("objective", 3)):
            mean = prediction[f"mean_{name}"][mask]
            ucb = prediction[f"ucb_{name}"][mask]
            truth = target[..., index][mask]
            residual = truth - mean
            result[name] = {
                "residual_bias_m": float(residual.mean()),
                "mae_m": float(np.abs(residual).mean()),
                "ucb_coverage": float((truth <= ucb).mean()),
                "ucb_violation": float((truth > ucb).mean()),
            }
        result["dual_probability"] = probability_metrics(dual[mask], prediction["p_dual"][mask])
        result["harm_probability"] = probability_metrics(harm[mask], prediction["p_harm"][mask])
        return result

    by_source = {
        name: local(finite & (source == index)) for index, name in enumerate(SOURCES)
    }
    by_route = {
        route: local(finite & (group[:, None] == route)) for route in np.unique(group)
    }
    return {"overall": local(finite), "by_proposal_source": by_source, "by_route": by_route}


def filter_summary(
    safe: np.ndarray,
    pool: np.ndarray,
    dual_vs_formal: np.ndarray,
    harm_vs_formal: np.ndarray,
) -> dict[str, Any]:
    admitted = pool & safe
    dual_input = pool & dual_vs_formal
    harm_input = pool & harm_vs_formal
    dual_admitted = admitted & dual_vs_formal
    harm_admitted = admitted & harm_vs_formal
    opportunity_state = dual_input.any(axis=1)
    recovered_state = dual_admitted.any(axis=1)
    return {
        "input_candidates": int(pool.sum()),
        "admitted_candidates": int(admitted.sum()),
        "candidate_coverage": float(admitted.sum() / max(int(pool.sum()), 1)),
        "states_with_any_admitted": int(admitted.any(axis=1).sum()),
        "state_coverage": float(admitted.any(axis=1).mean()),
        "dual_input_candidates": int(dual_input.sum()),
        "dual_admitted_candidates": int(dual_admitted.sum()),
        "harm_input_candidates": int(harm_input.sum()),
        "harm_admitted_candidates": int(harm_admitted.sum()),
        "dual_recall": wilson(int(dual_admitted.sum()), int(dual_input.sum())),
        "dual_purity": wilson(int(dual_admitted.sum()), int(admitted.sum())),
        "harm_rejection": wilson(int(harm_input.sum() - harm_admitted.sum()), int(harm_input.sum())),
        "opportunity_states": int(opportunity_state.sum()),
        "recovered_opportunity_states": int((opportunity_state & recovered_state).sum()),
        "opportunity_state_recall": wilson(int((opportunity_state & recovered_state).sum()), int(opportunity_state.sum())),
    }


def route_end_to_end(data: dict[str, Any], selected: np.ndarray) -> dict[str, Any]:
    result = {}
    for route in np.unique(data["group"].astype(str)):
        index = np.flatnonzero(data["group"].astype(str) == route)
        result[route] = evaluate_selection(data, selected[index], index)
    return result


def frozen_recon_summary() -> dict[str, Any]:
    cohorts = {}
    total = 0
    weighted = np.zeros(4, dtype=np.float64)
    for bank in "abcdefgh":
        path = prior.V70_ROOT / f"evaluations/recon_bank_{bank}/result.json"
        decision = read_json(path)["decisions"]["proposal_aware_v66_cem"]
        states = int(decision["states"])
        vector = np.asarray([decision["absolute_mean_m"][name] for name in METRICS])
        weighted += states * vector
        total += states
        cohorts[f"recon_bank_{bank}"] = {
            "states": states,
            "absolute_mean_m": decision["absolute_mean_m"],
            "absolute_p90_m": decision["absolute_p90_m"],
            "sha256": sha256_file(path),
        }
    return {
        "states": total,
        "pooled_mean_m": {name: float(weighted[index] / total) for index, name in enumerate(METRICS)},
        "pooled_p90_m": "not reconstructed: preregistered result-only binding preserves each cohort p90 exactly",
        "cohorts": cohorts,
        "behavior": "unchanged by construction; v76 has no RECON inference path",
    }


def main() -> None:
    for path in (RESULT, AUDIT, PREDICTIONS, PARAMETERS, MANIFEST):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite v76 artifact: {path}")
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required only to reconstruct frozen fixed-K17 safety outputs")
    prereg, hashes_before = verify_contract()
    started = time.perf_counter()
    scand_spec = next(value for value in v72.COHORTS if value["name"] == "scand_fresh_outer")
    data = v72.load_cohort(scand_spec)
    domain = np.full(len(data["record_key"]), "scand", dtype="U8")
    fold, fold_audit = v72.assign_route_folds(domain, data["group"])
    data["fold"] = fold
    pools, reconstruction_audit = prior.reconstruct_pools(args.device)
    fields = prior.unionize(pools, data)
    selected_cache = v72.load_npz(prior.SELECTED)
    formal_energy = selected_cache["archive_energy"].astype(np.float64)
    support = fields["support"] <= SUPPORT_THRESHOLD
    pool = fields["physical"] & support
    pool[:, 0] = False
    incumbent_metrics = data["true_metrics"][:, :1, :]
    target = data["true_metrics"] - incumbent_metrics
    dual_incumbent = (target[..., 0] < -TOLERANCE) & (target[..., 3] < -TOLERANCE)
    harm_incumbent = (target[..., 0] > TOLERANCE) | (target[..., 3] > TOLERANCE)
    formal = data["formal_index"]
    row = np.arange(len(formal))
    formal_metrics = data["true_metrics"][row, formal]
    relative_formal = data["true_metrics"] - formal_metrics[:, None]
    dual_formal = (relative_formal[..., 0] < -TOLERANCE) & (relative_formal[..., 3] < -TOLERANCE)
    harm_formal = (relative_formal[..., 0] > TOLERANCE) | (relative_formal[..., 3] > TOLERANCE)

    prediction_keys = ("mean_ate", "mean_objective", "ucb_ate", "ucb_objective", "p_dual", "p_harm")
    outer_predictions = {
        method: {
            **{key: np.full(pool.shape, np.nan, dtype=np.float32) for key in prediction_keys},
            "safe": np.zeros(pool.shape, dtype=bool),
        }
        for method in METHODS
    }
    nested = {
        **{key: np.full(pool.shape, np.nan, dtype=np.float32) for key in prediction_keys},
        "safe": np.zeros(pool.shape, dtype=bool),
        "method_code": np.full(len(formal), -1, dtype=np.int16),
    }
    parameter_audit: dict[str, Any] = {"outer_folds": {}}
    inner_audit: dict[str, Any] = {}

    for outer in range(5):
        outer_train = np.flatnonzero(fold != outer)
        outer_test = np.flatnonzero(fold == outer)
        inner_predictions = {
            method: {
                **{key: np.full(pool.shape, np.nan, dtype=np.float32) for key in prediction_keys},
                "safe": np.zeros(pool.shape, dtype=bool),
            }
            for method in METHODS
        }
        inner_records = []
        for inner in range(5):
            if inner == outer:
                continue
            inner_train = np.flatnonzero((fold != outer) & (fold != inner))
            inner_test = np.flatnonzero(fold == inner)
            overlap = set(data["group"][inner_train].astype(str)) & set(data["group"][inner_test].astype(str))
            if overlap:
                raise RuntimeError("inner route leakage")
            record = {"inner_fold": inner, "train_states": int(len(inner_train)), "test_states": int(len(inner_test)), "route_overlap": 0}
            for method in METHODS:
                family, profile = method.split("__", 1)
                parameters = fit_family(family, profile, fields, target, dual_incumbent, harm_incumbent, inner_train, pool, data["group"])
                applied = apply_family(parameters, fields, inner_test)
                for key, value in applied.items():
                    inner_predictions[method][key][inner_test] = value
            inner_records.append(record)
        method_records = {}
        feasible_methods = []
        for method in METHODS:
            selected = select_actions(data, fields, inner_predictions[method]["safe"], pool, formal_energy, outer_train)
            summary = evaluate_selection(data, selected, outer_train)
            feasible, checks = inner_feasible(summary)
            method_records[method] = {"feasible": feasible, "checks": checks, "selection": summary}
            if feasible:
                feasible_methods.append(method)
        if feasible_methods:
            chosen = sorted(
                feasible_methods,
                key=lambda method: (
                    method_records[method]["selection"]["delta_mean_m"]["objective"],
                    -method_records[method]["selection"]["replacements"],
                    0 if method.endswith("conservative") else 1,
                    method,
                ),
            )[0]
        else:
            chosen = None
        outer_parameter_record: dict[str, Any] = {"chosen": chosen, "methods": {}}
        for method_index, method in enumerate(METHODS):
            family, profile = method.split("__", 1)
            parameters = fit_family(family, profile, fields, target, dual_incumbent, harm_incumbent, outer_train, pool, data["group"])
            applied = apply_family(parameters, fields, outer_test)
            for key, value in applied.items():
                outer_predictions[method][key][outer_test] = value
            outer_parameter_record["methods"][method] = parameters
            if method == chosen:
                for key, value in applied.items():
                    nested[key][outer_test] = value
                nested["method_code"][outer_test] = method_index
        parameter_audit["outer_folds"][str(outer)] = outer_parameter_record
        inner_audit[str(outer)] = {
            "outer_train_states": int(len(outer_train)),
            "outer_test_states": int(len(outer_test)),
            "outer_train_routes": int(len(np.unique(data["group"][outer_train]))),
            "outer_test_routes": int(len(np.unique(data["group"][outer_test]))),
            "outer_route_overlap": int(len(set(data["group"][outer_train]) & set(data["group"][outer_test]))),
            "inner_folds": inner_records,
            "methods": method_records,
            "chosen": chosen,
        }

    original_safe = fields["current_safe"].astype(bool)
    predictions_for_report = {**outer_predictions, "nested_selected": nested}
    calibration = {
        method: calibration_summary(value, target, dual_incumbent, harm_incumbent, pool, data["group"], fields["source_pool"])
        for method, value in predictions_for_report.items()
    }
    filters = {
        "original_v70_safety": filter_summary(original_safe, pool, dual_formal, harm_formal),
        **{
            method: filter_summary(value["safe"], pool, dual_formal, harm_formal)
            for method, value in predictions_for_report.items()
        },
    }
    selections = {}
    selected_indices = {}
    for method, value in predictions_for_report.items():
        selected = select_actions(data, fields, value["safe"], pool, formal_energy)
        selected_indices[method] = selected
        selections[method] = evaluate_selection(data, selected)
        selections[method]["by_route"] = route_end_to_end(data, selected)

    formal_summary = evaluate_selection(data, formal.copy())
    formal_eval = read_json(prior.V70_ROOT / "evaluations/scand_fresh_outer/result.json")["decisions"]["proposal_aware_v66_cem"]
    for name in METRICS:
        if abs(formal_summary["metrics"]["mean_m"][name] - formal_eval["absolute_mean_m"][name]) > 2.0e-7:
            raise RuntimeError(f"formal SCAND baseline reconstruction mismatch: {name}")

    recon = frozen_recon_summary()
    aggregate = read_json(prior.V70_ROOT / "aggregate_result.json")["pooled_decisions"]["proposal_aware_v66_cem"]
    pooled = {}
    for method, summary in selections.items():
        pooled_mean = {}
        for name in METRICS:
            pooled_mean[name] = float(
                aggregate["absolute_mean_m"][name]
                + (summary["metrics"]["mean_m"][name] - formal_summary["metrics"]["mean_m"][name]) * 100.0 / 788.0
            )
        pooled[method] = {
            "states": 788,
            "mean_m": pooled_mean,
            "objective_gain_vs_formal_v70_m": float(aggregate["absolute_mean_m"]["objective"] - pooled_mean["objective"]),
            "p90_m": "not reconstructible from preregistered result-only RECON bindings; SCAND p90 is reported directly",
        }

    nested_summary = selections["nested_selected"]
    strict = nested_summary["strict_zero_mm"]
    success_checks = {
        "nonzero": nested_summary["replacements"] > 0,
        "scand_mean_ate": nested_summary["metrics"]["mean_m"]["ate"] <= formal_summary["metrics"]["mean_m"]["ate"] + 1.0e-12,
        "scand_mean_objective": nested_summary["metrics"]["mean_m"]["objective"] <= formal_summary["metrics"]["mean_m"]["objective"] + 1.0e-12,
        "scand_objective_p90": nested_summary["metrics"]["p90_m"]["objective"] <= formal_summary["metrics"]["p90_m"]["objective"] + 0.001,
        "incremental_dual_lcb": strict["dual"]["lcb"] >= 0.85,
        "incremental_harm_ucb": strict["harm"]["ucb"] <= 0.15,
        "pooled_objective_gain_1mm": pooled["nested_selected"]["objective_gain_vs_formal_v70_m"] >= 0.001,
    }
    passed = all(success_checks.values())
    original_filter = filters["original_v70_safety"]
    nested_filter = filters["nested_selected"]
    filter_improved = (
        nested_filter["dual_recall"]["rate"] is not None
        and original_filter["dual_recall"]["rate"] is not None
        and nested_filter["dual_recall"]["rate"] > original_filter["dual_recall"]["rate"]
        and nested_filter["harm_rejection"]["rate"] >= original_filter["harm_rejection"]["rate"]
    )
    if passed:
        verdict = "development_candidate_pass"
        failure_stage = None
    elif filter_improved:
        verdict = "safety_calibration_only_pass/end_to_end_fail"
        failure_stage = "frozen_scorer_or_archive_selection"
    elif all(value["chosen"] is None for value in inner_audit.values()):
        verdict = "v76_fail"
        failure_stage = "inner_route_risk_certification"
    else:
        verdict = "v76_fail"
        failure_stage = "calibration_or_filter"

    result = {
        "schema": "v76_scand_tail_safety_calibration_result_v1",
        "status": "completed_spent_development_stopped",
        "verdict": verdict,
        "failure_stage": failure_stage,
        "formal_v70_mutated": False,
        "reserve_accessed": False,
        "dwm_ltwm_rerun": False,
        "candidate_generation": False,
        "recon": recon,
        "formal_scand_v70": formal_summary,
        "calibration_layer": calibration,
        "filter_layer": filters,
        "end_to_end_scand": selections,
        "pooled_recon_plus_scand": pooled,
        "formal_success_gate": {"checks": success_checks, "passed": passed},
        "nested_fold_choices": {fold_name: value["chosen"] for fold_name, value in inner_audit.items()},
        "independent_validation_claim_allowed": False,
        "production_promotion_allowed": False,
        "next_step_if_passed": (
            "freeze v76 as a development candidate and pre-register one untouched independent SCAND-route validation; "
            "do not collect or run it automatically"
            if passed
            else None
        ),
        "stopped_after_v76": True,
        "runtime_seconds": float(time.perf_counter() - started),
    }
    prediction_arrays: dict[str, np.ndarray] = {
        "record_key": data["record_key"].astype("U256"),
        "route": data["group"].astype("U256"),
        "outer_fold": fold.astype(np.int16),
        "source_id": fields["source_pool"].astype(np.int8),
        "physical_support_pool": pool,
        "formal_v70_index": formal.astype(np.int16),
        "formal_v70_energy": formal_energy.astype(np.float32),
        "target_delta": target.astype(np.float32),
        "true_dual_vs_incumbent": dual_incumbent,
        "true_harm_vs_incumbent": harm_incumbent,
        "true_dual_vs_formal_v70": dual_formal,
        "true_harm_vs_formal_v70": harm_formal,
        "nested_method_code": nested["method_code"],
    }
    for method, value in predictions_for_report.items():
        prefix = method.replace("source_uncertainty_conformal_platt", "conformal").replace("location_scale_platt", "location_scale")
        for key, array in value.items():
            if key == "method_code":
                continue
            prediction_arrays[f"{prefix}__{key}"] = array
        prediction_arrays[f"{prefix}__selected_index"] = selected_indices[method].astype(np.int16)

    write_npz_atomic(PREDICTIONS, **prediction_arrays)
    write_json_atomic(PARAMETERS, parameter_audit)
    hashes_after = {name: sha256_file(ROOT / record["path"]) for name, record in prereg["inputs"].items()}
    immutable_equal = all(hashes_after[name] == record["sha256"] for name, record in prereg["inputs"].items())
    audit = {
        "schema": "v76_scand_tail_safety_calibration_audit_v1",
        "preregistration_sha256": sha256_file(PREREG),
        "input_hashes_before": hashes_before,
        "input_hashes_after": hashes_after,
        "all_inputs_unchanged": immutable_equal,
        "route_folds": fold_audit,
        "nested_route_audit": inner_audit,
        "route_overlap_violations": int(sum(value["outer_route_overlap"] for value in inner_audit.values())),
        "reconstruction": reconstruction_audit,
        "union_current_valid_exact": bool(np.array_equal(fields["current_valid"], data["valid"])),
        "union_current_safe_exact": bool(np.array_equal(fields["current_safe"], data["safe"])),
        "formal_scand_baseline_exact_tolerance_2e7": True,
        "candidate_rows": int(pool.size),
        "physical_support_rows": int(pool.sum()),
        "unobservable_pre_persistence_draws": True,
        "reserve_path_seen": False,
        "runner_sha256": sha256_file(Path(__file__)),
        "numeric_erratum": {
            "initial_run_preserved_at": "initial_numeric_solver_failure",
            "reason": "undamped Platt Newton iterations diverged in four outer folds",
            "scientific_contract_changed": False,
            "models_thresholds_profiles_changed": False,
            "correction": "standardized-coordinate damped Newton with backtracking for the same two-parameter Platt family",
        },
    }
    if not immutable_equal:
        raise RuntimeError("an immutable v70/input artifact changed during v76")
    write_json_atomic(RESULT, result)
    write_json_atomic(AUDIT, audit)
    manifest = {
        "schema": "v76_sha256_manifest_v1",
        "artifacts": {
            path.name: {"sha256": sha256_file(path), "bytes": path.stat().st_size}
            for path in (PREREG, RESULT, AUDIT, PREDICTIONS, PARAMETERS, Path(__file__))
        },
    }
    write_json_atomic(MANIFEST, manifest)
    print(json.dumps({
        "verdict": verdict,
        "failure_stage": failure_stage,
        "nested_choices": result["nested_fold_choices"],
        "replacements": nested_summary["replacements"],
        "dual": strict["dual"],
        "harm": strict["harm"],
        "scand_objective_delta_m": nested_summary["delta_mean_m"]["objective"],
        "pooled_objective_gain_m": pooled["nested_selected"]["objective_gain_vs_formal_v70_m"],
        "result": str(RESULT),
    }, indent=2))


if __name__ == "__main__":
    main()
