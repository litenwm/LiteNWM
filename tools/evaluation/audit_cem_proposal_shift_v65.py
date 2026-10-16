#!/usr/bin/env python3
"""Two-stage spent diagnostic for CEM set-context and proposal-tail shift.

``freeze`` records 64 deterministic candidate identities without opening any
outcome artifact. ``run`` then re-scores the frozen identities under three
contexts and evaluates fixed-K17 versus CEM-tail calibration on spent A-H.
"""

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

from scoring.spacing_support_v1 import threshold_for_quantile  # noqa: E402
from scoring.universal_dualdomain_raw_v2 import build_universal_features  # noqa: E402
from tools.evaluation.build_recon100_k17_source_v1 import metric_labels  # noqa: E402
from tools.planning.run_cem_development_v1 import (  # noqa: E402
    conservative_energy,
)
from tools.planning.run_cem_safe_elite_v55 import (  # noqa: E402
    candidate_safety_mask,
)
from tools.scoring.finetune_universal_dualdomain_safe_rank_v2 import (  # noqa: E402
    load_envelopes,
)
from tools.scoring.train_universal_dualdomain_raw_v2 import (  # noqa: E402
    ensemble_predict,
)


BLOCKS = ("actions_h4", "action_features", "static_features", "raw_ltwm_features")
BANKS = (
    "bank_a",
    "bank_b",
    "bank_c",
    "bank_d",
    "bank_e",
    "bank_f",
    "bank_g_v60",
    "bank_h_v60",
)
AF = set(BANKS[:6])
STRATA = (
    (0, "high", 0.05),
    (0, "middle", 0.50),
    (0, "low", 0.95),
    (1, "high", 0.05),
    (1, "middle", 0.50),
    (2, "high", 0.05),
    (2, "middle", 0.50),
    (2, "low", 0.95),
)
P_BINS = np.asarray((0.0, 0.5, 0.7, 0.8, 0.9, 0.95, 1.000001))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def spec(name: str) -> dict[str, Path]:
    if name in AF:
        letter = name[-1]
        bank = ROOT / "outputs/cem_development_v1/recon_target_calibration_v20" / name
        run_name = "cem_run_v57_lobo_gate" if name in {"bank_e", "bank_f"} else "cem_run_v55_safe_elite"
        return {
            "bank": bank,
            "base": bank / "features_cem_v30" / f"recon_target_bank_{letter}_cem_v30_features.npz",
            "run": bank / run_name,
            "outcome": bank / "source_labeled.npz",
        }
    bank = ROOT / "outputs/cem_development_v1/recon_cem_boundary_v60" / name
    return {
        "bank": bank,
        "base": bank / "features" / f"recon_{name}_features.npz",
        "run": bank / "cem_run_v60",
        "outcome": ROOT / "outputs/cem_development_v1/recon_cem_boundary_v60/evaluation_v60" / f"{name}_outcomes.npz",
    }


def load_feature(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        required = (*BLOCKS, "candidates", "majority_support", "record_key")
        missing = [key for key in required if key not in payload.files]
        if missing:
            raise RuntimeError(f"{path}: missing {missing}")
        return {key: payload[key].copy() for key in required}


def load_iteration(name: str, iteration: int) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    directory = spec(name)["run"] / f"iteration_{iteration:02d}"
    feature = load_feature(directory / "cem_features.npz")
    with np.load(directory / "frozen_scores.npz", allow_pickle=False) as payload:
        score = {key: payload[key].copy() for key in payload.files}
    if not np.array_equal(feature["record_key"].astype(str), score.get("record_key", feature["record_key"]).astype(str)):
        raise RuntimeError(f"{name} iteration {iteration}: score identity mismatch")
    return feature, score


def freeze_identities(preregistration: Path, output_dir: Path) -> None:
    prereg = read_json(preregistration)
    if prereg.get("status") != "frozen_before_v65_spent_outcome_read":
        raise RuntimeError("preregistration status mismatch")
    identity_path = output_dir / "context_identities.json"
    if identity_path.exists():
        raise FileExistsError(identity_path)
    records: list[dict[str, Any]] = []
    artifacts: dict[str, Any] = {}
    for name in BANKS:
        base_path = spec(name)["base"]
        base = load_feature(base_path)
        artifacts[name] = {"base": str(base_path), "base_sha256": sha256(base_path), "iterations": []}
        used_states: set[int] = set()
        for iteration, stratum, quantile in STRATA:
            directory = spec(name)["run"] / f"iteration_{iteration:02d}"
            feature_path = directory / "cem_features.npz"
            score_path = directory / "frozen_scores.npz"
            feature, score = load_iteration(name, iteration)
            if not np.array_equal(feature["record_key"].astype(str), base["record_key"].astype(str)):
                raise RuntimeError(f"{name}: base/iteration identity mismatch")
            if not any(item["iteration"] == iteration for item in artifacts[name]["iterations"]):
                artifacts[name]["iterations"].append({
                    "iteration": iteration,
                    "feature": str(feature_path),
                    "feature_sha256": sha256(feature_path),
                    "score": str(score_path),
                    "score_sha256": sha256(score_path),
                })
            valid = np.asarray(score["valid"], dtype=bool).copy()
            valid[:, 0] = False
            row, candidate = np.nonzero(valid & np.isfinite(score["energy"]))
            if not len(row):
                raise RuntimeError(f"{name} iteration {iteration}: no valid challengers")
            order = np.argsort(score["energy"][row, candidate], kind="stable")
            target_pos = int(round(quantile * (len(order) - 1)))
            scan = np.concatenate((order[target_pos:], order[:target_pos]))
            chosen = None
            for location in scan:
                state = int(row[location])
                if state not in used_states:
                    chosen = (state, int(candidate[location]))
                    break
            if chosen is None:
                chosen = (int(row[order[target_pos]]), int(candidate[order[target_pos]]))
            state, local = chosen
            used_states.add(state)
            records.append({
                "bank": name,
                "record_key": str(base["record_key"][state]),
                "state_index": state,
                "iteration": iteration,
                "candidate_index": local,
                "energy_stratum": stratum,
                "frozen_energy": float(score["energy"][state, local]),
                "frozen_p_dual": float(score["dual_probability"][state, local]),
                "frozen_p_harm": float(score["harm_probability"][state, local]),
                "outcome_opened_for_identity": False,
            })
    if len(records) != 64:
        raise RuntimeError(f"expected 64 identities, got {len(records)}")
    payload = {
        "schema": "cem_proposal_shift_v65_context_identities",
        "status": "frozen_before_v65_spent_outcome_read",
        "preregistration": str(preregistration),
        "preregistration_sha256": sha256(preregistration),
        "outcomes_opened": False,
        "records": records,
        "source_artifacts": artifacts,
    }
    write_json(identity_path, payload)
    print(json.dumps({"stage": "freeze", "identities": len(records), "output": str(identity_path), "sha256": sha256(identity_path)}, indent=2))


def compose_context(
    base: dict[str, np.ndarray],
    iterations: list[dict[str, np.ndarray]],
    identity: dict[str, Any],
    context: str,
) -> tuple[dict[str, np.ndarray], np.ndarray, int]:
    state = int(identity["state_index"])
    iteration = int(identity["iteration"])
    local = int(identity["candidate_index"])
    target_source = iterations[iteration]
    if context == "original":
        result = {key: target_source[key][state : state + 1].copy() for key in BLOCKS}
        majority = target_source["majority_support"][state : state + 1].copy()
        return result, majority, local

    sources: list[tuple[dict[str, np.ndarray], int]] = [(base, 0), (target_source, local)]
    if context == "fixed_anchor":
        sources.extend((base, index) for index in range(2, 17))
    elif context == "cem_distractors":
        pool = [(source, index, it) for it, source in enumerate(iterations) for index in range(1, 17)
                if not (it == iteration and index == local)]
        seed_bytes = hashlib.sha256(f"v65:{identity['bank']}:{identity['record_key']}".encode()).digest()
        rng = np.random.default_rng(int.from_bytes(seed_bytes[:8], "big"))
        order = rng.permutation(len(pool))[:15]
        sources.extend((pool[index][0], pool[index][1]) for index in order)
    else:
        raise ValueError(context)
    if len(sources) != 17:
        raise RuntimeError(f"{context}: expected 17 candidates")
    result = {
        key: np.stack([source[key][state, index] for source, index in sources])[None]
        for key in BLOCKS
    }
    majority = np.asarray([source["majority_support"][state, index] for source, index in sources], dtype=bool)[None]
    return result, majority, 1


def score_contexts(
    raw_sets: list[dict[str, np.ndarray]],
    majority_sets: list[np.ndarray],
    training: dict[str, Any],
    envelope: dict[str, Any],
    device: str,
) -> dict[str, np.ndarray]:
    raw = {key: np.concatenate([item[key] for item in raw_sets], axis=0) for key in BLOCKS}
    majority = np.concatenate(majority_sets, axis=0)
    feature, support_score, physical_guard = build_universal_features(raw, spacing_m=0.25, support_envelope=envelope)
    threshold = threshold_for_quantile(envelope, 0.999)
    valid = physical_guard & (support_score <= threshold) & majority
    valid[:, 0] = True
    domain = np.zeros(len(feature), dtype=np.int64)
    prediction = ensemble_predict(training["arms"]["D"]["checkpoints"]["full"], feature, domain, device=device)
    policy = training["arms"]["D"]["calibration"]["frontier"]["domains"]["recon"]["policy"]
    temperature = training["arms"]["D"]["temperature"]["domains"]["recon"]
    energy, p_dual, p_harm = conservative_energy(
        prediction,
        policy=policy,
        dual_temperature=float(temperature["deployment_dual_temperature"]),
        harm_temperature=float(temperature["deployment_harm_temperature"]),
        valid=valid,
    )
    safe = candidate_safety_mask(prediction, p_dual, p_harm, valid, policy)
    return {**prediction, "energy": energy, "p_dual": p_dual, "p_harm": p_harm, "valid": valid, "safe": safe}


def ece(probability: np.ndarray, target: np.ndarray, bins: int = 10) -> float:
    p = np.asarray(probability, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for index in range(bins):
        mask = (p >= edges[index]) & (p < edges[index + 1] if index + 1 < bins else p <= edges[index + 1])
        if mask.any():
            total += float(mask.mean()) * abs(float(p[mask].mean()) - float(y[mask].mean()))
    return float(total)


def binary_summary(p_dual: np.ndarray, p_harm: np.ndarray, dual: np.ndarray, harm: np.ndarray) -> dict[str, Any]:
    if not len(p_dual):
        return {"n": 0}
    return {
        "n": int(len(p_dual)),
        "empirical_dual": float(np.mean(dual)),
        "empirical_harm": float(np.mean(harm)),
        "mean_p_dual": float(np.mean(p_dual)),
        "mean_p_harm": float(np.mean(p_harm)),
        "dual_overconfidence": float(np.mean(p_dual) - np.mean(dual)),
        "harm_underestimate": float(np.mean(harm) - np.mean(p_harm)),
        "dual_brier": float(np.mean(np.square(p_dual - dual))),
        "harm_brier": float(np.mean(np.square(p_harm - harm))),
        "dual_ece10": ece(p_dual, dual),
        "harm_ece10": ece(p_harm, harm),
    }


def probability_bins(p_dual: np.ndarray, p_harm: np.ndarray, dual: np.ndarray, harm: np.ndarray) -> list[dict[str, Any]]:
    result = []
    for lower, upper in zip(P_BINS[:-1], P_BINS[1:]):
        mask = (p_dual >= lower) & (p_dual < upper)
        entry = binary_summary(p_dual[mask], p_harm[mask], dual[mask], harm[mask])
        entry.update({"p_dual_lower": float(lower), "p_dual_upper": float(min(upper, 1.0))})
        result.append(entry)
    return result


def ranks(value: np.ndarray) -> np.ndarray:
    order = np.argsort(value, kind="stable")
    result = np.empty(len(value), dtype=np.float64)
    result[order] = np.arange(len(value), dtype=np.float64)
    return result


def within_state_rank(predicted: np.ndarray, truth: np.ndarray, valid: np.ndarray) -> dict[str, Any]:
    correlations = []
    slopes = []
    for state in range(len(predicted)):
        mask = valid[state].copy()
        mask[0] = False
        if mask.sum() < 3:
            continue
        x = predicted[state, mask].astype(np.float64)
        y = truth[state, mask].astype(np.float64)
        corr = float(np.corrcoef(ranks(x), ranks(y))[0, 1])
        if np.isfinite(corr):
            correlations.append(corr)
        variance = float(np.var(x))
        if variance > 1.0e-12:
            slopes.append(float(np.cov(x, y, ddof=0)[0, 1] / variance))
    return {
        "states": int(len(correlations)),
        "spearman_mean": None if not correlations else float(np.mean(correlations)),
        "spearman_median": None if not correlations else float(np.median(correlations)),
        "true_vs_predicted_slope_mean": None if not slopes else float(np.mean(slopes)),
    }


def selected_vectors(score: dict[str, np.ndarray], dual: np.ndarray, harm: np.ndarray, mode: str) -> dict[str, np.ndarray]:
    allowed = score["safe"] if mode == "safe" else score["valid"].copy()
    allowed[:, 0] = False
    masked = np.where(allowed, score["energy"], np.inf)
    selected = masked.argmin(axis=1)
    exists = allowed.any(axis=1)
    row = np.flatnonzero(exists)
    col = selected[exists]
    return {
        "state": row,
        "candidate": col,
        "p_dual": score["p_dual"][row, col],
        "p_harm": score["p_harm"][row, col],
        "dual": dual[row, col],
        "harm": harm[row, col],
    }


def route_bootstrap(route: np.ndarray, dual: np.ndarray, harm: np.ndarray, p_dual: np.ndarray, seed: int) -> dict[str, Any]:
    unique = np.unique(route.astype(str))
    if not len(unique):
        return {}
    per_route = []
    for name in unique:
        mask = route == name
        per_route.append((int(mask.sum()), float(dual[mask].sum()), float(harm[mask].sum()), float(p_dual[mask].sum())))
    values = np.asarray(per_route, dtype=np.float64)
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, len(unique), size=(2000, len(unique)))
    sampled = values[draw].sum(axis=1)
    metrics = np.column_stack((sampled[:, 1] / sampled[:, 0], sampled[:, 2] / sampled[:, 0], sampled[:, 3] / sampled[:, 0] - sampled[:, 1] / sampled[:, 0]))
    return {
        name: [float(np.quantile(metrics[:, index], 0.025)), float(np.quantile(metrics[:, index], 0.975))]
        for index, name in enumerate(("dual_ci95", "harm_ci95", "dual_overconfidence_ci95"))
    }


def true_scores(absolute: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    delta = absolute - absolute[:, :1]
    dual = (delta[..., 0] < -1.0e-7) & (delta[..., 3] < -1.0e-7)
    harm = (delta[..., 0] > 1.0e-7) | (delta[..., 3] > 1.0e-7)
    cost = delta[..., 0] + 0.25 * delta[..., 1] + 0.25 * np.maximum(delta[..., 2], 0.0) + 0.50 * delta[..., 3] + 0.10 * harm
    return -cost.astype(np.float32), dual, harm


def bank_outcomes(name: str, base: dict[str, np.ndarray], iterations: list[dict[str, np.ndarray]]) -> tuple[list[np.ndarray], np.ndarray]:
    candidates = np.concatenate([base["candidates"]] + [item["candidates"][:, 1:] for item in iterations], axis=1)
    path = spec(name)["outcome"]
    with np.load(path, allow_pickle=False) as payload:
        if not np.array_equal(payload["record_key"].astype(str), base["record_key"].astype(str)):
            raise RuntimeError(f"{name}: outcome identity mismatch")
        if name in AF:
            absolute = metric_labels(candidates, payload["ground_truth_local_m"], 0.25, 0.4, 0.6)["values"]
            if not np.allclose(absolute[:, :17], payload["absolute_metrics"], atol=2.0e-6):
                raise RuntimeError(f"{name}: base outcome replay mismatch")
            route = payload["route_group"].astype(str)
        else:
            absolute = payload["absolute_metrics"].copy()
            route = payload["route_id"].astype(str)
    if absolute.shape[1] != 65:
        raise RuntimeError(f"{name}: expected K65 outcomes")
    return [absolute[:, :17]] + [absolute[:, 17 + 16 * it : 17 + 16 * (it + 1)] for it in range(3)], route


def run(preregistration: Path, output_dir: Path, device: str) -> None:
    prereg = read_json(preregistration)
    identity_path = output_dir / "context_identities.json"
    identities = read_json(identity_path)
    if identities.get("outcomes_opened") is not False or identities.get("preregistration_sha256") != sha256(preregistration):
        raise RuntimeError("identity freeze contract mismatch")
    result_path = output_dir / "result.json"
    if result_path.exists():
        raise FileExistsError(result_path)
    training_path = ROOT / "outputs/universal_dualdomain_raw_v2/training_v2/result.json"
    training = read_json(training_path)
    envelopes = load_envelopes(Path(training["support"]))
    envelope = envelopes["recon"]

    context_rows: list[dict[str, Any]] = []
    per_bank_context: dict[str, Any] = {}
    for name in BANKS:
        base = load_feature(spec(name)["base"])
        iterations = [load_iteration(name, it)[0] for it in range(3)]
        local_identities = [item for item in identities["records"] if item["bank"] == name]
        raw_sets: list[dict[str, np.ndarray]] = []
        majority_sets: list[np.ndarray] = []
        positions: list[int] = []
        descriptors: list[tuple[dict[str, Any], str]] = []
        for identity in local_identities:
            for context in ("original", "fixed_anchor", "cem_distractors"):
                raw, majority, position = compose_context(base, iterations, identity, context)
                raw_sets.append(raw)
                majority_sets.append(majority)
                positions.append(position)
                descriptors.append((identity, context))
        scored = score_contexts(raw_sets, majority_sets, training, envelope, device)
        by_identity: dict[tuple[int, int, int], dict[str, dict[str, Any]]] = {}
        for row, ((identity, context), position) in enumerate(zip(descriptors, positions)):
            values = {
                "p_dual": float(scored["p_dual"][row, position]),
                "p_harm": float(scored["p_harm"][row, position]),
                "energy": float(scored["energy"][row, position]),
                "rank_logit": float(scored["rank_logit"][row, position]),
                "mean": scored["mean"][row, position].astype(float).tolist(),
                "std": scored["std"][row, position].astype(float).tolist(),
                "valid": bool(scored["valid"][row, position]),
                "safe": bool(scored["safe"][row, position]),
            }
            key = (int(identity["state_index"]), int(identity["iteration"]), int(identity["candidate_index"]))
            by_identity.setdefault(key, {})[context] = values
        bank_rows = []
        for identity in local_identities:
            key = (int(identity["state_index"]), int(identity["iteration"]), int(identity["candidate_index"]))
            contexts = by_identity[key]
            original = contexts["original"]
            comparisons = {}
            for context in ("fixed_anchor", "cem_distractors"):
                other = contexts[context]
                comparisons[context] = {
                    "delta_p_dual": other["p_dual"] - original["p_dual"],
                    "delta_p_harm": other["p_harm"] - original["p_harm"],
                    "delta_energy": other["energy"] - original["energy"],
                    "safe_flip": bool(other["safe"] != original["safe"]),
                }
            entry = {"identity": identity, "contexts": contexts, "comparisons": comparisons}
            context_rows.append(entry)
            bank_rows.append(entry)
        def context_summary(rows: list[dict[str, Any]], context: str) -> dict[str, Any]:
            d_dual = np.asarray([row["comparisons"][context]["delta_p_dual"] for row in rows])
            d_harm = np.asarray([row["comparisons"][context]["delta_p_harm"] for row in rows])
            flips = np.asarray([row["comparisons"][context]["safe_flip"] for row in rows])
            return {
                "n": int(len(rows)),
                "safe_flip_rate": float(flips.mean()),
                "median_abs_delta_p_dual": float(np.median(np.abs(d_dual))),
                "median_abs_delta_p_harm": float(np.median(np.abs(d_harm))),
                "mean_delta_p_dual": float(d_dual.mean()),
                "mean_delta_p_harm": float(d_harm.mean()),
            }
        per_bank_context[name] = {context: context_summary(bank_rows, context) for context in ("fixed_anchor", "cem_distractors")}

    pooled_context = {}
    for context in ("fixed_anchor", "cem_distractors"):
        d_dual = np.asarray([row["comparisons"][context]["delta_p_dual"] for row in context_rows])
        d_harm = np.asarray([row["comparisons"][context]["delta_p_harm"] for row in context_rows])
        flips = np.asarray([row["comparisons"][context]["safe_flip"] for row in context_rows])
        pooled_context[context] = {
            "n": int(len(context_rows)),
            "safe_flip_rate": float(flips.mean()),
            "median_abs_delta_p_dual": float(np.median(np.abs(d_dual))),
            "median_abs_delta_p_harm": float(np.median(np.abs(d_harm))),
            "mean_delta_p_dual": float(d_dual.mean()),
            "mean_delta_p_harm": float(d_harm.mean()),
        }
    context_bank_hits = []
    for name, values in per_bank_context.items():
        hit = any(item["safe_flip_rate"] >= 0.10 or item["median_abs_delta_p_dual"] >= 0.05 or item["median_abs_delta_p_harm"] >= 0.05 for item in values.values())
        if hit:
            context_bank_hits.append(name)
    context_pooled_hit = any(item["safe_flip_rate"] >= 0.10 or item["median_abs_delta_p_dual"] >= 0.05 or item["median_abs_delta_p_harm"] >= 0.05 for item in pooled_context.values())
    root_a = context_pooled_hit and len(context_bank_hits) >= 2

    calibration_banks: dict[str, Any] = {}
    pooled_fixed_selected: list[dict[str, np.ndarray]] = []
    pooled_cem_selected: list[dict[str, np.ndarray]] = []
    fixed_rank = []
    cem_rank = []
    for bank_index, name in enumerate(BANKS):
        base = load_feature(spec(name)["base"])
        iterations = [load_iteration(name, it)[0] for it in range(3)]
        absolute_sets, route = bank_outcomes(name, base, iterations)
        score_sets = []
        raw_sources = [base] + iterations
        for source in raw_sources:
            score_sets.append(score_contexts([{key: source[key] for key in BLOCKS}], [source["majority_support"]], training, envelope, device))
        bank_result: dict[str, Any] = {"sets": {}}
        selected_fixed = None
        selected_cem_parts = []
        for set_index, (source, score, absolute) in enumerate(zip(raw_sources, score_sets, absolute_sets)):
            true_score, dual, harm = true_scores(absolute)
            valid = score["valid"].copy()
            valid[:, 0] = False
            group = "fixed" if set_index == 0 else f"cem_iteration_{set_index - 1}"
            bank_result["sets"][group] = {
                "all_valid": binary_summary(score["p_dual"][valid], score["p_harm"][valid], dual[valid], harm[valid]),
                "p_dual_bins": probability_bins(score["p_dual"][valid], score["p_harm"][valid], dual[valid], harm[valid]),
                "rank": within_state_rank(-score["energy"], true_score, score["valid"]),
            }
            selected = selected_vectors(score, dual, harm, "safe")
            summary = binary_summary(selected["p_dual"], selected["p_harm"], selected["dual"], selected["harm"])
            summary["route_clustered_ci95"] = route_bootstrap(route[selected["state"]], selected["dual"], selected["harm"], selected["p_dual"], 20260721 + bank_index * 10 + set_index)
            bank_result["sets"][group]["safe_top"] = summary
            selected["route"] = route[selected["state"]]
            if set_index == 0:
                selected_fixed = selected
                fixed_rank.append(bank_result["sets"][group]["rank"]["spearman_mean"])
            else:
                selected_cem_parts.append(selected)
                cem_rank.append(bank_result["sets"][group]["rank"]["spearman_mean"])
        if selected_fixed is None:
            raise RuntimeError("fixed selection missing")
        selected_cem = {key: np.concatenate([item[key] for item in selected_cem_parts]) for key in selected_cem_parts[0]}
        bank_result["fixed_safe_top"] = binary_summary(selected_fixed["p_dual"], selected_fixed["p_harm"], selected_fixed["dual"], selected_fixed["harm"])
        bank_result["cem_safe_top"] = binary_summary(selected_cem["p_dual"], selected_cem["p_harm"], selected_cem["dual"], selected_cem["harm"])
        calibration_banks[name] = bank_result
        pooled_fixed_selected.append(selected_fixed)
        pooled_cem_selected.append(selected_cem)

    def concatenate_selected(parts: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
        return {key: np.concatenate([item[key] for item in parts]) for key in parts[0]}
    fixed_selected = concatenate_selected(pooled_fixed_selected)
    cem_selected = concatenate_selected(pooled_cem_selected)
    fixed_summary = binary_summary(fixed_selected["p_dual"], fixed_selected["p_harm"], fixed_selected["dual"], fixed_selected["harm"])
    cem_summary = binary_summary(cem_selected["p_dual"], cem_selected["p_harm"], cem_selected["dual"], cem_selected["harm"])
    bank_b_hits = [
        name for name, value in calibration_banks.items()
        if value["cem_safe_top"].get("n", 0) and value["cem_safe_top"]["dual_overconfidence"] >= 0.08
    ]
    confidence_branch = (
        cem_summary["dual_overconfidence"] >= 0.08
        and len(bank_b_hits) >= 2
        and fixed_summary["empirical_dual"] - cem_summary["empirical_dual"] >= 0.03
    )
    valid_fixed_rank = np.asarray([value for value in fixed_rank if value is not None], dtype=np.float64)
    valid_cem_rank = np.asarray([value for value in cem_rank if value is not None], dtype=np.float64)
    rank_gap = float(valid_fixed_rank.mean() - valid_cem_rank.mean())
    rank_bank_hits = []
    for name, value in calibration_banks.items():
        fixed_value = value["sets"]["fixed"]["rank"]["spearman_mean"]
        cem_values = [value["sets"][f"cem_iteration_{it}"]["rank"]["spearman_mean"] for it in range(3)]
        if fixed_value is not None and all(item is not None for item in cem_values) and fixed_value - float(np.mean(cem_values)) >= 0.10:
            rank_bank_hits.append(name)
    rank_branch = rank_gap >= 0.10 and len(rank_bank_hits) >= 2
    root_b = confidence_branch or rank_branch
    decision = "A_set_context_instability" if root_a else ("B_proposal_tail_calibration" if root_b else "neither_stop_before_training")

    v61_path = ROOT / "outputs/cem_development_v1/recon_cem_fixed_harm_v61/crossfit/result.json"
    v61 = read_json(v61_path)
    result = {
        "schema": "cem_proposal_shift_v65_result",
        "status": "stage_A_complete",
        "decision": decision,
        "root_A": {
            "material": root_a,
            "pooled": pooled_context,
            "bank_hits": context_bank_hits,
            "per_bank": per_bank_context,
            "records": context_rows,
        },
        "root_B": {
            "material": root_b,
            "confidence_branch": confidence_branch,
            "rank_branch": rank_branch,
            "fixed_safe_top_pooled": fixed_summary,
            "cem_safe_top_pooled": cem_summary,
            "empirical_dual_drop_fixed_to_cem": float(fixed_summary["empirical_dual"] - cem_summary["empirical_dual"]),
            "mean_rank_gap_fixed_minus_cem": rank_gap,
            "overconfidence_bank_hits": bank_b_hits,
            "rank_bank_hits": rank_bank_hits,
            "per_bank": calibration_banks,
        },
        "authoritative_v61_separation": {
            "fixed_promotions": int(v61["frozen_fixed_promotions"]),
            "fixed_dual_successes": int(v61["signal_attribution"]["fixed_dual_successes"]),
            "fixed_harms": int(v61["signal_attribution"]["fixed_harms"]),
            "cem_net_added": int(v61["frozen_addonly_net_added"]),
            "cem_added_dual_successes": int(v61["cross_fitted_pooled"]["frozen_addonly"]["promotions"] - v61["signal_attribution"]["fixed_harms"] - v61["signal_attribution"]["fixed_dual_successes"] - (v61["cross_fitted_pooled"]["frozen_addonly"]["harm_count"] - v61["signal_attribution"]["fixed_harms"])),
            "cem_added_harms": int(v61["cross_fitted_pooled"]["frozen_addonly"]["harm_count"] - v61["signal_attribution"]["fixed_harms"]),
        },
        "stage_B_authorized": bool(root_a or root_b),
        "stage_B_branch": decision if root_a or root_b else None,
        "preregistration": str(preregistration),
        "preregistration_sha256": sha256(preregistration),
        "context_identities": str(identity_path),
        "context_identities_sha256": sha256(identity_path),
        "training_result": str(training_path),
        "training_result_sha256": sha256(training_path),
        "outcomes_are_spent": True,
        "new_candidates_generated": False,
        "model_trained": False,
        "thresholds_tuned": False,
        "reserve_accessed": False,
        "production_safe_claim_allowed": False,
    }
    write_json(result_path, result)
    print(json.dumps({
        "stage": "run",
        "decision": decision,
        "root_A": root_a,
        "root_B": root_b,
        "fixed_safe_top": fixed_summary,
        "cem_safe_top": cem_summary,
        "rank_gap": rank_gap,
        "output": str(result_path),
        "sha256": sha256(result_path),
    }, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "run"))
    parser.add_argument("--preregistration", type=Path, default=ROOT / "outputs/cem_development_v1/cem_proposal_shift_v65_preregistration.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/cem_development_v1/cem_proposal_shift_v65")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stage == "freeze":
        freeze_identities(args.preregistration, args.output_dir)
    else:
        run(args.preregistration, args.output_dir, args.device)


if __name__ == "__main__":
    main()
