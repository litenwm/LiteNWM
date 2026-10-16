#!/usr/bin/env python3
"""Post-hoc and strict nested risk/coverage diagnostics for the SCAND pilot."""

from __future__ import annotations

from collections import Counter
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from tools.evaluation import run_scand_existing_train_upperbound_pilot_v1 as pilot  # noqa: E402
from tools.evaluation import run_v77_scand_tail_joint_selector as v77  # noqa: E402


OUTPUT = pilot.BASE / "risk_coverage_frontier_v1"
PREREG = OUTPUT / "preregistration.json"
RESULT = OUTPUT / "result.json"
AUDIT = OUTPUT / "audit.json"
PREDICTIONS = OUTPUT / "nested_predictions.npz"
HASHES = OUTPUT / "sha256_manifest.json"
GRID = (0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20, 0.25, 0.35)
SIGMA = 1.0
GAIN_FLOOR = 0.002


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser.parse_args()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {key: payload[key].copy() for key in payload.files}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(jsonable(value), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def write_npz(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def verify_preregistration() -> tuple[dict[str, Any], dict[str, str]]:
    prereg = read_json(PREREG)
    if prereg.get("status") != "frozen_before_frontier_computation":
        raise RuntimeError("frontier preregistration is not frozen")
    if tuple(prereg["coverage_grid_fraction_of_all_states"]) != GRID:
        raise RuntimeError("coverage grid changed")
    if float(prereg["risk_score"]["ensemble_sigma"]) != SIGMA:
        raise RuntimeError("risk sigma changed")
    if float(prereg["risk_score"]["gain_floor_m"]) != GAIN_FLOOR:
        raise RuntimeError("gain floor changed")
    hashes = {}
    for name, item in prereg["inputs"].items():
        path = Path(item["path"])
        if "reserve" in str(path).lower():
            raise RuntimeError(f"forbidden input path: {name}")
        digest = sha256_file(path)
        if digest != item["sha256"]:
            raise RuntimeError(f"frozen input changed: {name}")
        hashes[name] = digest
    return prereg, hashes


def model_prefix(model: str) -> str:
    return "linear" if model == v77.MODELS[0] else "listwise"


def saved_predictions(payload: dict[str, np.ndarray], model: str) -> dict[str, np.ndarray]:
    prefix = model_prefix(model)
    return {
        key: payload[f"{prefix}__{key}"].astype(np.float32)
        for key in (
            "gain_mean", "gain_std", "p_dual_mean", "p_dual_std",
            "p_harm_mean", "p_harm_std",
        )
    }


def state_choice(
    prediction: dict[str, np.ndarray], pool: np.ndarray, formal: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gain_lcb = prediction["gain_mean"] - SIGMA * prediction["gain_std"]
    dual_lcb = prediction["p_dual_mean"] - SIGMA * prediction["p_dual_std"]
    harm_ucb = prediction["p_harm_mean"] + SIGMA * prediction["p_harm_std"]
    confidence = np.minimum(dual_lcb, 1.0 - harm_ucb)
    eligible = pool & (gain_lcb >= GAIN_FLOOR)
    eligible[np.arange(len(formal)), formal] = False
    selected = formal.copy()
    state_confidence = np.full(len(formal), -np.inf, dtype=np.float32)
    state_gain_lcb = np.full(len(formal), -np.inf, dtype=np.float32)
    for state in range(len(formal)):
        index = np.flatnonzero(eligible[state])
        if not len(index):
            continue
        order = np.lexsort((index, -gain_lcb[state, index], -confidence[state, index]))
        chosen = int(index[order[0]])
        selected[state] = chosen
        state_confidence[state] = confidence[state, chosen]
        state_gain_lcb[state] = gain_lcb[state, chosen]
    return selected, state_confidence, state_gain_lcb


def summary_checks(summary: dict[str, Any]) -> dict[str, bool]:
    return {
        "nonzero": summary["replacements"] > 0,
        "dual_wilson_lcb_85": summary["strict_zero_mm"]["dual"]["lcb"] >= 0.85,
        "harm_wilson_ucb_15": summary["strict_zero_mm"]["harm"]["ucb"] <= 0.15,
        "objective_improves": summary["delta_mean_m"]["objective"] < 0.0,
        "ate_improves": summary["delta_mean_m"]["ate"] < 0.0,
    }


def point_by_count(
    data: dict[str, Any], candidate: np.ndarray, confidence: np.ndarray,
    states: np.ndarray, fraction: float,
) -> tuple[dict[str, Any], float | None, np.ndarray]:
    finite = np.flatnonzero(np.isfinite(confidence))
    requested = int(math.ceil(fraction * len(states)))
    count = min(requested, len(finite))
    selected = data["formal_index"][states].copy()
    threshold = None
    if count:
        order = finite[np.argsort(-confidence[finite], kind="stable")]
        keep = order[:count]
        selected[keep] = candidate[keep]
        threshold = float(confidence[keep[-1]])
    summary = v77.evaluate_selection(data, selected, states)
    checks = summary_checks(summary)
    empirical_only = (
        summary["replacements"] > 0
        and summary["strict_zero_mm"]["dual"]["rate"] is not None
        and summary["strict_zero_mm"]["dual"]["rate"] >= 0.85
        and summary["strict_zero_mm"]["harm"]["rate"] <= 0.15
        and not (checks["dual_wilson_lcb_85"] and checks["harm_wilson_ucb_15"])
    )
    return {
        "target_coverage": fraction,
        "requested_states": requested,
        "threshold": threshold,
        "selection": summary,
        "checks": checks,
        "passes_all": all(checks.values()),
        "empirical_high_precision_but_wilson_insufficient": empirical_only,
    }, threshold, selected


def apply_threshold(
    data: dict[str, Any], candidate: np.ndarray, confidence: np.ndarray,
    states: np.ndarray, threshold: float | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    selected = data["formal_index"][states].copy()
    if threshold is not None:
        replace = np.isfinite(confidence) & (confidence >= threshold)
        selected[replace] = candidate[replace]
    return selected, v77.evaluate_selection(data, selected, states)


def replacement_routes(data: dict[str, Any], selected: np.ndarray) -> dict[str, int]:
    replacement = selected != data["formal_index"]
    return dict(sorted(Counter(data["group"][replacement].astype(str)).items()))


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    for path in (RESULT, AUDIT, PREDICTIONS, HASHES):
        if path.exists():
            raise FileExistsError(path)
    prereg, hashes_before = verify_preregistration()
    started = time.perf_counter()
    data, fields, feature, feature_names, pool, feature_audit = pilot.assemble()
    saved = load_npz(pilot.BASE / "oof_upperbound_v1/outer_oof_predictions.npz")
    fold = saved["outer_fold"].astype(np.int64)
    if not np.array_equal(saved["formal_v70_index"].astype(np.int64), data["formal_index"]):
        raise RuntimeError("saved formal index changed")
    if not np.array_equal(saved["physical_support_pool"], pool):
        raise RuntimeError("saved physical/support pool changed")
    row = np.arange(len(data["formal_index"]))
    gain = data["true_metrics"][row, data["formal_index"]][:, None] - data["true_metrics"]
    objective_gain = gain[..., 3].astype(np.float32)
    dual = (gain[..., 0] > v77.TOLERANCE) & (gain[..., 3] > v77.TOLERANCE)
    harm = (gain[..., 0] < -v77.TOLERANCE) | (gain[..., 3] < -v77.TOLERANCE)

    posthoc: dict[str, Any] = {}
    posthoc_choice: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    all_states = np.arange(len(row))
    for model in v77.MODELS:
        prediction = saved_predictions(saved, model)
        choice = state_choice(prediction, pool, data["formal_index"])
        posthoc_choice[model] = choice
        points = [point_by_count(data, choice[0], choice[1], all_states, value)[0] for value in GRID]
        posthoc[model] = {
            "role": "posthoc_diagnostic_only",
            "finite_state_candidates": int(np.isfinite(choice[1]).sum()),
            "points": points,
        }

    nested_selected = {model: data["formal_index"].copy() for model in v77.MODELS}
    nested_threshold = {model: np.full(5, np.nan, dtype=np.float32) for model in v77.MODELS}
    nested_target = {model: np.full(5, np.nan, dtype=np.float32) for model in v77.MODELS}
    inner_audit: dict[str, Any] = {}
    for outer in range(5):
        outer_train = np.flatnonzero(fold != outer)
        outer_test = np.flatnonzero(fold == outer)
        inner_prediction = {
            model: {
                key: np.full(pool.shape, np.nan, dtype=np.float32)
                for key in (
                    "gain_mean", "gain_std", "p_dual_mean", "p_dual_std",
                    "p_harm_mean", "p_harm_std",
                )
            }
            for model in v77.MODELS
        }
        fold_records = []
        for inner in range(5):
            if inner == outer:
                continue
            inner_train = np.flatnonzero((fold != outer) & (fold != inner))
            inner_test = np.flatnonzero(fold == inner)
            if set(data["group"][inner_train]) & set(data["group"][inner_test]):
                raise RuntimeError("inner route leakage")
            if set(data["bag"][inner_train]) & set(data["bag"][inner_test]):
                raise RuntimeError("inner bag leakage")
            for model in v77.MODELS:
                ensemble = v77.fit_ensemble(
                    model, feature, pool, objective_gain, dual, harm,
                    inner_train, data["group"], args.device,
                )
                prediction = v77.predict_ensemble(model, ensemble, feature[inner_test], args.device)
                for key, value in prediction.items():
                    inner_prediction[model][key][inner_test] = value
                del ensemble
                if args.device == "cuda":
                    torch.cuda.empty_cache()
            fold_records.append({
                "inner_fold": inner,
                "train_states": int(len(inner_train)),
                "test_states": int(len(inner_test)),
                "route_overlap": 0,
                "bag_overlap": 0,
            })
        model_records = {}
        for model in v77.MODELS:
            inner_local = {key: value[outer_train] for key, value in inner_prediction[model].items()}
            inner_choice = state_choice(
                inner_local, pool[outer_train], data["formal_index"][outer_train]
            )
            points = []
            feasible = []
            for value in GRID:
                point, threshold, _ = point_by_count(
                    data, inner_choice[0], inner_choice[1], outer_train, value
                )
                points.append(point)
                if point["passes_all"]:
                    feasible.append((value, threshold))
            chosen = max(feasible, key=lambda item: item[0]) if feasible else None
            threshold = chosen[1] if chosen is not None else None
            if chosen is not None:
                nested_threshold[model][outer] = threshold
                nested_target[model][outer] = chosen[0]
            outer_prediction = {
                key: value[outer_test]
                for key, value in saved_predictions(saved, model).items()
            }
            outer_choice = state_choice(
                outer_prediction, pool[outer_test], data["formal_index"][outer_test]
            )
            selected, outer_summary = apply_threshold(
                data, outer_choice[0], outer_choice[1], outer_test, threshold
            )
            nested_selected[model][outer_test] = selected
            model_records[model] = {
                "inner_points": points,
                "chosen_target_coverage": chosen[0] if chosen is not None else None,
                "chosen_threshold": threshold,
                "outer_selection": outer_summary,
            }
        inner_audit[str(outer)] = {
            "outer_train_states": int(len(outer_train)),
            "outer_test_states": int(len(outer_test)),
            "outer_route_overlap": 0,
            "outer_bag_overlap": 0,
            "inner_folds": fold_records,
            "models": model_records,
        }
        print(f"frontier outer={outer} complete", flush=True)

    nested = {}
    stable_models = []
    for model in v77.MODELS:
        summary = v77.evaluate_selection(data, nested_selected[model])
        routes = replacement_routes(data, nested_selected[model])
        checks = summary_checks(summary)
        checks["at_least_three_routes"] = len(routes) >= 3
        stable = all(checks.values())
        if stable:
            stable_models.append(model)
        nested[model] = {
            "selection": summary,
            "checks": checks,
            "stable_high_precision_region": stable,
            "replacement_routes": routes,
            "fold_target_coverage": nested_target[model],
            "fold_threshold": nested_threshold[model],
            "folds_with_inner_choice": int(np.isfinite(nested_target[model]).sum()),
        }

    posthoc_passing = {}
    for model in v77.MODELS:
        passing = [
            point for point in posthoc[model]["points"]
            if point["target_coverage"] >= 0.05 and point["passes_all"]
        ]
        posthoc_passing[model] = [point["target_coverage"] for point in passing]
    any_posthoc = any(posthoc_passing.values())
    if stable_models:
        verdict = "stable_high_precision_region_development_diagnostic"
    elif any_posthoc:
        verdict = "posthoc_only_region_not_stable_across_nested_routes"
    else:
        verdict = "risk_representation_insufficient_stop_threshold_tuning"
    result = {
        "schema": "scand_existing_train_risk_coverage_frontier_result_v1",
        "status": "completed_spent_development_diagnostic_stopped",
        "verdict": verdict,
        "posthoc_frontier": posthoc,
        "posthoc_passing_coverage_at_least_5pct": posthoc_passing,
        "nested_route_oof": nested,
        "stable_models": stable_models,
        "next_step": (
            "freeze strategy and perform one-time development validation on unused old-train bags"
            if stable_models else
            "do not validate thresholds; move to route-stable risk representation or independent proposal-tail data"
        ),
        "development_only": True,
        "promotion_allowed": False,
        "reserve_accessed": False,
        "unused_old_train_bags_accessed": False,
        "formal_v70_mutated": False,
        "runtime_seconds": float(time.perf_counter() - started),
    }
    arrays = {
        "record_key": data["record_key"].astype("U256"),
        "route": data["group"].astype("U256"),
        "bag": data["bag"].astype("U256"),
        "outer_fold": fold.astype(np.int16),
        "formal_index": data["formal_index"].astype(np.int16),
    }
    for model in v77.MODELS:
        prefix = model_prefix(model)
        arrays[f"{prefix}__posthoc_candidate"] = posthoc_choice[model][0].astype(np.int16)
        arrays[f"{prefix}__posthoc_confidence"] = posthoc_choice[model][1]
        arrays[f"{prefix}__posthoc_gain_lcb"] = posthoc_choice[model][2]
        arrays[f"{prefix}__nested_selected"] = nested_selected[model].astype(np.int16)
        arrays[f"{prefix}__nested_fold_threshold"] = nested_threshold[model]
        arrays[f"{prefix}__nested_fold_target"] = nested_target[model]
    write_npz(PREDICTIONS, **arrays)
    hashes_after = {
        name: sha256_file(Path(item["path"])) for name, item in prereg["inputs"].items()
    }
    if hashes_after != hashes_before:
        raise RuntimeError("frozen diagnostic input changed")
    audit = {
        "schema": "scand_existing_train_risk_coverage_frontier_audit_v1",
        "preregistration_sha256": sha256_file(PREREG),
        "input_hashes_before": hashes_before,
        "input_hashes_after": hashes_after,
        "all_inputs_unchanged": True,
        "route_overlap_violations": 0,
        "bag_overlap_violations": 0,
        "inner_outer_audit": inner_audit,
        "feature_contract": feature_audit,
        "feature_count": len(feature_names),
        "model_definition_changed": False,
        "threshold_grid_changed_after_results": False,
        "posthoc_never_used_for_nested_choice": True,
        "formal_v70_sha256": sha256_file(pilot.FORMAL_RESULT),
        "reserve_accessed": False,
        "runner_sha256": sha256_file(Path(__file__)),
    }
    write_json(RESULT, result)
    write_json(AUDIT, audit)
    write_json(HASHES, {
        "schema": "scand_existing_train_risk_coverage_frontier_sha256_manifest_v1",
        "artifacts": {
            str(path.relative_to(ROOT)): {"sha256": sha256_file(path), "bytes": path.stat().st_size}
            for path in (PREREG, RESULT, AUDIT, PREDICTIONS, Path(__file__))
        },
    })
    print(json.dumps(jsonable({
        "verdict": verdict,
        "posthoc_passing": posthoc_passing,
        "nested": nested,
        "runtime_seconds": result["runtime_seconds"],
    }), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
