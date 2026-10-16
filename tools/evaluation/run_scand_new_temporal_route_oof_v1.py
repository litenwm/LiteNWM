#!/usr/bin/env python3
"""Nested leave-one-route-out probe of a temporal/geometry SCAND risk head.

The proposal pool, formal-v70 fallback and all CEM/scorer artifacts are frozen.
Only label-free temporal/geometry sidecar features feed the risk head.  Inner
route-OOF predictions choose a frozen risk/coverage threshold; held-out routes
are used exactly once for the outer diagnostic.
"""

from __future__ import annotations

from collections import Counter
import argparse
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from planning.proposal_tail_sidecar_v1 import canonical_json, sha256_file  # noqa: E402
from tools.evaluation import run_scand_existing_train_risk_coverage_frontier_v1 as frontier  # noqa: E402
from tools.evaluation import run_scand_route_stable_risk_v1 as stable  # noqa: E402
from tools.evaluation import run_v77_scand_tail_joint_selector as v77  # noqa: E402
from tools.planning import build_proposal_tail_temporal_pair_v1 as pair  # noqa: E402


FEATURE_BLOCKS = (
    "action_structure_delta_to_formal_v70",
    "ltwm_temporal_summary_delta_to_formal_v70",
    "ltwm_temporal_order_statistics_delta_to_formal_v70",
    "ate_by_seed",
    "rtp_by_seed",
    "support_by_seed",
    "metric_mean",
    "metric_std",
    "dual_probability",
    "harm_probability",
    "support_score",
)
PREDICTION_KEYS = (
    "gain_mean", "gain_std", "p_dual_mean", "p_dual_std",
    "p_harm_mean", "p_harm_std",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--temporal-features", type=Path, required=True)
    parser.add_argument("--pilot-preregistration", type=Path, required=True)
    parser.add_argument("--formal-v70-result", type=Path, required=True)
    parser.add_argument("--recon-protection", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser.parse_args()


def load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {key: np.asarray(payload[key]) for key in payload.files}


def write_json_atomic(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(canonical_json(value) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_npz_atomic(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def input_paths(args: argparse.Namespace) -> dict[str, Path]:
    return {
        "collection_manifest": args.sidecar / "collection_manifest.json",
        "candidate_table": args.sidecar / "candidate_table.npz",
        "formal_snapshot": args.sidecar / "formal_selection_snapshot.npz",
        "labels": args.labels,
        "temporal_features": args.temporal_features,
        "pilot_preregistration": args.pilot_preregistration,
        "formal_v70_result": args.formal_v70_result,
        "recon_protection": args.recon_protection,
    }


def freeze(args: argparse.Namespace) -> None:
    prereg_path = args.output_dir / "preregistration.json"
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    states = [json.loads(line) for line in (args.sidecar / "states.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    routes = sorted({str(row["route"]) for row in states})
    if len(routes) != 4:
        raise RuntimeError(f"four development routes required, got {routes}")
    value = {
        "schema": "scand_new_temporal_route_oof_preregistration_v1",
        "status": "frozen_before_outer_route_predictions",
        "scope": "four-route development feasibility only; not independent validation",
        "routes": routes,
        "fold_rule": "sorted route name; one whole route per outer fold; inner leave-one-route-out over remaining three",
        "feature_blocks": list(FEATURE_BLOCKS),
        "state_transform": "within-state zscore plus percentile over physical+support candidates; no labels",
        "model": {
            "family": stable.MODEL_NAME,
            "seeds": list(stable.SEEDS),
            "epochs": stable.EPOCHS,
            "learning_rate": stable.LEARNING_RATE,
            "weight_decay": stable.WEIGHT_DECAY,
            "group_dro_eta": stable.DRO_ETA,
        },
        "risk_frontier": {
            "grid_fraction_of_states": list(frontier.GRID),
            "ensemble_sigma": frontier.SIGMA,
            "gain_floor_m": frontier.GAIN_FLOOR,
            "selection": "largest inner-route-OOF coverage point passing all frozen Wilson/benefit checks; otherwise formal-v70 fallback",
        },
        "success_gate": {
            "dual_candidate_auroc_min": 0.75,
            "harm_candidate_auroc_min": 0.75,
            "nonzero_risk_certified_replacements": True,
            "dual_wilson_lcb_min": 0.85,
            "harm_wilson_ucb_max": 0.15,
            "objective_relative_gain_min": 0.05,
            "ate_and_objective_noninferior": True,
            "recon_objective_degradation_max": 0.01,
        },
        "inputs": {name: {"path": str(path.resolve()), "sha256": sha256_file(path)} for name, path in input_paths(args).items()},
        "formal_v70_mutated": False,
        "reserve_accessed": False,
        "validation_labels_opened": False,
    }
    write_json_atomic(prereg_path, value)
    print(canonical_json({"status": value["status"], "preregistration": str(prereg_path), "sha256": sha256_file(prereg_path)}))


def verify(args: argparse.Namespace) -> dict:
    path = args.output_dir / "preregistration.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("status") != "frozen_before_outer_route_predictions":
        raise RuntimeError("OOF preregistration is not frozen")
    if tuple(value["feature_blocks"]) != FEATURE_BLOCKS:
        raise RuntimeError("feature blocks changed")
    if tuple(value["risk_frontier"]["grid_fraction_of_states"]) != frontier.GRID:
        raise RuntimeError("coverage grid changed")
    for name, item in value["inputs"].items():
        if sha256_file(Path(item["path"])) != item["sha256"]:
            raise RuntimeError(f"frozen input changed: {name}")
    return value


def assemble(args: argparse.Namespace) -> tuple[dict, np.ndarray, np.ndarray, list[str], dict]:
    table = load(args.sidecar / "candidate_table.npz")
    formal = load(args.sidecar / "formal_selection_snapshot.npz")
    labels = load(args.labels)
    temporal = load(args.temporal_features)
    if not np.array_equal(table["record_key"].astype(str), labels["record_key"].astype(str)):
        raise RuntimeError("label/table record mismatch")
    if not np.array_equal(table["candidate_hash"].astype(str), labels["candidate_hash"].astype(str)):
        raise RuntimeError("label/table candidate mismatch")
    if not np.array_equal(table["candidate_hash"].astype(str), temporal["candidate_hash"].astype(str)):
        raise RuntimeError("temporal/table candidate mismatch")
    formal_index = pair.formal_union_index(table, formal)
    if not np.array_equal(formal_index, temporal["formal_v70_union_index"].astype(np.int64)):
        raise RuntimeError("temporal formal mapping changed")
    pools = {source: load(args.sidecar / relative) for source, relative in pair.POOL_PATH.items()}
    physical, _ = pair.gather_union(table, pools, ("physical_mask",))
    support, _ = pair.gather_union(table, pools, ("support_mask",))
    pool = physical.astype(bool) & support.astype(bool)
    row = np.arange(len(formal_index))
    pool[row, formal_index] = True
    blocks = []
    names = []
    for name in FEATURE_BLOCKS:
        value = temporal[name].astype(np.float32)
        if value.shape[:2] != pool.shape:
            raise RuntimeError(f"bad feature shape {name}: {value.shape}")
        value = value.reshape(*pool.shape, -1)
        if not np.isfinite(value).all():
            raise RuntimeError(f"nonfinite feature block: {name}")
        blocks.append(value)
        names.extend(f"{name}_{index}" for index in range(value.shape[-1]))
    rounds = table["source_round"].astype(np.int64)
    source = np.eye(4, dtype=np.float32)[np.where(rounds < 0, 0, rounds + 1)]
    source_rank = table["source_candidate_index"].astype(np.float32)[..., None] / 16.0
    blocks.extend((source, source_rank))
    names.extend([f"proposal_source_{index}" for index in range(4)] + ["proposal_source_rank"])
    raw_feature = np.concatenate(blocks, axis=-1).astype(np.float32)
    feature = stable.state_normalize(raw_feature, pool, np.arange(raw_feature.shape[-1], dtype=np.int64))
    states = [json.loads(line) for line in (args.sidecar / "states.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    routes = np.asarray([str(item["route"]) for item in states])
    bags = np.asarray([str(item["bag"]) for item in states])
    unique_routes = sorted(np.unique(routes))
    fold = np.asarray([unique_routes.index(route) for route in routes], dtype=np.int64)
    data = {
        "record_key": table["record_key"].astype(str),
        "group": routes,
        "bag": bags,
        "formal_index": formal_index,
        "true_metrics": labels["true_metrics_m"].astype(np.float32),
    }
    audit = {
        "raw_feature_dimension": raw_feature.shape[-1],
        "state_normalized_dimension": feature.shape[-1],
        "feature_blocks": {name: list(temporal[name].shape[2:]) for name in FEATURE_BLOCKS},
        "uses_truth": False,
        "uses_route_bag_record_id_as_feature": False,
        "physical_support_candidates": int(pool.sum()),
        "routes": unique_routes,
    }
    return data, pool, feature, names, {"fold": fold, **audit}


def route_bootstrap(data: dict, selected: np.ndarray, repetitions: int = 10000) -> dict:
    row = np.arange(len(selected))
    delta = data["true_metrics"][row, selected, 3] - data["true_metrics"][row, data["formal_index"], 3]
    routes = np.unique(data["group"])
    local = [delta[data["group"] == route] for route in routes]
    rng = np.random.default_rng(2026072302)
    values = np.empty(repetitions)
    for index in range(repetitions):
        pick = rng.integers(0, len(routes), size=len(routes))
        values[index] = np.concatenate([local[item] for item in pick]).mean()
    return {"selected_minus_formal_m": float(delta.mean()), "route_cluster_bootstrap95_m": [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]}


def run(args: argparse.Namespace) -> None:
    prereg = verify(args)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    result_path = args.output_dir / "result.json"
    audit_path = args.output_dir / "audit.json"
    prediction_path = args.output_dir / "outer_oof_predictions.npz"
    parameter_path = args.output_dir / "model_parameters.npz"
    hash_path = args.output_dir / "sha256_manifest.json"
    for path in (result_path, audit_path, prediction_path, parameter_path, hash_path):
        if path.exists():
            raise FileExistsError(path)
    started = time.perf_counter()
    data, pool, feature, feature_names, feature_audit = assemble(args)
    fold = feature_audit.pop("fold")
    routes = sorted(np.unique(data["group"]))
    row = np.arange(len(fold))
    challenger = pool.copy()
    challenger[row, data["formal_index"]] = False
    gain = data["true_metrics"][row, data["formal_index"]][:, None] - data["true_metrics"]
    objective_gain = gain[..., 3].astype(np.float32)
    dual = (gain[..., 0] > v77.TOLERANCE) & (gain[..., 3] > v77.TOLERANCE)
    harm = (gain[..., 0] < -v77.TOLERANCE) | (gain[..., 3] < -v77.TOLERANCE)
    prediction = {key: np.full(pool.shape, np.nan, dtype=np.float32) for key in PREDICTION_KEYS}
    nested_selected = data["formal_index"].copy()
    nested_target = np.full(len(routes), np.nan, dtype=np.float32)
    nested_threshold = np.full(len(routes), np.nan, dtype=np.float32)
    parameters: dict[str, np.ndarray] = {}
    fold_audit = {}
    latency = []
    for outer in range(len(routes)):
        outer_train = np.flatnonzero(fold != outer)
        outer_test = np.flatnonzero(fold == outer)
        inner_prediction = {key: np.full(pool.shape, np.nan, dtype=np.float32) for key in PREDICTION_KEYS}
        inner_records = []
        for inner in range(len(routes)):
            if inner == outer:
                continue
            inner_train = np.flatnonzero((fold != outer) & (fold != inner))
            inner_test = np.flatnonzero(fold == inner)
            if set(data["group"][inner_train]) & set(data["group"][inner_test]):
                raise RuntimeError("inner route leakage")
            if set(data["bag"][inner_train]) & set(data["bag"][inner_test]):
                raise RuntimeError("inner bag leakage")
            models = stable.train_ensemble(feature, challenger, objective_gain, dual, harm, inner_train, data["group"], args.device)
            local = stable.predict_ensemble(models, feature[inner_test], args.device)
            for key, value in local.items():
                inner_prediction[key][inner_test] = value
            del models
            if args.device == "cuda":
                torch.cuda.empty_cache()
            inner_records.append({"held_out_route": routes[inner], "train_states": int(len(inner_train)), "test_states": int(len(inner_test)), "route_overlap": 0, "bag_overlap": 0})
        local_prediction = {key: value[outer_train] for key, value in inner_prediction.items()}
        choice = frontier.state_choice(local_prediction, pool[outer_train], data["formal_index"][outer_train])
        points = [frontier.point_by_count(data, choice[0], choice[1], outer_train, fraction)[0] for fraction in frontier.GRID]
        feasible = [(point["target_coverage"], point["threshold"]) for point in points if point["passes_all"]]
        chosen = max(feasible, key=lambda item: item[0]) if feasible else None
        if chosen is not None:
            nested_target[outer], nested_threshold[outer] = chosen
        models = stable.train_ensemble(feature, challenger, objective_gain, dual, harm, outer_train, data["group"], args.device)
        local = stable.predict_ensemble(models, feature[outer_test], args.device)
        for key, value in local.items():
            prediction[key][outer_test] = value
        outer_choice = frontier.state_choice(local, pool[outer_test], data["formal_index"][outer_test])
        selected, outer_summary = frontier.apply_threshold(data, outer_choice[0], outer_choice[1], outer_test, None if chosen is None else chosen[1])
        nested_selected[outer_test] = selected
        stable.serialize_models(f"outer{outer}", models, parameters)
        latency.append(stable.cpu_latency(models, feature[outer_test]))
        del models
        if args.device == "cuda":
            torch.cuda.empty_cache()
        fold_audit[str(outer)] = {"held_out_route": routes[outer], "outer_train_states": int(len(outer_train)), "outer_test_states": int(len(outer_test)), "inner": inner_records, "inner_points": points, "chosen": chosen, "outer_selection": outer_summary}
        print(f"temporal-route-oof outer={outer} route={routes[outer]} chosen={chosen}", flush=True)

    candidate = v77.candidate_metrics(prediction, objective_gain, dual, harm, challenger)
    nested = v77.evaluate_selection(data, nested_selected)
    checks = {
        "dual_candidate_auroc_min_075": candidate["dual"]["auroc"] >= 0.75,
        "harm_candidate_auroc_min_075": candidate["harm"]["auroc"] >= 0.75,
        **frontier.summary_checks(nested),
        "relative_objective_gain_5pct": (-nested["delta_mean_m"]["objective"] / max(nested["formal_metrics"]["mean_m"]["objective"], 1e-12)) >= 0.05,
        "recon_unchanged_by_scand_only_routing": sha256_file(args.recon_protection) == prereg["inputs"]["recon_protection"]["sha256"],
    }
    replacement = nested_selected != data["formal_index"]
    route_counts = dict(sorted(Counter(data["group"][replacement].astype(str)).items()))
    checks["replacement_spans_at_least_three_routes"] = len(route_counts) >= 3
    passed = all(checks.values())
    posthoc_choice = frontier.state_choice(prediction, pool, data["formal_index"])
    posthoc = [frontier.point_by_count(data, posthoc_choice[0], posthoc_choice[1], row, fraction)[0] for fraction in frontier.GRID]
    result = {
        "schema": "scand_new_temporal_route_oof_result_v1",
        "status": "completed_development_nested_route_oof",
        "verdict": "temporal_route_stable_development_candidate" if passed else "temporal_route_stable_fail_stop",
        "candidate_layer": candidate,
        "nested_route_oof": nested,
        "posthoc_frontier_diagnostic_only": posthoc,
        "folds_with_inner_certified_choice": int(np.isfinite(nested_target).sum()),
        "nested_fold_target": nested_target.tolist(),
        "nested_fold_threshold": nested_threshold.tolist(),
        "replacement_routes": route_counts,
        "route_cluster_objective": route_bootstrap(data, nested_selected),
        "success_gate": {"checks": checks, "passed": passed},
        "cpu_latency": {"mean_seconds_per_state": float(np.mean(latency)), "p90_fold_seconds_per_state": float(np.quantile(latency, 0.9)), "fold_values": latency},
        "runtime_seconds": float(time.perf_counter() - started),
        "development_only": True,
        "promotion_allowed": False,
        "formal_v70_mutated": False,
        "reserve_accessed": False,
        "validation_labels_opened": False,
    }
    write_npz_atomic(prediction_path, record_key=data["record_key"].astype("U256"), route=data["group"].astype("U256"), outer_fold=fold.astype(np.int16), formal_index=data["formal_index"].astype(np.int16), nested_selected=nested_selected.astype(np.int16), nested_fold_target=nested_target, nested_fold_threshold=nested_threshold, **prediction)
    write_npz_atomic(parameter_path, **parameters)
    audit = {"schema": "scand_new_temporal_route_oof_audit_v1", "preregistration_sha256": sha256_file(args.output_dir / "preregistration.json"), "feature_contract": feature_audit, "feature_names": feature_names, "fold_audit": fold_audit, "route_overlap_violations": 0, "bag_overlap_violations": 0, "outer_labels_used_for_threshold_choice": False, "formal_v70_sha256": sha256_file(args.formal_v70_result), "recon_protection_sha256": sha256_file(args.recon_protection), "reserve_accessed": False, "runner_sha256": sha256_file(Path(__file__))}
    write_json_atomic(result_path, result)
    write_json_atomic(audit_path, audit)
    write_json_atomic(hash_path, {"schema": "scand_new_temporal_route_oof_sha256_manifest_v1", "artifacts": {str(path.resolve()): {"sha256": sha256_file(path), "bytes": path.stat().st_size} for path in (args.output_dir / "preregistration.json", result_path, audit_path, prediction_path, parameter_path, Path(__file__))}})
    print(canonical_json({"verdict": result["verdict"], "candidate_layer": candidate, "nested": nested, "checks": checks, "runtime_seconds": result["runtime_seconds"]}))


def main() -> None:
    args = parse_args()
    if args.freeze_only:
        freeze(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
