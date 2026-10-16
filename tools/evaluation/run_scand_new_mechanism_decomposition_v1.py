#!/usr/bin/env python3
"""Adaptive development-only mechanism decomposition for frozen SCAND CEM data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from planning.proposal_tail_sidecar_v1 import canonical_json, sha256_file  # noqa: E402
from tools.planning import build_proposal_tail_temporal_pair_v1 as pair  # noqa: E402


TOLERANCE = 1.0e-7
SCALARS = ("action_amplitude", "curvature", "seed_disagreement", "temporal_residual", "support_margin")
QUANTILES = tuple(index / 10.0 for index in range(11))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--temporal-features", type=Path, required=True)
    parser.add_argument("--pilot-preregistration", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--freeze-only", action="store_true")
    return parser.parse_args()


def load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {key: np.asarray(payload[key]) for key in payload.files}


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(canonical_json(value) + "\n", encoding="utf-8")
    temporary.replace(path)


def metric_values_waypoint_units(candidates: np.ndarray, ground: np.ndarray) -> np.ndarray:
    value = np.asarray(candidates, dtype=np.float64) * 0.25
    truth = np.asarray(ground, dtype=np.float64)
    error = value - truth
    squared = np.sum(error * error, axis=-1)
    ate = np.sqrt(np.mean(squared, axis=-1))
    endpoint = np.sqrt(squared[..., -1])
    relative = np.diff(value, axis=-2) - np.diff(truth, axis=-2)
    rpe = np.sqrt(np.mean(np.sum(relative * relative, axis=-1), axis=-1))
    objective = 0.4 * ate + 0.6 * endpoint
    return np.stack((ate, rpe, endpoint, objective), axis=-1).astype(np.float32)


def load_core(args: argparse.Namespace):
    table = load(args.sidecar / "candidate_table.npz")
    formal = load(args.sidecar / "formal_selection_snapshot.npz")
    labels = load(args.labels)
    temporal = load(args.temporal_features)
    if not np.array_equal(table["candidate_hash"].astype(str), labels["candidate_hash"].astype(str)):
        raise RuntimeError("label candidate binding mismatch")
    if not np.array_equal(table["candidate_hash"].astype(str), temporal["candidate_hash"].astype(str)):
        raise RuntimeError("temporal candidate binding mismatch")
    pools = {source: load(args.sidecar / relative) for source, relative in pair.POOL_PATH.items()}
    physical, _ = pair.gather_union(table, pools, ("physical_mask",))
    support, _ = pair.gather_union(table, pools, ("support_mask",))
    safe, _ = pair.gather_union(table, pools, ("safe",))
    energy, _ = pair.gather_union(table, pools, ("energy",))
    formal_index = pair.formal_union_index(table, formal)
    rounds = table["source_round"].astype(np.int64)
    local = table["source_candidate_index"].astype(np.int64)
    incumbent = np.empty(len(rounds), dtype=np.int64)
    for state in range(len(rounds)):
        match = np.flatnonzero((rounds[state] == -1) & (local[state] == 0))
        if len(match) != 1:
            raise RuntimeError("incumbent mapping failure")
        incumbent[state] = match[0]
    states = [json.loads(line) for line in (args.sidecar / "states.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    route = np.asarray([str(item["route"]) for item in states])
    action = temporal["actions_h4_m_rad"].astype(np.float32)
    speed = np.linalg.norm(action[..., :2], axis=-1)
    scalar = {
        "action_amplitude": speed.max(axis=-1),
        "curvature": (np.abs(action[..., 2]) / (speed + 0.05)).max(axis=-1),
        "seed_disagreement": np.concatenate((temporal["ate_by_seed"], temporal["rtp_by_seed"], temporal["support_by_seed"]), axis=-1).std(axis=-1),
        "temporal_residual": np.sqrt(np.mean(np.square(temporal["ltwm_temporal_summary_delta_to_formal_v70"]), axis=(-3, -2, -1))),
        "support_margin": temporal["support_margin"].reshape(len(rounds), 65),
    }
    if any(not np.isfinite(value).all() for value in scalar.values()):
        raise RuntimeError("nonfinite mechanism scalar")
    return table, formal, labels, temporal, physical.astype(bool), support.astype(bool), safe.astype(bool), energy, formal_index, incumbent, route, scalar


def freeze(args: argparse.Namespace) -> None:
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table, formal, labels, temporal, physical, support, safe, energy, formal_index, incumbent, route, scalar = load_core(args)
    del formal, labels, temporal, safe, energy, formal_index, incumbent, route
    pool = physical & support
    bins = {name: np.quantile(value[pool], QUANTILES).tolist() for name, value in scalar.items()}
    prereg = {
        "schema": "scand_new_mechanism_decomposition_preregistration_v1",
        "status": "frozen_before_stratified_outcome_analysis",
        "role": "adaptive_diagnostic_only_not_for_promotion",
        "ordered_funnel": ["k17", "raw_draw_all", "raw_draw_physical", "raw_draw_physical_support", "accepted_population", "union65_physical_support", "old_safety", "scorer_archive_v70"],
        "references": ["candidate0_incumbent", "formal_v70"],
        "stratification_scalars": list(SCALARS),
        "quantile_grid": list(QUANTILES),
        "label_free_bin_edges": bins,
        "no_profile_or_threshold_selection": True,
        "inputs": {
            "collection_manifest": {"path": str((args.sidecar / "collection_manifest.json").resolve()), "sha256": sha256_file(args.sidecar / "collection_manifest.json")},
            "candidate_table": {"path": str((args.sidecar / "candidate_table.npz").resolve()), "sha256": sha256_file(args.sidecar / "candidate_table.npz")},
            "formal_snapshot": {"path": str((args.sidecar / "formal_selection_snapshot.npz").resolve()), "sha256": sha256_file(args.sidecar / "formal_selection_snapshot.npz")},
            "labels": {"path": str(args.labels.resolve()), "sha256": sha256_file(args.labels)},
            "temporal_features": {"path": str(args.temporal_features.resolve()), "sha256": sha256_file(args.temporal_features)},
            "pilot_preregistration": {"path": str(args.pilot_preregistration.resolve()), "sha256": sha256_file(args.pilot_preregistration)},
        },
        "reserve_accessed": False,
        "validation_labels_opened": False,
    }
    path = args.output_dir / "preregistration.json"
    write_json(path, prereg)
    print(canonical_json({"status": prereg["status"], "path": str(path), "sha256": sha256_file(path)}))


def wilson(successes: int, total: int) -> list[float | None]:
    if total == 0:
        return [None, None]
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [float(center - radius), float(center + radius)]


def candidate_stats(metrics: np.ndarray, mask: np.ndarray, reference_metrics: np.ndarray) -> dict:
    gain = reference_metrics[:, None] - metrics
    dual = (gain[..., 0] > TOLERANCE) & (gain[..., 3] > TOLERANCE)
    harm = (gain[..., 0] < -TOLERANCE) | (gain[..., 3] < -TOLERANCE)
    count = int(mask.sum())
    dual_count = int((mask & dual).sum())
    harm_count = int((mask & harm).sum())
    return {
        "candidates": count,
        "states_with_any": int(mask.any(axis=1).sum()),
        "dual_candidates": dual_count,
        "dual_rate": None if count == 0 else dual_count / count,
        "dual_wilson95": wilson(dual_count, count),
        "harm_candidates": harm_count,
        "harm_rate": None if count == 0 else harm_count / count,
        "harm_wilson95": wilson(harm_count, count),
        "states_with_dual_opportunity": int((mask & dual).any(axis=1).sum()),
    }


def oracle_stats(metrics: np.ndarray, mask: np.ndarray, reference_index: np.ndarray, reference_metrics: np.ndarray) -> dict:
    local = mask.copy()
    local[np.arange(len(local)), reference_index] = True
    selected = np.argmin(np.where(local, metrics[..., 3], np.inf), axis=1)
    chosen = metrics[np.arange(len(selected)), selected]
    gain = reference_metrics - chosen
    return {
        "objective_mean_m": float(chosen[:, 3].mean()),
        "objective_p90_m": float(np.quantile(chosen[:, 3], 0.9)),
        "objective_gain_vs_reference_m": float(gain[:, 3].mean()),
        "ate_gain_vs_reference_m": float(gain[:, 0].mean()),
        "states_changed": int((selected != reference_index).sum()),
    }


def run(args: argparse.Namespace) -> None:
    prereg_path = args.output_dir / "preregistration.json"
    prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
    if prereg.get("status") != "frozen_before_stratified_outcome_analysis":
        raise RuntimeError("mechanism preregistration not frozen")
    for item in prereg["inputs"].values():
        if sha256_file(Path(item["path"])) != item["sha256"]:
            raise RuntimeError("frozen mechanism input changed")
    result_path = args.output_dir / "result.json"
    audit_path = args.output_dir / "audit.json"
    if result_path.exists() or audit_path.exists():
        raise FileExistsError(result_path)
    table, formal, labels, temporal, physical, support, safe, energy, formal_index, incumbent, route, scalar = load_core(args)
    metrics = labels["true_metrics_m"].astype(np.float32)
    row = np.arange(len(metrics))
    incumbent_metrics = metrics[row, incumbent]
    formal_metrics = metrics[row, formal_index]
    rounds = table["source_round"].astype(np.int64)
    fixed = rounds == -1
    union_support = physical & support
    raw_parts = []
    for iteration in range(3):
        raw = load(args.sidecar / f"iteration_{iteration:02d}/raw_draws.npz")
        state = raw["state_index"].astype(np.int64)
        raw_metrics = metric_values_waypoint_units(raw["draw_clipped"], labels["ground_truth_local_m"][state])
        raw_parts.append({**raw, "metrics": raw_metrics, "iteration": np.full(len(state), iteration, dtype=np.int8)})
    raw_state = np.concatenate([item["state_index"] for item in raw_parts]).astype(np.int64)
    raw_metrics_flat = np.concatenate([item["metrics"] for item in raw_parts])
    raw_masks_flat = {
        "raw_draw_all": np.ones(len(raw_state), dtype=bool),
        "raw_draw_physical": np.concatenate([item["physical_guard"] for item in raw_parts]).astype(bool),
        "raw_draw_physical_support": np.concatenate([item["physical_guard"] & item["support_mask"] for item in raw_parts]).astype(bool),
        "accepted_population": np.concatenate([item["final_candidate_index"] >= 0 for item in raw_parts]),
    }
    max_per_state = int(max(np.sum(raw_state == state) for state in range(len(metrics))))
    raw_metrics = np.full((len(metrics), max_per_state, 4), np.inf, dtype=np.float32)
    raw_stage = {name: np.zeros((len(metrics), max_per_state), dtype=bool) for name in raw_masks_flat}
    cursor = np.zeros(len(metrics), dtype=np.int64)
    for index, state in enumerate(raw_state):
        local = cursor[state]
        raw_metrics[state, local] = raw_metrics_flat[index]
        for name, mask in raw_masks_flat.items():
            raw_stage[name][state, local] = mask[index]
        cursor[state] += 1

    funnel = {}
    stage65 = {
        "k17": fixed,
        "union65_physical_support": union_support,
        "old_safety": safe,
    }
    for reference_name, reference_index, reference_metrics in (
        ("candidate0_incumbent", incumbent, incumbent_metrics),
        ("formal_v70", formal_index, formal_metrics),
    ):
        local_result = {}
        for name, mask in stage65.items():
            local_result[name] = {"candidate": candidate_stats(metrics, mask, reference_metrics), "oracle": oracle_stats(metrics, mask, reference_index, reference_metrics)}
        for name, mask in raw_stage.items():
            fixed_valid = fixed if name == "raw_draw_all" else (fixed & physical if name == "raw_draw_physical" else fixed & union_support)
            combined_metrics = np.concatenate((metrics, raw_metrics), axis=1)
            combined_mask = np.concatenate((fixed_valid, mask), axis=1)
            combined_ref = reference_index
            local_result[name] = {"candidate": candidate_stats(combined_metrics, combined_mask, reference_metrics), "oracle": oracle_stats(combined_metrics, combined_mask, combined_ref, reference_metrics)}
        selected = formal_index
        selected_gain = reference_metrics - metrics[row, selected]
        changed = selected != reference_index
        dual_selected = changed & (selected_gain[:, 0] > TOLERANCE) & (selected_gain[:, 3] > TOLERANCE)
        harm_selected = changed & ((selected_gain[:, 0] < -TOLERANCE) | (selected_gain[:, 3] < -TOLERANCE))
        local_result["scorer_archive_v70"] = {
            "selected_states": len(selected), "changed_states": int(changed.sum()),
            "dual": int(dual_selected.sum()), "harm": int(harm_selected.sum()),
            "dual_rate_among_changed": None if not changed.any() else float(dual_selected.sum() / changed.sum()),
            "harm_rate_among_changed": None if not changed.any() else float(harm_selected.sum() / changed.sum()),
            "objective_gain_vs_reference_m": float(selected_gain[:, 3].mean()),
        }
        funnel[reference_name] = local_result

    gain = formal_metrics[:, None] - metrics
    dual = (gain[..., 0] > TOLERANCE) & (gain[..., 3] > TOLERANCE)
    harm = (gain[..., 0] < -TOLERANCE) | (gain[..., 3] < -TOLERANCE)
    strata = {}
    for name in SCALARS:
        edges = np.asarray(prereg["label_free_bin_edges"][name], dtype=np.float64)
        records = []
        for index in range(10):
            lower, upper = edges[index], edges[index + 1]
            bin_mask = union_support & (scalar[name] >= lower) & ((scalar[name] <= upper) if index == 9 else (scalar[name] < upper))
            count = int(bin_mask.sum())
            records.append({
                "quantile_bin": [QUANTILES[index], QUANTILES[index + 1]],
                "edge": [float(lower), float(upper)],
                "candidates": count,
                "dual": int((bin_mask & dual).sum()),
                "dual_rate": None if count == 0 else float((bin_mask & dual).sum() / count),
                "harm": int((bin_mask & harm).sum()),
                "harm_rate": None if count == 0 else float((bin_mask & harm).sum() / count),
                "old_safety_accept_rate": None if count == 0 else float((bin_mask & safe).sum() / count),
                "old_safety_dual_recall": None if not (bin_mask & dual).any() else float((bin_mask & dual & safe).sum() / (bin_mask & dual).sum()),
                "mean_objective_gain_m": None if count == 0 else float(gain[..., 3][bin_mask].mean()),
            })
        strata[name] = records

    by_route_round = {}
    for route_name in sorted(np.unique(route)):
        states = np.flatnonzero(route == route_name)
        by_route_round[route_name] = {}
        for round_value, round_name in ((-1, "fixed"), (0, "r1"), (1, "r2"), (2, "r3")):
            mask = np.zeros_like(union_support)
            mask[states] = (rounds[states] == round_value) & union_support[states]
            by_route_round[route_name][round_name] = {
                "candidates": int(mask.sum()),
                "dual_rate": None if not mask.any() else float((mask & dual).sum() / mask.sum()),
                "harm_rate": None if not mask.any() else float((mask & harm).sum() / mask.sum()),
                "old_safety_accept_rate": None if not mask.any() else float((mask & safe).sum() / mask.sum()),
                "dual_recall_after_old_safety": None if not (mask & dual).any() else float((mask & dual & safe).sum() / (mask & dual).sum()),
                "states_with_dual": int((mask & dual).any(axis=1).sum()),
            }

    result = {
        "schema": "scand_new_mechanism_decomposition_result_v1",
        "status": "adaptive_development_mechanism_diagnostic_complete",
        "scope": "not eligible for model promotion or threshold selection",
        "states": len(metrics), "routes": int(len(np.unique(route))), "union_candidates": int(metrics.shape[1]), "raw_draws": int(len(raw_state)),
        "funnel": funnel, "fixed_strata_deciles": strata, "route_round": by_route_round,
        "reserve_accessed": False, "validation_labels_opened": False,
    }
    write_json(result_path, result)
    audit = {
        "schema": "scand_new_mechanism_decomposition_audit_v1",
        "preregistration_sha256": sha256_file(prereg_path),
        "result_sha256": sha256_file(result_path),
        "input_hashes_unchanged": all(sha256_file(Path(item["path"])) == item["sha256"] for item in prereg["inputs"].values()),
        "raw_draw_metric_formula": "draw_clipped*0.25 versus ground_truth_local_m; objective=0.4*ATE+0.6*endpoint",
        "threshold_or_profile_selected": False,
        "formal_v70_mutated": False,
        "reserve_accessed": False,
    }
    write_json(audit_path, audit)
    print(canonical_json({"status": result["status"], "result": str(result_path), "result_sha256": sha256_file(result_path), "audit_sha256": sha256_file(audit_path)}))


def main() -> None:
    args = parse_args()
    if args.freeze_only:
        freeze(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
