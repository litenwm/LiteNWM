#!/usr/bin/env python3
"""Build a common spent-only challenger-vs-formal-v70 runtime feature table."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from planning.proposal_tail_sidecar_v1 import action_structure_features, canonical_json, sha256_file  # noqa: E402
from tools.evaluation import run_v72_proposal_tail_reranker as v72  # noqa: E402
from tools.evaluation import run_v74_direct_delta_nested as v74  # noqa: E402


TOL = 1.0e-7


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-root", type=Path, required=True)
    parser.add_argument("--new-root", type=Path, required=True)
    parser.add_argument("--v77-predictions", type=Path, required=True)
    parser.add_argument("--fresh-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    return parser.parse_args()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    require(path.is_file(), f"missing NPZ: {path}")
    with np.load(path, allow_pickle=False) as payload:
        return {key: payload[key].copy() for key in payload.files}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    require(path.is_file(), f"missing JSONL: {path}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def stable_array_sha(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def formal_delta(value: np.ndarray, formal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    selected = value[np.arange(len(formal)), formal]
    return value - selected[:, None], np.broadcast_to(selected[:, None], value.shape)


def seed_summary(ate: np.ndarray, rtp: np.ndarray, support: np.ndarray) -> np.ndarray:
    parts = []
    for value in (ate.astype(np.float32), rtp.astype(np.float32), support.astype(np.float32)):
        parts.extend((value.mean(axis=-1, keepdims=True), value.std(axis=-1, keepdims=True), np.ptp(value, axis=-1, keepdims=True)))
    return np.concatenate(parts, axis=-1).astype(np.float32)


def percentile(value: np.ndarray, mask: np.ndarray, *, descending: bool = False) -> np.ndarray:
    result = np.ones(value.shape, dtype=np.float32)
    for state in range(len(value)):
        index = np.flatnonzero(mask[state] & np.isfinite(value[state]))
        if len(index) == 0:
            continue
        order = index[np.argsort(value[state, index], kind="stable")]
        if descending:
            order = order[::-1]
        result[state, order] = np.arange(len(order), dtype=np.float32) / max(len(order) - 1, 1)
    return result[..., None]


def safe_channels(values: dict[str, np.ndarray], formal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    shape = values["energy"].shape
    observed = np.ones(shape, dtype=bool)
    for name in ("energy", "metric_mean", "metric_std", "dual_probability", "harm_probability", "rank_logit"):
        key = name + "_observed"
        if key in values:
            observed &= values[key].astype(bool)
    fields = [
        observed[..., None].astype(np.float32),
        values["energy"][..., None].astype(np.float32),
        values["metric_mean"].astype(np.float32),
        values["metric_std"].astype(np.float32),
        values["dual_probability"][..., None].astype(np.float32),
        values["harm_probability"][..., None].astype(np.float32),
        values["rank_logit"][..., None].astype(np.float32),
        values["support_score"][..., None].astype(np.float32),
    ]
    channel = np.concatenate(fields, axis=-1)
    channel[~np.isfinite(channel)] = 0.0
    return formal_delta(channel, formal)


def build_feature(
    candidates: np.ndarray,
    formal: np.ndarray,
    actions: np.ndarray,
    raw: np.ndarray,
    scorer: np.ndarray,
    ate_seed: np.ndarray,
    rtp_seed: np.ndarray,
    support_seed: np.ndarray,
    safe: dict[str, np.ndarray],
    source_round: np.ndarray,
    source_candidate: np.ndarray,
    pool: np.ndarray,
    old_safe: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    geometry = candidates.astype(np.float32) * 0.25
    action = actions.astype(np.float32)
    structure = action_structure_features(action)
    seeds = seed_summary(ate_seed, rtp_seed, support_seed)
    blocks = []
    names: list[str] = []
    for name, value in (
        ("ltwm_temporal", raw.astype(np.float32)),
        ("scorer_aux", scorer.astype(np.float32)),
        ("waypoint_m", geometry),
        ("actions_h4", action),
        ("action_structure", structure),
        ("seed_disagreement", seeds),
    ):
        delta, formal_value = formal_delta(value.reshape(*value.shape[:2], -1), formal)
        blocks.extend((delta, formal_value))
        names.extend([f"delta_{name}_{i}" for i in range(delta.shape[-1])])
        names.extend([f"formal_{name}_{i}" for i in range(formal_value.shape[-1])])
    safe_delta, safe_formal = safe_channels(safe, formal)
    blocks.extend((safe_delta, safe_formal))
    names.extend([f"delta_safety_{i}" for i in range(safe_delta.shape[-1])])
    names.extend([f"formal_safety_{i}" for i in range(safe_formal.shape[-1])])

    source_id = np.clip(source_round.astype(np.int64) + 1, 0, 3)
    source_onehot = np.eye(4, dtype=np.float32)[source_id]
    source_rank = np.clip(source_candidate.astype(np.float32) / 16.0, 0.0, 1.0)[..., None]
    ranks = np.concatenate((
        percentile(safe["energy"], pool),
        percentile(safe["dual_probability"], pool, descending=True),
        percentile(safe["harm_probability"], pool),
        percentile(safe["support_score"], pool),
        percentile(0.4 * scorer[..., 6] + 0.6 * scorer[..., 8], pool),
    ), axis=-1)
    flags = np.stack((pool.astype(np.float32), old_safe.astype(np.float32), np.arange(pool.shape[1])[None, :] == formal[:, None]), axis=-1).astype(np.float32)
    blocks.extend((source_onehot, source_rank, ranks, flags))
    names.extend([f"source_{i}" for i in range(4)] + ["source_candidate_rank"] + [f"within_state_rank_{i}" for i in range(5)] + ["physical_support", "old_safe", "is_formal"])
    feature = np.concatenate(blocks, axis=-1).astype(np.float32)
    require(np.isfinite(feature).all(), "nonfinite common runtime feature")
    require(feature.shape[-1] == len(names), "feature name mismatch")
    return feature, names


def sidecar_cohort(root: Path, cohort_name: str, camera_contract: str) -> dict[str, Any]:
    temporal = load_npz(root / "temporal_pair_features.npz")
    table = load_npz(root / "proposal_tail_sidecar/candidate_table.npz")
    labels = load_npz(root / "pilot_candidate_labels.npz")
    manifest_rows = load_jsonl(root / "pilot_manifest.windows.jsonl")
    record = table["record_key"].astype(str)
    require(len(record) == len(manifest_rows), f"manifest size mismatch: {cohort_name}")
    require(np.array_equal(record, labels["record_key"].astype(str)), f"label record mismatch: {cohort_name}")
    require(np.array_equal(table["candidate_hash"].astype(str), labels["candidate_hash"].astype(str)), f"candidate hash mismatch: {cohort_name}")
    require(np.array_equal(record, temporal["record_key_audit_only"].astype(str)), f"temporal record mismatch: {cohort_name}")
    require(np.array_equal(table["candidate_hash"].astype(str), temporal["candidate_hash"].astype(str)), f"temporal candidate mismatch: {cohort_name}")
    formal = temporal["formal_v70_union_index"].astype(np.int64)
    safe = {name: temporal[name] for name in ("energy", "metric_mean", "metric_std", "dual_probability", "harm_probability", "rank_logit", "support_score")}
    for name in list(safe):
        observed = name + "_observed"
        if observed in temporal:
            safe[observed] = temporal[observed]
    candidates = table["candidates_feature_waypoint_units"].astype(np.float32)
    pool = temporal["valid"].astype(bool)
    old_safe = temporal["safe"].astype(bool)
    feature, names = build_feature(
        candidates,
        formal,
        temporal["actions_h4_m_rad"],
        temporal["ltwm_temporal_summary"].reshape(len(record), 65, 112),
        temporal["scorer_v2_aux"],
        temporal["ate_by_seed"],
        temporal["rtp_by_seed"],
        temporal["support_by_seed"],
        safe,
        table["source_round"],
        table["source_candidate_index"],
        pool,
        old_safe,
    )
    return {
        "cohort": np.full(len(record), cohort_name),
        "camera_contract": np.full(len(record), camera_contract),
        "record": record,
        "route": np.asarray([str(row["route_group"]) for row in manifest_rows]),
        "trajectory": np.asarray([str(row["trajectory_id"]) for row in manifest_rows]),
        "robot": np.asarray([str(row["robot"]) for row in manifest_rows]),
        "candidates": candidates,
        "formal": formal,
        "pool": pool,
        "old_safe": old_safe,
        "feature": feature,
        "feature_names": names,
        "true_metrics": labels["true_metrics_m"].astype(np.float32),
        "source_round": table["source_round"].astype(np.int8),
        "source_candidate": table["source_candidate_index"].astype(np.int8),
        "input_hashes": {
            "candidate_table": sha256_file(root / "proposal_tail_sidecar/candidate_table.npz"),
            "temporal_pair_features": sha256_file(root / "temporal_pair_features.npz"),
            "labels": sha256_file(root / "pilot_candidate_labels.npz"),
            "manifest": sha256_file(root / "pilot_manifest.windows.jsonl"),
        },
    }


def fresh_cohort(v77_predictions: Path, fresh_manifest: Path) -> dict[str, Any]:
    spec = next(item for item in v72.COHORTS if item["name"] == "scand_fresh_outer")
    data = v72.load_cohort(spec)
    structural, structural_audit = v74.union_structural_for_cohort(spec, data)
    persisted = load_npz(v77_predictions)
    require(np.array_equal(data["record_key"].astype(str), persisted["record_key"].astype(str)), "v77 record mismatch")
    require(np.array_equal(data["formal_index"].astype(np.int64), persisted["formal_v70_index"].astype(np.int64)), "v77 formal mismatch")
    rows = load_jsonl(fresh_manifest)
    by_record = {str(row.get("record_key", f"{row['trajectory_id']}:{row['curr_time']}")): row for row in rows}
    metadata = []
    for record in data["record_key"].astype(str):
        require(record in by_record, f"fresh manifest missing record {record}")
        metadata.append(by_record[record])
    extension = data["extended"][..., -13:].astype(np.float32)
    safe = {
        "energy": extension[..., 1],
        "metric_mean": extension[..., 2:6],
        "metric_std": extension[..., 6:10],
        "dual_probability": extension[..., 10],
        "harm_probability": extension[..., 11],
        "rank_logit": extension[..., 12],
        "support_score": data["extended"][..., -(13 + 1)],
        "energy_observed": data["v70_score_available"],
        "metric_mean_observed": data["v70_score_available"],
        "metric_std_observed": data["v70_score_available"],
        "dual_probability_observed": data["v70_score_available"],
        "harm_probability_observed": data["v70_score_available"],
        "rank_logit_observed": data["v70_score_available"],
    }
    source_round = data["source_id"].astype(np.int8) - 1
    source_candidate = np.round(structural["source_rank"] * 16.0).astype(np.int8)
    pool = persisted["physical_support_pool"].astype(bool)
    feature, names = build_feature(
        data["candidates"].astype(np.float32),
        data["formal_index"].astype(np.int64),
        structural["actions_h4"],
        structural["raw_ltwm"],
        structural["scorer_aux"],
        structural["ate_seed"],
        structural["rtp_seed"],
        structural["support_seed"],
        safe,
        source_round,
        source_candidate,
        pool,
        data["safe"].astype(bool),
    )
    return {
        "cohort": np.full(len(data["record_key"]), "spent_fresh_v70_v77"),
        "camera_contract": np.full(len(data["record_key"]), "legacy_preprocessed_scand_contract"),
        "record": data["record_key"].astype(str),
        "route": np.asarray([str(row["route_group"]) for row in metadata]),
        "trajectory": np.asarray([str(row["trajectory_id"]) for row in metadata]),
        "robot": np.asarray([str(row["robot"]) for row in metadata]),
        "candidates": data["candidates"].astype(np.float32),
        "formal": data["formal_index"].astype(np.int64),
        "pool": pool,
        "old_safe": data["safe"].astype(bool),
        "feature": feature,
        "feature_names": names,
        "true_metrics": data["true_metrics"].astype(np.float32),
        "source_round": source_round,
        "source_candidate": source_candidate,
        "input_hashes": {**data["hashes"], **structural_audit["source_hashes"], "v77_predictions": sha256_file(v77_predictions), "manifest": sha256_file(fresh_manifest)},
    }


def main() -> None:
    args = parse_args()
    require(not args.output.exists(), f"refusing to overwrite: {args.output}")
    require(not args.audit.exists(), f"refusing to overwrite: {args.audit}")
    for path in (args.legacy_root, args.new_root, args.v77_predictions, args.fresh_manifest, args.output, args.audit):
        require("reserve" not in str(path).lower(), f"forbidden reserve-named path: {path}")

    cohorts = [
        sidecar_cohort(args.legacy_root, "spent_old_train_sidecar", "legacy_preprocessed_scand_contract"),
        fresh_cohort(args.v77_predictions, args.fresh_manifest),
        sidecar_cohort(args.new_root, "spent_new_mainrgb_development", "audited_canonical_main_rgb_320x240"),
    ]
    names = cohorts[0]["feature_names"]
    require(all(cohort["feature_names"] == names for cohort in cohorts), "common feature schema mismatch")
    record = np.concatenate([cohort["record"] for cohort in cohorts])
    require(len(set(record.tolist())) == len(record), "record duplication across spent cohorts")
    route = np.concatenate([cohort["route"] for cohort in cohorts])
    trajectory = np.concatenate([cohort["trajectory"] for cohort in cohorts])
    feature = np.concatenate([cohort["feature"] for cohort in cohorts])
    true_metrics = np.concatenate([cohort["true_metrics"] for cohort in cohorts])
    formal = np.concatenate([cohort["formal"] for cohort in cohorts])
    row = np.arange(len(record))
    formal_metrics = true_metrics[row, formal]
    gain = formal_metrics[:, None] - true_metrics
    dual = (gain[..., 0] > TOL) & (gain[..., 3] > TOL)
    helpful = gain[..., 3] > TOL
    harm = (gain[..., 0] < -TOL) | (gain[..., 3] < -TOL)
    arrays = {
        "record_key": record,
        "route_group": route,
        "trajectory_id": trajectory,
        "robot": np.concatenate([cohort["robot"] for cohort in cohorts]),
        "camera_contract": np.concatenate([cohort["camera_contract"] for cohort in cohorts]),
        "cohort": np.concatenate([cohort["cohort"] for cohort in cohorts]),
        "feature_names": np.asarray(names),
        "runtime_pair_feature": feature,
        "candidates_feature_waypoint_units": np.concatenate([cohort["candidates"] for cohort in cohorts]),
        "formal_v70_index": formal,
        "physical_support_pool": np.concatenate([cohort["pool"] for cohort in cohorts]),
        "old_safe": np.concatenate([cohort["old_safe"] for cohort in cohorts]),
        "source_round": np.concatenate([cohort["source_round"] for cohort in cohorts]),
        "source_candidate_index": np.concatenate([cohort["source_candidate"] for cohort in cohorts]),
        "true_metrics_m": true_metrics,
        "objective_gain_vs_formal_v70_m": gain[..., 3].astype(np.float32),
        "ate_gain_vs_formal_v70_m": gain[..., 0].astype(np.float32),
        "helpful_vs_formal_v70": helpful,
        "dual_vs_formal_v70": dual,
        "harm_vs_formal_v70": harm,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as handle:
        np.savez_compressed(handle, **arrays)

    route_counts = Counter(route.tolist())
    overlap_routes = sorted(name for name, count in Counter(np.concatenate([np.unique(cohort["route"]) for cohort in cohorts]).tolist()).items() if count > 1)
    cohort_audit = []
    for cohort in cohorts:
        cohort_audit.append({
            "name": str(cohort["cohort"][0]),
            "states": len(cohort["record"]),
            "routes": len(np.unique(cohort["route"])),
            "trajectories": len(np.unique(cohort["trajectory"])),
            "robots": dict(sorted(Counter(cohort["robot"].tolist()).items())),
            "camera_contract": str(cohort["camera_contract"][0]),
            "candidates_per_state": int(cohort["feature"].shape[1]),
            "physical_support_candidate_rows": int(cohort["pool"].sum()),
            "input_hashes": cohort["input_hashes"],
        })
    audit = {
        "schema": "scand_incremental_student_table_audit_v1",
        "status": "complete_spent_development_only",
        "states": len(record),
        "routes_after_shared_route_deduplication": len(route_counts),
        "trajectories": len(np.unique(trajectory)),
        "candidates_per_state": 65,
        "candidate_rows": int(len(record) * 65),
        "physical_support_candidate_rows": int(arrays["physical_support_pool"].sum()),
        "feature_dim": feature.shape[-1],
        "feature_names": names,
        "route_state_counts": dict(sorted(route_counts.items())),
        "route_groups_present_in_multiple_spent_sources_and_grouped_together": overlap_routes,
        "cohorts": cohort_audit,
        "target_contract": {
            "reference": "frozen formal-v70 selected/fallback action per state",
            "objective_gain": "formal objective minus challenger objective",
            "ate_gain": "formal ATE minus challenger ATE",
            "helpful": "strict positive objective gain",
            "dual": "strict positive ATE and objective gains",
            "harm": "negative ATE or objective gain",
        },
        "runtime_feature_contract": {
            "pairwise_delta": True,
            "formal_v70_context": True,
            "temporal_geometry": True,
            "three_seed_disagreement": True,
            "old_safety_as_feature_not_gate": True,
            "route_record_or_truth_in_features": False,
        },
        "field_completeness": {
            "candidate_pool_65": True,
            "formal_v70_selected_fallback": True,
            "temporal_ltwm_112": True,
            "geometry_and_actions": True,
            "three_seed_signals": True,
            "ground_truth_four_metrics": True,
            "camera_metadata": "canonical main-RGB audited only for the newest four-route cohort; historical processed cohorts are stratified as legacy contract",
        },
        "output": {"path": str(args.output.resolve()), "sha256": sha256_file(args.output), "runtime_feature_array_sha256": stable_array_sha(feature)},
        "development_aggregate_previously_seen": True,
        "sealed_validation_included": False,
        "reserve_content_accessed": False,
        "dwm_ltwm_rerun": False,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    args.audit.write_text(canonical_json(audit) + "\n", encoding="utf-8")
    print(canonical_json({
        "status": audit["status"],
        "states": audit["states"],
        "routes": audit["routes_after_shared_route_deduplication"],
        "trajectories": audit["trajectories"],
        "candidate_rows": audit["candidate_rows"],
        "feature_dim": audit["feature_dim"],
        "output_sha256": audit["output"]["sha256"],
        "audit": str(args.audit.resolve()),
    }))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(canonical_json({"status": "fail_closed", "error_type": type(error).__name__, "error": str(error), "reserve_content_accessed": False}), file=sys.stderr)
        raise SystemExit(2) from error
