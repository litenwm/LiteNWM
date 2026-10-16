#!/usr/bin/env python3
"""Seal v70, its reserve inputs, and its acceptance protocol before outcomes."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scoring.spacing_support_v1 import threshold_for_quantile  # noqa: E402
from tools.evaluation.export_recon_dense_dinov3_features_v1 import (  # noqa: E402
    write_json_atomic,
)
from tools.scoring.finetune_universal_dualdomain_safe_rank_v2 import (  # noqa: E402
    load_envelopes,
)
from tools.scoring.train_recon_dense_meta_dinov3_v1 import file_sha256  # noqa: E402


V70_DIR = ROOT / "outputs/cem_development_v1/dualdomain_cem_tail_v70_broad_hashfold_archive"
V70_RESULT = V70_DIR / "result.json"
V70_PREREG = ROOT / "outputs/cem_development_v1/dualdomain_cem_tail_v70_broad_hashfold_archive_preregistration.json"
V70_AGGREGATE = V70_DIR / "aggregate_result.json"
BASE_RESULT = ROOT / "outputs/universal_dualdomain_raw_v2/training_v2/result.json"
RESERVE = ROOT / "outputs/cem_development_v1/recon_target_calibration_v20/final_confirmation_reserve"
CONFIRMATION = V70_DIR / "reserve_confirmation"
SEAL_PATH = CONFIRMATION / "sealed_manifest.json"
TRAJECTORY_ROOT = Path("/home/rainbow/nwm/data/recon")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def artifact(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path.resolve()), "sha256": file_sha256(path)}


def assert_hash(path: Path, expected: str, label: str) -> None:
    observed = file_sha256(path)
    if observed != expected:
        raise RuntimeError(f"{label} SHA mismatch: {observed} != {expected}")


def manifest_rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def manifest_identities(path: Path) -> tuple[set[str], set[str], set[str]]:
    rows = manifest_rows(path)
    return (
        {str(row["trajectory_id"]) for row in rows},
        {str(row["route_id"]) for row in rows},
        {f"{row['trajectory_id']}:{int(row['current_t'])}" for row in rows},
    )


def npz_identities(path: Path) -> tuple[set[str], set[str], set[str]]:
    with np.load(path, allow_pickle=False) as payload:
        records = {str(value) for value in payload["record_key"].astype(str)}
        trajectories = {record.rpartition(":")[0] for record in records}
        routes = (
            {str(value).removeprefix("recon::") for value in payload["route_group"].astype(str)}
            if "route_group" in payload.files
            else set()
        )
    return trajectories, routes, records


def main() -> None:
    if SEAL_PATH.exists():
        raise FileExistsError(SEAL_PATH)
    forbidden_outputs = (
        CONFIRMATION / "preopen_reserve_pair_audit.json",
        CONFIRMATION / "evaluation/one_time_outcome_open_started.json",
        CONFIRMATION / "evaluation/ground_truth_once.npz",
        CONFIRMATION / "evaluation/result.json",
        CONFIRMATION / "evaluation/one_time_outcome_open_audit.json",
    )
    existing = [str(path) for path in forbidden_outputs if path.exists()]
    if existing:
        raise RuntimeError("reserve already opened or paired before seal: " + ", ".join(existing))

    v70 = read_json(V70_RESULT)
    prereg = read_json(V70_PREREG)
    aggregate = read_json(V70_AGGREGATE)
    base = read_json(BASE_RESULT)
    if v70.get("status") != "frozen_after_v69_spent_failure_before_v70_spent_replay":
        raise RuntimeError("v70 model is not in the expected frozen state")
    if prereg.get("status") != "frozen_after_v69_spent_failure_before_v70_training":
        raise RuntimeError("v70 preregistration status changed")
    if aggregate.get("status") != "promising_development_result":
        raise RuntimeError("v70 development aggregate is not frozen as promising")
    for name, value in (("v70", v70), ("prereg", prereg), ("aggregate", aggregate)):
        if value.get("reserve_accessed"):
            raise RuntimeError(f"{name} reports prior reserve access")
    if v70.get("A_to_H_used_by_v70_training_or_calibration"):
        raise RuntimeError("v70 reports A-H use in training/calibration")
    if v70.get("fresh_scand_used_by_v70_training_or_calibration"):
        raise RuntimeError("v70 reports fresh-SCAND use in training/calibration")
    if not v70.get("recon_output_bit_exact"):
        raise RuntimeError("v70 did not preserve RECON output bit exactness")

    expected_recon_policy = {
        "z": 0.5,
        "margin_ate": 0.01,
        "margin_objective": 0.01,
        "dual_threshold": 0.9,
        "harm_threshold": 0.2,
    }
    expected_scand_policy = {
        "z": 2.0,
        "margin_ate": 0.01,
        "margin_objective": 0.01,
        "dual_threshold": 0.5,
        "harm_threshold": 0.1,
    }
    recon_cal = v70["calibration"]["recon"]
    scand_cal = v70["calibration"]["scand"]
    if recon_cal.get("policy") != expected_recon_policy or not recon_cal.get("feasible"):
        raise RuntimeError("v70 RECON calibration contract changed")
    if scand_cal.get("policy") != expected_scand_policy or not scand_cal.get("feasible"):
        raise RuntimeError("v70 SCAND calibration contract changed")
    if (recon_cal["dual_temperature"], recon_cal["harm_temperature"]) != (3.0, 3.0):
        raise RuntimeError("v70 RECON temperatures changed")
    if (scand_cal["dual_temperature"], scand_cal["harm_temperature"]) != (1.5, 1.5):
        raise RuntimeError("v70 SCAND temperatures changed")
    if v70.get("route_universe") != {
        "broad_routes": 33,
        "exact_routes": 36,
        "fold_exact_routes": [3, 10, 4, 11, 8],
        "fold_states": [2352, 9588, 3828, 11264, 7776],
    }:
        raise RuntimeError("five-fold broad-hashfold route universe changed")

    full_checkpoints = v70["checkpoints"]["full"]
    oof_folds = v70["checkpoints"]["calibration_oof"]
    if len(full_checkpoints) != 3 or len(oof_folds) != 5:
        raise RuntimeError("v70 checkpoint ensemble cardinality changed")
    checkpoint_artifacts: list[dict[str, Any]] = []
    for record in full_checkpoints:
        path = Path(record["path"])
        assert_hash(path, record["sha256"], record["member_role"])
        checkpoint_artifacts.append({**record, "path": str(path.resolve())})
    oof_checkpoint_artifacts: list[dict[str, Any]] = []
    for expected_fold, fold in enumerate(oof_folds):
        if fold.get("heldout_fold") != expected_fold or len(fold.get("members", [])) != 3:
            raise RuntimeError("five-fold OOF checkpoint structure changed")
        members = []
        for record in fold["members"]:
            path = Path(record["path"])
            assert_hash(path, record["sha256"], record["member_role"])
            members.append({**record, "path": str(path.resolve())})
        oof_checkpoint_artifacts.append({**fold, "members": members})
    for path_key, sha_key in (
        ("preregistration", "preregistration_sha256"),
        ("full_checkpoint_source", "full_checkpoint_source_sha256"),
        ("oof_predictions", "oof_predictions_sha256"),
        ("frontier", "frontier_sha256"),
        ("archive_selection", "archive_selection_sha256"),
    ):
        assert_hash(Path(v70[path_key]), v70[sha_key], path_key)

    base_arm = base["arms"]["D"]
    if base.get("cem") is not False:
        raise RuntimeError("primary baseline is not the frozen Arm-D CEM=false result")
    base_recon_policy = base_arm["calibration"]["frontier"]["domains"]["recon"]["policy"]
    if base_recon_policy != expected_recon_policy:
        raise RuntimeError("original Arm-D RECON policy differs from v70 frozen policy")
    support_path = Path(base["support"])
    support_threshold = float(
        threshold_for_quantile(load_envelopes(support_path)["recon"], 0.999)
    )

    manifest = RESERVE / "manifest.windows.jsonl"
    manifest_audit_path = RESERVE / "manifest.windows.audit.json"
    excluded_path = RESERVE / "manifest.windows.excluded_identity.json"
    k17_path = RESERVE / "k17.npz"
    source_path = RESERVE / "source_label_free.npz"
    feature_path = RESERVE / "features/recon_target_final_reserve_v26_features.npz"
    latent_path = RESERVE / "features/recon_target_final_reserve_v26_latents.npz"
    old_freeze_path = RESERVE / "selection_v26/freeze.json"
    old_index_path = RESERVE / "selection_v26/selected_index.npz"
    manifest_audit = read_json(manifest_audit_path)
    k17_audit = read_json(RESERVE / "k17.audit.json")
    source_audit = read_json(RESERVE / "source_label_free.audit.json")
    feature_audit = read_json(RESERVE / "features/recon_target_final_reserve_v26_features.audit.json")
    old_freeze = read_json(old_freeze_path)
    if manifest_audit.get("status") != "manifest_frozen_rgb_only":
        raise RuntimeError("reserve manifest is not frozen RGB-only")
    if manifest_audit.get("labels_opened") or manifest_audit.get("future_pose_opened"):
        raise RuntimeError("reserve manifest audit reports label access")
    if manifest_audit.get("selected_route_overlap") or manifest_audit.get("selected_trajectory_overlap"):
        raise RuntimeError("reserve manifest audit reports identity overlap")
    if k17_audit.get("status") != "candidate_pool_frozen_before_partition_labels":
        raise RuntimeError("reserve candidate pool is not prelabel-frozen")
    if source_audit.get("future_pose_targets_computed") or source_audit.get("future_pose_targets_opened"):
        raise RuntimeError("reserve label-free source reports target access")
    if source_audit.get("labels_in_output") or feature_audit.get("labels_accessed") or feature_audit.get("labels_in_output"):
        raise RuntimeError("reserve feature artifacts contain or accessed labels")
    if old_freeze.get("status") != "selected_index_frozen_before_outcomes":
        raise RuntimeError("historical reserve selection is not pre-outcome frozen")
    if old_freeze.get("labels_opened") or old_freeze.get("outcomes_accessed"):
        raise RuntimeError("historical reserve freeze reports outcome access")

    declared_hashes = (
        (manifest, manifest_audit["manifest_sha256"], "reserve manifest"),
        (excluded_path, manifest_audit["excluded_identity_sha256"], "excluded identity"),
        (k17_path, k17_audit["candidate_pool_sha256"], "reserve K17"),
        (source_path, source_audit["output_sha256"], "reserve label-free source"),
        (feature_path, feature_audit["output_sha256"], "reserve base features"),
        (latent_path, feature_audit["latent_cache_sha256"], "reserve latent cache"),
        (old_index_path, old_freeze["artifacts"]["selected_index_sha256"], "old reserve selection"),
    )
    for path, expected_sha, label in declared_hashes:
        assert_hash(path, expected_sha, label)

    forbidden_names = {"source_labeled.npz", "ground_truth.npz", "ground_truth_once.npz", "result.json"}
    reserve_files = sorted(path for path in RESERVE.rglob("*") if path.is_file())
    bad_names = [str(path) for path in reserve_files if path.name in forbidden_names or "outcome_result" in path.name]
    if bad_names:
        raise RuntimeError("reserve already contains outcome artifacts: " + ", ".join(bad_names))
    forbidden_npz_keys = ("ground_truth", "label", "target", "future_pose")
    npz_key_audit: dict[str, list[str]] = {}
    for path in reserve_files:
        if path.suffix != ".npz":
            continue
        with np.load(path, allow_pickle=False) as payload:
            keys = list(payload.files)
        bad = [key for key in keys if any(token in key.lower() for token in forbidden_npz_keys)]
        if bad:
            raise RuntimeError(f"reserve NPZ has forbidden outcome keys: {path}: {bad}")
        npz_key_audit[str(path.resolve())] = keys

    rows = manifest_rows(manifest)
    if len(rows) != 100 or any(row.get("partition") != "fresh-outer" for row in rows):
        raise RuntimeError("reserve manifest state/partition contract changed")
    reserve_trajectories = {str(row["trajectory_id"]) for row in rows}
    reserve_routes = {str(row["route_id"]) for row in rows}
    reserve_records = np.asarray(
        [f"{row['trajectory_id']}:{int(row['current_t'])}" for row in rows]
    )
    if len(reserve_trajectories) != 25 or len(reserve_routes) != 25:
        raise RuntimeError("reserve cohort identity cardinality changed")
    with np.load(k17_path, allow_pickle=False) as candidate, np.load(feature_path, allow_pickle=False) as base_feature:
        if not np.array_equal(candidate["record_key"].astype(str), reserve_records):
            raise RuntimeError("reserve candidate/manifest record mismatch")
        if not np.array_equal(base_feature["record_key"].astype(str), reserve_records):
            raise RuntimeError("reserve feature/manifest record mismatch")
        base_candidates = np.concatenate(
            (
                candidate["candidates_waypoint_units"][:, :1].astype(np.float32),
                candidate["nomad_x16_waypoint_units"].astype(np.float32),
            ),
            axis=1,
        )
        if not np.array_equal(base_candidates, base_feature["candidates"].astype(np.float32)):
            raise RuntimeError("reserve candidate/base-feature bridge changed")

    excluded = read_json(excluded_path)
    used_trajectories = set(map(str, excluded["used_trajectories"]))
    used_routes = set(map(str, excluded["used_routes"]))
    if reserve_trajectories & used_trajectories or reserve_routes & used_routes:
        raise RuntimeError("reserve overlaps excluded used identity union")
    identity_sources: dict[str, Any] = {}
    train_cal_paths = (
        ROOT / "outputs/universal_dualdomain_raw_v2/features/recon_train_fit/recon_train_fit_universal_v2_features.npz",
        ROOT / "outputs/universal_dualdomain_raw_v2/features/recon_scorer_cal/recon_scorer_cal_universal_v2_features.npz",
    )
    for path in train_cal_paths:
        trajectories, routes, records = npz_identities(path)
        if reserve_trajectories & trajectories or reserve_routes & routes or set(reserve_records) & records:
            raise RuntimeError(f"reserve overlaps training/calibration identities: {path}")
        identity_sources[str(path.resolve())] = {
            "sha256": file_sha256(path),
            "trajectory_overlap": 0,
            "route_overlap": 0,
            "record_overlap": 0,
        }
    ah_manifests = [
        ROOT / f"outputs/cem_development_v1/recon_target_calibration_v20/bank_{letter}/manifest.windows.jsonl"
        for letter in "abcdef"
    ] + [
        ROOT / "outputs/cem_development_v1/recon_cem_boundary_v60/bank_g_v60/manifest.windows.jsonl",
        ROOT / "outputs/cem_development_v1/recon_cem_boundary_v60/bank_h_v60/manifest.windows.jsonl",
    ]
    for path in ah_manifests:
        trajectories, routes, records = manifest_identities(path)
        if reserve_trajectories & trajectories or reserve_routes & routes or set(reserve_records) & records:
            raise RuntimeError(f"reserve overlaps A-H identities: {path}")
        identity_sources[str(path.resolve())] = {
            "sha256": file_sha256(path),
            "trajectory_overlap": 0,
            "route_overlap": 0,
            "record_overlap": 0,
        }

    trajectory_files = sorted(
        str((TRAJECTORY_ROOT / trajectory / "traj_data.pkl").resolve())
        for trajectory in reserve_trajectories
    )
    missing_trajectories = [path for path in trajectory_files if not Path(path).is_file()]
    if missing_trajectories:
        raise FileNotFoundError("reserve raw trajectory paths are unclear/missing: " + ", ".join(missing_trajectories))

    scripts = {
        "v70_runner": artifact(ROOT / "tools/planning/run_dualdomain_cem_tail_v70_archive_replay.py"),
        "base_cem_runner": artifact(ROOT / "tools/planning/run_dualdomain_cem_tail_v66_replay.py"),
        "pair_binder": artifact(ROOT / "tools/evaluation/bind_dualdomain_cem_tail_v70_reserve_pair.py"),
        "one_time_evaluator": artifact(ROOT / "tools/evaluation/evaluate_dualdomain_cem_tail_v70_reserve_once.py"),
        "seal_builder": artifact(Path(__file__)),
    }
    all_reserve_artifacts = {
        str(path.relative_to(RESERVE)): artifact(path) for path in reserve_files
    }
    original_replay_dir = CONFIRMATION / "replays/recon_final_reserve/original_arm_d"
    v70_replay_dir = CONFIRMATION / "replays/recon_final_reserve/proposal_aware_v70"
    preopen_audit = CONFIRMATION / "preopen_reserve_pair_audit.json"
    evaluation_dir = CONFIRMATION / "evaluation"
    seal = {
        "schema": "dualdomain_cem_tail_v70_reserve_sealed_manifest",
        "status": "v70_sealed_before_reserve_outcomes",
        "sealed_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_contract": {
            "version": "v70_broad_hashfold_archive",
            "result": artifact(V70_RESULT),
            "preregistration": artifact(V70_PREREG),
            "development_aggregate": artifact(V70_AGGREGATE),
            "full_deployment_checkpoints": checkpoint_artifacts,
            "calibration_oof_checkpoints": oof_checkpoint_artifacts,
            "oof_predictions": artifact(Path(v70["oof_predictions"])),
            "calibration_frontier": artifact(Path(v70["frontier"])),
            "calibration_archive_selection": artifact(Path(v70["archive_selection"])),
            "route_universe": v70["route_universe"],
            "scand_policy": expected_scand_policy,
            "scand_dual_temperature": 1.5,
            "scand_harm_temperature": 1.5,
            "recon_policy": expected_recon_policy,
            "recon_dual_temperature": 3.0,
            "recon_harm_temperature": 3.0,
            "recon_output_bit_exact": True,
            "cross_round_safe_archive": True,
        },
        "primary_baseline_contract": {
            "name": "original_arm_d_fixed_cem_false",
            "result": artifact(BASE_RESULT),
            "model_trained_with_cem": False,
            "arm": "D",
            "policy": expected_recon_policy,
        },
        "cem_contract": {
            "iterations": 3,
            "population_challengers": 16,
            "set_cardinality_with_incumbent": 17,
            "elite_fraction": 0.25,
            "elite_count": 4,
            "proposal_scale": 0.70,
            "covariance_floor_waypoint_units": 0.015,
            "support_quantile": 0.999,
            "support_threshold": support_threshold,
            "seed": 20260730,
            "spacing_m": 0.25,
            "candidate_level_safety_before_elite": True,
            "safe_shortfall_rule": "retain previous proposal unchanged",
            "cross_round_safe_archive": True,
        },
        "reserve_inputs": {
            "manifest": artifact(manifest),
            "manifest_audit": artifact(manifest_audit_path),
            "excluded_identity": artifact(excluded_path),
            "candidate_pool": artifact(k17_path),
            "label_free_source": artifact(source_path),
            "base_features": artifact(feature_path),
            "latent_cache": artifact(latent_path),
            "historical_preoutcome_freeze": artifact(old_freeze_path),
            "historical_preoutcome_selection": artifact(old_index_path),
            "all_files": all_reserve_artifacts,
            "npz_key_audit": npz_key_audit,
        },
        "reserve_independence_audit": {
            "states": 100,
            "trajectories": 25,
            "routes": 25,
            "excluded_identity_union_sha256": file_sha256(excluded_path),
            "training_calibration_and_A_to_H_sources": identity_sources,
            "all_trajectory_route_record_overlaps": 0,
            "labels_present_in_sealed_inputs": False,
            "prior_reserve_outcome_access_detected": False,
        },
        "reserve_outcome_binding": {
            "trajectory_root": str(TRAJECTORY_ROOT.resolve()),
            "trajectory_files": trajectory_files,
            "trajectory_file_hashes_before_open": None,
            "trajectory_file_hash_policy": "paths sealed now; each raw file SHA-256 recorded during the one authorized outcome open",
            "window_binding": "manifest trajectory_id + current_t, H4 local_ground_truth",
        },
        "scripts": scripts,
        "expected_outputs": {
            "original_replay_dir": str(original_replay_dir.resolve()),
            "v70_replay_dir": str(v70_replay_dir.resolve()),
            "preopen_pair_audit": str(preopen_audit.resolve()),
            "evaluation_dir": str(evaluation_dir.resolve()),
        },
        "acceptance_protocol": {
            "primary_baseline": "original_arm_d_fixed_cem_false",
            "primary_challenger": "proposal_aware_v70_archive_cem",
            "coverage_minimum": 0.40,
            "dual_precision_minimum": 0.85,
            "harm_rate_maximum": 0.15,
            "absolute_mean_requirement": "all ate/rpe/endpoint/objective deltas <= 1e-7 m versus primary baseline",
            "absolute_p90_requirement": "all ate/rpe/endpoint/objective deltas <= 1e-7 m versus primary baseline",
            "recon_nondamage_requirement": "all original Arm-D/v70 paired selected-action arrays and artifact SHA-256 must be bit exact before outcomes",
            "uncertainty_report": "two-sided 95% Wilson intervals for dual precision and harm rate",
            "pass_rule": "all six frozen checks must pass",
        },
        "irreversibility": {
            "reserve_may_be_evaluated": "exactly once after both outcome-free replays and paired preopen audit pass",
            "reserve_outcomes_may_change_model": False,
            "reserve_outcomes_may_change_thresholds": False,
            "reserve_outcomes_may_change_selection_logic": False,
            "reserve_may_enter_training_or_calibration": False,
            "result_may_only_be_reported": True,
            "failure_may_trigger_retuning": False,
        },
        "reserve_accessed": False,
        "production_safe_claim_allowed": False,
    }
    CONFIRMATION.mkdir(parents=True, exist_ok=False)
    write_json_atomic(SEAL_PATH, seal)
    print(json.dumps(seal, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
