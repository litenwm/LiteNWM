#!/usr/bin/env python3
"""Open the sealed v70 RECON reserve once and report the frozen evaluation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from tools.evaluation.build_recon_inner_k17_source_v2 import local_ground_truth  # noqa: E402
from tools.evaluation.evaluate_dualdomain_cem_tail_v66_replay import (  # noqa: E402
    decision_metrics,
    metric_differences,
)
from tools.evaluation.export_recon_dense_dinov3_features_v1 import (  # noqa: E402
    load_npz,
    write_json_atomic,
    write_npz_atomic,
)
from tools.scoring.train_recon_dense_meta_dinov3_v1 import file_sha256  # noqa: E402


METRICS = ("ate", "rpe", "endpoint", "objective")
EXPECTED_STATUS = "candidate_level_safe_elite_actions_frozen_before_outcomes"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seal", type=Path, required=True)
    parser.add_argument("--preopen-audit", type=Path, required=True)
    parser.add_argument("--trajectory-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> list[float] | None:
    if trials <= 0:
        return None
    p = successes / trials
    denominator = 1.0 + z * z / trials
    center = (p + z * z / (2.0 * trials)) / denominator
    radius = z * math.sqrt(p * (1.0 - p) / trials + z * z / (4.0 * trials * trials)) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def load_replay(freeze_path: Path, role: str) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    freeze = read_json(freeze_path)
    if freeze.get("status") != EXPECTED_STATUS:
        raise RuntimeError(f"replay is not frozen: {freeze_path}")
    if freeze.get("model", {}).get("model_role") != role:
        raise RuntimeError(f"unexpected model role: {freeze_path}")
    if any(
        (
            freeze.get("labels_opened"),
            freeze.get("future_pose_outcomes_opened"),
            freeze.get("outcomes_used_by_cem"),
            freeze.get("reserve_accessed"),
        )
    ):
        raise RuntimeError(f"replay reports early outcome access: {freeze_path}")
    selected_path = Path(freeze["selected_actions"])
    if file_sha256(selected_path) != freeze["selected_actions_sha256"]:
        raise RuntimeError(f"selected actions changed: {selected_path}")
    return freeze, load_npz(selected_path)


def main() -> None:
    args = parse_args()
    seal = read_json(args.seal)
    preopen = read_json(args.preopen_audit)
    expected = seal["expected_outputs"]
    if seal.get("status") != "v70_sealed_before_reserve_outcomes":
        raise RuntimeError("v70 reserve seal is not active")
    if preopen.get("status") != "paired_actions_frozen_and_bit_exact_before_reserve_outcomes":
        raise RuntimeError("paired preopen audit did not pass")
    if preopen.get("seal_sha256") != file_sha256(args.seal):
        raise RuntimeError("preopen audit is bound to a different seal")
    if preopen.get("reserve_outcomes_accessed") or preopen.get("reserve_spent"):
        raise RuntimeError("preopen audit reports prior reserve access")
    if args.preopen_audit.resolve() != Path(expected["preopen_pair_audit"]).resolve():
        raise RuntimeError("preopen audit path differs from seal")
    if args.output_dir.resolve() != Path(expected["evaluation_dir"]).resolve():
        raise RuntimeError("evaluation directory differs from seal")
    if args.trajectory_root.resolve() != Path(seal["reserve_outcome_binding"]["trajectory_root"]).resolve():
        raise RuntimeError("trajectory root differs from seal")

    marker_path = args.output_dir / "one_time_outcome_open_started.json"
    ground_truth_path = args.output_dir / "ground_truth_once.npz"
    selected_metrics_path = args.output_dir / "selected_metrics.npz"
    result_path = args.output_dir / "result.json"
    final_audit_path = args.output_dir / "one_time_outcome_open_audit.json"
    for path in (marker_path, ground_truth_path, selected_metrics_path, result_path, final_audit_path):
        if path.exists():
            raise RuntimeError(f"reserve was already opened or evaluated: {path}")

    original_freeze_path = Path(preopen["original_replay_freeze"])
    v70_freeze_path = Path(preopen["v70_replay_freeze"])
    if file_sha256(original_freeze_path) != preopen["original_replay_freeze_sha256"]:
        raise RuntimeError("original replay freeze changed after preopen audit")
    if file_sha256(v70_freeze_path) != preopen["v70_replay_freeze_sha256"]:
        raise RuntimeError("v70 replay freeze changed after preopen audit")
    original_freeze, original = load_replay(original_freeze_path, "original_arm_d")
    v70_freeze, v70 = load_replay(v70_freeze_path, "proposal_aware_v66")
    expected_record = original["record_key"].astype(str)
    if not np.array_equal(v70["record_key"].astype(str), expected_record):
        raise RuntimeError("paired replay record identity changed")
    if not np.array_equal(
        original["final_cem_feature_waypoint_units"],
        v70["final_cem_feature_waypoint_units"],
        equal_nan=True,
    ) or not np.array_equal(
        original["final_cem_selected_index"],
        v70["final_cem_selected_index"],
    ):
        raise RuntimeError("RECON original/v70 CEM output lost bit exactness")

    manifest_path = Path(seal["reserve_inputs"]["manifest"]["path"])
    if file_sha256(manifest_path) != seal["reserve_inputs"]["manifest"]["sha256"]:
        raise RuntimeError("reserve manifest changed after seal")
    rows = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    manifest_record = np.asarray(
        [f"{row['trajectory_id']}:{int(row['current_t'])}" for row in rows]
    )
    if not np.array_equal(manifest_record, expected_record):
        raise RuntimeError("reserve manifest/replay identity mismatch")

    raw_paths = {
        str(row["trajectory_id"]): args.trajectory_root / str(row["trajectory_id"]) / "traj_data.pkl"
        for row in rows
    }
    sealed_paths = seal["reserve_outcome_binding"]["trajectory_files"]
    if sorted(str(path.resolve()) for path in raw_paths.values()) != sorted(sealed_paths):
        raise RuntimeError("reserve raw trajectory path set differs from seal")
    missing = [str(path) for path in raw_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing sealed reserve trajectories: " + ", ".join(missing))

    args.output_dir.mkdir(parents=True, exist_ok=False)
    opened_at = datetime.now(timezone.utc).isoformat()
    marker = {
        "schema": "dualdomain_cem_tail_v70_reserve_one_time_open_marker",
        "status": "one_time_reserve_outcome_open_started",
        "opened_at_utc": opened_at,
        "seal": str(args.seal.resolve()),
        "seal_sha256": file_sha256(args.seal),
        "preopen_audit": str(args.preopen_audit.resolve()),
        "preopen_audit_sha256": file_sha256(args.preopen_audit),
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": file_sha256(manifest_path),
        "trajectory_files": sorted(str(path.resolve()) for path in raw_paths.values()),
        "reserve_spent": True,
        "retuning_allowed": False,
    }
    write_json_atomic(marker_path, marker)

    raw_hashes = {
        trajectory: {
            "path": str(path.resolve()),
            "sha256": file_sha256(path),
        }
        for trajectory, path in sorted(raw_paths.items())
    }
    ground_truth = np.empty((len(rows), 4, 2), dtype=np.float32)
    for index, row in enumerate(rows):
        ground_truth[index] = local_ground_truth(
            raw_paths[str(row["trajectory_id"])], int(row["current_t"])
        )
    write_npz_atomic(
        ground_truth_path,
        record_key=expected_record,
        ground_truth_local_m=ground_truth,
    )

    decisions = {
        "original_arm_d_fixed_cem_false": (
            original["base_k17_feature_waypoint_units"].astype(np.float32),
            original["fixed_pool_selected_index"].astype(np.int64),
        ),
        "original_arm_d_archive_cem": (
            original["final_cem_feature_waypoint_units"].astype(np.float32),
            original["final_cem_selected_index"].astype(np.int64),
        ),
        "proposal_aware_v70_fixed": (
            v70["base_k17_feature_waypoint_units"].astype(np.float32),
            v70["fixed_pool_selected_index"].astype(np.int64),
        ),
        "proposal_aware_v70_archive_cem": (
            v70["final_cem_feature_waypoint_units"].astype(np.float32),
            v70["final_cem_selected_index"].astype(np.int64),
        ),
    }
    summaries: dict[str, Any] = {}
    selected_arrays: dict[str, np.ndarray] = {"record_key": expected_record}
    for name, (candidates, selected) in decisions.items():
        summary, arrays = decision_metrics(candidates, selected, ground_truth)
        summary["dual_wilson_95"] = wilson_interval(
            int(summary["dual_successes"]), int(summary["promotions"])
        )
        summary["harm_wilson_95"] = wilson_interval(
            int(summary["harm_count"]), int(summary["promotions"])
        )
        summaries[name] = summary
        for key, value in arrays.items():
            selected_arrays[f"{name}__{key}"] = value
    write_npz_atomic(selected_metrics_path, **selected_arrays)

    baseline_name = seal["acceptance_protocol"]["primary_baseline"]
    challenger_name = seal["acceptance_protocol"]["primary_challenger"]
    baseline = summaries[baseline_name]
    challenger = summaries[challenger_name]
    comparison = metric_differences(challenger, baseline)
    checks = {
        "coverage_at_least_0_40": challenger["coverage"] >= 0.40,
        "dual_at_least_0_85": challenger["dual_precision"] is not None
        and challenger["dual_precision"] >= 0.85,
        "harm_at_most_0_15": challenger["harm_rate"] is not None
        and challenger["harm_rate"] <= 0.15,
        "all_four_absolute_mean_nondegraded": comparison["all_four_mean_nondegraded"],
        "all_four_absolute_p90_nondegraded": comparison["all_four_p90_nondegraded"],
        "recon_original_v70_bit_exact": bool(preopen["all_selected_action_arrays_bit_exact"]),
    }
    checks["reserve_confirmation_pass"] = all(checks.values())
    status = (
        "reserve_confirmation_passed_no_retuning_allowed"
        if checks["reserve_confirmation_pass"]
        else "reserve_confirmation_failed_no_retuning_allowed"
    )
    result = {
        "schema": "dualdomain_cem_tail_v70_one_time_reserve_result",
        "status": status,
        "reserve_spent": True,
        "states": int(len(rows)),
        "decisions": summaries,
        "primary_baseline": baseline_name,
        "primary_challenger": challenger_name,
        "primary_comparison": comparison,
        "acceptance_checks": checks,
        "seal": str(args.seal.resolve()),
        "seal_sha256": file_sha256(args.seal),
        "preopen_audit": str(args.preopen_audit.resolve()),
        "preopen_audit_sha256": file_sha256(args.preopen_audit),
        "one_time_open_marker": str(marker_path.resolve()),
        "one_time_open_marker_sha256": file_sha256(marker_path),
        "ground_truth_once": str(ground_truth_path.resolve()),
        "ground_truth_once_sha256": file_sha256(ground_truth_path),
        "selected_metrics": str(selected_metrics_path.resolve()),
        "selected_metrics_sha256": file_sha256(selected_metrics_path),
        "raw_outcome_source_hashes": raw_hashes,
        "outcomes_used_by_cem": False,
        "reserve_used_by_training_or_calibration": False,
        "thresholds_tuned_on_reserve": False,
        "retuning_allowed_after_result": False,
        "production_safe_claim_allowed": False,
    }
    write_json_atomic(result_path, result)
    final_audit = {
        "schema": "dualdomain_cem_tail_v70_one_time_reserve_open_audit",
        "status": "reserve_opened_exactly_once_and_reported_without_retuning",
        "opened_at_utc": opened_at,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "seal_sha256": file_sha256(args.seal),
        "preopen_audit_sha256": file_sha256(args.preopen_audit),
        "marker_sha256": file_sha256(marker_path),
        "ground_truth_once_sha256": file_sha256(ground_truth_path),
        "selected_metrics_sha256": file_sha256(selected_metrics_path),
        "result": str(result_path.resolve()),
        "result_sha256": file_sha256(result_path),
        "raw_outcome_source_hashes": raw_hashes,
        "reserve_spent": True,
        "retuning_allowed": False,
        "reserve_may_enter_training_or_calibration": False,
    }
    write_json_atomic(final_audit_path, final_audit)
    print(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(final_audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
