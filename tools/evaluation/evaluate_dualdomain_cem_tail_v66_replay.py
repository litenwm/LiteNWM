#!/usr/bin/env python3
"""Open outcomes only after paired original/v66 CEM replays are frozen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from tools.evaluation.build_recon100_k17_source_v1 import metric_labels  # noqa: E402
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
    parser.add_argument("--original-replay-dir", type=Path, required=True)
    parser.add_argument("--v66-replay-dir", type=Path, required=True)
    parser.add_argument("--ground-truth-source", type=Path, required=True)
    parser.add_argument(
        "--cohort-role", choices=("a_to_h", "spent_scand"), required=True
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_frozen_replay(directory: Path, expected_role: str) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    freeze_path = directory / "prelabel_freeze.json"
    freeze = read_json(freeze_path)
    if freeze.get("status") != EXPECTED_STATUS:
        raise RuntimeError(f"replay is not frozen: {freeze_path}")
    if freeze.get("model", {}).get("model_role") != expected_role:
        raise RuntimeError(f"unexpected model role in {freeze_path}")
    if freeze.get("labels_opened") or freeze.get("future_pose_outcomes_opened"):
        raise RuntimeError(f"replay reports early outcome access: {freeze_path}")
    if freeze.get("reserve_accessed"):
        raise RuntimeError(f"replay reports reserve access: {freeze_path}")
    selected_path = Path(freeze["selected_actions"])
    if file_sha256(selected_path) != freeze["selected_actions_sha256"]:
        raise RuntimeError(f"selected action hash changed: {selected_path}")
    return freeze, load_npz(selected_path)


def bind_pair(original: dict[str, Any], v66: dict[str, Any]) -> None:
    for key in (
        "domain",
        "cohort",
        "states",
        "manifest_sha256",
        "candidate_pool_sha256",
        "base_features_sha256",
        "latent_cache_sha256",
        "support_checkpoint_sha256",
    ):
        if original.get(key) != v66.get(key):
            raise RuntimeError(f"paired replay mismatch for {key}")
    if original.get("universal_valid_contract") != v66.get("universal_valid_contract"):
        raise RuntimeError("paired replay validity contract mismatch")
    if original.get("config") != v66.get("config"):
        raise RuntimeError("paired replay CEM protocol mismatch")


def decision_arrays(
    original: dict[str, np.ndarray], v66: dict[str, np.ndarray]
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    return {
        "original_arm_d_fixed_cem_false": (
            original["base_k17_feature_waypoint_units"].astype(np.float32),
            original["fixed_pool_selected_index"].astype(np.int64),
        ),
        "original_arm_d_cem": (
            original["final_cem_feature_waypoint_units"].astype(np.float32),
            original["final_cem_selected_index"].astype(np.int64),
        ),
        "proposal_aware_v66_fixed": (
            v66["base_k17_feature_waypoint_units"].astype(np.float32),
            v66["fixed_pool_selected_index"].astype(np.int64),
        ),
        "proposal_aware_v66_cem": (
            v66["final_cem_feature_waypoint_units"].astype(np.float32),
            v66["final_cem_selected_index"].astype(np.int64),
        ),
    }


def decision_metrics(
    candidates: np.ndarray,
    selected: np.ndarray,
    ground_truth: np.ndarray,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    if candidates.ndim != 4 or candidates.shape[1:] != (17, 4, 2):
        raise RuntimeError(f"candidate contract changed: {candidates.shape}")
    if selected.shape != (len(candidates),):
        raise RuntimeError("selected index shape mismatch")
    if np.any((selected < 0) | (selected >= candidates.shape[1])):
        raise RuntimeError("selected index outside candidate set")
    labels = metric_labels(candidates, ground_truth, 0.25, 0.4, 0.6)
    row = np.arange(len(selected))
    absolute = labels["values"][row, selected, :4]
    delta = labels["targets"][row, selected, :4]
    promoted = selected != 0
    promotion_count = int(promoted.sum())
    dual = promoted & (delta[:, 0] < -1.0e-7) & (delta[:, 3] < -1.0e-7)
    harmed = promoted & ((delta[:, 0] > 1.0e-7) | (delta[:, 3] > 1.0e-7))
    summary = {
        "states": int(len(selected)),
        "promotions": promotion_count,
        "coverage": float(promoted.mean()),
        "dual_successes": int(dual.sum()),
        "dual_precision": (
            float(dual.sum() / promotion_count) if promotion_count else None
        ),
        "harm_count": int(harmed.sum()),
        "harm_rate": (
            float(harmed.sum() / promotion_count) if promotion_count else None
        ),
        "absolute_mean_m": {
            name: float(absolute[:, index].mean())
            for index, name in enumerate(METRICS)
        },
        "absolute_p90_m": {
            name: float(np.quantile(absolute[:, index], 0.9))
            for index, name in enumerate(METRICS)
        },
        "delta_mean_m": {
            name: float(delta[:, index].mean())
            for index, name in enumerate(METRICS)
        },
        "delta_p90_m": {
            name: float(np.quantile(delta[:, index], 0.9))
            for index, name in enumerate(METRICS)
        },
    }
    return summary, {
        "selected_index": selected,
        "promoted": promoted,
        "dual": dual,
        "harmed": harmed,
        "absolute_metrics": absolute,
        "delta_metrics": delta,
    }


def metric_differences(
    challenger: dict[str, Any], baseline: dict[str, Any]
) -> dict[str, Any]:
    mean_delta = {
        name: challenger["absolute_mean_m"][name] - baseline["absolute_mean_m"][name]
        for name in METRICS
    }
    p90_delta = {
        name: challenger["absolute_p90_m"][name] - baseline["absolute_p90_m"][name]
        for name in METRICS
    }
    tolerance = 1.0e-7
    return {
        "coverage_gain": challenger["coverage"] - baseline["coverage"],
        "absolute_mean_delta_m": mean_delta,
        "absolute_p90_delta_m": p90_delta,
        "all_four_mean_nondegraded": all(
            value <= tolerance for value in mean_delta.values()
        ),
        "all_four_p90_nondegraded": all(
            value <= tolerance for value in p90_delta.values()
        ),
    }


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(args.output_dir)
    original_freeze, original_arrays = load_frozen_replay(
        args.original_replay_dir, "original_arm_d"
    )
    v66_freeze, v66_arrays = load_frozen_replay(
        args.v66_replay_dir, "proposal_aware_v66"
    )
    bind_pair(original_freeze, v66_freeze)

    ground_truth_source = load_npz(args.ground_truth_source)
    if "ground_truth_local_m" not in ground_truth_source:
        raise RuntimeError("ground-truth source lacks ground_truth_local_m")
    expected_record = original_arrays["record_key"].astype(str)
    if not np.array_equal(v66_arrays["record_key"].astype(str), expected_record):
        raise RuntimeError("paired replay record identity mismatch")
    if not np.array_equal(
        ground_truth_source["record_key"].astype(str), expected_record
    ):
        raise RuntimeError("ground-truth/replay record identity mismatch")
    ground_truth = ground_truth_source["ground_truth_local_m"].astype(np.float32)
    if ground_truth.shape != (len(expected_record), 4, 2):
        raise RuntimeError(f"ground-truth contract changed: {ground_truth.shape}")

    summaries = {}
    selected_arrays = {"record_key": expected_record}
    for name, (candidates, selected) in decision_arrays(
        original_arrays, v66_arrays
    ).items():
        summary, arrays = decision_metrics(candidates, selected, ground_truth)
        summaries[name] = summary
        for key, value in arrays.items():
            selected_arrays[f"{name}__{key}"] = value

    baseline_name = "original_arm_d_fixed_cem_false"
    challenger_name = "proposal_aware_v66_cem"
    comparison = metric_differences(
        summaries[challenger_name], summaries[baseline_name]
    )
    challenger = summaries[challenger_name]
    development_checks = {
        "coverage_at_least_0_40": challenger["coverage"] >= 0.40,
        "coverage_gain_at_least_0_10": comparison["coverage_gain"] >= 0.10,
        "dual_at_least_0_85": (
            challenger["dual_precision"] is not None
            and challenger["dual_precision"] >= 0.85
        ),
        "harm_at_most_0_15": (
            challenger["harm_rate"] is not None
            and challenger["harm_rate"] <= 0.15
        ),
        "all_four_mean_nondegraded": comparison["all_four_mean_nondegraded"],
        "all_four_p90_nondegraded": comparison["all_four_p90_nondegraded"],
    }
    development_checks["cohort_pass"] = all(development_checks.values())

    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected_metrics_path = args.output_dir / "selected_metrics.npz"
    write_npz_atomic(selected_metrics_path, **selected_arrays)
    result = {
        "schema": "dualdomain_cem_tail_v66_outcome_replay_result",
        "status": "spent_development_outcomes_evaluated_without_retuning",
        "domain": original_freeze["domain"],
        "cohort": original_freeze["cohort"],
        "cohort_role": args.cohort_role,
        "states": int(len(expected_record)),
        "decisions": summaries,
        "primary_baseline": baseline_name,
        "primary_challenger": challenger_name,
        "primary_comparison": comparison,
        "development_checks": development_checks,
        "original_replay_freeze": str(args.original_replay_dir / "prelabel_freeze.json"),
        "original_replay_freeze_sha256": file_sha256(
            args.original_replay_dir / "prelabel_freeze.json"
        ),
        "v66_replay_freeze": str(args.v66_replay_dir / "prelabel_freeze.json"),
        "v66_replay_freeze_sha256": file_sha256(
            args.v66_replay_dir / "prelabel_freeze.json"
        ),
        "ground_truth_source": str(args.ground_truth_source),
        "ground_truth_source_sha256": file_sha256(args.ground_truth_source),
        "selected_metrics": str(selected_metrics_path),
        "selected_metrics_sha256": file_sha256(selected_metrics_path),
        "thresholds_tuned_on_this_cohort": False,
        "outcomes_used_by_cem": False,
        "A_to_H_accessed": args.cohort_role == "a_to_h",
        "reserve_accessed": False,
        "production_safe_claim_allowed": False,
    }
    result_path = args.output_dir / "result.json"
    write_json_atomic(result_path, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
