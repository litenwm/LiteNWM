#!/usr/bin/env python3
"""Open strict SCAND100 outcomes after selection freeze and report paired metrics."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import pickle
import sys
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tools.scoring.train_recon_dense_meta_dinov3_v1 import (  # noqa: E402
    file_sha256,
    write_json,
    write_npz,
)


DEFAULT_ROOT = REPO_ROOT / "outputs/scand_strict_h4_100_three_arm_v1"
METRICS = ("ate", "rpe", "endpoint", "objective")
ARM_ORDER = (
    "incumbent",
    "dinov3_full_meta_raw",
    "dinov3_full_meta_v2_stacked",
    "dinov3_scorer_v2_native",
    "dinov3_full_meta_raw_clean",
    "posthoc_pool_oracle",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--freeze",
        type=Path,
        default=DEFAULT_ROOT / "frozen_selection/pre_label_freeze.json",
    )
    parser.add_argument(
        "--navigation-eval",
        type=Path,
        default=REPO_ROOT
        / "data/evaluation/scand/strict_test_h4_100_seed17/navigation_eval.pkl",
    )
    parser.add_argument(
        "--sha256sums",
        type=Path,
        default=REPO_ROOT
        / "data/evaluation/scand/strict_test_h4_100_seed17/SHA256SUMS.txt",
    )
    parser.add_argument(
        "--data-root", type=Path, default=Path("/home/rainbow/nwm/data/scand")
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ROOT / "outcomes")
    parser.add_argument("--bootstrap-replicates", type=int, default=20000)
    parser.add_argument("--bootstrap-seed", type=int, default=170017)
    return parser.parse_args()


def sha_from_sum_file(path: Path, name: str) -> str:
    for line in path.read_text(encoding="utf-8").splitlines():
        digest, _, filename = line.partition("  ")
        if filename == name:
            return digest
    raise RuntimeError(f"{name} missing from {path}")


def verify_freeze(freeze_path: Path) -> dict[str, Any]:
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze.get("schema") != "scand100_k17_dense_meta_three_arm_pre_label_v1":
        raise RuntimeError("strict SCAND100 freeze schema mismatch")
    if freeze.get("status") != (
        "all_selections_frozen_before_scand_strict_test_outcomes_opened"
    ):
        raise RuntimeError("strict SCAND100 selection was not pre-label frozen")
    if freeze.get("labels_opened") is not False:
        raise RuntimeError("freeze does not assert labels remained closed")
    if freeze.get("fixed_pool_only") is not True or freeze.get("cem") is not False:
        raise RuntimeError("freeze fixed-pool/CEM contract changed")
    for path_key, hash_key in (
        ("candidate", "candidate_sha256"),
        ("candidate_audit", "candidate_audit_sha256"),
        ("preflight_audit", "preflight_audit_sha256"),
        ("manifest", "manifest_sha256"),
        ("features", "features_sha256"),
        ("policy", "policy_sha256"),
        ("raw_clean_policy", "raw_clean_policy_sha256"),
        ("inference", "inference_sha256"),
    ):
        artifact = Path(freeze[path_key])
        if file_sha256(artifact) != freeze[hash_key]:
            raise RuntimeError(f"post-freeze artifact changed: {path_key}")
    return freeze


def load_local_ground_truth(path: Path, timestep: int, horizon: int = 4) -> np.ndarray:
    with path.open("rb") as handle:
        raw = pickle.load(handle)
    positions = np.asarray(raw["position"], dtype=np.float64)[
        timestep : timestep + horizon + 1, :2
    ]
    yaw = np.asarray(raw["yaw"], dtype=np.float64).reshape(-1)[
        timestep : timestep + horizon + 1
    ]
    if positions.shape != (horizon + 1, 2) or yaw.shape != (horizon + 1,):
        raise RuntimeError(f"trajectory is shorter than H{horizon}: {path}")
    angle = float(yaw[0])
    rotation = np.asarray(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]],
        dtype=np.float64,
    )
    return ((positions - positions[0]).dot(rotation))[1:].astype(np.float32)


def candidate_metrics(candidates_m: np.ndarray, ground_truth_m: np.ndarray) -> np.ndarray:
    error = candidates_m - ground_truth_m[:, None]
    squared = np.sum(error * error, axis=-1)
    ate = np.sqrt(np.mean(squared, axis=-1))
    endpoint = np.sqrt(squared[..., -1])
    relative_error = np.diff(candidates_m, axis=2) - np.diff(
        ground_truth_m, axis=1
    )[:, None]
    rpe = np.sqrt(np.mean(np.sum(relative_error * relative_error, axis=-1), axis=-1))
    objective = np.float32(0.4) * ate + np.float32(0.6) * endpoint
    result = np.stack((ate, rpe, endpoint, objective), axis=-1).astype(np.float32)
    if result.shape != (100, 17, 4) or not np.isfinite(result).all():
        raise RuntimeError("strict SCAND100 metric tensor contract failed")
    return result


def wilson_lower(successes: int, trials: int, z: float = 1.959963984540054) -> float | None:
    if trials == 0:
        return None
    p = successes / trials
    denominator = 1.0 + z * z / trials
    center = p + z * z / (2.0 * trials)
    radius = z * math.sqrt(p * (1.0 - p) / trials + z * z / (4.0 * trials * trials))
    return float((center - radius) / denominator)


def absolute_summary(values: np.ndarray) -> dict[str, Any]:
    return {
        metric: {
            "mean_m": float(np.mean(values[:, index])),
            "p90_m": float(np.quantile(values[:, index], 0.9)),
        }
        for index, metric in enumerate(METRICS)
    }


def precision_summary(selected: np.ndarray, values: np.ndarray) -> dict[str, Any]:
    rows = np.arange(len(selected))
    promoted = selected != 0
    chosen = values[rows, selected]
    incumbent = values[:, 0]
    delta = chosen - incumbent
    ate_success = promoted & (delta[:, 0] < -1e-7)
    objective_success = promoted & (delta[:, 3] < -1e-7)
    dual_success = ate_success & objective_success
    trials = int(promoted.sum())
    result: dict[str, Any] = {"promotions": trials}
    for name, mask in (
        ("ate", ate_success),
        ("objective", objective_success),
        ("dual", dual_success),
    ):
        successes = int(mask.sum())
        result[f"{name}_successes"] = successes
        result[f"{name}_precision"] = float(successes / trials) if trials else None
        result[f"{name}_wilson_lower_95"] = wilson_lower(successes, trials)
    harm: dict[str, Any] = {}
    for index, metric in enumerate(METRICS):
        positive = delta[promoted, index]
        positive = positive[positive > 1e-7]
        harm[metric] = {
            "harmed_promotions": int(len(positive)),
            "mean_positive_delta_m": float(np.mean(positive)) if len(positive) else 0.0,
            "p90_positive_delta_m": float(np.quantile(positive, 0.9)) if len(positive) else 0.0,
            "max_positive_delta_m": float(np.max(positive)) if len(positive) else 0.0,
            "sum_positive_delta_m": float(np.sum(positive)) if len(positive) else 0.0,
        }
    result["harm_magnitude"] = harm
    result["mean_delta_vs_incumbent_m"] = {
        metric: float(np.mean(delta[:, index])) for index, metric in enumerate(METRICS)
    }
    return result


def cluster_bootstrap(
    delta: np.ndarray,
    clusters: np.ndarray,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    unique = np.unique(clusters.astype(str))
    members = [np.flatnonzero(clusters.astype(str) == value) for value in unique]
    rng = np.random.default_rng(seed)
    samples = np.empty((replicates, delta.shape[1]), dtype=np.float64)
    for replicate in range(replicates):
        drawn = rng.integers(0, len(unique), size=len(unique))
        index = np.concatenate([members[value] for value in drawn])
        samples[replicate] = np.mean(delta[index], axis=0)
    return {
        "clusters": int(len(unique)),
        "replicates": int(replicates),
        "seed": int(seed),
        "metrics": {
            metric: {
                "point_mean_delta_m": float(np.mean(delta[:, index])),
                "ci95_low_m": float(np.quantile(samples[:, index], 0.025)),
                "ci95_high_m": float(np.quantile(samples[:, index], 0.975)),
                "bootstrap_probability_mean_improves": float(np.mean(samples[:, index] < 0.0)),
            }
            for index, metric in enumerate(METRICS)
        },
    }


def main() -> None:
    args = parse_args()
    result_path = args.output_dir / "result.json"
    outcome_path = args.output_dir / "strict_outcomes.npz"
    if result_path.exists() or outcome_path.exists():
        raise FileExistsError("refusing to overwrite strict SCAND100 outcomes")
    if args.bootstrap_replicates < 1000:
        raise ValueError("use at least 1000 bootstrap replicates")

    # Everything above this point is label-free. The hash gate is checked before
    # navigation_eval.pkl or any future pose slice is deserialized below.
    freeze = verify_freeze(args.freeze)
    navigation_expected_sha = sha_from_sum_file(args.sha256sums, "navigation_eval.pkl")
    if file_sha256(args.navigation_eval) != navigation_expected_sha:
        raise RuntimeError("navigation_eval.pkl differs from the frozen benchmark package")

    with np.load(Path(freeze["inference"]), allow_pickle=False) as payload:
        record_key = payload["record_key"].astype(str)
        trajectory = payload["trajectory"].astype(str)
        route = payload["route"].astype(str)
        eval_index = payload["eval_index"].astype(np.int64)
        selections = {
            name: payload[f"{name}_selected_index"].astype(np.int64)
            for name in ARM_ORDER[1:-1]
        }
    with args.navigation_eval.open("rb") as handle:
        navigation_eval = pickle.load(handle)
    expected_navigation = [
        (str(name), int(key.rpartition(":")[2]), 4, 4)
        for name, key in zip(trajectory, record_key)
    ]
    if navigation_eval != expected_navigation:
        raise RuntimeError("navigation_eval order/identity differs from frozen inference")
    if not np.array_equal(eval_index, np.arange(100)):
        raise RuntimeError("navigation eval index is not canonical 0..99")

    with np.load(Path(freeze["candidate"]), allow_pickle=False) as payload:
        candidates_m = (
            payload["candidates_waypoint_units"].astype(np.float32) * np.float32(0.38)
        )
    ground_truth = np.empty((100, 4, 2), dtype=np.float32)
    cache: dict[str, dict[int, np.ndarray]] = {}
    for index, (name, key) in enumerate(zip(trajectory, record_key)):
        timestep = int(key.rpartition(":")[2])
        cache.setdefault(name, {})
        cache[name][timestep] = load_local_ground_truth(
            args.data_root / name / "traj_data.pkl", timestep
        )
        ground_truth[index] = cache[name][timestep]
    values = candidate_metrics(candidates_m, ground_truth)
    selections = {
        "incumbent": np.zeros(100, dtype=np.int64),
        **selections,
        "posthoc_pool_oracle": np.argmin(values[:, :, 3], axis=1).astype(np.int64),
    }
    rows = np.arange(100)
    selected_values = {name: values[rows, selected] for name, selected in selections.items()}
    absolute = {name: absolute_summary(selected_values[name]) for name in ARM_ORDER}
    promotion = {
        name: precision_summary(selections[name], values) for name in ARM_ORDER
    }

    comparisons: dict[str, tuple[np.ndarray, np.ndarray]] = {
        f"{name}_minus_incumbent": (selected_values[name], selected_values["incumbent"])
        for name in ARM_ORDER[1:]
    }
    comparisons["v2_stacked_minus_raw"] = (
        selected_values["dinov3_full_meta_v2_stacked"],
        selected_values["dinov3_full_meta_raw"],
    )
    comparisons["v2_stacked_minus_raw_clean_matched"] = (
        selected_values["dinov3_full_meta_v2_stacked"],
        selected_values["dinov3_full_meta_raw_clean"],
    )
    bootstrap: dict[str, Any] = {}
    for comparison_index, (name, (left, right)) in enumerate(comparisons.items()):
        delta = left - right
        bootstrap[name] = {
            "sign_convention": "left minus right; negative favors left",
            "trajectory_clustered": cluster_bootstrap(
                delta,
                trajectory,
                args.bootstrap_replicates,
                args.bootstrap_seed + comparison_index * 2,
            ),
            "route_clustered": cluster_bootstrap(
                delta,
                route,
                args.bootstrap_replicates,
                args.bootstrap_seed + comparison_index * 2 + 1,
            ),
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_npz(
        outcome_path,
        record_key=record_key,
        trajectory=trajectory,
        route=route,
        eval_index=eval_index,
        ground_truth_local_m=ground_truth,
        candidates_m=candidates_m,
        candidate_metrics_m=values,
        **{f"{name}_selected_index": selected for name, selected in selections.items()},
    )
    result = {
        **freeze,
        "schema": "scand100_k17_dense_meta_three_arm_zero_shot_result_v1",
        "status": "completed_first_zero_shot_scand_strict_h4_100_after_prelabel_freeze",
        "zero_shot_source": "RECON-trained frozen policies; no SCAND calibration",
        "navigation_eval": str(args.navigation_eval.resolve()),
        "navigation_eval_sha256": file_sha256(args.navigation_eval),
        "navigation_eval_deserialized_after_selection_freeze": True,
        "pose_outcomes_opened_after_selection_freeze": True,
        "labels_opened": True,
        "strict_outcomes": str(outcome_path.resolve()),
        "strict_outcomes_sha256": file_sha256(outcome_path),
        "objective_contract": "0.4 * RMS ATE + 0.6 * endpoint error",
        "rpe_contract": "RMS error of consecutive displacement vectors over H4",
        "oracle_contract": "post-hoc argmin objective over all frozen K17 candidates; unconstrained by scorer support",
        "absolute_metrics": absolute,
        "promotion_metrics": promotion,
        "paired_cluster_bootstrap": bootstrap,
        "matched_data_comparison": {
            "raw_clean": "824 RECON states, no scorer-v2 auxiliary predictions",
            "v2_stacked_clean": "same 824 RECON states plus frozen scorer-v2 auxiliary predictions",
            "comparison_key": "v2_stacked_minus_raw_clean_matched",
        },
        "scand_calibration_performed": False,
        "posthoc_retuning_performed": False,
        "fixed_pool_only": True,
        "cem": False,
    }
    write_json(result_path, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
