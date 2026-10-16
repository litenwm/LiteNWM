#!/usr/bin/env python3
"""Measure strict pose-matched natural action branches in high-res dual latents.

This is a data-sufficiency diagnostic, not a training-table exporter.  It only
uses the development split, never reads outcome labels, and writes aggregate
statistics.  A pair must come from different trajectory fragments of the same
original recording, agree in pose and latent history, and differ in H1-H4
actions.  This makes the observed difference between the two true H4 latent
residuals a substantially cleaner pseudo-counterfactual than visual-only
cross-recording matching.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import time
from typing import Any

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

POSE_RADIUS_M = 0.20
YAW_RADIUS_DEG = 5.0
MINIMUM_STATE_COSINE = 0.90
MAXIMUM_PARTNERS_PER_ANCHOR = 4
SIMILAR_ACTION_DISTANCE = 0.50
DIVERSE_ACTION_DISTANCE = 1.00


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latent-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="train")
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON object required: {path}")
    return value


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=False)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def recording_id(trajectory_id: str) -> str:
    """SCAND random-MDP shards end in a fragment number."""
    value = re.sub(r"_\d+$", "", trajectory_id)
    require(value != trajectory_id, f"unrecognized trajectory id: {trajectory_id}")
    return value


def robot_id(trajectory_id: str) -> str:
    if "_Spot_" in trajectory_id:
        return "spot"
    if "_Jackal_" in trajectory_id:
        return "jackal"
    return "unknown"


def wrap_angle(value: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(value), np.cos(value))


def normalize_rows(value: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(value, axis=1, keepdims=True)
    return value / np.maximum(norm, 1.0e-8)


def quantiles(value: np.ndarray) -> dict[str, float | None]:
    if not len(value):
        return {name: None for name in ("q10", "q25", "q50", "q75", "q90")}
    return {
        "q10": float(np.quantile(value, 0.10)),
        "q25": float(np.quantile(value, 0.25)),
        "q50": float(np.quantile(value, 0.50)),
        "q75": float(np.quantile(value, 0.75)),
        "q90": float(np.quantile(value, 0.90)),
    }


def pooled_frame_latents(shard: np.lib.npyio.NpzFile) -> tuple[np.ndarray, np.ndarray]:
    semantic = np.asarray(shard["semantic_latents"], dtype=np.float32).mean(
        axis=(1, 2),
    )
    geometry = np.asarray(shard["geometry_latents"], dtype=np.float32).mean(
        axis=(1, 2),
    )
    return semantic, geometry


def load_windows(
    latent_root: Path,
    split: str,
) -> dict[str, np.ndarray]:
    metadata = read_json(latent_root / "metadata.json")
    splits = read_json(latent_root / "splits.json")
    require(split in splits, f"missing split: {split}")
    allowed = set(map(str, splits[split]))
    rows: dict[str, list[np.ndarray] | list[str]] = {
        "current": [],
        "history_delta": [],
        "future_residual": [],
        "actions": [],
        "position": [],
        "yaw": [],
        "trajectory": [],
        "recording": [],
        "robot": [],
        "timestep": [],
    }
    loaded = 0
    for item in metadata["trajectories"]:
        trajectory = str(item["trajectory_id"])
        if trajectory not in allowed:
            continue
        path = latent_root / "shards" / str(item["shard_file"])
        with np.load(path, allow_pickle=False) as shard:
            semantic, geometry = pooled_frame_latents(shard)
            actions = np.asarray(shard["actions_continuous"], dtype=np.float32)
            position = np.asarray(shard["positions"], dtype=np.float32)
            yaw = np.asarray(shard["yaw"], dtype=np.float32)
        require(
            len(semantic) == len(geometry) == len(position) == len(yaw),
            f"frame alignment changed: {trajectory}",
        )
        require(len(actions) == len(semantic) - 1, f"action alignment changed: {trajectory}")
        if len(semantic) < 8:
            continue
        latent = np.concatenate((semantic, geometry), axis=1)
        timestep = np.arange(3, len(latent) - 4, dtype=np.int32)
        rows["current"].append(latent[timestep])
        rows["history_delta"].append(latent[timestep] - latent[timestep - 3])
        rows["future_residual"].append(latent[timestep + 4] - latent[timestep])
        rows["actions"].append(
            np.stack(
                [actions[index : index + 4].reshape(-1) for index in timestep],
                axis=0,
            )
        )
        rows["position"].append(position[timestep])
        rows["yaw"].append(yaw[timestep])
        rows["timestep"].append(timestep)
        count = len(timestep)
        rows["trajectory"].extend([trajectory] * count)
        rows["recording"].extend([recording_id(trajectory)] * count)
        rows["robot"].extend([robot_id(trajectory)] * count)
        loaded += 1
        if loaded % 25 == 0:
            print(json.dumps({"loaded_trajectories": loaded}), flush=True)
    require(rows["current"], "no development windows found")
    return {
        name: (
            np.concatenate(value).astype(np.float32)
            if name not in {"trajectory", "recording", "robot"}
            else np.asarray(value)
        )
        for name, value in rows.items()
    }


def build_state_signature(windows: dict[str, np.ndarray]) -> np.ndarray:
    current = windows["current"]
    history = windows["history_delta"]
    current = normalize_rows(current - current.mean(axis=0, keepdims=True))
    history = normalize_rows(history - history.mean(axis=0, keepdims=True))
    return normalize_rows(np.concatenate((current, 0.25 * history), axis=1))


def candidate_pose_pairs(
    windows: dict[str, np.ndarray],
    signature: np.ndarray,
) -> tuple[np.ndarray, dict[str, int]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, recording in enumerate(windows["recording"].astype(str)):
        grouped[recording].append(index)
    selected: list[tuple[int, int]] = []
    pose_candidates = 0
    yaw_candidates = 0
    state_candidates = 0
    for recording in sorted(grouped):
        local = np.asarray(grouped[recording], dtype=np.int64)
        if len(local) < 2:
            continue
        tree = cKDTree(windows["position"][local])
        neighborhoods = tree.query_ball_point(
            windows["position"][local],
            r=POSE_RADIUS_M,
        )
        for local_first, neighbors in enumerate(neighborhoods):
            first = int(local[local_first])
            candidates = []
            for local_second in neighbors:
                second = int(local[local_second])
                if second <= first:
                    continue
                if windows["trajectory"][first] == windows["trajectory"][second]:
                    continue
                pose_candidates += 1
                yaw_distance = abs(
                    float(
                        wrap_angle(
                            np.asarray(
                                windows["yaw"][first] - windows["yaw"][second]
                            )
                        )
                    )
                )
                if yaw_distance > math.radians(YAW_RADIUS_DEG):
                    continue
                yaw_candidates += 1
                cosine = float(signature[first] @ signature[second])
                if cosine < MINIMUM_STATE_COSINE:
                    continue
                state_candidates += 1
                candidates.append((cosine, second))
            candidates.sort(reverse=True)
            for _, second in candidates[:MAXIMUM_PARTNERS_PER_ANCHOR]:
                selected.append((first, second))
    if not selected:
        return np.empty((0, 2), dtype=np.int64), {
            "pose_candidates": pose_candidates,
            "yaw_candidates": yaw_candidates,
            "state_candidates": state_candidates,
        }
    return np.asarray(selected, dtype=np.int64), {
        "pose_candidates": pose_candidates,
        "yaw_candidates": yaw_candidates,
        "state_candidates": state_candidates,
    }


def scaled_distances(
    value: np.ndarray,
    pairs: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    difference = (value[pairs[:, 0]] - value[pairs[:, 1]]) / scale
    return np.sqrt(np.mean(difference * difference, axis=1))


def route_consistency(
    routes: np.ndarray,
    action_distance: np.ndarray,
    future_distance: np.ndarray,
) -> dict[str, Any]:
    rows = []
    for route in np.unique(routes):
        mask = routes == route
        if int(mask.sum()) < 10:
            continue
        association = spearmanr(
            action_distance[mask],
            future_distance[mask],
        ).statistic
        similar = mask & (action_distance <= SIMILAR_ACTION_DISTANCE)
        diverse = mask & (action_distance >= DIVERSE_ACTION_DISTANCE)
        ratio = None
        if similar.any() and diverse.any():
            denominator = float(np.median(future_distance[similar]))
            if denominator > 0:
                ratio = float(np.median(future_distance[diverse]) / denominator)
        rows.append(
            {
                "route": str(route),
                "pairs": int(mask.sum()),
                "association": (
                    float(association) if np.isfinite(association) else None
                ),
                "diverse_vs_similar_ratio": ratio,
            }
        )
    association_rows = [
        row["association"] for row in rows if row["association"] is not None
    ]
    ratio_rows = [
        row["diverse_vs_similar_ratio"]
        for row in rows
        if row["diverse_vs_similar_ratio"] is not None
    ]
    return {
        "eligible_routes": len(rows),
        "positive_association_routes": int(
            sum(value > 0 for value in association_rows)
        ),
        "ratio_above_one_routes": int(sum(value > 1 for value in ratio_rows)),
        "median_association": (
            float(np.median(association_rows)) if association_rows else None
        ),
        "median_diverse_vs_similar_ratio": (
            float(np.median(ratio_rows)) if ratio_rows else None
        ),
    }


def main() -> None:
    args = parse_args()
    latent_root = args.latent_root.resolve()
    output = args.output.resolve()
    require(not output.exists(), f"refusing to overwrite: {output}")
    require(args.split == "train", "only the development train split is authorized")
    started = time.perf_counter()
    windows = load_windows(latent_root, args.split)
    signature = build_state_signature(windows)
    pairs, funnel = candidate_pose_pairs(windows, signature)
    if len(pairs):
        action_scale = np.maximum(np.std(windows["actions"], axis=0), 1.0e-4)
        future_scale = np.maximum(
            np.std(windows["future_residual"], axis=0),
            1.0e-4,
        )
        action_distance = scaled_distances(
            windows["actions"],
            pairs,
            action_scale,
        )
        future_distance = scaled_distances(
            windows["future_residual"],
            pairs,
            future_scale,
        )
        position_distance = np.linalg.norm(
            windows["position"][pairs[:, 0]]
            - windows["position"][pairs[:, 1]],
            axis=1,
        )
        yaw_distance = np.abs(
            wrap_angle(
                windows["yaw"][pairs[:, 0]]
                - windows["yaw"][pairs[:, 1]]
            )
        )
        state_cosine = np.sum(
            signature[pairs[:, 0]] * signature[pairs[:, 1]],
            axis=1,
        )
        association = spearmanr(action_distance, future_distance).statistic
    else:
        action_distance = future_distance = position_distance = yaw_distance = (
            state_cosine
        ) = np.empty(0, dtype=np.float32)
        association = float("nan")
    similar = action_distance <= SIMILAR_ACTION_DISTANCE
    diverse = action_distance >= DIVERSE_ACTION_DISTANCE
    similar_median = (
        float(np.median(future_distance[similar])) if similar.any() else None
    )
    diverse_median = (
        float(np.median(future_distance[diverse])) if diverse.any() else None
    )
    ratio = (
        diverse_median / similar_median
        if similar_median is not None
        and diverse_median is not None
        and similar_median > 0
        else None
    )
    diverse_indices = (
        np.unique(pairs[diverse].reshape(-1))
        if diverse.any()
        else np.empty(0, dtype=np.int64)
    )
    diverse_routes = (
        np.unique(windows["recording"][diverse_indices].astype(str))
        if len(diverse_indices)
        else np.empty(0)
    )
    diverse_robots = (
        np.unique(windows["robot"][diverse_indices].astype(str))
        if len(diverse_indices)
        else np.empty(0)
    )
    pair_routes = (
        windows["recording"][pairs[:, 0]].astype(str)
        if len(pairs)
        else np.empty(0)
    )
    consistency = route_consistency(
        pair_routes,
        action_distance,
        future_distance,
    )
    checks = {
        "diverse_pairs_at_least_300": int(diverse.sum()) >= 300,
        "diverse_anchor_windows_at_least_300": len(diverse_indices) >= 300,
        "diverse_routes_at_least_12": len(diverse_routes) >= 12,
        "action_future_rank_correlation_at_least_0_15": bool(
            np.isfinite(association) and association >= 0.15
        ),
        "diverse_vs_similar_ratio_at_least_1_10": bool(
            ratio is not None and ratio >= 1.10
        ),
        "majority_routes_positive_association": bool(
            consistency["eligible_routes"] >= 12
            and consistency["positive_association_routes"]
            > consistency["eligible_routes"] / 2
        ),
    }
    go = all(checks.values())
    result = {
        "schema": "highres_dual_pose_matched_natural_branching_v1",
        "status": "complete",
        "decision": (
            "go_one_pose_matched_highres_dual_pseudobranch_pilot"
            if go
            else "no_go_local_trajectories_lack_strict_counterfactual_coverage"
        ),
        "go": go,
        "checks": checks,
        "thresholds": {
            "same_original_recording": True,
            "different_trajectory_fragment": True,
            "pose_radius_m": POSE_RADIUS_M,
            "yaw_radius_deg": YAW_RADIUS_DEG,
            "minimum_state_cosine": MINIMUM_STATE_COSINE,
            "maximum_partners_per_anchor": MAXIMUM_PARTNERS_PER_ANCHOR,
            "similar_action_standardized_rms": SIMILAR_ACTION_DISTANCE,
            "diverse_action_standardized_rms": DIVERSE_ACTION_DISTANCE,
        },
        "windows": int(len(signature)),
        "trajectories": int(len(np.unique(windows["trajectory"].astype(str)))),
        "recordings": int(len(np.unique(windows["recording"].astype(str)))),
        "robots": sorted(np.unique(windows["robot"].astype(str)).tolist()),
        "matching_funnel": {
            **funnel,
            "selected_pose_state_pairs": int(len(pairs)),
        },
        "selected_pairs": {
            "position_distance_m": quantiles(position_distance),
            "yaw_distance_deg": quantiles(np.degrees(yaw_distance)),
            "state_cosine": quantiles(state_cosine),
            "action_distance": quantiles(action_distance),
            "future_residual_distance": quantiles(future_distance),
            "action_future_rank_correlation": (
                float(association) if np.isfinite(association) else None
            ),
        },
        "similar_action_pairs": {
            "pairs": int(similar.sum()),
            "future_residual_distance": quantiles(future_distance[similar]),
        },
        "diverse_action_pairs": {
            "pairs": int(diverse.sum()),
            "anchor_windows": int(len(diverse_indices)),
            "recordings": int(len(diverse_routes)),
            "robots": sorted(diverse_robots.tolist()),
            "future_residual_distance": quantiles(future_distance[diverse]),
        },
        "diverse_vs_similar_median_future_residual_distance_ratio": ratio,
        "route_consistency": consistency,
        "inputs": {
            "latent_root": str(latent_root),
            "metadata_sha256": sha256_file(latent_root / "metadata.json"),
            "splits_sha256": sha256_file(latent_root / "splits.json"),
            "runner": str(Path(__file__).resolve()),
            "runner_sha256": sha256_file(Path(__file__).resolve()),
        },
        "runtime_seconds": float(time.perf_counter() - started),
        "outcome_labels_used": False,
        "pair_indices_persisted": False,
        "new_training_table_created": False,
        "sealed_validation_opened": False,
    }
    write_json_atomic(output, result)
    print(
        json.dumps(
            {
                "decision": result["decision"],
                "pairs": int(len(pairs)),
                "diverse_pairs": int(diverse.sum()),
                "diverse_routes": int(len(diverse_routes)),
                "association": result["selected_pairs"][
                    "action_future_rank_correlation"
                ],
                "ratio": ratio,
                "output": str(output),
                "output_sha256": sha256_file(output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(
            json.dumps(
                {
                    "status": "fail_closed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "sealed_validation_opened": False,
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        raise SystemExit(2) from error
