#!/usr/bin/env python3
"""Build a K17 RECON100 source with the dense-meta action feature contract."""

from __future__ import annotations

import argparse
import json
import math
import pickle
from pathlib import Path
import re
import sys
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from planning.nomad_action_adapter import nomad_waypoints_to_local_actions  # noqa: E402
from scoring.features import action_trajectory_features  # noqa: E402


HORIZON = 4
WAYPOINT_SPACING_M = 0.25


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("features", "labels"), required=True)
    parser.add_argument("--candidate", type=Path, default=Path("/home/rainbow/ndwm/DWM/reports/recon_h4_scorer_candidates_v1/strict_external100.npz"))
    parser.add_argument("--manifest", type=Path, default=Path("/home/rainbow/ndwm/DWM/reports/recon_h4_scorer_candidates_v1/strict_external100.json"))
    parser.add_argument("--legacy-labels", type=Path, default=Path("/home/rainbow/ndwm/DWM/reports/recon_h4_scorer_labels_v2/strict_external100/labels.npz"))
    parser.add_argument("--trajectory-root", type=Path, default=Path("/home/rainbow/rrnwm/data/recon"))
    parser.add_argument("--feature-output", type=Path, required=True)
    parser.add_argument("--label-output", type=Path)
    return parser.parse_args()


def write_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def trimmed_mean75(pool: np.ndarray) -> np.ndarray:
    center = pool.mean(axis=0)
    distance = np.mean(np.sum((pool - center[None]) ** 2, axis=2), axis=1)
    keep = max(2, int(np.ceil(0.75 * len(pool))))
    return pool[np.argsort(distance, kind="stable")[:keep]].mean(axis=0).astype(np.float32)


def medoid_index(pool: np.ndarray) -> int:
    flat = pool.reshape(len(pool), -1)
    distance = np.linalg.norm(flat[:, None] - flat[None], axis=-1)
    return int(np.argmin(distance.mean(axis=1)))


def action_set_features(candidates_raw: np.ndarray, reference_raw: np.ndarray) -> np.ndarray:
    actions = nomad_waypoints_to_local_actions(candidates_raw, dataset="recon")
    reference_actions = nomad_waypoints_to_local_actions(reference_raw, dataset="recon")
    with torch.inference_mode():
        candidate_base, _ = action_trajectory_features(torch.from_numpy(actions))
        reference_base, _ = action_trajectory_features(torch.from_numpy(reference_actions))
    candidate_base = candidate_base.numpy().astype(np.float32)
    reference_base = reference_base.numpy().astype(np.float32)
    center = np.median(reference_base, axis=1, keepdims=True)
    scale = np.maximum(reference_base.std(axis=1, keepdims=True), 1.0e-5)
    relative = np.clip((candidate_base - center) / scale, -8.0, 8.0)
    candidates_m = candidates_raw * WAYPOINT_SPACING_M
    reference_m = reference_raw * WAYPOINT_SPACING_M
    candidate_flat = candidates_m.reshape(candidates_m.shape[0], candidates_m.shape[1], -1)
    reference_flat = reference_m.reshape(reference_m.shape[0], reference_m.shape[1], -1)
    median_path = np.median(reference_flat, axis=1, keepdims=True)
    medoid = np.asarray([medoid_index(value) for value in reference_m])
    medoid_path = reference_flat[np.arange(len(reference_flat)), medoid][:, None]

    def scalar(flat: np.ndarray, pool: np.ndarray) -> np.ndarray:
        return np.stack((np.linalg.norm(flat - median_path, axis=-1),
                         np.linalg.norm(flat - medoid_path, axis=-1),
                         np.linalg.norm(pool[:, :, -1], axis=-1)), axis=-1)

    candidate_scalar = scalar(candidate_flat, candidates_m)
    reference_scalar = scalar(reference_flat, reference_m)
    scalar_center = np.median(reference_scalar, axis=1, keepdims=True)
    scalar_scale = np.maximum(reference_scalar.std(axis=1, keepdims=True), 1.0e-5)
    set_relative = np.clip((candidate_scalar - scalar_center) / scalar_scale, -8.0, 8.0)
    result = np.concatenate((candidate_base, relative, set_relative), axis=-1).astype(np.float32)
    if result.shape[-1] != 83:
        raise RuntimeError(f"action-set dimension changed: {result.shape}")
    return result


def wrap_angle(value: np.ndarray) -> np.ndarray:
    return (value + np.pi) % (2.0 * np.pi) - np.pi


def motion_parts(candidates_m: np.ndarray, records: list[dict[str, Any]], root: Path) -> tuple[np.ndarray, np.ndarray]:
    states = []
    specifics = []
    cache: dict[str, dict[str, Any]] = {}
    horizon = candidates_m.shape[2]
    steps = np.arange(1, horizon + 1, dtype=np.float32)[:, None]
    for group, record in enumerate(records):
        route = str(record["traj_name"])
        if route not in cache:
            with (root / route / "traj_data.pkl").open("rb") as handle:
                cache[route] = pickle.load(handle)
        trajectory = cache[route]
        timestep = int(record["curr_time"])
        positions = np.asarray(trajectory["position"][timestep - 3:timestep + 1, :2], dtype=np.float32)
        yaw = np.asarray(trajectory["yaw"][timestep - 3:timestep + 1], dtype=np.float32)
        global_velocity = np.diff(positions, axis=0)
        cosine, sine = math.cos(float(yaw[-1])), math.sin(float(yaw[-1]))
        rotation = np.asarray([[cosine, sine], [-sine, cosine]], dtype=np.float32)
        local_velocity = global_velocity @ rotation.T
        speed = np.linalg.norm(local_velocity, axis=1)
        yaw_delta = wrap_angle(np.diff(yaw)).astype(np.float32)
        weights = np.asarray([0.2, 0.3, 0.5], dtype=np.float32)
        velocity_weighted = (local_velocity * weights[:, None]).sum(axis=0)
        speed_weighted = float((speed * weights).sum())
        yaw_rate = float((yaw_delta * weights).sum())
        expected_last = steps * local_velocity[-1]
        expected_weighted = steps * velocity_weighted
        expected_twist = np.zeros((horizon, 2), dtype=np.float32)
        for future_step in range(horizon):
            heading = (future_step + 0.5) * yaw_rate
            increment = speed_weighted * np.asarray([math.cos(heading), math.sin(heading)], dtype=np.float32)
            if future_step:
                expected_twist[future_step] = expected_twist[future_step - 1]
            expected_twist[future_step] += increment
        states.append(np.concatenate((local_velocity.reshape(-1), speed, yaw_delta,
                                      np.asarray([speed.mean(), speed[-1], speed_weighted,
                                                  speed[-1] - speed[-2], yaw_delta.mean(), yaw_rate],
                                                 dtype=np.float32))))
        candidates = candidates_m[group]
        candidate_steps = np.diff(np.concatenate((np.zeros((len(candidates), 1, 2), dtype=np.float32), candidates), axis=1), axis=1)
        candidate_speed = np.linalg.norm(candidate_steps, axis=-1)

        def residual(expected: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            distance = np.linalg.norm(candidates - expected[None], axis=-1)
            return distance.mean(axis=1), distance[:, -1]

        last_mean, last_final = residual(expected_last)
        weighted_mean, weighted_final = residual(expected_weighted)
        twist_mean, twist_final = residual(expected_twist)
        expected_direction = velocity_weighted / max(float(np.linalg.norm(velocity_weighted)), 1.0e-6)
        candidate_direction = candidates[:, -1] / np.maximum(np.linalg.norm(candidates[:, -1], axis=1, keepdims=True), 1.0e-6)
        specifics.append(np.stack((last_mean, last_final, weighted_mean, weighted_final,
                                   twist_mean, twist_final,
                                   np.abs(candidate_speed.mean(axis=1) - speed_weighted),
                                   np.abs(candidate_speed[:, 0] - speed[-1]),
                                   np.abs(np.linalg.norm(candidates[:, -1], axis=1) / horizon - speed_weighted),
                                   1.0 - candidate_direction @ expected_direction), axis=-1))
    return np.stack(states).astype(np.float32), np.stack(specifics).astype(np.float32)


def history_features(candidates_raw: np.ndarray, reference_raw: np.ndarray,
                     records: list[dict[str, Any]], root: Path) -> np.ndarray:
    state, candidate_specific = motion_parts(candidates_raw * WAYPOINT_SPACING_M, records, root)
    _, reference_specific = motion_parts(reference_raw * WAYPOINT_SPACING_M, records, root)
    center = np.median(reference_specific, axis=1, keepdims=True)
    scale = np.maximum(reference_specific.std(axis=1, keepdims=True), 1.0e-5)
    relative = np.clip((candidate_specific - center) / scale, -8.0, 8.0)
    repeated = np.repeat(state[:, None], candidates_raw.shape[1], axis=1)
    result = np.concatenate((repeated, candidate_specific, relative), axis=-1).astype(np.float32)
    if result.shape[-1] != 38:
        raise RuntimeError(f"history feature dimension changed: {result.shape}")
    return result


def feature_stage(args: argparse.Namespace) -> None:
    records = list(json.loads(args.manifest.read_text(encoding="utf-8"))["records"])
    with np.load(args.candidate, allow_pickle=False) as payload:
        reference = payload["candidates"].astype(np.float32)
    if reference.shape != (len(records), 16, HORIZON, 2):
        raise RuntimeError(f"RECON100 candidate shape changed: {reference.shape}")
    incumbent = np.stack([trimmed_mean75(value) for value in reference])
    candidates = np.concatenate((incumbent[:, None], reference), axis=1).astype(np.float32)
    action_set = action_set_features(candidates, reference)
    motion = history_features(candidates, reference, records, args.trajectory_root)
    raw = np.concatenate((action_set, motion), axis=-1).astype(np.float32)
    action = np.concatenate((raw, raw - raw[:, :1]), axis=-1).astype(np.float32)
    if action.shape != (len(records), 17, 242):
        raise RuntimeError(f"action feature contract changed: {action.shape}")
    record_key = np.asarray([f"{row['traj_name']}:{int(row['curr_time'])}" for row in records])
    family = np.asarray([re.sub(r"_r\d+$", "", str(row["traj_name"])) for row in records])
    write_npz(args.feature_output, candidates=candidates, action_features=action,
              record_key=record_key, family=family)
    print(json.dumps({"stage": "features", "states": len(records), "candidates": 17,
                      "labels_accessed": False, "output": str(args.feature_output)}))


def metric_labels(candidates: np.ndarray, ground_truth: np.ndarray,
                  spacing: float, ate_weight: float, endpoint_weight: float) -> dict[str, np.ndarray]:
    candidates_m = candidates.astype(np.float64) * spacing
    ground_truth = ground_truth.astype(np.float64)
    error = candidates_m - ground_truth[:, None]
    squared = np.sum(error * error, axis=-1)
    ate = np.sqrt(np.mean(squared, axis=-1))
    endpoint = np.sqrt(squared[..., -1])
    rpe = np.sqrt(np.mean(np.sum((np.diff(candidates_m, axis=2) - np.diff(ground_truth, axis=1)[:, None]) ** 2, axis=-1), axis=-1))
    objective = ate_weight * ate + endpoint_weight * endpoint
    values = (ate, rpe, endpoint, objective)
    delta = [value - value[:, :1] for value in values]
    return {"values": np.stack(values, axis=-1).astype(np.float32),
            "targets": np.concatenate((np.stack(delta, axis=-1),
                                        (delta[0] < -1e-7)[..., None],
                                        (delta[0] > 1e-7)[..., None],
                                        ((delta[0] < -1e-7) & (delta[1] <= 0) & (delta[2] <= 0) & (delta[3] <= 0))[..., None],
                                        ((delta[1] > .005) | (delta[2] > .020) | (delta[3] > .010))[..., None]), axis=-1).astype(np.float32)}


def label_stage(args: argparse.Namespace) -> None:
    if args.label_output is None:
        raise ValueError("--label-output is required for labels stage")
    with np.load(args.feature_output, allow_pickle=False) as source:
        candidates = source["candidates"].copy()
        record_key = source["record_key"].astype(str)
        family = source["family"].astype(str)
    with np.load(args.legacy_labels, allow_pickle=False) as legacy:
        if not np.array_equal(record_key, legacy["record_key"].astype(str)):
            raise RuntimeError("legacy label order differs from K17 source")
        labels = metric_labels(candidates, legacy["ground_truth_local_m"],
                               float(legacy["waypoint_spacing_m"]),
                               float(legacy["objective_ate_weight"]),
                               float(legacy["objective_endpoint_weight"]))
    write_npz(args.label_output, record_key=record_key, family=family,
              targets=labels["targets"], absolute_metrics=labels["values"])
    print(json.dumps({"stage": "labels", "states": len(record_key), "labels_accessed": True,
                      "output": str(args.label_output)}))


def main() -> None:
    args = parse_args()
    if args.stage == "features":
        feature_stage(args)
    else:
        label_stage(args)


if __name__ == "__main__":
    main()
