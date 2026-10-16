#!/usr/bin/env python3
"""Open labels after NWM selection and report trajectory-ranking diagnostics."""

import argparse
import json
import math
import pickle
from pathlib import Path

import numpy as np
import yaml


def rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    result = np.empty(len(values), dtype=np.float64)
    result[order] = np.arange(len(values), dtype=np.float64)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--trajectory-root", type=Path, required=True)
    parser.add_argument("--data-config", type=Path, required=True)
    parser.add_argument("--spacing", type=float, required=True)
    args = parser.parse_args()

    rows = json.loads(args.manifest.read_text(encoding="utf-8"))["records"]
    stats = yaml.safe_load(args.data_config.read_text(encoding="utf-8"))["action_stats"]
    action_min = np.asarray(stats["min"], dtype=np.float64)
    action_max = np.asarray(stats["max"], dtype=np.float64)
    trace = np.load(args.trace, allow_pickle=False)
    results = []

    for index, row in enumerate(rows):
        with (args.trajectory_root / row["trajectory_id"] / "traj_data.pkl").open("rb") as handle:
            raw = pickle.load(handle)
        current = int(row["current_t"])
        times = [current] + [int(value) for value in row["gt_times"]]
        position = np.asarray(raw["position"], dtype=np.float64)[times, :2]
        yaw = float(np.asarray(raw["yaw"]).reshape(-1)[current])
        rotation = np.asarray(
            [[math.cos(yaw), -math.sin(yaw)],
             [math.sin(yaw), math.cos(yaw)]],
            dtype=np.float64,
        )
        ground_truth = ((position - position[0]).dot(rotation))[1:]
        parameters = trace["candidate_parameter"][index].astype(np.float64)
        native_xy = (parameters[:, :2] + 1.0) / 2.0
        native_xy = native_xy * (action_max - action_min) + action_min
        steps = np.arange(1, len(row["gt_times"]) + 1, dtype=np.float64)
        candidate_path = steps[None, :, None] * native_xy[:, None, :]
        candidate_path *= args.spacing
        candidate_error = candidate_path - ground_truth[None, :, :]
        candidate_ate = np.sqrt(np.mean(np.sum(candidate_error ** 2, axis=-1), axis=-1))
        candidate_ate /= args.spacing
        selected_error = trace["path_m"][index] - ground_truth
        selected_ate = np.sqrt(np.mean(np.sum(selected_error ** 2, axis=-1)))
        selected_ate /= args.spacing
        loss = trace["mean_candidate_lpips"][index].astype(np.float64)
        best = int(np.argmin(loss))
        rho = float(np.corrcoef(rank(loss), rank(candidate_ate))[0, 1])
        results.append({
            "record_key": str(trace["record_key"][index]),
            "selected_ate": float(selected_ate),
            "candidate_oracle_ate": float(candidate_ate.min()),
            "lpips_best_candidate_ate": float(candidate_ate[best]),
            "lpips_ate_spearman": rho,
        })

    summary = {
        "states": results,
        "selected_ate_mean": float(np.mean([item["selected_ate"] for item in results])),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
