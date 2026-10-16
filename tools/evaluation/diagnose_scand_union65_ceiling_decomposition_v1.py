#!/usr/bin/env python3
"""Decompose union65 oracle value into recall, reranking, and rejection ceilings."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--path-predictions", type=Path, required=True)
    parser.add_argument("--v4-predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def perfect_rerank(gain: np.ndarray, order: np.ndarray, k: int) -> float:
    values = np.take_along_axis(gain, order[:, :k], axis=1)
    return float(values.max(axis=1).mean())


def main() -> None:
    args = parse_args()
    with np.load(args.table, allow_pickle=True) as archive:
        data = {key: archive[key] for key in archive.files}
    with np.load(args.path_predictions, allow_pickle=True) as archive:
        path = {key: archive[key] for key in archive.files}
    with np.load(args.v4_predictions, allow_pickle=True) as archive:
        v4 = {key: archive[key] for key in archive.files}

    record = data["record_key"].astype(str)
    if not np.array_equal(record, path["record_key"].astype(str)):
        raise ValueError("path record order mismatch")
    if not np.array_equal(record, v4["record_key"].astype(str)):
        raise ValueError("v4 record order mismatch")

    n = len(record)
    formal = data["formal_v70_index"].astype(np.int64)
    mask = data["physical_support_pool"].astype(bool).copy()
    mask[np.arange(n), formal] = True
    ate = data["true_metrics_m"][:, :, 0].astype(np.float32)
    formal_ate = ate[np.arange(n), formal]
    gain = formal_ate[:, None] - ate
    candidates_m = (
        data["candidates_feature_waypoint_units"].astype(np.float32) * 0.25
    )

    result: dict[str, object] = {
        "schema": "scand_union65_ceiling_decomposition_v1",
        "status": "complete_spent_development_diagnostic_only",
        "counterfactual_action_outcomes_used": False,
        "states": n,
        "full_oracle_gain_m": float(
            np.max(np.where(mask, gain, -np.inf), axis=1).mean()
        ),
        "inputs": {
            "table": {"path": str(args.table), "sha256": sha256_file(args.table)},
            "path_predictions": {
                "path": str(args.path_predictions),
                "sha256": sha256_file(args.path_predictions),
            },
            "v4_predictions": {
                "path": str(args.v4_predictions),
                "sha256": sha256_file(args.v4_predictions),
            },
        },
    }

    rankings: dict[str, np.ndarray] = {}
    for regime, key in (
        ("route_disjoint_path", "route_disjoint_predicted_path_m"),
        ("trajectory_disjoint_path", "trajectory_disjoint_predicted_path_m"),
    ):
        predicted = path[key].astype(np.float32)
        distance = np.sqrt(
            np.mean(
                np.sum((candidates_m - predicted[:, None]) ** 2, axis=-1),
                axis=-1,
            )
        )
        order = np.argsort(-np.where(mask, -distance, -np.inf), axis=1)
        selected_gain = gain[np.arange(n), order[:, 0]]
        item: dict[str, float] = {
            "raw_top1_gain_m": float(selected_gain.mean()),
            "perfect_accept_reject_gain_m": float(
                np.maximum(selected_gain, 0.0).mean()
            ),
            "positive_choice_fraction": float(np.mean(selected_gain > 0.0)),
        }
        for k in (1, 3, 5, 10, 20):
            item[f"perfect_rerank_top{k}_gain_m"] = perfect_rerank(gain, order, k)
        result[regime] = item
        rankings[regime] = order

    v4_score = np.where(mask, v4["utility_score"].astype(np.float32), -np.inf)
    v4_order = np.argsort(-v4_score, axis=1)
    v4_selected_gain = gain[np.arange(n), v4["safe_selected_index"].astype(int)]
    v4_result: dict[str, float] = {
        "actual_safe_gain_m": float(v4_selected_gain.mean()),
        "perfect_accept_reject_same_choices_gain_m": float(
            np.maximum(v4_selected_gain, 0.0).mean()
        ),
    }
    path_order = rankings["route_disjoint_path"]
    for k in (1, 3, 5, 10, 20):
        v4_result[f"perfect_rerank_top{k}_gain_m"] = perfect_rerank(
            gain, v4_order, k
        )
        union_best = np.asarray(
            [
                gain[row, np.unique(np.r_[v4_order[row, :k], path_order[row, :k]])].max()
                for row in range(n)
            ],
            dtype=np.float32,
        )
        v4_result[f"perfect_rerank_union_top{k}_gain_m"] = float(union_best.mean())
    result["v4"] = v4_result

    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(output), "full_oracle_gain_m": result["full_oracle_gain_m"]}))


if __name__ == "__main__":
    main()
