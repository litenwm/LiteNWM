#!/usr/bin/env python3
"""Replace four provenance columns with route-disjoint path-ranking features.

This is a spent-development diagnostic table.  The predicted path is learned
only from ordinary logged futures after removing every route present in the
union65 evaluation table; it uses no counterfactual action outcomes.
"""

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
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite: {output}")

    with np.load(args.table, allow_pickle=True) as archive:
        data = {key: archive[key] for key in archive.files}
    with np.load(args.path_predictions, allow_pickle=True) as archive:
        predictions = {key: archive[key] for key in archive.files}

    record_key = data["record_key"].astype(str)
    if not np.array_equal(record_key, predictions["record_key"].astype(str)):
        raise ValueError("record order mismatch")

    feature = data["runtime_pair_feature"].astype(np.float32).copy()
    names = data["feature_names"].astype(str).copy()
    candidates_m = (
        data["candidates_feature_waypoint_units"].astype(np.float32) * 0.25
    )
    predicted_m = predictions["route_disjoint_predicted_path_m"].astype(np.float32)
    distance = np.sqrt(
        np.mean(np.sum((candidates_m - predicted_m[:, None]) ** 2, axis=-1), axis=-1)
    ).astype(np.float32)
    formal = data["formal_v70_index"].astype(np.int64)
    formal_distance = distance[np.arange(len(distance)), formal]
    advantage = (formal_distance[:, None] - distance).astype(np.float32)
    order = np.argsort(distance, axis=1)
    rank = np.empty_like(order, dtype=np.float32)
    rank[np.arange(len(rank))[:, None], order] = np.arange(
        distance.shape[1], dtype=np.float32
    )
    rank /= max(distance.shape[1] - 1, 1)
    best_margin = (distance.min(axis=1, keepdims=True) - distance).astype(np.float32)

    replaced = ["source_0", "source_1", "source_2", "source_3"]
    index = [int(np.flatnonzero(names == name)[0]) for name in replaced]
    values = [distance, advantage, rank, best_margin]
    new_names = [
        "route_disjoint_path_rmse_m",
        "route_disjoint_path_advantage_vs_formal_m",
        "route_disjoint_path_rank_fraction",
        "route_disjoint_path_best_margin_m",
    ]
    for column, value, name in zip(index, values, new_names, strict=True):
        feature[..., column] = value
        names[column] = name

    data["runtime_pair_feature"] = feature
    data["feature_names"] = names
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **data)
    print(
        json.dumps(
            {
                "output": str(output),
                "sha256": sha256_file(output),
                "shape": list(feature.shape),
                "replaced_columns": dict(zip(replaced, new_names, strict=True)),
                "counterfactual_action_outcomes_used": False,
                "path_prediction_regime": "dense ordinary logs, union-route-disjoint",
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
