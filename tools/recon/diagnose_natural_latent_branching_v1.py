#!/usr/bin/env python3
"""Test whether existing trajectories contain usable natural action branches.

The diagnostic searches for mutual cross-recording nearest neighbors in a
pooled DINO history-latent representation.  It then asks whether visually
matched windows with different H4 actions have systematically different true
future-latent residuals.  Only aggregate statistics are written; no new
training table or pair registry is created.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import time
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
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


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(canonical_json(value) + "\n", encoding="utf-8")
    temporary.replace(path)


def recording_id(trajectory_id: str) -> str:
    return re.sub(r"_r\d+$", "", trajectory_id)


def pool_latent(value: np.ndarray) -> np.ndarray:
    """Pool 16x16x384 tokens to a 4x4x32 route-matching signature."""
    array = value.astype(np.float32, copy=False)
    array = array.reshape(len(array), 16, 16, 32, 12).mean(axis=-1)
    array = array.reshape(len(array), 4, 4, 4, 4, 32).mean(axis=(2, 4))
    return array.reshape(len(array), 512).astype(np.float32)


def normalize(value: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(value, axis=1, keepdims=True)
    return value / np.maximum(norm, 1.0e-8)


def rank_correlation(first: np.ndarray, second: np.ndarray) -> float | None:
    if len(first) < 3:
        return None
    first_rank = np.empty(len(first), dtype=np.float64)
    second_rank = np.empty(len(second), dtype=np.float64)
    first_rank[np.argsort(first, kind="stable")] = np.arange(len(first))
    second_rank[np.argsort(second, kind="stable")] = np.arange(len(second))
    return float(np.corrcoef(first_rank, second_rank)[0, 1])


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


def load_windows(
    latent_root: Path,
    split: str,
) -> dict[str, np.ndarray]:
    metadata = read_json(latent_root / "metadata.json")
    splits = read_json(latent_root / "splits.json")
    allowed = set(map(str, splits[split]))
    current_rows = []
    history_rows = []
    future_rows = []
    action_rows = []
    trajectory_rows = []
    recording_rows = []
    timestep_rows = []
    for item in metadata["trajectories"]:
        trajectory_id = str(item["trajectory_id"])
        if trajectory_id not in allowed:
            continue
        path = latent_root / "shards" / str(item["shard_file"])
        with np.load(path, allow_pickle=False) as payload:
            latent = np.asarray(payload["patch_latents"])
            actions = np.asarray(payload["actions_continuous"], dtype=np.float32)
        if len(latent) < 8:
            continue
        pooled = pool_latent(latent)
        timestep = np.arange(3, len(latent) - 4, dtype=np.int32)
        current_rows.append(pooled[timestep])
        history_rows.append(pooled[timestep] - pooled[timestep - 3])
        future_rows.append(pooled[timestep + 4] - pooled[timestep])
        action_rows.append(
            np.stack(
                [actions[index : index + 4].reshape(-1) for index in timestep],
                axis=0,
            )
        )
        trajectory_rows.extend([trajectory_id] * len(timestep))
        recording_rows.extend([recording_id(trajectory_id)] * len(timestep))
        timestep_rows.append(timestep)
    require(current_rows, "no windows found")
    return {
        "current": np.concatenate(current_rows).astype(np.float32),
        "history_motion": np.concatenate(history_rows).astype(np.float32),
        "future_residual": np.concatenate(future_rows).astype(np.float32),
        "actions": np.concatenate(action_rows).astype(np.float32),
        "trajectory": np.asarray(trajectory_rows),
        "recording": np.asarray(recording_rows),
        "timestep": np.concatenate(timestep_rows),
    }


def build_signature(windows: dict[str, np.ndarray]) -> np.ndarray:
    current = windows["current"]
    centered = normalize(current - current.mean(axis=0, keepdims=True))
    motion = windows["history_motion"]
    motion = normalize(motion - motion.mean(axis=0, keepdims=True))
    signature = np.concatenate((centered, 0.25 * motion), axis=1)
    return normalize(signature).astype(np.float32)


def cross_recording_random_pairs(
    recording_index: np.ndarray,
    count: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    size = len(recording_index)
    first = rng.integers(0, size, size=count, dtype=np.int64)
    second = rng.integers(0, size, size=count, dtype=np.int64)
    same = recording_index[first] == recording_index[second]
    while same.any():
        second[same] = rng.integers(0, size, size=int(same.sum()), dtype=np.int64)
        same = recording_index[first] == recording_index[second]
    return first, second


def nearest_cross_recording(
    signature: np.ndarray,
    recording_index: np.ndarray,
    top_k: int,
    chunk_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    database = torch.from_numpy(signature).to(device)
    recording_tensor = torch.from_numpy(recording_index).to(device)
    all_indices = []
    all_values = []
    with torch.inference_mode():
        for start in range(0, len(signature), chunk_size):
            end = min(start + chunk_size, len(signature))
            query = database[start:end]
            similarity = query @ database.T
            same = (
                recording_tensor[start:end, None]
                == recording_tensor[None, :]
            )
            similarity.masked_fill_(same, -torch.inf)
            values, indices = torch.topk(
                similarity,
                k=top_k,
                dim=1,
                largest=True,
                sorted=True,
            )
            all_indices.append(indices.cpu().numpy().astype(np.int32))
            all_values.append(values.cpu().numpy().astype(np.float32))
    return np.concatenate(all_indices), np.concatenate(all_values)


def mutual_pairs(
    neighbor: np.ndarray,
    similarity: np.ndarray,
    minimum_similarity: float,
) -> tuple[np.ndarray, np.ndarray]:
    directed: dict[tuple[int, int], float] = {}
    for first in range(len(neighbor)):
        for position, second in enumerate(neighbor[first]):
            value = float(similarity[first, position])
            if value >= minimum_similarity:
                directed[(first, int(second))] = value
    pairs = []
    values = []
    for (first, second), value in directed.items():
        if first < second and (second, first) in directed:
            pairs.append((first, second))
            values.append(min(value, directed[(second, first)]))
    if not pairs:
        return np.empty((0, 2), dtype=np.int64), np.empty(0, dtype=np.float32)
    return np.asarray(pairs, dtype=np.int64), np.asarray(values, dtype=np.float32)


def main() -> None:
    args = parse_args()
    contract_path = args.contract.resolve()
    output = args.output.resolve()
    require(not output.exists(), f"refusing to overwrite: {output}")
    contract = read_json(contract_path)
    require(
        contract.get("status") == "frozen_before_single_natural_branching_diagnostic",
        "contract not frozen",
    )
    latent_entry = contract["input"]["latent_root"]
    latent_root = (ROOT / latent_entry["path"]).resolve()
    require(latent_root.is_dir(), f"missing latent root: {latent_root}")
    require(
        sha256_file(latent_root / "metadata.json") == latent_entry["metadata_sha256"],
        "latent metadata changed",
    )
    require(
        sha256_file(latent_root / "splits.json") == latent_entry["splits_sha256"],
        "latent splits changed",
    )
    device = torch.device(args.device)
    require(device.type != "cuda" or torch.cuda.is_available(), "CUDA unavailable")
    started = time.perf_counter()
    windows = load_windows(latent_root, contract["input"]["split"])
    signature = build_signature(windows)
    unique_recording, recording_index = np.unique(
        windows["recording"].astype(str),
        return_inverse=True,
    )
    random_first, random_second = cross_recording_random_pairs(
        recording_index,
        int(contract["matching"]["random_pair_count"]),
        int(contract["matching"]["seed"]),
    )
    random_similarity = np.sum(
        signature[random_first] * signature[random_second],
        axis=1,
    )
    action_scale = np.std(windows["actions"], axis=0)
    action_scale = np.maximum(action_scale, 1.0e-4)

    def action_distance(first: np.ndarray, second: np.ndarray) -> np.ndarray:
        difference = (
            windows["actions"][first] - windows["actions"][second]
        ) / action_scale
        return np.sqrt(np.mean(difference * difference, axis=1))

    random_action_distance = action_distance(random_first, random_second)
    state_threshold = float(
        np.quantile(
            random_similarity,
            float(contract["matching"]["random_state_similarity_quantile"]),
        )
    )
    similar_action_threshold = float(
        np.quantile(
            random_action_distance,
            float(contract["matching"]["similar_action_quantile"]),
        )
    )
    diverse_action_threshold = float(
        np.quantile(
            random_action_distance,
            float(contract["matching"]["diverse_action_quantile"]),
        )
    )
    neighbor, similarity = nearest_cross_recording(
        signature,
        recording_index,
        int(contract["matching"]["top_k"]),
        int(contract["matching"]["query_chunk_size"]),
        device,
    )
    pairs, pair_similarity = mutual_pairs(
        neighbor,
        similarity,
        state_threshold,
    )
    if len(pairs):
        pair_action_distance = action_distance(pairs[:, 0], pairs[:, 1])
        residual = windows["future_residual"]
        residual_scale = float(np.sqrt(np.mean(residual * residual)))
        future_difference = residual[pairs[:, 0]] - residual[pairs[:, 1]]
        pair_future_distance = np.sqrt(
            np.mean(future_difference * future_difference, axis=1)
        ) / max(residual_scale, 1.0e-8)
    else:
        pair_action_distance = np.empty(0, dtype=np.float32)
        pair_future_distance = np.empty(0, dtype=np.float32)
    similar_mask = pair_action_distance <= similar_action_threshold
    diverse_mask = pair_action_distance >= diverse_action_threshold
    similar_future = pair_future_distance[similar_mask]
    diverse_future = pair_future_distance[diverse_mask]
    future_ratio = (
        float(np.median(diverse_future) / np.median(similar_future))
        if len(diverse_future) and len(similar_future)
        and float(np.median(similar_future)) > 0.0
        else None
    )
    diverse_anchors = (
        np.unique(pairs[diverse_mask].reshape(-1))
        if diverse_mask.any()
        else np.empty(0, dtype=np.int64)
    )
    diverse_recordings = (
        np.unique(recording_index[diverse_anchors])
        if len(diverse_anchors)
        else np.empty(0, dtype=np.int64)
    )
    association = rank_correlation(pair_action_distance, pair_future_distance)
    gates = contract["go_gate"]
    checks = {
        "mutual_state_pairs": len(pairs)
        >= int(gates["minimum_mutual_state_pairs"]),
        "diverse_action_pairs": int(diverse_mask.sum())
        >= int(gates["minimum_diverse_action_pairs"]),
        "diverse_anchor_windows": len(diverse_anchors)
        >= int(gates["minimum_diverse_anchor_windows"]),
        "diverse_recordings": len(diverse_recordings)
        >= int(gates["minimum_diverse_recordings"]),
        "action_future_rank_correlation": association is not None
        and association
        >= float(gates["minimum_action_future_rank_correlation"]),
        "diverse_vs_similar_future_ratio": future_ratio is not None
        and future_ratio
        >= float(gates["minimum_diverse_vs_similar_future_ratio"]),
    }
    go = all(checks.values())
    result = {
        "schema": "natural_cross_recording_latent_branching_diagnostic_v1",
        "status": "complete",
        "decision": (
            "go_existing_trajectories_support_pseudo_branching_pilot"
            if go
            else "no_go_existing_trajectories_do_not_identify_counterfactual_action_futures"
        ),
        "go": go,
        "checks": checks,
        "windows": int(len(signature)),
        "trajectories": int(len(np.unique(windows["trajectory"].astype(str)))),
        "recordings": int(len(unique_recording)),
        "representation": contract["representation"],
        "thresholds_derived_from_random_cross_recording_pairs": {
            "state_similarity": state_threshold,
            "similar_action_distance": similar_action_threshold,
            "diverse_action_distance": diverse_action_threshold,
        },
        "random_cross_recording": {
            "pairs": int(len(random_first)),
            "state_similarity": quantiles(random_similarity),
            "action_distance": quantiles(random_action_distance),
        },
        "mutual_high_similarity_pairs": {
            "pairs": int(len(pairs)),
            "state_similarity": quantiles(pair_similarity),
            "action_distance": quantiles(pair_action_distance),
            "future_residual_distance": quantiles(pair_future_distance),
            "action_future_rank_correlation": association,
        },
        "similar_action_matched_pairs": {
            "pairs": int(similar_mask.sum()),
            "future_residual_distance": quantiles(similar_future),
        },
        "diverse_action_matched_pairs": {
            "pairs": int(diverse_mask.sum()),
            "anchor_windows": int(len(diverse_anchors)),
            "recordings": int(len(diverse_recordings)),
            "future_residual_distance": quantiles(diverse_future),
        },
        "diverse_vs_similar_median_future_residual_distance_ratio": future_ratio,
        "inputs": {
            "contract": {
                "path": str(contract_path),
                "sha256": sha256_file(contract_path),
            },
            "latent_root": {
                "path": str(latent_root),
                "metadata_sha256": sha256_file(latent_root / "metadata.json"),
                "splits_sha256": sha256_file(latent_root / "splits.json"),
            },
            "runner": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
        },
        "runtime_seconds": float(time.perf_counter() - started),
        "outcome_labels_used": False,
        "new_training_table_created": False,
        "sealed_validation_opened": False,
    }
    output.parent.mkdir(parents=True, exist_ok=False)
    write_json_atomic(output, result)
    print(
        canonical_json(
            {
                "decision": result["decision"],
                "windows": result["windows"],
                "mutual_pairs": len(pairs),
                "diverse_pairs": int(diverse_mask.sum()),
                "action_future_rank_correlation": association,
                "diverse_vs_similar_ratio": future_ratio,
                "output": str(output),
                "output_sha256": sha256_file(output),
            }
        )
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(
            canonical_json(
                {
                    "status": "fail_closed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "sealed_validation_opened": False,
                }
            ),
            file=sys.stderr,
        )
        raise SystemExit(2) from error
