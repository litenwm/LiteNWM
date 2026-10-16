"""Strict H4 inner-partition and exclusion-manifest utilities.

The functions in this module deliberately operate on split metadata before
window generation.  In particular, SCAND is partitioned by undirected route
group, which keeps every physical bag, segment, and reverse traversal in one
inner partition.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any

import numpy as np


INNER_PARTITIONS = (
    "train-fit",
    "scorer-cal",
    "cem-cal",
    "internal-holdout",
)
OUTER_SPLITS = ("train", "val", "test")
DEFAULT_INNER_FRACTIONS = {
    "train-fit": 0.70,
    "scorer-cal": 0.10,
    "cem-cal": 0.10,
    "internal-holdout": 0.10,
}


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(canonical_json(dict(row)) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary.replace(path)


def normalize_scand_route(left: str, right: str) -> str:
    """Return the split-lock-compatible undirected SCAND route identifier."""

    return "__".join(sorted((left.strip().lower(), right.strip().lower())))


def parse_scand_trajectory_id(trajectory_id: str) -> dict[str, str]:
    """Parse a canonical SCAND trajectory identifier.

    Expected IDs have the form
    ``random_mdps_<survey>_<robot>_<from>_<to>_<dow>_<month>_<day>_<bag>_<segment>``.
    The physical bag ID excludes only the final segment suffix.
    """

    parts = str(trajectory_id).split("_")
    if len(parts) != 11 or parts[:2] != ["random", "mdps"]:
        raise ValueError(f"unsupported SCAND trajectory ID: {trajectory_id}")
    return {
        "survey": parts[2],
        "robot": parts[3],
        "left": parts[4],
        "right": parts[5],
        "bag_id": "_".join(parts[:-1]),
        "segment_id": parts[-1],
        "route_group": normalize_scand_route(parts[4], parts[5]),
    }


def recon_route_id(trajectory_id: str) -> str:
    """Strip only RECON repeat suffixes, matching the outer split lock."""

    return re.sub(r"_r\d+$", "", str(trajectory_id))


def validate_inner_fractions(
    fractions: Mapping[str, float],
) -> dict[str, float]:
    normalized = {partition: float(fractions[partition]) for partition in INNER_PARTITIONS}
    if any(not math.isfinite(value) or value <= 0.0 for value in normalized.values()):
        raise ValueError("every inner-partition fraction must be finite and positive")
    total = sum(normalized.values())
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1.0e-9):
        raise ValueError(f"inner-partition fractions must sum to one, got {total}")
    return normalized


def _partition_totals(
    assignment: Mapping[str, str],
    group_stats: Mapping[str, Mapping[str, int]],
) -> dict[str, dict[str, int]]:
    metrics = ("h4_windows", "trajectories", "bags", "route_groups")
    totals = {
        partition: {metric: 0 for metric in metrics}
        for partition in INNER_PARTITIONS
    }
    for group, partition in assignment.items():
        stats = group_stats[group]
        for metric in metrics:
            totals[partition][metric] += int(stats[metric])
    return totals


def _allocation_objective(
    assignment: Mapping[str, str],
    group_stats: Mapping[str, Mapping[str, int]],
    fractions: Mapping[str, float],
) -> float:
    totals = _partition_totals(assignment, group_stats)
    global_totals = {
        metric: sum(int(stats[metric]) for stats in group_stats.values())
        for metric in ("h4_windows", "trajectories", "bags", "route_groups")
    }
    weights = {
        "h4_windows": 1.0,
        "trajectories": 0.30,
        "bags": 0.30,
        "route_groups": 0.10,
    }
    score = 0.0
    for partition in INNER_PARTITIONS:
        if totals[partition]["route_groups"] == 0:
            score += 1.0e6
        for metric, weight in weights.items():
            target = fractions[partition] * global_totals[metric]
            scale = max(target, 1.0)
            score += weight * ((totals[partition][metric] - target) / scale) ** 2
    return score


def _allocation_objective_from_totals(
    totals: Mapping[str, Mapping[str, int]],
    global_totals: Mapping[str, int],
    fractions: Mapping[str, float],
) -> float:
    weights = {
        "h4_windows": 1.0,
        "trajectories": 0.30,
        "bags": 0.30,
        "route_groups": 0.10,
    }
    score = 0.0
    for partition in INNER_PARTITIONS:
        if int(totals[partition]["route_groups"]) == 0:
            score += 1.0e6
        for metric, weight in weights.items():
            target = fractions[partition] * int(global_totals[metric])
            scale = max(target, 1.0)
            score += weight * (
                (int(totals[partition][metric]) - target) / scale
            ) ** 2
    return score


def allocate_route_groups(
    group_stats: Mapping[str, Mapping[str, int]],
    *,
    fractions: Mapping[str, float] = DEFAULT_INNER_FRACTIONS,
    seed: int = 20260719,
    local_search_passes: int = 8,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Allocate route groups deterministically while balancing data volume.

    Only pre-label statistics are accepted: H4 availability, trajectory count,
    physical-bag count, and group count.
    """

    fractions = validate_inner_fractions(fractions)
    if len(group_stats) < len(INNER_PARTITIONS):
        raise ValueError(
            f"at least {len(INNER_PARTITIONS)} route groups are required"
        )
    required = {"h4_windows", "trajectories", "bags", "route_groups"}
    for group, stats in group_stats.items():
        missing = required.difference(stats)
        if missing:
            raise ValueError(f"group {group} is missing statistics: {sorted(missing)}")
        if int(stats["route_groups"]) != 1:
            raise ValueError(f"group {group} must contribute one route group")
        if int(stats["h4_windows"]) <= 0:
            raise ValueError(f"group {group} has no H4 windows")

    ordered_groups = sorted(
        group_stats,
        key=lambda group: (
            -int(group_stats[group]["h4_windows"]),
            -int(group_stats[group]["trajectories"]),
            -int(group_stats[group]["bags"]),
            sha256_text(f"route-order|{seed}|{group}"),
            group,
        ),
    )
    assignment: dict[str, str] = {}
    metrics = ("h4_windows", "trajectories", "bags", "route_groups")
    global_totals = {
        metric: sum(int(stats[metric]) for stats in group_stats.values())
        for metric in metrics
    }
    running_totals = {
        partition: {metric: 0 for metric in metrics}
        for partition in INNER_PARTITIONS
    }
    for index, group in enumerate(ordered_groups):
        empty = [
            partition
            for partition in INNER_PARTITIONS
            if partition not in assignment.values()
        ]
        remaining_after = len(ordered_groups) - index - 1
        options = (
            empty
            if empty and remaining_after < len(empty)
            else list(INNER_PARTITIONS)
        )
        candidates = []
        for partition in options:
            proposal_totals = {
                key: dict(value) for key, value in running_totals.items()
            }
            for metric in metrics:
                proposal_totals[partition][metric] += int(
                    group_stats[group][metric]
                )
            candidates.append(
                (
                    _allocation_objective_from_totals(
                        proposal_totals,
                        global_totals,
                        fractions,
                    ),
                    sha256_text(f"route-choice|{seed}|{group}|{partition}"),
                    partition,
                )
            )
        chosen = min(candidates)[2]
        assignment[group] = chosen
        for metric in metrics:
            running_totals[chosen][metric] += int(group_stats[group][metric])

    current_score = _allocation_objective_from_totals(
        running_totals,
        global_totals,
        fractions,
    )
    effective_local_search_passes = (
        max(0, int(local_search_passes)) if len(group_stats) <= 256 else 0
    )
    for _ in range(effective_local_search_passes):
        improved = False
        partition_group_counts = {
            partition: sum(value == partition for value in assignment.values())
            for partition in INNER_PARTITIONS
        }
        for group in sorted(
            assignment,
            key=lambda value: (sha256_text(f"move|{seed}|{value}"), value),
        ):
            source = assignment[group]
            if partition_group_counts[source] <= 1:
                continue
            best = (current_score, source)
            for target in INNER_PARTITIONS:
                if target == source:
                    continue
                proposal = dict(assignment)
                proposal[group] = target
                score = _allocation_objective(proposal, group_stats, fractions)
                candidate = (
                    score,
                    sha256_text(f"move-choice|{seed}|{group}|{target}"),
                    target,
                )
                incumbent = (
                    best[0],
                    sha256_text(f"move-choice|{seed}|{group}|{best[1]}"),
                    best[1],
                )
                if candidate < incumbent:
                    best = (score, target)
            if best[1] != source and best[0] < current_score - 1.0e-12:
                assignment[group] = best[1]
                partition_group_counts[source] -= 1
                partition_group_counts[best[1]] += 1
                current_score = best[0]
                improved = True

        groups = sorted(
            assignment,
            key=lambda value: (sha256_text(f"swap|{seed}|{value}"), value),
        )
        best_swap: tuple[float, str, str] | None = None
        for left_index, left in enumerate(groups):
            for right in groups[left_index + 1 :]:
                if assignment[left] == assignment[right]:
                    continue
                proposal = dict(assignment)
                proposal[left], proposal[right] = (
                    assignment[right],
                    assignment[left],
                )
                score = _allocation_objective(proposal, group_stats, fractions)
                candidate = (
                    score,
                    sha256_text(f"swap-choice|{seed}|{left}|{right}"),
                    left,
                    right,
                )
                if score < current_score - 1.0e-12 and (
                    best_swap is None
                    or candidate
                    < (
                        best_swap[0],
                        sha256_text(
                            f"swap-choice|{seed}|{best_swap[1]}|{best_swap[2]}"
                        ),
                        best_swap[1],
                        best_swap[2],
                    )
                ):
                    best_swap = (score, left, right)
        if best_swap is not None:
            _, left, right = best_swap
            assignment[left], assignment[right] = (
                assignment[right],
                assignment[left],
            )
            current_score = best_swap[0]
            improved = True
        if not improved:
            break

    totals = _partition_totals(assignment, group_stats)
    if any(totals[partition]["route_groups"] == 0 for partition in INNER_PARTITIONS):
        raise RuntimeError("route allocation produced an empty partition")
    return assignment, {
        "algorithm": "descending_h4_greedy_plus_deterministic_move_swap_v1",
        "seed": int(seed),
        "fractions": dict(fractions),
        "objective": current_score,
        "requested_local_search_passes": int(local_search_passes),
        "effective_local_search_passes": effective_local_search_passes,
        "partition_totals_before_window_filter": totals,
    }


def _pairwise_overlap(
    values: Mapping[str, Iterable[str]],
) -> dict[str, int]:
    sets = {key: set(map(str, value)) for key, value in values.items()}
    output: dict[str, int] = {}
    for index, left in enumerate(INNER_PARTITIONS):
        for right in INNER_PARTITIONS[index + 1 :]:
            output[f"{left}__{right}"] = len(sets[left] & sets[right])
    return output


def _validate_scand_outer_lock(
    metadata: Mapping[str, Any],
    splits: Mapping[str, Sequence[str]],
    split_lock: Mapping[str, Any],
    *,
    split_lock_sha256: str,
) -> tuple[dict[str, Mapping[str, Any]], dict[str, dict[str, str]]]:
    if metadata.get("split_lock_dataset") != "scand":
        raise RuntimeError("latent root is not bound to the SCAND split")
    if metadata.get("split_lock_sha256") != split_lock_sha256:
        raise RuntimeError("SCAND latent root and split-lock hash differ")
    if splits.get("test"):
        raise RuntimeError("train/val latent root unexpectedly exposes test data")
    policy = split_lock.get("policy", {})
    if policy.get("history_frames") != 4 or policy.get("future_frames") != 4:
        raise RuntimeError("split lock is not 4-history -> H4")
    locked = split_lock["scand"]["splits"]
    for split in ("train", "val"):
        if set(map(str, splits.get(split, ()))) != set(map(str, locked[split])):
            raise RuntimeError(f"SCAND latent {split} membership differs from lock")
    outer_sets = {
        split: set(map(str, locked[split])) for split in OUTER_SPLITS
    }
    if (
        outer_sets["train"] & outer_sets["val"]
        or outer_sets["train"] & outer_sets["test"]
        or outer_sets["val"] & outer_sets["test"]
    ):
        raise RuntimeError("SCAND exact outer-split overlap detected")

    records = {
        str(row["trajectory_id"]): row for row in metadata["trajectories"]
    }
    if set(records) != outer_sets["train"] | outer_sets["val"]:
        raise RuntimeError("SCAND train/val metadata identity differs from splits")
    parsed = {
        trajectory: parse_scand_trajectory_id(trajectory)
        for trajectory in outer_sets["train"] | outer_sets["val"] | outer_sets["test"]
    }
    route_split = {
        str(row["route_group"]): str(row["split"])
        for row in split_lock["scand"]["route_assignment"]
    }
    for split, trajectories in outer_sets.items():
        for trajectory in trajectories:
            route_group = parsed[trajectory]["route_group"]
            if route_split.get(route_group) != split:
                raise RuntimeError(
                    f"route group {route_group} is not locked to {split}"
                )
    for bag_id in {value["bag_id"] for value in parsed.values()}:
        bag_routes = {
            value["route_group"]
            for value in parsed.values()
            if value["bag_id"] == bag_id
        }
        if len(bag_routes) != 1:
            raise RuntimeError(f"physical bag spans route groups: {bag_id}")
    return records, parsed


def build_scand_inner_manifests(
    *,
    latent_root: Path,
    split_lock_path: Path,
    out_root: Path,
    split_seed: int = 20260719,
    fractions: Mapping[str, float] = DEFAULT_INNER_FRACTIONS,
    history: int = 4,
    horizon: int = 4,
    max_windows_per_bag: int = 256,
    min_goal_displacement_m: float = 0.5,
) -> dict[str, Any]:
    """Build strict four-way SCAND outer-train manifests."""

    latent_root = Path(latent_root)
    split_lock_path = Path(split_lock_path)
    out_root = Path(out_root)
    if history != 4 or horizon != 4:
        raise ValueError("this builder is frozen to 4-history -> H4")
    if max_windows_per_bag <= 0:
        raise ValueError("max_windows_per_bag must be positive")
    if min_goal_displacement_m < 0.0:
        raise ValueError("min_goal_displacement_m must be non-negative")
    if out_root.exists():
        raise FileExistsError(out_root)
    fractions = validate_inner_fractions(fractions)

    metadata_path = latent_root / "metadata.json"
    splits_path = latent_root / "splits.json"
    metadata = read_json(metadata_path)
    splits = read_json(splits_path)
    split_lock = read_json(split_lock_path)
    lock_sha = file_sha256(split_lock_path)
    records, parsed = _validate_scand_outer_lock(
        metadata,
        splits,
        split_lock,
        split_lock_sha256=lock_sha,
    )
    train_ids = set(map(str, splits["train"]))

    route_records: dict[str, list[str]] = defaultdict(list)
    for trajectory in sorted(train_ids):
        route_records[parsed[trajectory]["route_group"]].append(trajectory)
    group_stats = {}
    for route_group, trajectories in sorted(route_records.items()):
        group_stats[route_group] = {
            "h4_windows": sum(
                max(0, int(records[trajectory]["num_frames"]) - history - horizon + 1)
                for trajectory in trajectories
            ),
            "trajectories": len(trajectories),
            "bags": len({parsed[trajectory]["bag_id"] for trajectory in trajectories}),
            "route_groups": 1,
        }
    route_assignment, allocation = allocate_route_groups(
        group_stats,
        fractions=fractions,
        seed=split_seed,
    )
    trajectory_assignment = {
        trajectory: route_assignment[parsed[trajectory]["route_group"]]
        for trajectory in train_ids
    }
    bag_assignment: dict[str, str] = {}
    for trajectory in train_ids:
        bag_id = parsed[trajectory]["bag_id"]
        partition = trajectory_assignment[trajectory]
        previous = bag_assignment.setdefault(bag_id, partition)
        if previous != partition:
            raise RuntimeError(f"physical bag crosses inner partitions: {bag_id}")

    eligible_by_bag: dict[str, list[dict[str, Any]]] = defaultdict(list)
    eligible_before_cap: dict[str, int] = {}
    for trajectory in sorted(train_ids):
        record = records[trajectory]
        shard_path = latent_root / "shards" / str(record["shard_file"])
        with np.load(shard_path, allow_pickle=False) as shard:
            image_paths = shard["image_paths"].astype(str)
            positions = shard["positions"].astype(np.float64)
        if len(image_paths) != int(record["num_frames"]):
            raise RuntimeError(f"frame count differs from metadata: {trajectory}")
        if len(positions) != len(image_paths):
            raise RuntimeError(f"position/image length mismatch: {trajectory}")
        parsed_row = parsed[trajectory]
        partition = trajectory_assignment[trajectory]
        for current_t in range(history - 1, len(image_paths) - horizon):
            goal_t = current_t + horizon
            displacement = float(
                np.linalg.norm(
                    positions[goal_t, :2] - positions[current_t, :2]
                )
            )
            if displacement < min_goal_displacement_m:
                continue
            raw_uid = (
                f"strict-h4-inner-v3|{split_seed}|{partition}|{trajectory}|"
                f"{current_t}|{goal_t}"
            )
            eligible_by_bag[parsed_row["bag_id"]].append(
                {
                    "window_uid": sha256_text(raw_uid)[:24],
                    "dataset": "scand",
                    "partition": partition,
                    "trajectory_id": trajectory,
                    "route_id": parsed_row["route_group"],
                    "route_group": parsed_row["route_group"],
                    "bag_id": parsed_row["bag_id"],
                    "shard_file": str(record["shard_file"]),
                    "current_t": current_t,
                    "goal_t": goal_t,
                    "history_timesteps": list(
                        range(current_t - history + 1, current_t + 1)
                    ),
                    "context_image_paths": [
                        str(image_paths[index])
                        for index in range(current_t - history + 1, current_t + 1)
                    ],
                    "current_image_path": str(image_paths[current_t]),
                    "goal_image_path": str(image_paths[goal_t]),
                    "start_goal_displacement_m": displacement,
                }
            )

    selected_by_partition: dict[str, list[dict[str, Any]]] = {
        partition: [] for partition in INNER_PARTITIONS
    }
    for bag_id, rows in sorted(eligible_by_bag.items()):
        eligible_before_cap[bag_id] = len(rows)
        selected = sorted(
            rows,
            key=lambda row: (
                sha256_text(
                    f"bag-window-cap|{split_seed}|{bag_id}|{row['window_uid']}"
                ),
                row["window_uid"],
            ),
        )[:max_windows_per_bag]
        if selected:
            partition = bag_assignment[bag_id]
            if any(row["partition"] != partition for row in selected):
                raise RuntimeError(f"bag selection crosses partitions: {bag_id}")
            selected_by_partition[partition].extend(selected)

    for partition in INNER_PARTITIONS:
        selected_by_partition[partition].sort(
            key=lambda row: (
                row["route_group"],
                row["bag_id"],
                row["trajectory_id"],
                row["current_t"],
                row["window_uid"],
            )
        )
        if not selected_by_partition[partition]:
            raise RuntimeError(f"no eligible windows in {partition}")

    route_sets = {
        partition: {row["route_group"] for row in rows}
        for partition, rows in selected_by_partition.items()
    }
    bag_sets = {
        partition: {row["bag_id"] for row in rows}
        for partition, rows in selected_by_partition.items()
    }
    trajectory_sets = {
        partition: {row["trajectory_id"] for row in rows}
        for partition, rows in selected_by_partition.items()
    }
    route_overlap = _pairwise_overlap(route_sets)
    bag_overlap = _pairwise_overlap(bag_sets)
    trajectory_overlap = _pairwise_overlap(trajectory_sets)
    if any(route_overlap.values()) or any(bag_overlap.values()) or any(
        trajectory_overlap.values()
    ):
        raise RuntimeError("inner-partition identity overlap detected")

    out_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{out_root.name}.tmp-", dir=out_root.parent)
    )
    try:
        manifests: dict[str, Any] = {}
        for partition in INNER_PARTITIONS:
            filename = f"{partition}.windows.jsonl"
            temporary_path = temporary / filename
            _write_jsonl(temporary_path, selected_by_partition[partition])
            manifests[partition] = {
                "path": filename,
                "sha256": file_sha256(temporary_path),
                "windows": len(selected_by_partition[partition]),
                "trajectories": len(trajectory_sets[partition]),
                "bags": len(bag_sets[partition]),
                "route_groups": len(route_sets[partition]),
            }

        assignment_payload = {
            "schema": "strict_scand_h4_inner_assignment_v3",
            "seed": int(split_seed),
            "fractions": dict(fractions),
            "allocation": allocation,
            "route_assignment": dict(sorted(route_assignment.items())),
            "trajectory_assignment": dict(sorted(trajectory_assignment.items())),
            "bag_assignment": dict(sorted(bag_assignment.items())),
            "group_stats_before_window_filter": group_stats,
            "eligible_windows_before_bag_cap": dict(
                sorted(eligible_before_cap.items())
            ),
            "partition_before_window_filter": allocation[
                "partition_totals_before_window_filter"
            ],
            "label_or_model_output_used_for_assignment": False,
        }
        _write_json(temporary / "partition_assignment.json", assignment_payload)
        summary = {
            "decision": "STRICT_SCAND_H4_FOUR_WAY_MANIFESTS_READY",
            "schema": "strict_scand_h4_inner_manifests_v3",
            "dataset": "scand",
            "outer_scope": "train-only",
            "protocol": {
                "history": history,
                "horizon": horizon,
                "goal_offset": horizon,
                "split_seed": int(split_seed),
                "fractions": dict(fractions),
                "max_windows_per_physical_bag": int(max_windows_per_bag),
                "min_goal_displacement_m": float(min_goal_displacement_m),
                "partition_before_window_generation": True,
                "partition_unit": "undirected_route_group",
            },
            "manifests": manifests,
            "overlap_audit": {
                "route_group": route_overlap,
                "physical_bag": bag_overlap,
                "trajectory": trajectory_overlap,
            },
            "leakage_guard": {
                "outer_train_only": True,
                "outer_val_metadata_verified": True,
                "outer_val_examples_materialized": False,
                "test_metadata_verified_from_split_lock": True,
                "test_examples_materialized": False,
                "test_data_accessed": False,
                "labels_used_for_partitioning": False,
                "model_outputs_used_for_partitioning": False,
            },
            "latent_root": str(latent_root),
            "latent_metadata_sha256": file_sha256(metadata_path),
            "latent_splits_sha256": file_sha256(splits_path),
            "split_lock": str(split_lock_path),
            "split_lock_sha256": lock_sha,
            "assignment_sha256": file_sha256(
                temporary / "partition_assignment.json"
            ),
        }
        _write_json(temporary / "manifest_summary.json", summary)
        temporary.replace(out_root)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return read_json(out_root / "manifest_summary.json")


def build_recon_h4_exclusion_manifest(
    *,
    latent_root: Path,
    split_lock_path: Path,
    out_file: Path,
    history: int = 4,
    horizon: int = 4,
) -> dict[str, Any]:
    """Record deterministic RECON train/val exclusions without opening test."""

    latent_root = Path(latent_root)
    split_lock_path = Path(split_lock_path)
    out_file = Path(out_file)
    if history <= 0 or horizon <= 0:
        raise ValueError("history and horizon must be positive")
    if out_file.exists():
        raise FileExistsError(out_file)
    minimum_frames = history + horizon
    metadata_path = latent_root / "metadata.json"
    splits_path = latent_root / "splits.json"
    metadata = read_json(metadata_path)
    splits = read_json(splits_path)
    split_lock = read_json(split_lock_path)
    lock_sha = file_sha256(split_lock_path)
    if metadata.get("split_lock_dataset") != "recon":
        raise RuntimeError("latent root is not bound to the RECON split")
    if metadata.get("split_lock_sha256") != lock_sha:
        raise RuntimeError("RECON latent root and split-lock hash differ")
    if splits.get("test"):
        raise RuntimeError("train/val RECON latent root unexpectedly exposes test")
    policy = split_lock.get("policy", {})
    if policy.get("history_frames") != history or policy.get(
        "future_frames"
    ) != horizon:
        raise RuntimeError("requested temporal protocol differs from split lock")

    locked = split_lock["recon"]["splits"]
    records = {
        str(row["trajectory_id"]): row for row in metadata["trajectories"]
    }
    split_reports: dict[str, Any] = {}
    exclusions: list[dict[str, Any]] = []
    for split in ("train", "val"):
        root_ids = list(map(str, splits.get(split, ())))
        locked_ids = list(map(str, locked[split]))
        if set(root_ids) != set(locked_ids):
            raise RuntimeError(f"RECON latent {split} membership differs from lock")
        missing = sorted(set(root_ids) - set(records))
        if missing:
            raise RuntimeError(
                f"RECON metadata is missing {len(missing)} {split} trajectories"
            )
        eligible_ids = []
        excluded_ids = []
        windows = 0
        for trajectory in sorted(root_ids):
            num_frames = int(records[trajectory]["num_frames"])
            if num_frames < minimum_frames:
                excluded_ids.append(trajectory)
                exclusions.append(
                    {
                        "trajectory_id": trajectory,
                        "split": split,
                        "num_frames": num_frames,
                        "minimum_frames": minimum_frames,
                        "reason": "insufficient_frames_for_history_plus_horizon",
                    }
                )
            else:
                eligible_ids.append(trajectory)
                windows += num_frames - minimum_frames + 1
        split_reports[split] = {
            "locked_trajectories": len(locked_ids),
            "eligible_trajectories": len(eligible_ids),
            "excluded_trajectories": len(excluded_ids),
            "eligible_h4_windows": windows,
            "eligible_assignment_sha256": sha256_text(
                "\n".join(sorted(eligible_ids)) + "\n"
            ),
            "excluded_assignment_sha256": sha256_text(
                "\n".join(sorted(excluded_ids)) + "\n"
            ),
        }
    split_reports["test"] = {
        "locked_trajectories": len(locked["test"]),
        "status": "SEALED_NOT_MATERIALIZED",
        "eligible_trajectories": None,
        "excluded_trajectories": None,
        "eligible_h4_windows": None,
    }
    audit = split_lock["recon"].get("audit", {})
    if any(int(value) for value in audit.get("exact_overlap", {}).values()):
        raise RuntimeError("RECON exact outer-split overlap detected")
    if any(int(value) for value in audit.get("group_overlap", {}).values()):
        raise RuntimeError("RECON group outer-split overlap detected")

    report = {
        "decision": "RECON_H4_EXCLUSION_AUDIT_PASS",
        "schema": "recon_h4_exclusion_manifest_v1",
        "dataset": "recon",
        "protocol": {
            "history": int(history),
            "horizon": int(horizon),
            "minimum_frames": int(minimum_frames),
            "eligible_windows_per_trajectory": "num_frames-history-horizon+1",
        },
        "splits": split_reports,
        "exclusions": exclusions,
        "exclusion_count": len(exclusions),
        "exclusion_reasons": {
            "insufficient_frames_for_history_plus_horizon": len(exclusions)
        },
        "outer_overlap_audit": {
            "exact_overlap": audit.get("exact_overlap"),
            "group_overlap": audit.get("group_overlap"),
        },
        "leakage_guard": {
            "test_split_ids_read_from_lock_for_count_only": True,
            "test_latents_materialized": False,
            "test_examples_opened": False,
            "test_eligibility_computed": False,
        },
        "latent_root": str(latent_root),
        "latent_metadata_sha256": file_sha256(metadata_path),
        "latent_splits_sha256": file_sha256(splits_path),
        "split_lock": str(split_lock_path),
        "split_lock_sha256": lock_sha,
    }
    _write_json(out_file, report)
    return report


def _validate_recon_outer_lock(
    metadata: Mapping[str, Any],
    splits: Mapping[str, Sequence[str]],
    split_lock: Mapping[str, Any],
    *,
    split_lock_sha256: str,
) -> dict[str, Mapping[str, Any]]:
    if metadata.get("split_lock_dataset") != "recon":
        raise RuntimeError("latent root is not bound to the RECON split")
    if metadata.get("split_lock_sha256") != split_lock_sha256:
        raise RuntimeError("RECON latent root and split-lock hash differ")
    if splits.get("test"):
        raise RuntimeError("train/val latent root unexpectedly exposes test data")
    policy = split_lock.get("policy", {})
    if policy.get("history_frames") != 4 or policy.get("future_frames") != 4:
        raise RuntimeError("split lock is not 4-history -> H4")
    locked = {
        split: set(map(str, split_lock["recon"]["splits"][split]))
        for split in OUTER_SPLITS
    }
    for split in ("train", "val"):
        if set(map(str, splits.get(split, ()))) != locked[split]:
            raise RuntimeError(f"RECON latent {split} membership differs from lock")
    route_sets = {
        split: {recon_route_id(value) for value in values}
        for split, values in locked.items()
    }
    if (
        route_sets["train"] & route_sets["val"]
        or route_sets["train"] & route_sets["test"]
        or route_sets["val"] & route_sets["test"]
    ):
        raise RuntimeError("RECON outer route-group overlap detected")
    records = {
        str(row["trajectory_id"]): row for row in metadata["trajectories"]
    }
    if set(records) != locked["train"] | locked["val"]:
        raise RuntimeError("RECON train/val metadata identity differs from splits")
    return records


def _round_robin_route_windows(
    *,
    route: str,
    trajectory_timesteps: Mapping[str, Sequence[int]],
    quota: int,
    seed: int,
    partition: str,
) -> list[tuple[str, int]]:
    trajectory_order = sorted(
        trajectory_timesteps,
        key=lambda trajectory: (
            sha256_text(
                f"recon-trajectory|{seed}|{partition}|{route}|{trajectory}"
            ),
            trajectory,
        ),
    )
    ordered_timesteps = {
        trajectory: sorted(
            map(int, timesteps),
            key=lambda timestep: (
                sha256_text(
                    f"recon-window|{seed}|{partition}|{route}|"
                    f"{trajectory}|{timestep}"
                ),
                timestep,
            ),
        )
        for trajectory, timesteps in trajectory_timesteps.items()
    }
    selected: list[tuple[str, int]] = []
    round_index = 0
    while len(selected) < quota:
        progressed = False
        for trajectory in trajectory_order:
            values = ordered_timesteps[trajectory]
            if round_index >= len(values):
                continue
            selected.append((trajectory, values[round_index]))
            progressed = True
            if len(selected) == quota:
                break
        if not progressed:
            break
        round_index += 1
    return selected


def build_recon_inner_manifests(
    *,
    latent_root: Path,
    split_lock_path: Path,
    exclusion_manifest_path: Path,
    out_root: Path,
    split_seed: int = 20260719,
    fractions: Mapping[str, float] = DEFAULT_INNER_FRACTIONS,
    quota_per_route: Mapping[str, int] | None = None,
    history: int = 4,
    horizon: int = 4,
    min_goal_displacement_m: float = 0.5,
) -> dict[str, Any]:
    """Build route-disjoint four-way RECON outer-train manifests."""

    latent_root = Path(latent_root)
    split_lock_path = Path(split_lock_path)
    exclusion_manifest_path = Path(exclusion_manifest_path)
    out_root = Path(out_root)
    if history != 4 or horizon != 4:
        raise ValueError("this builder is frozen to 4-history -> H4")
    if out_root.exists():
        raise FileExistsError(out_root)
    fractions = validate_inner_fractions(fractions)
    quota_per_route = dict(
        quota_per_route
        or {
            "train-fit": 3,
            "scorer-cal": 2,
            "cem-cal": 2,
            "internal-holdout": 2,
        }
    )
    if set(quota_per_route) != set(INNER_PARTITIONS):
        raise ValueError("quota_per_route must define all inner partitions")
    if any(int(value) <= 0 for value in quota_per_route.values()):
        raise ValueError("every route quota must be positive")

    metadata_path = latent_root / "metadata.json"
    splits_path = latent_root / "splits.json"
    metadata = read_json(metadata_path)
    splits = read_json(splits_path)
    split_lock = read_json(split_lock_path)
    lock_sha = file_sha256(split_lock_path)
    records = _validate_recon_outer_lock(
        metadata,
        splits,
        split_lock,
        split_lock_sha256=lock_sha,
    )
    exclusions = read_json(exclusion_manifest_path)
    if exclusions.get("decision") != "RECON_H4_EXCLUSION_AUDIT_PASS":
        raise RuntimeError("RECON exclusion manifest is not in a pass state")
    if exclusions.get("split_lock_sha256") != lock_sha:
        raise RuntimeError("RECON exclusion manifest/split-lock mismatch")
    excluded_short = {
        str(row["trajectory_id"])
        for row in exclusions["exclusions"]
        if row["split"] == "train"
    }

    train_ids = set(map(str, splits["train"]))
    valid_timesteps: dict[str, list[int]] = {}
    no_displacement: list[str] = []
    for trajectory in sorted(train_ids):
        record = records[trajectory]
        num_frames = int(record["num_frames"])
        if num_frames < history + horizon:
            if trajectory not in excluded_short:
                raise RuntimeError(
                    f"short trajectory missing from exclusion manifest: {trajectory}"
                )
            valid_timesteps[trajectory] = []
            continue
        if trajectory in excluded_short:
            raise RuntimeError(
                f"eligible trajectory incorrectly excluded: {trajectory}"
            )
        shard_path = latent_root / "shards" / str(record["shard_file"])
        with np.load(shard_path, allow_pickle=False) as shard:
            positions = shard["positions"].astype(np.float64)
        if len(positions) != num_frames:
            raise RuntimeError(f"frame count differs from metadata: {trajectory}")
        values = [
            current_t
            for current_t in range(history - 1, num_frames - horizon)
            if float(
                np.linalg.norm(
                    positions[current_t + horizon, :2]
                    - positions[current_t, :2]
                )
            )
            >= min_goal_displacement_m
        ]
        valid_timesteps[trajectory] = values
        if not values:
            no_displacement.append(trajectory)

    by_route: dict[str, dict[str, list[int]]] = defaultdict(dict)
    for trajectory, timesteps in valid_timesteps.items():
        if timesteps:
            by_route[recon_route_id(trajectory)][trajectory] = timesteps
    if len(by_route) < len(INNER_PARTITIONS):
        raise RuntimeError("not enough eligible RECON routes for four partitions")
    group_stats = {
        route: {
            "h4_windows": sum(len(values) for values in trajectories.values()),
            "trajectories": len(trajectories),
            "bags": len(trajectories),
            "route_groups": 1,
        }
        for route, trajectories in by_route.items()
    }
    route_assignment, allocation = allocate_route_groups(
        group_stats,
        fractions=fractions,
        seed=split_seed,
    )
    trajectory_assignment = {
        trajectory: route_assignment[route]
        for route, trajectories in by_route.items()
        for trajectory in trajectories
    }

    selected_by_partition: dict[str, list[dict[str, Any]]] = {
        partition: [] for partition in INNER_PARTITIONS
    }
    for route in sorted(by_route):
        partition = route_assignment[route]
        selected = _round_robin_route_windows(
            route=route,
            trajectory_timesteps=by_route[route],
            quota=int(quota_per_route[partition]),
            seed=split_seed,
            partition=partition,
        )
        for trajectory, current_t in selected:
            record = records[trajectory]
            shard_path = latent_root / "shards" / str(record["shard_file"])
            with np.load(shard_path, allow_pickle=False) as shard:
                image_paths = shard["image_paths"].astype(str)
            goal_t = current_t + horizon
            raw_uid = (
                f"strict-recon-h4-inner-v3|{split_seed}|{partition}|"
                f"{trajectory}|{current_t}|{goal_t}"
            )
            selected_by_partition[partition].append(
                {
                    "window_uid": sha256_text(raw_uid)[:24],
                    "dataset": "recon",
                    "partition": partition,
                    "trajectory_id": trajectory,
                    "route_id": route,
                    "shard_file": str(record["shard_file"]),
                    "current_t": int(current_t),
                    "goal_t": int(goal_t),
                    "history_timesteps": list(
                        range(current_t - history + 1, current_t + 1)
                    ),
                    "context_image_paths": [
                        str(image_paths[index])
                        for index in range(current_t - history + 1, current_t + 1)
                    ],
                    "current_image_path": str(image_paths[current_t]),
                    "goal_image_path": str(image_paths[goal_t]),
                }
            )
    for partition in INNER_PARTITIONS:
        selected_by_partition[partition].sort(
            key=lambda row: (
                row["route_id"],
                row["trajectory_id"],
                row["current_t"],
                row["window_uid"],
            )
        )
        if not selected_by_partition[partition]:
            raise RuntimeError(f"no eligible RECON windows in {partition}")

    route_sets = {
        partition: {row["route_id"] for row in rows}
        for partition, rows in selected_by_partition.items()
    }
    trajectory_sets = {
        partition: {row["trajectory_id"] for row in rows}
        for partition, rows in selected_by_partition.items()
    }
    route_overlap = _pairwise_overlap(route_sets)
    trajectory_overlap = _pairwise_overlap(trajectory_sets)
    if any(route_overlap.values()) or any(trajectory_overlap.values()):
        raise RuntimeError("RECON inner-partition overlap detected")

    out_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{out_root.name}.tmp-", dir=out_root.parent)
    )
    try:
        manifests: dict[str, Any] = {}
        for partition in INNER_PARTITIONS:
            filename = f"{partition}.windows.jsonl"
            path = temporary / filename
            _write_jsonl(path, selected_by_partition[partition])
            manifests[partition] = {
                "path": filename,
                "sha256": file_sha256(path),
                "windows": len(selected_by_partition[partition]),
                "trajectories": len(trajectory_sets[partition]),
                "route_groups": len(route_sets[partition]),
            }
        assignment_payload = {
            "schema": "strict_recon_h4_inner_assignment_v3",
            "seed": int(split_seed),
            "fractions": dict(fractions),
            "allocation": allocation,
            "route_assignment": dict(sorted(route_assignment.items())),
            "trajectory_assignment": dict(sorted(trajectory_assignment.items())),
            "group_stats_after_geometry_gate": group_stats,
            "label_or_model_output_used_for_assignment": False,
        }
        _write_json(temporary / "partition_assignment.json", assignment_payload)
        summary = {
            "decision": "STRICT_RECON_H4_FOUR_WAY_MANIFESTS_READY",
            "schema": "strict_recon_h4_inner_manifests_v3",
            "dataset": "recon",
            "outer_scope": "train-only",
            "protocol": {
                "history": history,
                "horizon": horizon,
                "goal_offset": horizon,
                "split_seed": int(split_seed),
                "fractions": dict(fractions),
                "quota_per_route": {
                    key: int(value) for key, value in quota_per_route.items()
                },
                "min_goal_displacement_m": float(min_goal_displacement_m),
                "partition_before_window_generation": True,
                "partition_unit": "repeat-stripped route_id",
            },
            "manifests": manifests,
            "overlap_audit": {
                "route_group": route_overlap,
                "trajectory": trajectory_overlap,
            },
            "exclusions": {
                "short_trajectories": len(excluded_short),
                "no_minimum_displacement_trajectories": len(no_displacement),
                "no_minimum_displacement_assignment_sha256": sha256_text(
                    "\n".join(sorted(no_displacement)) + "\n"
                ),
            },
            "leakage_guard": {
                "outer_train_only": True,
                "outer_val_examples_materialized": False,
                "test_examples_materialized": False,
                "test_data_accessed": False,
                "labels_used_for_partitioning": False,
                "model_outputs_used_for_partitioning": False,
            },
            "latent_root": str(latent_root),
            "latent_metadata_sha256": file_sha256(metadata_path),
            "latent_splits_sha256": file_sha256(splits_path),
            "split_lock": str(split_lock_path),
            "split_lock_sha256": lock_sha,
            "exclusion_manifest": str(exclusion_manifest_path),
            "exclusion_manifest_sha256": file_sha256(exclusion_manifest_path),
            "assignment_sha256": file_sha256(
                temporary / "partition_assignment.json"
            ),
        }
        _write_json(temporary / "manifest_summary.json", summary)
        temporary.replace(out_root)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return read_json(out_root / "manifest_summary.json")


def validate_recon_inner_manifests(
    *,
    asset_root: Path,
    split_lock_path: Path,
    out_file: Path | None = None,
) -> dict[str, Any]:
    """Independently validate a built RECON four-way manifest directory."""

    asset_root = Path(asset_root)
    split_lock_path = Path(split_lock_path)
    out_file = Path(out_file) if out_file is not None else None
    if out_file is not None and out_file.exists():
        raise FileExistsError(out_file)
    summary_path = asset_root / "manifest_summary.json"
    assignment_path = asset_root / "partition_assignment.json"
    summary = read_json(summary_path)
    assignment = read_json(assignment_path)
    split_lock = read_json(split_lock_path)
    lock_sha = file_sha256(split_lock_path)
    if summary.get("decision") != "STRICT_RECON_H4_FOUR_WAY_MANIFESTS_READY":
        raise RuntimeError("RECON manifest summary is not in a ready state")
    if summary.get("split_lock_sha256") != lock_sha:
        raise RuntimeError("RECON manifest/split-lock hash mismatch")
    if summary.get("assignment_sha256") != file_sha256(assignment_path):
        raise RuntimeError("RECON assignment hash mismatch")
    if assignment.get("label_or_model_output_used_for_assignment") is not False:
        raise RuntimeError("RECON assignment is not label/model-output clean")

    locked = {
        split: set(map(str, split_lock["recon"]["splits"][split]))
        for split in OUTER_SPLITS
    }
    route_assignment = {
        str(key): str(value)
        for key, value in assignment["route_assignment"].items()
    }
    trajectory_assignment = {
        str(key): str(value)
        for key, value in assignment["trajectory_assignment"].items()
    }
    route_sets: dict[str, set[str]] = {}
    trajectory_sets: dict[str, set[str]] = {}
    all_uids: set[str] = set()
    partition_counts: dict[str, Any] = {}
    for partition in INNER_PARTITIONS:
        entry = summary["manifests"][partition]
        path = asset_root / str(entry["path"])
        if file_sha256(path) != entry["sha256"]:
            raise RuntimeError(f"RECON manifest hash mismatch: {partition}")
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(rows) != int(entry["windows"]) or not rows:
            raise RuntimeError(f"RECON manifest count mismatch: {partition}")
        for row in rows:
            trajectory = str(row["trajectory_id"])
            route = str(row["route_id"])
            if row["dataset"] != "recon" or row["partition"] != partition:
                raise RuntimeError("RECON row domain/partition mismatch")
            if trajectory not in locked["train"]:
                raise RuntimeError("RECON manifest exposes a non-train trajectory")
            if trajectory in locked["val"] or trajectory in locked["test"]:
                raise RuntimeError("RECON manifest exposes outer val/test")
            if route != recon_route_id(trajectory):
                raise RuntimeError("RECON route identity differs from trajectory ID")
            if route_assignment.get(route) != partition:
                raise RuntimeError("RECON row differs from route assignment")
            if trajectory_assignment.get(trajectory) != partition:
                raise RuntimeError("RECON row differs from trajectory assignment")
            current_t = int(row["current_t"])
            if int(row["goal_t"]) - current_t != 4:
                raise RuntimeError("RECON goal offset is not H4")
            if list(map(int, row["history_timesteps"])) != list(
                range(current_t - 3, current_t + 1)
            ):
                raise RuntimeError("RECON history is not four consecutive frames")
            uid = str(row["window_uid"])
            if uid in all_uids:
                raise RuntimeError(f"duplicate RECON window UID: {uid}")
            all_uids.add(uid)
        route_sets[partition] = {str(row["route_id"]) for row in rows}
        trajectory_sets[partition] = {
            str(row["trajectory_id"]) for row in rows
        }
        partition_counts[partition] = {
            "windows": len(rows),
            "trajectories": len(trajectory_sets[partition]),
            "route_groups": len(route_sets[partition]),
        }
        for key, value in partition_counts[partition].items():
            if int(entry[key]) != value:
                raise RuntimeError(f"RECON {partition} {key} count mismatch")
    overlaps = {
        "route_group": _pairwise_overlap(route_sets),
        "trajectory": _pairwise_overlap(trajectory_sets),
    }
    if any(
        count
        for category in overlaps.values()
        for count in category.values()
    ):
        raise RuntimeError("independent RECON overlap audit failed")
    if overlaps != summary["overlap_audit"]:
        raise RuntimeError("independent RECON overlap differs from build summary")
    report = {
        "decision": "STRICT_RECON_H4_FOUR_WAY_INDEPENDENT_AUDIT_PASS",
        "schema": "strict_recon_h4_inner_independent_audit_v1",
        "asset_root": str(asset_root),
        "manifest_summary_sha256": file_sha256(summary_path),
        "partition_assignment_sha256": file_sha256(assignment_path),
        "split_lock_sha256": lock_sha,
        "partitions": partition_counts,
        "overlap_audit": overlaps,
        "window_uid_count": len(all_uids),
        "leakage_guard": {
            "outer_train_only": True,
            "outer_val_rows": 0,
            "test_rows": 0,
            "labels_read": False,
            "model_outputs_read": False,
        },
    }
    if out_file is not None:
        _write_json(out_file, report)
    return report


def validate_scand_inner_manifests(
    *,
    asset_root: Path,
    split_lock_path: Path,
    out_file: Path | None = None,
) -> dict[str, Any]:
    """Independently validate a built SCAND four-way manifest directory."""

    asset_root = Path(asset_root)
    split_lock_path = Path(split_lock_path)
    out_file = Path(out_file) if out_file is not None else None
    if out_file is not None and out_file.exists():
        raise FileExistsError(out_file)
    summary_path = asset_root / "manifest_summary.json"
    assignment_path = asset_root / "partition_assignment.json"
    summary = read_json(summary_path)
    assignment = read_json(assignment_path)
    split_lock = read_json(split_lock_path)
    lock_sha = file_sha256(split_lock_path)
    if summary.get("decision") != "STRICT_SCAND_H4_FOUR_WAY_MANIFESTS_READY":
        raise RuntimeError("SCAND manifest summary is not in a ready state")
    if summary.get("split_lock_sha256") != lock_sha:
        raise RuntimeError("SCAND manifest/split-lock hash mismatch")
    if summary.get("assignment_sha256") != file_sha256(assignment_path):
        raise RuntimeError("SCAND assignment hash mismatch")
    if assignment.get("label_or_model_output_used_for_assignment") is not False:
        raise RuntimeError("partition assignment is not label/model-output clean")

    locked = {
        split: set(map(str, split_lock["scand"]["splits"][split]))
        for split in OUTER_SPLITS
    }
    trajectory_assignment = {
        str(key): str(value)
        for key, value in assignment["trajectory_assignment"].items()
    }
    if set(trajectory_assignment) != locked["train"]:
        raise RuntimeError("inner assignment does not cover exact outer-train")
    route_assignment = {
        str(key): str(value)
        for key, value in assignment["route_assignment"].items()
    }
    bag_assignment = {
        str(key): str(value)
        for key, value in assignment["bag_assignment"].items()
    }

    rows_by_partition: dict[str, list[dict[str, Any]]] = {}
    all_window_uids: set[str] = set()
    route_sets: dict[str, set[str]] = {}
    bag_sets: dict[str, set[str]] = {}
    trajectory_sets: dict[str, set[str]] = {}
    required = {
        "window_uid",
        "dataset",
        "partition",
        "trajectory_id",
        "route_id",
        "route_group",
        "bag_id",
        "shard_file",
        "current_t",
        "goal_t",
        "history_timesteps",
    }
    for partition in INNER_PARTITIONS:
        manifest_entry = summary["manifests"][partition]
        path = asset_root / str(manifest_entry["path"])
        if file_sha256(path) != manifest_entry["sha256"]:
            raise RuntimeError(f"manifest hash mismatch: {partition}")
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(rows) != int(manifest_entry["windows"]):
            raise RuntimeError(f"manifest window count mismatch: {partition}")
        if not rows:
            raise RuntimeError(f"manifest partition is empty: {partition}")
        for row in rows:
            missing = required.difference(row)
            if missing:
                raise RuntimeError(
                    f"{partition} row is missing fields: {sorted(missing)}"
                )
            if row["dataset"] != "scand" or row["partition"] != partition:
                raise RuntimeError(f"row domain/partition mismatch: {partition}")
            trajectory = str(row["trajectory_id"])
            if trajectory not in locked["train"]:
                raise RuntimeError("manifest exposes a non-train trajectory")
            if trajectory in locked["val"] or trajectory in locked["test"]:
                raise RuntimeError("manifest exposes outer val/test")
            parsed = parse_scand_trajectory_id(trajectory)
            route_group = str(row["route_group"])
            bag_id = str(row["bag_id"])
            if route_group != parsed["route_group"] or row["route_id"] != route_group:
                raise RuntimeError("row route identity differs from trajectory ID")
            if bag_id != parsed["bag_id"]:
                raise RuntimeError("row physical-bag identity differs from trajectory ID")
            if trajectory_assignment.get(trajectory) != partition:
                raise RuntimeError("row differs from trajectory assignment")
            if route_assignment.get(route_group) != partition:
                raise RuntimeError("row differs from route assignment")
            if bag_assignment.get(bag_id) != partition:
                raise RuntimeError("row differs from physical-bag assignment")
            current_t = int(row["current_t"])
            goal_t = int(row["goal_t"])
            history_timesteps = list(map(int, row["history_timesteps"]))
            if goal_t - current_t != 4:
                raise RuntimeError("manifest goal offset is not H4")
            if history_timesteps != list(range(current_t - 3, current_t + 1)):
                raise RuntimeError("manifest history is not four consecutive frames")
            window_uid = str(row["window_uid"])
            if window_uid in all_window_uids:
                raise RuntimeError(f"duplicate window UID: {window_uid}")
            all_window_uids.add(window_uid)
        rows_by_partition[partition] = rows
        route_sets[partition] = {str(row["route_group"]) for row in rows}
        bag_sets[partition] = {str(row["bag_id"]) for row in rows}
        trajectory_sets[partition] = {
            str(row["trajectory_id"]) for row in rows
        }
        expected = {
            "route_groups": len(route_sets[partition]),
            "bags": len(bag_sets[partition]),
            "trajectories": len(trajectory_sets[partition]),
        }
        for key, value in expected.items():
            if int(manifest_entry[key]) != value:
                raise RuntimeError(f"{partition} {key} count mismatch")

    overlaps = {
        "route_group": _pairwise_overlap(route_sets),
        "physical_bag": _pairwise_overlap(bag_sets),
        "trajectory": _pairwise_overlap(trajectory_sets),
    }
    if any(
        count
        for category in overlaps.values()
        for count in category.values()
    ):
        raise RuntimeError("independent overlap audit failed")
    if overlaps != summary["overlap_audit"]:
        raise RuntimeError("independent overlap audit differs from build summary")

    report = {
        "decision": "STRICT_SCAND_H4_FOUR_WAY_INDEPENDENT_AUDIT_PASS",
        "schema": "strict_scand_h4_inner_independent_audit_v1",
        "asset_root": str(asset_root),
        "manifest_summary_sha256": file_sha256(summary_path),
        "partition_assignment_sha256": file_sha256(assignment_path),
        "split_lock_sha256": lock_sha,
        "partitions": {
            partition: {
                "windows": len(rows_by_partition[partition]),
                "trajectories": len(trajectory_sets[partition]),
                "physical_bags": len(bag_sets[partition]),
                "route_groups": len(route_sets[partition]),
            }
            for partition in INNER_PARTITIONS
        },
        "overlap_audit": overlaps,
        "window_uid_count": len(all_window_uids),
        "leakage_guard": {
            "outer_train_only": True,
            "outer_val_rows": 0,
            "test_rows": 0,
            "labels_read": False,
            "model_outputs_read": False,
        },
    }
    if out_file is not None:
        _write_json(out_file, report)
    return report


def assert_manifest_is_reproducible(
    first_root: Path,
    second_root: Path,
) -> None:
    """Assert byte-identical deterministic outputs from two build roots."""

    first_files = {
        path.relative_to(first_root): file_sha256(path)
        for path in first_root.rglob("*")
        if path.is_file()
    }
    second_files = {
        path.relative_to(second_root): file_sha256(path)
        for path in second_root.rglob("*")
        if path.is_file()
    }
    if first_files != second_files:
        raise AssertionError(
            f"manifest builds differ: first={first_files}, second={second_files}"
        )


__all__ = [
    "DEFAULT_INNER_FRACTIONS",
    "INNER_PARTITIONS",
    "allocate_route_groups",
    "assert_manifest_is_reproducible",
    "build_recon_h4_exclusion_manifest",
    "build_recon_inner_manifests",
    "build_scand_inner_manifests",
    "normalize_scand_route",
    "parse_scand_trajectory_id",
    "recon_route_id",
    "validate_recon_inner_manifests",
    "validate_scand_inner_manifests",
    "validate_inner_fractions",
]
