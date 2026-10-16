#!/usr/bin/env python3
"""Build a deduplicated, route-isolated SCAND expansion split and source view."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from recon_latents.split_lock import (  # noqa: E402
    SPLITS,
    assert_disjoint_splits,
    file_sha256,
    load_split_lock,
    normalize_scand_route,
    validate_parent_split_reuse,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-split-lock", type=Path, required=True)
    parser.add_argument("--old-scand-root", type=Path, required=True)
    parser.add_argument("--new-scand-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--canonical-root", type=Path, required=True)
    parser.add_argument("--allocation-seed", type=int, default=20260718)
    parser.add_argument("--minimum-frames", type=int, default=8)
    parser.add_argument("--novel-train-fraction", type=float, default=0.70)
    parser.add_argument("--novel-val-fraction", type=float, default=0.15)
    parser.add_argument("--novel-test-fraction", type=float, default=0.15)
    parser.add_argument("--new-archive-sha256", default=None)
    parser.add_argument("--new-tree-sha256", default=None)
    parser.add_argument("--parent-freeze-id", default=None)
    return parser.parse_args()


def parse_scand_id(trajectory_id: str) -> dict[str, str]:
    parts = trajectory_id.split("_")
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


def stable_tiebreak(seed: int, route: str, split: str = "") -> str:
    return hashlib.sha256(f"{seed}|{route}|{split}".encode()).hexdigest()


def allocate_novel_routes(
    route_stats: dict[str, dict[str, int]],
    *,
    seed: int,
    fractions: dict[str, float],
) -> tuple[dict[str, str], dict[str, Any]]:
    metrics = ("h4_windows", "trajectories")
    totals_all = {
        metric: sum(route_stats[route][metric] for route in route_stats)
        for metric in metrics
    }
    fraction_sum = sum(fractions.values())
    if any(value <= 0 for value in totals_all.values()) or fraction_sum <= 0:
        raise ValueError("novel route allocation requires positive data/fractions")
    fractions = {key: value / fraction_sum for key, value in fractions.items()}
    targets = {
        metric: {
            split: totals_all[metric] * fractions[split] for split in SPLITS
        }
        for metric in metrics
    }
    totals = {
        metric: {split: 0 for split in SPLITS}
        for metric in metrics
    }
    counts = {split: 0 for split in SPLITS}
    assignment: dict[str, str] = {}

    def objective(candidate: dict[str, dict[str, int]]) -> float:
        return sum(
            (
                (
                    candidate[metric][split] - targets[metric][split]
                )
                / max(targets[metric][split], 1.0)
            )
            ** 2
            for metric in metrics
            for split in SPLITS
        ) / len(metrics)

    ordered = sorted(
        route_stats,
        key=lambda route: (
            -route_stats[route]["h4_windows"],
            -route_stats[route]["trajectories"],
            stable_tiebreak(seed, route),
        ),
    )
    for route in ordered:
        choices = []
        for split in SPLITS:
            projected = {
                metric: dict(totals[metric]) for metric in metrics
            }
            for metric in metrics:
                projected[metric][split] += route_stats[route][metric]
            choices.append(
                (
                    objective(projected),
                    stable_tiebreak(seed, route, split),
                    split,
                )
            )
        split = min(choices)[2]
        assignment[route] = split
        for metric in metrics:
            totals[metric][split] += route_stats[route][metric]
        counts[split] += 1

    # A validation/test result based on only one or two route groups is fragile.
    # Move the smallest train groups if the deterministic allocation undershoots.
    for required_split in ("val", "test"):
        while counts[required_split] < 4:
            candidates = [
                route for route, split in assignment.items() if split == "train"
            ]
            if not candidates:
                raise RuntimeError(f"cannot supply four novel {required_split} routes")
            choices = []
            for route in candidates:
                projected = {
                    metric: dict(totals[metric]) for metric in metrics
                }
                for metric in metrics:
                    projected[metric]["train"] -= route_stats[route][metric]
                    projected[metric][required_split] += route_stats[route][metric]
                choices.append(
                    (
                        objective(projected),
                        route_stats[route]["h4_windows"],
                        route_stats[route]["trajectories"],
                        stable_tiebreak(seed, route, required_split),
                        route,
                    )
                )
            route = min(choices)[4]
            assignment[route] = required_split
            for metric in metrics:
                totals[metric]["train"] -= route_stats[route][metric]
                totals[metric][required_split] += route_stats[route][metric]
            counts["train"] -= 1
            counts[required_split] += 1

    # Greedy placement is deterministic but can strand one highly fragmented
    # route in validation. Improve the joint metric objective using only source
    # metadata (never model outcomes): single-route moves and pairwise swaps.
    local_search_steps = 0
    while local_search_steps < 1000:
        current_objective = objective(totals)
        candidates: list[
            tuple[float, str, str, str, str | None]
        ] = []
        for route, source_split in assignment.items():
            for destination_split in SPLITS:
                if destination_split == source_split:
                    continue
                if source_split in {"val", "test"} and counts[source_split] <= 4:
                    continue
                projected = {
                    metric: dict(totals[metric]) for metric in metrics
                }
                for metric in metrics:
                    value = route_stats[route][metric]
                    projected[metric][source_split] -= value
                    projected[metric][destination_split] += value
                candidates.append(
                    (
                        objective(projected),
                        stable_tiebreak(
                            seed,
                            route,
                            f"move:{source_split}:{destination_split}",
                        ),
                        "move",
                        route,
                        destination_split,
                    )
                )
        routes = sorted(assignment)
        for index, left in enumerate(routes):
            left_split = assignment[left]
            for right in routes[index + 1 :]:
                right_split = assignment[right]
                if left_split == right_split:
                    continue
                projected = {
                    metric: dict(totals[metric]) for metric in metrics
                }
                for metric in metrics:
                    left_value = route_stats[left][metric]
                    right_value = route_stats[right][metric]
                    projected[metric][left_split] += right_value - left_value
                    projected[metric][right_split] += left_value - right_value
                candidates.append(
                    (
                        objective(projected),
                        stable_tiebreak(seed, left, f"swap:{right}"),
                        "swap",
                        left,
                        right,
                    )
                )
        if not candidates:
            break
        best = min(candidates)
        if best[0] >= current_objective - 1e-12:
            break
        _, _, operation, left, right = best
        left_split = assignment[left]
        if operation == "move":
            assert right is not None
            destination_split = right
            for metric in metrics:
                value = route_stats[left][metric]
                totals[metric][left_split] -= value
                totals[metric][destination_split] += value
            assignment[left] = destination_split
            counts[left_split] -= 1
            counts[destination_split] += 1
        else:
            assert right is not None
            right_split = assignment[right]
            for metric in metrics:
                left_value = route_stats[left][metric]
                right_value = route_stats[right][metric]
                totals[metric][left_split] += right_value - left_value
                totals[metric][right_split] += left_value - right_value
            assignment[left] = right_split
            assignment[right] = left_split
        local_search_steps += 1

    return assignment, {
        "algorithm": (
            "descending_h4_greedy_plus_deterministic_move_swap_local_search_v3"
        ),
        "allocation_seed": seed,
        "fractions": fractions,
        "target_metrics": targets,
        "actual_metrics": totals,
        "route_counts": counts,
        "local_search_steps": local_search_steps,
        "objective": objective(totals),
    }


def scan_root(root: Path, source: str) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for path in sorted(item for item in root.iterdir() if item.is_dir()):
        images = sorted(path.glob("*.jpg"), key=lambda item: int(item.stem))
        pkl_path = path / "traj_data.pkl"
        if not pkl_path.is_file():
            raise FileNotFoundError(f"missing traj_data.pkl: {path}")
        parsed = parse_scand_id(path.name)
        records[path.name] = {
            "trajectory_id": path.name,
            "source": source,
            "source_dir": str(path),
            "traj_data_sha256": file_sha256(pkl_path),
            "num_frames": len(images),
            "h4_windows": max(len(images) - 7, 0),
            **parsed,
        }
    return records


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    if args.minimum_frames != 8:
        raise ValueError("this H4 protocol requires exactly 4 history + 4 future frames")
    if args.output_dir.exists() or args.canonical_root.exists():
        raise FileExistsError("output paths must not already exist")
    parent, parent_sha = load_split_lock(args.parent_split_lock)
    old_records = scan_root(args.old_scand_root, "frozen_v1")
    new_records = scan_root(args.new_scand_root, "scand_320")

    duplicate_ids = sorted(set(old_records) & set(new_records))
    duplicate_audit = []
    for trajectory_id in duplicate_ids:
        old = old_records[trajectory_id]
        new = new_records[trajectory_id]
        duplicate_audit.append(
            {
                "trajectory_id": trajectory_id,
                "decision": "retain_frozen_v1_exclude_new_alternate_conversion",
                "old_num_frames": old["num_frames"],
                "new_num_frames": new["num_frames"],
                "old_traj_data_sha256": old["traj_data_sha256"],
                "new_traj_data_sha256": new["traj_data_sha256"],
                "traj_data_bytes_equal": (
                    old["traj_data_sha256"] == new["traj_data_sha256"]
                ),
            }
        )

    canonical = dict(old_records)
    canonical.update(
        {
            trajectory_id: record
            for trajectory_id, record in new_records.items()
            if trajectory_id not in old_records
        }
    )
    insufficient = sorted(
        (
            {
                "trajectory_id": trajectory_id,
                "source": record["source"],
                "num_frames": record["num_frames"],
                "reason": "fewer_than_8_frames_no_h4_window",
            }
            for trajectory_id, record in canonical.items()
            if record["num_frames"] < args.minimum_frames
        ),
        key=lambda item: item["trajectory_id"],
    )
    eligible = {
        trajectory_id: record
        for trajectory_id, record in canonical.items()
        if record["num_frames"] >= args.minimum_frames
    }

    parent_assignment = {
        str(trajectory_id): split
        for split in SPLITS
        for trajectory_id in parent["scand"]["splits"][split]
    }
    missing_parent = sorted(set(parent_assignment) - set(eligible))
    if missing_parent:
        raise ValueError(f"parent trajectories missing from canonical set: {missing_parent}")
    inherited_route_assignment: dict[str, str] = {}
    for trajectory_id, split in parent_assignment.items():
        route = eligible[trajectory_id]["route_group"]
        previous = inherited_route_assignment.setdefault(route, split)
        if previous != split:
            raise ValueError(f"parent route has conflicting splits: {route}")

    route_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in eligible.values():
        route_records[record["route_group"]].append(record)
    novel_route_stats = {
        route: {
            "h4_windows": sum(item["h4_windows"] for item in records),
            "trajectories": len(records),
        }
        for route, records in route_records.items()
        if route not in inherited_route_assignment
    }
    novel_assignment, allocation_report = allocate_novel_routes(
        novel_route_stats,
        seed=args.allocation_seed,
        fractions={
            "train": args.novel_train_fraction,
            "val": args.novel_val_fraction,
            "test": args.novel_test_fraction,
        },
    )
    route_assignment = {**inherited_route_assignment, **novel_assignment}
    splits = {
        split: sorted(
            trajectory_id
            for trajectory_id, record in eligible.items()
            if route_assignment[record["route_group"]] == split
        )
        for split in SPLITS
    }
    groups = {
        trajectory_id: record["route_group"]
        for trajectory_id, record in eligible.items()
    }
    isolation_audit = assert_disjoint_splits(splits, group_for_id=groups)

    args.output_dir.mkdir(parents=True)
    args.canonical_root.mkdir(parents=True)
    for trajectory_id, record in sorted(eligible.items()):
        os.symlink(record["source_dir"], args.canonical_root / trajectory_id)

    with Image.open(
        next(
            iter(
                sorted(
                    (Path(record["source_dir"]) for record in eligible.values()),
                    key=str,
                )
            )
        )
        / "0.jpg"
    ) as sample:
        native_image_size = list(sample.size)

    route_rows = []
    for route, records in sorted(route_records.items()):
        split = route_assignment[route]
        route_rows.append(
            {
                "route_group": route,
                "split": split,
                "assignment_origin": (
                    "inherited_parent_lock"
                    if route in inherited_route_assignment
                    else "deterministic_novel_allocation"
                ),
                "trajectory_count": len(records),
                "bag_count": len({item["bag_id"] for item in records}),
                "frame_count": sum(item["num_frames"] for item in records),
                "h4_windows": sum(item["h4_windows"] for item in records),
                "robots": dict(Counter(item["robot"] for item in records)),
            }
        )
    split_summary = {
        split: {
            "trajectories": len(splits[split]),
            "bags": len({eligible[item]["bag_id"] for item in splits[split]}),
            "routes": len({eligible[item]["route_group"] for item in splits[split]}),
            "frames": sum(eligible[item]["num_frames"] for item in splits[split]),
            "h4_windows": sum(eligible[item]["h4_windows"] for item in splits[split]),
            "robots": dict(Counter(eligible[item]["robot"] for item in splits[split])),
        }
        for split in SPLITS
    }
    cohorts = {
        f"{origin}_{split}": sorted(
            trajectory_id
            for trajectory_id in splits[split]
            if (
                (eligible[trajectory_id]["route_group"] in inherited_route_assignment)
                == (origin == "inherited_route")
            )
        )
        for split in SPLITS
        for origin in ("inherited_route", "novel_route")
    }
    lock = {
        "schema_version": 2,
        "experiment": "ltwm_dinov3_recon_scand_expansion_v2",
        "lineage": {
            "parent_split_lock": str(args.parent_split_lock),
            "parent_split_lock_sha256": parent_sha,
            "parent_freeze_id": args.parent_freeze_id,
            "duplicate_policy": (
                "frozen_v1_wins_on_same_trajectory_id; alternate conversion excluded"
            ),
        },
        "policy": {
            "history_frames": 4,
            "future_frames": 4,
            "minimum_window_frames": 8,
            "split_before_window_generation": True,
            "same_bag_single_split": True,
            "reverse_route_group_single_split": True,
            "parent_assignments_preserved": True,
            "train_only_statistics": True,
            "test_locked_until_three_seed_checkpoint_freeze": True,
            "test_latents_forbidden_during_training": True,
            "unverified_warm_start_forbidden": True,
        },
        "recon": {
            **parent["recon"],
            "splits": {
                split: sorted(map(str, parent["recon"]["splits"][split]))
                for split in SPLITS
            },
            "lineage": "unchanged_from_parent",
        },
        "scand": {
            "canonical_root": str(args.canonical_root),
            "old_scand_root": str(args.old_scand_root),
            "new_scand_root": str(args.new_scand_root),
            "new_archive_sha256": args.new_archive_sha256,
            "new_tree_sha256": args.new_tree_sha256,
            "native_image_size": native_image_size,
            "splits_file": "scand_strict_splits.json",
            "splits": splits,
            "evaluation_cohorts": cohorts,
            "split_summary": split_summary,
            "route_assignment": route_rows,
            "deduplication": {
                "same_id_overlap_count": len(duplicate_ids),
                "same_id_overlap": duplicate_audit,
                "insufficient_h4_count": len(insufficient),
                "eligible_trajectory_count": len(eligible),
            },
            "allocation": allocation_report,
            "isolation_audit": isolation_audit,
        },
        "training_preregistration": {
            "backbone": "DINOv3-L/16 frozen pretrained checkpoint",
            "adapter": "frozen parent adapter; verified parent-lock reuse",
            "world_model": "unchanged 8,001,960-parameter LTWM H4 architecture",
            "initialization": "matching parent seed 1701/1702/1703 checkpoint",
            "optimizer_state": "reset",
            "seeds": [1701, 1702, 1703],
            "epochs": 1,
            "coverage": "at least one complete SCAND-train H4 pass per seed",
            "recon_batches_per_scand_batch": 3,
            "learning_rate": 0.00005,
            "model_selection": "SCAND validation only with RECON validation guard",
            "test_unlock": (
                "only after all three checkpoints, commands, hashes, and validation "
                "decisions are frozen"
            ),
        },
    }
    lock_path = args.output_dir / "split_lock.json"
    write_json(lock_path, lock)
    lock_sha = file_sha256(lock_path)
    (lock_path.with_suffix(".json.sha256")).write_text(
        f"{lock_sha}  split_lock.json\n", encoding="utf-8"
    )
    write_json(args.output_dir / "scand_strict_splits.json", splits)
    write_json(
        args.output_dir / "recon_strict_splits.json",
        lock["recon"]["splits"],
    )
    write_json(args.output_dir / "dedup_audit.json", {
        "same_id_overlap": duplicate_audit,
        "insufficient_h4": insufficient,
    })
    write_json(args.output_dir / "route_assignment.json", {
        "allocation": allocation_report,
        "routes": route_rows,
    })
    with (args.output_dir / "canonical_source_manifest.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for trajectory_id, record in sorted(eligible.items()):
            handle.write(
                json.dumps(
                    {
                        **record,
                        "split": route_assignment[record["route_group"]],
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    write_json(
        args.output_dir / "build_report.json",
        {
            "status": "PASS",
            "split_lock_sha256": lock_sha,
            "parent_split_lock_sha256": parent_sha,
            "raw_trajectory_counts": {
                "old": len(old_records),
                "new": len(new_records),
            },
            "same_id_overlap_count": len(duplicate_ids),
            "insufficient_h4_count": len(insufficient),
            "eligible_trajectory_count": len(eligible),
            "split_summary": split_summary,
            "isolation_audit": isolation_audit,
            "test_accessed": False,
        },
    )
    validate_parent_split_reuse(
        lock,
        parent_split_lock=args.parent_split_lock,
        artifact_split_lock_sha256=parent_sha,
        unchanged_datasets=("recon",),
        preserved_datasets=("scand",),
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "split_lock": str(lock_path),
                "split_lock_sha256": lock_sha,
                "canonical_root": str(args.canonical_root),
                "eligible_trajectories": len(eligible),
                "same_id_duplicates_excluded": len(duplicate_ids),
                "insufficient_h4_excluded": len(insufficient),
                "split_summary": split_summary,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
