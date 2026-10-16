#!/usr/bin/env python3
"""Build a model-independent SCAND strict-test H4-100 benchmark.

The benchmark covers every trajectory in the frozen SCAND outer-test split,
balances trajectory-length and temporal-position quartiles, and never reads
model predictions or scores.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import pickle
import shutil
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ASSETS = ROOT / "configs/local_assets.json"
DEFAULT_OUTPUT = ROOT / "data/evaluation/scand/strict_test_h4_100_seed17"
CANONICAL_MANIFEST = ROOT / "data/split_lock/canonical_source_manifest.jsonl"
ROUTE_ASSIGNMENT = ROOT / "data/split_lock/route_assignment.json"
SCAND_SPLITS = ROOT / "data/split_lock/scand_strict_splits.json"
SPLIT_LOCK = ROOT / "data/split_lock/split_lock.json"

CONTEXT_FRAMES = 4
HORIZON = 4
TARGET_WINDOWS = 100
TEMPORAL_FRACTIONS = (0.125, 0.375, 0.625, 0.875)
TEMPORAL_NAMES = ("early", "mid_early", "mid_late", "late")
LENGTH_NAMES = ("Q1_shortest", "Q2", "Q3", "Q4_longest")


@dataclass
class Trajectory:
    trajectory_id: str
    canonical_num_frames: int
    num_frames: int
    metadata: dict[str, Any]
    trajectory_dir: Path
    length_bin: int = -1
    windows: int = 0
    temporal_bins: tuple[int, ...] = ()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_key(seed: int, namespace: str, *parts: object) -> str:
    payload = "|".join((str(seed), namespace, *(str(part) for part in parts)))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def scand_root_from_assets(path: Path) -> Path:
    payload = load_json(path)
    return Path(payload["datasets"]["scand"]["trajectory_root"]).expanduser()


def contiguous_image_count(trajectory_dir: Path) -> int:
    indices = {
        int(path.stem)
        for path in trajectory_dir.glob("*.jpg")
        if path.stem.isdigit()
    }
    count = 0
    while count in indices:
        count += 1
    if len(indices) != count:
        unexpected = sorted(index for index in indices if index >= count)[:5]
        raise RuntimeError(
            f"non-contiguous JPEG sequence in {trajectory_dir}: "
            f"first missing={count}, later indices={unexpected}"
        )
    return count


def local_num_frames(trajectory_dir: Path) -> int:
    pose_path = trajectory_dir / "traj_data.pkl"
    if not pose_path.is_file():
        raise FileNotFoundError(pose_path)
    with pose_path.open("rb") as handle:
        trajectory_data = pickle.load(handle)
    if not isinstance(trajectory_data, dict):
        raise RuntimeError(f"invalid traj_data.pkl mapping: {pose_path}")
    try:
        position_count = len(trajectory_data["position"])
        yaw_count = len(trajectory_data["yaw"])
    except (KeyError, TypeError) as exc:
        raise RuntimeError(f"invalid trajectory arrays: {pose_path}") from exc
    return min(contiguous_image_count(trajectory_dir), position_count, yaw_count)


def load_test_trajectories(scand_root: Path) -> list[Trajectory]:
    strict_splits = load_json(SCAND_SPLITS)
    test_ids = set(strict_splits["test"])
    if test_ids & set(strict_splits["train"]):
        raise RuntimeError("SCAND test/train trajectory overlap")
    if test_ids & set(strict_splits["val"]):
        raise RuntimeError("SCAND test/val trajectory overlap")

    canonical: dict[str, dict[str, Any]] = {}
    for line in CANONICAL_MANIFEST.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        trajectory_id = record["trajectory_id"]
        if trajectory_id in test_ids:
            canonical[trajectory_id] = record
    missing_metadata = sorted(test_ids - canonical.keys())
    if missing_metadata:
        raise RuntimeError(f"test trajectories absent from canonical manifest: {missing_metadata}")

    trajectories: list[Trajectory] = []
    for trajectory_id in sorted(test_ids):
        trajectory_dir = scand_root / trajectory_id
        if not trajectory_dir.is_dir():
            raise FileNotFoundError(trajectory_dir)
        num_frames = local_num_frames(trajectory_dir)
        if num_frames < CONTEXT_FRAMES + HORIZON:
            raise RuntimeError(
                f"strict-test trajectory has no H4 window: {trajectory_id} "
                f"({num_frames} usable frames)"
            )
        metadata = canonical[trajectory_id]
        trajectories.append(
            Trajectory(
                trajectory_id=trajectory_id,
                canonical_num_frames=int(metadata["num_frames"]),
                num_frames=num_frames,
                metadata=metadata,
                trajectory_dir=trajectory_dir,
            )
        )
    return trajectories


def balanced_bin_sizes(total: int, bins: int) -> list[int]:
    quotient, remainder = divmod(total, bins)
    return [quotient + (index < remainder) for index in range(bins)]


def assign_length_bins_and_window_counts(
    trajectories: list[Trajectory], seed: int
) -> list[list[Trajectory]]:
    ordered = sorted(
        trajectories, key=lambda item: (item.num_frames, item.trajectory_id)
    )
    bin_sizes = balanced_bin_sizes(len(ordered), len(LENGTH_NAMES))
    base_windows, extra_windows = divmod(TARGET_WINDOWS, len(ordered))
    if base_windows < 1:
        raise RuntimeError("target does not cover every strict-test trajectory")

    bins: list[list[Trajectory]] = []
    offset = 0
    assigned_extras = 0
    for length_bin, bin_size in enumerate(bin_sizes):
        members = ordered[offset : offset + bin_size]
        offset += bin_size
        target_for_bin = TARGET_WINDOWS // len(LENGTH_NAMES)
        extras_for_bin = target_for_bin - base_windows * len(members)
        if extras_for_bin < 0 or extras_for_bin > len(members):
            raise RuntimeError("cannot satisfy length-quartile window balance")
        ranked = sorted(
            members,
            key=lambda item: stable_key(
                seed, "scand_strict_h4_100_extra", item.trajectory_id
            ),
        )
        extra_ids = {
            item.trajectory_id for item in ranked[:extras_for_bin]
        }
        for item in members:
            item.length_bin = length_bin
            item.windows = base_windows + (item.trajectory_id in extra_ids)
        assigned_extras += extras_for_bin
        bins.append(members)

    if assigned_extras != extra_windows:
        raise RuntimeError(
            f"length-bin allocation assigned {assigned_extras} extras, "
            f"expected {extra_windows}"
        )
    return bins


def temporal_times(num_frames: int) -> tuple[int, ...]:
    span = num_frames - (CONTEXT_FRAMES + HORIZON)
    return tuple(
        CONTEXT_FRAMES - 1 + round(fraction * span)
        for fraction in TEMPORAL_FRACTIONS
    )


def assign_temporal_bins(trajectories: list[Trajectory], seed: int) -> None:
    target_per_bin = TARGET_WINDOWS // len(TEMPORAL_FRACTIONS)
    capacities = (target_per_bin,) * len(TEMPORAL_FRACTIONS)

    candidates: dict[str, list[tuple[int, ...]]] = {}
    for item in trajectories:
        times = temporal_times(item.num_frames)
        candidates[item.trajectory_id] = [
            combination
            for combination in itertools.combinations(
                range(len(TEMPORAL_FRACTIONS)), item.windows
            )
            if len({times[index] for index in combination}) == item.windows
        ]
        if not candidates[item.trajectory_id]:
            raise RuntimeError(
                f"cannot select {item.windows} distinct temporal windows from "
                f"{item.trajectory_id}"
            )

    ordered = sorted(
        trajectories,
        key=lambda item: (
            len(candidates[item.trajectory_id]),
            stable_key(seed, "scand_strict_h4_100_temporal_order", item.trajectory_id),
        ),
    )
    remaining_window_counts = [0] * (len(ordered) + 1)
    for index in range(len(ordered) - 1, -1, -1):
        remaining_window_counts[index] = (
            remaining_window_counts[index + 1] + ordered[index].windows
        )

    @lru_cache(maxsize=None)
    def solve(index: int, remaining: tuple[int, ...]) -> tuple[tuple[int, ...], ...] | None:
        if index == len(ordered):
            return () if all(value == 0 for value in remaining) else None
        if sum(remaining) != remaining_window_counts[index]:
            return None
        trajectories_left = len(ordered) - index
        if any(value > trajectories_left for value in remaining):
            return None

        item = ordered[index]
        choices = []
        for combination in candidates[item.trajectory_id]:
            if any(remaining[bin_index] <= 0 for bin_index in combination):
                continue
            urgency = -sum(remaining[bin_index] for bin_index in combination)
            choices.append(
                (
                    urgency,
                    stable_key(
                        seed,
                        "scand_strict_h4_100_temporal_choice",
                        item.trajectory_id,
                        ",".join(str(value) for value in combination),
                    ),
                    combination,
                )
            )
        for _, _, combination in sorted(choices):
            next_remaining = list(remaining)
            for bin_index in combination:
                next_remaining[bin_index] -= 1
            suffix = solve(index + 1, tuple(next_remaining))
            if suffix is not None:
                return (combination, *suffix)
        return None

    solution = solve(0, capacities)
    if solution is None:
        raise RuntimeError("could not balance the four temporal bins")
    for item, temporal_bins in zip(ordered, solution):
        item.temporal_bins = temporal_bins


def route_origins() -> dict[str, str]:
    payload = load_json(ROUTE_ASSIGNMENT)
    return {
        item["route_group"]: item["assignment_origin"]
        for item in payload["routes"]
    }


def build_records(
    trajectories: list[Trajectory], seed: int
) -> list[dict[str, Any]]:
    origins = route_origins()
    records: list[dict[str, Any]] = []
    for item in trajectories:
        times = temporal_times(item.num_frames)
        length_rank = sorted(
            (
                candidate
                for candidate in trajectories
                if candidate.length_bin == item.length_bin
            ),
            key=lambda candidate: stable_key(
                seed, "scand_strict_h4_100_length_rank", candidate.trajectory_id
            ),
        ).index(item)
        for trajectory_window_index, temporal_bin in enumerate(item.temporal_bins):
            curr_time = times[temporal_bin]
            record = {
                "trajectory_id": item.trajectory_id,
                "curr_time": curr_time,
                "context_times": list(range(curr_time - 3, curr_time + 1)),
                "goal_time": curr_time + HORIZON,
                "gt_times": list(range(curr_time + 1, curr_time + HORIZON + 1)),
                "canonical_num_frames": item.canonical_num_frames,
                "num_frames": item.num_frames,
                "length_bin": item.length_bin,
                "length_bin_name": LENGTH_NAMES[item.length_bin],
                "selection_rank_in_length_bin": length_rank,
                "temporal_bin": temporal_bin,
                "temporal_bin_name": TEMPORAL_NAMES[temporal_bin],
                "temporal_fraction": TEMPORAL_FRACTIONS[temporal_bin],
                "trajectory_window_index": trajectory_window_index,
                "strict_split": "test",
                "route_group": item.metadata["route_group"],
                "route_assignment_origin": origins[item.metadata["route_group"]],
                "robot": item.metadata["robot"],
                "survey": item.metadata["survey"],
                "source": item.metadata["source"],
                "bag_id": item.metadata["bag_id"],
            }
            records.append(record)

    records.sort(
        key=lambda record: stable_key(
            seed,
            "scand_strict_h4_100_record",
            record["trajectory_id"],
            record["temporal_bin"],
        )
    )
    for eval_index, record in enumerate(records):
        record["eval_index"] = eval_index
    return records


def count_by(records: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        value = str(record[key])
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def validate_records(
    records: list[dict[str, Any]],
    trajectories: list[Trajectory],
    expected_resolution: tuple[int, int],
) -> dict[str, int]:
    if len(records) != TARGET_WINDOWS:
        raise RuntimeError(f"expected {TARGET_WINDOWS} records, found {len(records)}")
    test_ids = {item.trajectory_id for item in trajectories}
    record_ids = {record["trajectory_id"] for record in records}
    if record_ids != test_ids:
        raise RuntimeError("H4-100 does not cover every strict-test trajectory")
    if count_by(records, "length_bin") != {str(index): 25 for index in range(4)}:
        raise RuntimeError("length bins are not exactly balanced")
    if count_by(records, "temporal_bin") != {str(index): 25 for index in range(4)}:
        raise RuntimeError("temporal bins are not exactly balanced")

    by_id = {item.trajectory_id: item for item in trajectories}
    checked_images: set[Path] = set()
    for record in records:
        item = by_id[record["trajectory_id"]]
        referenced_times = record["context_times"] + record["gt_times"]
        if min(referenced_times) < 0 or max(referenced_times) >= item.num_frames:
            raise RuntimeError(f"out-of-range H4 record: {record}")
        for time_index in referenced_times:
            image_path = item.trajectory_dir / f"{time_index}.jpg"
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            if image_path in checked_images:
                continue
            with Image.open(image_path) as image:
                if image.size != expected_resolution:
                    raise RuntimeError(
                        f"unexpected SCAND resolution {image.size}: {image_path}; "
                        f"expected {expected_resolution}"
                    )
                image.verify()
            checked_images.add(image_path)
    return {
        "unique_images_checked": len(checked_images),
        "canonical_local_frame_count_mismatches": sum(
            item.canonical_num_frames != item.num_frames for item in trajectories
        ),
    }


def build_manifest(
    records: list[dict[str, Any]],
    trajectories: list[Trajectory],
    length_bins: list[list[Trajectory]],
    seed: int,
    scand_root: Path,
    audit: dict[str, int],
) -> dict[str, Any]:
    split_lock = load_json(SPLIT_LOCK)
    split_lock_sha = sha256(SPLIT_LOCK)
    route_summary = []
    for route_group, windows in count_by(records, "route_group").items():
        route_summary.append(
            {
                "route_group": route_group,
                "selected_windows": windows,
                "selected_trajectories": len(
                    {
                        record["trajectory_id"]
                        for record in records
                        if record["route_group"] == route_group
                    }
                ),
            }
        )
    return {
        "schema_version": 1,
        "name": "scand_strict_test_h4_100_seed17",
        "purpose": (
            "Model-independent paired comparison of frozen trajectory scorers "
            "on the project SCAND strict outer-test split"
        ),
        "dataset": "scand",
        "strict_split": "test",
        "context_frames": CONTEXT_FRAMES,
        "horizon": HORIZON,
        "index_tuple": "(trajectory_id, curr_time, 4, 4)",
        "valid_curr_time_rule": "3 <= curr_time <= local_num_frames - 5",
        "selection_seed": seed,
        "selection_key": "SHA256(seed|namespace|trajectory_id|temporal_bin)",
        "selection_uses_model_outputs": False,
        "selected_windows": len(records),
        "selected_unique_trajectories": len(
            {record["trajectory_id"] for record in records}
        ),
        "strict_test_trajectories": len(trajectories),
        "all_strict_test_trajectories_covered": True,
        "windows_per_trajectory": count_by(
            [
                {
                    "windows": sum(
                        record["trajectory_id"] == item.trajectory_id
                        for record in records
                    )
                }
                for item in trajectories
            ],
            "windows",
        ),
        "length_bins": [
            {
                "index": index,
                "name": LENGTH_NAMES[index],
                "min_num_frames": min(item.num_frames for item in members),
                "max_num_frames": max(item.num_frames for item in members),
                "population_trajectories": len(members),
                "selected_windows": sum(item.windows for item in members),
            }
            for index, members in enumerate(length_bins)
        ],
        "temporal_bins": [
            {
                "index": index,
                "name": TEMPORAL_NAMES[index],
                "fraction": TEMPORAL_FRACTIONS[index],
                "selected_windows": sum(
                    record["temporal_bin"] == index for record in records
                ),
            }
            for index in range(len(TEMPORAL_FRACTIONS))
        ],
        "route_summary": route_summary,
        "robot_counts": count_by(records, "robot"),
        "survey_counts": count_by(records, "survey"),
        "source_counts": count_by(records, "source"),
        "local_data_root": str(scand_root.resolve()),
        "expected_image_resolution": [320, 240],
        "data_audit": audit,
        "split_contract": {
            "release_id": split_lock.get("release_id"),
            "split_lock_sha256": split_lock_sha,
            "scand_strict_splits_sha256": sha256(SCAND_SPLITS),
            "canonical_source_manifest_sha256": sha256(CANONICAL_MANIFEST),
            "train_overlap": 0,
            "val_overlap": 0,
        },
        "records": records,
    }


def render_readme(manifest: dict[str, Any]) -> str:
    mismatches = manifest["data_audit"]["canonical_local_frame_count_mismatches"]
    return f"""# SCAND strict-test H4-100 (seed 17)

This benchmark freezes 100 H4 windows for a paired comparison of trajectory
scorers. It uses only the 43 trajectories in the project SCAND strict outer-test
split, covers every one of those trajectories, and has zero trajectory overlap
with the project train and validation splits.

## Model-independent selection

- The selection uses seed 17 and SHA-256 ordering only. No model prediction,
  score, metric, checkpoint, proposal, or label is read during selection.
- Every strict-test trajectory contributes two windows; 14 deterministically
  selected trajectories contribute a third window.
- Each trajectory-length quartile contributes exactly 25 windows.
- Each temporal-position quartile contributes exactly 25 windows.
- Every case uses context `t-3..t`, future ground truth `t+1..t+4`, and goal
  frame `t+4`.

The local SCAND copy contains {mismatches} trajectories whose usable local frame
count is shorter than the historical canonical count. Selection and range checks
therefore use the contiguous local JPEG/position/yaw intersection; both counts
remain recorded per case for auditability.

## Fair scorer comparison

Use `navigation_eval.pkl` unchanged for both scorers. To isolate scorer quality,
freeze one candidate/proposal pool per `eval_index` and give the exact same pool,
ordering, labels, normalization, and candidate mask to both scorers. Do not tune
thresholds, temperatures, or weights on this outer-test set. Report paired
per-case deltas and a trajectory-clustered confidence interval because multiple
windows can come from one trajectory.

If a scorer trained, calibrated, selected a checkpoint, or tuned hyperparameters
on any of these 43 trajectories, this benchmark is not strict for that scorer.
Audit both scorers' training provenance before making a strict comparison claim.

`navigation_eval.pkl` is the canonical runner input. `manifest.json` and
`manifest.csv` preserve selection provenance, and `SHA256SUMS.txt` freezes the
package bytes.
"""


def write_package(output: Path, manifest: dict[str, Any], force: bool) -> None:
    if output.exists():
        if not force:
            raise FileExistsError(f"output exists; pass --force to replace: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True)

    records = manifest["records"]
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    csv_fields = [
        "eval_index",
        "trajectory_id",
        "num_frames",
        "canonical_num_frames",
        "route_group",
        "robot",
        "survey",
        "length_bin",
        "length_bin_name",
        "temporal_bin",
        "temporal_bin_name",
        "temporal_fraction",
        "curr_time",
        "context_times",
        "goal_time",
        "gt_times",
        "trajectory_window_index",
        "strict_split",
    ]
    with (output / "manifest.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            row = dict(record)
            row["context_times"] = ";".join(map(str, row["context_times"]))
            row["gt_times"] = ";".join(map(str, row["gt_times"]))
            writer.writerow(row)

    navigation_eval = [
        (record["trajectory_id"], record["curr_time"], HORIZON, HORIZON)
        for record in records
    ]
    with (output / "navigation_eval.pkl").open("wb") as handle:
        pickle.dump(navigation_eval, handle, protocol=4)
    (output / "traj_names.txt").write_text(
        "".join(f"{record['trajectory_id']}\n" for record in records),
        encoding="utf-8",
        newline="\n",
    )
    (output / "README.md").write_text(
        render_readme(manifest), encoding="utf-8", newline="\n"
    )

    frozen_files = (
        "navigation_eval.pkl",
        "manifest.csv",
        "manifest.json",
        "traj_names.txt",
        "README.md",
    )
    (output / "SHA256SUMS.txt").write_text(
        "".join(f"{sha256(output / name)}  {name}\n" for name in frozen_files),
        encoding="utf-8",
        newline="\n",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, default=DEFAULT_ASSETS)
    parser.add_argument("--scand-root", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    scand_root = args.scand_root or scand_root_from_assets(args.assets)
    trajectories = load_test_trajectories(scand_root)
    length_bins = assign_length_bins_and_window_counts(trajectories, args.seed)
    assign_temporal_bins(trajectories, args.seed)
    records = build_records(trajectories, args.seed)
    audit = validate_records(records, trajectories, expected_resolution=(320, 240))
    manifest = build_manifest(
        records, trajectories, length_bins, args.seed, scand_root, audit
    )
    write_package(args.output, manifest, args.force)

    print(
        "PASS: "
        f"{len(records)} SCAND strict-test H4 windows, "
        f"{len(trajectories)} trajectories, "
        "25 windows per length bin, 25 per temporal bin, "
        "zero train/val overlap"
    )
    print(f"output: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
