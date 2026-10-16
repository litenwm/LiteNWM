#!/usr/bin/env python3
"""Build a model-independent SACSoN strict-test H4-100 benchmark."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import re
import shutil
from pathlib import Path
from typing import Any

from PIL import Image

CONTEXT_FRAMES = 4
HORIZON = 4
TARGET_WINDOWS = 100
TEMPORAL_FRACTIONS = (0.125, 0.375, 0.625, 0.875)
TEMPORAL_NAMES = ("early", "mid_early", "mid_late", "late")
LENGTH_NAMES = ("Q1_shortest", "Q2", "Q3", "Q4_longest")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable(seed: int, namespace: str, *parts: object) -> str:
    value = "|".join((str(seed), namespace, *(str(part) for part in parts)))
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_names(path: Path) -> list[str]:
    values = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    values = [value for value in values if value]
    if len(values) != len(set(values)):
        raise ValueError(f"duplicate trajectory identities in {path}")
    return values


def session_id(trajectory_id: str) -> str:
    value = re.sub(r"_\d+_\d+$", "", trajectory_id)
    if value == trajectory_id:
        raise ValueError(f"unsupported SACSoN trajectory ID: {trajectory_id}")
    return value


def local_num_frames(path: Path) -> int:
    image_indices = {
        int(item.stem) for item in path.glob("*.jpg") if item.stem.isdigit()
    }
    count = 0
    while count in image_indices:
        count += 1
    if len(image_indices) != count:
        raise RuntimeError(f"non-contiguous JPEG sequence in {path}")
    with (path / "traj_data.pkl").open("rb") as handle:
        trajectory_data = pickle.load(handle)
    if not isinstance(trajectory_data, dict):
        raise RuntimeError(f"invalid trajectory data in {path}")
    return min(count, len(trajectory_data["position"]), len(trajectory_data["yaw"]))


def scan_test_trajectories(
    root: Path, test_ids: list[str], split_lock: dict[str, Any]
) -> list[dict[str, Any]]:
    test = set(map(str, split_lock["sacson"]["splits"]["test"]))
    train = set(map(str, split_lock["sacson"]["splits"]["train"]))
    val = set(map(str, split_lock["sacson"]["splits"]["val"]))
    listed = set(test_ids)
    if listed != test:
        raise ValueError(
            "test list must exactly equal active SACSoN test membership: "
            f"listed_only={sorted(listed-test)[:5]} lock_only={sorted(test-listed)[:5]}"
        )
    if listed & train or listed & val:
        raise ValueError("SACSoN test list overlaps active train/val membership")

    records = []
    for trajectory_id in sorted(test_ids):
        path = root / trajectory_id
        if not path.is_dir():
            raise FileNotFoundError(path)
        num_frames = local_num_frames(path)
        if num_frames < CONTEXT_FRAMES + HORIZON:
            continue
        records.append(
            {
                "trajectory_id": trajectory_id,
                "trajectory_dir": path,
                "num_frames": num_frames,
                "h4_windows": max(num_frames - 7, 0),
                "session_id": session_id(trajectory_id),
            }
        )
    if len(records) < TARGET_WINDOWS:
        raise RuntimeError(
            f"only {len(records)} SACSoN test trajectories are H4-eligible; "
            f"cannot select {TARGET_WINDOWS} unique windows"
        )
    return records


def select_records(records: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    ordered = sorted(records, key=lambda row: (row["num_frames"], row["trajectory_id"]))
    quotient, remainder = divmod(len(ordered), len(LENGTH_NAMES))
    selected: list[dict[str, Any]] = []
    offset = 0
    for length_bin in range(len(LENGTH_NAMES)):
        size = quotient + (length_bin < remainder)
        members = ordered[offset : offset + size]
        offset += size
        ranked = sorted(
            members,
            key=lambda row: stable(seed, "sacson_h4_100_select", row["trajectory_id"]),
        )
        for row in ranked[: TARGET_WINDOWS // len(LENGTH_NAMES)]:
            selected.append({**row, "length_bin": length_bin})
    if len(selected) != TARGET_WINDOWS:
        raise RuntimeError(f"selected {len(selected)} windows, expected {TARGET_WINDOWS}")

    temporal_loads = [0] * len(TEMPORAL_FRACTIONS)
    for row in sorted(
        selected,
        key=lambda item: stable(seed, "sacson_h4_100_temporal", item["trajectory_id"]),
    ):
        lowest = min(temporal_loads)
        choices = [index for index, value in enumerate(temporal_loads) if value == lowest]
        temporal_bin = min(
            choices,
            key=lambda index: stable(
                seed, "sacson_h4_100_temporal_choice", row["trajectory_id"], index
            ),
        )
        temporal_loads[temporal_bin] += 1
        span = row["num_frames"] - (CONTEXT_FRAMES + HORIZON)
        current_t = CONTEXT_FRAMES - 1 + round(TEMPORAL_FRACTIONS[temporal_bin] * span)
        row.update(
            {
                "temporal_bin": temporal_bin,
                "curr_time": current_t,
                "context_times": list(range(current_t - 3, current_t + 1)),
                "goal_time": current_t + HORIZON,
                "gt_times": list(range(current_t + 1, current_t + HORIZON + 1)),
            }
        )
    return sorted(
        selected,
        key=lambda row: stable(seed, "sacson_h4_100_record", row["trajectory_id"]),
    )


def count_by(records: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in records:
        value = str(row[key])
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def validate_records(records: list[dict[str, Any]]) -> dict[str, int]:
    if len(records) != TARGET_WINDOWS:
        raise RuntimeError(f"expected {TARGET_WINDOWS} records, found {len(records)}")
    if len({row["trajectory_id"] for row in records}) != TARGET_WINDOWS:
        raise RuntimeError("SACSoN H4-100 must use one window per trajectory")
    if count_by(records, "length_bin") != {str(index): 25 for index in range(4)}:
        raise RuntimeError("length quartiles are not exactly balanced")
    if count_by(records, "temporal_bin") != {str(index): 25 for index in range(4)}:
        raise RuntimeError("temporal quartiles are not exactly balanced")
    checked_images: set[Path] = set()
    for row in records:
        referenced = row["context_times"] + row["gt_times"]
        if min(referenced) < 0 or max(referenced) >= row["num_frames"]:
            raise RuntimeError(f"out-of-range SACSoN H4 record: {row}")
        for time_index in referenced:
            path = row["trajectory_dir"] / f"{time_index}.jpg"
            if not path.is_file():
                raise FileNotFoundError(path)
            if path in checked_images:
                continue
            with Image.open(path) as image:
                if image.size != (320, 240):
                    raise RuntimeError(
                        f"unexpected SACSoN resolution {image.size}; expected (320, 240): {path}"
                    )
                image.verify()
            checked_images.add(path)
    return {"unique_images_checked": len(checked_images)}


def build_manifest(
    records: list[dict[str, Any]],
    all_test_records: list[dict[str, Any]],
    split_lock_path: Path,
    test_list_path: Path,
    root: Path,
    seed: int,
    audit: dict[str, int],
) -> dict[str, Any]:
    split_lock = json.loads(split_lock_path.read_text(encoding="utf-8"))
    return {
        "schema_version": 1,
        "name": "sacson_strict_test_h4_100_seed17",
        "purpose": "Model-independent SACSoN outer-test H4-100 benchmark",
        "dataset": "sacson",
        "strict_split": "test",
        "context_frames": CONTEXT_FRAMES,
        "horizon": HORIZON,
        "index_tuple": "(trajectory_id, curr_time, 4, 4)",
        "valid_curr_time_rule": "3 <= curr_time <= local_num_frames - 5",
        "selection_seed": seed,
        "selection_key": "SHA256(seed|namespace|trajectory_id)",
        "selection_uses_model_outputs": False,
        "selected_windows": len(records),
        "selected_unique_trajectories": len({row["trajectory_id"] for row in records}),
        "sealed_outer_test_trajectories": len(read_names(test_list_path)),
        "h4_eligible_outer_test_trajectories": len(all_test_records),
        "all_selected_from_sealed_outer_test": True,
        "windows_per_trajectory": {"1": len(records)},
        "length_bins": [
            {
                "index": index,
                "name": LENGTH_NAMES[index],
                "selected_windows": sum(row["length_bin"] == index for row in records),
            }
            for index in range(4)
        ],
        "temporal_bins": [
            {
                "index": index,
                "name": TEMPORAL_NAMES[index],
                "fraction": TEMPORAL_FRACTIONS[index],
                "selected_windows": sum(row["temporal_bin"] == index for row in records),
            }
            for index in range(4)
        ],
        "session_counts": count_by(records, "session_id"),
        "expected_image_resolution": [320, 240],
        "data_audit": audit,
        "split_contract": {
            "split_lock_sha256": sha256(split_lock_path),
            "test_list_sha256": sha256(test_list_path),
            "train_overlap": 0,
            "val_overlap": 0,
            "outer_test_protocol": split_lock["sacson"].get("outer_test_protocol"),
        },
        "records": [
            {key: value for key, value in row.items() if key != "trajectory_dir"}
            for row in records
        ],
        "local_data_root": str(root.resolve()),
    }


def write_package(output: Path, manifest: dict[str, Any], force: bool) -> None:
    if output.exists():
        if not force:
            raise FileExistsError(f"output exists; pass --force to replace: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True)
    records = manifest["records"]
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    fields = [
        "eval_index", "trajectory_id", "num_frames", "session_id", "length_bin",
        "length_bin_name", "temporal_bin", "temporal_bin_name", "temporal_fraction",
        "curr_time", "context_times", "goal_time", "gt_times", "strict_split",
    ]
    with (output / "manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for index, record in enumerate(records):
            row = dict(record)
            row["eval_index"] = index
            row["length_bin_name"] = LENGTH_NAMES[row["length_bin"]]
            row["temporal_bin_name"] = TEMPORAL_NAMES[row["temporal_bin"]]
            row["temporal_fraction"] = TEMPORAL_FRACTIONS[row["temporal_bin"]]
            row["strict_split"] = "test"
            row["context_times"] = ";".join(map(str, row["context_times"]))
            row["gt_times"] = ";".join(map(str, row["gt_times"]))
            writer.writerow(row)
    with (output / "navigation_eval.pkl").open("wb") as handle:
        pickle.dump(
            [(row["trajectory_id"], row["curr_time"], HORIZON, HORIZON) for row in records],
            handle,
            protocol=4,
        )
    (output / "traj_names.txt").write_text(
        "".join(f"{row['trajectory_id']}\n" for row in records), encoding="utf-8"
    )
    (output / "README.md").write_text(
        "# SACSoN strict-test H4-100 (seed 17)\n\n"
        "This package freezes 100 H4 windows selected from the already sealed "
        "SACSoN outer-test membership. It uses one window per unique H4-eligible "
        "test trajectory, exactly 25 windows per trajectory-length quartile and "
        "exactly 25 per temporal-position quartile. Selection uses SHA-256 ordering "
        "only; no model outputs, proposals, labels, or checkpoints are read.\n\n"
        "The active train and validation memberships are checked for zero trajectory "
        "overlap before the package is written. Freeze this package before any model "
        "selection or threshold tuning, and reserve it for the final independent test.\n\n"
        "All referenced images are verified as 320x240 JPEGs. `navigation_eval.pkl` "
        "is the canonical runner input; `manifest.json` and `manifest.csv` preserve "
        "the selection provenance.\n",
        encoding="utf-8",
    )
    frozen = ("navigation_eval.pkl", "manifest.csv", "manifest.json", "traj_names.txt", "README.md")
    (output / "SHA256SUMS.txt").write_text(
        "".join(f"{sha256(output / name)}  {name}\n" for name in frozen), encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sacson-root", type=Path, required=True)
    parser.add_argument("--split-lock", type=Path, required=True)
    parser.add_argument("--test-list", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    split_lock = json.loads(args.split_lock.read_text(encoding="utf-8"))
    test_ids = read_names(args.test_list)
    all_test_records = scan_test_trajectories(args.sacson_root, test_ids, split_lock)
    records = select_records(all_test_records, args.seed)
    audit = validate_records(records)
    manifest = build_manifest(
        records, all_test_records, args.split_lock, args.test_list,
        args.sacson_root, args.seed, audit
    )
    write_package(args.output, manifest, args.force)
    print(
        f"PASS: {len(records)} SACSoN strict-test H4 windows, "
        f"{len(all_test_records)} H4-eligible outer-test trajectories, "
        "25 windows per length quartile, 25 per temporal quartile, "
        "zero train/val trajectory overlap"
    )
    print(f"output: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
