#!/usr/bin/env python3
"""Build route/session-disjoint SACSoN scorer inner manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re


PARTITIONS = ("train-fit", "scorer-cal", "cem-cal", "internal-holdout")
FRACTIONS = {
    "train-fit": 0.70,
    "scorer-cal": 0.10,
    "cem-cal": 0.10,
    "internal-holdout": 0.10,
}


def session_id(trajectory_id: str) -> str:
    value = re.sub(r"_\d+_\d+$", "", trajectory_id)
    if value == trajectory_id:
        raise ValueError(f"unsupported SACSoN trajectory id: {trajectory_id}")
    return value


def stable(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}|{value}".encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latent-root", type=Path, required=True)
    parser.add_argument("--split-lock", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260727)
    args = parser.parse_args()

    split_lock = json.loads(args.split_lock.read_text(encoding="utf-8"))
    outer_train = set(map(str, split_lock["sacson"]["splits"]["train"]))
    metadata = json.loads((args.latent_root / "metadata.json").read_text())
    records = {
        str(row["trajectory_id"]): int(row["num_frames"])
        for row in metadata["trajectories"]
        if str(row["trajectory_id"]) in outer_train
    }
    groups: dict[str, list[tuple[str, int]]] = {}
    for trajectory_id, frames in records.items():
        if frames < 8:
            continue
        groups.setdefault(session_id(trajectory_id), []).append(
            (trajectory_id, frames)
        )
    if len(groups) < len(PARTITIONS):
        raise ValueError("not enough SACSoN sessions for four partitions")

    ordered = sorted(
        groups,
        key=lambda name: (
            -sum(frames - 7 for _, frames in groups[name]),
            stable(args.seed, name),
            name,
        ),
    )
    loads = {partition: 0 for partition in PARTITIONS}
    assignment: dict[str, str] = {}
    for index, group in enumerate(ordered):
        missing = [p for p in PARTITIONS if p not in assignment.values()]
        if missing and len(ordered) - index <= len(missing):
            chosen = missing[0]
        else:
            chosen = min(
                PARTITIONS,
                key=lambda p: (loads[p] / FRACTIONS[p], p),
            )
        assignment[group] = chosen
        loads[chosen] += sum(frames - 7 for _, frames in groups[group])

    args.out_root.mkdir(parents=True, exist_ok=False)
    counts = {partition: 0 for partition in PARTITIONS}
    for partition in PARTITIONS:
        rows: list[dict[str, object]] = []
        for group in sorted(assignment):
            if assignment[group] != partition:
                continue
            for trajectory_id, frames in sorted(groups[group]):
                for current_t in range(3, frames - 4):
                    rows.append(
                        {
                            "window_uid": (
                                f"sacson:{partition}:{trajectory_id}:{current_t}"
                            ),
                            "dataset": "sacson",
                            "partition": partition,
                            "trajectory_id": trajectory_id,
                            "route_id": group,
                            "current_t": current_t,
                            "goal_t": current_t + 4,
                            "history_timesteps": list(
                                range(current_t - 3, current_t + 1)
                            ),
                        }
                    )
        counts[partition] = len(rows)
        path = args.out_root / f"{partition}.windows.jsonl"
        path.write_text(
            "".join(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                for row in rows
            ),
            encoding="utf-8",
        )

    report = {
        "schema": "sacson_inner_manifests_v1",
        "outer_train_trajectories": len(outer_train),
        "eligible_trajectories": len(records),
        "session_groups": len(groups),
        "partition_session_counts": {
            partition: sum(assignment[group] == partition for group in assignment)
            for partition in PARTITIONS
        },
        "partition_window_counts": counts,
        "split_lock": str(args.split_lock),
    }
    (args.out_root / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
