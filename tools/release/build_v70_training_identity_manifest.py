#!/usr/bin/env python3
"""Export label-free identities for every project dataset used by frozen v70."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "data_splits/v70_training_identity"

SOURCES = (
    (
        "recon_train_fit",
        "recon",
        "train-fit",
        ROOT / "data/inner_manifests/recon/train-fit.windows.jsonl",
    ),
    (
        "recon_scorer_cal",
        "recon",
        "scorer-cal",
        ROOT / "data/inner_manifests/recon/scorer-cal.windows.jsonl",
    ),
    (
        "scand_train_fit",
        "scand",
        "train-fit",
        ROOT / "data/inner_manifests/scand/train-fit.windows.jsonl",
    ),
    (
        "scand_scorer_cal",
        "scand",
        "scorer-cal",
        ROOT
        / (
            "outputs/scand_domain_adaptation_v1/manifests/"
            "scorer_cal_rgb_available.windows.jsonl"
        ),
    ),
)

STRICT_SPLITS = {
    "recon": ROOT / "data/split_lock/recon_strict_splits.json",
    "scand": ROOT / "data/split_lock/scand_strict_splits.json",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def normalized_identity(
    row: dict[str, Any],
    *,
    source: str,
    dataset: str,
    partition: str,
) -> dict[str, Any]:
    trajectory_id = str(row["trajectory_id"])
    current_t = int(row["current_t"])
    route_group = row.get("route_group") or row.get("route_id")
    return {
        "dataset": dataset,
        "source": source,
        "partition": partition,
        "record_key": f"{trajectory_id}:{current_t}",
        "trajectory_id": trajectory_id,
        "current_t": current_t,
        "goal_t": int(row["goal_t"]),
        "route_group": None if route_group is None else str(route_group),
        "bag_id": row.get("bag_id"),
        "window_uid": row.get("window_uid"),
    }


def build(output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite: {output}")
    output.mkdir(parents=True)
    manifests_dir = output / "source_manifests"
    splits_dir = output / "strict_splits"
    manifests_dir.mkdir()
    splits_dir.mkdir()

    identities: list[dict[str, Any]] = []
    source_summary: list[dict[str, Any]] = []
    for source, dataset, partition, path in SOURCES:
        rows = load_jsonl(path)
        normalized = [
            normalized_identity(
                row,
                source=source,
                dataset=dataset,
                partition=partition,
            )
            for row in rows
        ]
        identities.extend(normalized)
        copied = manifests_dir / f"{source}.windows.jsonl"
        shutil.copy2(path, copied)
        source_summary.append(
            {
                "source": source,
                "dataset": dataset,
                "partition": partition,
                "windows": len(normalized),
                "trajectories": len(
                    {row["trajectory_id"] for row in normalized}
                ),
                "route_groups": len(
                    {
                        row["route_group"]
                        for row in normalized
                        if row["route_group"] is not None
                    }
                ),
                "input_sha256": sha256(path),
                "copied_manifest": str(copied.relative_to(output)),
            }
        )

    identities.sort(
        key=lambda row: (
            row["dataset"],
            row["trajectory_id"],
            row["current_t"],
            row["source"],
        )
    )
    identity_path = output / "v70_training_identities.jsonl"
    identity_path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n"
            for row in identities
        ),
        encoding="utf-8",
    )

    strict_summary: dict[str, Any] = {}
    for dataset, path in STRICT_SPLITS.items():
        payload = json.loads(path.read_text(encoding="utf-8"))
        copied = splits_dir / path.name
        shutil.copy2(path, copied)
        strict_summary[dataset] = {
            "sha256": sha256(path),
            "train_trajectories": len(payload["train"]),
            "val_trajectories": len(payload["val"]),
            "test_trajectories": len(payload["test"]),
            "copied_split": str(copied.relative_to(output)),
        }

    summary = {
        "schema": "v70_training_identity_manifest_v1",
        "status": "complete_label_free_identity_export",
        "purpose": (
            "Prevent record, trajectory, or route overlap when constructing "
            "new v70 evaluation cohorts."
        ),
        "identity_levels": ["record_key", "trajectory_id", "route_group"],
        "strict_rule": (
            "A new project test must come only from the frozen strict test "
            "partition and should have zero trajectory/route overlap with "
            "v70_training_identities.jsonl."
        ),
        "arm_d_sources": source_summary,
        "strict_project_splits": strict_summary,
        "identity_rows": len(identities),
        "unique_record_keys": len(
            {(row["dataset"], row["record_key"]) for row in identities}
        ),
        "unique_trajectories": {
            dataset: len(
                {
                    row["trajectory_id"]
                    for row in identities
                    if row["dataset"] == dataset
                }
            )
            for dataset in STRICT_SPLITS
        },
        "identity_jsonl": identity_path.name,
        "identity_jsonl_sha256": sha256(identity_path),
        "not_training": {
            "recon_A_to_H": "spent development evaluation only",
            "fresh_scand_100": "spent development evaluation only",
            "reserve": "one-time evaluation only; never training/calibration",
        },
        "external_pretraining": {
            "dinov3": (
                "Upstream LVD-1689M pretraining; exact image identities are "
                "not enumerated by this project."
            ),
            "nomad": (
                "Upstream nomad.pth config enables RECON training and leaves "
                "SCAND disabled; use the official NoMaD-unseen RECON split."
            ),
        },
        "contains_labels": False,
        "contains_reserve_identities": False,
    }
    summary_path = output / "manifest.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    summary = build(args.output)
    print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
