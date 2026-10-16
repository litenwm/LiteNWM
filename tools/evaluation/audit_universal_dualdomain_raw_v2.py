#!/usr/bin/env python3
"""Freeze the data, feature-availability, and leakage audit for universal v2."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
REPEAT_SUFFIX = re.compile(r"_r\d+$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT
        / "configs/scoring/universal_dualdomain_raw_v2.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT
        / "outputs/universal_dualdomain_raw_v2/preflight/"
        "data_feature_audit.json",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def record_keys(values: list[dict[str, Any]]) -> set[str]:
    return {
        f"{value['trajectory_id']}:{int(value['current_t'])}"
        for value in values
    }


def manifest_summary(path: Path, dataset: str) -> dict[str, Any]:
    values = rows(path)
    result = {
        "path": str(path),
        "sha256": sha256(path),
        "states": len(values),
        "trajectories": len({str(value["trajectory_id"]) for value in values}),
        "route_groups": len(
            {
                str(value["route_id"] if dataset == "recon" else value["route_group"])
                for value in values
            }
        ),
    }
    if dataset == "scand":
        result["bags"] = len({str(value["bag_id"]) for value in values})
    return result


def pairwise_overlap(
    manifests: dict[str, list[dict[str, Any]]], dataset: str
) -> dict[str, dict[str, int]]:
    identities: dict[str, dict[str, set[str]]] = {}
    for name, values in manifests.items():
        identities[name] = {
            "record": record_keys(values),
            "trajectory": {str(value["trajectory_id"]) for value in values},
            "route": {
                str(value["route_id"] if dataset == "recon" else value["route_group"])
                for value in values
            },
        }
        if dataset == "scand":
            identities[name]["bag"] = {str(value["bag_id"]) for value in values}
    result: dict[str, dict[str, int]] = {}
    names = list(manifests)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            result[f"{left}__{right}"] = {
                key: len(identities[left][key].intersection(identities[right][key]))
                for key in identities[left]
            }
    return result


def npz_records(path: Path) -> tuple[np.ndarray, dict[str, tuple[list[int], str]]]:
    with np.load(path, allow_pickle=False) as payload:
        records = payload["record_key"].astype(str)
        contract = {
            key: (list(payload[key].shape), str(payload[key].dtype))
            for key in payload.files
        }
    return records, contract


def route_from_record(value: str) -> str:
    trajectory = str(value).rpartition(":")[0]
    return REPEAT_SUFFIX.sub("", trajectory)


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config.get("cem") is not False:
        raise RuntimeError("universal v2 config unexpectedly enables CEM")

    manifest_paths = {
        dataset: {
            partition: REPO_ROOT
            / f"data/inner_manifests/{dataset}/{partition}.windows.jsonl"
            for partition in (
                "train-fit",
                "scorer-cal",
                "cem-cal",
                "internal-holdout",
            )
        }
        for dataset in ("recon", "scand")
    }
    manifest_rows = {
        dataset: {
            partition: rows(path)
            for partition, path in paths.items()
        }
        for dataset, paths in manifest_paths.items()
    }
    manifests = {
        dataset: {
            partition: manifest_summary(path, dataset)
            for partition, path in paths.items()
        }
        for dataset, paths in manifest_paths.items()
    }
    overlaps = {
        dataset: pairwise_overlap(manifest_rows[dataset], dataset)
        for dataset in ("recon", "scand")
    }
    if any(
        count
        for dataset in overlaps.values()
        for pair in dataset.values()
        for key, count in pair.items()
        if key != "record"
    ):
        raise RuntimeError("canonical four-way group leakage detected")

    recon_train = record_keys(manifest_rows["recon"]["train-fit"])
    recon_cal = record_keys(manifest_rows["recon"]["scorer-cal"])
    recon_all = {
        partition: record_keys(values)
        for partition, values in manifest_rows["recon"].items()
    }
    legacy_train_path = (
        REPO_ROOT
        / "outputs/recon_dense_meta_dinov3_v1/features/dense_train_features.npz"
    )
    legacy_cal_path = (
        REPO_ROOT
        / "outputs/recon_dense_meta_dinov3_v1/features/dense_cal_features.npz"
    )
    legacy_train, legacy_train_contract = npz_records(legacy_train_path)
    legacy_cal, legacy_cal_contract = npz_records(legacy_cal_path)
    legacy_train_set = set(legacy_train.tolist())
    legacy_cal_set = set(legacy_cal.tolist())

    legacy_split_membership = {
        partition: len(legacy_train_set.intersection(keys))
        for partition, keys in recon_all.items()
    }
    split_lock_path = REPO_ROOT / "data/split_lock/split_lock.json"
    split_lock_membership: Counter[str] = Counter()
    if split_lock_path.is_file():
        split_lock = json.loads(split_lock_path.read_text(encoding="utf-8"))
        split_lists = split_lock.get("recon", {}).get("splits", {})
        assignment = {
            str(trajectory): str(split)
            for split, trajectories in split_lists.items()
            for trajectory in trajectories
        }
        for record in legacy_train:
            trajectory = str(record).rpartition(":")[0]
            value = assignment.get(trajectory)
            split_lock_membership[str(value or "unresolved")] += 1

    scand_train_feature = (
        REPO_ROOT
        / "outputs/scand_domain_adaptation_v1/features/train_fit/"
        "scand_inner_train_fit_spacing038_features.npz"
    )
    scand_cal_feature = (
        REPO_ROOT
        / "outputs/scand_domain_adaptation_v1/features/scorer_cal/"
        "scand_inner_scorer_cal_spacing038_features.npz"
    )
    scand_train_records, scand_train_contract = npz_records(scand_train_feature)
    scand_cal_records, scand_cal_contract = npz_records(scand_cal_feature)
    scand_train_expected = record_keys(manifest_rows["scand"]["train-fit"])
    scand_cal_expected = record_keys(manifest_rows["scand"]["scorer-cal"])

    report = {
        "schema": "universal_dualdomain_raw_v2_data_feature_audit",
        "status": "canonical_identity_frozen_recon_reexport_required",
        "config": str(args.config),
        "config_sha256": sha256(args.config),
        "manifests": manifests,
        "canonical_group_overlap": overlaps,
        "canonical_train_total_states": (
            manifests["recon"]["train-fit"]["states"]
            + manifests["scand"]["train-fit"]["states"]
        ),
        "canonical_train_total_k17_candidates": (
            manifests["recon"]["train-fit"]["states"]
            + manifests["scand"]["train-fit"]["states"]
        )
        * 17,
        "legacy_recon_cache": {
            "train_path": str(legacy_train_path),
            "train_sha256": sha256(legacy_train_path),
            "train_states": len(legacy_train),
            "train_contract": legacy_train_contract,
            "canonical_train_exact_overlap": len(
                legacy_train_set.intersection(recon_train)
            ),
            "legacy_states_outside_canonical_train": len(
                legacy_train_set.difference(recon_train)
            ),
            "canonical_train_states_missing": len(
                recon_train.difference(legacy_train_set)
            ),
            "canonical_four_way_exact_membership": legacy_split_membership,
            "upstream_split_lock_membership": dict(split_lock_membership),
            "cal_path": str(legacy_cal_path),
            "cal_sha256": sha256(legacy_cal_path),
            "cal_states": len(legacy_cal),
            "canonical_scorer_cal_exact_overlap": len(
                legacy_cal_set.intersection(recon_cal)
            ),
            "role": "legacy_noncanonical_spent_baseline_A_only",
        },
        "feature_availability": {
            "recon": {
                "canonical_train_required": len(recon_train),
                "canonical_train_reexport": len(recon_train),
                "canonical_scorer_cal_required": len(recon_cal),
                "canonical_scorer_cal_reexport": len(recon_cal),
                "reason": "legacy pools are identity-incompatible; do not mix 25 overlapping states with newly sampled K17",
            },
            "scand": {
                "train_feature": str(scand_train_feature),
                "train_feature_sha256": sha256(scand_train_feature),
                "train_exact_match": (
                    set(scand_train_records.tolist()) == scand_train_expected
                ),
                "train_states": len(scand_train_records),
                "train_contract": scand_train_contract,
                "scorer_cal_feature": str(scand_cal_feature),
                "scorer_cal_feature_sha256": sha256(scand_cal_feature),
                "scorer_cal_states": len(scand_cal_records),
                "scorer_cal_missing_rgb_records": sorted(
                    scand_cal_expected.difference(set(scand_cal_records.tolist()))
                ),
                "scorer_cal_contract": scand_cal_contract,
            },
        },
        "outer_availability": {
            "recon": "strict fresh H4-100 available; manifest not yet frozen",
            "scand": "no unused route+bag-disjoint H4 outer remains",
        },
        "primary_forbidden_features": config["feature_contract"][
            "forbidden_blocks"
        ],
        "cem_cal_accessed": False,
        "cem": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
