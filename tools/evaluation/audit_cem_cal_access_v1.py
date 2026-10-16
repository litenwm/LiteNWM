#!/usr/bin/env python3
"""Audit reserved RECON/SCAND cem-cal identity and historical artifact use."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tools.scoring.train_recon_dense_meta_dinov3_v1 import (  # noqa: E402
    file_sha256,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT
        / "outputs/cem_development_v1/preflight/cem_cal_access_audit.json",
    )
    return parser.parse_args()


def rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def identity(values: list[dict[str, Any]]) -> dict[str, set[str]]:
    return {
        "record": {
            f"{value['trajectory_id']}:{int(value['current_t'])}"
            for value in values
        },
        "trajectory": {str(value["trajectory_id"]) for value in values},
        "route": {
            str(value.get("route_group", value.get("route_id")))
            for value in values
        },
        "bag": {
            str(value["bag_id"])
            for value in values
            if value.get("bag_id") is not None
        },
    }


def split_manifest(domain: str, split: str) -> Path:
    path = REPO_ROOT / f"data/inner_manifests/{domain}/{split}.windows.jsonl"
    if (
        domain == "scand"
        and split == "scorer-cal"
        and not path.exists()
    ):
        path = (
            REPO_ROOT
            / "outputs/scand_domain_adaptation_v1/manifests/"
            "scorer_cal_rgb_available.windows.jsonl"
        )
    return path


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    domains = {}
    cem_record_by_domain = {}
    for domain in ("recon", "scand"):
        manifests = {
            split: split_manifest(domain, split)
            for split in ("train-fit", "scorer-cal", "cem-cal")
        }
        values = {
            split: rows(path) for split, path in manifests.items()
        }
        ids = {split: identity(value) for split, value in values.items()}
        cem = ids["cem-cal"]
        cem_record_by_domain[domain] = cem["record"]
        domains[domain] = {
            "cem_cal": {
                "manifest": str(manifests["cem-cal"]),
                "manifest_sha256": file_sha256(manifests["cem-cal"]),
                "windows": len(values["cem-cal"]),
                "trajectories": len(cem["trajectory"]),
                "route_groups": len(cem["route"]),
                "bags": len(cem["bag"]),
            },
            "split_overlap": {
                split: {
                    key: len(ids[split][key] & cem[key])
                    for key in ("record", "trajectory", "route", "bag")
                }
                for split in ("train-fit", "scorer-cal")
            },
        }

    matches = {"recon": [], "scand": []}
    failures = []
    scanned = 0
    for root in (REPO_ROOT / "outputs", REPO_ROOT / "data/evaluation"):
        for path in sorted(root.rglob("*.npz")):
            scanned += 1
            try:
                with np.load(path, allow_pickle=False) as payload:
                    if "record_key" not in payload.files:
                        continue
                    observed = set(payload["record_key"].astype(str))
            except Exception as exc:
                failures.append({"path": str(path), "error": str(exc)})
                continue
            for domain in ("recon", "scand"):
                overlap = observed & cem_record_by_domain[domain]
                if overlap:
                    matches[domain].append(
                        {
                            "path": str(path),
                            "sha256": file_sha256(path),
                            "record_overlap": len(overlap),
                        }
                    )
    for domain in ("recon", "scand"):
        domains[domain]["historical_npz_identity_matches"] = matches[domain]
        domains[domain]["historically_accessed"] = bool(matches[domain])
        split_clean = all(
            not any(value.values())
            for value in domains[domain]["split_overlap"].values()
        )
        domains[domain]["legal_for_cem_development"] = bool(split_clean)

    report = {
        "schema": "dual_domain_cem_cal_access_audit_v1",
        "status": "identity_only_preflight_complete",
        "domains": domains,
        "npz_files_scanned": scanned,
        "scan_failures": failures,
        "outcomes_opened_by_this_audit": False,
        "identity_fields_only": True,
        "cem": False,
    }
    write_json(args.output, report)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "output_sha256": file_sha256(args.output),
                **report,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
