#!/usr/bin/env python3
"""Fail-closed audit for proposal and optional ATE/RTP label artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from scoring.trajectory_artifact_contract_v2 import (  # noqa: E402
    file_sha256,
    load_inner_manifest,
    INNER_PARTITIONS,
    load_npz,
    validate_outer_train_membership,
    validate_label_artifact,
    validate_proposal_artifact,
    write_json_atomic,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("recon", "scand"), required=True)
    parser.add_argument("--partition", choices=INNER_PARTITIONS, required=True)
    parser.add_argument("--inner-manifest", type=Path, required=True)
    parser.add_argument("--proposals", type=Path, required=True)
    parser.add_argument("--labels", type=Path, default=None)
    parser.add_argument("--split-lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_inner_manifest(
        args.inner_manifest,
        dataset=args.dataset,
        partition=args.partition,
    )
    validate_outer_train_membership(args.split_lock, rows, dataset=args.dataset)
    split_lock_sha = file_sha256(args.split_lock)
    proposal = load_npz(args.proposals)
    proposal_metadata = validate_proposal_artifact(
        proposal,
        rows,
        dataset=args.dataset,
        partition=args.partition,
        split_lock_sha256=split_lock_sha,
    )
    manifest_sha = file_sha256(args.inner_manifest)
    if proposal_metadata.get("inner_manifest_sha256") != manifest_sha:
        raise RuntimeError("proposal inner-manifest hash mismatch")
    label_metadata = None
    if args.labels is not None:
        labels = load_npz(args.labels)
        label_metadata = validate_label_artifact(
            labels,
            proposal,
            dataset=args.dataset,
            partition=args.partition,
            split_lock_sha256=split_lock_sha,
        )
        if label_metadata.get("inner_manifest_sha256") != manifest_sha:
            raise RuntimeError("label inner-manifest hash mismatch")
        if label_metadata.get("proposal_sha256") != file_sha256(args.proposals):
            raise RuntimeError("label proposal hash mismatch")
    report = {
        "schema": "trajectory_artifact_audit_v2",
        "status": "PASS",
        "dataset": args.dataset,
        "partition": args.partition,
        "windows": len(rows),
        "candidates": int(proposal["actions_h4"].shape[1]),
        "split_lock_sha256": split_lock_sha,
        "inner_manifest_sha256": manifest_sha,
        "proposal_sha256": file_sha256(args.proposals),
        "proposal_metadata": proposal_metadata,
        "labels_present": args.labels is not None,
        "labels_sha256": file_sha256(args.labels) if args.labels is not None else None,
        "label_metadata": label_metadata,
    }
    if args.output is not None:
        write_json_atomic(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
