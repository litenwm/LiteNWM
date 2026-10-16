#!/usr/bin/env python3
"""Backfill sidecar state metadata from a frozen, label-free input manifest.

This tool only replaces ``states.jsonl`` and its binding in the sidecar
collection manifest. Candidate, score, RNG, raw-draw and formal-selection
artifacts are verified unchanged before and after the operation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from planning.proposal_tail_sidecar_v1 import (  # noqa: E402
    canonical_json,
    sha256_file,
    snapshot_states,
)


IMMUTABLE_FILES = (
    "candidate_table.npz",
    "formal_selection_snapshot.npz",
    "fixed_pool/candidates_and_scores.npz",
    "iteration_00/candidates_and_scores.npz",
    "iteration_00/raw_draws.npz",
    "iteration_01/candidates_and_scores.npz",
    "iteration_01/raw_draws.npz",
    "iteration_02/candidates_and_scores.npz",
    "iteration_02/raw_draws.npz",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_text_atomic(path: Path, text: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    states_path = args.collection / "states.jsonl"
    collection_manifest_path = args.collection / "collection_manifest.json"
    old_states = read_jsonl(states_path)
    source_rows = read_jsonl(args.source_manifest)
    source = {(str(row["trajectory_id"]), int(row["current_t"])): row for row in source_rows}
    if len(source) != len(source_rows):
        raise RuntimeError("duplicate source trajectory/current identity")

    immutable_before = {name: sha256_file(args.collection / name) for name in IMMUTABLE_FILES}
    enriched = []
    for old in old_states:
        key = (str(old["trajectory"]), int(old["current_frame_index"]))
        if key not in source:
            raise RuntimeError(f"missing source metadata for {key}")
        fresh = snapshot_states([source[key]])[0]
        for identity_key in (
            "record_key", "record_key_hash", "route", "route_hash", "day_hash",
            "bag", "bag_hash", "trajectory", "trajectory_hash", "window", "window_hash",
            "history_frame_indices", "current_frame_index", "goal_frame_index",
        ):
            if old[identity_key] != fresh[identity_key]:
                raise RuntimeError(f"identity mismatch for {key}: {identity_key}")
        if fresh["timestamps"]["history"]["value"] is None:
            raise RuntimeError(f"missing history timestamps for {key}")
        if fresh["timestamps"]["current"]["value"] is None or fresh["timestamps"]["goal"]["value"] is None:
            raise RuntimeError(f"missing current/goal timestamps for {key}")
        for camera_key in ("width", "height", "encoding", "file_paths", "file_sizes", "file_sha256"):
            if fresh["camera"][camera_key]["value"] is None:
                raise RuntimeError(f"missing camera {camera_key} for {key}")
        enriched.append(fresh)

    old_states_sha = sha256_file(states_path)
    write_text_atomic(states_path, "".join(canonical_json(row) + "\n" for row in enriched))
    new_states_sha = sha256_file(states_path)

    manifest = json.loads(collection_manifest_path.read_text(encoding="utf-8"))
    if manifest["files"].get("states.jsonl") != old_states_sha:
        raise RuntimeError("collection manifest did not bind original states.jsonl")
    manifest["files"]["states.jsonl"] = new_states_sha
    manifest["metadata_backfill"] = {
        "schema": "proposal_tail_sidecar_metadata_backfill_v1",
        "label_free": True,
        "source_manifest": str(args.source_manifest.resolve()),
        "source_manifest_sha256": sha256_file(args.source_manifest),
        "states_before_sha256": old_states_sha,
        "states_after_sha256": new_states_sha,
        "matched_states": len(enriched),
        "formal_behavior_influence": False,
    }
    write_text_atomic(collection_manifest_path, canonical_json(manifest) + "\n")

    immutable_after = {name: sha256_file(args.collection / name) for name in IMMUTABLE_FILES}
    if immutable_before != immutable_after:
        raise RuntimeError("immutable planner artifact changed during metadata backfill")
    audit = {
        "schema": "proposal_tail_sidecar_metadata_backfill_audit_v1",
        "status": "pass",
        "collection": str(args.collection.resolve()),
        "source_manifest": str(args.source_manifest.resolve()),
        "source_manifest_sha256": sha256_file(args.source_manifest),
        "states": len(enriched),
        "states_before_sha256": old_states_sha,
        "states_after_sha256": new_states_sha,
        "collection_manifest_sha256": sha256_file(collection_manifest_path),
        "immutable_artifact_sha256": immutable_after,
        "labels_accessed": False,
        "formal_behavior_influence": False,
    }
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    write_text_atomic(args.audit_output, canonical_json(audit) + "\n")
    print(canonical_json(audit))


if __name__ == "__main__":
    main()
