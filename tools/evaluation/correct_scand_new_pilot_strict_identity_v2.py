#!/usr/bin/env python3
"""Apply the frozen strict100 identity exclusion to the new SCAND pilot."""

from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from planning.proposal_tail_sidecar_v1 import canonical_json, sha256_file  # noqa: E402


SOURCE = ROOT / "outputs/cem_development_v1/scand_new_routefit10_v4_development_pilot_v1"
VALIDATION_SOURCE = ROOT / "data/proposal_tail_collection_v1/incoming/one_time_validation_routefit10_v4/manifest.windows.jsonl"
STRICT = ROOT / "data/evaluation/scand/strict_test_h4_100_seed17/manifest.json"
OUTPUT = ROOT / "outputs/cem_development_v1/scand_new_routefit10_v4_development_pilot_v2_strict_identity"


def write_new(path: Path, content: str) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    spent = {row["trajectory_id"] for row in json.loads(STRICT.read_text(encoding="utf-8"))["records"]}
    dev_all = [json.loads(line) for line in (SOURCE / "pilot_manifest.windows.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    val_all = [json.loads(line) for line in VALIDATION_SOURCE.read_text(encoding="utf-8").splitlines() if line.strip()]
    dev_overlap = sorted({row["trajectory_id"] for row in dev_all} & spent)
    val_overlap = sorted({row["trajectory_id"] for row in val_all} & spent)
    dev = [row for row in dev_all if row["trajectory_id"] not in spent]
    val = [row for row in val_all if row["trajectory_id"] not in spent]
    if (len(dev), len({row["route_group"] for row in dev}), len(val), len({row["route_group"] for row in val})) != (160, 4, 120, 3):
        raise RuntimeError("strict identity correction cardinality changed")
    view = OUTPUT / "trajectory_view"
    view.mkdir(parents=True)
    for trajectory in sorted({row["trajectory_id"] for row in dev}):
        (view / trajectory).symlink_to((SOURCE / "trajectory_view" / trajectory).resolve(), target_is_directory=True)
    dev_path = OUTPUT / "pilot_manifest.windows.jsonl"
    val_path = OUTPUT / "sealed_validation_identity_clear.manifest.windows.jsonl"
    write_new(dev_path, "".join(canonical_json(row) + "\n" for row in dev))
    write_new(val_path, "".join(canonical_json(row) + "\n" for row in val))
    prereg = {
        "schema": "scand_new_routefit10_v4_strict_identity_preregistration_v2",
        "status": "frozen_before_candidate_generation_and_outcomes",
        "scope": "four-route feasibility diagnostic only; statistically underpowered; not independent validation",
        "identity_rule": "exclude any trajectory present in spent strict100, regardless of v70-v77 registry status",
        "inputs": {
            "source_preregistration": {"path": str(SOURCE / "preregistration.json"), "sha256": sha256_file(SOURCE / "preregistration.json")},
            "strict100_identity_only": {"path": str(STRICT), "sha256": sha256_file(STRICT)},
            "manifest": {"path": str(dev_path), "sha256": sha256_file(dev_path)},
            "sealed_validation_manifest": {"path": str(val_path), "sha256": sha256_file(val_path), "labels_opened": False},
        },
        "excluded_development_trajectories": dev_overlap,
        "excluded_validation_trajectories": val_overlap,
        "development": {"states": len(dev), "routes": sorted({row["route_group"] for row in dev}), "trajectories": sorted({row["trajectory_id"] for row in dev})},
        "sealed_validation": {"states": len(val), "routes": sorted({row["route_group"] for row in val}), "trajectories": sorted({row["trajectory_id"] for row in val}), "labels_opened": False},
        "staged_gate": {
            "first": "measure physical+support proposal oracle; stop if no cross-route opportunity",
            "second": "leave-one-route-out candidate risk/ranking probe only if oracle passes",
            "candidate_dual_and_harm_auroc_min": 0.75,
            "risk_certified_replacements_nonzero": True,
            "dual_wilson_lcb_min": 0.85,
            "harm_wilson_ucb_max": 0.15,
        },
        "formal_v70_unchanged": True,
        "labels_or_pose_opened": False,
        "reserve_accessed": False,
    }
    prereg_path = OUTPUT / "preregistration.json"
    write_new(prereg_path, json.dumps(prereg, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    audit = {
        "schema": "scand_new_routefit10_v4_strict_identity_audit_v2",
        "status": "complete_label_free",
        "development_states": len(dev),
        "development_routes": len({row["route_group"] for row in dev}),
        "sealed_validation_states": len(val),
        "sealed_validation_routes": len({row["route_group"] for row in val}),
        "development_overlap_removed": dev_overlap,
        "validation_overlap_removed": val_overlap,
        "manifest_sha256": sha256_file(dev_path),
        "sealed_validation_manifest_sha256": sha256_file(val_path),
        "preregistration_sha256": sha256_file(prereg_path),
        "labels_or_outcomes_opened": False,
        "reserve_accessed": False,
    }
    audit_path = OUTPUT / "identity_correction.audit.json"
    write_new(audit_path, json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({**audit, "audit_sha256": sha256_file(audit_path)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
