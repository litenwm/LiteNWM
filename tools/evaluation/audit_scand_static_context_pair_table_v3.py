#!/usr/bin/env python3
"""Final static-context audit with trajectory and route-group scopes separated."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from planning.proposal_tail_sidecar_v1 import canonical_json, sha256_file  # noqa: E402


BASE = ROOT / "outputs/cem_development_v1/incremental_outcome_student_v1/static_context_student_v1"
TABLE = BASE / "spent_runtime_pair_static_context_table.npz"
OLD_AUDIT = BASE / "spent_runtime_pair_static_context_table.audit.json"
OUTPUT = BASE / "spent_runtime_pair_static_context_table.audit.v3.json"
DEFINITION = ROOT / "tools/evaluation/export_recon_dense_dinov3_features_v1.py"
PILOT_PREREG = (
    ROOT
    / "outputs/cem_development_v1/scand_new_routefit10_v4_development_pilot_v2_strict_identity"
    / "preregistration.json"
)
SEALED_MANIFEST = PILOT_PREREG.parent / "sealed_validation_identity_clear.manifest.windows.jsonl"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    require(not OUTPUT.exists(), f"refusing to overwrite: {OUTPUT}")
    with np.load(TABLE, allow_pickle=False) as payload:
        feature = payload["runtime_pair_feature"].copy()
        routes = payload["route_group"].astype(str)
        trajectories = payload["trajectory_id"].astype(str)
        records = payload["record_key"].astype(str)
        cohorts = payload["cohort"].astype(str)
    require(feature.shape == (800, 65, 561), f"unexpected feature shape: {feature.shape}")
    static = feature[:, :, -210:]
    require(np.isfinite(static).all(), "static context contains nonfinite values")
    require(
        np.array_equal(static, np.broadcast_to(static[:, :1], static.shape)),
        "210 static features are not exactly candidate-independent",
    )
    pilot = json.loads(PILOT_PREREG.read_text(encoding="utf-8"))
    sealed_routes = set(pilot["sealed_validation"]["routes"])
    sealed_trajectories = set(pilot["sealed_validation"]["trajectories"])
    trajectory_overlap = sorted(sealed_trajectories.intersection(set(trajectories)))
    route_group_overlap = sorted(sealed_routes.intersection(set(routes)))
    require(not trajectory_overlap, f"sealed validation trajectory leaked: {trajectory_overlap}")
    original = json.loads(OLD_AUDIT.read_text(encoding="utf-8"))
    audit = {
        "schema": "scand_static_context_pair_table_final_audit_v3",
        "status": "complete_adaptive_development_alignment_verified",
        "table": {
            "path": str(TABLE),
            "sha256": sha256_file(TABLE),
            "shape": list(feature.shape),
            "states": 800,
            "candidate_rows": 52000,
            "routes": int(len(np.unique(routes))),
            "trajectories": int(len(np.unique(trajectories))),
            "records": int(len(np.unique(records))),
            "cohorts": sorted(np.unique(cohorts).tolist()),
        },
        "representation": {
            "base_dimensions": 351,
            "appended_static_dimensions": 210,
            "total_dimensions": 561,
            "candidate_independent_exact": True,
            "candidate_varying_static_dimensions": 0,
            "runtime_definition": "pool_static_features(history_latent, goal_latent)",
            "definition_source": {
                "path": str(DEFINITION),
                "sha256": sha256_file(DEFINITION),
            },
            "online_inputs_only": True,
            "label_route_record_truth_features": False,
        },
        "alignment": {
            "all_52000_record_candidate_rows_exactly_mapped": True,
            "mapping_evidence": (
                "builder reconstructed every fixed/round union candidate and required "
                "exact float32 action equality before copying cached static_features"
            ),
            "record_candidate_static_alignment_sha256": original[
                "candidate_record_static_alignment_sha256"
            ],
            "source_hashes": original["source_hashes"],
        },
        "superseded_original_audit": {
            "path": str(OLD_AUDIT),
            "sha256": sha256_file(OLD_AUDIT),
            "error": (
                "float32 std<1e-8 falsely reported zero invariant dimensions; "
                "exact equality proves all 210 dimensions are invariant across candidates"
            ),
            "table_rewritten": False,
        },
        "adaptive_development_disclosure": {
            "prior_42_route_outer_results_influenced_representation_choice": True,
            "same_outer_folds_are_unbiased_certification": False,
            "allowed_use": (
                "one adaptive development experiment; all gates must pass before an "
                "immutable freeze and one-time sealed validation"
            ),
        },
        "sealed_validation": {
            "manifest_path": str(SEALED_MANIFEST),
            "manifest_sha256": sha256_file(SEALED_MANIFEST),
            "expected_manifest_sha256": (
                "98aac829a1ffd571a91c8be6d3f6c0749e65e95a791110d30f4eaa89a9215b78"
            ),
            "trajectory_overlap_with_development": trajectory_overlap,
            "route_group_name_overlap_with_development": route_group_overlap,
            "scope_warning": (
                "The sealed set is trajectory/bag-disjoint but two route-group names "
                "were already present in spent development. A future result can support "
                "new-trajectory evidence, not wholly unseen-route-group generalization."
            ),
            "labels_or_outcomes_loaded": False,
        },
        "reserve_accessed": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    require(
        audit["sealed_validation"]["manifest_sha256"]
        == audit["sealed_validation"]["expected_manifest_sha256"],
        "sealed validation manifest hash changed",
    )
    OUTPUT.write_text(canonical_json(audit) + "\n", encoding="utf-8")
    print(
        canonical_json(
            {
                "status": audit["status"],
                "output": str(OUTPUT),
                "sha256": sha256_file(OUTPUT),
                "table_sha256": audit["table"]["sha256"],
                "candidate_independent_exact": True,
                "sealed_trajectory_overlap": trajectory_overlap,
                "sealed_route_group_name_overlap": route_group_overlap,
            }
        )
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(
            canonical_json(
                {
                    "status": "fail_closed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "sealed_validation_opened": False,
                }
            ),
            file=sys.stderr,
        )
        raise SystemExit(2) from error
