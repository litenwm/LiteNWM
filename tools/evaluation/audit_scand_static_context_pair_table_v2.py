#!/usr/bin/env python3
"""Correct and strengthen the static-context table audit without rewriting it."""

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
OUTPUT = BASE / "spent_runtime_pair_static_context_table.audit.v2.json"
DEFINITION = ROOT / "tools/evaluation/export_recon_dense_dinov3_features_v1.py"
SEALED_MANIFEST = (
    ROOT
    / "outputs/cem_development_v1/scand_new_routefit10_v4_development_pilot_v2_strict_identity"
    / "sealed_validation_identity_clear.manifest.windows.jsonl"
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    require(not OUTPUT.exists(), f"refusing to overwrite: {OUTPUT}")
    require(TABLE.is_file() and OLD_AUDIT.is_file(), "missing table or original audit")
    with np.load(TABLE, allow_pickle=False) as payload:
        feature = payload["runtime_pair_feature"].copy()
        routes = payload["route_group"].astype(str)
        records = payload["record_key"].astype(str)
        cohorts = payload["cohort"].astype(str)
    require(feature.shape == (800, 65, 561), f"unexpected feature shape: {feature.shape}")
    static = feature[:, :, -210:]
    exact_candidate_independent = bool(
        np.array_equal(static, np.broadcast_to(static[:, :1], static.shape))
    )
    require(exact_candidate_independent, "210 static features are not exactly repeated")
    require(np.isfinite(static).all(), "static context contains nonfinite values")
    validation_routes = {"ahg__gdc", "library__union", "thompson__thompson"}
    overlap = sorted(validation_routes.intersection(set(routes)))
    require(not overlap, f"sealed validation leaked into table: {overlap}")
    audit = {
        "schema": "scand_static_context_pair_table_corrected_audit_v2",
        "status": "complete_adaptive_development_alignment_verified",
        "supersedes_original_audit_only": {
            "path": str(OLD_AUDIT),
            "sha256": sha256_file(OLD_AUDIT),
            "reason": (
                "The original audit used float32 std<1e-8 and falsely reported "
                "zero invariant dimensions despite exact equality."
            ),
            "table_rewritten": False,
        },
        "table": {
            "path": str(TABLE),
            "sha256": sha256_file(TABLE),
            "shape": list(feature.shape),
            "states": int(feature.shape[0]),
            "candidate_rows": int(feature.shape[0] * feature.shape[1]),
            "routes": int(len(np.unique(routes))),
            "records": int(len(np.unique(records))),
            "cohorts": sorted(np.unique(cohorts).tolist()),
        },
        "representation": {
            "base_dimensions": 351,
            "appended_static_dimensions": 210,
            "total_dimensions": 561,
            "candidate_independent_exact": exact_candidate_independent,
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
                "builder reconstructed every union candidate from fixed/round cache "
                "and required exact float32 equality before copying static_features"
            ),
            "original_alignment_sha256": json.loads(
                OLD_AUDIT.read_text(encoding="utf-8")
            )["candidate_record_static_alignment_sha256"],
        },
        "adaptive_development_disclosure": {
            "prior_42_route_outer_results_influenced_representation_choice": True,
            "same_outer_folds_are_unbiased_certification": False,
            "allowed_use": (
                "single adaptive development experiment; if all gates pass, freeze "
                "before one-time sealed validation"
            ),
        },
        "sealed_validation": {
            "manifest_path": str(SEALED_MANIFEST),
            "manifest_sha256": sha256_file(SEALED_MANIFEST),
            "expected_sha256": (
                "98aac829a1ffd571a91c8be6d3f6c0749e65e95a791110d30f4eaa89a9215b78"
            ),
            "routes": sorted(validation_routes),
            "route_overlap_with_table": overlap,
            "labels_or_outcomes_loaded": False,
        },
        "reserve_accessed": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    require(
        audit["sealed_validation"]["manifest_sha256"]
        == audit["sealed_validation"]["expected_sha256"],
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
                "candidate_independent_exact": exact_candidate_independent,
                "sealed_validation_route_overlap": overlap,
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
