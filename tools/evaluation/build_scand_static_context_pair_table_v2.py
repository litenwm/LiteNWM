#!/usr/bin/env python3
"""Build the frozen 351+210 static-context development table.

This supersedes the unexecuted v1 draft, which proposed a 420-dimensional
delta/context expansion.  The frozen experiment appends exactly the existing
210-dimensional, candidate-independent runtime ``static_features`` block.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import sys

import numpy as np

from tools.evaluation import build_scand_static_context_pair_table_v1 as base


def main() -> None:
    base.require(
        not base.OUTPUT.exists() and not base.AUDIT.exists(),
        "refusing to overwrite static-context table",
    )
    source = base.load_npz(base.SOURCE)
    legacy, legacy_hash = base.sidecar_static(base.LEGACY)
    fresh, fresh_hash = base.fresh_static()
    new, new_hash = base.sidecar_static(base.NEW)
    static = np.concatenate((legacy, fresh, new), axis=0)
    base.require(static.shape == (800, 65, 210), f"static shape mismatch: {static.shape}")
    formal = source["formal_v70_index"].astype(np.int64)
    selected = static[np.arange(len(formal)), formal]
    base.require(
        np.array_equal(static, np.broadcast_to(selected[:, None], static.shape)),
        "static feature must be candidate-independent",
    )
    feature = np.concatenate(
        (source["runtime_pair_feature"].astype(np.float32), static), axis=-1
    )
    base.require(
        feature.shape == (800, 65, 561),
        f"extended feature shape mismatch: {feature.shape}",
    )
    base.require(np.isfinite(feature).all(), "extended feature contains NaN/Inf")
    names = np.concatenate(
        (
            source["feature_names"].astype(str),
            np.asarray([f"static_dinov3_context_{i}" for i in range(210)]),
        )
    )
    arrays = {
        key: value
        for key, value in source.items()
        if key not in {"runtime_pair_feature", "feature_names"}
    }
    arrays["runtime_pair_feature"] = feature
    arrays["feature_names"] = names
    base.OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with base.OUTPUT.open("wb") as handle:
        np.savez_compressed(handle, **arrays)

    candidate_std = static.std(axis=1)
    invariant = np.median(candidate_std, axis=0) < 1.0e-8
    alignment = hashlib.sha256()
    for state, record_key in enumerate(source["record_key"].astype(str)):
        for candidate in range(65):
            alignment.update(record_key.encode("utf-8"))
            alignment.update(candidate.to_bytes(2, "little", signed=False))
            alignment.update(
                np.ascontiguousarray(
                    source["candidates_feature_waypoint_units"][state, candidate],
                    dtype=np.float32,
                ).tobytes()
            )
            alignment.update(
                np.ascontiguousarray(static[state, candidate], dtype=np.float32).tobytes()
            )
            alignment.update(
                int(source["source_round"][state, candidate]).to_bytes(
                    2, "little", signed=True
                )
            )
            alignment.update(
                int(source["source_candidate_index"][state, candidate]).to_bytes(
                    2, "little", signed=True
                )
            )

    route = source["route_group"].astype(str)
    formal_static = selected.astype(np.float64)
    total_var = formal_static.var(axis=0)
    within = np.zeros(210, dtype=np.float64)
    for name in np.unique(route):
        local = route == name
        within += local.mean() * formal_static[local].var(axis=0)
    eta = np.where(
        total_var > 1.0e-12,
        np.clip(1.0 - within / total_var, 0.0, 1.0),
        0.0,
    )
    audit = {
        "schema": "scand_static_context_pair_table_audit_v2",
        "status": "complete_adaptive_spent_development_representation",
        "states": 800,
        "routes": int(len(np.unique(route))),
        "candidates_per_state": 65,
        "candidate_rows": 52000,
        "base_feature_dim": 351,
        "static_context_dim": 210,
        "static_context_added_dim": 210,
        "output_feature_dim": 561,
        "static_candidate_invariant_dimensions": int(invariant.sum()),
        "static_candidate_varying_dimensions": int((~invariant).sum()),
        "formal_static_route_eta_squared": {
            "median": float(np.median(eta)),
            "p90": float(np.quantile(eta, 0.9)),
            "max": float(eta.max()),
        },
        "candidate_mapping": (
            "exact float32 candidate union/order verified for all 52,000 "
            "fixed+three-round candidate rows"
        ),
        "candidate_record_static_alignment_sha256": alignment.hexdigest(),
        "runtime_availability": {
            "definition": (
                "pool_static_features(history_latent, goal_latent); no labels, "
                "route IDs, record IDs, or true outcomes"
            ),
            "candidate_independent": bool(invariant.all()),
            "all_52000_rows_finite": True,
        },
        "adaptive_selection_disclosure": {
            "prior_outer_results_influenced_representation_family": True,
            "certification_status": "adaptive_development_only",
            "interpretation": (
                "The same outer folds are not an unbiased generalization estimate. "
                "Only a future one-time sealed validation can provide independent evidence."
            ),
        },
        "source_hashes": {
            "legacy": legacy_hash,
            "fresh": fresh_hash,
            "new": new_hash,
            "base_table": base.sha256_file(base.SOURCE),
            "builder_v2": base.sha256_file(base.Path(__file__).resolve()),
        },
        "output": {
            "path": str(base.OUTPUT),
            "sha256": base.sha256_file(base.OUTPUT),
        },
        "dwm_ltwm_scorer_rerun": False,
        "targets_or_thresholds_changed": False,
        "sealed_validation_included": False,
        "reserve_accessed": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    base.AUDIT.write_text(base.canonical_json(audit) + "\n", encoding="utf-8")
    print(
        base.canonical_json(
            {
                "status": audit["status"],
                "output": str(base.OUTPUT),
                "sha256": audit["output"]["sha256"],
                "feature_dim": 561,
                "invariant_static_dims": audit[
                    "static_candidate_invariant_dimensions"
                ],
                "alignment_sha256": audit[
                    "candidate_record_static_alignment_sha256"
                ],
                "route_eta2": audit["formal_static_route_eta_squared"],
            }
        )
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(
            base.canonical_json(
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
