#!/usr/bin/env python3
"""Idempotent final entrypoint for the route-fit SCAND download plan."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from planning.proposal_tail_sidecar_v1 import canonical_json  # noqa: E402
from tools.planning import download_prepare_new_scand_bags_v1 as v1  # noqa: E402
from tools.planning import download_prepare_new_scand_bags_v5 as v5  # noqa: E402


SCHEMA = "download_prepare_new_scand_bags_v6"


def freeze_corrected_plan(
    args: Any,
    selected: Sequence[Mapping[str, Any]],
    identity_audit: Mapping[str, Any],
    correction: Mapping[str, Any],
    path: Path,
) -> dict[str, Any]:
    plan = {
        "schema": SCHEMA + "_plan",
        "status": "frozen_before_network_and_labels",
        "selection_rule": (
            "one longest-duration public bag per route; seeded route subset; "
            "validation platform-stratified then seeded fill; route-disjoint; "
            "exclude pre-label observed route unable to supply 40 H4 windows; "
            "deterministically rebalance one validation route to development"
        ),
        "seed": args.seed,
        "identity_audit": dict(identity_audit),
        "contract": {
            "sample_rate_hz": v1.SAMPLE_RATE_HZ,
            "processed_resolution": [v1.PROCESSED_WIDTH, v1.PROCESSED_HEIGHT],
            "aspect_processing": "center crop to 4:3 then bicubic resize",
            "processed_encoding": "jpeg",
            "jpeg_quality": v1.JPEG_QUALITY,
            "history_frames": v1.HISTORY_FRAMES,
            "goal_offset_frames": v1.GOAL_OFFSET_FRAMES,
            "windows_per_route": args.windows_per_route,
            "odom_or_truth_opened": False,
            "labels_opened": False,
        },
        "requested": {
            "development_routes": args.development_routes,
            "one_time_validation_routes": args.validation_routes,
        },
        "final_scientific_target": {
            "development_routes": v1.FINAL_TARGET_DEVELOPMENT_ROUTES,
            "one_time_validation_routes": v1.FINAL_TARGET_VALIDATION_ROUTES,
            "this_public_download_alone_satisfies_target": False,
        },
        "selected": list(selected),
        "route_fit_correction": dict(correction),
        "network_started": False,
        "reserve_content_accessed": False,
    }
    v1.write_text_exact(path, canonical_json(plan) + "\n")
    return plan


def main() -> None:
    v5.install_transport()
    args = v1.parse_args()
    v1.reject_forbidden_path(args.incoming_root)
    v1.require(args.retries >= 1, "retries must be positive")
    v1.require(args.timeout_seconds > 0, "timeout must be positive")
    candidates, identity_audit, forbidden = v1.filtered_candidates(args)
    selected, correction = v5.corrected_selection(args, candidates)
    plan_path = (
        args.incoming_root
        / "quarantine/official_scand_selective_v1/download_plan.routefit_v3.json"
    )
    plan = freeze_corrected_plan(
        args,
        selected,
        identity_audit,
        correction,
        plan_path,
    )
    if not args.execute:
        print(
            canonical_json(
                {
                    "status": "routefit_plan_frozen_no_network",
                    "plan": str(plan_path),
                    "plan_sha256": v1.sha256_file(plan_path),
                    "selected_bags": len(selected),
                    "selected_routes": len({row["route_group"] for row in selected}),
                    "development_routes": args.development_routes,
                    "one_time_validation_routes": args.validation_routes,
                    "excluded_short_route": v5.EXCLUDED_ROUTE,
                    "execute_required_for_download": True,
                }
            )
        )
        return
    result = v1.execute(args, plan, forbidden, plan_path)
    print(canonical_json(result))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(
            canonical_json(
                {
                    "schema": SCHEMA + "_failure",
                    "status": "fail_closed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "network_may_have_started": "--execute" in sys.argv,
                    "reserve_content_accessed": False,
                }
            ),
            file=sys.stderr,
        )
        raise SystemExit(2) from error
