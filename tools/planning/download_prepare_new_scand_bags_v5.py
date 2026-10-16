#!/usr/bin/env python3
"""Route-fit SCAND ingest after excluding one provably too-short bag.

The original frozen 18-route plan included the 9.6-second gdc__gdc bag.  Its
successfully downloaded camera stream contains 36 frames at the frozen 4 Hz,
which permits only 29 H4 windows; the contract requires 40 per route.  This
pre-label correction excludes that route rather than lowering the window
contract, then deterministically rebalances the remaining 17 routes to 12
development and 5 one-time validation routes.

All v4 transport, official TDR size/MD5, identity, no-label, and camera
processing safeguards remain unchanged.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from planning.proposal_tail_sidecar_v1 import canonical_json  # noqa: E402
from tools.planning import download_prepare_new_scand_bags_v1 as v1  # noqa: E402
from tools.planning import download_prepare_new_scand_bags_v2 as v2  # noqa: E402
from tools.planning import download_prepare_new_scand_bags_v3 as v3  # noqa: E402
from tools.planning import download_prepare_new_scand_bags_v4 as v4  # noqa: E402


SCHEMA = "download_prepare_new_scand_bags_v5"
EXCLUDED_FILE = "A_Jackal_GDC_GDC_Fri_Oct_29_11"
EXCLUDED_ROUTE = "gdc__gdc"
OBSERVED_FRAMES = 36
OBSERVED_ELIGIBLE_H4_WINDOWS = 29
REQUIRED_WINDOWS = 40
CORRECTED_DEVELOPMENT_ROUTES = 12
CORRECTED_VALIDATION_ROUTES = 5


def corrected_selection(
    args: Any,
    candidates: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    v2.require(args.development_routes == 12, "v5 requires the frozen pre-correction development count 12")
    v2.require(args.validation_routes == 6, "v5 requires the frozen pre-correction validation count 6")
    v2.require(args.windows_per_route == REQUIRED_WINDOWS, "v5 does not allow changing windows per route")
    original = v1.select_plan(args, candidates)
    excluded = [row for row in original if row["official_file_name"] == EXCLUDED_FILE]
    v2.require(len(excluded) == 1, "short-route exclusion did not resolve uniquely")
    v2.require(excluded[0]["route_group"] == EXCLUDED_ROUTE, "short-route identity changed")
    v2.require(excluded[0]["cohort_role"] == "development", "short route was not in development")
    selected = [dict(row) for row in original if row["official_file_name"] != EXCLUDED_FILE]
    validation = [row for row in selected if row["cohort_role"] == "one_time_validation"]
    v2.require(len(validation) == 6, "unexpected validation count before rebalance")
    moved = min(
        validation,
        key=lambda row: v1.stable_key(args.seed, "routefit-rebalance", row["route_group"]),
    )
    moved["cohort_role"] = "development"
    args.development_routes = CORRECTED_DEVELOPMENT_ROUTES
    args.validation_routes = CORRECTED_VALIDATION_ROUTES
    v2.require(
        sum(row["cohort_role"] == "development" for row in selected)
        == CORRECTED_DEVELOPMENT_ROUTES,
        "corrected development count mismatch",
    )
    v2.require(
        sum(row["cohort_role"] == "one_time_validation" for row in selected)
        == CORRECTED_VALIDATION_ROUTES,
        "corrected validation count mismatch",
    )
    v2.require(
        not (
            {row["route_group"] for row in selected if row["cohort_role"] == "development"}
            & {
                row["route_group"]
                for row in selected
                if row["cohort_role"] == "one_time_validation"
            }
        ),
        "corrected cohorts overlap by route",
    )
    correction = {
        "reason": "actual 4 Hz camera stream cannot satisfy frozen 40-window H4 contract",
        "excluded_official_file_name": EXCLUDED_FILE,
        "excluded_route_group": EXCLUDED_ROUTE,
        "public_duration_seconds": excluded[0]["duration_seconds"],
        "observed_frames": OBSERVED_FRAMES,
        "observed_eligible_h4_windows": OBSERVED_ELIGIBLE_H4_WINDOWS,
        "required_windows": REQUIRED_WINDOWS,
        "excluded_bag_and_partial_frames_retained_for_audit": True,
        "labels_opened": False,
        "moved_validation_route_to_development": moved["route_group"],
        "corrected_development_routes": CORRECTED_DEVELOPMENT_ROUTES,
        "corrected_one_time_validation_routes": CORRECTED_VALIDATION_ROUTES,
    }
    return selected, correction


def install_transport() -> None:
    v2.fetch_tdr_catalog = v3.fetch_tdr_catalog_from_snapshot
    v2._ORIGINAL_DOWNLOAD_ONE = v4.windows_curl_download_one
    v2.install_patches()


def main() -> None:
    install_transport()
    args = v1.parse_args()
    v1.reject_forbidden_path(args.incoming_root)
    v1.require(args.retries >= 1, "retries must be positive")
    v1.require(args.timeout_seconds > 0, "timeout must be positive")
    candidates, identity_audit, forbidden = v1.filtered_candidates(args)
    selected, correction = corrected_selection(args, candidates)
    plan_path = (
        args.incoming_root
        / "quarantine/official_scand_selective_v1/download_plan.routefit_v2.json"
    )
    plan = v1.freeze_plan(args, selected, identity_audit, plan_path)
    plan.update(
        {
            "schema": SCHEMA + "_plan",
            "selection_rule": (
                plan["selection_rule"]
                + "; exclude pre-label observed route unable to supply 40 H4 windows; "
                + "deterministically rebalance one validation route to development"
            ),
            "route_fit_correction": correction,
        }
    )
    v1.write_json_atomic(plan_path, plan)
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
                    "excluded_short_route": EXCLUDED_ROUTE,
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
