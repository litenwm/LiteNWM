#!/usr/bin/env python3
"""Build strict expanded-SCAND inner splits and RECON H4 exclusions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from scoring.strict_inner_partitions import (  # noqa: E402
    build_recon_h4_exclusion_manifest,
    build_recon_inner_manifests,
    build_scand_inner_manifests,
    validate_recon_inner_manifests,
    validate_scand_inner_manifests,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    scand = commands.add_parser(
        "scand-manifests",
        help="Build route-group/bag-disjoint four-way outer-train manifests.",
    )
    scand.add_argument("--latent-root", type=Path, required=True)
    scand.add_argument("--split-lock", type=Path, required=True)
    scand.add_argument("--out-root", type=Path, required=True)
    scand.add_argument("--split-seed", type=int, default=20260719)
    scand.add_argument("--train-fit-fraction", type=float, default=0.70)
    scand.add_argument("--scorer-cal-fraction", type=float, default=0.10)
    scand.add_argument("--cem-cal-fraction", type=float, default=0.10)
    scand.add_argument("--internal-holdout-fraction", type=float, default=0.10)
    scand.add_argument("--max-windows-per-bag", type=int, default=256)
    scand.add_argument("--min-goal-displacement-m", type=float, default=0.5)

    recon = commands.add_parser(
        "recon-exclusions",
        help="Audit deterministic 4-history/H4 short-trajectory exclusions.",
    )
    recon.add_argument("--latent-root", type=Path, required=True)
    recon.add_argument("--split-lock", type=Path, required=True)
    recon.add_argument("--out-file", type=Path, required=True)
    recon.add_argument("--history", type=int, default=4)
    recon.add_argument("--horizon", type=int, default=4)

    recon_manifests = commands.add_parser(
        "recon-manifests",
        help="Build route-disjoint four-way RECON outer-train manifests.",
    )
    recon_manifests.add_argument("--latent-root", type=Path, required=True)
    recon_manifests.add_argument("--split-lock", type=Path, required=True)
    recon_manifests.add_argument(
        "--exclusion-manifest", type=Path, required=True
    )
    recon_manifests.add_argument("--out-root", type=Path, required=True)
    recon_manifests.add_argument("--split-seed", type=int, default=20260719)
    recon_manifests.add_argument("--train-fit-fraction", type=float, default=0.70)
    recon_manifests.add_argument("--scorer-cal-fraction", type=float, default=0.10)
    recon_manifests.add_argument("--cem-cal-fraction", type=float, default=0.10)
    recon_manifests.add_argument(
        "--internal-holdout-fraction", type=float, default=0.10
    )
    recon_manifests.add_argument("--train-fit-quota-per-route", type=int, default=3)
    recon_manifests.add_argument("--other-quota-per-route", type=int, default=2)
    recon_manifests.add_argument(
        "--min-goal-displacement-m", type=float, default=0.5
    )

    validate = commands.add_parser(
        "validate-scand",
        help="Independently validate a built four-way SCAND manifest root.",
    )
    validate.add_argument("--asset-root", type=Path, required=True)
    validate.add_argument("--split-lock", type=Path, required=True)
    validate.add_argument("--out-file", type=Path, default=None)

    validate_recon = commands.add_parser(
        "validate-recon",
        help="Independently validate a built four-way RECON manifest root.",
    )
    validate_recon.add_argument("--asset-root", type=Path, required=True)
    validate_recon.add_argument("--split-lock", type=Path, required=True)
    validate_recon.add_argument("--out-file", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "scand-manifests":
        report = build_scand_inner_manifests(
            latent_root=args.latent_root,
            split_lock_path=args.split_lock,
            out_root=args.out_root,
            split_seed=args.split_seed,
            fractions={
                "train-fit": args.train_fit_fraction,
                "scorer-cal": args.scorer_cal_fraction,
                "cem-cal": args.cem_cal_fraction,
                "internal-holdout": args.internal_holdout_fraction,
            },
            max_windows_per_bag=args.max_windows_per_bag,
            min_goal_displacement_m=args.min_goal_displacement_m,
        )
    elif args.command == "recon-exclusions":
        report = build_recon_h4_exclusion_manifest(
            latent_root=args.latent_root,
            split_lock_path=args.split_lock,
            out_file=args.out_file,
            history=args.history,
            horizon=args.horizon,
        )
    elif args.command == "recon-manifests":
        report = build_recon_inner_manifests(
            latent_root=args.latent_root,
            split_lock_path=args.split_lock,
            exclusion_manifest_path=args.exclusion_manifest,
            out_root=args.out_root,
            split_seed=args.split_seed,
            fractions={
                "train-fit": args.train_fit_fraction,
                "scorer-cal": args.scorer_cal_fraction,
                "cem-cal": args.cem_cal_fraction,
                "internal-holdout": args.internal_holdout_fraction,
            },
            quota_per_route={
                "train-fit": args.train_fit_quota_per_route,
                "scorer-cal": args.other_quota_per_route,
                "cem-cal": args.other_quota_per_route,
                "internal-holdout": args.other_quota_per_route,
            },
            min_goal_displacement_m=args.min_goal_displacement_m,
        )
    elif args.command == "validate-scand":
        report = validate_scand_inner_manifests(
            asset_root=args.asset_root,
            split_lock_path=args.split_lock,
            out_file=args.out_file,
        )
    elif args.command == "validate-recon":
        report = validate_recon_inner_manifests(
            asset_root=args.asset_root,
            split_lock_path=args.split_lock,
            out_file=args.out_file,
        )
    else:  # pragma: no cover
        raise AssertionError(args.command)
    printable = report
    if args.command == "recon-exclusions":
        printable = {
            key: report[key]
            for key in (
                "decision",
                "dataset",
                "schema",
                "protocol",
                "splits",
                "exclusion_count",
                "exclusion_reasons",
                "leakage_guard",
                "split_lock_sha256",
            )
        }
        printable["exclusions_written_to"] = str(args.out_file)
    print(json.dumps(printable, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
