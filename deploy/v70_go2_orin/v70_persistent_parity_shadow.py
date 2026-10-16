#!/usr/bin/env python3
"""Compare persistent in-memory V70 scoring against staged shadow artifacts.

The script reuses the staged DINO float16 boundary and NoMaD samples so the
comparison isolates online-feature, LTWM, selector, and 3x16 CEM equivalence.
It initializes no camera, ROS publisher, SportClient, or motion executor.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEPLOY_DIR = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(DEPLOY_DIR)]

from v70_pose_features import PoseSample  # noqa: E402
from v70_persistent_planner import (  # noqa: E402
    PersistentV70Planner,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dino", required=True, type=Path)
    parser.add_argument("--nomad", required=True, type=Path)
    parser.add_argument("--online", required=True, type=Path)
    parser.add_argument("--selector", required=True, type=Path)
    parser.add_argument("--cem", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--atol",
        type=float,
        default=1.0e-7,
    )
    return parser.parse_args()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {
            key: np.asarray(payload[key]).copy()
            for key in payload.files
        }


def compare_array(
    name: str,
    actual: np.ndarray,
    expected: np.ndarray,
    *,
    atol: float,
) -> dict[str, Any]:
    left = np.asarray(actual)
    right = np.asarray(expected)
    report: dict[str, Any] = {
        "name": name,
        "actual_shape": list(left.shape),
        "expected_shape": list(right.shape),
        "actual_dtype": str(left.dtype),
        "expected_dtype": str(right.dtype),
        "exact": False,
        "within_tolerance": False,
        "max_abs_diff": None,
    }
    if left.shape != right.shape:
        return report
    report["exact"] = bool(np.array_equal(left, right))
    if left.dtype.kind in "biufc" and right.dtype.kind in "biufc":
        finite = np.isfinite(left) & np.isfinite(right)
        if finite.any():
            max_diff = float(
                np.max(
                    np.abs(
                        left[finite].astype(np.float64)
                        - right[finite].astype(np.float64)
                    )
                )
            )
        else:
            max_diff = 0.0
        same_nonfinite = np.array_equal(
            np.isposinf(left),
            np.isposinf(right),
        ) and np.array_equal(
            np.isneginf(left),
            np.isneginf(right),
        )
        report["max_abs_diff"] = max_diff
        report["within_tolerance"] = bool(
            same_nonfinite
            and np.allclose(
                left,
                right,
                rtol=0.0,
                atol=atol,
                equal_nan=True,
            )
        )
    else:
        report["within_tolerance"] = report["exact"]
    return report


def main() -> int:
    args = parse_args()
    if args.atol < 0.0:
        raise ValueError("--atol must be non-negative")
    print("PERSISTENT V70 PARITY SHADOW")
    print("No camera or ROS publisher is initialized.")
    print("No Unitree motion API is imported.")

    dino_staged = load_npz(args.dino)
    online_staged = load_npz(args.online)
    selector_staged = load_npz(args.selector)
    cem_staged = load_npz(args.cem)
    nomad_staged = np.load(args.nomad, allow_pickle=False)

    dino = {
        "history_latent": dino_staged[
            "history_latent"
        ].astype(np.float32),
        "goal_latent": dino_staged[
            "goal_latent"
        ].astype(np.float32),
        "static_features": dino_staged[
            "static_features"
        ].astype(np.float32),
    }
    pose_array = online_staged["pose_samples"].astype(np.float64)
    poses = [
        PoseSample(
            receive_time=float(row[0]),
            x=float(row[1]),
            y=float(row[2]),
            yaw=float(row[3]),
        )
        for row in pose_array
    ]

    planner = PersistentV70Planner()
    result = planner.score_and_plan(
        dino=dino,
        nomad_candidates=np.asarray(
            nomad_staged,
            dtype=np.float32,
        ),
        poses=poses,
    )

    mappings: list[
        tuple[str, np.ndarray, np.ndarray]
    ] = [
        (
            "online.candidates_waypoint_units",
            result["base_candidates"][0],
            online_staged["candidates_waypoint_units"],
        ),
        (
            "online.actions_h4",
            result["fixed"]["actions_h4"][0],
            online_staged["actions_h4"],
        ),
        (
            "online.action_features",
            result["fixed"]["action_features"][0],
            online_staged["action_features"],
        ),
        (
            "selector.universal_features",
            result["fixed"]["universal_features"],
            selector_staged["universal_features"],
        ),
        (
            "selector.energy",
            result["fixed"]["energy"],
            selector_staged["energy"],
        ),
        (
            "selector.safe",
            result["fixed"]["safe"],
            selector_staged["safe"],
        ),
        (
            "selector.valid",
            result["fixed"]["valid"],
            selector_staged["valid"],
        ),
        (
            "selector.mean",
            result["fixed"]["prediction_mean"],
            selector_staged["mean"],
        ),
        (
            "selector.std",
            result["fixed"]["prediction_std"],
            selector_staged["std"],
        ),
    ]
    for iteration, round_result in enumerate(result["rounds"]):
        evaluation = round_result["evaluation"]
        prefix = f"round_{iteration}"
        mappings.extend(
            [
                (
                    f"{prefix}.candidates",
                    round_result["candidates"],
                    cem_staged[f"{prefix}_candidates"],
                ),
                (
                    f"{prefix}.energy",
                    evaluation["energy"],
                    cem_staged[f"{prefix}_energy"],
                ),
                (
                    f"{prefix}.safe",
                    evaluation["safe"],
                    cem_staged[f"{prefix}_safe"],
                ),
                (
                    f"{prefix}.valid",
                    evaluation["valid"],
                    cem_staged[f"{prefix}_valid"],
                ),
                (
                    f"{prefix}.elite_index",
                    round_result["elite_index"],
                    cem_staged[f"{prefix}_elite_index"],
                ),
            ]
        )
    mappings.extend(
        [
            (
                "cem.selected_actions_h4",
                result["selected_actions_h4"],
                cem_staged["selected_actions_h4"],
            ),
            (
                "cem.selected_first_action",
                result["selected_first_action"],
                cem_staged["selected_first_action"],
            ),
            (
                "cem.archive_candidate",
                result["archive_candidate_waypoint_units"],
                cem_staged[
                    "archive_candidate_waypoint_units"
                ],
            ),
            (
                "cem.final_proposal_mean",
                result["final_proposal_mean"],
                cem_staged["final_proposal_mean"],
            ),
            (
                "cem.final_proposal_covariance",
                result["final_proposal_covariance"],
                cem_staged["final_proposal_covariance"],
            ),
        ]
    )

    comparisons = [
        compare_array(name, actual, expected, atol=args.atol)
        for name, actual, expected in mappings
    ]
    failures = [
        value
        for value in comparisons
        if not value["within_tolerance"]
    ]
    report = {
        "schema": "v70_persistent_parity_report_v1",
        "status": "PASS" if not failures else "FAIL",
        "atol": args.atol,
        "comparisons": comparisons,
        "failures": failures,
        "persistent_summary": planner.summary(
            {
                **result,
                "dino": {
                    "current_goal_nmse": float(
                        dino_staged[
                            "current_goal_nmse"
                        ].reshape(-1)[0]
                    ),
                    "current_goal_cosine_distance": float(
                        dino_staged[
                            "current_goal_cosine_distance"
                        ].reshape(-1)[0]
                    ),
                    "current_goal_norm_ratio": float(
                        dino_staged[
                            "current_goal_norm_ratio"
                        ].reshape(-1)[0]
                    ),
                    "encode_5_images_ms": 0.0,
                },
                "nomad": {
                    "temporal_distance": float("nan"),
                    "candidate_generation_ms": 0.0,
                },
                "planning_total_ms": (
                    result["fixed_pool_planning_ms"]
                    + result["cem_planning_ms"]
                ),
            }
        ),
    }
    # JSON does not carry the artificial NaN used only in this downstream
    # parity mode.
    report["persistent_summary"][
        "nomad_temporal_distance"
    ] = None

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    print(json.dumps(
        {
            "status": report["status"],
            "comparisons": len(comparisons),
            "failures": len(failures),
            "output": str(args.output),
        },
        indent=2,
    ))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
