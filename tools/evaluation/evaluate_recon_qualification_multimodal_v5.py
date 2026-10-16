#!/usr/bin/env python3
"""Evaluate frozen multimodal-v5 on qualification_v4 and cumulative evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from tools.evaluation.evaluate_recon_fresh_outer_universal_v2 import (  # noqa: E402
    deployment_checks,
)
from tools.evaluation.evaluate_scand_fresh_outer_v1 import method_metrics  # noqa: E402
from tools.scoring.train_recon_dense_meta_dinov3_three_arm_v1 import (  # noqa: E402
    load_features,
)
from tools.scoring.train_recon_dense_meta_dinov3_v1 import (  # noqa: E402
    file_sha256,
    write_json,
)
from tools.scoring.train_universal_dualdomain_raw_v2 import (  # noqa: E402
    safety_metrics,
)


ROOT = REPO_ROOT / "outputs/universal_dualdomain_raw_v2"
OUTER = ROOT / "recon_fresh_outer_qualification_v4"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--freeze",
        type=Path,
        default=OUTER / "prelabel_multimodal_v5/prelabel_freeze.json",
    )
    parser.add_argument(
        "--selected",
        type=Path,
        default=OUTER / "prelabel_multimodal_v5/selected_index.npz",
    )
    parser.add_argument(
        "--baseline-result",
        type=Path,
        default=OUTER / "baseline_result.json",
    )
    parser.add_argument(
        "--calibration-result",
        type=Path,
        default=ROOT / "recon_route_multimodal_v5/result.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTER / "multimodal_v5_qualification_result.json",
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260725)
    return parser.parse_args()


def load_development_targets(
    calibration: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    audit = calibration["data_and_leakage_audit"]
    scorer = load_features(
        Path(audit["scorer_cal"]["features"]), labels_expected=True
    )
    target_parts = [scorer["targets"][..., :4].astype(np.float32)]
    for name in ("spent_outer_1", "spent_outer_2"):
        with np.load(audit[name]["outcomes"], allow_pickle=False) as payload:
            absolute = payload["absolute_metrics"].astype(np.float32)
        target_parts.append(absolute - absolute[:, :1])
    targets = np.concatenate(target_parts)
    with np.load(
        calibration["calibration_predictions"], allow_pickle=False
    ) as payload:
        selected = payload["deployment_oof_selected"].astype(np.int64)
        cohort = payload["cohort"].astype(str)
    if selected.shape != (610,) or targets.shape != (610, 17, 4):
        raise RuntimeError("v5 development evidence shape changed")
    return selected, targets, cohort


def safety_pass(metrics: dict[str, Any]) -> bool:
    return bool(
        metrics["coverage"] >= 0.40
        and metrics["dual_precision"] is not None
        and metrics["dual_precision"] >= 0.90
        and metrics["dual_wilson_lower_95"] is not None
        and metrics["dual_wilson_lower_95"] >= 0.80
        and metrics["harm_rate"] is not None
        and metrics["harm_rate"] <= 0.10
    )


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    freeze = json.loads(args.freeze.read_text(encoding="utf-8"))
    if (
        freeze.get("schema")
        != "recon_multimodal_v5_qualification_v4_prelabel_freeze"
        or freeze.get("status")
        != "selected_index_frozen_before_future_pose_labels"
        or freeze.get("labels_opened")
        or freeze.get("future_pose_outcomes_opened")
        or freeze.get("cem")
        or freeze.get("evidence_role")
        != "fresh_outer_qualification_once"
    ):
        raise RuntimeError("v5 qualification prelabel freeze is invalid")
    if file_sha256(args.selected) != freeze["selected_index_sha256"]:
        raise RuntimeError("v5 qualification selected index changed")
    baseline = json.loads(
        args.baseline_result.read_text(encoding="utf-8")
    )
    if (
        baseline.get("status")
        != "fresh_route_disjoint_outer_evaluated_after_prelabel_freeze"
        or baseline.get("labels_opened_only_after_selection_freeze")
        is not True
        or baseline.get("historically_spent")
        or baseline.get("cem")
    ):
        raise RuntimeError("qualification baseline outcome provenance invalid")
    outcomes_path = Path(baseline["outcomes"])
    if file_sha256(outcomes_path) != baseline["outcomes_sha256"]:
        raise RuntimeError("qualification outcomes changed")
    with np.load(outcomes_path, allow_pickle=False) as payload:
        record_key = payload["record_key"].astype(str)
        trajectory = payload["trajectory"].astype(str)
        route = payload["route_group"].astype(str)
        absolute = payload["absolute_metrics"].astype(np.float32)
    with np.load(args.selected, allow_pickle=False) as payload:
        if not np.array_equal(payload["record_key"].astype(str), record_key):
            raise RuntimeError("v5 selection/outcome identity mismatch")
        selected = payload["multimodal_v5"].astype(np.int64)
    fresh = method_metrics(
        selected,
        absolute,
        trajectory,
        route,
        replicates=args.bootstrap_replicates,
        seed=args.bootstrap_seed,
    )
    incumbent = baseline["methods"]["incumbent"]
    native = baseline["methods"]["native_scorer_v2"]
    fresh_checks = deployment_checks(fresh, incumbent, native)

    calibration = json.loads(
        args.calibration_result.read_text(encoding="utf-8")
    )
    if (
        calibration.get("schema")
        != "recon_route_multimodal_calibration_v5"
        or file_sha256(args.calibration_result)
        != freeze["calibration_result_sha256"]
    ):
        raise RuntimeError("v5 calibration provenance changed")
    dev_selected, dev_targets, cohort = load_development_targets(calibration)
    fresh_targets = absolute - absolute[:, :1]
    cumulative_all = safety_metrics(
        np.concatenate((dev_selected, selected)),
        np.concatenate((dev_targets, fresh_targets)),
    )
    spent = np.char.startswith(cohort, "spent_outer_")
    cumulative_outer300 = safety_metrics(
        np.concatenate((dev_selected[spent], selected)),
        np.concatenate((dev_targets[spent], fresh_targets)),
    )
    development = {
        name: safety_metrics(
            dev_selected[cohort == name],
            dev_targets[cohort == name],
        )
        for name in np.unique(cohort)
    }
    development["spent_pooled_200"] = safety_metrics(
        dev_selected[spent], dev_targets[spent]
    )

    qualification_pass = bool(fresh_checks["overall_pass"])
    report = {
        "schema": "recon_multimodal_v5_qualification_v4_result",
        "status": "qualification_v4_evaluated_once_after_full_freeze",
        "evidence_role": "fresh_outer_qualification_once_now_spent",
        "selection_freeze": str(args.freeze),
        "selection_freeze_sha256": file_sha256(args.freeze),
        "selected_index": str(args.selected),
        "selected_index_sha256": file_sha256(args.selected),
        "baseline_result": str(args.baseline_result),
        "baseline_result_sha256": file_sha256(args.baseline_result),
        "outcomes": str(outcomes_path),
        "outcomes_sha256": file_sha256(outcomes_path),
        "fresh_qualification_v4": {
            "metrics": fresh,
            "deployment_checks": fresh_checks,
            "passed": qualification_pass,
        },
        "development_per_cohort": development,
        "known_heterogeneity": {
            "spent_outer_1_dual_precision": development[
                "spent_outer_1"
            ]["dual_precision"],
            "spent_outer_1_wilson": development[
                "spent_outer_1"
            ]["dual_wilson_lower_95"],
            "spent_outer_1_harm_rate": development[
                "spent_outer_1"
            ]["harm_rate"],
            "interpretation": (
                "v5 has material historical cohort heterogeneity even if "
                "pooled or fresh evidence passes"
            ),
        },
        "cumulative_safety": {
            "scorer_cal_plus_outer_710": {
                "metrics": cumulative_all,
                "passed": safety_pass(cumulative_all),
            },
            "three_outer_cohorts_300": {
                "metrics": cumulative_outer300,
                "passed": safety_pass(cumulative_outer300),
            },
        },
        "scope_limited_cem_qualification": {
            "eligible": qualification_pass,
            "scope": (
                "RECON development/calibration then a new frozen fresh outer"
                if qualification_pass
                else "development-only CEM; no formal qualification"
            ),
            "scand_external_status": (
                "blocked: no unused route/bag-disjoint fresh outer"
            ),
        },
        "no_v5_adaptation_after_v4": True,
        "qualification_v4_now_spent": True,
        "labels_opened_only_after_manifest_feature_policy_selected_sha_freeze": True,
        "bootstrap_replicates": args.bootstrap_replicates,
        "fixed_pool_only": True,
        "cem": False,
    }
    write_json(args.output, report)
    print(
        json.dumps(
            {
                "fresh_qualification_v4": report[
                    "fresh_qualification_v4"
                ],
                "development_per_cohort": development,
                "cumulative_safety": report["cumulative_safety"],
                "scope_limited_cem_qualification": report[
                    "scope_limited_cem_qualification"
                ],
                "result": str(args.output),
                "result_sha256": file_sha256(args.output),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
