#!/usr/bin/env python3
"""Freeze multimodal-v5 choices on sealed RECON qualification_v4."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from scoring.route_multimodal_calibration_v5 import (  # noqa: E402
    calibrated_probabilities_v5,
    multimodal_risk_features,
)
from scoring.route_robust_calibration_v3 import (  # noqa: E402
    risk_features,
    risk_nonconformity,
    select_fixed_pool,
)
from scoring.spacing_support_v1 import threshold_for_quantile  # noqa: E402
from scoring.universal_dualdomain_raw_v2 import build_universal_features  # noqa: E402
from tools.evaluation.freeze_recon_fresh_outer_route_robust_v3 import (  # noqa: E402
    read_manifest,
)
from tools.scoring.calibrate_universal_route_robust_v3 import (  # noqa: E402
    ensemble_with_members,
)
from tools.scoring.finetune_universal_dualdomain_safe_rank_v2 import (  # noqa: E402
    load_envelopes,
)
from tools.scoring.freeze_recon_route_multimodal_v5_mapping import (  # noqa: E402
    load_calibrator,
)
from tools.scoring.train_recon_dense_meta_dinov3_three_arm_v1 import (  # noqa: E402
    load_features,
)
from tools.scoring.train_recon_dense_meta_dinov3_v1 import (  # noqa: E402
    file_sha256,
    write_json,
    write_npz,
)


ROOT = REPO_ROOT / "outputs/universal_dualdomain_raw_v2"
OUTER = ROOT / "recon_fresh_outer_qualification_v4"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features",
        type=Path,
        default=OUTER
        / "features/"
        "recon_fresh_outer_qualification_h4_100_seed20260725_features.npz",
    )
    parser.add_argument(
        "--feature-audit",
        type=Path,
        default=OUTER
        / "features/"
        "recon_fresh_outer_qualification_h4_100_seed20260725_features.audit.json",
    )
    parser.add_argument(
        "--candidate", type=Path, default=OUTER / "fresh_outer_k17.npz"
    )
    parser.add_argument(
        "--manifest", type=Path, default=OUTER / "manifest.windows.jsonl"
    )
    parser.add_argument(
        "--manifest-audit",
        type=Path,
        default=OUTER / "manifest.windows.audit.json",
    )
    parser.add_argument(
        "--training-result",
        type=Path,
        default=ROOT / "training_v2/result.json",
    )
    parser.add_argument(
        "--calibration-result",
        type=Path,
        default=ROOT / "recon_route_multimodal_v5/result.json",
    )
    parser.add_argument(
        "--deployment-mapping",
        type=Path,
        default=ROOT
        / "recon_route_multimodal_v5/deployment_mapping.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTER / "prelabel_multimodal_v5",
    )
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


def validate_identity(
    args: argparse.Namespace,
    data: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    manifest_audit = json.loads(
        args.manifest_audit.read_text(encoding="utf-8")
    )
    if (
        manifest_audit.get("status") != "manifest_frozen_rgb_only"
        or manifest_audit.get("labels_opened")
        or manifest_audit.get("future_pose_opened")
        or manifest_audit.get("selected_trajectory_overlap")
        or manifest_audit.get("selected_route_overlap")
        or manifest_audit.get("states") != 100
        or manifest_audit.get("route_groups") != 25
    ):
        raise RuntimeError("qualification_v4 is not fresh and label-sealed")
    if manifest_audit["manifest_sha256"] != file_sha256(args.manifest):
        raise RuntimeError("qualification manifest changed")
    rows = read_manifest(args.manifest)
    record_key = np.asarray(
        [f"{row['trajectory_id']}:{int(row['current_t'])}" for row in rows]
    )
    trajectory = np.asarray([str(row["trajectory_id"]) for row in rows])
    route = np.asarray([str(row["route_id"]) for row in rows])
    feature_audit = json.loads(
        args.feature_audit.read_text(encoding="utf-8")
    )
    metadata = data["metadata"]
    if (
        metadata.get("labels_accessed")
        or metadata.get("labels_in_output")
        or feature_audit.get("labels_accessed")
        or feature_audit.get("labels_in_output")
    ):
        raise RuntimeError("qualification features contain labels")
    if feature_audit["output_sha256"] != file_sha256(args.features):
        raise RuntimeError("qualification feature artifact changed")
    if not np.array_equal(data["record_key"].astype(str), record_key):
        raise RuntimeError("qualification feature/manifest identity mismatch")
    with np.load(args.candidate, allow_pickle=False) as payload:
        if not np.array_equal(
            payload["record_key"].astype(str), record_key
        ):
            raise RuntimeError("qualification candidate identity mismatch")
        if payload["candidates_waypoint_units"].shape != (100, 17, 4, 2):
            raise RuntimeError("qualification pool is not K17")
    return record_key, trajectory, route, manifest_audit


def main() -> None:
    args = parse_args()
    selected_path = args.output_dir / "selected_index.npz"
    freeze_path = args.output_dir / "prelabel_freeze.json"
    if selected_path.exists() or freeze_path.exists():
        raise FileExistsError(selected_path)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    data = load_features(args.features, labels_expected=False)
    record_key, trajectory, route, manifest_audit = validate_identity(
        args, data
    )
    training = json.loads(args.training_result.read_text(encoding="utf-8"))
    calibration = json.loads(
        args.calibration_result.read_text(encoding="utf-8")
    )
    mapping = json.loads(
        args.deployment_mapping.read_text(encoding="utf-8")
    )
    if (
        training.get("cem") is not False
        or calibration.get("schema")
        != "recon_route_multimodal_calibration_v5"
        or calibration.get("qualification_v4_accessed") is not False
        or mapping.get("schema")
        != "recon_route_multimodal_v5_deployment_mapping"
        or mapping.get("qualification_v4_accessed") is not False
        or mapping.get("qualification_v4_covariates_used") is not False
        or mapping.get("outer_outcomes_used") is not False
        or mapping.get("cem") is not False
    ):
        raise RuntimeError("multimodal-v5 freeze provenance is invalid")
    if (
        mapping["calibration_result_sha256"]
        != file_sha256(args.calibration_result)
    ):
        raise RuntimeError("calibration changed after deployment mapping")

    envelopes = load_envelopes(Path(training["support"]))
    universal, support, guard = build_universal_features(
        data,
        spacing_m=0.25,
        support_envelope=envelopes["recon"],
    )
    valid = guard & (
        support <= threshold_for_quantile(envelopes["recon"], 0.999)
    )
    valid[:, 0] = True
    prediction, base_members = ensemble_with_members(
        training["arms"]["D"]["checkpoints"]["full"],
        universal,
        np.zeros(len(universal), dtype=np.int64),
        device=args.device,
    )
    base_risk = risk_features(
        prediction,
        base_members,
        support,
        robot_context=np.zeros((len(universal), 2), dtype=np.float32),
    )
    risk = multimodal_risk_features(base_risk, universal)
    del base_members

    dual_members = []
    harm_members = []
    ood_members = []
    checkpoint_audit = []
    for record in calibration["checkpoints"]["full"]:
        path = Path(record["path"])
        if record["sha256"] != file_sha256(path):
            raise RuntimeError(f"calibrator checkpoint changed: {path}")
        model, stats = load_calibrator(path, args.device)
        dual, harm = calibrated_probabilities_v5(
            model, risk, stats, device=args.device
        )
        dual_members.append(dual)
        harm_members.append(harm)
        ood_members.append(risk_nonconformity(risk, stats))
        checkpoint_audit.append(
            {"path": str(path), "sha256": file_sha256(path)}
        )
        del model
        if args.device == "cuda":
            torch.cuda.empty_cache()
    dual_stack = np.stack(dual_members)
    harm_stack = np.stack(harm_members)
    p_dual = np.clip(
        dual_stack.mean(0) - dual_stack.std(0), 0.0, 1.0
    )
    p_harm = np.clip(
        harm_stack.mean(0) + harm_stack.std(0), 0.0, 1.0
    )
    disagreement = dual_stack.std(0) + harm_stack.std(0)
    ood = np.stack(ood_members).max(0)
    policy = mapping["mapped_full_policy"]
    selected = select_fixed_pool(
        prediction,
        p_dual,
        p_harm,
        valid,
        policy,
        risk_nonconformity_score=ood,
        calibrator_disagreement=disagreement,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_npz(
        selected_path,
        record_key=record_key,
        trajectory=trajectory,
        route_group=route,
        multimodal_v5=selected,
        p_dual=p_dual,
        p_harm=p_harm,
        risk_ood=ood,
        calibrator_disagreement=disagreement,
    )
    freeze = {
        "schema": "recon_multimodal_v5_qualification_v4_prelabel_freeze",
        "status": "selected_index_frozen_before_future_pose_labels",
        "evidence_role": "fresh_outer_qualification_once",
        "states": 100,
        "trajectories": int(len(np.unique(trajectory))),
        "route_groups": int(len(np.unique(route))),
        "promotions": int((selected != 0).sum()),
        "coverage": float((selected != 0).mean()),
        "policy": policy,
        "manifest": str(args.manifest),
        "manifest_sha256": file_sha256(args.manifest),
        "manifest_audit": str(args.manifest_audit),
        "manifest_audit_sha256": file_sha256(args.manifest_audit),
        "candidate_pool": str(args.candidate),
        "candidate_pool_sha256": file_sha256(args.candidate),
        "feature_artifact": str(args.features),
        "feature_artifact_sha256": file_sha256(args.features),
        "training_result": str(args.training_result),
        "training_result_sha256": file_sha256(args.training_result),
        "calibration_result": str(args.calibration_result),
        "calibration_result_sha256": file_sha256(args.calibration_result),
        "deployment_mapping": str(args.deployment_mapping),
        "deployment_mapping_sha256": file_sha256(
            args.deployment_mapping
        ),
        "calibrator_checkpoints": checkpoint_audit,
        "selected_index": str(selected_path),
        "selected_index_sha256": file_sha256(selected_path),
        "support_candidate_acceptance": float(valid[:, 1:].mean()),
        "development_evidence_disclosure": {
            "group_oof": calibration["nested_group_oof"][
                "cohort_metrics"
            ],
            "deployment_oof": calibration["deployment_policy"][
                "cohort_metrics"
            ],
            "known_heterogeneity": (
                "spent_outer_1 deployment OOF dual=0.8372; "
                "must remain visible in cumulative conclusion"
            ),
        },
        "manifest_was_label_sealed": (
            manifest_audit.get("labels_opened") is False
            and manifest_audit.get("future_pose_opened") is False
        ),
        "causal_decision_scope": "current_window_only",
        "route_quota_used": False,
        "labels_opened": False,
        "future_pose_outcomes_opened": False,
        "fixed_pool_only": True,
        "cem_cal_accessed": False,
        "cem": False,
    }
    write_json(freeze_path, freeze)
    print(json.dumps(freeze, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
