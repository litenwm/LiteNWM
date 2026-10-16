#!/usr/bin/env python3
"""Freeze route-robust v3 choices on a label-sealed RECON fixed K17 pool."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from scoring.route_robust_calibration_v3 import (  # noqa: E402
    RISK_FEATURE_DIM,
    RouteRiskCalibratorV3,
    calibrated_probabilities,
    risk_features,
    risk_nonconformity,
    select_fixed_pool,
)
from scoring.spacing_support_v1 import threshold_for_quantile  # noqa: E402
from scoring.universal_dualdomain_raw_v2 import build_universal_features  # noqa: E402
from tools.scoring.calibrate_universal_route_robust_v3 import (  # noqa: E402
    ensemble_with_members,
)
from tools.scoring.finetune_universal_dualdomain_safe_rank_v2 import (  # noqa: E402
    load_envelopes,
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
OUTER = ROOT / "recon_fresh_outer_route_v3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features",
        type=Path,
        default=OUTER
        / "features/recon_fresh_outer_route_h4_100_seed20260724_features.npz",
    )
    parser.add_argument(
        "--feature-audit",
        type=Path,
        default=OUTER
        / "features/recon_fresh_outer_route_h4_100_seed20260724_features.audit.json",
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
        default=ROOT / "route_robust_v3_fixed_q90/result.json",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=OUTER / "prelabel_route_robust_v3"
    )
    parser.add_argument(
        "--deployment-mode",
        choices=("crossfit-hash", "full"),
        default="crossfit-hash",
    )
    parser.add_argument(
        "--evidence-role",
        choices=("fresh_outer", "label_blind_preflight_spent"),
        default="fresh_outer",
    )
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


def route_deployment_fold(route: str, folds: int) -> int:
    digest = hashlib.sha256(
        f"route-risk-v3|recon|{route}".encode()
    ).digest()
    return int.from_bytes(digest[:8], "big") % folds


def read_manifest(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_calibrator(
    record: dict[str, Any], device: str
) -> tuple[RouteRiskCalibratorV3, dict[str, np.ndarray]]:
    path = Path(record["path"])
    if file_sha256(path) != record["sha256"]:
        raise RuntimeError(f"calibrator checkpoint changed: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (
        payload.get("schema") != "route_robust_risk_calibrator_v3"
        or int(payload["input_dim"]) != RISK_FEATURE_DIM
    ):
        raise RuntimeError("route-risk calibrator schema changed")
    model = RouteRiskCalibratorV3(
        input_dim=int(payload["input_dim"]),
        hidden_dim=int(payload["hidden_dim"]),
    ).to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    stats = {
        key: np.asarray(value, dtype=np.float32)
        for key, value in payload["normalization"].items()
    }
    return model, stats


def main() -> None:
    args = parse_args()
    selected_path = args.output_dir / "selected_index.npz"
    freeze_path = args.output_dir / "prelabel_freeze.json"
    if selected_path.exists() or freeze_path.exists():
        raise FileExistsError(selected_path)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    manifest_audit = json.loads(args.manifest_audit.read_text())
    if (
        manifest_audit.get("status") != "manifest_frozen_rgb_only"
        or manifest_audit.get("labels_opened")
        or manifest_audit.get("future_pose_opened")
        or manifest_audit.get("selected_trajectory_overlap")
        or manifest_audit.get("selected_route_overlap")
        or manifest_audit.get("states") != 100
        or manifest_audit.get("route_groups") != 25
    ):
        raise RuntimeError("RECON outer manifest is not fresh and label-sealed")
    if manifest_audit["manifest_sha256"] != file_sha256(args.manifest):
        raise RuntimeError("manifest changed after freeze")

    rows = read_manifest(args.manifest)
    record_key = np.asarray(
        [f"{row['trajectory_id']}:{int(row['current_t'])}" for row in rows]
    )
    trajectory = np.asarray([str(row["trajectory_id"]) for row in rows])
    route = np.asarray([str(row["route_id"]) for row in rows])
    data = load_features(args.features, labels_expected=False)
    feature_audit = json.loads(args.feature_audit.read_text())
    metadata = data["metadata"]
    if (
        metadata.get("labels_accessed")
        or metadata.get("labels_in_output")
        or feature_audit.get("labels_accessed")
        or feature_audit.get("labels_in_output")
    ):
        raise RuntimeError("feature artifact contains future labels")
    if feature_audit["output_sha256"] != file_sha256(args.features):
        raise RuntimeError("feature artifact changed after label-free export")
    if not np.array_equal(data["record_key"].astype(str), record_key):
        raise RuntimeError("feature/manifest identity mismatch")
    with np.load(args.candidate, allow_pickle=False) as payload:
        if not np.array_equal(payload["record_key"].astype(str), record_key):
            raise RuntimeError("candidate/manifest identity mismatch")
        if payload["candidates_waypoint_units"].shape != (100, 17, 4, 2):
            raise RuntimeError("candidate pool is not frozen K17")

    training = json.loads(args.training_result.read_text())
    calibration = json.loads(args.calibration_result.read_text())
    if (
        training.get("cem") is not False
        or calibration.get("cem") is not False
        or not calibration.get("fixed_pool_only")
        or calibration.get("outer_labels_accessed")
        or calibration.get("cem_cal_accessed")
        or not calibration["domains"]["recon"]["nested_group_oof"][
            "target_feasible"
        ]
        or not calibration["domains"]["scand"]["nested_group_oof"][
            "target_feasible"
        ]
    ):
        raise RuntimeError("route-risk calibration is not the frozen q0.90 gate")
    envelopes = load_envelopes(Path(training["support"]))
    universal, support_score, guard = build_universal_features(
        data,
        spacing_m=0.25,
        support_envelope=envelopes["recon"],
    )
    valid = guard & (
        support_score
        <= threshold_for_quantile(envelopes["recon"], 0.999)
    )
    valid[:, 0] = True
    prediction, base_members = ensemble_with_members(
        training["arms"]["D"]["checkpoints"]["full"],
        universal,
        np.zeros(len(universal), dtype=np.int64),
        device=args.device,
    )
    risk = risk_features(
        prediction,
        base_members,
        support_score,
        robot_context=np.zeros((len(universal), 2), dtype=np.float32),
    )
    del base_members

    deployment_records: list[dict[str, Any]] = []
    if args.deployment_mode == "full":
        member_groups = [
            (
                np.arange(len(risk)),
                calibration["domains"]["recon"]["checkpoints"]["full"],
            )
        ]
        deployment_fold = np.full(len(risk), -1, dtype=np.int64)
    else:
        folds = int(calibration["domains"]["recon"]["folds"])
        deployment_fold = np.asarray(
            [route_deployment_fold(value, folds) for value in route],
            dtype=np.int64,
        )
        by_fold = {
            int(value["heldout_fold"]): value["members"]
            for value in calibration["domains"]["recon"]["checkpoints"]["oof"]
        }
        if set(by_fold) != set(range(folds)):
            raise RuntimeError("crossfit checkpoint fold set changed")
        member_groups = [
            (np.flatnonzero(deployment_fold == fold), by_fold[fold])
            for fold in range(folds)
        ]

    p_dual = np.empty(valid.shape, dtype=np.float32)
    p_harm = np.empty(valid.shape, dtype=np.float32)
    calibrator_disagreement = np.empty(valid.shape, dtype=np.float32)
    risk_ood = np.empty(valid.shape, dtype=np.float32)
    for index, records in member_groups:
        if not len(index):
            continue
        dual_members = []
        harm_members = []
        ood_members = []
        for record in records:
            model, stats = load_calibrator(record, args.device)
            member_dual, member_harm = calibrated_probabilities(
                model, risk[index], stats, device=args.device
            )
            dual_members.append(member_dual)
            harm_members.append(member_harm)
            ood_members.append(risk_nonconformity(risk[index], stats))
            del model
        dual_stack = np.stack(dual_members)
        harm_stack = np.stack(harm_members)
        p_dual[index] = np.clip(
            dual_stack.mean(axis=0) - dual_stack.std(axis=0),
            0.0,
            1.0,
        )
        p_harm[index] = np.clip(
            harm_stack.mean(axis=0) + harm_stack.std(axis=0),
            0.0,
            1.0,
        )
        calibrator_disagreement[index] = (
            dual_stack.std(axis=0) + harm_stack.std(axis=0)
        )
        risk_ood[index] = np.stack(ood_members).max(axis=0)
        deployment_records.extend(records)

    policy = calibration["domains"]["recon"]["deployment_policy"]["policy"]
    selected = select_fixed_pool(
        prediction,
        p_dual,
        p_harm,
        valid,
        policy,
        risk_nonconformity_score=risk_ood,
        calibrator_disagreement=calibrator_disagreement,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_npz(
        selected_path,
        record_key=record_key,
        trajectory=trajectory,
        route_group=route,
        route_robust_v3=selected,
        p_dual=p_dual,
        p_harm=p_harm,
        risk_ood=risk_ood,
        calibrator_disagreement=calibrator_disagreement,
    )
    freeze = {
        "schema": "route_robust_v3_recon_fresh_prelabel_freeze",
        "status": "selected_index_frozen_before_future_pose_labels",
        "evidence_role": args.evidence_role,
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
        "calibrator_checkpoints": deployment_records,
        "deployment_mode": args.deployment_mode,
        "deployment_fold_rule": (
            "sha256(route-risk-v3|recon|route) modulo folds"
            if args.deployment_mode == "crossfit-hash"
            else "full_calibrator_ensemble"
        ),
        "deployment_fold_counts": {
            str(value): int((deployment_fold == value).sum())
            for value in np.unique(deployment_fold)
        },
        "selected_index": str(selected_path),
        "selected_index_sha256": file_sha256(selected_path),
        "support_candidate_acceptance": float(valid[:, 1:].mean()),
        "causal_decision_scope": "current_window_only",
        "robot_context": "recon_unknown_all_zero",
        "labels_opened": False,
        "future_pose_outcomes_opened": False,
        "historically_spent": (
            args.evidence_role == "label_blind_preflight_spent"
        ),
        "fixed_pool_only": True,
        "cem_cal_accessed": False,
        "cem": False,
    }
    write_json(freeze_path, freeze)
    print(json.dumps(freeze, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
