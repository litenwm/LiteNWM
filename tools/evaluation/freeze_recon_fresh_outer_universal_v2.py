#!/usr/bin/env python3
"""Freeze universal-v2 and native scorer-v2 choices on fresh RECON H4-100."""

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

from scoring.safe_promotion_v1 import greedy_fixed_pool_selection  # noqa: E402
from scoring.spacing_support_v1 import (  # noqa: E402
    support_mask,
    threshold_for_quantile,
)
from scoring.universal_dualdomain_raw_v2 import build_universal_features  # noqa: E402
from tools.evaluation.freeze_scand_internal_holdout_selections_v1 import (  # noqa: E402
    load_members as load_legacy_members,
    load_support as load_legacy_support,
    predict_modes as predict_legacy_modes,
    raw_selection as legacy_raw_selection,
)
from tools.scoring.finetune_universal_dualdomain_safe_rank_v2 import (  # noqa: E402
    load_envelopes,
    policy_selected,
)
from tools.scoring.train_recon_dense_meta_dinov3_three_arm_v1 import (  # noqa: E402
    load_features,
)
from tools.scoring.train_recon_dense_meta_dinov3_v1 import (  # noqa: E402
    file_sha256,
    write_json,
    write_npz,
)
from tools.scoring.train_universal_dualdomain_raw_v2 import (  # noqa: E402
    ensemble_predict,
    selected_for_policy,
    sigmoid,
)


ROOT = REPO_ROOT / "outputs/universal_dualdomain_raw_v2"
FRESH_ROOT = ROOT / "recon_fresh_outer_v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features",
        type=Path,
        default=FRESH_ROOT
        / "features/recon_fresh_outer_h4_100_seed20260721_features.npz",
    )
    parser.add_argument(
        "--feature-audit",
        type=Path,
        default=FRESH_ROOT
        / "features/recon_fresh_outer_h4_100_seed20260721_features.audit.json",
    )
    parser.add_argument(
        "--candidate", type=Path, default=FRESH_ROOT / "fresh_outer_k17.npz"
    )
    parser.add_argument(
        "--manifest", type=Path, default=FRESH_ROOT / "manifest.windows.jsonl"
    )
    parser.add_argument(
        "--manifest-audit",
        type=Path,
        default=FRESH_ROOT / "manifest.windows.audit.json",
    )
    parser.add_argument(
        "--training-result",
        type=Path,
        default=ROOT / "training_v2/result.json",
    )
    parser.add_argument(
        "--safe-rank-result",
        type=Path,
        default=ROOT / "safe_rank_v2/result.json",
    )
    parser.add_argument(
        "--legacy-balanced-result",
        type=Path,
        default=REPO_ROOT
        / "outputs/scand_domain_adaptation_v1/training_v1/result.json",
    )
    parser.add_argument(
        "--native-policy",
        type=Path,
        default=REPO_ROOT
        / "outputs/recon_dense_meta_dinov3_v1/"
        "three_arm_training/frozen_policy.json",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=FRESH_ROOT / "prelabel_freeze"
    )
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


def read_manifest(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def arm_policy(
    result: dict[str, Any], arm: str
) -> tuple[dict[str, float], float, float]:
    value = result["arms"][arm]
    if arm == "B":
        frontier = value["calibration"]["frontier"]
        temperature = value["temperature"]["record"]
    else:
        frontier = value["calibration"]["frontier"]["domains"]["recon"]
        temperature = value["temperature"]["domains"]["recon"]
    return (
        frontier["policy"],
        float(temperature["deployment_dual_temperature"]),
        float(temperature["deployment_harm_temperature"]),
    )


def main() -> None:
    args = parse_args()
    selected_path = args.output_dir / "selected_index.npz"
    freeze_path = args.output_dir / "prelabel_freeze.json"
    if selected_path.exists() or freeze_path.exists():
        raise FileExistsError(selected_path)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    manifest_audit = json.loads(args.manifest_audit.read_text(encoding="utf-8"))
    if (
        manifest_audit.get("status") != "manifest_frozen_rgb_only"
        or manifest_audit.get("labels_opened")
        or manifest_audit.get("future_pose_opened")
        or manifest_audit.get("selected_trajectory_overlap")
        or manifest_audit.get("selected_route_overlap")
        or manifest_audit.get("states") != 100
        or manifest_audit.get("trajectories") != 25
        or manifest_audit.get("route_groups") != 25
    ):
        raise RuntimeError("fresh manifest is not label-free and disjoint")
    if manifest_audit["manifest_sha256"] != file_sha256(args.manifest):
        raise RuntimeError("manifest changed after RGB-only freeze")
    rows = read_manifest(args.manifest)
    record_key = np.asarray(
        [f"{row['trajectory_id']}:{int(row['current_t'])}" for row in rows]
    )
    trajectory = np.asarray([str(row["trajectory_id"]) for row in rows])
    route = np.asarray([str(row["route_id"]) for row in rows])
    if (
        len(rows) != 100
        or len(np.unique(trajectory)) != 25
        or len(np.unique(route)) != 25
    ):
        raise RuntimeError("fresh RECON H4-100 clustering contract changed")

    feature_audit = json.loads(args.feature_audit.read_text(encoding="utf-8"))
    data = load_features(args.features, labels_expected=False)
    metadata = data["metadata"]
    if (
        metadata.get("labels_accessed")
        or metadata.get("labels_in_output")
        or feature_audit.get("labels_accessed")
        or feature_audit.get("labels_in_output")
    ):
        raise RuntimeError("prelabel feature artifact contains labels")
    if feature_audit["output_sha256"] != file_sha256(args.features):
        raise RuntimeError("feature artifact changed after export")
    if not np.array_equal(data["record_key"].astype(str), record_key):
        raise RuntimeError("feature/manifest identity mismatch")
    with np.load(args.candidate, allow_pickle=False) as payload:
        if not np.array_equal(payload["record_key"].astype(str), record_key):
            raise RuntimeError("candidate/manifest identity mismatch")
        candidate_count = int(
            payload["candidates_waypoint_units"].shape[1]
        )
    if candidate_count != 17:
        raise RuntimeError("fresh pool is no longer K17")

    training = json.loads(args.training_result.read_text(encoding="utf-8"))
    safe_rank = json.loads(args.safe_rank_result.read_text(encoding="utf-8"))
    if (
        training.get("cem") is not False
        or safe_rank.get("cem") is not False
        or not training.get("primary_excludes_scorer_v2_predictions")
        or not safe_rank.get("primary_excludes_scorer_v2_predictions")
    ):
        raise RuntimeError("universal model contract is not raw-only/CEM-false")
    envelopes = load_envelopes(Path(training["support"]))
    feature, support_score, guard = build_universal_features(
        data,
        spacing_m=0.25,
        support_envelope=envelopes["recon"],
    )
    support_threshold = threshold_for_quantile(envelopes["recon"], 0.999)
    valid = guard & (support_score <= support_threshold)
    valid[:, 0] = True
    domain = np.zeros(len(feature), dtype=np.int64)

    arrays: dict[str, np.ndarray] = {
        "record_key": record_key,
        "trajectory": trajectory,
        "route_group": route,
    }
    selection_audit: dict[str, Any] = {}
    for arm in ("B", "C", "D"):
        prediction = ensemble_predict(
            training["arms"][arm]["checkpoints"]["full"],
            feature,
            domain,
            device=args.device,
        )
        policy, dual_temperature, harm_temperature = arm_policy(training, arm)
        selected = selected_for_policy(
            prediction,
            sigmoid(prediction["dual_logit"] / dual_temperature),
            sigmoid(prediction["harm_logit"] / harm_temperature),
            valid,
            policy,
        )
        name = f"universal_{arm.lower()}"
        arrays[name] = selected
        selection_audit[name] = {
            "promotions": int((selected != 0).sum()),
            "promotion_fraction": float((selected != 0).mean()),
            "calibration_feasible_both_domains": bool(
                training["arms"][arm]["calibration"]["feasible"]
            ),
            "policy": policy,
            "dual_temperature": dual_temperature,
            "harm_temperature": harm_temperature,
        }

    safe_prediction = ensemble_predict(
        safe_rank["checkpoints"]["full"],
        feature,
        domain,
        device=args.device,
    )
    safe_selected = policy_selected(
        safe_prediction,
        valid,
        safe_rank["calibration"]["recon"]["policy"],
    )
    arrays["safe_rank_failed_ablation"] = safe_selected
    selection_audit["safe_rank_failed_ablation"] = {
        "promotions": int((safe_selected != 0).sum()),
        "promotion_fraction": float((safe_selected != 0).mean()),
        "calibration_feasible": bool(
            safe_rank["calibration"]["recon"]["feasible"]
        ),
    }

    legacy = json.loads(args.legacy_balanced_result.read_text(encoding="utf-8"))
    if (
        legacy.get("status") != "calibrated_six_arm_fixed_pool_cem_false"
        or legacy.get("cem") is not False
    ):
        raise RuntimeError("legacy balanced support+raw result is not frozen")
    legacy_regime = legacy["regimes"]["domain_balanced"]
    legacy_members = load_legacy_members(
        legacy_regime["training"], args.device
    )
    legacy_predictions = predict_legacy_modes(
        legacy_members, data, args.device
    )
    legacy_calibration = legacy_regime["calibration"]
    legacy_support = load_legacy_support(
        Path(legacy_calibration["support_checkpoint"]),
        legacy_calibration["support_checkpoint_sha256"],
    )
    legacy_policy = legacy_calibration["arms"]["support_plus_raw"]
    legacy_valid, legacy_support_audit = support_mask(
        legacy_support,
        data["actions_h4"],
        np.full(len(data["record_key"]), 0.25, dtype=np.float32),
        quantile=float(legacy_policy["support_quantile"]),
    )
    legacy_selected, _ = legacy_raw_selection(
        legacy_predictions,
        data,
        legacy_valid,
        legacy_policy,
    )
    arrays["legacy_balanced_support_plus_raw"] = legacy_selected
    selection_audit["legacy_balanced_support_plus_raw"] = {
        "promotions": int((legacy_selected != 0).sum()),
        "promotion_fraction": float((legacy_selected != 0).mean()),
        "calibration_feasible": bool(legacy_policy["feasible_gate"]),
        "support": legacy_support_audit,
    }
    del legacy_members, legacy_predictions
    if args.device == "cuda":
        torch.cuda.empty_cache()

    native_policy = json.loads(args.native_policy.read_text(encoding="utf-8"))
    native = native_policy["arms"]["dinov3_scorer_v2_native"]
    native_selected = greedy_fixed_pool_selection(
        data["ate_by_seed"],
        data["rtp_by_seed"],
        ate_scale=float(native["ate_scale"]),
        rtp_scale=float(native["rtp_scale"]),
        valid=data["majority_support"],
    )
    arrays["native_scorer_v2"] = native_selected
    selection_audit["native_scorer_v2"] = {
        "promotions": int((native_selected != 0).sum()),
        "promotion_fraction": float((native_selected != 0).mean()),
        "feature_contract": native["feature_contract"],
        "ate_scale": float(native["ate_scale"]),
        "rtp_scale": float(native["rtp_scale"]),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_npz(selected_path, **arrays)
    freeze = {
        "schema": "universal_dualdomain_raw_v2_recon_fresh_prelabel_freeze",
        "status": "all_fixed_pool_selections_frozen_before_future_pose_labels",
        "evidence_role": "fresh route-disjoint outer H4-100",
        "states": 100,
        "trajectories": 25,
        "route_groups": 25,
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
        "safe_rank_result": str(args.safe_rank_result),
        "safe_rank_result_sha256": file_sha256(args.safe_rank_result),
        "legacy_balanced_result": str(args.legacy_balanced_result),
        "legacy_balanced_result_sha256": file_sha256(
            args.legacy_balanced_result
        ),
        "native_policy": str(args.native_policy),
        "native_policy_sha256": file_sha256(args.native_policy),
        "selected_index": str(selected_path),
        "selected_index_sha256": file_sha256(selected_path),
        "support_candidate_acceptance": float(valid[:, 1:].mean()),
        "support_window_acceptance": float(valid[:, 1:].any(axis=1).mean()),
        "native_majority_support_candidate_acceptance": float(
            data["majority_support"][:, 1:].mean()
        ),
        "selection_audit": selection_audit,
        "labels_opened": False,
        "future_pose_outcomes_opened": False,
        "historically_spent": False,
        "fixed_pool_only": True,
        "cem_cal_accessed": False,
        "cem": False,
    }
    write_json(freeze_path, freeze)
    print(json.dumps(freeze, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
