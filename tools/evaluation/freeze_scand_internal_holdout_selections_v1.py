#!/usr/bin/env python3
"""Apply the frozen six-arm policies to sealed SCAND holdout features."""

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

from scoring.dense_meta_v1 import (  # noqa: E402
    GroupFirstNet,
    MODES,
    build_meta_features,
    ensemble_predict,
    feature_by_mode,
    predict_ridge,
    residual_rows,
    selected_indices,
)
from scoring.safe_promotion_v1 import (  # noqa: E402
    conservative_utility,
    paired_relative_gains,
)
from scoring.spacing_support_v1 import support_mask  # noqa: E402
from tools.scoring.train_recon_dense_meta_dinov3_three_arm_v1 import (  # noqa: E402
    load_features,
)
from tools.scoring.train_recon_dense_meta_dinov3_v1 import (  # noqa: E402
    file_sha256,
    write_json,
    write_npz,
)


DEFAULT_ROOT = REPO_ROOT / "outputs/scand_domain_adaptation_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features",
        type=Path,
        default=DEFAULT_ROOT
        / "features/internal_holdout/scand_internal_holdout_spacing038_features.npz",
    )
    parser.add_argument(
        "--candidate",
        type=Path,
        default=DEFAULT_ROOT / "candidates/internal_holdout_k17.npz",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_ROOT
        / "manifests/internal_holdout_rgb_available.windows.jsonl",
    )
    parser.add_argument(
        "--training-result",
        type=Path,
        default=DEFAULT_ROOT / "training_v1/result.json",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_ROOT / "holdout_freeze_v1"
    )
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(canonical_json(list(array.shape)).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def load_members(
    training: dict[str, Any], device: str
) -> dict[str, list[tuple[torch.nn.Module, dict[str, np.ndarray]]]]:
    result: dict[str, list[tuple[torch.nn.Module, dict[str, np.ndarray]]]] = {}
    for mode in MODES:
        result[mode] = []
        records = training["checkpoints"][mode]
        if len(records) != 5:
            raise RuntimeError(f"{mode} checkpoint ensemble is not K5")
        for record in records:
            path = Path(record["path"])
            if file_sha256(path) != record["sha256"]:
                raise RuntimeError(f"checkpoint hash changed: {path}")
            payload = torch.load(path, map_location="cpu", weights_only=False)
            if payload.get("feature_variant") != "raw" or payload.get(
                "scorer_v2_predictions_in_features"
            ):
                raise RuntimeError("holdout inference received a stacked checkpoint")
            config = payload["config"]
            model = GroupFirstNet(
                int(payload["input_dim"]),
                int(config["hidden"]),
                float(config["dropout"]),
            ).to(device)
            model.load_state_dict(payload["state_dict"], strict=True)
            model.eval().requires_grad_(False)
            stats = {
                key: np.asarray(value, dtype=np.float32)
                for key, value in payload["normalization"].items()
            }
            result[mode].append((model, stats))
    return result


def load_support(path: Path, expected_sha: str) -> dict[str, Any]:
    if file_sha256(path) != expected_sha:
        raise RuntimeError(f"support checkpoint hash changed: {path}")
    with np.load(path, allow_pickle=False) as payload:
        metadata = json.loads(str(payload["metadata"].item()))
        result = {
            key: payload[key].copy()
            for key in payload.files
            if key != "metadata"
        }
    result.update(metadata)
    return result


def predict_modes(
    members: dict[str, list[tuple[torch.nn.Module, dict[str, np.ndarray]]]],
    data: dict[str, Any],
    device: str,
) -> dict[str, dict[str, np.ndarray]]:
    return {
        mode: ensemble_predict(
            members[mode],
            feature_by_mode(data, mode, variant="raw"),
            device=device,
        )
        for mode in MODES
    }


def native_selection(
    data: dict[str, Any],
    valid: np.ndarray,
    policy: dict[str, Any],
    ate_scale: float,
    rtp_scale: float,
) -> np.ndarray:
    gains = paired_relative_gains(
        data["ate_by_seed"], data["rtp_by_seed"], uncertainty_beta=1.0
    )
    utility = conservative_utility(
        gains, ate_scale=ate_scale, rtp_scale=rtp_scale
    )
    allowed = np.asarray(valid, dtype=bool).copy()
    allowed[:, 0] = False
    ranked = np.where(allowed, utility, -np.inf)
    challenger = ranked.argmax(axis=1).astype(np.int64)
    exists = allowed.any(axis=1)
    challenger = np.where(exists, challenger, 0)
    row = np.arange(len(challenger))
    score = np.where(exists, utility[row, challenger], -np.inf)
    threshold = float(policy["promotion_threshold"])
    return np.where(exists & (score >= threshold), challenger, 0).astype(np.int64)


def raw_selection(
    predictions: dict[str, dict[str, np.ndarray]],
    data: dict[str, Any],
    valid: np.ndarray,
    policy: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    rows = {}
    selections = {}
    for mode in MODES:
        mode_policy = policy["policies"][mode]
        rows[mode] = residual_rows(
            predictions["action"],
            predictions[mode],
            alpha=float(mode_policy["alpha"]),
            gate=str(mode_policy["gate"]),
            risk=str(mode_policy["risk"]),
            valid=valid,
        )
        selections[mode] = selected_indices(
            rows[mode], float(mode_policy["threshold"])
        )
    feature = build_meta_features(
        predictions,
        rows,
        {
            key: np.asarray(data[key])
            for key in ("action_features", "static_features", "raw_ltwm_features")
        },
        variant="raw",
    )
    meta = policy["meta"]
    ridge = {
        "feature_mean": np.asarray(meta["feature_mean"], dtype=np.float32),
        "feature_scale": np.asarray(meta["feature_scale"], dtype=np.float32),
        "coefficient": np.asarray(meta["coefficient"], dtype=np.float32),
        "intercept": float(meta["intercept"]),
    }
    score = predict_ridge(ridge, feature)
    selected = np.where(
        score >= float(meta["threshold"]),
        selections["full"],
        selections["static"],
    ).astype(np.int64)
    return selected, score


def selection_audit(selected: np.ndarray, valid: np.ndarray) -> dict[str, Any]:
    row = np.arange(len(selected))
    if not valid[row, selected].all():
        raise RuntimeError("frozen policy selected an unsupported candidate")
    return {
        "states": int(len(selected)),
        "promotions": int((selected != 0).sum()),
        "selected_index_sha256": array_sha256(selected),
    }


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    data = load_features(args.features, labels_expected=False)
    if data["metadata"]["split"] != "scand_internal_holdout_spacing038":
        raise RuntimeError("holdout feature split changed")
    if data["metadata"]["labels_accessed"] or data["metadata"]["labels_in_output"]:
        raise RuntimeError("holdout features unexpectedly contain labels")
    training = json.loads(args.training_result.read_text(encoding="utf-8"))
    if training.get("status") != "calibrated_six_arm_fixed_pool_cem_false":
        raise RuntimeError("training result is not frozen/calibrated")
    if training.get("spent_strict100_outcomes_used") or training.get(
        "internal_holdout_accessed"
    ):
        raise RuntimeError("training provenance violates sealed holdout")
    if training.get("cem"):
        raise RuntimeError("CEM must remain disabled")
    if len(data["record_key"]) != 984:
        raise RuntimeError("sealed RGB-available holdout count changed")
    args.output_dir.mkdir(parents=True)
    spacing = np.full(len(data["record_key"]), 0.38, dtype=np.float32)
    arrays: dict[str, np.ndarray] = {
        "record_key": data["record_key"].astype(str),
        "family": data["family"].astype(str),
    }
    audit: dict[str, Any] = {}
    ate_scale = float(training["scorer_v2"]["ate_scale"])
    rtp_scale = float(training["scorer_v2"]["rtp_scale"])
    for regime_name in ("scand_only", "domain_balanced"):
        regime = training["regimes"][regime_name]
        members = load_members(regime["training"], args.device)
        predictions = predict_modes(members, data, args.device)
        calibration = regime["calibration"]
        support = load_support(
            Path(calibration["support_checkpoint"]),
            calibration["support_checkpoint_sha256"],
        )
        regime_audit: dict[str, Any] = {}
        for arm_name in ("support_only", "raw_only", "support_plus_raw"):
            policy = calibration["arms"][arm_name]
            if arm_name == "raw_only":
                valid = data["majority_support"].astype(bool)
                support_info = {
                    "source": "frozen_scorer_v2_majority_support",
                    "candidate_acceptance": float(valid[:, 1:].mean()),
                    "no_supported_proposal_windows": int(
                        (~valid[:, 1:].any(axis=1)).sum()
                    ),
                }
            else:
                valid, support_info = support_mask(
                    support,
                    data["actions_h4"],
                    spacing,
                    quantile=float(policy["support_quantile"]),
                )
            if arm_name == "support_only":
                selected = native_selection(
                    data, valid, policy, ate_scale, rtp_scale
                )
                meta_score = np.zeros(len(selected), dtype=np.float32)
            else:
                selected, meta_score = raw_selection(
                    predictions, data, valid, policy
                )
            label = f"{regime_name}__{arm_name}"
            arrays[label] = selected
            arrays[f"{label}__meta_score"] = meta_score
            regime_audit[arm_name] = {
                **selection_audit(selected, valid),
                "support": support_info,
            }
        audit[regime_name] = regime_audit
        del members, predictions
        if args.device == "cuda":
            torch.cuda.empty_cache()
    selected_path = args.output_dir / "selected_index.npz"
    write_npz(selected_path, **arrays)
    checkpoint_hashes = sorted(
        {
            record["sha256"]
            for regime in training["regimes"].values()
            for records in regime["training"]["checkpoints"].values()
            for record in records
        }
    )
    freeze = {
        "schema": "scand_internal_holdout_prelabel_freeze_v1",
        "status": "selected_indices_frozen_before_future_pose_labels",
        "states": int(len(data["record_key"])),
        "trajectories": 21,
        "route_groups": 4,
        "manifest": str(args.manifest),
        "manifest_sha256": file_sha256(args.manifest),
        "candidate_pool": str(args.candidate),
        "candidate_pool_sha256": file_sha256(args.candidate),
        "feature_artifact": str(args.features),
        "feature_artifact_sha256": file_sha256(args.features),
        "training_result": str(args.training_result),
        "training_result_sha256": file_sha256(args.training_result),
        "model_checkpoint_sha256": checkpoint_hashes,
        "selected_index": str(selected_path),
        "selected_index_sha256": file_sha256(selected_path),
        "selection_audit": audit,
        "labels_opened": False,
        "future_pose_outcomes_opened": False,
        "spent_strict100_outcomes_used": False,
        "cem_cal_accessed": False,
        "fixed_pool_only": True,
        "cem": False,
    }
    write_json(args.output_dir / "prelabel_freeze.json", freeze)
    print(json.dumps(freeze, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
