#!/usr/bin/env python3
"""Map OOF route-risk thresholds to full-ensemble scorer-cal scales."""

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

from scoring.route_robust_calibration_v3 import (  # noqa: E402
    RISK_FEATURE_DIM,
    RouteRiskCalibratorV3,
    calibrated_probabilities,
    risk_features,
    risk_nonconformity,
)
from scoring.spacing_support_v1 import threshold_for_quantile  # noqa: E402
from scoring.universal_dualdomain_raw_v2 import build_universal_features  # noqa: E402
from tools.scoring.calibrate_universal_route_robust_v3 import (  # noqa: E402
    challenger_for_rank,
    ensemble_with_members,
    robot_feature_context,
    select_challenger,
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
)
from tools.scoring.train_universal_dualdomain_raw_v2 import load_group  # noqa: E402


ROOT = REPO_ROOT / "outputs/universal_dualdomain_raw_v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--training-result", type=Path, default=ROOT / "training_v2/result.json"
    )
    parser.add_argument(
        "--calibration-result",
        type=Path,
        default=ROOT / "route_robust_v3_fixed_q90/result.json",
    )
    parser.add_argument(
        "--recon-features",
        type=Path,
        default=ROOT
        / "features/recon_scorer_cal/recon_scorer_cal_universal_v2_features.npz",
    )
    parser.add_argument(
        "--recon-source",
        type=Path,
        default=ROOT / "sources/recon_scorer_cal_source.npz",
    )
    parser.add_argument(
        "--scand-features",
        type=Path,
        default=REPO_ROOT
        / "outputs/scand_domain_adaptation_v1/features/scorer_cal/"
        "scand_inner_scorer_cal_spacing038_features.npz",
    )
    parser.add_argument(
        "--scand-source",
        type=Path,
        default=REPO_ROOT
        / "outputs/scand_domain_adaptation_v1/sources/scorer_cal_source.npz",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "route_robust_v3_fixed_q90/deployment_mapping.json",
    )
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


def load_calibrator(
    record: dict[str, Any], device: str
) -> tuple[RouteRiskCalibratorV3, dict[str, np.ndarray]]:
    path = Path(record["path"])
    if file_sha256(path) != record["sha256"]:
        raise RuntimeError(f"checkpoint changed: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (
        payload.get("schema") != "route_robust_risk_calibrator_v3"
        or int(payload["input_dim"]) != RISK_FEATURE_DIM
    ):
        raise RuntimeError("calibrator schema changed")
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


def percentile(reference: np.ndarray, threshold: float) -> float:
    value = np.asarray(reference, dtype=np.float64)
    if not len(value) or not np.isfinite(value).all():
        raise RuntimeError("invalid scorer-cal reference distribution")
    return float(np.mean(value <= threshold))


def mapped_threshold(
    oof: np.ndarray, full: np.ndarray, threshold: float
) -> dict[str, float]:
    quantile = percentile(oof, threshold)
    return {
        "oof_threshold": float(threshold),
        "reference_quantile": quantile,
        "full_threshold": float(np.quantile(full, quantile)),
    }


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    training = json.loads(args.training_result.read_text())
    calibration = json.loads(args.calibration_result.read_text())
    if (
        training.get("cem") is not False
        or calibration.get("cem") is not False
        or calibration.get("outer_labels_accessed")
        or calibration.get("cem_cal_accessed")
    ):
        raise RuntimeError("invalid calibration provenance")
    predictions_path = Path(calibration["calibration_predictions"])
    with np.load(predictions_path, allow_pickle=False) as payload:
        oof_arrays = {key: payload[key].copy() for key in payload.files}

    envelopes = load_envelopes(Path(training["support"]))
    inputs = {
        "recon": (args.recon_features, args.recon_source, 0.25, 0),
        "scand": (args.scand_features, args.scand_source, 0.38, 1),
    }
    results = {}
    for domain, (feature_path, source_path, spacing, domain_index) in inputs.items():
        data = load_features(feature_path, labels_expected=True)
        group = load_group(source_path, data["record_key"], domain)
        universal, support_score, guard = build_universal_features(
            data,
            spacing_m=spacing,
            support_envelope=envelopes[domain],
        )
        valid = guard & (
            support_score
            <= threshold_for_quantile(envelopes[domain], 0.999)
        )
        valid[:, 0] = True
        prediction, base_members = ensemble_with_members(
            training["arms"]["D"]["checkpoints"]["full"],
            universal,
            np.full(len(universal), domain_index, dtype=np.int64),
            device=args.device,
        )
        risk = risk_features(
            prediction,
            base_members,
            support_score,
            robot_context=robot_feature_context(group),
        )
        del base_members

        dual_members = []
        harm_members = []
        ood_members = []
        full_records = calibration["domains"][domain]["checkpoints"]["full"]
        for record in full_records:
            model, stats = load_calibrator(record, args.device)
            member_dual, member_harm = calibrated_probabilities(
                model, risk, stats, device=args.device
            )
            dual_members.append(member_dual)
            harm_members.append(member_harm)
            ood_members.append(risk_nonconformity(risk, stats))
            del model
        dual_stack = np.stack(dual_members)
        harm_stack = np.stack(harm_members)
        full_dual = np.clip(
            dual_stack.mean(axis=0) - dual_stack.std(axis=0), 0.0, 1.0
        )
        full_harm = np.clip(
            harm_stack.mean(axis=0) + harm_stack.std(axis=0), 0.0, 1.0
        )
        full_disagreement = (
            dual_stack.std(axis=0) + harm_stack.std(axis=0)
        ).astype(np.float32)
        full_ood = np.stack(ood_members).max(axis=0).astype(np.float32)

        key = f"{domain}__record_key"
        if not np.array_equal(
            oof_arrays[key].astype(str), data["record_key"].astype(str)
        ):
            raise RuntimeError(f"{domain} OOF identity changed")
        oof_dual = oof_arrays[f"{domain}__oof_p_dual"]
        oof_harm = oof_arrays[f"{domain}__oof_p_harm"]
        oof_ood = oof_arrays[f"{domain}__oof_risk_ood"]
        oof_disagreement = oof_arrays[
            f"{domain}__oof_calibrator_disagreement"
        ]
        old_policy = calibration["domains"][domain][
            "deployment_policy"
        ]["policy"]
        oof_challenger, oof_exists = challenger_for_rank(
            prediction,
            oof_dual,
            oof_harm,
            valid,
            harm_weight=float(old_policy["harm_rank_weight"]),
            uncertainty_weight=float(old_policy["uncertainty_rank_weight"]),
        )
        full_challenger, full_exists = challenger_for_rank(
            prediction,
            full_dual,
            full_harm,
            valid,
            harm_weight=float(old_policy["harm_rank_weight"]),
            uncertainty_weight=float(old_policy["uncertainty_rank_weight"]),
        )
        if not oof_exists.all() or not full_exists.all():
            raise RuntimeError(f"{domain} lost a scorer-cal challenger")
        row = np.arange(len(valid))
        mappings = {
            "dual_threshold": mapped_threshold(
                oof_dual[row, oof_challenger],
                full_dual[row, full_challenger],
                float(old_policy["dual_threshold"]),
            ),
            "harm_threshold": mapped_threshold(
                oof_harm[row, oof_challenger],
                full_harm[row, full_challenger],
                float(old_policy["harm_threshold"]),
            ),
            "risk_ood_threshold": mapped_threshold(
                oof_ood[row, oof_challenger],
                full_ood[row, full_challenger],
                float(old_policy["risk_ood_threshold"]),
            ),
            "calibrator_disagreement_threshold": mapped_threshold(
                oof_disagreement[row, oof_challenger],
                full_disagreement[row, full_challenger],
                float(old_policy["calibrator_disagreement_threshold"]),
            ),
        }
        mapped_policy = dict(old_policy)
        for name, value in mappings.items():
            mapped_policy[name] = value["full_threshold"]
        selected = select_challenger(
            prediction,
            full_dual,
            full_harm,
            full_ood,
            full_disagreement,
            full_challenger,
            full_exists,
            type("Policy", (), mapped_policy)(),
        )
        results[domain] = {
            "states": len(valid),
            "groups": int(len(np.unique(group))),
            "old_oof_policy": old_policy,
            "quantile_mappings": mappings,
            "mapped_full_policy": mapped_policy,
            "label_free_scorer_cal_coverage": float(
                (selected != 0).mean()
            ),
            "full_calibrator_checkpoints": full_records,
        }

    report = {
        "schema": "route_robust_v3_full_ensemble_quantile_mapping",
        "status": "deployment_thresholds_mapped_on_scorer_cal",
        "training_result": str(args.training_result),
        "training_result_sha256": file_sha256(args.training_result),
        "calibration_result": str(args.calibration_result),
        "calibration_result_sha256": file_sha256(args.calibration_result),
        "calibration_predictions": str(predictions_path),
        "calibration_predictions_sha256": file_sha256(predictions_path),
        "domains": results,
        "mapping_rule": (
            "preserve each OOF challenger threshold percentile in the "
            "full-ensemble scorer-cal challenger distribution"
        ),
        "outcome_values_used_for_mapping": False,
        "outer_features_accessed": False,
        "outer_labels_accessed": False,
        "cem_cal_accessed": False,
        "fixed_pool_only": True,
        "cem": False,
    }
    write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
