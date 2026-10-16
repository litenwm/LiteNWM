#!/usr/bin/env python3
"""Map v5 OOF policy quantiles onto its full calibrator ensemble."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from scoring.route_multimodal_calibration_v5 import (  # noqa: E402
    RouteMultimodalCalibratorV5,
    calibrated_probabilities_v5,
    multimodal_risk_features,
)
from scoring.route_robust_calibration_v3 import (  # noqa: E402
    RouteRiskPolicyV3,
    risk_features,
    risk_nonconformity,
)
from scoring.spacing_support_v1 import threshold_for_quantile  # noqa: E402
from scoring.universal_dualdomain_raw_v2 import build_universal_features  # noqa: E402
from tools.scoring.calibrate_recon_route_expanded_v4 import (  # noqa: E402
    cohort_metrics,
    load_development_data,
)
from tools.scoring.calibrate_universal_route_robust_v3 import (  # noqa: E402
    challenger_for_rank,
    ensemble_with_members,
    select_challenger,
)
from tools.scoring.finetune_universal_dualdomain_safe_rank_v2 import (  # noqa: E402
    load_envelopes,
)
from tools.scoring.train_recon_dense_meta_dinov3_v1 import (  # noqa: E402
    file_sha256,
    write_json,
)
from tools.scoring.train_universal_dualdomain_raw_v2 import (  # noqa: E402
    safety_metrics,
)


ROOT = REPO_ROOT / "outputs/universal_dualdomain_raw_v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--calibration-result",
        type=Path,
        default=ROOT / "recon_route_multimodal_v5/result.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "recon_route_multimodal_v5/deployment_mapping.json",
    )
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument(
        "--scorer-cal-features",
        type=Path,
        default=ROOT
        / "features/recon_scorer_cal/recon_scorer_cal_universal_v2_features.npz",
    )
    parser.add_argument(
        "--scorer-cal-source",
        type=Path,
        default=ROOT / "sources/recon_scorer_cal_source.npz",
    )
    parser.add_argument(
        "--spent-features",
        type=Path,
        nargs=2,
        default=[
            ROOT
            / "recon_fresh_outer_v2/features/"
            "recon_fresh_outer_h4_100_seed20260721_features.npz",
            ROOT
            / "recon_fresh_outer_confirm_v2/features/"
            "recon_fresh_outer_confirm_h4_100_seed20260722_features.npz",
        ],
    )
    parser.add_argument(
        "--spent-outcomes",
        type=Path,
        nargs=2,
        default=[
            ROOT / "recon_fresh_outer_v2/fresh_outer_outcomes.npz",
            ROOT / "recon_fresh_outer_confirm_v2/fresh_outer_outcomes.npz",
        ],
    )
    return parser.parse_args()


def load_calibrator(
    path: Path, device: str
) -> tuple[RouteMultimodalCalibratorV5, dict[str, np.ndarray]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("schema") != "route_multimodal_risk_calibrator_v5":
        raise RuntimeError(f"unexpected calibrator schema: {path}")
    model = RouteMultimodalCalibratorV5(
        input_dim=int(payload["input_dim"]),
        hidden_dim=int(payload["hidden_dim"]),
        dropout=float(payload["dropout"]),
    ).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    stats = {
        key: np.asarray(value, dtype=np.float32)
        for key, value in payload["normalization"].items()
    }
    return model, stats


def threshold_quantile(reference: np.ndarray, threshold: float) -> float:
    values = np.asarray(reference, dtype=np.float64)
    if not len(values) or not np.isfinite(values).all():
        raise RuntimeError("invalid deployment mapping reference")
    return float(np.mean(values <= float(threshold)))


def mapped_threshold(reference: np.ndarray, quantile: float) -> float:
    return float(np.quantile(np.asarray(reference), quantile))


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    calibration = json.loads(
        args.calibration_result.read_text(encoding="utf-8")
    )
    if calibration.get("schema") != "recon_route_multimodal_calibration_v5":
        raise RuntimeError("mapping requires the frozen v5 development result")
    if calibration.get("qualification_v4_accessed") is not False:
        raise RuntimeError("qualification_v4 must remain sealed")
    training_path = Path(calibration["base_training_result"])
    training = json.loads(training_path.read_text(encoding="utf-8"))
    datasets, leakage_audit = load_development_data(args)
    envelopes = load_envelopes(Path(training["support"]))

    universal_parts = []
    support_parts = []
    valid_parts = []
    targets_parts = []
    cohort_parts = []
    for data in datasets:
        universal, support, guard = build_universal_features(
            data,
            spacing_m=0.25,
            support_envelope=envelopes["recon"],
        )
        valid = guard & (
            support
            <= threshold_for_quantile(envelopes["recon"], 0.999)
        )
        valid[:, 0] = True
        universal_parts.append(universal)
        support_parts.append(support)
        valid_parts.append(valid)
        targets_parts.append(data["targets"][..., :4].astype(np.float32))
        cohort_parts.append(data["cohort"].astype(str))
    universal = np.concatenate(universal_parts)
    support = np.concatenate(support_parts)
    valid = np.concatenate(valid_parts)
    targets = np.concatenate(targets_parts)
    cohort = np.concatenate(cohort_parts)

    prediction, member_prediction = ensemble_with_members(
        training["arms"]["D"]["checkpoints"]["full"],
        universal,
        np.zeros(len(universal), dtype=np.int64),
        device=args.device,
    )
    base_risk = risk_features(
        prediction,
        member_prediction,
        support,
        robot_context=np.zeros((len(universal), 2), dtype=np.float32),
    )
    risk = multimodal_risk_features(base_risk, universal)

    dual_members = []
    harm_members = []
    ood_members = []
    checkpoint_audit = []
    for record in calibration["checkpoints"]["full"]:
        path = Path(record["path"])
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
    full_dual = np.clip(
        dual_stack.mean(0) - dual_stack.std(0), 0.0, 1.0
    )
    full_harm = np.clip(
        harm_stack.mean(0) + harm_stack.std(0), 0.0, 1.0
    )
    full_disagreement = dual_stack.std(0) + harm_stack.std(0)
    full_ood = np.stack(ood_members).max(0)

    with np.load(
        calibration["calibration_predictions"], allow_pickle=False
    ) as payload:
        oof_dual = payload["oof_p_dual"].astype(np.float32)
        oof_harm = payload["oof_p_harm"].astype(np.float32)
        oof_ood = payload["oof_risk_ood"].astype(np.float32)
        oof_disagreement = payload[
            "oof_calibrator_disagreement"
        ].astype(np.float32)
    policy = RouteRiskPolicyV3(
        **calibration["deployment_policy"]["policy"]
    )
    oof_challenger, oof_exists = challenger_for_rank(
        prediction,
        oof_dual,
        oof_harm,
        valid,
        harm_weight=policy.harm_rank_weight,
        uncertainty_weight=policy.uncertainty_rank_weight,
    )
    full_challenger, full_exists = challenger_for_rank(
        prediction,
        full_dual,
        full_harm,
        valid,
        harm_weight=policy.harm_rank_weight,
        uncertainty_weight=policy.uncertainty_rank_weight,
    )
    row = np.arange(len(valid))
    quantiles = {
        "dual_threshold": threshold_quantile(
            oof_dual[row, oof_challenger][oof_exists],
            policy.dual_threshold,
        ),
        "harm_threshold": threshold_quantile(
            oof_harm[row, oof_challenger][oof_exists],
            policy.harm_threshold,
        ),
        "risk_ood_threshold": threshold_quantile(
            oof_ood[row, oof_challenger][oof_exists],
            policy.risk_ood_threshold,
        ),
        "calibrator_disagreement_threshold": threshold_quantile(
            oof_disagreement[row, oof_challenger][oof_exists],
            policy.calibrator_disagreement_threshold,
        ),
    }
    full_reference = {
        "dual_threshold": full_dual[row, full_challenger][full_exists],
        "harm_threshold": full_harm[row, full_challenger][full_exists],
        "risk_ood_threshold": full_ood[row, full_challenger][full_exists],
        "calibrator_disagreement_threshold": full_disagreement[
            row, full_challenger
        ][full_exists],
    }
    mapped = {
        **calibration["deployment_policy"]["policy"],
        **{
            key: mapped_threshold(full_reference[key], value)
            for key, value in quantiles.items()
        },
    }
    mapped_policy = RouteRiskPolicyV3(**mapped)
    selected = select_challenger(
        prediction,
        full_dual,
        full_harm,
        full_ood,
        full_disagreement,
        full_challenger,
        full_exists,
        mapped_policy,
    )
    report = {
        "schema": "recon_route_multimodal_v5_deployment_mapping",
        "status": "frozen_calibration_quantile_mapping",
        "evidence_role": "development_only",
        "calibration_result": str(args.calibration_result),
        "calibration_result_sha256": file_sha256(args.calibration_result),
        "base_training_result": str(training_path),
        "base_training_result_sha256": file_sha256(training_path),
        "full_checkpoints": checkpoint_audit,
        "oof_policy": calibration["deployment_policy"]["policy"],
        "oof_to_full_quantiles": quantiles,
        "mapped_full_policy": mapped,
        "development_full_mapped": {
            "metrics": safety_metrics(selected, targets),
            "cohort_metrics": cohort_metrics(
                selected, targets, cohort
            ),
        },
        "data_and_leakage_audit": leakage_audit,
        "mapping_contract": (
            "threshold percentile transferred from group-OOF challenger "
            "scores to full-ensemble challenger scores; no outer covariate"
        ),
        "qualification_v4_accessed": False,
        "qualification_v4_covariates_used": False,
        "outer_outcomes_used": False,
        "fixed_pool_only": True,
        "cem": False,
    }
    write_json(args.output, report)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "sha256": file_sha256(args.output),
                "mapped_full_policy": mapped,
                "development_full_mapped": report[
                    "development_full_mapped"
                ],
                "qualification_v4_accessed": False,
                "cem": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
