#!/usr/bin/env python3
"""Train final v69 SCAND heads after feasible route-heldout calibration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scoring.universal_dualdomain_raw_v2 import build_universal_targets
from tools.evaluation.export_recon_dense_dinov3_features_v1 import (
    write_json_atomic,
    write_npz_atomic,
)
from tools.scoring import train_dualdomain_cem_tail_v69_scand_route_crossfit as crossfit
from tools.scoring import train_dualdomain_cem_tail_v67_head_hard_negative as head_training
from tools.scoring.train_recon_dense_meta_dinov3_v1 import file_sha256
from tools.scoring.train_universal_dualdomain_raw_v2 import ensemble_predict


def main() -> None:
    args = parse_args()
    result_path = args.output_dir / "training" / "result.json"
    if result_path.exists():
        raise FileExistsError(result_path)
    prereg = crossfit.read_json(crossfit.PREREG)
    if prereg.get("status") != "frozen_before_route_crossfit_training":
        raise RuntimeError("v69 preregistration changed")
    crossfit_path = args.output_dir / "crossfit_result.json"
    calibration_result = crossfit.read_json(crossfit_path)
    if calibration_result.get("status") != "route_crossfit_scorer_cal_feasible":
        raise RuntimeError("route crossfit is not feasible; final training forbidden")
    if calibration_result.get("A_to_H_accessed") or calibration_result.get("reserve_accessed"):
        raise RuntimeError("route crossfit reports forbidden access")
    base = crossfit.read_json(crossfit.BASE_RESULT)
    envelopes = crossfit.implementation.load_envelopes(Path(base["support"]))
    train, adaptation, combined, targets, hard, hard_audit = crossfit.load_combined(
        envelopes
    )
    weights = crossfit.combined_state_weights(
        combined["group"], combined["set_type"]
    )
    all_index = np.arange(len(combined["feature"]), dtype=np.int64)
    records = []
    old_multiplier = head_training.HARD_MULTIPLIER
    head_training.HARD_MULTIPLIER = crossfit.HARD_MULTIPLIER
    try:
        for member_index, initial in enumerate(
            base["arms"]["D"]["checkpoints"]["full"]
        ):
            records.append(head_training.fine_tune_member(
                initial=initial,
                feature=combined["feature"],
                targets=targets,
                hard_negative=hard,
                base_weight=weights,
                train_index=all_index,
                seed=20261310 + member_index,
                device=args.device,
                checkpoint=(
                    args.output_dir
                    / "training/checkpoints"
                    / f"arm_D_full_seed{member_index}.pt"
                ),
                role=f"full_seed_{member_index}",
                epochs=args.epochs,
            ))
    finally:
        head_training.HARD_MULTIPLIER = old_multiplier

    recon = crossfit.implementation.load_domain_split(
        "recon", "scorer_cal", envelopes
    )
    original_recon = ensemble_predict(
        base["arms"]["D"]["checkpoints"]["full"],
        recon["feature"],
        recon["domain"],
        device=args.device,
    )
    final_recon = crossfit.predict_crossfit_ensemble(
        records, recon["feature"], recon["domain"], device=args.device
    )
    recon_max_error = {
        key: float(
            np.abs(final_recon[key] - original_recon[key]).max(initial=0.0)
        )
        for key in original_recon
    }
    if any(value != 0.0 for value in recon_max_error.values()):
        raise RuntimeError(f"final RECON output changed: {recon_max_error}")
    recon_policy = base["arms"]["D"]["calibration"]["frontier"]["domains"][
        "recon"
    ]["policy"]
    recon_temperature = base["arms"]["D"]["temperature"]["domains"]["recon"]
    recon_metrics, recon_selected = head_training.exact_policy_metrics(
        final_recon,
        recon["targets_raw"],
        recon["valid"],
        recon_policy,
        float(recon_temperature["deployment_dual_temperature"]),
        float(recon_temperature["deployment_harm_temperature"]),
    )

    frozen_scand = calibration_result["calibration"]
    final_scand_prediction = crossfit.predict_crossfit_ensemble(
        records,
        adaptation["feature"],
        adaptation["domain"],
        device=args.device,
    )
    final_scand_metrics, final_scand_selected = head_training.exact_policy_metrics(
        final_scand_prediction,
        adaptation["targets_raw"],
        adaptation["valid"],
        frozen_scand["policy"],
        float(frozen_scand["dual_temperature"]),
        float(frozen_scand["harm_temperature"]),
    )
    selected_path = args.output_dir / "training/calibration_selected_index.npz"
    write_npz_atomic(
        selected_path,
        recon_selected_index=recon_selected,
        scand_route_crossfit_selected_index=crossfit.load_npz(
            calibration_result["crossfit_oof_predictions"]
        )["selected_index"],
        scand_final_in_sample_selected_index=final_scand_selected,
    )
    calibration = {
        "recon": {
            "dual_temperature": float(
                recon_temperature["deployment_dual_temperature"]
            ),
            "harm_temperature": float(
                recon_temperature["deployment_harm_temperature"]
            ),
            "policy": recon_policy,
            "metrics": recon_metrics,
            "feasible": True,
            "policy_source": "bit-exact original Arm-D RECON policy",
        },
        "scand": {
            "dual_temperature": float(frozen_scand["dual_temperature"]),
            "harm_temperature": float(frozen_scand["harm_temperature"]),
            "policy": frozen_scand["policy"],
            "metrics": frozen_scand["metrics"],
            "feasible": True,
            "policy_source": "five-route heldout crossfit frozen before final training",
            "route_crossfit_feasible_policy_count": frozen_scand[
                "feasible_policy_count"
            ],
            "final_model_in_sample_diagnostic_only": final_scand_metrics,
        },
    }
    result = {
        "schema": "dualdomain_cem_tail_v69_scand_route_crossfit_training_result",
        "status": "trained_and_frozen_before_A_to_H_replay",
        "checkpoints": {
            "full": records,
            "oof": [],
            "calibration_route_crossfit": calibration_result["folds"],
        },
        "train_states": int(len(combined["feature"])),
        "train_fit_states": int(len(train["feature"])),
        "adaptation_states": int(len(adaptation["feature"])),
        "hard_negative_audit": hard_audit,
        "calibration": calibration,
        "calibration_selected_index": str(selected_path),
        "calibration_selected_index_sha256": file_sha256(selected_path),
        "recon_output_bit_exact": True,
        "recon_output_max_abs_error_vs_original": recon_max_error,
        "route_crossfit_result": str(crossfit_path),
        "route_crossfit_result_sha256": file_sha256(crossfit_path),
        "base_result": str(crossfit.BASE_RESULT),
        "base_result_sha256": file_sha256(crossfit.BASE_RESULT),
        "preregistration": str(crossfit.PREREG),
        "preregistration_sha256": file_sha256(crossfit.PREREG),
        "calibration_decision": "candidate-level safety mask then lowest conservative energy",
        "scorer_cal_independent_evidence_claim_allowed": False,
        "A_to_H_accessed": False,
        "reserve_accessed": False,
        "production_safe_claim_allowed": False,
    }
    write_json_atomic(result_path, result)
    print(json.dumps({
        "stage": "v69_finalize",
        "status": result["status"],
        "calibration": calibration,
        "recon_output_bit_exact": True,
        "output": str(result_path),
        "sha256": file_sha256(result_path),
    }, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=crossfit.OUTPUT)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--epochs", type=int, default=5)
    return parser.parse_args()


if __name__ == "__main__":
    main()
