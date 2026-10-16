#!/usr/bin/env python3
"""Minimal v67: train only the SCAND head on frozen false-safe hard negatives."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scoring.universal_dualdomain_raw_v2 import (  # noqa: E402
    DOMAINS,
    UniversalDualDomainRawV2,
    UniversalModelConfig,
    build_universal_targets,
    normalize_features,
    set_seed,
)
from tools.evaluation.export_recon_dense_dinov3_features_v1 import (  # noqa: E402
    load_npz,
    write_json_atomic,
    write_npz_atomic,
)
from tools.planning.run_cem_safe_elite_v55 import (  # noqa: E402
    candidate_safety_mask,
    select_lowest_energy_safe,
)
from tools.scoring import run_train_dualdomain_cem_tail_v66_frozen as v66  # noqa: E402
from tools.scoring.run_train_dualdomain_cem_tail_v66_candidate_level import (  # noqa: E402
    candidate_energy,
    choose_candidate_level_cem_frontier,
)
from tools.scoring.train_recon_dense_meta_dinov3_v1 import file_sha256  # noqa: E402
from tools.scoring.train_universal_dualdomain_raw_v2 import (  # noqa: E402
    binary_nll,
    ensemble_predict,
    positive_weight,
    sigmoid,
    safety_metrics,
    policy_feasible,
)


implementation = v66.implementation
BASE_RESULT = implementation.BASE_RESULT
BASE_CONFIG = implementation.BASE_CONFIG
PREREG = (
    ROOT
    / "outputs/cem_development_v1/"
    "dualdomain_cem_tail_v67_head_hard_negative_preregistration.json"
)
OUTPUT = (
    ROOT
    / "outputs/cem_development_v1/"
    "dualdomain_cem_tail_v67_head_hard_negative/training"
)
HARD_MULTIPLIER = 6.0
SET_WEIGHTS = implementation.SET_WEIGHTS
TEMP_GRID = implementation.TEMP_GRID


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def scand_state_weights(group: np.ndarray, set_type: np.ndarray) -> np.ndarray:
    result = np.zeros(len(group), dtype=np.float64)
    group = group.astype(str)
    set_type = set_type.astype(str)
    for name, total in SET_WEIGHTS.items():
        mask = set_type == name
        routes = np.unique(group[mask])
        if not len(routes):
            raise RuntimeError(f"missing SCAND set type {name}")
        for route in routes:
            local = mask & (group == route)
            result[local] = float(total) / len(routes) / int(local.sum())
    if not np.isclose(result.sum(), 1.0):
        raise RuntimeError(f"SCAND state weights sum to {result.sum()}")
    return result


def load_scand_hard_negatives(
    train: dict[str, np.ndarray], targets: dict[str, np.ndarray]
) -> tuple[np.ndarray, dict[str, Any]]:
    states = len(train["feature"]) // 4
    hard = np.zeros(targets["harm"].shape, dtype=bool)
    per_iteration = []
    for iteration in range(3):
        score_path = (
            implementation.HARVEST
            / "scand_train_fit"
            / f"iteration_{iteration:02d}/scores.npz"
        )
        score = load_npz(score_path)
        start = (iteration + 1) * states
        stop = start + states
        local = (
            score["safe"].astype(bool)
            & targets["harm"][start:stop].astype(bool)
        )
        local[:, 0] = False
        hard[start:stop] = local
        per_iteration.append({
            "iteration": iteration,
            "original_safe_candidates": int(score["safe"][:, 1:].sum()),
            "false_safe_harm_candidates": int(local[:, 1:].sum()),
            "score_sha256": file_sha256(score_path),
        })
    audit = {
        "definition": "original route-OOF safe AND true harm",
        "multiplier": HARD_MULTIPLIER,
        "total": int(hard.sum()),
        "effective_extra_candidate_mass": float((HARD_MULTIPLIER - 1.0) * hard.sum()),
        "per_iteration": per_iteration,
    }
    return hard, audit


def member_state_weights(base: np.ndarray, train_index: np.ndarray) -> np.ndarray:
    result = np.zeros_like(base)
    local = base[train_index]
    result[train_index] = local / local.sum()
    return result


def fine_tune_member(
    *,
    initial: dict[str, Any],
    feature: np.ndarray,
    targets: dict[str, np.ndarray],
    hard_negative: np.ndarray,
    base_weight: np.ndarray,
    train_index: np.ndarray,
    seed: int,
    device: str,
    checkpoint: Path,
    role: str,
    epochs: int,
) -> dict[str, Any]:
    if checkpoint.exists():
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if payload.get("schema") != "dualdomain_cem_tail_v67_scand_head_member":
            raise RuntimeError(f"unexpected existing checkpoint: {checkpoint}")
        return {
            "path": str(checkpoint),
            "sha256": file_sha256(checkpoint),
            "seed": int(payload["seed"]),
            "member_role": role,
            "train_states": int(payload["train_states"]),
            "resumed": True,
        }
    initial_path = Path(initial["path"])
    if file_sha256(initial_path) != initial["sha256"]:
        raise RuntimeError(f"initial checkpoint changed: {initial_path}")
    payload = torch.load(initial_path, map_location="cpu", weights_only=False)
    config = UniversalModelConfig(**payload["model_config"])
    model = UniversalDualDomainRawV2(config).to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.heads[DOMAINS.index("scand")].parameters():
        parameter.requires_grad_(True)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    normalization = {
        key: np.asarray(value, dtype=np.float32)
        for key, value in payload["normalization"].items()
    }
    target_norm = {
        key: np.asarray(value, dtype=np.float32)
        for key, value in payload["target_normalization"].items()
    }
    set_seed(seed)
    optimizer = torch.optim.AdamW(trainable, lr=1.0e-4, weight_decay=1.0e-4)
    pos_weight = {
        name: torch.tensor(positive_weight(targets[name][train_index]), device=device)
        for name in ("helpful", "dual", "harm")
    }
    loss_weights = read_json(BASE_CONFIG)["training"]["loss_weights"]
    local_weight = member_state_weights(base_weight, train_index)
    rng = np.random.default_rng(seed)
    domain_id = DOMAINS.index("scand")
    epoch_audit = []
    started = time.perf_counter()
    for epoch in range(epochs):
        order = rng.permutation(train_index)
        model.train()
        total_loss = 0.0
        batches = 0
        for start in range(0, len(order), 32):
            index = order[start : start + 32]
            local_domain = np.full(len(index), domain_id, dtype=np.int64)
            x = torch.from_numpy(
                normalize_features(feature[index], local_domain, normalization)
            ).to(device)
            d = torch.from_numpy(local_domain).to(device)
            output = model(x, d)
            candidate_weight = torch.from_numpy(
                1.0
                + (HARD_MULTIPLIER - 1.0)
                * hard_negative[index].astype(np.float32)
            ).to(device)
            candidate_denominator = candidate_weight.sum(dim=1).clamp_min(1.0)
            target_mean = target_norm["target_mean"][local_domain]
            target_scale = target_norm["target_scale"][local_domain]
            expected_delta = torch.from_numpy(
                (targets["delta"][index] - target_mean[:, None])
                / target_scale[:, None]
            ).to(device)
            inverse_variance = torch.exp(-output["logvar"])
            regression = 0.5 * (
                inverse_variance * (output["mean"] - expected_delta).square()
                + output["logvar"]
            ).mean(dim=2)
            state_loss = float(loss_weights["regression"]) * (
                regression * candidate_weight
            ).sum(dim=1) / candidate_denominator
            for name, logit, key in (
                ("helpful", "helpful_logit", "helpful_bce"),
                ("dual", "dual_logit", "dual_bce"),
                ("harm", "harm_logit", "harm_bce"),
            ):
                expected = torch.from_numpy(targets[name][index]).to(device)
                value = nn.functional.binary_cross_entropy_with_logits(
                    output[logit],
                    expected,
                    pos_weight=pos_weight[name],
                    reduction="none",
                )
                state_loss += float(loss_weights[key]) * (
                    value * candidate_weight
                ).sum(dim=1) / candidate_denominator
            pair_score = output["rank_logit"][:, 1:] - output["rank_logit"][:, :1]
            pair_dual = torch.from_numpy(targets["dual"][index, 1:]).to(device)
            pair_harm = torch.from_numpy(targets["harm"][index, 1:]).to(device)
            pair_loss = nn.functional.softplus(
                -(pair_dual * 2.0 - 1.0) * pair_score
            )
            pair_multiplier = 1.0 + pair_harm * (
                float(loss_weights["harmful_pair_multiplier"]) - 1.0
            )
            hard_pair = candidate_weight[:, 1:]
            state_loss += float(loss_weights["incumbent_pairwise"]) * (
                pair_loss * pair_multiplier * hard_pair
            ).sum(dim=1) / hard_pair.sum(dim=1).clamp_min(1.0)
            distribution = torch.from_numpy(targets["list_distribution"][index]).to(device)
            state_loss += float(loss_weights["listwise"]) * (
                -(distribution * torch.log_softmax(output["rank_logit"][:, 1:], dim=1)).sum(dim=1)
            )
            sample_weight = torch.from_numpy(
                local_weight[index] * len(train_index)
            ).to(device=device, dtype=state_loss.dtype)
            loss = (state_loss * sample_weight).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 5.0)
            optimizer.step()
            total_loss += float(loss.detach().cpu())
            batches += 1
        if epoch == 0 or (epoch + 1) % 5 == 0 or epoch + 1 == epochs:
            record = {
                "member_role": role,
                "epoch": epoch + 1,
                "epochs": epochs,
                "states": int(len(order)),
                "mean_batch_loss": total_loss / max(batches, 1),
                "elapsed_sec": time.perf_counter() - started,
            }
            epoch_audit.append(record)
            print(json.dumps(record), flush=True)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "schema": "dualdomain_cem_tail_v67_scand_head_member",
        "model_config": payload["model_config"],
        "normalization": normalization,
        "target_normalization": target_norm,
        "state_dict": model.cpu().state_dict(),
        "initial_checkpoint": str(initial_path),
        "initial_checkpoint_sha256": initial["sha256"],
        "seed": seed,
        "member_role": role,
        "train_states": int(len(train_index)),
        "epochs": epochs,
        "epoch_audit": epoch_audit,
        "trainable_parameters": "SCAND domain-specific head only",
        "hard_negative_multiplier": HARD_MULTIPLIER,
        "A_to_H_accessed": False,
        "reserve_accessed": False,
    }, checkpoint)
    return {
        "path": str(checkpoint),
        "sha256": file_sha256(checkpoint),
        "seed": seed,
        "member_role": role,
        "train_states": int(len(train_index)),
        "resumed": False,
    }


def fit_temperature(logit: np.ndarray, target: np.ndarray) -> float:
    return float(min(TEMP_GRID, key=lambda value: binary_nll(logit, target, float(value))))


def exact_policy_metrics(
    prediction: dict[str, np.ndarray],
    targets_raw: np.ndarray,
    valid: np.ndarray,
    policy: dict[str, float],
    dual_temperature: float,
    harm_temperature: float,
) -> tuple[dict[str, Any], np.ndarray]:
    p_dual = sigmoid(prediction["dual_logit"] / dual_temperature)
    p_harm = sigmoid(prediction["harm_logit"] / harm_temperature)
    energy = candidate_energy(prediction, p_dual, p_harm, valid, policy)
    safe = candidate_safety_mask(prediction, p_dual, p_harm, valid, policy)
    selected = select_lowest_energy_safe(energy, safe)
    return safety_metrics(selected, targets_raw), selected


def main() -> None:
    args = parse_args()
    result_path = args.output_dir / "result.json"
    if result_path.exists():
        raise FileExistsError(result_path)
    prereg = read_json(PREREG)
    if prereg.get("status") != "frozen_before_v67_training":
        raise RuntimeError("v67 preregistration changed")
    if prereg.get("A_to_H_accessed") or prereg.get("reserve_accessed"):
        raise RuntimeError("v67 preregistration reports forbidden access")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    base = read_json(BASE_RESULT)
    envelopes = implementation.load_envelopes(Path(base["support"]))
    train = implementation.load_domain_split("scand", "train_fit", envelopes)
    targets = build_universal_targets(train["targets_raw"])
    hard_negative, hard_audit = load_scand_hard_negatives(train, targets)
    base_weight = scand_state_weights(train["group"], train["set_type"])
    checkpoints = {"oof": [], "full": []}
    base_arm = base["arms"]["D"]
    for heldout, initial in enumerate(base_arm["checkpoints"]["oof"]):
        index = np.flatnonzero(train["fold_id"] != heldout)
        checkpoints["oof"].append(fine_tune_member(
            initial=initial,
            feature=train["feature"],
            targets=targets,
            hard_negative=hard_negative,
            base_weight=base_weight,
            train_index=index,
            seed=20260910 + heldout,
            device=args.device,
            checkpoint=args.output_dir / "checkpoints" / f"arm_D_oof_fold{heldout}.pt",
            role=f"oof_fold_{heldout}",
            epochs=args.epochs,
        ))
    all_index = np.arange(len(train["feature"]), dtype=np.int64)
    for member_index, initial in enumerate(base_arm["checkpoints"]["full"]):
        checkpoints["full"].append(fine_tune_member(
            initial=initial,
            feature=train["feature"],
            targets=targets,
            hard_negative=hard_negative,
            base_weight=base_weight,
            train_index=all_index,
            seed=20261010 + member_index,
            device=args.device,
            checkpoint=args.output_dir / "checkpoints" / f"arm_D_full_seed{member_index}.pt",
            role=f"full_seed_{member_index}",
            epochs=args.epochs,
        ))

    calibration_contract = {
        "minimum_coverage": 0.4,
        "target_dual_precision": 0.85,
        "target_wilson_lower_95": 0.80,
        "maximum_harm_rate": 0.15,
    }
    selected_arrays = {}
    calibration = {}

    recon = implementation.load_domain_split("recon", "scorer_cal", envelopes)
    original_recon_prediction = ensemble_predict(
        base_arm["checkpoints"]["full"], recon["feature"], recon["domain"], device=args.device
    )
    v67_recon_prediction = ensemble_predict(
        checkpoints["full"], recon["feature"], recon["domain"], device=args.device
    )
    recon_exact = {
        key: float(np.abs(v67_recon_prediction[key] - original_recon_prediction[key]).max(initial=0.0))
        for key in original_recon_prediction
    }
    if any(value != 0.0 for value in recon_exact.values()):
        raise RuntimeError(f"frozen RECON output changed: {recon_exact}")
    recon_policy = base_arm["calibration"]["frontier"]["domains"]["recon"]["policy"]
    recon_temperature = base_arm["temperature"]["domains"]["recon"]
    recon_metrics, recon_selected = exact_policy_metrics(
        v67_recon_prediction,
        recon["targets_raw"],
        recon["valid"],
        recon_policy,
        float(recon_temperature["deployment_dual_temperature"]),
        float(recon_temperature["deployment_harm_temperature"]),
    )
    calibration["recon"] = {
        "dual_temperature": float(recon_temperature["deployment_dual_temperature"]),
        "harm_temperature": float(recon_temperature["deployment_harm_temperature"]),
        "policy": recon_policy,
        "metrics": recon_metrics,
        "feasible": policy_feasible({"recon": recon_metrics}, calibration_contract),
        "policy_source": "bit-exact original Arm-D RECON policy",
    }
    selected_arrays["recon_selected_index"] = recon_selected

    scand = implementation.load_domain_split("scand", "scorer_cal", envelopes)
    scand_prediction = ensemble_predict(
        checkpoints["full"], scand["feature"], scand["domain"], device=args.device
    )
    scand_targets = build_universal_targets(scand["targets_raw"])
    dual_temperature = fit_temperature(scand_prediction["dual_logit"], scand_targets["dual"])
    harm_temperature = fit_temperature(scand_prediction["harm_logit"], scand_targets["harm"])
    p_dual = sigmoid(scand_prediction["dual_logit"] / dual_temperature)
    p_harm = sigmoid(scand_prediction["harm_logit"] / harm_temperature)
    frontier = choose_candidate_level_cem_frontier(
        {"scand": {
            "prediction": scand_prediction,
            "p_dual": p_dual,
            "p_harm": p_harm,
            "valid": scand["valid"],
            "targets": scand["targets_raw"],
        }},
        calibration_contract,
        shared_policy=False,
    )["domains"]["scand"]
    calibration["scand"] = {
        "dual_temperature": dual_temperature,
        "harm_temperature": harm_temperature,
        "policy": frontier["policy"],
        "metrics": frontier["metrics"],
        "feasible": frontier["feasible"],
        "policy_source": "candidate-level scorer-cal frontier",
    }
    selected_arrays["scand_selected_index"] = frontier["selected"]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected_path = args.output_dir / "calibration_selected_index.npz"
    write_npz_atomic(selected_path, **selected_arrays)
    both_feasible = bool(calibration["recon"]["feasible"] and calibration["scand"]["feasible"])
    result = {
        "schema": "dualdomain_cem_tail_v67_head_hard_negative_training_result",
        "status": (
            "trained_and_frozen_before_A_to_H_replay"
            if both_feasible
            else "scorer_cal_infeasible_A_to_H_sealed"
        ),
        "checkpoints": checkpoints,
        "train_states": int(len(train["feature"])),
        "hard_negative_audit": hard_audit,
        "calibration": calibration,
        "calibration_selected_index": str(selected_path),
        "calibration_selected_index_sha256": file_sha256(selected_path),
        "recon_output_max_abs_error_vs_original": recon_exact,
        "recon_output_bit_exact": True,
        "base_result": str(BASE_RESULT),
        "base_result_sha256": file_sha256(BASE_RESULT),
        "preregistration": str(PREREG),
        "preregistration_sha256": file_sha256(PREREG),
        "calibration_decision": "candidate-level safety mask then lowest conservative energy",
        "A_to_H_accessed": False,
        "reserve_accessed": False,
        "production_safe_claim_allowed": False,
    }
    write_json_atomic(result_path, result)
    print(json.dumps({
        "stage": "train_v67",
        "status": result["status"],
        "hard_negative_audit": hard_audit,
        "calibration": calibration,
        "recon_output_bit_exact": True,
        "output": str(result_path),
        "sha256": file_sha256(result_path),
    }, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--epochs", type=int, default=10)
    return parser.parse_args()


if __name__ == "__main__":
    main()
