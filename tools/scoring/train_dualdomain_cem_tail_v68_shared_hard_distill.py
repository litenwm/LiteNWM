#!/usr/bin/env python3
"""Fine-tune v66 on hard negatives while distilling original RECON outputs."""

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
    write_json_atomic,
    write_npz_atomic,
)
from tools.scoring import run_train_dualdomain_cem_tail_v66_frozen as v66  # noqa: E402
from tools.scoring import train_dualdomain_cem_tail_v67_head_hard_negative as v67  # noqa: E402
from tools.scoring.run_train_dualdomain_cem_tail_v66_candidate_level import (  # noqa: E402
    choose_candidate_level_cem_frontier,
)
from tools.scoring.train_recon_dense_meta_dinov3_v1 import file_sha256  # noqa: E402
from tools.scoring.train_universal_dualdomain_raw_v2 import (  # noqa: E402
    binary_nll,
    ensemble_predict,
    positive_weight,
    sigmoid,
)


implementation = v66.implementation
BASE_RESULT = implementation.BASE_RESULT
V66_RESULT = implementation.OUTPUT / "result.json"
BASE_CONFIG = implementation.BASE_CONFIG
PREREG = (
    ROOT
    / "outputs/cem_development_v1/"
    "dualdomain_cem_tail_v68_shared_hard_distill_preregistration.json"
)
OUTPUT = (
    ROOT
    / "outputs/cem_development_v1/"
    "dualdomain_cem_tail_v68_shared_hard_distill/training"
)
HARD_MULTIPLIER = 6.0
DISTILL_MULTIPLIER = 2.0
TEMP_GRID = implementation.TEMP_GRID


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def joint_data(envelopes):
    parts = [
        implementation.load_domain_split(domain, "train_fit", envelopes)
        for domain in DOMAINS
    ]
    train = {
        key: np.concatenate([part[key] for part in parts])
        for key in parts[0]
    }
    targets = build_universal_targets(train["targets_raw"])
    scand_targets = build_universal_targets(parts[1]["targets_raw"])
    scand_hard, hard_audit = v67.load_scand_hard_negatives(
        parts[1], scand_targets
    )
    hard = np.concatenate(
        [np.zeros_like(parts[0]["targets_raw"][..., 0], dtype=bool), scand_hard]
    )
    weights = implementation.training_weights(
        train["domain"], train["group"], train["set_type"]
    )
    return train, targets, hard, weights, hard_audit


def load_model(record, device):
    path = Path(record["path"])
    if file_sha256(path) != record["sha256"]:
        raise RuntimeError(f"checkpoint hash changed: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not payload.get("primary_excludes_scorer_v2_predictions"):
        raise RuntimeError(f"raw-only contract missing: {path}")
    model = UniversalDualDomainRawV2(
        UniversalModelConfig(**payload["model_config"])
    ).to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    normalization = {
        key: np.asarray(value, dtype=np.float32)
        for key, value in payload["normalization"].items()
    }
    target_norm = {
        key: np.asarray(value, dtype=np.float32)
        for key, value in payload["target_normalization"].items()
    }
    return payload, model, normalization, target_norm


def local_state_weights(
    base: np.ndarray, domain: np.ndarray, train_index: np.ndarray
) -> np.ndarray:
    result = np.zeros_like(base)
    for domain_index in range(len(DOMAINS)):
        local = train_index[domain[train_index] == domain_index]
        value = base[local]
        result[local] = value * (0.5 / value.sum())
    return result


def distillation_per_state(student, teacher):
    values = []
    for key in ("mean", "logvar"):
        values.append((student[key] - teacher[key]).square().mean(dim=(1, 2)))
    for key in ("helpful_logit", "dual_logit", "harm_logit", "rank_logit"):
        values.append((student[key] - teacher[key]).square().mean(dim=1))
    return torch.stack(values, dim=1).mean(dim=1)


def fine_tune_member(
    *,
    student_initial,
    teacher_initial,
    feature,
    targets,
    domain,
    hard_negative,
    base_weight,
    train_index,
    seed,
    device,
    checkpoint,
    role,
    epochs,
):
    if checkpoint.exists():
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if payload.get("schema") != "dualdomain_cem_tail_v68_member":
            raise RuntimeError(f"unexpected existing checkpoint: {checkpoint}")
        return {
            "path": str(checkpoint),
            "sha256": file_sha256(checkpoint),
            "seed": int(payload["seed"]),
            "member_role": role,
            "train_states": int(payload["train_states"]),
            "resumed": True,
        }
    student_payload, model, normalization, target_norm = load_model(
        student_initial, device
    )
    _, teacher, teacher_normalization, teacher_target_norm = load_model(
        teacher_initial, device
    )
    for key in normalization:
        if not np.array_equal(normalization[key], teacher_normalization[key]):
            raise RuntimeError("student/teacher feature normalization mismatch")
    for key in target_norm:
        if not np.array_equal(target_norm[key], teacher_target_norm[key]):
            raise RuntimeError("student/teacher target normalization mismatch")
    teacher.eval().requires_grad_(False)
    model.train().requires_grad_(True)
    set_seed(seed)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=5.0e-5, weight_decay=1.0e-4
    )
    pos_weight = {
        name: torch.tensor(
            positive_weight(targets[name][train_index]), device=device
        )
        for name in ("helpful", "dual", "harm")
    }
    loss_weights = read_json(BASE_CONFIG)["training"]["loss_weights"]
    state_weight = local_state_weights(base_weight, domain, train_index)
    rng = np.random.default_rng(seed)
    epoch_audit = []
    started = time.perf_counter()
    for epoch in range(epochs):
        order = rng.permutation(train_index)
        model.train()
        total_loss = 0.0
        total_distill = 0.0
        batches = 0
        for start in range(0, len(order), 32):
            index = order[start : start + 32]
            local_domain = domain[index].astype(np.int64)
            x = torch.from_numpy(
                normalize_features(feature[index], local_domain, normalization)
            ).to(device)
            d = torch.from_numpy(local_domain).to(device)
            student = model(x, d)
            with torch.inference_mode():
                teacher_output = teacher(x, d)
            candidate_weight = torch.from_numpy(
                1.0
                + (HARD_MULTIPLIER - 1.0)
                * hard_negative[index].astype(np.float32)
            ).to(device)
            denominator = candidate_weight.sum(dim=1).clamp_min(1.0)
            target_mean = target_norm["target_mean"][local_domain]
            target_scale = target_norm["target_scale"][local_domain]
            expected_delta = torch.from_numpy(
                (targets["delta"][index] - target_mean[:, None])
                / target_scale[:, None]
            ).to(device)
            inverse_variance = torch.exp(-student["logvar"])
            regression = 0.5 * (
                inverse_variance
                * (student["mean"] - expected_delta).square()
                + student["logvar"]
            ).mean(dim=2)
            state_loss = float(loss_weights["regression"]) * (
                regression * candidate_weight
            ).sum(dim=1) / denominator
            for name, logit, key in (
                ("helpful", "helpful_logit", "helpful_bce"),
                ("dual", "dual_logit", "dual_bce"),
                ("harm", "harm_logit", "harm_bce"),
            ):
                expected = torch.from_numpy(targets[name][index]).to(device)
                value = nn.functional.binary_cross_entropy_with_logits(
                    student[logit],
                    expected,
                    pos_weight=pos_weight[name],
                    reduction="none",
                )
                state_loss += float(loss_weights[key]) * (
                    value * candidate_weight
                ).sum(dim=1) / denominator
            pair_score = student["rank_logit"][:, 1:] - student["rank_logit"][:, :1]
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
            distribution = torch.from_numpy(
                targets["list_distribution"][index]
            ).to(device)
            state_loss += float(loss_weights["listwise"]) * (
                -(
                    distribution
                    * torch.log_softmax(student["rank_logit"][:, 1:], dim=1)
                ).sum(dim=1)
            )
            distill = distillation_per_state(student, teacher_output)
            recon = torch.from_numpy((local_domain == 0).astype(np.float32)).to(device)
            state_loss += DISTILL_MULTIPLIER * distill * recon
            sample_weight = torch.from_numpy(
                state_weight[index] * len(train_index)
            ).to(device=device, dtype=state_loss.dtype)
            loss = (state_loss * sample_weight).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.detach().cpu())
            total_distill += float((distill * recon).mean().detach().cpu())
            batches += 1
        record = {
            "member_role": role,
            "epoch": epoch + 1,
            "epochs": epochs,
            "states": int(len(order)),
            "mean_batch_loss": total_loss / max(batches, 1),
            "mean_recon_distillation": total_distill / max(batches, 1),
            "elapsed_sec": time.perf_counter() - started,
        }
        epoch_audit.append(record)
        print(json.dumps(record), flush=True)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "schema": "dualdomain_cem_tail_v68_member",
        "model_config": student_payload["model_config"],
        "normalization": normalization,
        "target_normalization": target_norm,
        "state_dict": model.cpu().state_dict(),
        "primary_excludes_scorer_v2_predictions": True,
        "student_initial_checkpoint": student_initial,
        "teacher_initial_checkpoint": teacher_initial,
        "seed": seed,
        "member_role": role,
        "train_states": int(len(train_index)),
        "epochs": epochs,
        "epoch_audit": epoch_audit,
        "hard_negative_multiplier": HARD_MULTIPLIER,
        "recon_distillation_multiplier": DISTILL_MULTIPLIER,
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


def fit_temperature(logit, target):
    return float(
        min(TEMP_GRID, key=lambda value: binary_nll(logit, target, float(value)))
    )


def calibrate(checkpoints, envelopes, device):
    calibration = {}
    selected_arrays = {}
    contract = {
        "minimum_coverage": 0.4,
        "target_dual_precision": 0.85,
        "target_wilson_lower_95": 0.80,
        "maximum_harm_rate": 0.15,
    }
    for domain in DOMAINS:
        part = implementation.load_domain_split(domain, "scorer_cal", envelopes)
        prediction = ensemble_predict(
            checkpoints, part["feature"], part["domain"], device=device
        )
        targets = build_universal_targets(part["targets_raw"])
        dual_temperature = fit_temperature(
            prediction["dual_logit"], targets["dual"]
        )
        harm_temperature = fit_temperature(
            prediction["harm_logit"], targets["harm"]
        )
        p_dual = sigmoid(prediction["dual_logit"] / dual_temperature)
        p_harm = sigmoid(prediction["harm_logit"] / harm_temperature)
        frontier = choose_candidate_level_cem_frontier(
            {domain: {
                "prediction": prediction,
                "p_dual": p_dual,
                "p_harm": p_harm,
                "valid": part["valid"],
                "targets": part["targets_raw"],
            }},
            contract,
            shared_policy=False,
        )["domains"][domain]
        calibration[domain] = {
            "dual_temperature": dual_temperature,
            "harm_temperature": harm_temperature,
            "policy": frontier["policy"],
            "metrics": frontier["metrics"],
            "feasible": frontier["feasible"],
        }
        selected_arrays[f"{domain}_selected_index"] = frontier["selected"]
    return calibration, selected_arrays


def main() -> None:
    args = parse_args()
    prereg = read_json(PREREG)
    if prereg.get("status") != "frozen_before_v68_full_trial":
        raise RuntimeError("v68 preregistration changed")
    if prereg.get("A_to_H_accessed") or prereg.get("reserve_accessed"):
        raise RuntimeError("v68 preregistration reports forbidden access")
    base = read_json(BASE_RESULT)
    parent = read_json(V66_RESULT)
    if parent.get("status") != "trained_and_frozen_before_A_to_H_replay":
        raise RuntimeError("v66 parent is not frozen")
    envelopes = implementation.load_envelopes(Path(base["support"]))
    train, targets, hard, weights, hard_audit = joint_data(envelopes)
    checkpoints = {"full": [], "oof": []}
    all_index = np.arange(len(train["feature"]), dtype=np.int64)
    for member_index, (student, teacher) in enumerate(
        zip(parent["checkpoints"]["full"], base["arms"]["D"]["checkpoints"]["full"])
    ):
        checkpoints["full"].append(fine_tune_member(
            student_initial=student,
            teacher_initial=teacher,
            feature=train["feature"],
            targets=targets,
            domain=train["domain"],
            hard_negative=hard,
            base_weight=weights,
            train_index=all_index,
            seed=20261110 + member_index,
            device=args.device,
            checkpoint=args.output_dir / "checkpoints" / f"arm_D_full_seed{member_index}.pt",
            role=f"full_seed_{member_index}",
            epochs=args.epochs,
        ))
    calibration, selected_arrays = calibrate(
        checkpoints["full"], envelopes, args.device
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected_path = args.output_dir / "full_trial_selected_index.npz"
    if not selected_path.exists():
        write_npz_atomic(selected_path, **selected_arrays)
    result = {
        "schema": "dualdomain_cem_tail_v68_full_trial_result",
        "status": (
            "full_trial_scorer_cal_feasible"
            if all(value["feasible"] for value in calibration.values())
            else "full_trial_scorer_cal_infeasible_A_to_H_sealed"
        ),
        "checkpoints": checkpoints,
        "train_states": int(len(train["feature"])),
        "hard_negative_audit": hard_audit,
        "calibration": calibration,
        "selected_index": str(selected_path),
        "selected_index_sha256": file_sha256(selected_path),
        "parent_v66_result": str(V66_RESULT),
        "parent_v66_result_sha256": file_sha256(V66_RESULT),
        "preregistration": str(PREREG),
        "preregistration_sha256": file_sha256(PREREG),
        "A_to_H_accessed": False,
        "reserve_accessed": False,
        "production_safe_claim_allowed": False,
    }
    result_path = args.output_dir / "full_trial_result.json"
    if result_path.exists():
        raise FileExistsError(result_path)
    write_json_atomic(result_path, result)
    print(json.dumps({
        "stage": "v68_full_trial",
        "status": result["status"],
        "calibration": calibration,
        "output": str(result_path),
        "sha256": file_sha256(result_path),
    }, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--epochs", type=int, default=5)
    return parser.parse_args()


if __name__ == "__main__":
    main()
