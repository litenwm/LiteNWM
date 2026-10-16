#!/usr/bin/env python3
"""Fine-tune Arm D on fixed K17 plus route-OOF CEM proposal populations."""

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

from scoring.spacing_support_v1 import threshold_for_quantile  # noqa: E402
from scoring.universal_dualdomain_raw_v2 import (  # noqa: E402
    DOMAINS,
    UniversalDualDomainRawV2,
    UniversalModelConfig,
    build_universal_features,
    build_universal_targets,
    normalize_features,
    set_seed,
)
from tools.evaluation.export_recon_dense_dinov3_features_v1 import (  # noqa: E402
    load_npz,
    write_json_atomic,
    write_npz_atomic,
)
from tools.scoring.finetune_universal_dualdomain_safe_rank_v2 import load_envelopes  # noqa: E402
from tools.scoring.harvest_dualdomain_cem_tail_v66 import paths as source_paths  # noqa: E402
from tools.scoring.train_recon_dense_meta_dinov3_v1 import file_sha256  # noqa: E402
from tools.scoring.train_universal_dualdomain_raw_v2 import (  # noqa: E402
    binary_nll,
    choose_frontier,
    ensemble_predict,
    load_member,
    positive_weight,
    predict_member,
    sigmoid,
)


BASE_RESULT = ROOT / "outputs/universal_dualdomain_raw_v2/training_v2/result.json"
BASE_CONFIG = ROOT / "configs/scoring/universal_dualdomain_raw_v2.json"
PREREG = ROOT / "outputs/cem_development_v1/dualdomain_cem_tail_v66_preregistration.json"
WEIGHT_AMENDMENT = ROOT / "outputs/cem_development_v1/dualdomain_cem_tail_v66_training_weight_amendment.json"
HARVEST = ROOT / "outputs/cem_development_v1/dualdomain_cem_tail_v66/harvest"
OUTPUT = ROOT / "outputs/cem_development_v1/dualdomain_cem_tail_v66/training"
SET_WEIGHTS = {"fixed": 0.5, "cem_iteration_0": 1.0 / 6.0, "cem_iteration_1": 1.0 / 6.0, "cem_iteration_2": 1.0 / 6.0}
TEMP_GRID = np.asarray((0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0), dtype=np.float64)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_domain_split(domain: str, split: str, envelopes: dict[str, dict[str, Any]]) -> dict[str, np.ndarray]:
    base_path = source_paths(domain, split)["feature"]
    source_path = source_paths(domain, split)["source"]
    base = load_npz(base_path)
    source = load_npz(source_path)
    if not np.array_equal(base["record_key"].astype(str), source["record_key"].astype(str)):
        raise RuntimeError(f"{domain}/{split}: base/source identity mismatch")
    features = []
    targets = []
    valid = []
    groups = []
    set_types = []
    fold_ids = []
    spacing = 0.25 if domain == "recon" else 0.38
    threshold = threshold_for_quantile(envelopes[domain], 0.999)
    domain_index = DOMAINS.index(domain)
    raw_sets = [("fixed", base, base["targets"].astype(np.float32), None)]
    harvest_root = HARVEST / f"{domain}_{split}"
    harvest_result = read_json(harvest_root / "result.json")
    if harvest_result.get("status") != "complete" or harvest_result.get("reserve_accessed"):
        raise RuntimeError(f"{domain}/{split}: harvest contract failed")
    for iteration in range(3):
        directory = harvest_root / f"iteration_{iteration:02d}"
        raw = load_npz(directory / "features.npz")
        label = load_npz(directory / "targets.npz")
        if not np.array_equal(raw["record_key"].astype(str), source["record_key"].astype(str)) or not np.array_equal(label["record_key"].astype(str), source["record_key"].astype(str)):
            raise RuntimeError(f"{domain}/{split}/iteration{iteration}: identity mismatch")
        raw_sets.append((f"cem_iteration_{iteration}", raw, label["targets"].astype(np.float32), label["fold_id"].astype(np.int64)))
    for set_type, raw, target, stored_fold in raw_sets:
        feature, support, guard = build_universal_features(raw, spacing_m=spacing, support_envelope=envelopes[domain])
        local_valid = guard & (support <= threshold) & np.asarray(raw["majority_support"], dtype=bool)
        local_valid[:, 0] = True
        features.append(feature)
        targets.append(target)
        valid.append(local_valid)
        groups.append(source["family"].astype(str))
        set_types.append(np.full(len(feature), set_type))
        if stored_fold is None:
            from scoring.universal_dualdomain_raw_v2 import group_fold
            fold_ids.append(np.asarray([group_fold(domain, value, 5) for value in source["family"].astype(str)], dtype=np.int64))
        else:
            fold_ids.append(stored_fold)
    count = sum(len(value) for value in features)
    return {
        "feature": np.concatenate(features),
        "targets_raw": np.concatenate(targets),
        "valid": np.concatenate(valid),
        "group": np.concatenate(groups),
        "set_type": np.concatenate(set_types),
        "fold_id": np.concatenate(fold_ids),
        "domain": np.full(count, domain_index, dtype=np.int64),
        "record_key": np.tile(source["record_key"].astype(str), 4),
    }


def training_weights(domain: np.ndarray, group: np.ndarray, set_type: np.ndarray) -> np.ndarray:
    result = np.zeros(len(domain), dtype=np.float64)
    for domain_index in range(len(DOMAINS)):
        for type_name, type_weight in SET_WEIGHTS.items():
            mask = (domain == domain_index) & (set_type == type_name)
            routes = np.unique(group[mask])
            if not len(routes):
                raise RuntimeError(f"missing {DOMAINS[domain_index]}/{type_name}")
            for route in routes:
                local = mask & (group == route)
                result[local] = 0.5 * type_weight / len(routes) / int(local.sum())
    if not np.isclose(result.sum(), 1.0):
        raise RuntimeError(f"training weights sum to {result.sum()}")
    return result


def member_weights(base: np.ndarray, domain: np.ndarray, train_index: np.ndarray) -> np.ndarray:
    result = np.zeros_like(base)
    for domain_index in range(len(DOMAINS)):
        local = train_index[domain[train_index] == domain_index]
        value = base[local]
        result[local] = value * (0.5 / value.sum())
    return result


def fine_tune_member(
    *,
    initial: dict[str, Any],
    feature: np.ndarray,
    targets: dict[str, np.ndarray],
    domain: np.ndarray,
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
        if payload.get("schema") != "dualdomain_cem_tail_v66_member":
            raise RuntimeError(f"unexpected existing checkpoint: {checkpoint}")
        return {"path": str(checkpoint), "sha256": file_sha256(checkpoint), "seed": int(payload["seed"]), "member_role": role, "train_states": int(payload["train_states"]), "resumed": True}
    initial_path = Path(initial["path"])
    if file_sha256(initial_path) != initial["sha256"]:
        raise RuntimeError(f"initial checkpoint changed: {initial_path}")
    payload = torch.load(initial_path, map_location="cpu", weights_only=False)
    config = UniversalModelConfig(**payload["model_config"])
    model = UniversalDualDomainRawV2(config).to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    normalization = {key: np.asarray(value, dtype=np.float32) for key, value in payload["normalization"].items()}
    target_norm = {key: np.asarray(value, dtype=np.float32) for key, value in payload["target_normalization"].items()}
    set_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-4, weight_decay=1.0e-4)
    pos_weight = {name: torch.tensor(positive_weight(targets[name][train_index]), device=device) for name in ("helpful", "dual", "harm")}
    weights = read_json(BASE_CONFIG)["training"]["loss_weights"]
    local_weight = member_weights(base_weight, domain, train_index)
    rng = np.random.default_rng(seed)
    epoch_audit = []
    started = time.perf_counter()
    for epoch in range(epochs):
        order = rng.permutation(train_index)
        model.train()
        total_loss = 0.0
        batches = 0
        for start in range(0, len(order), 32):
            index = order[start : start + 32]
            local_domain = domain[index]
            x = torch.from_numpy(normalize_features(feature[index], local_domain, normalization)).to(device)
            d = torch.from_numpy(local_domain.astype(np.int64)).to(device)
            output = model(x, d)
            target_mean = target_norm["target_mean"][local_domain]
            target_scale = target_norm["target_scale"][local_domain]
            expected_delta = torch.from_numpy((targets["delta"][index] - target_mean[:, None]) / target_scale[:, None]).to(device)
            inverse_variance = torch.exp(-output["logvar"])
            regression = 0.5 * (inverse_variance * (output["mean"] - expected_delta).square() + output["logvar"])
            state_loss = float(weights["regression"]) * regression.mean(dim=(1, 2))
            for name, logit, key in (("helpful", "helpful_logit", "helpful_bce"), ("dual", "dual_logit", "dual_bce"), ("harm", "harm_logit", "harm_bce")):
                expected = torch.from_numpy(targets[name][index]).to(device)
                value = nn.functional.binary_cross_entropy_with_logits(output[logit], expected, pos_weight=pos_weight[name], reduction="none").mean(dim=1)
                state_loss += float(weights[key]) * value
            pair_score = output["rank_logit"][:, 1:] - output["rank_logit"][:, :1]
            pair_dual = torch.from_numpy(targets["dual"][index, 1:]).to(device)
            pair_harm = torch.from_numpy(targets["harm"][index, 1:]).to(device)
            pair_loss = nn.functional.softplus(-(pair_dual * 2.0 - 1.0) * pair_score)
            pair_multiplier = 1.0 + pair_harm * (float(weights["harmful_pair_multiplier"]) - 1.0)
            state_loss += float(weights["incumbent_pairwise"]) * (pair_loss * pair_multiplier).mean(dim=1)
            distribution = torch.from_numpy(targets["list_distribution"][index]).to(device)
            state_loss += float(weights["listwise"]) * (-(distribution * torch.log_softmax(output["rank_logit"][:, 1:], dim=1)).sum(dim=1))
            sample_weight = torch.from_numpy(local_weight[index] * len(train_index)).to(device=device, dtype=state_loss.dtype)
            loss = (state_loss * sample_weight).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.detach().cpu())
            batches += 1
        if epoch == 0 or (epoch + 1) % 5 == 0 or epoch + 1 == epochs:
            record = {"member_role": role, "epoch": epoch + 1, "epochs": epochs, "states": int(len(order)), "mean_batch_loss": total_loss / max(batches, 1), "elapsed_sec": time.perf_counter() - started}
            epoch_audit.append(record)
            print(json.dumps(record), flush=True)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "schema": "dualdomain_cem_tail_v66_member",
        "model_config": payload["model_config"],
        "normalization": normalization,
        "target_normalization": target_norm,
        "state_dict": model.cpu().state_dict(),
        "primary_excludes_scorer_v2_predictions": True,
        "initial_checkpoint": str(initial_path),
        "initial_checkpoint_sha256": initial["sha256"],
        "seed": seed,
        "member_role": role,
        "train_states": int(len(train_index)),
        "epochs": epochs,
        "epoch_audit": epoch_audit,
        "cem_proposal_aware": True,
        "A_to_H_accessed": False,
        "reserve_accessed": False,
    }, checkpoint)
    return {"path": str(checkpoint), "sha256": file_sha256(checkpoint), "seed": seed, "member_role": role, "train_states": int(len(train_index)), "resumed": False}


def fit_temperature(logit: np.ndarray, target: np.ndarray) -> float:
    return float(min(TEMP_GRID, key=lambda value: binary_nll(logit, target, float(value))))


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and (args.output_dir / "result.json").exists():
        raise FileExistsError(args.output_dir / "result.json")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    prereg = read_json(PREREG)
    amendment = read_json(WEIGHT_AMENDMENT)
    if prereg.get("status") != "frozen_before_proposal_harvest" or amendment.get("status") != "frozen_before_any_v66_training":
        raise RuntimeError("v66 frozen protocol mismatch")
    base_result = read_json(BASE_RESULT)
    envelopes = load_envelopes(Path(base_result["support"]))
    train_parts = [load_domain_split(domain, "train_fit", envelopes) for domain in DOMAINS]
    cal_parts = [load_domain_split(domain, "scorer_cal", envelopes) for domain in DOMAINS]
    train = {key: np.concatenate([part[key] for part in train_parts]) for key in train_parts[0]}
    train_targets = build_universal_targets(train["targets_raw"])
    base_weight = training_weights(train["domain"], train["group"], train["set_type"])
    checkpoints = {"oof": [], "full": []}
    for heldout, initial in enumerate(base_result["arms"]["D"]["checkpoints"]["oof"]):
        index = np.flatnonzero(train["fold_id"] != heldout)
        checkpoints["oof"].append(fine_tune_member(initial=initial, feature=train["feature"], targets=train_targets, domain=train["domain"], base_weight=base_weight, train_index=index, seed=20260730 + heldout, device=args.device, checkpoint=args.output_dir / "checkpoints" / f"arm_D_oof_fold{heldout}.pt", role=f"oof_fold_{heldout}", epochs=args.epochs))
    all_index = np.arange(len(train["feature"]), dtype=np.int64)
    for member_index, initial in enumerate(base_result["arms"]["D"]["checkpoints"]["full"]):
        checkpoints["full"].append(fine_tune_member(initial=initial, feature=train["feature"], targets=train_targets, domain=train["domain"], base_weight=base_weight, train_index=all_index, seed=20260830 + member_index, device=args.device, checkpoint=args.output_dir / "checkpoints" / f"arm_D_full_seed{member_index}.pt", role=f"full_seed_{member_index}", epochs=args.epochs))

    calibration_domains = {}
    selected_arrays = {}
    for domain, part in zip(DOMAINS, cal_parts):
        prediction = ensemble_predict(checkpoints["full"], part["feature"], part["domain"], device=args.device)
        targets = build_universal_targets(part["targets_raw"])
        dual_temperature = fit_temperature(prediction["dual_logit"], targets["dual"])
        harm_temperature = fit_temperature(prediction["harm_logit"], targets["harm"])
        p_dual = sigmoid(prediction["dual_logit"] / dual_temperature)
        p_harm = sigmoid(prediction["harm_logit"] / harm_temperature)
        calibration_domains[domain] = {
            "prediction": prediction,
            "p_dual": p_dual,
            "p_harm": p_harm,
            "valid": part["valid"],
            "targets": part["targets_raw"],
            "dual_temperature": dual_temperature,
            "harm_temperature": harm_temperature,
        }
    calibration_contract = {
        "minimum_coverage": 0.4,
        "target_dual_precision": 0.85,
        "target_wilson_lower_95": 0.80,
        "maximum_harm_rate": 0.15,
    }
    frontier_input = {
        domain: {key: value[key] for key in ("prediction", "p_dual", "p_harm", "valid", "targets")}
        for domain, value in calibration_domains.items()
    }
    frontier = choose_frontier(frontier_input, calibration_contract, shared_policy=False)
    for domain in DOMAINS:
        selected_arrays[f"{domain}_selected_index"] = frontier["domains"][domain]["selected"]
    selected_path = args.output_dir / "calibration_selected_index.npz"
    write_npz_atomic(selected_path, **selected_arrays)
    calibration_summary = {
        domain: {
            "dual_temperature": value["dual_temperature"],
            "harm_temperature": value["harm_temperature"],
            "policy": frontier["domains"][domain]["policy"],
            "metrics": frontier["domains"][domain]["metrics"],
            "feasible": frontier["domains"][domain]["feasible"],
        }
        for domain, value in calibration_domains.items()
    }
    result = {
        "schema": "dualdomain_cem_tail_v66_training_result",
        "status": "trained_and_frozen_before_A_to_H_replay",
        "checkpoints": checkpoints,
        "train_states": int(len(train["feature"])),
        "train_original_states": {domain: int(len(part["feature"]) // 4) for domain, part in zip(DOMAINS, train_parts)},
        "calibration_states": {domain: int(len(part["feature"])) for domain, part in zip(DOMAINS, cal_parts)},
        "calibration": calibration_summary,
        "calibration_selected_index": str(selected_path),
        "calibration_selected_index_sha256": file_sha256(selected_path),
        "base_result": str(BASE_RESULT),
        "base_result_sha256": file_sha256(BASE_RESULT),
        "preregistration": str(PREREG),
        "preregistration_sha256": file_sha256(PREREG),
        "training_weight_amendment": str(WEIGHT_AMENDMENT),
        "training_weight_amendment_sha256": file_sha256(WEIGHT_AMENDMENT),
        "A_to_H_accessed": False,
        "reserve_accessed": False,
        "production_safe_claim_allowed": False,
    }
    result_path = args.output_dir / "result.json"
    write_json_atomic(result_path, result)
    print(json.dumps({"stage": "train", "status": result["status"], "calibration": calibration_summary, "output": str(result_path), "sha256": file_sha256(result_path)}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--epochs", type=int, default=20)
    return parser.parse_args()


if __name__ == "__main__":
    main()
