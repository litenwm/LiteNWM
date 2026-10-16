#!/usr/bin/env python3
"""Fine-tune frozen arm-D members with a dual-safe setwise ranking objective."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from scoring.universal_dualdomain_raw_v2 import (  # noqa: E402
    DOMAINS,
    build_universal_features,
    build_universal_targets,
    group_fold,
    normalize_features,
    set_seed,
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
    concatenate,
    ensemble_predict,
    load_group,
    load_member,
    positive_weight,
    predict_member,
    route_domain_weights,
    safety_metrics,
)


ROOT = REPO_ROOT / "outputs/universal_dualdomain_raw_v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT
        / "configs/scoring/universal_dualdomain_raw_v2_safe_rank.json",
    )
    parser.add_argument(
        "--base-result", type=Path, default=ROOT / "training_v2/result.json"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "safe_rank_v2"
    )
    parser.add_argument(
        "--recon-train",
        type=Path,
        default=ROOT
        / "features/recon_train_fit/recon_train_fit_universal_v2_features.npz",
    )
    parser.add_argument(
        "--recon-cal",
        type=Path,
        default=ROOT
        / "features/recon_scorer_cal/recon_scorer_cal_universal_v2_features.npz",
    )
    parser.add_argument(
        "--recon-train-source",
        type=Path,
        default=ROOT / "sources/recon_train_fit_source.npz",
    )
    parser.add_argument(
        "--scand-train",
        type=Path,
        default=REPO_ROOT
        / "outputs/scand_domain_adaptation_v1/features/train_fit/"
        "scand_inner_train_fit_spacing038_features.npz",
    )
    parser.add_argument(
        "--scand-cal",
        type=Path,
        default=REPO_ROOT
        / "outputs/scand_domain_adaptation_v1/features/scorer_cal/"
        "scand_inner_scorer_cal_spacing038_features.npz",
    )
    parser.add_argument(
        "--scand-train-source",
        type=Path,
        default=REPO_ROOT
        / "outputs/scand_domain_adaptation_v1/sources/train_fit_source.npz",
    )
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--epochs", type=int)
    return parser.parse_args()


def load_envelopes(path: Path) -> dict[str, dict[str, Any]]:
    with np.load(path, allow_pickle=False) as payload:
        metadata = json.loads(str(payload["metadata"].item()))
        result: dict[str, dict[str, Any]] = {
            domain: dict(values) for domain, values in metadata.items()
        }
        for key in payload.files:
            if key == "metadata":
                continue
            domain, name = key.split("__", maxsplit=1)
            result.setdefault(domain, {})[name] = payload[key]
    return result


def member_weights(
    state_weight: np.ndarray,
    domain: np.ndarray,
    train_index: np.ndarray,
) -> np.ndarray:
    result = np.zeros_like(state_weight, dtype=np.float64)
    for domain_index in range(len(DOMAINS)):
        local = train_index[domain[train_index] == domain_index]
        if not len(local):
            raise RuntimeError(f"member lacks {DOMAINS[domain_index]} states")
        value = state_weight[local].astype(np.float64)
        result[local] = value * (0.5 / value.sum())
    if not np.isclose(result[train_index].sum(), 1.0):
        raise RuntimeError("member weights do not sum to one")
    return result


def safe_distribution(
    dual: np.ndarray,
    utility: np.ndarray,
    temperature: float,
) -> tuple[np.ndarray, np.ndarray]:
    safe = np.asarray(dual[:, 1:], dtype=bool)
    score = -np.asarray(utility[:, 1:], dtype=np.float64) / temperature
    score = np.where(safe, score, -np.inf)
    maximum = np.max(score, axis=1, keepdims=True)
    has_safe = np.isfinite(maximum[:, 0])
    maximum = np.where(np.isfinite(maximum), maximum, 0.0)
    weight = np.where(safe, np.exp(np.clip(score - maximum, -30.0, 0.0)), 0.0)
    denominator = weight.sum(axis=1, keepdims=True)
    distribution = np.divide(
        weight,
        denominator,
        out=np.zeros_like(weight),
        where=denominator > 0.0,
    )
    return distribution.astype(np.float32), has_safe


def fine_tune_member(
    *,
    base_record: dict[str, Any],
    feature: np.ndarray,
    targets: dict[str, np.ndarray],
    domain: np.ndarray,
    state_weight: np.ndarray,
    train_index: np.ndarray,
    protocol: dict[str, Any],
    seed: int,
    device: str,
    checkpoint: Path,
) -> dict[str, Any]:
    set_seed(seed)
    member = load_member(base_record, device)
    model, normalization, target_norm = member
    model.train().requires_grad_(True)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(protocol["learning_rate"]),
        weight_decay=float(protocol["weight_decay"]),
    )
    epochs = int(protocol["fine_tune_epochs"])
    batch_states = int(protocol["batch_states"])
    loss_weight = protocol["loss_weights"]
    local_weight = member_weights(state_weight, domain, train_index)
    dual_pos_weight = torch.tensor(
        positive_weight(targets["dual"][train_index]), device=device
    )
    harm_pos_weight = torch.tensor(
        positive_weight(targets["harm"][train_index]), device=device
    )
    distribution, has_safe = safe_distribution(
        targets["dual"],
        targets["utility"],
        float(protocol["safe_listwise_temperature_m"]),
    )
    rng = np.random.default_rng(seed)
    started = time.perf_counter()
    audit = []
    for epoch in range(epochs):
        order = rng.permutation(train_index)
        if len(np.unique(order)) != len(train_index):
            raise RuntimeError("fine-tune epoch does not cover states exactly once")
        accumulated = 0.0
        batches = 0
        for start in range(0, len(order), batch_states):
            index = order[start : start + batch_states]
            local_domain = domain[index]
            normalized = normalize_features(
                feature[index], local_domain, normalization
            )
            output = model(
                torch.from_numpy(normalized).to(device),
                torch.from_numpy(local_domain.astype(np.int64)).to(device),
            )
            expected_delta = (
                targets["delta"][index]
                - target_norm["target_mean"][local_domain, None]
            ) / target_norm["target_scale"][local_domain, None]
            expected_delta_tensor = torch.from_numpy(expected_delta).to(device)
            inverse_variance = torch.exp(-output["logvar"])
            regression = 0.5 * (
                inverse_variance
                * (output["mean"] - expected_delta_tensor).square()
                + output["logvar"]
            )
            state_loss = float(loss_weight["regression"]) * regression.mean(
                dim=(1, 2)
            )

            expected_dual = torch.from_numpy(targets["dual"][index]).to(device)
            expected_harm = torch.from_numpy(targets["harm"][index]).to(device)
            dual_loss = nn.functional.binary_cross_entropy_with_logits(
                output["dual_logit"],
                expected_dual,
                pos_weight=dual_pos_weight,
                reduction="none",
            ).mean(dim=1)
            harm_loss = nn.functional.binary_cross_entropy_with_logits(
                output["harm_logit"],
                expected_harm,
                pos_weight=harm_pos_weight,
                reduction="none",
            ).mean(dim=1)
            state_loss += float(loss_weight["dual_bce"]) * dual_loss
            state_loss += float(loss_weight["harm_bce"]) * harm_loss

            pair_score = output["rank_logit"][:, 1:] - output["rank_logit"][:, :1]
            pair_dual = expected_dual[:, 1:]
            pair_harm = expected_harm[:, 1:]
            signed = pair_dual * 2.0 - 1.0
            pair_loss = nn.functional.softplus(-signed * pair_score)
            multiplier = 1.0 + pair_harm * (
                float(loss_weight["harmful_pair_multiplier"]) - 1.0
            )
            state_loss += float(loss_weight["incumbent_pairwise"]) * (
                pair_loss * multiplier
            ).mean(dim=1)

            expected_distribution = torch.from_numpy(distribution[index]).to(device)
            listwise = -(
                expected_distribution
                * torch.log_softmax(output["rank_logit"][:, 1:], dim=1)
            ).sum(dim=1)
            safe_mask = torch.from_numpy(has_safe[index].astype(np.float32)).to(
                device
            )
            state_loss += (
                float(loss_weight["safe_listwise"]) * listwise * safe_mask
            )

            weight = torch.from_numpy(
                local_weight[index] * len(train_index)
            ).to(device=device, dtype=state_loss.dtype)
            loss = (state_loss * weight).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(protocol["gradient_clip_norm"])
            )
            optimizer.step()
            accumulated += float(loss.detach().cpu())
            batches += 1
        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch + 1 == epochs:
            record = {
                "member_role": base_record["member_role"],
                "epoch": epoch + 1,
                "epochs": epochs,
                "states_seen_exactly_once": len(order),
                "mean_batch_loss": accumulated / max(batches, 1),
                "elapsed_sec": time.perf_counter() - started,
            }
            audit.append(record)
            print(json.dumps(record), flush=True)

    base_payload = torch.load(
        Path(base_record["path"]), map_location="cpu", weights_only=False
    )
    payload = {
        "schema": "universal_dualdomain_raw_v2_safe_rank_member",
        "arm": "D-safe-rank",
        "member_role": base_record["member_role"],
        "seed": seed,
        "model_config": asdict(model.config),
        "normalization": normalization,
        "target_normalization": target_norm,
        "state_dict": model.cpu().state_dict(),
        "base_checkpoint": base_record["path"],
        "base_checkpoint_sha256": base_record["sha256"],
        "base_schema": base_payload["schema"],
        "primary_excludes_scorer_v2_predictions": True,
        "fine_tune_objective": (
            "dual/harm BCE + harmful pairwise + dual-safe setwise CE"
        ),
        "train_states": int(len(train_index)),
        "domain_weighting": (
            "0.50/0.50 per member; inverse route frequency; no state discarded"
        ),
        "epoch_coverage": "each supplied train state exactly once per epoch",
        "epoch_audit": audit,
        "cem": False,
    }
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, checkpoint)
    return {
        "path": str(checkpoint),
        "sha256": file_sha256(checkpoint),
        "member_role": base_record["member_role"],
        "seed": seed,
        "train_states": int(len(train_index)),
    }


def score_components(
    prediction: dict[str, np.ndarray],
    challenger: np.ndarray,
    z: float,
) -> dict[str, np.ndarray]:
    row = np.arange(len(challenger))
    ucb_ate = (
        prediction["mean"][row, challenger, 0]
        + z * prediction["std"][row, challenger, 0]
    )
    ucb_objective = (
        prediction["mean"][row, challenger, 3]
        + z * prediction["std"][row, challenger, 3]
    )
    return {
        "dual_harm": (
            prediction["dual_logit"][row, challenger]
            - prediction["harm_logit"][row, challenger]
        ),
        "rank_margin": (
            prediction["rank_logit"][row, challenger]
            - prediction["rank_logit"][row, 0]
        ),
        "conservative": np.minimum(-ucb_ate, -ucb_objective),
        "ucb_ate": ucb_ate,
        "ucb_objective": ucb_objective,
    }


def normalize_score(value: np.ndarray, eligible: np.ndarray) -> tuple[np.ndarray, float, float]:
    center = float(np.median(value[eligible]))
    scale = float(
        max(
            1.4826 * np.median(np.abs(value[eligible] - center)),
            np.std(value[eligible]),
            1.0e-6,
        )
    )
    return (value - center) / scale, center, scale


def policy_selected(
    prediction: dict[str, np.ndarray],
    valid: np.ndarray,
    policy: dict[str, Any],
) -> np.ndarray:
    rank = np.where(valid, prediction["rank_logit"], -np.inf)
    rank[:, 0] = -np.inf
    challenger = rank.argmax(axis=1).astype(np.int64)
    exists = np.isfinite(rank).any(axis=1)
    components = score_components(prediction, challenger, float(policy["z"]))
    eligible = (
        exists
        & (components["ucb_ate"] < -float(policy["margin"]))
        & (components["ucb_objective"] < -float(policy["margin"]))
    )
    if policy["score_type"] == "composite":
        score = np.zeros(len(challenger), dtype=np.float64)
        for name in ("dual_harm", "rank_margin", "conservative"):
            spec = policy["score_normalization"][name]
            score += (components[name] - spec["center"]) / spec["scale"]
    else:
        score = components[policy["score_type"]]
    promote = eligible & (score >= float(policy["threshold"]))
    return np.where(promote, challenger, 0).astype(np.int64)


def choose_continuous_policy(
    prediction: dict[str, np.ndarray],
    valid: np.ndarray,
    targets: np.ndarray,
    calibration: dict[str, Any],
) -> dict[str, Any]:
    rank = np.where(valid, prediction["rank_logit"], -np.inf)
    rank[:, 0] = -np.inf
    challenger = rank.argmax(axis=1).astype(np.int64)
    exists = np.isfinite(rank).any(axis=1)
    minimum = int(math.ceil(float(calibration["minimum_coverage"]) * len(rank)))
    candidates = []
    for z in (0.0, 0.5, 1.0, 1.5, 2.0):
        components = score_components(prediction, challenger, z)
        for margin in (0.0, 0.0025, 0.005, 0.010):
            eligible = (
                exists
                & (components["ucb_ate"] < -margin)
                & (components["ucb_objective"] < -margin)
            )
            if int(eligible.sum()) < minimum:
                continue
            normalized = {}
            normalization = {}
            for name in ("dual_harm", "rank_margin", "conservative"):
                value, center, scale = normalize_score(components[name], eligible)
                normalized[name] = value
                normalization[name] = {"center": center, "scale": scale}
            score_values = {
                "dual_harm": components["dual_harm"],
                "rank_margin": components["rank_margin"],
                "conservative": components["conservative"],
                "composite": (
                    normalized["dual_harm"]
                    + normalized["rank_margin"]
                    + normalized["conservative"]
                ),
            }
            for score_type, score in score_values.items():
                eligible_index = np.flatnonzero(eligible)
                order = eligible_index[
                    np.argsort(-score[eligible_index], kind="stable")
                ]
                for promoted in range(minimum, len(order) + 1):
                    threshold = float(score[order[promoted - 1]])
                    selected = np.zeros(len(rank), dtype=np.int64)
                    selected[order[:promoted]] = challenger[order[:promoted]]
                    metrics = safety_metrics(selected, targets)
                    feasible = (
                        metrics["dual_precision"] is not None
                        and metrics["dual_precision"]
                        >= float(calibration["target_dual_precision"])
                        and metrics["dual_wilson_lower_95"] is not None
                        and metrics["dual_wilson_lower_95"]
                        >= float(calibration["target_wilson_lower_95"])
                        and metrics["harm_rate"] is not None
                        and metrics["harm_rate"]
                        <= float(calibration["maximum_harm_rate"])
                    )
                    candidates.append(
                        {
                            "feasible": feasible,
                            "metrics": metrics,
                            "policy": {
                                "z": z,
                                "margin": margin,
                                "score_type": score_type,
                                "threshold": threshold,
                                "score_normalization": (
                                    normalization if score_type == "composite" else {}
                                ),
                            },
                            "selected": selected,
                        }
                    )
    feasible = [value for value in candidates if value["feasible"]]
    if feasible:
        best = min(
            feasible,
            key=lambda value: (
                -value["metrics"]["coverage"],
                value["metrics"]["net_delta_per_state_m"]["objective"],
            ),
        )
    else:
        best = min(
            candidates,
            key=lambda value: (
                -(value["metrics"]["dual_precision"] or 0.0),
                value["metrics"]["harm_rate"] or 1.0,
                value["metrics"]["net_delta_per_state_m"]["objective"],
            ),
        )
    return {
        "feasible": bool(feasible),
        "policy": best["policy"],
        "metrics": best["metrics"],
        "selected": best["selected"],
        "evaluated_prefixes": len(candidates),
    }


def main() -> None:
    args = parse_args()
    if (args.output_dir / "result.json").exists():
        raise FileExistsError(args.output_dir / "result.json")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    protocol = json.loads(args.config.read_text(encoding="utf-8"))
    if args.epochs is not None:
        protocol["fine_tune_epochs"] = int(args.epochs)
    if protocol.get("cem") is not False:
        raise RuntimeError("safe-rank protocol must keep CEM disabled")
    base = json.loads(args.base_result.read_text(encoding="utf-8"))
    if base.get("cem") is not False or not base.get(
        "primary_excludes_scorer_v2_predictions"
    ):
        raise RuntimeError("base result is not frozen raw-only CEM=false")

    recon_train = load_features(args.recon_train, labels_expected=True)
    scand_train = load_features(args.scand_train, labels_expected=True)
    recon_cal = load_features(args.recon_cal, labels_expected=True)
    scand_cal = load_features(args.scand_cal, labels_expected=True)
    if (
        len(recon_train["record_key"]),
        len(scand_train["record_key"]),
        len(recon_cal["record_key"]),
        len(scand_cal["record_key"]),
    ) != (5292, 7489, 410, 1213):
        raise RuntimeError("safe-rank full data contract changed")

    train_domain = concatenate(
        [
            np.zeros(len(recon_train["record_key"]), dtype=np.int64),
            np.ones(len(scand_train["record_key"]), dtype=np.int64),
        ]
    )
    train_groups = concatenate(
        [
            np.char.add(
                "recon::",
                load_group(
                    args.recon_train_source,
                    recon_train["record_key"],
                    "recon",
                ),
            ),
            np.char.add(
                "scand::",
                load_group(
                    args.scand_train_source,
                    scand_train["record_key"],
                    "scand",
                ),
            ),
        ]
    )
    fold_id = np.asarray(
        [
            group_fold(DOMAINS[int(domain)], str(group), 5)
            for domain, group in zip(train_domain, train_groups)
        ],
        dtype=np.int64,
    )
    state_weight = route_domain_weights(train_domain, train_groups)
    envelopes = load_envelopes(Path(base["support"]))
    recon_train_feature, recon_train_score, recon_train_guard = (
        build_universal_features(
            recon_train,
            spacing_m=0.25,
            support_envelope=envelopes["recon"],
        )
    )
    scand_train_feature, scand_train_score, scand_train_guard = (
        build_universal_features(
            scand_train,
            spacing_m=0.38,
            support_envelope=envelopes["scand"],
        )
    )
    train_feature = concatenate([recon_train_feature, scand_train_feature])
    train_targets_raw = concatenate(
        [recon_train["targets"], scand_train["targets"]]
    )
    train_targets = build_universal_targets(train_targets_raw)
    train_valid = {}
    offset = 0
    for name, score, guard, count in (
        ("recon", recon_train_score, recon_train_guard, len(recon_train_feature)),
        ("scand", scand_train_score, scand_train_guard, len(scand_train_feature)),
    ):
        threshold = float(
            np.quantile(
                np.asarray(envelopes[name]["fit_nonconformity"]),
                0.999,
                method="higher",
            )
        )
        value = guard & (score <= threshold)
        value[:, 0] = True
        train_valid[name] = value
        offset += count

    checkpoints = {"oof": [], "full": []}
    oof_prediction = {
        key: np.empty(
            (len(train_feature), 17, 4)
            if key in {"mean", "std"}
            else (len(train_feature), 17),
            dtype=np.float32,
        )
        for key in (
            "mean",
            "std",
            "helpful_logit",
            "dual_logit",
            "harm_logit",
            "rank_logit",
        )
    }
    for base_record in base["arms"]["D"]["checkpoints"]["oof"]:
        heldout = int(str(base_record["member_role"]).rpartition("_")[-1])
        train_index = np.flatnonzero(fold_id != heldout)
        test_index = np.flatnonzero(fold_id == heldout)
        record = fine_tune_member(
            base_record=base_record,
            feature=train_feature,
            targets=train_targets,
            domain=train_domain,
            state_weight=state_weight,
            train_index=train_index,
            protocol=protocol,
            seed=int(protocol["seed"]) + heldout,
            device=args.device,
            checkpoint=args.output_dir
            / f"checkpoints/safe_rank_oof_fold{heldout}.pt",
        )
        checkpoints["oof"].append(record)
        prediction = predict_member(
            load_member(record, args.device),
            train_feature[test_index],
            train_domain[test_index],
            device=args.device,
        )
        for key in oof_prediction:
            if key == "std":
                oof_prediction[key][test_index] = np.sqrt(
                    np.maximum(prediction["variance"], 1.0e-8)
                )
            else:
                oof_prediction[key][test_index] = prediction[key]

    all_index = np.arange(len(train_feature), dtype=np.int64)
    for seed_index, base_record in enumerate(
        base["arms"]["D"]["checkpoints"]["full"]
    ):
        record = fine_tune_member(
            base_record=base_record,
            feature=train_feature,
            targets=train_targets,
            domain=train_domain,
            state_weight=state_weight,
            train_index=all_index,
            protocol=protocol,
            seed=int(protocol["seed"]) + 100 + seed_index,
            device=args.device,
            checkpoint=args.output_dir
            / f"checkpoints/safe_rank_full_seed{seed_index}.pt",
        )
        checkpoints["full"].append(record)

    cal_data = {"recon": recon_cal, "scand": scand_cal}
    cal_feature = {}
    cal_valid = {}
    cal_domain = {}
    for domain_index, (name, spacing) in enumerate(
        (("recon", 0.25), ("scand", 0.38))
    ):
        feature, score, guard = build_universal_features(
            cal_data[name],
            spacing_m=spacing,
            support_envelope=envelopes[name],
        )
        threshold = float(
            np.quantile(
                np.asarray(envelopes[name]["fit_nonconformity"]),
                0.999,
                method="higher",
            )
        )
        valid = guard & (score <= threshold)
        valid[:, 0] = True
        cal_feature[name] = feature
        cal_valid[name] = valid
        cal_domain[name] = np.full(
            len(feature), domain_index, dtype=np.int64
        )
    cal_prediction = {
        name: ensemble_predict(
            checkpoints["full"],
            cal_feature[name],
            cal_domain[name],
            device=args.device,
        )
        for name in DOMAINS
    }
    calibration = {
        name: choose_continuous_policy(
            cal_prediction[name],
            cal_valid[name],
            cal_data[name]["targets"],
            protocol["calibration"],
        )
        for name in DOMAINS
    }
    oof_metrics = {}
    for domain_index, name in enumerate(DOMAINS):
        index = np.flatnonzero(train_domain == domain_index)
        prediction = {
            key: value[index] for key, value in oof_prediction.items()
        }
        selected = policy_selected(
            prediction,
            train_valid[name],
            calibration[name]["policy"],
        )
        oof_metrics[name] = safety_metrics(
            selected, train_targets_raw[index]
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected_path = args.output_dir / "calibration/selected_index.npz"
    write_npz(
        selected_path,
        recon=calibration["recon"].pop("selected"),
        scand=calibration["scand"].pop("selected"),
    )
    result = {
        "schema": "universal_dualdomain_raw_v2_safe_rank_result",
        "status": "fine_tuned_and_continuous_gate_calibrated",
        "config": str(args.config),
        "config_sha256": file_sha256(args.config),
        "base_result": str(args.base_result),
        "base_result_sha256": file_sha256(args.base_result),
        "base_arm": "D",
        "train_states": {
            "recon": len(recon_train["record_key"]),
            "scand": len(scand_train["record_key"]),
            "total": len(train_feature),
            "k17_candidates": int(len(train_feature) * 17),
        },
        "checkpoints": checkpoints,
        "calibration": calibration,
        "train_group_oof_metrics": oof_metrics,
        "support": base["support"],
        "support_sha256": base["support_sha256"],
        "selected_index": str(selected_path),
        "selected_index_sha256": file_sha256(selected_path),
        "primary_excludes_scorer_v2_predictions": True,
        "scorer_cal_role": "policy calibration only; not fresh outer evidence",
        "cem_cal_accessed": False,
        "fixed_pool_only": True,
        "cem": False,
    }
    result_path = args.output_dir / "result.json"
    write_json(result_path, result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "result": str(result_path),
                "result_sha256": file_sha256(result_path),
                "calibration": {
                    name: {
                        "feasible": calibration[name]["feasible"],
                        "metrics": calibration[name]["metrics"],
                        "policy": calibration[name]["policy"],
                    }
                    for name in DOMAINS
                },
                "cem": False,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
