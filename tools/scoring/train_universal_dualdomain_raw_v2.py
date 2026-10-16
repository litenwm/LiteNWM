#!/usr/bin/env python3
"""Train and calibrate full-data universal RECON+SCAND raw scorer v2."""

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

from scoring.dense_meta_v1 import wilson_lower  # noqa: E402
from scoring.spacing_support_v1 import (  # noqa: E402
    fit_spacing_envelope,
    score_spacing_envelope,
    threshold_for_quantile,
)
from scoring.universal_dualdomain_raw_v2 import (  # noqa: E402
    DOMAINS,
    UniversalDualDomainRawV2,
    UniversalModelConfig,
    build_universal_features,
    build_universal_targets,
    feature_normalization,
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


ROOT = REPO_ROOT / "outputs/universal_dualdomain_raw_v2"
ARMS = ("B", "C", "D")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT
        / "configs/scoring/universal_dualdomain_raw_v2.json",
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
        "--recon-cal-source",
        type=Path,
        default=ROOT / "sources/recon_scorer_cal_source.npz",
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
    parser.add_argument(
        "--scand-cal-source",
        type=Path,
        default=REPO_ROOT
        / "outputs/scand_domain_adaptation_v1/sources/scorer_cal_source.npz",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "training_v2")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--epochs", type=int)
    return parser.parse_args()


def load_group(path: Path, expected_record: np.ndarray, dataset: str) -> np.ndarray:
    with np.load(path, allow_pickle=False) as payload:
        if not np.array_equal(payload["record_key"].astype(str), expected_record):
            raise RuntimeError(f"{dataset} source/feature identity mismatch")
        route = payload["route_group"].astype(str)
        if dataset == "scand":
            bag = payload["bag_id"].astype(str)
            return np.char.add(np.char.add(route, "::bag::"), bag)
        return route


def concatenate(values: list[np.ndarray]) -> np.ndarray:
    return np.concatenate(values, axis=0)


def route_domain_weights(domain: np.ndarray, group: np.ndarray) -> np.ndarray:
    result = np.empty(len(domain), dtype=np.float32)
    for domain_index in range(len(DOMAINS)):
        in_domain = domain == domain_index
        names, counts = np.unique(group[in_domain], return_counts=True)
        count_by_group = dict(zip(names.tolist(), counts.tolist()))
        domain_groups = len(names)
        for index in np.flatnonzero(in_domain):
            result[index] = np.float32(
                0.5 / (domain_groups * count_by_group[str(group[index])])
            )
    if not np.allclose(
        [result[domain == index].sum() for index in range(len(DOMAINS))],
        [0.5, 0.5],
        atol=1.0e-5,
    ):
        raise RuntimeError("domain/route weights do not sum to 0.5 per domain")
    return result


def target_normalization(
    delta: np.ndarray, domain: np.ndarray
) -> dict[str, np.ndarray]:
    mean = []
    scale = []
    for domain_index in range(len(DOMAINS)):
        value = delta[domain == domain_index].reshape(-1, 4)
        mean.append(value.mean(axis=0))
        scale.append(np.maximum(value.std(axis=0), 0.005))
    return {
        "target_mean": np.stack(mean).astype(np.float32),
        "target_scale": np.stack(scale).astype(np.float32),
    }


def positive_weight(value: np.ndarray, maximum: float = 20.0) -> float:
    positive = float(value.sum())
    negative = float(value.size - positive)
    return min(negative / max(positive, 1.0), maximum)


def arm_model_config(arm: str, protocol: dict[str, Any]) -> UniversalModelConfig:
    training = protocol["training"]
    return UniversalModelConfig(
        input_dim=int(protocol["feature_contract"]["numeric_input_dim"]),
        hidden_dim=int(training["hidden_dim"]),
        domain_embedding_dim=int(
            protocol["feature_contract"]["domain_embedding_dim"]
        ),
        dropout=float(training["dropout"]),
        domain_conditioned=arm in {"C", "D"},
        setwise_context=arm == "D",
        heteroscedastic=arm == "D",
    )


def train_member(
    *,
    arm: str,
    feature: np.ndarray,
    targets: dict[str, np.ndarray],
    domain: np.ndarray,
    state_weight: np.ndarray,
    train_index: np.ndarray,
    protocol: dict[str, Any],
    seed: int,
    device: str,
    checkpoint: Path,
    member_role: str,
) -> dict[str, Any]:
    config = arm_model_config(arm, protocol)
    training = protocol["training"]
    epochs = int(training["epochs"])
    batch_states = int(training["batch_states"])
    set_seed(seed)
    normalization = feature_normalization(
        feature[train_index],
        domain[train_index],
        domain_conditioned=config.domain_conditioned,
    )
    target_norm = target_normalization(
        targets["delta"][train_index], domain[train_index]
    )
    model = UniversalDualDomainRawV2(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    pos_weight = {
        name: torch.tensor(
            positive_weight(targets[name][train_index]), device=device
        )
        for name in ("helpful", "dual", "harm")
    }
    weights = protocol["training"]["loss_weights"]
    member_state_weight = np.zeros_like(state_weight, dtype=np.float64)
    for domain_index in range(len(DOMAINS)):
        local_index = train_index[domain[train_index] == domain_index]
        if not len(local_index):
            raise RuntimeError(
                f"training member {member_role} has no {DOMAINS[domain_index]} states"
            )
        local_weight = state_weight[local_index].astype(np.float64)
        local_total = float(local_weight.sum())
        if local_total <= 0.0:
            raise RuntimeError("non-positive route/domain weight")
        member_state_weight[local_index] = local_weight * (0.5 / local_total)
    if not np.isclose(member_state_weight[train_index].sum(), 1.0):
        raise RuntimeError("member route/domain weights do not sum to one")
    rng = np.random.default_rng(seed)
    started = time.perf_counter()
    epoch_audit = []
    for epoch in range(epochs):
        order = rng.permutation(train_index)
        if len(np.unique(order)) != len(train_index):
            raise RuntimeError("epoch state coverage is not exactly once")
        model.train()
        accumulated = 0.0
        batches = 0
        for start in range(0, len(order), batch_states):
            index = order[start : start + batch_states]
            local_domain = domain[index]
            normalized = normalize_features(
                feature[index], local_domain, normalization
            )
            x = torch.from_numpy(normalized).to(device)
            d = torch.from_numpy(local_domain.astype(np.int64)).to(device)
            output = model(x, d)
            target_mean = target_norm["target_mean"][local_domain]
            target_scale = target_norm["target_scale"][local_domain]
            true_delta = (
                targets["delta"][index] - target_mean[:, None]
            ) / target_scale[:, None]
            expected_delta = torch.from_numpy(true_delta).to(device)
            state_losses = torch.zeros(len(index), device=device)
            if arm == "D":
                inverse_variance = torch.exp(-output["logvar"])
                regression = 0.5 * (
                    inverse_variance
                    * (output["mean"] - expected_delta).square()
                    + output["logvar"]
                )
                state_losses += float(weights["regression"]) * regression.mean(
                    dim=(1, 2)
                )
            else:
                regression = nn.functional.smooth_l1_loss(
                    output["mean"], expected_delta, reduction="none"
                )
                state_losses += float(weights["regression"]) * regression.mean(
                    dim=(1, 2)
                )
            for name, logit_name, loss_weight in (
                ("helpful", "helpful_logit", "helpful_bce"),
                ("dual", "dual_logit", "dual_bce"),
                ("harm", "harm_logit", "harm_bce"),
            ):
                expected = torch.from_numpy(targets[name][index]).to(device)
                loss = nn.functional.binary_cross_entropy_with_logits(
                    output[logit_name],
                    expected,
                    pos_weight=pos_weight[name],
                    reduction="none",
                ).mean(dim=1)
                state_losses += float(weights[loss_weight]) * loss

            pair_score = output["rank_logit"][:, 1:] - output["rank_logit"][:, :1]
            pair_dual = torch.from_numpy(
                targets["dual"][index, 1:]
            ).to(device)
            pair_harm = torch.from_numpy(
                targets["harm"][index, 1:]
            ).to(device)
            signed = pair_dual * 2.0 - 1.0
            pair_loss = nn.functional.softplus(-signed * pair_score)
            pair_multiplier = 1.0 + pair_harm * (
                float(weights["harmful_pair_multiplier"]) - 1.0
            )
            state_losses += float(weights["incumbent_pairwise"]) * (
                pair_loss * pair_multiplier
            ).mean(dim=1)
            if arm == "D":
                distribution = torch.from_numpy(
                    targets["list_distribution"][index]
                ).to(device)
                listwise = -(
                    distribution
                    * torch.log_softmax(output["rank_logit"][:, 1:], dim=1)
                ).sum(dim=1)
                state_losses += float(weights["listwise"]) * listwise

            local_weight = torch.from_numpy(
                member_state_weight[index] * len(train_index)
            ).to(device=device, dtype=state_losses.dtype)
            loss = (state_losses * local_weight).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(training["gradient_clip_norm"])
            )
            optimizer.step()
            accumulated += float(loss.detach().cpu())
            batches += 1
        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch + 1 == epochs:
            record = {
                "arm": arm,
                "member_role": member_role,
                "epoch": epoch + 1,
                "epochs": epochs,
                "states_seen_exactly_once": len(order),
                "mean_batch_loss": accumulated / max(batches, 1),
                "elapsed_sec": time.perf_counter() - started,
            }
            epoch_audit.append(record)
            print(json.dumps(record), flush=True)

    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "universal_dualdomain_raw_v2_member",
        "arm": arm,
        "member_role": member_role,
        "seed": seed,
        "model_config": asdict(config),
        "normalization": normalization,
        "target_normalization": target_norm,
        "state_dict": model.cpu().state_dict(),
        "primary_excludes_scorer_v2_predictions": True,
        "train_states": int(len(train_index)),
        "domain_weighting": "0.50/0.50 per member; inverse route frequency; no state discarded",
        "epoch_coverage": "each supplied train state exactly once per epoch",
        "epoch_audit": epoch_audit,
        "cem": False,
    }
    torch.save(payload, checkpoint)
    return {
        "path": str(checkpoint),
        "sha256": file_sha256(checkpoint),
        "seed": seed,
        "member_role": member_role,
        "train_states": int(len(train_index)),
    }


def load_member(
    record: dict[str, Any], device: str
) -> tuple[UniversalDualDomainRawV2, dict[str, np.ndarray], dict[str, np.ndarray]]:
    path = Path(record["path"])
    if file_sha256(path) != record["sha256"]:
        raise RuntimeError(f"checkpoint hash changed: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not payload.get("primary_excludes_scorer_v2_predictions"):
        raise RuntimeError("primary checkpoint feature contract is not raw-only")
    config = UniversalModelConfig(**payload["model_config"])
    model = UniversalDualDomainRawV2(config).to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval().requires_grad_(False)
    normalization = {
        key: np.asarray(value, dtype=np.float32)
        for key, value in payload["normalization"].items()
    }
    target_norm = {
        key: np.asarray(value, dtype=np.float32)
        for key, value in payload["target_normalization"].items()
    }
    return model, normalization, target_norm


def predict_member(
    member: tuple[
        UniversalDualDomainRawV2,
        dict[str, np.ndarray],
        dict[str, np.ndarray],
    ],
    feature: np.ndarray,
    domain: np.ndarray,
    *,
    device: str,
    batch_states: int = 128,
) -> dict[str, np.ndarray]:
    model, normalization, target_norm = member
    values: dict[str, list[np.ndarray]] = {
        key: []
        for key in ("mean", "variance", "helpful_logit", "dual_logit", "harm_logit", "rank_logit")
    }
    for start in range(0, len(feature), batch_states):
        end = min(start + batch_states, len(feature))
        local_domain = domain[start:end]
        normalized = normalize_features(
            feature[start:end], local_domain, normalization
        )
        with torch.inference_mode():
            output = model(
                torch.from_numpy(normalized).to(device),
                torch.from_numpy(local_domain.astype(np.int64)).to(device),
            )
        scale = target_norm["target_scale"][local_domain, None]
        mean = (
            output["mean"].cpu().numpy() * scale
            + target_norm["target_mean"][local_domain, None]
        )
        variance = np.exp(output["logvar"].cpu().numpy()) * np.square(scale)
        values["mean"].append(mean.astype(np.float32))
        values["variance"].append(variance.astype(np.float32))
        for key in ("helpful_logit", "dual_logit", "harm_logit", "rank_logit"):
            values[key].append(output[key].cpu().numpy().astype(np.float32))
    return {key: np.concatenate(value) for key, value in values.items()}


def ensemble_predict(
    records: list[dict[str, Any]],
    feature: np.ndarray,
    domain: np.ndarray,
    *,
    device: str,
) -> dict[str, np.ndarray]:
    predictions = []
    for record in records:
        member = load_member(record, device)
        predictions.append(
            predict_member(member, feature, domain, device=device)
        )
        del member
        if device == "cuda":
            torch.cuda.empty_cache()
    means = np.stack([value["mean"] for value in predictions])
    variances = np.stack([value["variance"] for value in predictions])
    ensemble_mean = means.mean(axis=0)
    total_variance = (
        variances.mean(axis=0)
        + np.square(means).mean(axis=0)
        - np.square(ensemble_mean)
    )
    return {
        "mean": ensemble_mean.astype(np.float32),
        "std": np.sqrt(np.maximum(total_variance, 1.0e-8)).astype(np.float32),
        **{
            key: np.stack([value[key] for value in predictions]).mean(axis=0)
            for key in ("helpful_logit", "dual_logit", "harm_logit", "rank_logit")
        },
    }


def sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -30.0, 30.0)))


def binary_nll(logit: np.ndarray, target: np.ndarray, temperature: float) -> float:
    value = np.asarray(logit, dtype=np.float64) / temperature
    expected = np.asarray(target, dtype=np.float64)
    return float(
        (np.maximum(value, 0.0) - value * expected + np.log1p(np.exp(-np.abs(value)))).mean()
    )


def crossfit_temperature(
    dual_logit: np.ndarray,
    harm_logit: np.ndarray,
    target: dict[str, np.ndarray],
    groups: np.ndarray,
    *,
    folds: int,
) -> dict[str, Any]:
    temperatures = np.asarray((0.5, 0.75, 1.0, 1.5, 2.0, 3.0), dtype=np.float64)
    fold_id = np.asarray(
        [group_fold("recon" if str(value).startswith("recon") else "scand", str(value), folds)
         for value in groups],
        dtype=np.int64,
    )
    oof_dual = np.empty_like(dual_logit, dtype=np.float32)
    oof_harm = np.empty_like(harm_logit, dtype=np.float32)
    fold_records = []
    for heldout in range(folds):
        fit = fold_id != heldout
        test = fold_id == heldout
        if not fit.any() or not test.any():
            continue
        dual_temp = min(
            temperatures,
            key=lambda value: binary_nll(
                dual_logit[fit], target["dual"][fit], float(value)
            ),
        )
        harm_temp = min(
            temperatures,
            key=lambda value: binary_nll(
                harm_logit[fit], target["harm"][fit], float(value)
            ),
        )
        oof_dual[test] = sigmoid(dual_logit[test] / dual_temp)
        oof_harm[test] = sigmoid(harm_logit[test] / harm_temp)
        fold_records.append(
            {
                "heldout_fold": heldout,
                "fit_groups": int(len(np.unique(groups[fit]))),
                "heldout_groups": int(len(np.unique(groups[test]))),
                "dual_temperature": float(dual_temp),
                "harm_temperature": float(harm_temp),
            }
        )
    dual_full = min(
        temperatures,
        key=lambda value: binary_nll(
            dual_logit, target["dual"], float(value)
        ),
    )
    harm_full = min(
        temperatures,
        key=lambda value: binary_nll(
            harm_logit, target["harm"], float(value)
        ),
    )
    return {
        "oof_p_dual": oof_dual,
        "oof_p_harm": oof_harm,
        "deployment_dual_temperature": float(dual_full),
        "deployment_harm_temperature": float(harm_full),
        "folds": fold_records,
    }


def selected_for_policy(
    prediction: dict[str, np.ndarray],
    p_dual: np.ndarray,
    p_harm: np.ndarray,
    valid: np.ndarray,
    policy: dict[str, float],
) -> np.ndarray:
    allowed = np.asarray(valid, dtype=bool).copy()
    allowed[:, 0] = False
    rank = np.where(allowed, prediction["rank_logit"], -np.inf)
    challenger = rank.argmax(axis=1).astype(np.int64)
    exists = allowed.any(axis=1)
    challenger = np.where(exists, challenger, 0)
    row = np.arange(len(challenger))
    z = float(policy["z"])
    ucb_ate = prediction["mean"][row, challenger, 0] + z * prediction["std"][
        row, challenger, 0
    ]
    ucb_objective = prediction["mean"][row, challenger, 3] + z * prediction[
        "std"
    ][row, challenger, 3]
    promote = (
        exists
        & (ucb_ate < -float(policy["margin_ate"]))
        & (ucb_objective < -float(policy["margin_objective"]))
        & (p_dual[row, challenger] >= float(policy["dual_threshold"]))
        & (p_harm[row, challenger] <= float(policy["harm_threshold"]))
    )
    return np.where(promote, challenger, 0).astype(np.int64)


def safety_metrics(selected: np.ndarray, targets: np.ndarray) -> dict[str, Any]:
    row = np.arange(len(selected))
    delta = targets[row, selected, :4]
    promote = selected != 0
    count = int(promote.sum())
    ate_success = int((delta[promote, 0] < -1.0e-7).sum())
    objective_success = int((delta[promote, 3] < -1.0e-7).sum())
    dual_success = int(
        (
            (delta[promote, 0] < -1.0e-7)
            & (delta[promote, 3] < -1.0e-7)
        ).sum()
    )
    harmed = promote & (
        (delta[:, 0] > 1.0e-7) | (delta[:, 3] > 1.0e-7)
    )
    return {
        "states": int(len(selected)),
        "promotions": count,
        "coverage": float(count / len(selected)),
        "ate_precision": ate_success / count if count else None,
        "objective_precision": objective_success / count if count else None,
        "dual_precision": dual_success / count if count else None,
        "dual_wilson_lower_95": wilson_lower(dual_success, count),
        "harm_count": int(harmed.sum()),
        "harm_rate": float(harmed.sum() / count) if count else None,
        "mean_harm_magnitude_m": (
            float(np.maximum(delta[harmed, 0], delta[harmed, 3]).mean())
            if harmed.any()
            else 0.0
        ),
        "net_delta_per_state_m": {
            name: float(delta[:, metric].mean())
            for metric, name in enumerate(("ate", "rpe", "endpoint", "objective"))
        },
    }


def policy_feasible(
    per_domain: dict[str, dict[str, Any]], calibration: dict[str, Any]
) -> bool:
    return all(
        value["coverage"] >= float(calibration["minimum_coverage"])
        and value["dual_precision"] is not None
        and value["dual_precision"] >= float(calibration["target_dual_precision"])
        and value["dual_wilson_lower_95"] is not None
        and value["dual_wilson_lower_95"]
        >= float(calibration["target_wilson_lower_95"])
        and value["harm_rate"] is not None
        and value["harm_rate"] <= float(calibration["maximum_harm_rate"])
        for value in per_domain.values()
    )


def policy_grid() -> list[dict[str, float]]:
    return [
        {
            "z": z,
            "margin_ate": margin,
            "margin_objective": margin,
            "dual_threshold": dual,
            "harm_threshold": harm,
        }
        for z in (0.0, 0.5, 1.0, 1.5, 2.0)
        for margin in (0.0, 0.0025, 0.005, 0.010)
        for dual in (0.50, 0.60, 0.70, 0.80, 0.85, 0.90)
        for harm in (0.05, 0.10, 0.20, 0.30, 0.40)
    ]


def choose_frontier(
    domains: dict[str, dict[str, Any]],
    calibration: dict[str, Any],
    *,
    shared_policy: bool,
) -> dict[str, Any]:
    grid = policy_grid()
    if shared_policy:
        frontier = []
        for policy in grid:
            metrics = {}
            selections = {}
            for name, value in domains.items():
                selected = selected_for_policy(
                    value["prediction"],
                    value["p_dual"],
                    value["p_harm"],
                    value["valid"],
                    policy,
                )
                selections[name] = selected
                metrics[name] = safety_metrics(selected, value["targets"])
            frontier.append(
                {
                    "policy": policy,
                    "metrics": metrics,
                    "feasible": policy_feasible(metrics, calibration),
                    "macro_coverage": float(
                        np.mean([metric["coverage"] for metric in metrics.values()])
                    ),
                    "macro_objective_delta": float(
                        np.mean(
                            [
                                metric["net_delta_per_state_m"]["objective"]
                                for metric in metrics.values()
                            ]
                        )
                    ),
                    "selections": selections,
                }
            )
        feasible = [value for value in frontier if value["feasible"]]
        pool = feasible or frontier
        best = min(
            pool,
            key=lambda value: (
                -value["macro_coverage"] if feasible else 0.0,
                value["macro_objective_delta"],
                -min(
                    metric["dual_precision"] or 0.0
                    for metric in value["metrics"].values()
                ),
            ),
        )
        return {
            "shared": True,
            "feasible": bool(feasible),
            "policy": best["policy"],
            "metrics": best["metrics"],
            "selections": best["selections"],
            "frontier": [
                {key: value[key] for key in ("policy", "metrics", "feasible", "macro_coverage", "macro_objective_delta")}
                for value in frontier
            ],
        }

    results = {}
    for name, value in domains.items():
        frontier = []
        for policy in grid:
            selected = selected_for_policy(
                value["prediction"],
                value["p_dual"],
                value["p_harm"],
                value["valid"],
                policy,
            )
            metrics = safety_metrics(selected, value["targets"])
            feasible = policy_feasible({name: metrics}, calibration)
            frontier.append(
                {
                    "policy": policy,
                    "metrics": metrics,
                    "feasible": feasible,
                    "selected": selected,
                }
            )
        feasible_values = [entry for entry in frontier if entry["feasible"]]
        pool = feasible_values or frontier
        best = min(
            pool,
            key=lambda entry: (
                -entry["metrics"]["coverage"] if feasible_values else 0.0,
                entry["metrics"]["net_delta_per_state_m"]["objective"],
                -(entry["metrics"]["dual_precision"] or 0.0),
            ),
        )
        results[name] = {
            "feasible": bool(feasible_values),
            "policy": best["policy"],
            "metrics": best["metrics"],
            "selected": best["selected"],
            "frontier": [
                {key: entry[key] for key in ("policy", "metrics", "feasible")}
                for entry in frontier
            ],
        }
    return {"shared": False, "domains": results}


def save_support_bundle(
    path: Path, envelopes: dict[str, dict[str, Any]]
) -> None:
    arrays: dict[str, np.ndarray] = {}
    metadata = {}
    for domain, envelope in envelopes.items():
        for key, value in envelope.items():
            if isinstance(value, np.ndarray):
                arrays[f"{domain}__{key}"] = value
            else:
                metadata.setdefault(domain, {})[key] = value
    write_npz(path, **arrays, metadata=np.asarray(json.dumps(metadata, sort_keys=True)))


def main() -> None:
    args = parse_args()
    result_path = args.output_dir / "result.json"
    if result_path.exists():
        raise FileExistsError(result_path)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    protocol = json.loads(args.config.read_text(encoding="utf-8"))
    if args.epochs is not None:
        protocol["training"]["epochs"] = int(args.epochs)
    if protocol.get("cem"):
        raise RuntimeError("CEM must remain disabled")

    recon_train = load_features(args.recon_train, labels_expected=True)
    scand_train = load_features(args.scand_train, labels_expected=True)
    recon_cal = load_features(args.recon_cal, labels_expected=True)
    scand_cal = load_features(args.scand_cal, labels_expected=True)
    expected = ((len(recon_train["record_key"]), len(scand_train["record_key"])),
                (len(recon_cal["record_key"]), len(scand_cal["record_key"])))
    if expected != ((5292, 7489), (410, 1213)):
        raise RuntimeError(f"full-data state contract changed: {expected}")

    train_groups = concatenate(
        [
            np.char.add(
                "recon::",
                load_group(args.recon_train_source, recon_train["record_key"], "recon"),
            ),
            np.char.add(
                "scand::",
                load_group(args.scand_train_source, scand_train["record_key"], "scand"),
            ),
        ]
    )
    cal_groups = {
        "recon": np.char.add(
            "recon::",
            load_group(args.recon_cal_source, recon_cal["record_key"], "recon"),
        ),
        "scand": np.char.add(
            "scand::",
            load_group(args.scand_cal_source, scand_cal["record_key"], "scand"),
        ),
    }
    train_domain = concatenate(
        [
            np.zeros(len(recon_train["record_key"]), dtype=np.int64),
            np.ones(len(scand_train["record_key"]), dtype=np.int64),
        ]
    )
    cal_domain = {
        "recon": np.zeros(len(recon_cal["record_key"]), dtype=np.int64),
        "scand": np.ones(len(scand_cal["record_key"]), dtype=np.int64),
    }

    envelopes = {
        "recon": fit_spacing_envelope(
            recon_train["actions_h4"],
            np.full(len(recon_train["record_key"]), 0.25, dtype=np.float32),
        ),
        "scand": fit_spacing_envelope(
            scand_train["actions_h4"],
            np.full(len(scand_train["record_key"]), 0.38, dtype=np.float32),
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    support_path = args.output_dir / "support/dualdomain_spacing_envelopes.npz"
    save_support_bundle(support_path, envelopes)

    recon_train_feature, _, recon_train_guard = build_universal_features(
        recon_train, spacing_m=0.25, support_envelope=envelopes["recon"]
    )
    scand_train_feature, _, scand_train_guard = build_universal_features(
        scand_train, spacing_m=0.38, support_envelope=envelopes["scand"]
    )
    train_feature = concatenate([recon_train_feature, scand_train_feature])
    train_targets_raw = concatenate(
        [recon_train["targets"], scand_train["targets"]]
    )
    train_targets = build_universal_targets(train_targets_raw)
    state_weight = route_domain_weights(train_domain, train_groups)
    folds = int(protocol["training"]["folds"])
    fold_id = np.asarray(
        [
            group_fold(DOMAINS[int(domain)], str(group), folds)
            for domain, group in zip(train_domain, train_groups)
        ],
        dtype=np.int64,
    )

    cal_feature = {}
    cal_valid = {}
    for name, data, spacing in (
        ("recon", recon_cal, 0.25),
        ("scand", scand_cal, 0.38),
    ):
        feature, score, guard = build_universal_features(
            data, spacing_m=spacing, support_envelope=envelopes[name]
        )
        threshold = threshold_for_quantile(envelopes[name], 0.999)
        valid = guard & (score <= threshold)
        valid[:, 0] = True
        cal_feature[name] = feature
        cal_valid[name] = valid

    arm_results: dict[str, Any] = {}
    selected_arrays: dict[str, np.ndarray] = {}
    for arm_index, arm in enumerate(ARMS):
        checkpoints = {"oof": [], "full": []}
        oof_prediction = {
            key: np.empty(
                (len(train_feature), 17, 4)
                if key in {"mean", "std"}
                else (len(train_feature), 17),
                dtype=np.float32,
            )
            for key in ("mean", "std", "helpful_logit", "dual_logit", "harm_logit", "rank_logit")
        }
        for heldout in range(folds):
            train_index = np.flatnonzero(fold_id != heldout)
            test_index = np.flatnonzero(fold_id == heldout)
            record = train_member(
                arm=arm,
                feature=train_feature,
                targets=train_targets,
                domain=train_domain,
                state_weight=state_weight,
                train_index=train_index,
                protocol=protocol,
                seed=int(protocol["seed"]) + arm_index * 100 + heldout,
                device=args.device,
                checkpoint=args.output_dir
                / f"checkpoints/arm_{arm}_oof_fold{heldout}.pt",
                member_role=f"oof_fold_{heldout}",
            )
            checkpoints["oof"].append(record)
            member = load_member(record, args.device)
            prediction = predict_member(
                member,
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
            del member
            if args.device == "cuda":
                torch.cuda.empty_cache()

        all_index = np.arange(len(train_feature), dtype=np.int64)
        for seed_index, seed in enumerate(
            protocol["training"]["full_data_member_seeds"]
        ):
            record = train_member(
                arm=arm,
                feature=train_feature,
                targets=train_targets,
                domain=train_domain,
                state_weight=state_weight,
                train_index=all_index,
                protocol=protocol,
                seed=int(seed) + arm_index * 100,
                device=args.device,
                checkpoint=args.output_dir
                / f"checkpoints/arm_{arm}_full_seed{seed_index}.pt",
                member_role=f"full_seed_{seed_index}",
            )
            checkpoints["full"].append(record)

        cal_predictions = {
            name: ensemble_predict(
                checkpoints["full"],
                cal_feature[name],
                cal_domain[name],
                device=args.device,
            )
            for name in DOMAINS
        }
        cal_targets = {
            "recon": build_universal_targets(recon_cal["targets"]),
            "scand": build_universal_targets(scand_cal["targets"]),
        }
        calibration_domains = {}
        if arm == "B":
            pooled_dual = concatenate(
                [cal_predictions[name]["dual_logit"] for name in DOMAINS]
            )
            pooled_harm = concatenate(
                [cal_predictions[name]["harm_logit"] for name in DOMAINS]
            )
            pooled_target = {
                "dual": concatenate([cal_targets[name]["dual"] for name in DOMAINS]),
                "harm": concatenate([cal_targets[name]["harm"] for name in DOMAINS]),
            }
            pooled_groups = concatenate([cal_groups[name] for name in DOMAINS])
            temperature = crossfit_temperature(
                pooled_dual,
                pooled_harm,
                pooled_target,
                pooled_groups,
                folds=5,
            )
            offset = 0
            for name in DOMAINS:
                count = len(cal_predictions[name]["dual_logit"])
                calibration_domains[name] = {
                    "prediction": cal_predictions[name],
                    "p_dual": temperature["oof_p_dual"][offset : offset + count],
                    "p_harm": temperature["oof_p_harm"][offset : offset + count],
                    "valid": cal_valid[name],
                    "targets": (
                        recon_cal["targets"] if name == "recon" else scand_cal["targets"]
                    ),
                }
                offset += count
            temperature_record: Any = {
                "shared": True,
                "record": {
                    key: value
                    for key, value in temperature.items()
                    if not key.startswith("oof_")
                },
            }
        else:
            temperature_record = {"shared": False, "domains": {}}
            for name in DOMAINS:
                temperature = crossfit_temperature(
                    cal_predictions[name]["dual_logit"],
                    cal_predictions[name]["harm_logit"],
                    cal_targets[name],
                    cal_groups[name],
                    folds=5 if name == "recon" else 4,
                )
                calibration_domains[name] = {
                    "prediction": cal_predictions[name],
                    "p_dual": temperature["oof_p_dual"],
                    "p_harm": temperature["oof_p_harm"],
                    "valid": cal_valid[name],
                    "targets": (
                        recon_cal["targets"] if name == "recon" else scand_cal["targets"]
                    ),
                }
                temperature_record["domains"][name] = {
                    key: value
                    for key, value in temperature.items()
                    if not key.startswith("oof_")
                }

        frontier = choose_frontier(
            calibration_domains,
            protocol["calibration"],
            shared_policy=arm == "B",
        )
        if arm == "B":
            for name in DOMAINS:
                selected_arrays[f"arm_{arm}__{name}"] = frontier["selections"][name]
        else:
            for name in DOMAINS:
                selected_arrays[f"arm_{arm}__{name}"] = frontier["domains"][name][
                    "selected"
                ]

        # Apply the frozen calibration policy to route-disjoint train OOF predictions.
        oof_metrics = {}
        for domain_index, name in enumerate(DOMAINS):
            index = np.flatnonzero(train_domain == domain_index)
            if arm == "B":
                policy = frontier["policy"]
                dual_temperature = temperature_record["record"][
                    "deployment_dual_temperature"
                ]
                harm_temperature = temperature_record["record"][
                    "deployment_harm_temperature"
                ]
            else:
                policy = frontier["domains"][name]["policy"]
                dual_temperature = temperature_record["domains"][name][
                    "deployment_dual_temperature"
                ]
                harm_temperature = temperature_record["domains"][name][
                    "deployment_harm_temperature"
                ]
            prediction = {key: value[index] for key, value in oof_prediction.items()}
            p_dual = sigmoid(prediction["dual_logit"] / dual_temperature)
            p_harm = sigmoid(prediction["harm_logit"] / harm_temperature)
            train_data = recon_train if name == "recon" else scand_train
            _, score, guard = build_universal_features(
                train_data,
                spacing_m=0.25 if name == "recon" else 0.38,
                support_envelope=envelopes[name],
            )
            threshold = threshold_for_quantile(envelopes[name], 0.999)
            valid = guard & (score <= threshold)
            valid[:, 0] = True
            selected = selected_for_policy(
                prediction, p_dual, p_harm, valid, policy
            )
            oof_metrics[name] = safety_metrics(
                selected, train_targets_raw[index]
            )

        arm_results[arm] = {
            "checkpoints": checkpoints,
            "temperature": temperature_record,
            "calibration": {
                "shared_policy": arm == "B",
                "feasible": (
                    frontier["feasible"]
                    if arm == "B"
                    else all(
                        frontier["domains"][name]["feasible"] for name in DOMAINS
                    )
                ),
                "frontier": frontier,
            },
            "train_group_oof_metrics": oof_metrics,
        }
        # Remove arrays before JSON serialization; selections are frozen separately.
        if arm == "B":
            arm_results[arm]["calibration"]["frontier"].pop("selections")
        else:
            for name in DOMAINS:
                arm_results[arm]["calibration"]["frontier"]["domains"][name].pop(
                    "selected"
                )

    selected_path = args.output_dir / "calibration/selected_index.npz"
    write_npz(selected_path, **selected_arrays)
    report = {
        "schema": "universal_dualdomain_raw_v2_training_result",
        "status": "trained_full_canonical_and_crossfit_calibrated",
        "config": str(args.config),
        "config_sha256": file_sha256(args.config),
        "data_sha256": {
            "recon_train": file_sha256(args.recon_train),
            "recon_cal": file_sha256(args.recon_cal),
            "scand_train": file_sha256(args.scand_train),
            "scand_cal": file_sha256(args.scand_cal),
        },
        "train_states": {
            "recon": len(recon_train["record_key"]),
            "scand": len(scand_train["record_key"]),
            "total": len(train_feature),
            "k17_candidates": int(len(train_feature) * 17),
        },
        "scorer_cal_states": {
            "recon": len(recon_cal["record_key"]),
            "scand": len(scand_cal["record_key"]),
        },
        "feature_dim": int(train_feature.shape[-1]),
        "primary_excludes_scorer_v2_predictions": True,
        "support_role": "OOD/physical guard and uncertainty input only",
        "support": str(support_path),
        "support_sha256": file_sha256(support_path),
        "selected_index": str(selected_path),
        "selected_index_sha256": file_sha256(selected_path),
        "arms": arm_results,
        "legacy_A_role": "legacy_noncanonical_spent baseline only",
        "cem_cal_accessed": False,
        "fixed_pool_only": True,
        "cem": False,
    }
    write_json(result_path, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "result": str(result_path),
                "result_sha256": file_sha256(result_path),
                "train_states": report["train_states"],
                "scorer_cal_states": report["scorer_cal_states"],
                "calibration_feasible": {
                    arm: value["calibration"]["feasible"]
                    for arm, value in arm_results.items()
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
