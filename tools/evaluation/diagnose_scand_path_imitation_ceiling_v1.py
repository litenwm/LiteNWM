#!/usr/bin/env python3
"""Diagnose how much union65 ATE oracle value plain path imitation can recover.

The 32 mm union65 oracle is defined by geometric distance to the logged future
path, not by executing counterfactual actions.  This spent-development
diagnostic therefore predicts the four logged future waypoints directly from
the existing runtime-safe state summary, then selects the candidate nearest to
that predicted path.

It reports three distinct regimes:
* route OOF: entire route groups held out;
* random OOF: states held out while routes may be shared;
* resubstitution: training-fit/optimization ceiling only.

No sealed validation data is read and no result is eligible for promotion.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import sys
import time
from typing import Any

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [
    str(ROOT),
    str(ROOT / "src"),
    str(ROOT / "tools" / "evaluation"),
]

from planning.proposal_tail_sidecar_v1 import canonical_json, sha256_file  # noqa: E402
from run_scand_union65_listwise_selector_v1 import (  # noqa: E402
    assign_folds,
    choice_mask,
    load_npz,
    require,
    selection_stats,
    state_weights,
    write_json_atomic,
)


GROUND_TRUTH_SOURCES = (
    ROOT
    / "outputs/cem_development_v1/scand_existing_train_upperbound_pilot_v1_corrected"
    / "pilot_candidate_labels.npz",
    ROOT
    / "outputs/scand_domain_adaptation_v1/fresh_outer_v1/outcomes"
    / "fresh_outer_outcomes.npz",
    ROOT
    / "outputs/cem_development_v1"
    / "scand_new_routefit10_v4_development_pilot_v2_strict_identity"
    / "pilot_candidate_labels.npz",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--ensemble-size", type=int, default=2)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_ground_truth(record: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    mapping: dict[str, np.ndarray] = {}
    sources = []
    for path in GROUND_TRUTH_SOURCES:
        require(path.is_file(), f"ground-truth source missing: {path}")
        with np.load(path, allow_pickle=False) as payload:
            key = payload["record_key"].astype(str)
            value = payload["ground_truth_local_m"].astype(np.float32)
        require(value.shape == (len(key), 4, 2), f"bad truth shape: {path}")
        for name, path_value in zip(key.tolist(), value):
            require(name not in mapping, f"duplicate truth record: {name}")
            mapping[name] = path_value
        sources.append(
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "states": int(len(key)),
            }
        )
    missing = [name for name in record.astype(str) if name not in mapping]
    require(not missing, f"missing ground truth for {len(missing)} records")
    return np.stack([mapping[name] for name in record.astype(str)]), {
        "sources": sources,
        "combined_records": int(len(mapping)),
    }


def masked_moments(feature: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    weight = mask.astype(np.float32)[..., None]
    count = weight.sum(axis=1).clip(min=1.0)
    mean = (feature * weight).sum(axis=1) / count
    variance = (((feature - mean[:, None]) ** 2) * weight).sum(axis=1) / count
    return mean.astype(np.float32), np.sqrt(np.maximum(variance, 0.0)).astype(np.float32)


def one_hot(value: np.ndarray) -> tuple[np.ndarray, list[str]]:
    names = sorted(np.unique(value.astype(str)).tolist())
    index = {name: position for position, name in enumerate(names)}
    result = np.zeros((len(value), len(names)), dtype=np.float32)
    result[np.arange(len(value)), [index[name] for name in value.astype(str)]] = 1.0
    return result, names


def state_descriptor(data: dict[str, np.ndarray]) -> tuple[np.ndarray, dict[str, Any]]:
    feature = data["runtime_pair_feature"].astype(np.float32)
    states = np.arange(len(feature))
    formal = data["formal_v70_index"].astype(np.int64)
    mask = choice_mask(data, states)
    formal_feature = feature[states, formal]
    mean, std = masked_moments(feature, mask)
    cohort, cohort_names = one_hot(data["cohort"])
    robot, robot_names = one_hot(data["robot"])
    descriptor = np.concatenate((formal_feature, mean, std, cohort, robot), axis=1)
    require(np.isfinite(descriptor).all(), "nonfinite state descriptor")
    return descriptor.astype(np.float32), {
        "dimension": int(descriptor.shape[1]),
        "blocks": {
            "formal_runtime_pair_feature": int(formal_feature.shape[1]),
            "masked_candidate_set_mean": int(mean.shape[1]),
            "masked_candidate_set_std": int(std.shape[1]),
            "cohort_onehot": int(cohort.shape[1]),
            "robot_onehot": int(robot.shape[1]),
        },
        "cohort_names": cohort_names,
        "robot_names": robot_names,
    }


class PathResidualNetwork(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dimension, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 8),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


def fit_path_model(
    descriptor: np.ndarray,
    residual: np.ndarray,
    routes: np.ndarray,
    train: np.ndarray,
    device: torch.device,
    seed: int,
    epochs: int,
) -> tuple[PathResidualNetwork, np.ndarray, np.ndarray, dict[str, float]]:
    set_seed(seed)
    x = descriptor[train].astype(np.float32)
    mean = x.mean(axis=0)
    std = np.maximum(x.std(axis=0), 1.0e-5)
    x = ((x - mean) / std).astype(np.float32)
    y = (residual[train].reshape(len(train), 8) / 0.1).astype(np.float32)
    weight = state_weights(routes, train)
    model = PathResidualNetwork(x.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8.0e-4, weight_decay=3.0e-4)
    x_tensor = torch.from_numpy(x)
    y_tensor = torch.from_numpy(y)
    weight_tensor = torch.from_numpy(weight)
    rng = np.random.default_rng(seed)
    batch_size = 64
    final_loss = math.nan
    model.train()
    for _ in range(epochs):
        order = rng.permutation(len(train))
        total = 0.0
        total_weight = 0.0
        for start in range(0, len(order), batch_size):
            index = torch.from_numpy(order[start : start + batch_size])
            xb = x_tensor[index].to(device)
            yb = y_tensor[index].to(device)
            wb = weight_tensor[index].to(device)
            predicted = model(xb)
            per_state = F.smooth_l1_loss(
                predicted, yb, reduction="none", beta=0.5
            ).mean(dim=1)
            loss = (per_state * wb).sum() / wb.sum()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            local_weight = float(wb.sum().detach().cpu())
            total += float(loss.detach().cpu()) * local_weight
            total_weight += local_weight
        final_loss = total / max(total_weight, 1.0)
    return model, mean.astype(np.float32), std.astype(np.float32), {
        "states": int(len(train)),
        "routes": int(len(np.unique(routes[train]))),
        "last_epoch_loss": float(final_loss),
    }


def predict_path_model(
    model: PathResidualNetwork,
    mean: np.ndarray,
    std: np.ndarray,
    descriptor: np.ndarray,
    formal_path: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    x = ((descriptor - mean) / std).astype(np.float32)
    result = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(x), 256):
            result.append(
                model(torch.from_numpy(x[start : start + 256]).to(device))
                .cpu()
                .numpy()
                .astype(np.float32)
            )
    residual = np.concatenate(result).reshape(-1, 4, 2) * 0.1
    return (formal_path + residual).astype(np.float32)


def candidate_distance(candidates_m: np.ndarray, path_m: np.ndarray) -> np.ndarray:
    squared = ((candidates_m - path_m[:, None]) ** 2).sum(axis=-1)
    return np.sqrt(squared.mean(axis=-1)).astype(np.float32)


def evaluate_paths(
    predicted: np.ndarray,
    truth: np.ndarray,
    data: dict[str, np.ndarray],
) -> dict[str, Any]:
    states = np.arange(len(predicted))
    mask = choice_mask(data, states)
    distance = candidate_distance(
        data["candidates_feature_waypoint_units"].astype(np.float32) * 0.25,
        predicted,
    )
    selected = np.argmin(np.where(mask, distance, np.inf), axis=1).astype(np.int64)
    stats = selection_stats(data, states, selected)
    oracle_gain = float(
        np.max(
            np.where(
                mask,
                data["ate_gain_vs_formal_v70_m"].astype(np.float32),
                -np.inf,
            ),
            axis=1,
        ).mean()
    )
    stats["ate_oracle_gain_ratio"] = (
        stats["ate_gain_m"] / oracle_gain if oracle_gain > 0.0 else None
    )
    stats["path_rmse_m"] = float(
        np.sqrt(np.mean(np.sum((predicted - truth) ** 2, axis=-1)))
    )
    oracle_index = np.argmax(
        np.where(
            mask,
            data["ate_gain_vs_formal_v70_m"].astype(np.float32),
            -np.inf,
        ),
        axis=1,
    )
    stats["exact_ate_oracle_recall"] = float((selected == oracle_index).mean())
    order = np.argsort(np.where(mask, distance, np.inf), axis=1)
    top_k = {}
    true_gain = data["ate_gain_vs_formal_v70_m"].astype(np.float32)
    oracle_per_state = np.max(np.where(mask, true_gain, -np.inf), axis=1)
    opportunity = oracle_per_state > 0.0
    for k in (1, 3, 5, 10):
        local = order[:, :k]
        best = np.max(np.take_along_axis(true_gain, local, axis=1), axis=1)
        ratio = np.divide(
            np.maximum(best, 0.0),
            oracle_per_state,
            out=np.ones_like(best),
            where=oracle_per_state > 0.0,
        )
        top_k[str(k)] = {
            "mean_best_ate_gain_fraction": float(ratio[opportunity].mean()),
            "exact_oracle_candidate_recall": float(
                (local == oracle_index[:, None]).any(axis=1)[opportunity].mean()
            ),
        }
    stats["ranking"] = top_k
    return stats


def crossfit(
    descriptor: np.ndarray,
    residual: np.ndarray,
    formal_path: np.ndarray,
    truth: np.ndarray,
    data: dict[str, np.ndarray],
    assignment: np.ndarray,
    name: str,
    device: torch.device,
    epochs: int,
    ensemble_size: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    routes = data["route_group"].astype(str)
    predicted = np.full_like(truth, np.nan)
    folds = int(assignment.max()) + 1
    records = []
    for fold in range(folds):
        train = np.flatnonzero(assignment != fold)
        test = np.flatnonzero(assignment == fold)
        members = []
        fits = []
        for member in range(ensemble_size):
            model, mean, std, fit = fit_path_model(
                descriptor,
                residual,
                routes,
                train,
                device,
                20260740 + (0 if name == "route_oof" else 1000) + fold * 10 + member,
                epochs,
            )
            members.append(
                predict_path_model(
                    model,
                    mean,
                    std,
                    descriptor[test],
                    formal_path[test],
                    device,
                )
            )
            fits.append({"member": member, **fit})
            del model
        predicted[test] = np.mean(np.stack(members), axis=0)
        records.append(
            {
                "fold": fold,
                "train_states": int(len(train)),
                "test_states": int(len(test)),
                "train_routes": int(len(np.unique(routes[train]))),
                "test_routes": int(len(np.unique(routes[test]))),
                "route_overlap": int(
                    len(set(routes[train].tolist()) & set(routes[test].tolist()))
                ),
                "fits": fits,
            }
        )
    require(np.isfinite(predicted).all(), f"{name} predictions incomplete")
    return predicted, {"folds": records, "metrics": evaluate_paths(predicted, truth, data)}


def main() -> None:
    args = parse_args()
    table_path = args.table.resolve()
    output = args.output_dir.resolve()
    require(not output.exists(), f"refusing to overwrite: {output}")
    require("validation" not in str(table_path).lower(), "validation-like table rejected")
    device = torch.device(args.device)
    require(device.type != "cuda" or torch.cuda.is_available(), "CUDA unavailable")
    data = load_npz(table_path)
    feature = data["runtime_pair_feature"].astype(np.float32)
    require(feature.shape == (800, 65, 351), f"unexpected table: {feature.shape}")
    truth, truth_audit = load_ground_truth(data["record_key"].astype(str))
    candidates_m = data["candidates_feature_waypoint_units"].astype(np.float32) * 0.25
    recomputed_ate = candidate_distance(candidates_m, truth)
    require(
        np.allclose(recomputed_ate, data["true_metrics_m"][..., 0], atol=2.0e-6),
        "ATE labels are not geometric distance to logged future",
    )
    formal = data["formal_v70_index"].astype(np.int64)
    formal_path = candidates_m[np.arange(len(formal)), formal]
    residual = truth - formal_path
    descriptor, descriptor_audit = state_descriptor(data)
    routes = data["route_group"].astype(str)
    route_fold, route_fold_audit = assign_folds(routes, 5)
    rng = np.random.default_rng(20260740)
    random_fold = np.empty(len(routes), dtype=np.int8)
    random_fold[rng.permutation(len(routes))] = np.arange(len(routes)) % 5

    started = time.perf_counter()
    route_prediction, route_result = crossfit(
        descriptor,
        residual,
        formal_path,
        truth,
        data,
        route_fold,
        "route_oof",
        device,
        args.epochs,
        args.ensemble_size,
    )
    print(
        canonical_json(
            {
                "stage": "route_oof",
                "ate_gain_m": route_result["metrics"]["ate_gain_m"],
                "oracle_ratio": route_result["metrics"]["ate_oracle_gain_ratio"],
                "elapsed_seconds": time.perf_counter() - started,
            }
        ),
        flush=True,
    )
    random_prediction, random_result = crossfit(
        descriptor,
        residual,
        formal_path,
        truth,
        data,
        random_fold,
        "random_oof",
        device,
        args.epochs,
        args.ensemble_size,
    )
    members = []
    fits = []
    states = np.arange(len(routes))
    for member in range(args.ensemble_size):
        model, mean, std, fit = fit_path_model(
            descriptor,
            residual,
            routes,
            states,
            device,
            20262740 + member,
            args.epochs,
        )
        members.append(
            predict_path_model(
                model, mean, std, descriptor, formal_path, device
            )
        )
        fits.append({"member": member, **fit})
        del model
    fitted_prediction = np.mean(np.stack(members), axis=0)
    fitted_result = {
        "fits": fits,
        "metrics": evaluate_paths(fitted_prediction, truth, data),
    }
    formal_result = evaluate_paths(formal_path, truth, data)

    output.mkdir(parents=True, exist_ok=False)
    prediction_path = output / "path_predictions.npz"
    result_path = output / "result.json"
    with prediction_path.open("wb") as handle:
        np.savez_compressed(
            handle,
            record_key=data["record_key"],
            route_group=routes,
            route_fold=route_fold,
            random_fold=random_fold,
            ground_truth_local_m=truth,
            formal_path_m=formal_path,
            route_oof_predicted_path_m=route_prediction,
            random_oof_predicted_path_m=random_prediction,
            fitted_predicted_path_m=fitted_prediction,
        )
    result = {
        "schema": "scand_path_imitation_ceiling_diagnostic_v1",
        "status": "complete_spent_development_diagnostic_only",
        "promotion_allowed": False,
        "sealed_validation_opened": False,
        "question": (
            "Can direct logged-future path imitation recover union65 ATE oracle "
            "value without counterfactual action outcomes?"
        ),
        "states": int(len(routes)),
        "routes": int(len(np.unique(routes))),
        "descriptor": descriptor_audit,
        "label_identity_audit": {
            "ate_recomputed_from_candidate_vs_logged_future": True,
            "maximum_absolute_ate_recompute_error_m": float(
                np.max(np.abs(recomputed_ate - data["true_metrics_m"][..., 0]))
            ),
            **truth_audit,
        },
        "formal_path_as_prediction": formal_result,
        "route_oof": route_result,
        "random_state_oof": random_result,
        "training_resubstitution": fitted_result,
        "route_fold_audit": route_fold_audit,
        "runtime_seconds": float(time.perf_counter() - started),
        "inputs": {
            "table": {"path": str(table_path), "sha256": sha256_file(table_path)},
            "runner": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
        },
        "artifacts": {
            "predictions": {
                "path": str(prediction_path),
                "sha256": sha256_file(prediction_path),
            }
        },
    }
    write_json_atomic(result_path, result)
    print(
        canonical_json(
            {
                "route_oof_ate_gain_m": route_result["metrics"]["ate_gain_m"],
                "random_oof_ate_gain_m": random_result["metrics"]["ate_gain_m"],
                "training_fit_ate_gain_m": fitted_result["metrics"]["ate_gain_m"],
                "result": str(result_path),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
