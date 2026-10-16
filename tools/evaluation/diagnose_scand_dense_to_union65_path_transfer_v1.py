#!/usr/bin/env python3
"""Transfer ordinary-log path imitation to the spent union65 action table.

Two leakage controls are reported:
* route-disjoint: remove every dense-training route present in union65;
* trajectory-disjoint: remove overlapping trajectories, but permit other
  trajectories from the same route.

Only feature blocks shared exactly by the dense K17 artifacts and the union65
runtime table are used.  This is a spent-development diagnostic, not a
promotion result.
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
    bootstrap_gain,
    choice_mask,
    load_npz,
    require,
    selection_stats,
    write_json_atomic,
)


DENSE_SOURCE = (
    ROOT / "outputs/scand_domain_adaptation_v1/sources/train_fit_source.npz"
)
DENSE_FEATURE = (
    ROOT
    / "outputs/scand_domain_adaptation_v1/features/train_fit"
    / "scand_inner_train_fit_spacing038_features.npz"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--ensemble-size", type=int, default=2)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def summarize(
    value: np.ndarray,
    incumbent: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    flat = value.reshape(value.shape[0], value.shape[1], -1).astype(np.float32)
    row = np.arange(len(flat))
    first = flat[row, incumbent]
    weight = mask.astype(np.float32)[..., None]
    count = weight.sum(axis=1).clip(min=1.0)
    mean = (flat * weight).sum(axis=1) / count
    variance = (((flat - mean[:, None]) ** 2) * weight).sum(axis=1) / count
    std = np.sqrt(np.maximum(variance, 0.0))
    return np.concatenate((first, mean, std), axis=1).astype(np.float32)


def seed_summary(
    ate_seed: np.ndarray,
    rtp_seed: np.ndarray,
    support_seed: np.ndarray,
) -> np.ndarray:
    parts = []
    for value in (
        ate_seed.astype(np.float32),
        rtp_seed.astype(np.float32),
        support_seed.astype(np.float32),
    ):
        parts.extend(
            (
                value.mean(axis=-1, keepdims=True),
                value.std(axis=-1, keepdims=True),
                np.ptp(value, axis=-1, keepdims=True),
            )
        )
    return np.concatenate(parts, axis=-1).astype(np.float32)


def dense_data() -> dict[str, np.ndarray]:
    with np.load(DENSE_SOURCE, allow_pickle=False) as payload:
        source = {
            key: np.asarray(payload[key]).copy()
            for key in (
                "candidates",
                "ground_truth_local_m",
                "record_key",
                "route_group",
                "trajectory",
            )
        }
    with np.load(DENSE_FEATURE, allow_pickle=False) as payload:
        record = payload["record_key"].astype(str)
        require(
            np.array_equal(record, source["record_key"].astype(str)),
            "dense identity mismatch",
        )
        candidates = payload["candidates"].astype(np.float32)
        require(np.allclose(candidates, source["candidates"]), "dense candidate mismatch")
        actions = payload["actions_h4"].astype(np.float32)
        raw = payload["raw_ltwm_features"].astype(np.float32)
        scorer = payload["scorer_v2_aux_features"].astype(np.float32)
        ate_seed = np.moveaxis(payload["ate_by_seed"].astype(np.float32), 0, -1)
        rtp_seed = np.moveaxis(payload["rtp_by_seed"].astype(np.float32), 0, -1)
        support_seed = np.moveaxis(
            payload["support_by_seed"].astype(np.float32), 0, -1
        )
    states = len(candidates)
    incumbent = np.zeros(states, dtype=np.int64)
    mask = np.ones(candidates.shape[:2], dtype=bool)
    seeds = seed_summary(ate_seed, rtp_seed, support_seed)
    descriptor = np.concatenate(
        [
            summarize(value, incumbent, mask)
            for value in (candidates, actions, raw, scorer, seeds)
        ],
        axis=1,
    ).astype(np.float32)
    source["descriptor"] = descriptor
    source["incumbent"] = incumbent
    return source


def union_block(
    feature: np.ndarray,
    names: list[str],
    delta_prefix: str,
    formal_prefix: str,
) -> np.ndarray:
    delta_index = [i for i, name in enumerate(names) if name.startswith(delta_prefix)]
    formal_index = [i for i, name in enumerate(names) if name.startswith(formal_prefix)]
    require(delta_index and len(delta_index) == len(formal_index), f"bad block {delta_prefix}")
    return (feature[..., delta_index] + feature[..., formal_index]).astype(np.float32)


def union_data(table_path: Path) -> dict[str, np.ndarray]:
    data = load_npz(table_path)
    feature = data["runtime_pair_feature"].astype(np.float32)
    names = data["feature_names"].astype(str).tolist()
    states = np.arange(len(feature))
    incumbent = data["formal_v70_index"].astype(np.int64)
    mask = choice_mask(data, states)
    candidates = data["candidates_feature_waypoint_units"].astype(np.float32)
    actions = union_block(feature, names, "delta_actions_h4_", "formal_actions_h4_")
    raw = union_block(feature, names, "delta_ltwm_temporal_", "formal_ltwm_temporal_")
    scorer = union_block(feature, names, "delta_scorer_aux_", "formal_scorer_aux_")
    seeds = union_block(
        feature, names, "delta_seed_disagreement_", "formal_seed_disagreement_"
    )
    descriptor = np.concatenate(
        [
            summarize(value, incumbent, mask)
            for value in (candidates, actions, raw, scorer, seeds)
        ],
        axis=1,
    ).astype(np.float32)
    data["descriptor"] = descriptor
    data["incumbent"] = incumbent
    return data


class TransferPathNetwork(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dimension, 384),
            nn.LayerNorm(384),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(384, 192),
            nn.GELU(),
            nn.Linear(192, 96),
            nn.GELU(),
            nn.Linear(96, 8),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


def route_weights(route: np.ndarray) -> np.ndarray:
    unique, count = np.unique(route.astype(str), return_counts=True)
    lookup = dict(zip(unique.tolist(), count.tolist()))
    weight = np.asarray([1.0 / lookup[name] for name in route.astype(str)], np.float32)
    return weight / weight.mean()


def fit_predict(
    train: dict[str, np.ndarray],
    train_mask: np.ndarray,
    test: dict[str, np.ndarray],
    device: torch.device,
    epochs: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, float]]:
    set_seed(seed)
    x = train["descriptor"][train_mask].astype(np.float32)
    mean = x.mean(axis=0)
    std = np.maximum(x.std(axis=0), 1.0e-5)
    x = ((x - mean) / std).astype(np.float32)
    candidate_m = train["candidates"][train_mask].astype(np.float32) * 0.25
    baseline = candidate_m[
        np.arange(len(candidate_m)), train["incumbent"][train_mask]
    ]
    y = (
        (
            train["ground_truth_local_m"][train_mask].astype(np.float32)
            - baseline
        ).reshape(len(x), 8)
        / 0.1
    ).astype(np.float32)
    weight = route_weights(train["route_group"][train_mask])
    model = TransferPathNetwork(x.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=7.0e-4, weight_decay=5.0e-4)
    x_tensor = torch.from_numpy(x)
    y_tensor = torch.from_numpy(y)
    weight_tensor = torch.from_numpy(weight)
    rng = np.random.default_rng(seed)
    batch_size = 256
    final_loss = math.nan
    model.train()
    for _ in range(epochs):
        order = rng.permutation(len(x))
        total = 0.0
        total_weight = 0.0
        for start in range(0, len(order), batch_size):
            index = torch.from_numpy(order[start : start + batch_size])
            xb = x_tensor[index].to(device)
            yb = y_tensor[index].to(device)
            wb = weight_tensor[index].to(device)
            prediction = model(xb)
            per_state = F.smooth_l1_loss(
                prediction, yb, reduction="none", beta=0.5
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
    test_x = ((test["descriptor"].astype(np.float32) - mean) / std).astype(np.float32)
    outputs = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(test_x), 256):
            outputs.append(
                model(torch.from_numpy(test_x[start : start + 256]).to(device))
                .cpu()
                .numpy()
                .astype(np.float32)
            )
    residual = np.concatenate(outputs).reshape(-1, 4, 2) * 0.1
    test_candidates_m = (
        test["candidates_feature_waypoint_units"].astype(np.float32) * 0.25
    )
    formal_path = test_candidates_m[
        np.arange(len(test_candidates_m)), test["incumbent"]
    ]
    del model
    return formal_path + residual, {
        "seed": seed,
        "train_states": int(train_mask.sum()),
        "train_routes": int(
            len(np.unique(train["route_group"][train_mask].astype(str)))
        ),
        "last_epoch_loss": float(final_loss),
    }


def candidate_distance(candidates_m: np.ndarray, path_m: np.ndarray) -> np.ndarray:
    return np.sqrt(
        np.mean(
            np.sum((candidates_m - path_m[:, None]) ** 2, axis=-1),
            axis=-1,
        )
    ).astype(np.float32)


def evaluate_prediction(
    data: dict[str, np.ndarray],
    predicted: np.ndarray,
    physical_only: bool,
) -> dict[str, Any]:
    states = np.arange(len(predicted))
    candidates_m = (
        data["candidates_feature_waypoint_units"].astype(np.float32) * 0.25
    )
    score = -candidate_distance(candidates_m, predicted)
    mask = (
        choice_mask(data, states)
        if physical_only
        else np.ones(score.shape, dtype=bool)
    )
    formal = data["formal_v70_index"].astype(np.int64)
    mask[np.arange(len(mask)), formal] = True
    selected = np.argmax(np.where(mask, score, -np.inf), axis=1).astype(np.int64)
    stats = selection_stats(data, states, selected)
    true_gain = data["ate_gain_vs_formal_v70_m"].astype(np.float32)
    oracle = np.argmax(np.where(mask, true_gain, -np.inf), axis=1)
    oracle_gain = np.max(np.where(mask, true_gain, -np.inf), axis=1)
    stats["oracle_ate_gain_m"] = float(oracle_gain.mean())
    stats["oracle_gain_ratio"] = (
        stats["ate_gain_m"] / stats["oracle_ate_gain_m"]
        if stats["oracle_ate_gain_m"] > 0.0
        else None
    )
    stats["exact_oracle_recall"] = float((selected == oracle).mean())
    order = np.argsort(np.where(mask, score, -np.inf), axis=1)[:, ::-1]
    opportunity = oracle_gain > 0.0
    ranking = {}
    for k in (1, 3, 5, 10):
        local = order[:, :k]
        best = np.max(np.take_along_axis(true_gain, local, axis=1), axis=1)
        ratio = np.divide(
            np.maximum(best, 0.0),
            oracle_gain,
            out=np.ones_like(best),
            where=oracle_gain > 0.0,
        )
        ranking[str(k)] = {
            "mean_best_ate_gain_fraction": float(ratio[opportunity].mean()),
            "exact_oracle_candidate_recall": float(
                (local == oracle[:, None]).any(axis=1)[opportunity].mean()
            ),
        }
    stats["ranking"] = ranking
    selected_gain = true_gain[np.arange(len(selected)), selected].astype(np.float64)
    selected_gain[selected == formal] = 0.0
    stats["route_cluster_bootstrap_ate_gain"] = bootstrap_gain(
        selected_gain, data["route_group"].astype(str), draws=5000
    )
    return stats


def regime(
    name: str,
    train: dict[str, np.ndarray],
    train_mask: np.ndarray,
    test: dict[str, np.ndarray],
    device: torch.device,
    epochs: int,
    ensemble_size: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    predictions = []
    fits = []
    for member in range(ensemble_size):
        prediction, fit = fit_predict(
            train,
            train_mask,
            test,
            device,
            epochs,
            20260760 + (0 if name == "route_disjoint" else 100) + member,
        )
        predictions.append(prediction)
        fits.append(fit)
    predicted = np.mean(np.stack(predictions), axis=0)
    return predicted, {
        "fits": fits,
        "all_union65_candidates": evaluate_prediction(test, predicted, False),
        "physical_support_pool": evaluate_prediction(test, predicted, True),
    }


def main() -> None:
    args = parse_args()
    table_path = args.table.resolve()
    output = args.output_dir.resolve()
    require(not output.exists(), f"refusing to overwrite: {output}")
    require("validation" not in str(table_path).lower(), "validation-like path rejected")
    device = torch.device(args.device)
    require(device.type != "cuda" or torch.cuda.is_available(), "CUDA unavailable")
    train = dense_data()
    test = union_data(table_path)
    require(
        train["descriptor"].shape[1] == test["descriptor"].shape[1],
        "shared descriptor dimension mismatch",
    )
    union_routes = set(test["route_group"].astype(str).tolist())
    union_trajectories = set(test["trajectory_id"].astype(str).tolist())
    train_routes = train["route_group"].astype(str)
    train_trajectories = train["trajectory"].astype(str)
    route_disjoint_mask = ~np.isin(train_routes, list(union_routes))
    trajectory_disjoint_mask = ~np.isin(train_trajectories, list(union_trajectories))
    require(route_disjoint_mask.sum() > 1000, "too few route-disjoint train states")
    require(trajectory_disjoint_mask.sum() > 1000, "too few trajectory-disjoint states")
    started = time.perf_counter()
    route_prediction, route_result = regime(
        "route_disjoint",
        train,
        route_disjoint_mask,
        test,
        device,
        args.epochs,
        args.ensemble_size,
    )
    print(
        canonical_json(
            {
                "stage": "route_disjoint",
                "train_states": int(route_disjoint_mask.sum()),
                "ate_gain_m": route_result["physical_support_pool"]["ate_gain_m"],
                "elapsed_seconds": time.perf_counter() - started,
            }
        ),
        flush=True,
    )
    trajectory_prediction, trajectory_result = regime(
        "trajectory_disjoint",
        train,
        trajectory_disjoint_mask,
        test,
        device,
        args.epochs,
        args.ensemble_size,
    )
    output.mkdir(parents=True, exist_ok=False)
    prediction_path = output / "path_predictions.npz"
    result_path = output / "result.json"
    with prediction_path.open("wb") as handle:
        np.savez_compressed(
            handle,
            record_key=test["record_key"],
            route_group=test["route_group"],
            route_disjoint_predicted_path_m=route_prediction,
            trajectory_disjoint_predicted_path_m=trajectory_prediction,
        )
    result = {
        "schema": "scand_dense_to_union65_path_transfer_diagnostic_v1",
        "status": "complete_spent_development_diagnostic_only",
        "promotion_allowed": False,
        "counterfactual_action_outcomes_used": False,
        "states": int(len(test["record_key"])),
        "union_routes": int(len(union_routes)),
        "shared_descriptor_dimension": int(train["descriptor"].shape[1]),
        "shared_blocks": [
            "candidate geometry",
            "H4 actions",
            "LTWM rollout summaries",
            "frozen scorer summaries",
            "three-seed mean/std/range",
        ],
        "overlap_audit": {
            "train_routes": int(len(np.unique(train_routes))),
            "union_routes": int(len(union_routes)),
            "overlapping_routes": int(len(set(train_routes.tolist()) & union_routes)),
            "train_trajectories": int(len(np.unique(train_trajectories))),
            "union_trajectories": int(len(union_trajectories)),
            "overlapping_trajectories": int(
                len(set(train_trajectories.tolist()) & union_trajectories)
            ),
            "route_disjoint_train_states": int(route_disjoint_mask.sum()),
            "trajectory_disjoint_train_states": int(trajectory_disjoint_mask.sum()),
        },
        "route_disjoint": route_result,
        "trajectory_disjoint": trajectory_result,
        "runtime_seconds": float(time.perf_counter() - started),
        "inputs": {
            "dense_source": {
                "path": str(DENSE_SOURCE),
                "sha256": sha256_file(DENSE_SOURCE),
            },
            "dense_feature": {
                "path": str(DENSE_FEATURE),
                "sha256": sha256_file(DENSE_FEATURE),
            },
            "union65_table": {
                "path": str(table_path),
                "sha256": sha256_file(table_path),
            },
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
                "route_disjoint_physical_ate_gain_m": route_result[
                    "physical_support_pool"
                ]["ate_gain_m"],
                "route_disjoint_physical_oracle_ratio": route_result[
                    "physical_support_pool"
                ]["oracle_gain_ratio"],
                "trajectory_disjoint_physical_ate_gain_m": trajectory_result[
                    "physical_support_pool"
                ]["ate_gain_m"],
                "trajectory_disjoint_physical_oracle_ratio": trajectory_result[
                    "physical_support_pool"
                ]["oracle_gain_ratio"],
                "result": str(result_path),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
