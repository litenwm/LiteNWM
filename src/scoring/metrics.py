"""Route-clustered ranking and selection metrics for independent scorers."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Sequence

import numpy as np


EPS = 1.0e-12


def mean_rank(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(len(array), dtype=np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and array[order[end]] == array[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def spearman(values: Sequence[float], targets: Sequence[float]) -> float:
    x = mean_rank(values)
    y = mean_rank(targets)
    if x.std() <= EPS or y.std() <= EPS:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def kendall_tau_b(values: Sequence[float], targets: Sequence[float]) -> float:
    x = np.asarray(values, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    concordant = discordant = tie_x = tie_y = 0
    for left in range(len(x) - 1):
        dx = x[left] - x[left + 1 :]
        dy = y[left] - y[left + 1 :]
        product = dx * dy
        concordant += int(np.sum(product > 0))
        discordant += int(np.sum(product < 0))
        tie_x += int(np.sum((dx == 0) & (dy != 0)))
        tie_y += int(np.sum((dy == 0) & (dx != 0)))
    denominator = math.sqrt(
        (concordant + discordant + tie_x)
        * (concordant + discordant + tie_y)
    )
    return 0.0 if denominator <= EPS else float((concordant - discordant) / denominator)


def route_bootstrap(
    values: Sequence[float],
    routes: Sequence[str],
    *,
    samples: int,
    seed: int,
) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for value, route in zip(values, routes):
        grouped[str(route)].append(float(value))
    route_values = np.asarray(
        [np.mean(grouped[key]) for key in sorted(grouped)], dtype=np.float64
    )
    rng = np.random.default_rng(seed)
    draws = route_values[
        rng.integers(0, len(route_values), size=(samples, len(route_values)))
    ].mean(axis=1)
    return {
        "mean": float(route_values.mean()),
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
        "positive_route_fraction": float(np.mean(route_values > 0)),
        "routes": int(len(route_values)),
    }


def evaluate_scores(
    scores: np.ndarray,
    utility: np.ndarray,
    group_uid: np.ndarray,
    routes: np.ndarray,
    candidate_index: np.ndarray,
    valid: np.ndarray,
    *,
    bootstrap_samples: int,
    seed: int,
    reference_regret: np.ndarray | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = []
    group_order = []
    for group in group_uid:
        value = str(group)
        if value not in group_order:
            group_order.append(value)
    for group in group_order:
        mask = (group_uid == group) & valid
        group_scores = scores[mask].astype(np.float64)
        group_utility = utility[mask].astype(np.float64)
        group_candidates = candidate_index[mask]
        group_routes = routes[mask]
        selected = int(np.argmax(group_scores))
        oracle = int(np.argmax(group_utility))
        ordered = np.argsort(-group_scores)
        span = float(group_utility.max() - group_utility.min())
        regret = float(group_utility.max() - group_utility[selected])
        rows.append(
            {
                "group_uid": group,
                "route": str(group_routes[selected]),
                "selected_candidate_index": int(group_candidates[selected]),
                "oracle_candidate_index": int(group_candidates[oracle]),
                "spearman": spearman(group_scores, group_utility),
                "kendall_tau_b": kendall_tau_b(group_scores, group_utility),
                "top1": float(selected == oracle),
                "recall_at_3": float(oracle in ordered[:3]),
                "recall_at_5": float(oracle in ordered[:5]),
                "recall_at_8": float(oracle in ordered[:8]),
                "recall_at_12": float(oracle in ordered[:12]),
                "regret": regret,
                "normalized_regret": regret / max(span, EPS),
                "selected_utility": float(group_utility[selected]),
                "selected_minus_mean": float(
                    group_utility[selected] - group_utility.mean()
                ),
            }
        )
    route_values = [row["route"] for row in rows]
    summary = {
        "groups": len(rows),
        "spearman_mean": float(np.mean([row["spearman"] for row in rows])),
        "spearman_bootstrap": route_bootstrap(
            [row["spearman"] for row in rows],
            route_values,
            samples=bootstrap_samples,
            seed=seed,
        ),
        "kendall_tau_b_mean": float(
            np.mean([row["kendall_tau_b"] for row in rows])
        ),
        "kendall_bootstrap": route_bootstrap(
            [row["kendall_tau_b"] for row in rows],
            route_values,
            samples=bootstrap_samples,
            seed=seed + 1,
        ),
        "top1": float(np.mean([row["top1"] for row in rows])),
        "recall_at_3": float(np.mean([row["recall_at_3"] for row in rows])),
        "recall_at_5": float(np.mean([row["recall_at_5"] for row in rows])),
        "recall_at_8": float(np.mean([row["recall_at_8"] for row in rows])),
        "recall_at_12": float(np.mean([row["recall_at_12"] for row in rows])),
        "regret_mean": float(np.mean([row["regret"] for row in rows])),
        "regret_median": float(np.median([row["regret"] for row in rows])),
        "normalized_regret_mean": float(
            np.mean([row["normalized_regret"] for row in rows])
        ),
        "selected_utility_mean": float(
            np.mean([row["selected_utility"] for row in rows])
        ),
        "selected_minus_mean_bootstrap": route_bootstrap(
            [row["selected_minus_mean"] for row in rows],
            route_values,
            samples=bootstrap_samples,
            seed=seed + 2,
        ),
    }
    if reference_regret is not None:
        if len(reference_regret) != len(rows):
            raise ValueError("reference regret does not match group count")
        improvement = reference_regret - np.asarray(
            [row["regret"] for row in rows], dtype=np.float64
        )
        summary["reference_regret_improvement_bootstrap"] = route_bootstrap(
            improvement,
            route_values,
            samples=bootstrap_samples,
            seed=seed + 3,
        )
    return summary, rows


__all__ = [
    "evaluate_scores",
    "kendall_tau_b",
    "mean_rank",
    "route_bootstrap",
    "spearman",
]
