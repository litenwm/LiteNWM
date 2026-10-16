#!/usr/bin/env python3
"""Measure where frozen formal-v70 leaves CEM oracle value unrealized.

This is a development-only research diagnostic.  It reads the already-spent
four-route SCAND sidecar and its bound labels, then compares the formal-v70
fallback with:

* the frozen K17 pool,
* each CEM round in isolation,
* the cumulative raw-draw pool after every CEM round, and
* the final compressed 65-candidate union.

It never reads sealed validation data and does not modify the planner, scorer,
candidate pool, or CEM random-number stream.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from tools.planning import build_proposal_tail_temporal_pair_v1 as pair  # noqa: E402


TOLERANCE = 1.0e-7


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {name: np.asarray(payload[name]) for name in payload.files}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def metric_values_waypoint_units(
    candidates: np.ndarray,
    ground_truth_local_m: np.ndarray,
) -> np.ndarray:
    value = np.asarray(candidates, dtype=np.float64)
    if value.ndim != 3 or value.shape[-1] != 8:
        raise RuntimeError(f"expected flattened raw draws [N,C,8], got {value.shape}")
    value = value.reshape(*value.shape[:2], 4, 2) * 0.25
    truth = np.asarray(ground_truth_local_m, dtype=np.float64)[:, None]
    error = value - truth
    squared = np.sum(error * error, axis=-1)
    ate = np.sqrt(np.mean(squared, axis=-1))
    endpoint = np.sqrt(squared[..., -1])
    relative = np.diff(value, axis=-2) - np.diff(truth, axis=-2)
    rpe = np.sqrt(np.mean(np.sum(relative * relative, axis=-1), axis=-1))
    objective = 0.4 * ate + 0.6 * endpoint
    return np.stack((ate, rpe, endpoint, objective), axis=-1).astype(np.float32)


def reshape_raw(
    payload: dict[str, np.ndarray],
    states: int,
    ground_truth_local_m: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    state_index = payload["state_index"].astype(np.int64)
    if state_index.min() != 0 or state_index.max() != states - 1:
        raise RuntimeError("raw draw state coverage changed")
    counts = np.bincount(state_index, minlength=states)
    if not np.all(counts == counts[0]):
        raise RuntimeError("raw draw count is not constant across states")
    draws_per_state = int(counts[0])
    draw = payload["draw_clipped"].astype(np.float32).reshape(states, draws_per_state, 8)
    expected_state = np.repeat(np.arange(states, dtype=np.int64), draws_per_state)
    if not np.array_equal(state_index, expected_state):
        raise RuntimeError("raw draws are not in deterministic state-major order")
    valid = (
        payload["physical_guard"].astype(bool)
        & payload["support_mask"].astype(bool)
    ).reshape(states, draws_per_state)
    metrics = metric_values_waypoint_units(draw, ground_truth_local_m)
    return metrics, valid


def select_oracle(
    formal_metrics: np.ndarray,
    metric_blocks: list[np.ndarray],
    mask_blocks: list[np.ndarray],
    route: np.ndarray,
) -> dict[str, Any]:
    states = len(formal_metrics)
    metrics = np.concatenate(
        (formal_metrics[:, None], *metric_blocks),
        axis=1,
    )
    masks = np.concatenate(
        (np.ones((states, 1), dtype=bool), *mask_blocks),
        axis=1,
    )
    selected = np.argmin(np.where(masks, metrics[..., 3], np.inf), axis=1)
    chosen = metrics[np.arange(states), selected]
    gain = formal_metrics - chosen
    dual = (gain[:, 0] > TOLERANCE) & (gain[:, 3] > TOLERANCE)
    harm = (gain[:, 0] < -TOLERANCE) | (gain[:, 3] < -TOLERANCE)
    changed = selected != 0
    by_route = {}
    for name in sorted(np.unique(route)):
        local = route == name
        by_route[str(name)] = {
            "states": int(local.sum()),
            "objective_gain_m": float(gain[local, 3].mean()),
            "ate_gain_m": float(gain[local, 0].mean()),
            "states_changed": int(changed[local].sum()),
            "states_dual": int(dual[local].sum()),
        }
    return {
        "candidate_count_including_formal": int(metrics.shape[1]),
        "valid_candidate_count_including_formal": int(masks.sum()),
        "states": states,
        "states_changed": int(changed.sum()),
        "states_dual": int(dual.sum()),
        "states_harm": int(harm.sum()),
        "objective_gain_m": float(gain[:, 3].mean()),
        "ate_gain_m": float(gain[:, 0].mean()),
        "objective_mean_m": float(chosen[:, 3].mean()),
        "objective_p90_m": float(np.quantile(chosen[:, 3], 0.9)),
        "by_route": by_route,
    }


def main() -> None:
    args = parse_args()
    sidecar = args.sidecar.resolve()
    labels_path = args.labels.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)

    table = load_npz(sidecar / "candidate_table.npz")
    formal = load_npz(sidecar / "formal_selection_snapshot.npz")
    labels = load_npz(labels_path)
    candidate_hash = table["candidate_hash"].astype(str)
    if not np.array_equal(candidate_hash, labels["candidate_hash"].astype(str)):
        raise RuntimeError("candidate/label binding mismatch")

    states = candidate_hash.shape[0]
    true_metrics = labels["true_metrics_m"].astype(np.float32)
    if true_metrics.shape != (states, 65, 4):
        raise RuntimeError(f"unexpected label metric shape: {true_metrics.shape}")
    formal_index = pair.formal_union_index(table, formal)
    formal_metrics = true_metrics[np.arange(states), formal_index]

    state_rows = [
        json.loads(line)
        for line in (sidecar / "states.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    route = np.asarray([str(row["route"]) for row in state_rows])
    if route.shape != (states,):
        raise RuntimeError("state route binding mismatch")

    pools = {
        source: load_npz(sidecar / relative)
        for source, relative in pair.POOL_PATH.items()
    }
    physical, _ = pair.gather_union(table, pools, ("physical_mask",))
    support, _ = pair.gather_union(table, pools, ("support_mask",))
    if physical is None or support is None:
        raise RuntimeError("union masks unavailable")
    union_valid = physical.astype(bool) & support.astype(bool)
    source_round = table["source_round"].astype(np.int64)
    fixed_mask = union_valid & (source_round == -1)

    raw_metric_blocks: list[np.ndarray] = []
    raw_mask_blocks: list[np.ndarray] = []
    stages: dict[str, Any] = {
        "formal_v70": {
            "states": states,
            "objective_mean_m": float(formal_metrics[:, 3].mean()),
            "objective_p90_m": float(np.quantile(formal_metrics[:, 3], 0.9)),
            "ate_mean_m": float(formal_metrics[:, 0].mean()),
            "objective_gain_m": 0.0,
            "ate_gain_m": 0.0,
        },
        "fixed_k17_oracle": select_oracle(
            formal_metrics,
            [true_metrics],
            [fixed_mask],
            route,
        ),
    }

    ground_truth = labels["ground_truth_local_m"].astype(np.float32)
    for iteration in range(3):
        raw_path = sidecar / f"iteration_{iteration:02d}" / "raw_draws.npz"
        raw_metrics, raw_valid = reshape_raw(
            load_npz(raw_path),
            states,
            ground_truth,
        )
        raw_metric_blocks.append(raw_metrics)
        raw_mask_blocks.append(raw_valid)
        stages[f"round_{iteration + 1}_only_oracle"] = select_oracle(
            formal_metrics,
            [raw_metrics],
            [raw_valid],
            route,
        )
        stages[f"through_round_{iteration + 1}_oracle"] = select_oracle(
            formal_metrics,
            [true_metrics, *raw_metric_blocks],
            [fixed_mask, *raw_mask_blocks],
            route,
        )

    stages["union65_oracle"] = select_oracle(
        formal_metrics,
        [true_metrics],
        [union_valid],
        route,
    )

    raw_gain = stages["through_round_3_oracle"]["objective_gain_m"]
    union_gain = stages["union65_oracle"]["objective_gain_m"]
    fixed_gain = stages["fixed_k17_oracle"]["objective_gain_m"]
    result = {
        "schema": "scand_cem_ceiling_research_v1",
        "status": "development_only_complete",
        "scope": "spent four-route SCAND development; not evidence for promotion",
        "question": "Is frozen formal-v70 limited by search, candidate compression, or final selection?",
        "states": states,
        "routes": int(len(np.unique(route))),
        "stages": stages,
        "loss_decomposition_m": {
            "raw_search_oracle_gain": raw_gain,
            "fixed_k17_oracle_gain": fixed_gain,
            "union65_oracle_gain": union_gain,
            "cem_raw_increment_beyond_fixed_k17": raw_gain - fixed_gain,
            "compression_loss_raw_to_union65": raw_gain - union_gain,
            "selector_unrealized_union65_gain": union_gain,
        },
        "decision": {
            "primary_bottleneck": "final_selector",
            "next_experiment": "structured H1-H4 x seed pair encoder on frozen union65 candidates",
            "increase_cem_budget_now": False,
            "reason": (
                "The final selector leaves the entire union65 oracle gain unrealized; "
                "search-budget changes would add candidates before safe relative-to-formal "
                "selection is solved."
            ),
        },
        "inputs": {
            "candidate_table": {
                "path": str((sidecar / "candidate_table.npz").resolve()),
                "sha256": sha256_file(sidecar / "candidate_table.npz"),
            },
            "formal_snapshot": {
                "path": str((sidecar / "formal_selection_snapshot.npz").resolve()),
                "sha256": sha256_file(sidecar / "formal_selection_snapshot.npz"),
            },
            "labels": {
                "path": str(labels_path),
                "sha256": sha256_file(labels_path),
            },
        },
        "formal_v70_mutated": False,
        "sealed_validation_opened": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(canonical_json(result) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(
        canonical_json(
            {
                "status": result["status"],
                "output": str(output),
                "sha256": sha256_file(output),
                "loss_decomposition_m": result["loss_decomposition_m"],
            }
        )
    )


if __name__ == "__main__":
    main()
