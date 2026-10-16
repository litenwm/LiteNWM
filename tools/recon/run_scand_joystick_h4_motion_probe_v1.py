#!/usr/bin/env python3
"""Held-route H4 test of whether SCAND joystick adds future-motion information."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_scand_command_motion_probe_v1 import Ridge, metrics  # noqa: E402
from run_scand_joystick_motion_probe_v1 import read_bag, transitions  # noqa: E402


HORIZON = 4
LAG_GRID_SECONDS = tuple(float(value) for value in np.arange(0.0, 0.501, 0.05))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-bag", type=Path, action="append", required=True)
    parser.add_argument("--held-bag", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ridge", type=float, default=1.0e-2)
    return parser.parse_args()


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.std() < 1.0e-10 or y.std() < 1.0e-10:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def select_axes(train_steps: list[dict[str, np.ndarray]]) -> tuple[list[int], dict[str, Any]]:
    dims = train_steps[0]["joystick"].shape[1]
    per_axis: dict[str, Any] = {}
    selected: list[int] = []
    for axis in range(dims):
        per_target = []
        for target_dim in range(3):
            correlations = [
                safe_corr(row["joystick"][:, axis], row["target"][:, target_dim])
                for row in train_steps
            ]
            consistent_sign = all(value >= 0 for value in correlations) or all(
                value <= 0 for value in correlations
            )
            score = min(abs(value) for value in correlations) if consistent_sign else 0.0
            per_target.append(
                {
                    "target_dim": target_dim,
                    "route_correlations": correlations,
                    "consistent_score": score,
                }
            )
        best = max(item["consistent_score"] for item in per_target)
        if best >= 0.5:
            selected.append(axis)
        per_axis[str(axis)] = {"targets": per_target, "best_consistent_score": best}
    if not selected:
        raise RuntimeError("no joystick axis has stable train-route correlation >= 0.5")
    return selected, per_axis


def integrate_h4(actions: np.ndarray) -> np.ndarray:
    result = np.zeros(3, dtype=np.float64)
    for dx_local, dy_local, dyaw in actions:
        cosine = np.cos(result[2])
        sine = np.sin(result[2])
        result[0] += cosine * dx_local - sine * dy_local
        result[1] += sine * dx_local + cosine * dy_local
        result[2] = (result[2] + dyaw + np.pi) % (2.0 * np.pi) - np.pi
    return result


def windows(step: dict[str, np.ndarray], axes: list[int]) -> dict[str, np.ndarray]:
    count = len(step["target"]) - HORIZON + 1
    state_rows = []
    full_rows = []
    targets = []
    for start in range(count):
        stop = start + HORIZON
        state = step["state"][start]
        command = step["joystick"][start:stop, axes].reshape(-1)
        command_delta = np.diff(command.reshape(HORIZON, len(axes)), axis=0, prepend=command.reshape(HORIZON, len(axes))[:1]).reshape(-1)
        state_rows.append(state)
        full_rows.append(np.concatenate((state, command, command_delta)))
        targets.append(integrate_h4(step["target"][start:stop]))
    return {
        "state": np.asarray(state_rows),
        "full": np.asarray(full_rows),
        "target": np.asarray(targets),
    }


def score(
    train: list[dict[str, np.ndarray]],
    held: dict[str, np.ndarray],
    key: str,
    ridge: float,
) -> dict[str, Any]:
    x = np.concatenate([row[key] for row in train])
    y = np.concatenate([row["target"] for row in train])
    prediction = Ridge(ridge).fit(x, y).predict(held[key])
    return metrics(held["target"], prediction, y.mean(axis=0), y.var(axis=0))


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    if len(args.train_bag) < 2:
        raise ValueError("two train routes are required")
    raw_train = [read_bag(path) for path in args.train_bag]
    raw_held = read_bag(args.held_bag)

    lag_rows = []
    for lag in LAG_GRID_SECONDS:
        train_steps = [transitions(raw, lag) for raw in raw_train]
        axes, _ = select_axes(train_steps)
        loro_scores = []
        for held_index in range(len(raw_train)):
            fit = [
                windows(step, axes)
                for index, step in enumerate(train_steps)
                if index != held_index
            ]
            held = windows(train_steps[held_index], axes)
            loro_scores.append(score(fit, held, "full", args.ridge)["mean_nmse"])
        lag_rows.append(
            {
                "lag_seconds": lag,
                "axes": axes,
                "mean_loro_nmse": float(np.mean(loro_scores)),
            }
        )
    selected = min(lag_rows, key=lambda row: (row["mean_loro_nmse"], row["lag_seconds"]))
    selected_lag = float(selected["lag_seconds"])
    train_steps = [transitions(raw, selected_lag) for raw in raw_train]
    axes, axis_evidence = select_axes(train_steps)
    train = [windows(step, axes) for step in train_steps]
    held = windows(transitions(raw_held, selected_lag), axes)

    state_score = score(train, held, "state", args.ridge)
    full_score = score(train, held, "full", args.ridge)
    gain = (state_score["mean_nmse"] - full_score["mean_nmse"]) / max(
        state_score["mean_nmse"], 1.0e-10
    )
    result = {
        "status": "GO_JOYSTICK_H4_MOTION" if gain >= 0.05 else "NO_GO_JOYSTICK_H4_MOTION",
        "interpretation": (
            "train-selected joystick axes add held-route H4 motion information"
            if gain >= 0.05
            else "joystick does not add stable held-route H4 information beyond current motion state"
        ),
        "protocol": {
            "train_bags": [str(path) for path in args.train_bag],
            "held_bag": str(args.held_bag),
            "horizon": HORIZON,
            "selected_lag_seconds": selected_lag,
            "selected_axes": axes,
            "lag_and_axes_selected_on_train_routes_only": True,
            "ate_or_objective_loaded": False,
            "sealed_validation_opened": False,
            "formal_v70_modified": False,
            "frozen_highres_dual_modified": False,
        },
        "axis_selection": axis_evidence,
        "lag_selection": lag_rows,
        "sample_counts": {
            "train_windows": int(sum(len(row["target"]) for row in train)),
            "held_windows": int(len(held["target"])),
        },
        "held_route": {
            "state_only_ridge": state_score,
            "state_plus_joystick_h4_ridge": full_score,
            "relative_nmse_gain_from_joystick": float(gain),
        },
        "runtime_seconds": float(time.perf_counter() - started),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
