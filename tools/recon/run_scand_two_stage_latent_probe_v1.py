#!/usr/bin/env python3
"""Feed a held-route joystick-to-motion model into the frozen high-res LTWM."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from recon_latents.models_highres_dual_v1 import (  # noqa: E402
    HighResDualLatentWMConfig,
    HighResMultiLayerDualLatentWM,
)
from run_scand_command_motion_probe_v1 import Ridge, metrics  # noqa: E402


HISTORY = 4
HORIZON = 4
JOY_AXES = (0, 4)
JOY_LAG_SECONDS = 0.5
JOY_TOPIC = "/joystick"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-pair",
        nargs=2,
        metavar=("BAG", "LATENT"),
        action="append",
        required=True,
    )
    parser.add_argument(
        "--held-pair",
        nargs=2,
        metavar=("BAG", "LATENT"),
        required=True,
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--ridge", type=float, default=1.0e-2)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_joystick(path: Path) -> np.ndarray:
    from rosbags.highlevel import AnyReader
    from rosbags.typesys import Stores, get_typestore

    typestore = get_typestore(Stores.ROS1_NOETIC)
    rows: list[tuple[float, ...]] = []
    with AnyReader([path], default_typestore=typestore) as reader:
        connections = [
            connection
            for connection in reader.connections
            if connection.topic == JOY_TOPIC
        ]
        if len(connections) != 1:
            raise ValueError(f"{path.name}: expected one {JOY_TOPIC} connection")
        connection = connections[0]
        for _, timestamp, rawdata in reader.messages(connections=connections):
            message = reader.deserialize(rawdata, connection.msgtype)
            rows.append((float(timestamp) * 1.0e-9, *map(float, message.axes)))
    return np.asarray(rows, dtype=np.float64)


def held_value_at(stream: np.ndarray, query: np.ndarray) -> np.ndarray:
    indices = np.searchsorted(stream[:, 0], query, side="right") - 1
    indices = np.clip(indices, 0, len(stream) - 1)
    return stream[indices, 1:]


def load_route(bag: Path, latent: Path) -> dict[str, Any]:
    joystick = read_joystick(bag)
    with np.load(latent, allow_pickle=False) as shard:
        semantic = shard["semantic_latents"].astype(np.float32)
        geometry = shard["geometry_latents"].astype(np.float32)
        actions = shard["actions_continuous"].astype(np.float64)
        timestamps = shard["timestamps_ns"].astype(np.float64) * 1.0e-9
        trajectory_id = str(shard["trajectory_id"])
    if len(actions) != len(timestamps) - 1:
        raise ValueError(f"{latent}: action/frame mismatch")
    query = 0.5 * (timestamps[:-1] + timestamps[1:]) - JOY_LAG_SECONDS
    joystick_per_transition = held_value_at(joystick, query)[:, JOY_AXES]

    state_rows = []
    full_rows = []
    target_rows = []
    starts = []
    max_start = len(timestamps) - HISTORY - HORIZON
    for history_start in range(max_start + 1):
        t = history_start + HISTORY - 1
        past = actions[t - (HISTORY - 1) : t]
        future_joy = joystick_per_transition[t : t + HORIZON]
        joy_delta = np.diff(future_joy, axis=0, prepend=future_joy[:1])
        state = past.reshape(-1)
        state_rows.append(state)
        full_rows.append(np.concatenate((state, future_joy.reshape(-1), joy_delta.reshape(-1))))
        target_rows.append(actions[t : t + HORIZON].reshape(-1))
        starts.append(history_start)
    return {
        "trajectory_id": trajectory_id,
        "semantic": semantic,
        "geometry": geometry,
        "actions": actions,
        "timestamps": timestamps,
        "state": np.asarray(state_rows),
        "full": np.asarray(full_rows),
        "target": np.asarray(target_rows),
        "starts": np.asarray(starts, dtype=np.int64),
    }


def endpoint(actions: np.ndarray) -> np.ndarray:
    flat = actions.reshape(-1, HORIZON, 3)
    result = np.zeros((len(flat), 3), dtype=np.float64)
    for index, sequence in enumerate(flat):
        x = y = yaw = 0.0
        for dx_local, dy_local, dyaw in sequence:
            cosine = np.cos(yaw)
            sine = np.sin(yaw)
            x += cosine * dx_local - sine * dy_local
            y += sine * dx_local + cosine * dy_local
            yaw = (yaw + dyaw + np.pi) % (2.0 * np.pi) - np.pi
        result[index] = (x, y, yaw)
    return result


def action_report(
    train_targets: np.ndarray,
    target: np.ndarray,
    prediction: np.ndarray,
) -> dict[str, Any]:
    base = metrics(
        target,
        prediction,
        train_targets.mean(axis=0),
        train_targets.var(axis=0),
    )
    target_endpoint = endpoint(target)
    predicted_endpoint = endpoint(prediction)
    delta = predicted_endpoint - target_endpoint
    delta[:, 2] = (delta[:, 2] + np.pi) % (2.0 * np.pi) - np.pi
    base["endpoint_translation_mae_m"] = float(
        np.linalg.vector_norm(delta[:, :2], axis=1).mean()
    )
    base["endpoint_yaw_mae_deg"] = float(np.degrees(np.abs(delta[:, 2])).mean())
    return base


def flatten_tokens(value: np.ndarray) -> np.ndarray:
    return value.reshape(*value.shape[:-3], value.shape[-3] * value.shape[-2], value.shape[-1])


def latent_nmse(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    error = (prediction.float() - target.float()).square().mean(dim=(-1, -2))
    denominator = target.float().square().mean(dim=(-1, -2)).clamp_min(1.0e-8)
    return error / denominator


def evaluate_latents(
    route: dict[str, Any],
    checkpoint: Path,
    action_sets: dict[str, np.ndarray],
    batch_size: int,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the frozen LTWM probe")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = HighResDualLatentWMConfig(**payload["config"])
    model = HighResMultiLayerDualLatentWM(config).cuda().eval()
    model.load_state_dict(payload["model"], strict=True)

    names = list(action_sets)
    dual_rows = {name: [] for name in names}
    semantic_rows = {name: [] for name in names}
    geometry_rows = {name: [] for name in names}
    baseline_rows = []
    starts = route["starts"]
    with torch.no_grad():
        for offset in range(0, len(starts), batch_size):
            batch_starts = starts[offset : offset + batch_size]
            semantic_history = torch.from_numpy(
                np.stack(
                    [
                        flatten_tokens(route["semantic"][start : start + HISTORY])
                        for start in batch_starts
                    ]
                )
            ).cuda()
            geometry_history = torch.from_numpy(
                np.stack(
                    [
                        flatten_tokens(route["geometry"][start : start + HISTORY])
                        for start in batch_starts
                    ]
                )
            ).cuda()
            semantic_future = torch.from_numpy(
                np.stack(
                    [
                        flatten_tokens(
                            route["semantic"][
                                start + HISTORY : start + HISTORY + HORIZON
                            ]
                        )
                        for start in batch_starts
                    ]
                )
            ).cuda()
            geometry_future = torch.from_numpy(
                np.stack(
                    [
                        flatten_tokens(
                            route["geometry"][
                                start + HISTORY : start + HISTORY + HORIZON
                            ]
                        )
                        for start in batch_starts
                    ]
                )
            ).cuda()
            candidates = torch.from_numpy(
                np.stack(
                    [
                        action_sets[name][offset : offset + len(batch_starts)].reshape(
                            len(batch_starts), HORIZON, 3
                        )
                        for name in names
                    ],
                    axis=1,
                ).astype(np.float32)
            ).cuda()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = model(semantic_history, geometry_history, candidates)
            predicted_semantic = output["predicted_semantic_tokens"]
            predicted_geometry = output["predicted_geometry_tokens"]
            for candidate_index, name in enumerate(names):
                semantic_score = latent_nmse(
                    predicted_semantic[:, candidate_index],
                    semantic_future,
                )[:, -1]
                geometry_score = latent_nmse(
                    predicted_geometry[:, candidate_index],
                    geometry_future,
                )[:, -1]
                semantic_rows[name].append(semantic_score.cpu().numpy())
                geometry_rows[name].append(geometry_score.cpu().numpy())
                dual_rows[name].append(
                    (0.5 * (semantic_score + geometry_score)).cpu().numpy()
                )
            current_semantic = semantic_history[:, -1, None].expand_as(semantic_future)
            current_geometry = geometry_history[:, -1, None].expand_as(geometry_future)
            baseline_rows.append(
                (
                    0.5
                    * (
                        latent_nmse(current_semantic, semantic_future)[:, -1]
                        + latent_nmse(current_geometry, geometry_future)[:, -1]
                    )
                ).cpu().numpy()
            )
    baseline = np.concatenate(baseline_rows)
    result = {
        "no_change_dual_h4_nmse": float(baseline.mean()),
        "windows": int(len(baseline)),
        "action_sources": {},
    }
    for name in names:
        dual = np.concatenate(dual_rows[name])
        result["action_sources"][name] = {
            "semantic_h4_nmse": float(np.concatenate(semantic_rows[name]).mean()),
            "geometry_h4_nmse": float(np.concatenate(geometry_rows[name]).mean()),
            "dual_h4_nmse": float(dual.mean()),
            "relative_improvement_over_no_change": float(
                (baseline.mean() - dual.mean()) / baseline.mean()
            ),
        }
    return result


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    if len(args.train_pair) < 2:
        raise ValueError("at least two train pairs are required")
    train = [
        load_route(Path(bag), Path(latent))
        for bag, latent in args.train_pair
    ]
    held = load_route(Path(args.held_pair[0]), Path(args.held_pair[1]))
    train_target = np.concatenate([route["target"] for route in train])

    state_x = np.concatenate([route["state"] for route in train])
    full_x = np.concatenate([route["full"] for route in train])
    state_model = Ridge(args.ridge).fit(state_x, train_target)
    full_model = Ridge(args.ridge).fit(full_x, train_target)
    state_prediction = state_model.predict(held["state"])
    full_prediction = full_model.predict(held["full"])
    ground_truth = held["target"]
    zero = np.zeros_like(ground_truth)

    action_sets = {
        "ground_truth_achieved_motion": ground_truth,
        "state_only_predicted_motion": state_prediction,
        "joystick_plus_state_predicted_motion": full_prediction,
        "zero_motion": zero,
    }
    latent = evaluate_latents(held, args.checkpoint, action_sets, args.batch_size)
    state_latent = latent["action_sources"]["state_only_predicted_motion"]["dual_h4_nmse"]
    joystick_latent = latent["action_sources"]["joystick_plus_state_predicted_motion"]["dual_h4_nmse"]
    latent_gain = (state_latent - joystick_latent) / max(state_latent, 1.0e-10)

    result = {
        "status": (
            "GO_TWO_STAGE_LATENT"
            if latent_gain >= 0.02
            else "NO_GO_TWO_STAGE_LATENT"
        ),
        "interpretation": (
            "real joystick improves held-route future-latent prediction through predicted achieved motion"
            if latent_gain >= 0.02
            else "motion improvement does not transfer materially through the frozen LTWM"
        ),
        "protocol": {
            "train_pairs": args.train_pair,
            "held_pair": args.held_pair,
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": sha256(args.checkpoint),
            "joystick_axes": list(JOY_AXES),
            "joystick_lag_seconds": JOY_LAG_SECONDS,
            "axes_and_lag_frozen_from_train_only_h4_probe": True,
            "ate_or_objective_loaded": False,
            "sealed_validation_opened": False,
            "formal_v70_modified": False,
            "frozen_highres_dual_modified": False,
        },
        "sample_counts": {
            "train_windows": int(len(train_target)),
            "held_windows": int(len(ground_truth)),
        },
        "held_action_prediction": {
            "state_only": action_report(train_target, ground_truth, state_prediction),
            "joystick_plus_state": action_report(train_target, ground_truth, full_prediction),
        },
        "held_future_latent": latent,
        "relative_dual_h4_nmse_gain_from_joystick": float(latent_gain),
        "runtime_seconds": float(time.perf_counter() - started),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
