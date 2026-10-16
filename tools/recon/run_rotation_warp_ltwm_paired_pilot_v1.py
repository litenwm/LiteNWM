#!/usr/bin/env python3
"""Paired held-route pilot for a rotation-warp dual-latent world model.

The baseline is the already completed legacy320 model from the frozen
raw-resolution paired pilot.  This runner reuses exactly the same latent
shards, route-balanced schedule, architecture weights, loss, seed, and held
route.  The only intervention is replacing the no-change output reference
with a fixed analytic rotation warp derived without outcome labels.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

import run_raw_resolution_paired_pilot_v1 as raw_pilot
from recon_latents.losses_highres_dual_stable_v1 import (
    HighResDualLatentLossWeights,
    compute_highres_dual_latent_loss,
)
from recon_latents.models_highres_dual_rotation_warp_v1 import (
    RotationWarpHighResDualLatentWMConfig,
    RotationWarpHighResMultiLayerDualLatentWM,
)
from recon_latents.models_highres_dual_v1 import (
    HighResDualLatentWMConfig,
    HighResMultiLayerDualLatentWM,
)
from train_highres_multilayer_dual_ltwm_v1 import corrupt_actions


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PARENT = (
    ROOT / "outputs/highres_dual_ltwm_v1/raw_resolution_paired_pilot_v1"
)
DEFAULT_GEOMETRY = (
    ROOT / "outputs/highres_dual_ltwm_v1/"
    "external_camera_rotation_proxy_validation_v1/result.json"
)
DEFAULT_OUTPUT = (
    ROOT / "outputs/highres_dual_ltwm_v1/"
    "rotation_warp_ltwm_paired_pilot_v1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--geometry", type=Path, default=DEFAULT_GEOMETRY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--amp", choices=("bf16", "fp16", "off"), default="bf16")
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(canonical_json(value) + "\n", encoding="utf-8")
    temporary.replace(path)


def initialization_matches_baseline(
    warp_config: RotationWarpHighResDualLatentWMConfig,
    seed: int,
) -> bool:
    raw_pilot.set_seed(seed)
    baseline = HighResMultiLayerDualLatentWM(
        HighResDualLatentWMConfig()
    )
    raw_pilot.set_seed(seed)
    warp = RotationWarpHighResMultiLayerDualLatentWM(warp_config)
    baseline_parameters = dict(baseline.named_parameters())
    warp_parameters = dict(warp.named_parameters())
    if baseline_parameters.keys() != warp_parameters.keys():
        return False
    return all(
        torch.equal(baseline_parameters[name], warp_parameters[name])
        for name in baseline_parameters
    )


def train_warp_model(
    cache: raw_pilot.ShardCache,
    refs: list[raw_pilot.WindowRef],
    schedule: np.ndarray,
    *,
    config: RotationWarpHighResDualLatentWMConfig,
    seed: int,
    device: torch.device,
    amp: str,
) -> tuple[RotationWarpHighResMultiLayerDualLatentWM, dict[str, Any]]:
    raw_pilot.set_seed(seed)
    model = RotationWarpHighResMultiLayerDualLatentWM(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=2.0e-4,
        weight_decay=1.0e-4,
    )
    rows = []
    started = time.perf_counter()
    model.train()
    for step, indices in enumerate(schedule, start=1):
        batch = cache.batch([refs[int(index)] for index in indices])
        semantic_history, geometry_history, semantic_future, geometry_future, actions = (
            value.to(device, non_blocking=True) for value in batch
        )
        optimizer.zero_grad(set_to_none=True)
        with raw_pilot.amp_context(amp):
            output = model(
                semantic_history,
                geometry_history,
                actions,
            )
            corrupted_output = None
            if (step - 1) % 4 == 0:
                corrupted_output = model(
                    semantic_history,
                    geometry_history,
                    corrupt_actions(actions, (step - 1) // 4),
                )
            loss, components = compute_highres_dual_latent_loss(
                output,
                semantic_future,
                geometry_future,
                semantic_history,
                geometry_history,
                actions,
                weights=HighResDualLatentLossWeights(),
                corrupted_output=corrupted_output,
            )
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0,
        )
        optimizer.step()
        rows.append(
            {
                "loss": float(loss.detach()),
                "gradient_norm": float(gradient_norm),
                **{
                    key: float(value.detach())
                    for key, value in components.items()
                },
            }
        )
        if step % 100 == 0 or step == len(schedule):
            print(
                canonical_json(
                    {
                        "stage": "train_rotation_warp",
                        "step": step,
                        "steps": len(schedule),
                        "loss_last_100": float(
                            np.mean([row["loss"] for row in rows[-100:]])
                        ),
                        "elapsed_seconds": (
                            time.perf_counter() - started
                        ),
                    }
                ),
                flush=True,
            )
    return model, {
        "config": asdict(config),
        "steps": len(schedule),
        "batch_size": int(schedule.shape[1]),
        "learning_rate": 2.0e-4,
        "weight_decay": 1.0e-4,
        "loss_weights": asdict(HighResDualLatentLossWeights()),
        "mean_loss": float(np.mean([row["loss"] for row in rows])),
        "last_100_loss": float(
            np.mean([row["loss"] for row in rows[-100:]])
        ),
        "mean_gradient_norm": float(
            np.mean([row["gradient_norm"] for row in rows])
        ),
        "parameter_count": int(
            sum(value.numel() for value in model.parameters())
        ),
        "runtime_seconds": time.perf_counter() - started,
    }


def metric_consistency(
    observed: dict[str, Any],
    recorded: dict[str, Any],
) -> dict[str, float]:
    keys = (
        "semantic_h4_nmse",
        "geometry_h4_nmse",
        "dual_h4_nmse",
        "negative_action_gap",
        "negative_gap_fraction_of_true_error",
        "pairwise_action_win_rate",
        "retrieval_top1",
        "mean_reciprocal_rank",
        "mean_rank",
    )
    return {
        key: abs(float(observed[key]) - float(recorded[key]))
        for key in keys
    }


def main() -> None:
    args = parse_args()
    parent = args.parent.resolve()
    geometry_path = args.geometry.resolve()
    output = args.output.resolve()
    result_path = output / "result.json"
    require(not result_path.exists(), f"refusing to overwrite {result_path}")
    require(torch.cuda.is_available(), "CUDA unavailable")
    parent_result_path = parent / "result.json"
    baseline_checkpoint_path = parent / "legacy320_pilot_final.pt"
    require(parent_result_path.is_file(), "missing parent result")
    require(baseline_checkpoint_path.is_file(), "missing baseline checkpoint")
    require(geometry_path.is_file(), "missing rotation geometry result")
    parent_result = json.loads(parent_result_path.read_text(encoding="utf-8"))
    geometry_result = json.loads(geometry_path.read_text(encoding="utf-8"))
    require(
        geometry_result["status"] == "GO_ROTATION_ONLY_GEOMETRY_PROXY",
        "rotation geometry proxy did not pass",
    )
    expected_baseline_sha = parent_result["checkpoints"]["legacy320"][
        "sha256"
    ]
    require(
        sha256_file(baseline_checkpoint_path) == expected_baseline_sha,
        "baseline checkpoint hash mismatch",
    )
    baseline_payload = torch.load(
        baseline_checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    seed = int(baseline_payload["seed"])
    steps = int(baseline_payload["training"]["steps"])
    batch_size = int(baseline_payload["training"]["batch_size"])
    require(
        baseline_payload["mode"] == "legacy320",
        "wrong parent checkpoint mode",
    )

    geometry = geometry_result["joint_train_fit"]
    scale = 448.0 / 640.0
    axis = geometry["axis_camera"]
    config = RotationWarpHighResDualLatentWMConfig(
        image_width=448,
        image_height=336,
        effective_focal_px=float(geometry["focal_px"]) * scale,
        principal_x_px=224.0,
        principal_y_px=168.0,
        camera_yaw_axis_x=float(axis[0]),
        camera_yaw_axis_y=float(axis[1]),
        camera_yaw_axis_z=float(axis[2]),
    )
    require(
        initialization_matches_baseline(config, seed),
        "warp trainable initialization differs from baseline",
    )

    paths = {
        spec.trajectory_id: (
            parent / "latents" / "legacy320" / f"{spec.trajectory_id}.npz"
        )
        for spec in raw_pilot.DEFAULT_BAGS
    }
    for path in paths.values():
        require(path.is_file(), f"missing latent shard: {path}")
    split_by_trajectory = {
        spec.trajectory_id: spec.split for spec in raw_pilot.DEFAULT_BAGS
    }
    train_refs, train_by_trajectory = raw_pilot.window_refs(
        paths,
        split_by_trajectory,
        "train",
    )
    val_refs, _ = raw_pilot.window_refs(
        paths,
        split_by_trajectory,
        "val",
    )
    schedule = raw_pilot.route_balanced_schedule(
        train_by_trajectory,
        steps=steps,
        batch_size=batch_size,
        seed=seed + 1,
    )
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    cache = raw_pilot.ShardCache(paths)

    baseline_config = HighResDualLatentWMConfig(
        **baseline_payload["config"]
    )
    baseline_model = HighResMultiLayerDualLatentWM(
        baseline_config
    ).to(device)
    baseline_model.load_state_dict(baseline_payload["model"], strict=True)
    baseline_evaluation = raw_pilot.evaluate_model(
        baseline_model,
        cache,
        val_refs,
        batch_size=args.eval_batch_size,
        device=device,
        amp=args.amp,
    )
    recorded_baseline = parent_result["mode_results"]["legacy320"][
        "held_route_evaluation"
    ]
    consistency = metric_consistency(
        baseline_evaluation,
        recorded_baseline,
    )
    require(
        max(consistency.values()) <= 1.0e-6,
        f"baseline reevaluation mismatch: {consistency}",
    )
    del baseline_model
    torch.cuda.empty_cache()

    started = time.perf_counter()
    warp_model, training = train_warp_model(
        cache,
        train_refs,
        schedule,
        config=config,
        seed=seed,
        device=device,
        amp=args.amp,
    )
    warp_evaluation = raw_pilot.evaluate_model(
        warp_model,
        cache,
        val_refs,
        batch_size=args.eval_batch_size,
        device=device,
        amp=args.amp,
    )
    checkpoint = output / "rotation_warp_pilot_final.pt"
    torch.save(
        {
            "model": warp_model.state_dict(),
            "config": training["config"],
            "training": training,
            "evaluation": warp_evaluation,
            "seed": seed,
            "parent_baseline": {
                "path": str(baseline_checkpoint_path),
                "sha256": expected_baseline_sha,
            },
            "rotation_geometry": {
                "path": str(geometry_path),
                "sha256": sha256_file(geometry_path),
            },
            "outcome_gradients": False,
            "sealed_validation_opened": False,
            "scientific_status": "rotation_warp_development_pilot",
        },
        checkpoint,
    )
    del warp_model
    torch.cuda.empty_cache()

    baseline = baseline_evaluation
    warp = warp_evaluation
    comparisons = {
        "semantic_h4_nmse_change": (
            warp["semantic_h4_nmse"] - baseline["semantic_h4_nmse"]
        ),
        "geometry_h4_nmse_change": (
            warp["geometry_h4_nmse"] - baseline["geometry_h4_nmse"]
        ),
        "dual_h4_nmse_change": (
            warp["dual_h4_nmse"] - baseline["dual_h4_nmse"]
        ),
        "dual_h4_nmse_relative_change": (
            warp["dual_h4_nmse"] / baseline["dual_h4_nmse"] - 1.0
        ),
        "negative_gap_fraction_change": (
            warp["negative_gap_fraction_of_true_error"]
            - baseline["negative_gap_fraction_of_true_error"]
        ),
        "pairwise_action_win_rate_change": (
            warp["pairwise_action_win_rate"]
            - baseline["pairwise_action_win_rate"]
        ),
        "retrieval_top1_change": (
            warp["retrieval_top1"] - baseline["retrieval_top1"]
        ),
        "mean_rank_change": warp["mean_rank"] - baseline["mean_rank"],
    }
    checks = {
        "future_dual_h4_improved": (
            warp["dual_h4_nmse"] < baseline["dual_h4_nmse"]
        ),
        "future_geometry_h4_improved": (
            warp["geometry_h4_nmse"] < baseline["geometry_h4_nmse"]
        ),
        "negative_gap_fraction_improved": (
            warp["negative_gap_fraction_of_true_error"]
            > baseline["negative_gap_fraction_of_true_error"]
        ),
        "pairwise_action_win_rate_improved": (
            warp["pairwise_action_win_rate"]
            > baseline["pairwise_action_win_rate"]
        ),
        "mean_rank_improved": warp["mean_rank"] < baseline["mean_rank"],
    }
    go = all(checks.values())
    result = {
        "status": (
            "GO_ROTATION_WARP_LTWM"
            if go
            else "NO_GO_ROTATION_WARP_LTWM"
        ),
        "interpretation": (
            "rotation geometry improved real-future latent prediction and "
            "held-route action discrimination"
            if go
            else "rotation geometry did not jointly improve real-future "
            "latent prediction and held-route action discrimination"
        ),
        "protocol": {
            "parent_mode": "legacy320",
            "train_trajectories": [
                spec.trajectory_id
                for spec in raw_pilot.DEFAULT_BAGS
                if spec.split == "train"
            ],
            "held_route_trajectories": [
                spec.trajectory_id
                for spec in raw_pilot.DEFAULT_BAGS
                if spec.split == "val"
            ],
            "train_windows": len(train_refs),
            "held_route_windows": len(val_refs),
            "same_latents_schedule_architecture_parameters_seed_loss": True,
            "only_intervention": (
                "candidate-yaw analytic rotation warp as output reference"
            ),
            "dino_frozen": True,
            "outcome_labels_loaded": False,
            "ate_or_objective_gradients": False,
            "sealed_validation_opened": False,
            "formal_v70_modified": False,
            "frozen_highres_dual_modified": False,
        },
        "lineage": {
            "runner": str(Path(__file__).resolve()),
            "runner_sha256": sha256_file(Path(__file__).resolve()),
            "model_source": str(
                ROOT
                / "src/recon_latents/"
                "models_highres_dual_rotation_warp_v1.py"
            ),
            "model_source_sha256": sha256_file(
                ROOT
                / "src/recon_latents/"
                "models_highres_dual_rotation_warp_v1.py"
            ),
            "parent_result": {
                "path": str(parent_result_path),
                "sha256": sha256_file(parent_result_path),
            },
            "baseline_checkpoint": {
                "path": str(baseline_checkpoint_path),
                "sha256": expected_baseline_sha,
            },
            "rotation_geometry": {
                "path": str(geometry_path),
                "sha256": sha256_file(geometry_path),
            },
            "latent_shards": {
                trajectory_id: {
                    "path": str(path),
                    "sha256": sha256_file(path),
                }
                for trajectory_id, path in paths.items()
            },
            "baseline_reevaluation_max_abs_difference": max(
                consistency.values()
            ),
            "initial_trainable_parameters_bitwise_equal": True,
        },
        "baseline_held_route_evaluation": baseline,
        "rotation_warp_held_route_evaluation": warp,
        "comparisons": comparisons,
        "checks": checks,
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": sha256_file(checkpoint),
        },
        "training": training,
        "runtime_seconds": time.perf_counter() - started,
    }
    write_json_atomic(result_path, result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "checks": checks,
                "comparisons": comparisons,
                "baseline": baseline,
                "rotation_warp": warp,
                "checkpoint": result["checkpoint"],
                "result": str(result_path),
                "result_sha256": sha256_file(result_path),
                "runtime_seconds": result["runtime_seconds"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
