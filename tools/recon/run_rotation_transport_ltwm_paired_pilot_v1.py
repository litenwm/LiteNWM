#!/usr/bin/env python3
"""Paired pilot for zero-initialized analytic rotation transport.

This is the single corrective experiment after the hard-warp pilot showed
large action-identifiability gains but an over-constrained geometry branch.
The model begins exactly at the original LTWM baseline and learns only two
bounded transport strengths from real-future latent supervision.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

import run_raw_resolution_paired_pilot_v1 as raw_pilot
import run_rotation_warp_ltwm_paired_pilot_v1 as hard_pilot
from recon_latents.losses_highres_dual_stable_v1 import (
    HighResDualLatentLossWeights,
    compute_highres_dual_latent_loss,
)
from recon_latents.models_highres_dual_rotation_transport_v1 import (
    RotationTransportHighResMultiLayerDualLatentWM,
)
from recon_latents.models_highres_dual_rotation_warp_v1 import (
    RotationWarpHighResDualLatentWMConfig,
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
DEFAULT_HARD_RESULT = (
    ROOT / "outputs/highres_dual_ltwm_v1/"
    "rotation_warp_ltwm_paired_pilot_v1/result.json"
)
DEFAULT_OUTPUT = (
    ROOT / "outputs/highres_dual_ltwm_v1/"
    "rotation_transport_ltwm_paired_pilot_v1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--geometry", type=Path, default=DEFAULT_GEOMETRY)
    parser.add_argument("--hard-result", type=Path, default=DEFAULT_HARD_RESULT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--amp", choices=("bf16", "fp16", "off"), default="bf16")
    return parser.parse_args()


def initialization_matches_baseline(
    config: RotationWarpHighResDualLatentWMConfig,
    seed: int,
) -> bool:
    raw_pilot.set_seed(seed)
    baseline = HighResMultiLayerDualLatentWM(
        HighResDualLatentWMConfig()
    )
    raw_pilot.set_seed(seed)
    transport = RotationTransportHighResMultiLayerDualLatentWM(config)
    baseline_parameters = dict(baseline.named_parameters())
    transport_parameters = dict(transport.named_parameters())
    if set(transport_parameters) - set(baseline_parameters) != {
        "rotation_transport_raw_scale"
    }:
        return False
    return all(
        torch.equal(
            baseline_parameters[name],
            transport_parameters[name],
        )
        for name in baseline_parameters
    ) and torch.equal(
        transport.rotation_transport_raw_scale,
        torch.zeros(2),
    )


def train_model(
    cache: raw_pilot.ShardCache,
    refs: list[raw_pilot.WindowRef],
    schedule: np.ndarray,
    *,
    config: RotationWarpHighResDualLatentWMConfig,
    seed: int,
    device: torch.device,
    amp: str,
) -> tuple[
    RotationTransportHighResMultiLayerDualLatentWM,
    dict[str, Any],
]:
    raw_pilot.set_seed(seed)
    model = RotationTransportHighResMultiLayerDualLatentWM(config).to(device)
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
                "semantic_transport_scale": float(
                    torch.tanh(
                        model.rotation_transport_raw_scale[0]
                    ).detach()
                ),
                "geometry_transport_scale": float(
                    torch.tanh(
                        model.rotation_transport_raw_scale[1]
                    ).detach()
                ),
                **{
                    key: float(value.detach())
                    for key, value in components.items()
                },
            }
        )
        if step % 100 == 0 or step == len(schedule):
            print(
                hard_pilot.canonical_json(
                    {
                        "stage": "train_rotation_transport",
                        "step": step,
                        "steps": len(schedule),
                        "loss_last_100": float(
                            np.mean([row["loss"] for row in rows[-100:]])
                        ),
                        "transport_scale": [
                            rows[-1]["semantic_transport_scale"],
                            rows[-1]["geometry_transport_scale"],
                        ],
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
        "final_transport_scale": [
            rows[-1]["semantic_transport_scale"],
            rows[-1]["geometry_transport_scale"],
        ],
        "parameter_count": int(
            sum(value.numel() for value in model.parameters())
        ),
        "runtime_seconds": time.perf_counter() - started,
    }


def main() -> None:
    args = parse_args()
    parent = args.parent.resolve()
    geometry_path = args.geometry.resolve()
    hard_result_path = args.hard_result.resolve()
    output = args.output.resolve()
    result_path = output / "result.json"
    hard_pilot.require(
        not result_path.exists(),
        f"refusing to overwrite {result_path}",
    )
    hard_pilot.require(torch.cuda.is_available(), "CUDA unavailable")
    parent_result_path = parent / "result.json"
    baseline_checkpoint_path = parent / "legacy320_pilot_final.pt"
    for path in (
        parent_result_path,
        baseline_checkpoint_path,
        geometry_path,
        hard_result_path,
    ):
        hard_pilot.require(path.is_file(), f"missing input: {path}")
    parent_result = json.loads(parent_result_path.read_text(encoding="utf-8"))
    geometry_result = json.loads(geometry_path.read_text(encoding="utf-8"))
    hard_result = json.loads(hard_result_path.read_text(encoding="utf-8"))
    hard_pilot.require(
        geometry_result["status"] == "GO_ROTATION_ONLY_GEOMETRY_PROXY",
        "rotation geometry proxy did not pass",
    )
    hard_pilot.require(
        hard_result["status"] == "NO_GO_ROTATION_WARP_LTWM",
        "corrective pilot requires the recorded hard-warp no-go",
    )
    expected_baseline_sha = parent_result["checkpoints"]["legacy320"][
        "sha256"
    ]
    hard_pilot.require(
        hard_pilot.sha256_file(baseline_checkpoint_path)
        == expected_baseline_sha,
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
    hard_pilot.require(
        initialization_matches_baseline(config, seed),
        "transport initialization differs from baseline",
    )

    paths = {
        spec.trajectory_id: (
            parent / "latents" / "legacy320" / f"{spec.trajectory_id}.npz"
        )
        for spec in raw_pilot.DEFAULT_BAGS
    }
    for path in paths.values():
        hard_pilot.require(
            path.is_file(),
            f"missing latent shard: {path}",
        )
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

    baseline_model = HighResMultiLayerDualLatentWM(
        HighResDualLatentWMConfig(**baseline_payload["config"])
    ).to(device)
    baseline_model.load_state_dict(baseline_payload["model"], strict=True)
    baseline = raw_pilot.evaluate_model(
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
    consistency = hard_pilot.metric_consistency(
        baseline,
        recorded_baseline,
    )
    hard_pilot.require(
        max(consistency.values()) <= 1.0e-6,
        f"baseline reevaluation mismatch: {consistency}",
    )
    del baseline_model
    torch.cuda.empty_cache()

    started = time.perf_counter()
    model, training = train_model(
        cache,
        train_refs,
        schedule,
        config=config,
        seed=seed,
        device=device,
        amp=args.amp,
    )
    transport = raw_pilot.evaluate_model(
        model,
        cache,
        val_refs,
        batch_size=args.eval_batch_size,
        device=device,
        amp=args.amp,
    )
    checkpoint = output / "rotation_transport_pilot_final.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "config": training["config"],
            "training": training,
            "evaluation": transport,
            "seed": seed,
            "parent_baseline": {
                "path": str(baseline_checkpoint_path),
                "sha256": expected_baseline_sha,
            },
            "rotation_geometry": {
                "path": str(geometry_path),
                "sha256": hard_pilot.sha256_file(geometry_path),
            },
            "hard_warp_result": {
                "path": str(hard_result_path),
                "sha256": hard_pilot.sha256_file(hard_result_path),
            },
            "outcome_gradients": False,
            "sealed_validation_opened": False,
            "scientific_status": (
                "adaptive_development_rotation_transport_pilot"
            ),
        },
        checkpoint,
    )
    del model
    torch.cuda.empty_cache()

    comparisons = {
        "semantic_h4_nmse_change": (
            transport["semantic_h4_nmse"] - baseline["semantic_h4_nmse"]
        ),
        "geometry_h4_nmse_change": (
            transport["geometry_h4_nmse"] - baseline["geometry_h4_nmse"]
        ),
        "dual_h4_nmse_change": (
            transport["dual_h4_nmse"] - baseline["dual_h4_nmse"]
        ),
        "dual_h4_nmse_relative_change": (
            transport["dual_h4_nmse"] / baseline["dual_h4_nmse"] - 1.0
        ),
        "negative_gap_fraction_change": (
            transport["negative_gap_fraction_of_true_error"]
            - baseline["negative_gap_fraction_of_true_error"]
        ),
        "pairwise_action_win_rate_change": (
            transport["pairwise_action_win_rate"]
            - baseline["pairwise_action_win_rate"]
        ),
        "retrieval_top1_change": (
            transport["retrieval_top1"] - baseline["retrieval_top1"]
        ),
        "mean_rank_change": (
            transport["mean_rank"] - baseline["mean_rank"]
        ),
    }
    checks = {
        "future_dual_h4_improved": (
            transport["dual_h4_nmse"] < baseline["dual_h4_nmse"]
        ),
        "future_geometry_h4_not_worse": (
            transport["geometry_h4_nmse"]
            <= baseline["geometry_h4_nmse"]
        ),
        "negative_gap_fraction_improved": (
            transport["negative_gap_fraction_of_true_error"]
            > baseline["negative_gap_fraction_of_true_error"]
        ),
        "pairwise_action_win_rate_improved": (
            transport["pairwise_action_win_rate"]
            > baseline["pairwise_action_win_rate"]
        ),
        "mean_rank_improved": (
            transport["mean_rank"] < baseline["mean_rank"]
        ),
    }
    go = all(checks.values())
    result = {
        "status": (
            "GO_ROTATION_TRANSPORT_LTWM"
            if go
            else "NO_GO_ROTATION_TRANSPORT_LTWM"
        ),
        "interpretation": (
            "soft analytic rotation transport jointly improved held-route "
            "real-future prediction and action discrimination"
            if go
            else "soft analytic rotation transport did not jointly improve "
            "held-route future prediction and action discrimination"
        ),
        "scientific_status": (
            "adaptive development result; held route was already observed "
            "by the hard-warp diagnostic and is not independent confirmation"
        ),
        "protocol": {
            "same_latents_schedule_base_parameters_seed_loss": True,
            "base_initialization_bitwise_equal": True,
            "transport_initialization_exactly_zero": True,
            "extra_trainable_parameters": 2,
            "only_intervention": (
                "learned bounded scales on analytic "
                "warped-current rotation transport"
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
            "runner_sha256": hard_pilot.sha256_file(
                Path(__file__).resolve()
            ),
            "model_source": str(
                ROOT
                / "src/recon_latents/"
                "models_highres_dual_rotation_transport_v1.py"
            ),
            "model_source_sha256": hard_pilot.sha256_file(
                ROOT
                / "src/recon_latents/"
                "models_highres_dual_rotation_transport_v1.py"
            ),
            "parent_result": {
                "path": str(parent_result_path),
                "sha256": hard_pilot.sha256_file(parent_result_path),
            },
            "baseline_checkpoint": {
                "path": str(baseline_checkpoint_path),
                "sha256": expected_baseline_sha,
            },
            "rotation_geometry": {
                "path": str(geometry_path),
                "sha256": hard_pilot.sha256_file(geometry_path),
            },
            "hard_warp_result": {
                "path": str(hard_result_path),
                "sha256": hard_pilot.sha256_file(hard_result_path),
            },
            "baseline_reevaluation_max_abs_difference": max(
                consistency.values()
            ),
        },
        "baseline_held_route_evaluation": baseline,
        "rotation_transport_held_route_evaluation": transport,
        "comparisons": comparisons,
        "checks": checks,
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": hard_pilot.sha256_file(checkpoint),
        },
        "training": training,
        "runtime_seconds": time.perf_counter() - started,
    }
    hard_pilot.write_json_atomic(result_path, result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "checks": checks,
                "comparisons": comparisons,
                "final_transport_scale": training[
                    "final_transport_scale"
                ],
                "baseline": baseline,
                "rotation_transport": transport,
                "checkpoint": result["checkpoint"],
                "result": str(result_path),
                "result_sha256": hard_pilot.sha256_file(result_path),
                "runtime_seconds": result["runtime_seconds"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
