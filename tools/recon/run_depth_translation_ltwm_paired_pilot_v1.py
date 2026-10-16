#!/usr/bin/env python3
"""Paired pilot for depth-aware translation transport on a frozen LTWM.

The frozen baseline is the passed rotation-transport checkpoint.  Only a
token-depth head and two zero-initialized SE(3)-minus-rotation transport
strengths are trained on the original development routes.
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
import run_rotation_warp_ltwm_paired_pilot_v1 as common
from recon_latents.losses_highres_dual_stable_v1 import (
    HighResDualLatentLossWeights,
    compute_highres_dual_latent_loss,
)
from recon_latents.models_highres_dual_depth_translation_v1 import (
    DepthTranslationResidualConfig,
    FrozenRotationTransportDepthTranslationResidual,
)
from recon_latents.models_highres_dual_rotation_transport_v1 import (
    RotationTransportHighResMultiLayerDualLatentWM,
)
from recon_latents.models_highres_dual_rotation_warp_v1 import (
    RotationWarpHighResDualLatentWMConfig,
)
from train_highres_multilayer_dual_ltwm_v1 import corrupt_actions


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PARENT = (
    ROOT / "outputs/highres_dual_ltwm_v1/raw_resolution_paired_pilot_v1"
)
DEFAULT_ROTATION = (
    ROOT / "outputs/highres_dual_ltwm_v1/"
    "rotation_transport_ltwm_paired_pilot_v1"
)
DEFAULT_TRANSLATION_GEOMETRY = (
    ROOT / "outputs/highres_dual_ltwm_v1/"
    "external_camera_translation_axis_pilot_v1/result.json"
)
DEFAULT_OUTPUT = (
    ROOT / "outputs/highres_dual_ltwm_v1/"
    "depth_translation_ltwm_paired_pilot_v1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--rotation", type=Path, default=DEFAULT_ROTATION)
    parser.add_argument(
        "--translation-geometry",
        type=Path,
        default=DEFAULT_TRANSLATION_GEOMETRY,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--amp", choices=("bf16", "fp16", "off"), default="bf16")
    return parser.parse_args()


def load_frozen_base(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[
    RotationTransportHighResMultiLayerDualLatentWM,
    dict[str, Any],
]:
    payload = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    base = RotationTransportHighResMultiLayerDualLatentWM(
        RotationWarpHighResDualLatentWMConfig(**payload["config"])
    )
    base.load_state_dict(payload["model"], strict=True)
    return base.eval().to(device), payload


def train_adapter(
    model: FrozenRotationTransportDepthTranslationResidual,
    cache: raw_pilot.ShardCache,
    refs: list[raw_pilot.WindowRef],
    schedule: np.ndarray,
    *,
    device: torch.device,
    amp: str,
) -> dict[str, Any]:
    trainable = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable,
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
            trainable,
            1.0,
        )
        optimizer.step()
        with torch.no_grad():
            scale = torch.tanh(
                model.translation_transport_raw_scale
            )
            mean_depth = float(output["predicted_depth_m"].mean())
        rows.append(
            {
                "loss": float(loss.detach()),
                "gradient_norm": float(gradient_norm),
                "semantic_translation_scale": float(scale[0]),
                "geometry_translation_scale": float(scale[1]),
                "mean_depth_m": mean_depth,
                **{
                    key: float(value.detach())
                    for key, value in components.items()
                },
            }
        )
        if step % 100 == 0 or step == len(schedule):
            print(
                common.canonical_json(
                    {
                        "stage": "train_depth_translation",
                        "step": step,
                        "steps": len(schedule),
                        "loss_last_100": float(
                            np.mean([row["loss"] for row in rows[-100:]])
                        ),
                        "translation_scale": [
                            rows[-1]["semantic_translation_scale"],
                            rows[-1]["geometry_translation_scale"],
                        ],
                        "mean_depth_m": mean_depth,
                        "elapsed_seconds": (
                            time.perf_counter() - started
                        ),
                    }
                ),
                flush=True,
            )
    return {
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
        "final_translation_scale": [
            rows[-1]["semantic_translation_scale"],
            rows[-1]["geometry_translation_scale"],
        ],
        "final_mean_depth_m": rows[-1]["mean_depth_m"],
        "trainable_parameter_count": int(
            sum(parameter.numel() for parameter in trainable)
        ),
        "runtime_seconds": time.perf_counter() - started,
    }


@torch.no_grad()
def depth_summary(
    model: FrozenRotationTransportDepthTranslationResidual,
    cache: raw_pilot.ShardCache,
    refs: list[raw_pilot.WindowRef],
    *,
    batch_size: int,
    device: torch.device,
    amp: str,
) -> dict[str, float]:
    values = []
    model.eval()
    for start in range(0, len(refs), batch_size):
        batch = cache.batch(refs[start : start + batch_size])
        semantic_history, geometry_history, _, _, actions = (
            value.to(device, non_blocking=True) for value in batch
        )
        with raw_pilot.amp_context(amp):
            output = model(
                semantic_history,
                geometry_history,
                actions,
            )
        values.append(output["predicted_depth_m"].float().cpu())
    depth = torch.cat(values).numpy()
    return {
        "mean_m": float(depth.mean()),
        "std_m": float(depth.std()),
        "p05_m": float(np.quantile(depth, 0.05)),
        "median_m": float(np.median(depth)),
        "p95_m": float(np.quantile(depth, 0.95)),
    }


def main() -> None:
    args = parse_args()
    parent = args.parent.resolve()
    rotation_dir = args.rotation.resolve()
    geometry_path = args.translation_geometry.resolve()
    output = args.output.resolve()
    result_path = output / "result.json"
    common.require(
        not result_path.exists(),
        f"refusing to overwrite {result_path}",
    )
    common.require(torch.cuda.is_available(), "CUDA unavailable")
    parent_result_path = parent / "result.json"
    rotation_result_path = rotation_dir / "result.json"
    rotation_checkpoint_path = (
        rotation_dir / "rotation_transport_pilot_final.pt"
    )
    for path in (
        parent_result_path,
        rotation_result_path,
        rotation_checkpoint_path,
        geometry_path,
    ):
        common.require(path.is_file(), f"missing input: {path}")
    rotation_result = json.loads(
        rotation_result_path.read_text(encoding="utf-8")
    )
    geometry = json.loads(geometry_path.read_text(encoding="utf-8"))
    common.require(
        rotation_result["status"] == "GO_ROTATION_TRANSPORT_LTWM",
        "rotation transport did not pass development",
    )
    common.require(
        geometry["status"] == "GO_TRANSLATION_ORIENTATION_PROXY",
        "translation orientation proxy did not pass",
    )
    common.require(
        common.sha256_file(rotation_checkpoint_path)
        == rotation_result["checkpoint"]["sha256"],
        "rotation checkpoint hash mismatch",
    )
    orientation = geometry["joint_train"]
    adapter_config = DepthTranslationResidualConfig(
        body_forward_axis_camera=tuple(
            map(float, orientation["axis"])
        ),
        body_lateral_axis_camera=tuple(
            map(float, orientation["lateral_axis_camera"])
        ),
    )

    paths = {
        spec.trajectory_id: (
            parent / "latents" / "legacy320" / f"{spec.trajectory_id}.npz"
        )
        for spec in raw_pilot.DEFAULT_BAGS
    }
    for path in paths.values():
        common.require(path.is_file(), f"missing latent shard: {path}")
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
    device = torch.device(args.device)
    base, base_payload = load_frozen_base(
        rotation_checkpoint_path,
        device,
    )
    seed = int(base_payload["seed"])
    steps = int(base_payload["training"]["steps"])
    batch_size = int(base_payload["training"]["batch_size"])
    schedule = raw_pilot.route_balanced_schedule(
        train_by_trajectory,
        steps=steps,
        batch_size=batch_size,
        seed=seed + 1,
    )
    cache = raw_pilot.ShardCache(paths)
    base_evaluation = raw_pilot.evaluate_model(
        base,
        cache,
        val_refs,
        batch_size=args.eval_batch_size,
        device=device,
        amp=args.amp,
    )
    consistency = common.metric_consistency(
        base_evaluation,
        rotation_result["rotation_transport_held_route_evaluation"],
    )
    common.require(
        max(consistency.values()) <= 1.0e-6,
        f"rotation baseline mismatch: {consistency}",
    )
    model = FrozenRotationTransportDepthTranslationResidual(
        base,
        adapter_config,
    ).to(device)
    common.require(
        torch.equal(
            model.translation_transport_raw_scale,
            torch.zeros(2, device=device),
        ),
        "translation transport is not zero initialized",
    )
    raw_pilot.set_seed(seed)
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    training = train_adapter(
        model,
        cache,
        train_refs,
        schedule,
        device=device,
        amp=args.amp,
    )
    evaluation = raw_pilot.evaluate_model(
        model,
        cache,
        val_refs,
        batch_size=args.eval_batch_size,
        device=device,
        amp=args.amp,
    )
    depths = depth_summary(
        model,
        cache,
        val_refs,
        batch_size=args.eval_batch_size,
        device=device,
        amp=args.amp,
    )
    checkpoint = output / "depth_translation_adapter_final.pt"
    torch.save(
        {
            "adapter": model.adapter_state_dict(),
            "adapter_config": asdict(adapter_config),
            "training": training,
            "evaluation": evaluation,
            "depth_summary": depths,
            "seed": seed,
            "rotation_base": {
                "path": str(rotation_checkpoint_path),
                "sha256": common.sha256_file(rotation_checkpoint_path),
            },
            "translation_geometry": {
                "path": str(geometry_path),
                "sha256": common.sha256_file(geometry_path),
            },
            "outcome_gradients": False,
            "sealed_validation_opened": False,
            "scientific_status": (
                "depth_translation_development_adapter_pilot"
            ),
        },
        checkpoint,
    )
    del model
    torch.cuda.empty_cache()

    baseline = base_evaluation
    value = evaluation
    comparisons = {
        "semantic_h4_nmse_change": (
            value["semantic_h4_nmse"] - baseline["semantic_h4_nmse"]
        ),
        "geometry_h4_nmse_change": (
            value["geometry_h4_nmse"] - baseline["geometry_h4_nmse"]
        ),
        "dual_h4_nmse_change": (
            value["dual_h4_nmse"] - baseline["dual_h4_nmse"]
        ),
        "dual_h4_nmse_relative_change": (
            value["dual_h4_nmse"] / baseline["dual_h4_nmse"] - 1.0
        ),
        "negative_gap_fraction_change": (
            value["negative_gap_fraction_of_true_error"]
            - baseline["negative_gap_fraction_of_true_error"]
        ),
        "pairwise_action_win_rate_change": (
            value["pairwise_action_win_rate"]
            - baseline["pairwise_action_win_rate"]
        ),
        "retrieval_top1_change": (
            value["retrieval_top1"] - baseline["retrieval_top1"]
        ),
        "mean_rank_change": value["mean_rank"] - baseline["mean_rank"],
    }
    checks = {
        "future_dual_h4_improved": (
            comparisons["dual_h4_nmse_change"] < 0.0
        ),
        "future_geometry_h4_not_worse": (
            comparisons["geometry_h4_nmse_change"] <= 0.0
        ),
        "negative_gap_fraction_improved": (
            comparisons["negative_gap_fraction_change"] > 0.0
        ),
        "pairwise_action_win_rate_improved": (
            comparisons["pairwise_action_win_rate_change"] > 0.0
        ),
        "mean_rank_improved": comparisons["mean_rank_change"] < 0.0,
        "depth_is_noncollapsed": depths["std_m"] >= 0.05,
    }
    go = all(checks.values())
    result = {
        "status": (
            "GO_DEPTH_TRANSLATION_LTWM"
            if go
            else "NO_GO_DEPTH_TRANSLATION_LTWM"
        ),
        "interpretation": (
            "metric-depth translation transport improved future latents and "
            "action discrimination beyond frozen rotation transport"
            if go
            else "depth translation transport did not jointly improve future "
            "latents and action discrimination"
        ),
        "protocol": {
            "rotation_base_frozen": True,
            "initial_prediction_exactly_equals_rotation_base": True,
            "trainable_parameter_count": training[
                "trainable_parameter_count"
            ],
            "dino_frozen": True,
            "outcome_labels_loaded": False,
            "ate_or_objective_gradients": False,
            "sealed_validation_opened": False,
            "formal_v70_modified": False,
            "frozen_highres_dual_modified": False,
        },
        "lineage": {
            "runner": str(Path(__file__).resolve()),
            "runner_sha256": common.sha256_file(Path(__file__).resolve()),
            "model_source": str(
                ROOT
                / "src/recon_latents/"
                "models_highres_dual_depth_translation_v1.py"
            ),
            "model_source_sha256": common.sha256_file(
                ROOT
                / "src/recon_latents/"
                "models_highres_dual_depth_translation_v1.py"
            ),
            "rotation_checkpoint": {
                "path": str(rotation_checkpoint_path),
                "sha256": common.sha256_file(rotation_checkpoint_path),
            },
            "translation_geometry": {
                "path": str(geometry_path),
                "sha256": common.sha256_file(geometry_path),
            },
        },
        "rotation_baseline_held_route_evaluation": baseline,
        "depth_translation_held_route_evaluation": value,
        "depth_summary": depths,
        "comparisons": comparisons,
        "checks": checks,
        "training": training,
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": common.sha256_file(checkpoint),
        },
        "runtime_seconds": time.perf_counter() - started,
    }
    common.write_json_atomic(result_path, result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "checks": checks,
                "comparisons": comparisons,
                "depth_summary": depths,
                "training": training,
                "baseline": baseline,
                "depth_translation": value,
                "checkpoint": result["checkpoint"],
                "result": str(result_path),
                "result_sha256": common.sha256_file(result_path),
                "runtime_seconds": result["runtime_seconds"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
