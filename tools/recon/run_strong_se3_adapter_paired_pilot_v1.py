#!/usr/bin/env python3
"""Paired pilot for calibrated SE(3) residuals on the frozen strong LTWM."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import torch

import run_depth_translation_ltwm_paired_pilot_v1 as depth_pilot
import run_raw_resolution_paired_pilot_v1 as raw_pilot
import run_rotation_warp_ltwm_paired_pilot_v1 as common
from recon_latents.models_highres_dual_depth_translation_v1 import (
    DepthTranslationResidualConfig,
)
from recon_latents.models_highres_dual_rotation_transport_v1 import (
    RotationTransportHighResMultiLayerDualLatentWM,
)
from recon_latents.models_highres_dual_rotation_warp_v1 import (
    RotationWarpHighResDualLatentWMConfig,
)
from recon_latents.models_highres_dual_strong_se3_adapter_v1 import (
    FrozenStrongHighResCalibratedSE3Residual,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LATENTS = (
    ROOT / "outputs/highres_dual_ltwm_v1/raw_resolution_paired_pilot_v1"
)
DEFAULT_STRONG = (
    ROOT / "outputs/highres_dual_ltwm_v1/frozen/"
    "highres_multilayer_dual_ltwm_scand_v1.pt"
)
DEFAULT_ROTATION_GEOMETRY = (
    ROOT / "outputs/highres_dual_ltwm_v1/"
    "external_camera_rotation_proxy_validation_v1/result.json"
)
DEFAULT_TRANSLATION_GEOMETRY = (
    ROOT / "outputs/highres_dual_ltwm_v1/"
    "external_camera_translation_axis_pilot_v1/result.json"
)
DEFAULT_OUTPUT = (
    ROOT / "outputs/highres_dual_ltwm_v1/"
    "strong_se3_adapter_paired_pilot_v1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latents", type=Path, default=DEFAULT_LATENTS)
    parser.add_argument("--strong-checkpoint", type=Path, default=DEFAULT_STRONG)
    parser.add_argument(
        "--rotation-geometry",
        type=Path,
        default=DEFAULT_ROTATION_GEOMETRY,
    )
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


def load_strong_transport_base(
    checkpoint_path: Path,
    rotation_geometry: dict,
    device: torch.device,
) -> tuple[
    RotationTransportHighResMultiLayerDualLatentWM,
    dict,
    list[str],
]:
    payload = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    geometry = rotation_geometry["joint_train_fit"]
    axis = geometry["axis_camera"]
    config = RotationWarpHighResDualLatentWMConfig(
        **payload["config"],
        image_width=448,
        image_height=336,
        effective_focal_px=float(geometry["focal_px"]) * (448.0 / 640.0),
        principal_x_px=224.0,
        principal_y_px=168.0,
        camera_yaw_axis_x=float(axis[0]),
        camera_yaw_axis_y=float(axis[1]),
        camera_yaw_axis_z=float(axis[2]),
    )
    model = RotationTransportHighResMultiLayerDualLatentWM(config)
    missing, unexpected = model.load_state_dict(
        payload["model"],
        strict=False,
    )
    common.require(not unexpected, f"unexpected strong keys: {unexpected}")
    expected_missing = {
        "rotation_transport_raw_scale",
        "camera_yaw_axis",
        "output_camera_rays",
    }
    common.require(
        set(missing) == expected_missing,
        f"unexpected missing strong keys: {missing}",
    )
    common.require(
        torch.equal(
            model.rotation_transport_raw_scale,
            torch.zeros(2),
        ),
        "strong rotation transport is not zero initialized",
    )
    return model.eval().to(device), payload, missing


def main() -> None:
    args = parse_args()
    latent_root = args.latents.resolve()
    strong_path = args.strong_checkpoint.resolve()
    rotation_path = args.rotation_geometry.resolve()
    translation_path = args.translation_geometry.resolve()
    output = args.output.resolve()
    result_path = output / "result.json"
    common.require(
        not result_path.exists(),
        f"refusing to overwrite {result_path}",
    )
    common.require(torch.cuda.is_available(), "CUDA unavailable")
    for path in (strong_path, rotation_path, translation_path):
        common.require(path.is_file(), f"missing input: {path}")
    rotation = json.loads(rotation_path.read_text(encoding="utf-8"))
    translation = json.loads(translation_path.read_text(encoding="utf-8"))
    common.require(
        rotation["status"] == "GO_ROTATION_ONLY_GEOMETRY_PROXY",
        "rotation geometry did not pass",
    )
    common.require(
        translation["status"] == "GO_TRANSLATION_ORIENTATION_PROXY",
        "translation geometry did not pass",
    )
    strong_sha_before = common.sha256_file(strong_path)
    device = torch.device(args.device)
    base, strong_payload, missing = load_strong_transport_base(
        strong_path,
        rotation,
        device,
    )
    common.require(
        strong_payload.get("outcome_gradients") is False,
        "strong checkpoint outcome leakage",
    )
    common.require(
        strong_payload.get("sealed_test_opened") is False,
        "strong checkpoint sealed test leakage",
    )

    paths = {
        spec.trajectory_id: (
            latent_root
            / "latents"
            / "legacy320"
            / f"{spec.trajectory_id}.npz"
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
    cache = raw_pilot.ShardCache(paths)
    baseline = raw_pilot.evaluate_model(
        base,
        cache,
        val_refs,
        batch_size=args.eval_batch_size,
        device=device,
        amp=args.amp,
    )
    orientation = translation["joint_train"]
    adapter_config = DepthTranslationResidualConfig(
        body_forward_axis_camera=tuple(
            map(float, orientation["axis"])
        ),
        body_lateral_axis_camera=tuple(
            map(float, orientation["lateral_axis_camera"])
        ),
    )
    model = FrozenStrongHighResCalibratedSE3Residual(
        base,
        adapter_config,
    ).to(device)
    common.require(
        torch.equal(
            model.base.rotation_transport_raw_scale,
            torch.zeros(2, device=device),
        )
        and torch.equal(
            model.translation_transport_raw_scale,
            torch.zeros(2, device=device),
        ),
        "SE3 adapter is not exactly zero initialized",
    )
    seed = 2026072408
    raw_pilot.set_seed(seed)
    schedule = raw_pilot.route_balanced_schedule(
        train_by_trajectory,
        steps=500,
        batch_size=2,
        seed=seed + 1,
    )
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    training = depth_pilot.train_adapter(
        model,
        cache,
        train_refs,
        schedule,
        device=device,
        amp=args.amp,
    )
    training["final_rotation_scale"] = (
        torch.tanh(model.base.rotation_transport_raw_scale)
        .detach()
        .cpu()
        .tolist()
    )
    evaluation = raw_pilot.evaluate_model(
        model,
        cache,
        val_refs,
        batch_size=args.eval_batch_size,
        device=device,
        amp=args.amp,
    )
    depths = depth_pilot.depth_summary(
        model,
        cache,
        val_refs,
        batch_size=args.eval_batch_size,
        device=device,
        amp=args.amp,
    )
    checkpoint = output / "strong_se3_adapter_final.pt"
    torch.save(
        {
            "adapter": model.adapter_state_dict(),
            "adapter_config": asdict(adapter_config),
            "transport_config": asdict(model.base.config),
            "training": training,
            "evaluation": evaluation,
            "depth_summary": depths,
            "seed": seed,
            "strong_base": {
                "path": str(strong_path),
                "sha256": strong_sha_before,
            },
            "rotation_geometry": {
                "path": str(rotation_path),
                "sha256": common.sha256_file(rotation_path),
            },
            "translation_geometry": {
                "path": str(translation_path),
                "sha256": common.sha256_file(translation_path),
            },
            "outcome_gradients": False,
            "sealed_validation_opened": False,
            "scientific_status": "strong_base_se3_development_adapter",
        },
        checkpoint,
    )
    del model
    torch.cuda.empty_cache()

    comparisons = {
        "semantic_h4_nmse_change": (
            evaluation["semantic_h4_nmse"] - baseline["semantic_h4_nmse"]
        ),
        "geometry_h4_nmse_change": (
            evaluation["geometry_h4_nmse"] - baseline["geometry_h4_nmse"]
        ),
        "dual_h4_nmse_change": (
            evaluation["dual_h4_nmse"] - baseline["dual_h4_nmse"]
        ),
        "dual_h4_nmse_relative_change": (
            evaluation["dual_h4_nmse"] / baseline["dual_h4_nmse"] - 1.0
        ),
        "negative_gap_fraction_change": (
            evaluation["negative_gap_fraction_of_true_error"]
            - baseline["negative_gap_fraction_of_true_error"]
        ),
        "pairwise_action_win_rate_change": (
            evaluation["pairwise_action_win_rate"]
            - baseline["pairwise_action_win_rate"]
        ),
        "retrieval_top1_change": (
            evaluation["retrieval_top1"] - baseline["retrieval_top1"]
        ),
        "mean_rank_change": evaluation["mean_rank"] - baseline["mean_rank"],
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
        "strong_checkpoint_unchanged": (
            common.sha256_file(strong_path) == strong_sha_before
        ),
    }
    go = all(checks.values())
    result = {
        "status": (
            "GO_STRONG_SE3_ADAPTER"
            if go
            else "NO_GO_STRONG_SE3_ADAPTER"
        ),
        "interpretation": (
            "calibrated SE3 residual improved the actual frozen strong LTWM"
            if go
            else "calibrated SE3 residual did not improve the frozen strong LTWM"
        ),
        "protocol": {
            "strong_base_frozen": True,
            "initial_prediction_exactly_equals_strong_base": True,
            "trainable_parameter_count": training[
                "trainable_parameter_count"
            ],
            "dino_frozen": True,
            "outcome_labels_loaded": False,
            "ate_or_objective_gradients": False,
            "sealed_validation_opened": False,
            "formal_v70_modified": False,
        },
        "lineage": {
            "runner": str(Path(__file__).resolve()),
            "runner_sha256": common.sha256_file(Path(__file__).resolve()),
            "model_source": str(
                ROOT
                / "src/recon_latents/"
                "models_highres_dual_strong_se3_adapter_v1.py"
            ),
            "model_source_sha256": common.sha256_file(
                ROOT
                / "src/recon_latents/"
                "models_highres_dual_strong_se3_adapter_v1.py"
            ),
            "strong_checkpoint": {
                "path": str(strong_path),
                "sha256": strong_sha_before,
            },
            "rotation_geometry": {
                "path": str(rotation_path),
                "sha256": common.sha256_file(rotation_path),
            },
            "translation_geometry": {
                "path": str(translation_path),
                "sha256": common.sha256_file(translation_path),
            },
            "strong_to_transport_missing_keys": missing,
        },
        "strong_baseline_held_route_evaluation": baseline,
        "strong_se3_held_route_evaluation": evaluation,
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
                "training": training,
                "depth_summary": depths,
                "baseline": baseline,
                "strong_se3": evaluation,
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
