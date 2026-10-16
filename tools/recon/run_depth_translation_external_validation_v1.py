#!/usr/bin/env python3
"""Frozen external validation of the depth-translation LTWM adapter."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
from typing import Any

import torch

import run_depth_translation_ltwm_paired_pilot_v1 as development
import run_raw_resolution_paired_pilot_v1 as raw_pilot
import run_rotation_transport_external_bag_validation_v1 as external
import run_rotation_warp_ltwm_paired_pilot_v1 as common
from recon_latents.models_highres_dual_depth_translation_v1 import (
    DepthTranslationResidualConfig,
    FrozenRotationTransportDepthTranslationResidual,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXTERNAL = (
    ROOT / "outputs/highres_dual_ltwm_v1/"
    "rotation_transport_external_bag_validation_v1"
)
DEFAULT_DEVELOPMENT = (
    ROOT / "outputs/highres_dual_ltwm_v1/"
    "depth_translation_ltwm_paired_pilot_v1"
)
DEFAULT_OUTPUT = (
    ROOT / "outputs/highres_dual_ltwm_v1/"
    "depth_translation_external_validation_v1/result.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--external", type=Path, default=DEFAULT_EXTERNAL)
    parser.add_argument("--development", type=Path, default=DEFAULT_DEVELOPMENT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--amp", choices=("bf16", "fp16", "off"), default="bf16")
    return parser.parse_args()


def load_adapter(
    development_dir: Path,
    device: torch.device,
) -> tuple[
    FrozenRotationTransportDepthTranslationResidual,
    dict[str, Any],
]:
    result = json.loads(
        (development_dir / "result.json").read_text(encoding="utf-8")
    )
    checkpoint_path = (
        development_dir / "depth_translation_adapter_final.pt"
    )
    common.require(
        common.sha256_file(checkpoint_path)
        == result["checkpoint"]["sha256"],
        "development adapter hash mismatch",
    )
    payload = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    rotation_path = Path(payload["rotation_base"]["path"])
    common.require(
        common.sha256_file(rotation_path)
        == payload["rotation_base"]["sha256"],
        "rotation base hash mismatch",
    )
    base, _ = development.load_frozen_base(rotation_path, device)
    model = FrozenRotationTransportDepthTranslationResidual(
        base,
        DepthTranslationResidualConfig(**payload["adapter_config"]),
    ).to(device)
    missing, unexpected = model.load_state_dict(
        payload["adapter"],
        strict=False,
    )
    common.require(not unexpected, f"unexpected adapter keys: {unexpected}")
    common.require(
        all(name.startswith("base.") for name in missing),
        f"missing non-base adapter keys: {missing}",
    )
    return model.eval(), {
        "result": result,
        "payload": payload,
        "checkpoint_path": checkpoint_path,
        "rotation_path": rotation_path,
    }


def main() -> None:
    args = parse_args()
    external_dir = args.external.resolve()
    development_dir = args.development.resolve()
    result_path = args.output.resolve()
    common.require(
        not result_path.exists(),
        f"refusing to overwrite {result_path}",
    )
    common.require(torch.cuda.is_available(), "CUDA unavailable")
    external_result_path = external_dir / "result.json"
    development_result_path = development_dir / "result.json"
    for path in (external_result_path, development_result_path):
        common.require(path.is_file(), f"missing input: {path}")
    external_result = json.loads(
        external_result_path.read_text(encoding="utf-8")
    )
    development_result = json.loads(
        development_result_path.read_text(encoding="utf-8")
    )
    common.require(
        development_result["status"] == "GO_DEPTH_TRANSLATION_LTWM",
        "depth translation development result did not pass",
    )
    common.require(
        external_result["protocol"]["model_training_or_tuning"] is False,
        "external latent source was not frozen inference",
    )

    paths = {
        path.stem: path
        for path in sorted((external_dir / "latents").glob("*.npz"))
    }
    common.require(len(paths) == 3, "expected exactly three external shards")
    expected_shards = {
        Path(row["latent_shard"]).stem: row["latent_shard_sha256"]
        for row in external_result["lineage"]["bags"]
    }
    common.require(
        set(paths) == set(expected_shards),
        "external shard identity mismatch",
    )
    for trajectory_id, path in paths.items():
        common.require(
            common.sha256_file(path) == expected_shards[trajectory_id],
            f"external shard hash mismatch: {trajectory_id}",
        )
    split = {trajectory_id: "external" for trajectory_id in paths}
    pooled_refs, refs_by_trajectory = raw_pilot.window_refs(
        paths,
        split,
        "external",
    )
    cache = raw_pilot.ShardCache(paths)
    device = torch.device(args.device)
    model, lineage = load_adapter(development_dir, device)
    base = model.base
    started = time.perf_counter()

    pooled_base = raw_pilot.evaluate_model(
        base,
        cache,
        pooled_refs,
        batch_size=args.eval_batch_size,
        device=device,
        amp=args.amp,
    )
    consistency = common.metric_consistency(
        pooled_base,
        external_result["pooled"]["transport"],
    )
    common.require(
        max(consistency.values()) <= 1.0e-6,
        f"external rotation baseline mismatch: {consistency}",
    )
    pooled_adapter = raw_pilot.evaluate_model(
        model,
        cache,
        pooled_refs,
        batch_size=args.eval_batch_size,
        device=device,
        amp=args.amp,
    )
    pooled_depth = development.depth_summary(
        model,
        cache,
        pooled_refs,
        batch_size=args.eval_batch_size,
        device=device,
        amp=args.amp,
    )
    pooled_changes = external.metric_changes(
        pooled_base,
        pooled_adapter,
    )

    per_route = {}
    routes_dual_improved = 0
    routes_pairwise_improved = 0
    for trajectory_id in sorted(refs_by_trajectory):
        refs = [
            pooled_refs[index]
            for index in refs_by_trajectory[trajectory_id]
        ]
        baseline = raw_pilot.evaluate_model(
            base,
            cache,
            refs,
            batch_size=args.eval_batch_size,
            device=device,
            amp=args.amp,
        )
        adapter = raw_pilot.evaluate_model(
            model,
            cache,
            refs,
            batch_size=args.eval_batch_size,
            device=device,
            amp=args.amp,
        )
        changes = external.metric_changes(baseline, adapter)
        depths = development.depth_summary(
            model,
            cache,
            refs,
            batch_size=args.eval_batch_size,
            device=device,
            amp=args.amp,
        )
        if changes["dual_h4_nmse_change"] < 0.0:
            routes_dual_improved += 1
        if changes["pairwise_action_win_rate_change"] > 0.0:
            routes_pairwise_improved += 1
        per_route[trajectory_id] = {
            "rotation_baseline": baseline,
            "depth_translation": adapter,
            "depth_summary": depths,
            "changes": changes,
        }
    checks = {
        "pooled_future_dual_h4_improved": (
            pooled_changes["dual_h4_nmse_change"] < 0.0
        ),
        "pooled_future_geometry_h4_not_worse": (
            pooled_changes["geometry_h4_nmse_change"] <= 0.0
        ),
        "pooled_negative_gap_fraction_improved": (
            pooled_changes["negative_gap_fraction_change"] > 0.0
        ),
        "pooled_pairwise_action_win_rate_improved": (
            pooled_changes["pairwise_action_win_rate_change"] > 0.0
        ),
        "pooled_mean_rank_improved": (
            pooled_changes["mean_rank_change"] < 0.0
        ),
        "dual_h4_improved_on_at_least_two_routes": (
            routes_dual_improved >= 2
        ),
        "pairwise_improved_on_at_least_two_routes": (
            routes_pairwise_improved >= 2
        ),
        "external_depth_is_noncollapsed": pooled_depth["std_m"] >= 0.05,
    }
    go = all(checks.values())
    result = {
        "status": (
            "GO_EXTERNAL_DEPTH_TRANSLATION_LTWM"
            if go
            else "NO_GO_EXTERNAL_DEPTH_TRANSLATION_LTWM"
        ),
        "decision": (
            "retain depth-aware SE3 transport as the leading LTWM branch"
            if go
            else "do not promote depth-aware SE3 transport"
        ),
        "protocol": {
            "checkpoint_frozen_before_external_inference": True,
            "external_latents_reused_without_reencoding": True,
            "model_training_or_tuning": False,
            "threshold_or_gate_tuning": False,
            "outcome_labels_loaded": False,
            "ate_or_objective_gradients": False,
            "sealed_validation_opened": False,
            "formal_v70_modified": False,
            "frozen_highres_dual_modified": False,
        },
        "lineage": {
            "runner": str(Path(__file__).resolve()),
            "runner_sha256": common.sha256_file(Path(__file__).resolve()),
            "external_result": {
                "path": str(external_result_path),
                "sha256": common.sha256_file(external_result_path),
            },
            "development_result": {
                "path": str(development_result_path),
                "sha256": common.sha256_file(development_result_path),
            },
            "adapter_checkpoint": {
                "path": str(lineage["checkpoint_path"]),
                "sha256": common.sha256_file(
                    lineage["checkpoint_path"]
                ),
            },
            "rotation_checkpoint": {
                "path": str(lineage["rotation_path"]),
                "sha256": common.sha256_file(lineage["rotation_path"]),
            },
            "external_shards": {
                trajectory_id: {
                    "path": str(path),
                    "sha256": common.sha256_file(path),
                }
                for trajectory_id, path in paths.items()
            },
            "rotation_baseline_reevaluation_max_abs_difference": max(
                consistency.values()
            ),
        },
        "pooled": {
            "rotation_baseline": pooled_base,
            "depth_translation": pooled_adapter,
            "depth_summary": pooled_depth,
            "changes": pooled_changes,
        },
        "per_route": per_route,
        "route_consistency": {
            "routes": len(per_route),
            "dual_h4_improved_routes": routes_dual_improved,
            "pairwise_improved_routes": routes_pairwise_improved,
        },
        "checks": checks,
        "runtime_seconds": time.perf_counter() - started,
    }
    common.write_json_atomic(result_path, result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "checks": checks,
                "pooled": result["pooled"],
                "route_consistency": result["route_consistency"],
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
