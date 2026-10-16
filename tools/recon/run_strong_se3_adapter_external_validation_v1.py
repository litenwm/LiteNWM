#!/usr/bin/env python3
"""Frozen external validation of the strong-base calibrated SE(3) adapter."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import torch

import run_depth_translation_ltwm_paired_pilot_v1 as depth_pilot
import run_raw_resolution_paired_pilot_v1 as raw_pilot
import run_rotation_transport_external_bag_validation_v1 as external
import run_rotation_warp_ltwm_paired_pilot_v1 as common
import run_strong_se3_adapter_paired_pilot_v1 as development
from recon_latents.models_highres_dual_depth_translation_v1 import (
    DepthTranslationResidualConfig,
)
from recon_latents.models_highres_dual_strong_se3_adapter_v1 import (
    FrozenStrongHighResCalibratedSE3Residual,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXTERNAL = (
    ROOT / "outputs/highres_dual_ltwm_v1/"
    "rotation_transport_external_bag_validation_v1"
)
DEFAULT_DEVELOPMENT = (
    ROOT / "outputs/highres_dual_ltwm_v1/"
    "strong_se3_adapter_paired_pilot_v1"
)
DEFAULT_OUTPUT = (
    ROOT / "outputs/highres_dual_ltwm_v1/"
    "strong_se3_adapter_external_validation_v1/result.json"
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
    adapter_path = development_dir / "strong_se3_adapter_final.pt"
    for path in (
        external_result_path,
        development_result_path,
        adapter_path,
    ):
        common.require(path.is_file(), f"missing input: {path}")
    external_result = json.loads(
        external_result_path.read_text(encoding="utf-8")
    )
    development_result = json.loads(
        development_result_path.read_text(encoding="utf-8")
    )
    common.require(
        development_result["status"] == "GO_STRONG_SE3_ADAPTER",
        "strong SE3 development result did not pass",
    )
    common.require(
        common.sha256_file(adapter_path)
        == development_result["checkpoint"]["sha256"],
        "strong SE3 adapter hash mismatch",
    )
    payload = torch.load(
        adapter_path,
        map_location="cpu",
        weights_only=False,
    )
    strong_path = Path(payload["strong_base"]["path"])
    rotation_path = Path(payload["rotation_geometry"]["path"])
    common.require(
        common.sha256_file(strong_path)
        == payload["strong_base"]["sha256"],
        "strong base hash mismatch",
    )
    common.require(
        common.sha256_file(rotation_path)
        == payload["rotation_geometry"]["sha256"],
        "rotation geometry hash mismatch",
    )
    rotation = json.loads(rotation_path.read_text(encoding="utf-8"))

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
    baseline, _, _ = development.load_strong_transport_base(
        strong_path,
        rotation,
        device,
    )
    adapted_base, _, _ = development.load_strong_transport_base(
        strong_path,
        rotation,
        device,
    )
    model = FrozenStrongHighResCalibratedSE3Residual(
        adapted_base,
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
    common.require(
        "base.rotation_transport_raw_scale" not in missing,
        "rotation adapter scale was not restored",
    )
    model.eval()
    started = time.perf_counter()
    pooled_baseline = raw_pilot.evaluate_model(
        baseline,
        cache,
        pooled_refs,
        batch_size=args.eval_batch_size,
        device=device,
        amp=args.amp,
    )
    pooled_adapter = raw_pilot.evaluate_model(
        model,
        cache,
        pooled_refs,
        batch_size=args.eval_batch_size,
        device=device,
        amp=args.amp,
    )
    pooled_depth = depth_pilot.depth_summary(
        model,
        cache,
        pooled_refs,
        batch_size=args.eval_batch_size,
        device=device,
        amp=args.amp,
    )
    pooled_changes = external.metric_changes(
        pooled_baseline,
        pooled_adapter,
    )
    per_route = {}
    routes_dual_improved = 0
    routes_pairwise_improved = 0
    routes_top1_not_worse = 0
    for trajectory_id in sorted(refs_by_trajectory):
        refs = [
            pooled_refs[index]
            for index in refs_by_trajectory[trajectory_id]
        ]
        base_value = raw_pilot.evaluate_model(
            baseline,
            cache,
            refs,
            batch_size=args.eval_batch_size,
            device=device,
            amp=args.amp,
        )
        adapter_value = raw_pilot.evaluate_model(
            model,
            cache,
            refs,
            batch_size=args.eval_batch_size,
            device=device,
            amp=args.amp,
        )
        changes = external.metric_changes(base_value, adapter_value)
        if changes["dual_h4_nmse_change"] < 0.0:
            routes_dual_improved += 1
        if changes["pairwise_action_win_rate_change"] > 0.0:
            routes_pairwise_improved += 1
        if changes["retrieval_top1_change"] >= 0.0:
            routes_top1_not_worse += 1
        per_route[trajectory_id] = {
            "strong_baseline": base_value,
            "strong_se3": adapter_value,
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
        "pooled_retrieval_top1_not_worse": (
            pooled_changes["retrieval_top1_change"] >= 0.0
        ),
        "dual_h4_improved_on_at_least_two_routes": (
            routes_dual_improved >= 2
        ),
        "pairwise_improved_on_at_least_two_routes": (
            routes_pairwise_improved >= 2
        ),
        "top1_not_worse_on_at_least_two_routes": (
            routes_top1_not_worse >= 2
        ),
        "external_depth_is_noncollapsed": pooled_depth["std_m"] >= 0.05,
    }
    go = all(checks.values())
    result = {
        "status": (
            "GO_EXTERNAL_STRONG_SE3_ADAPTER"
            if go
            else "NO_GO_EXTERNAL_STRONG_SE3_ADAPTER"
        ),
        "decision": (
            "rescore Spot union65 with the frozen strong SE3 adapter"
            if go
            else "do not rescore union65; retain original frozen highres LTWM"
        ),
        "protocol": {
            "adapter_frozen_before_external_inference": True,
            "external_latents_reused_without_reencoding": True,
            "model_training_or_tuning": False,
            "top1_noninferiority_predeclared": True,
            "outcome_labels_loaded": False,
            "ate_or_objective_gradients": False,
            "sealed_validation_opened": False,
            "formal_v70_modified": False,
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
            "adapter": {
                "path": str(adapter_path),
                "sha256": common.sha256_file(adapter_path),
            },
            "strong_base": {
                "path": str(strong_path),
                "sha256": common.sha256_file(strong_path),
            },
        },
        "pooled": {
            "strong_baseline": pooled_baseline,
            "strong_se3": pooled_adapter,
            "depth_summary": pooled_depth,
            "changes": pooled_changes,
        },
        "per_route": per_route,
        "route_consistency": {
            "routes": len(per_route),
            "dual_h4_improved_routes": routes_dual_improved,
            "pairwise_improved_routes": routes_pairwise_improved,
            "top1_not_worse_routes": routes_top1_not_worse,
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
