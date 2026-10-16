#!/usr/bin/env python3
"""Fixed full-validation comparison for dual-latent LTWM checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import recon_latents.losses_highres_dual_stable_v1  # noqa: E402,F401
import train_highres_multilayer_dual_ltwm_v1 as shared  # noqa: E402
from recon_latents.dataset_highres_dual_v1 import (  # noqa: E402
    HighResDualLatentWindowDataset,
)
from recon_latents.models_highres_dual_v1 import (  # noqa: E402
    HighResDualLatentWMConfig,
    HighResMultiLayerDualLatentWM,
)
from recon_latents.split_lock import file_sha256, load_split_lock  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latent-root", type=Path, required=True)
    parser.add_argument("--dataset", choices=("recon", "scand"), required=True)
    parser.add_argument("--split-lock", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--amp", choices=("bf16", "fp16", "off"), default="bf16")
    parser.add_argument("--seed", type=int, default=20260724)
    return parser.parse_args()


def amp_context(amp: str):
    if amp == "off":
        return torch.autocast(device_type="cuda", enabled=False)
    dtype = torch.bfloat16 if amp == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def branch_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    current: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    prediction = prediction.float()
    target = target.float()
    current = current.float()
    error = (prediction - target).square().mean(dim=(-1, -2))
    denominator = target.square().mean(dim=(-1, -2)).clamp_min(1.0e-8)
    nmse = error / denominator
    pred_delta = prediction - current[:, None]
    target_delta = target - current[:, None]
    ratio = (
        torch.linalg.vector_norm(pred_delta.flatten(-2), dim=-1)
        / torch.linalg.vector_norm(
            target_delta.flatten(-2),
            dim=-1,
        ).clamp_min(1.0e-8)
    )
    return nmse.cpu().numpy(), ratio.cpu().numpy()


def append_metrics(
    record: dict[str, list[np.ndarray] | list[str]],
    *,
    semantic_prediction: torch.Tensor,
    geometry_prediction: torch.Tensor,
    semantic_future: torch.Tensor,
    geometry_future: torch.Tensor,
    semantic_current: torch.Tensor,
    geometry_current: torch.Tensor,
    routes: list[str],
    prefix: str = "",
) -> None:
    semantic_nmse, semantic_ratio = branch_metrics(
        semantic_prediction,
        semantic_future,
        semantic_current,
    )
    geometry_nmse, geometry_ratio = branch_metrics(
        geometry_prediction,
        geometry_future,
        geometry_current,
    )
    record[f"{prefix}semantic_nmse"].append(semantic_nmse)
    record[f"{prefix}geometry_nmse"].append(geometry_nmse)
    record[f"{prefix}semantic_delta_ratio"].append(semantic_ratio)
    record[f"{prefix}geometry_delta_ratio"].append(geometry_ratio)
    if not prefix:
        record["routes"].extend(routes)


def new_record() -> dict[str, list[np.ndarray] | list[str]]:
    return {
        "routes": [],
        "semantic_nmse": [],
        "geometry_nmse": [],
        "semantic_delta_ratio": [],
        "geometry_delta_ratio": [],
        "corrupt_semantic_nmse": [],
        "corrupt_geometry_nmse": [],
        "corrupt_semantic_delta_ratio": [],
        "corrupt_geometry_delta_ratio": [],
    }


def concatenate_record(
    record: dict[str, list[np.ndarray] | list[str]],
) -> dict[str, np.ndarray | list[str]]:
    return {
        key: (
            value
            if key == "routes"
            else np.concatenate(value, axis=0)
        )
        for key, value in record.items()
    }


def summarize(
    record: dict[str, np.ndarray | list[str]],
    baseline: dict[str, np.ndarray | list[str]],
) -> dict[str, object]:
    semantic = record["semantic_nmse"]
    geometry = record["geometry_nmse"]
    corrupt_semantic = record["corrupt_semantic_nmse"]
    corrupt_geometry = record["corrupt_geometry_nmse"]
    baseline_dual = 0.5 * (
        baseline["semantic_nmse"][:, -1]
        + baseline["geometry_nmse"][:, -1]
    )
    dual = 0.5 * (semantic[:, -1] + geometry[:, -1])
    corrupt_dual = 0.5 * (
        corrupt_semantic[:, -1] + corrupt_geometry[:, -1]
    )
    gap = corrupt_dual - dual
    routes = np.asarray(record["routes"])
    route_rows = []
    for route in sorted(set(routes.tolist())):
        mask = routes == route
        route_rows.append(
            {
                "route": route,
                "windows": int(mask.sum()),
                "dual_h4_nmse": float(dual[mask].mean()),
                "baseline_dual_h4_nmse": float(baseline_dual[mask].mean()),
                "dual_h4_improvement": float(
                    baseline_dual[mask].mean() - dual[mask].mean()
                ),
                "corrupted_nmse_gap_h4": float(gap[mask].mean()),
            }
        )
    improvements = np.asarray(
        [row["dual_h4_improvement"] for row in route_rows]
    )
    route_gaps = np.asarray(
        [row["corrupted_nmse_gap_h4"] for row in route_rows]
    )
    return {
        "semantic_nmse_by_horizon": semantic.mean(axis=0).tolist(),
        "geometry_nmse_by_horizon": geometry.mean(axis=0).tolist(),
        "semantic_delta_ratio_by_horizon": record[
            "semantic_delta_ratio"
        ].mean(axis=0).tolist(),
        "geometry_delta_ratio_by_horizon": record[
            "geometry_delta_ratio"
        ].mean(axis=0).tolist(),
        "dual_h4_nmse": float(dual.mean()),
        "baseline_dual_h4_nmse": float(baseline_dual.mean()),
        "dual_h4_improvement": float(baseline_dual.mean() - dual.mean()),
        "dual_h4_relative_improvement": float(
            (baseline_dual.mean() - dual.mean()) / baseline_dual.mean()
        ),
        "corrupted_nmse_gap_h4": float(gap.mean()),
        "corrupted_gap_fraction_of_true_error": float(
            gap.mean() / max(float(dual.mean()), 1.0e-8)
        ),
        "routes_improved_over_baseline": int((improvements > 0).sum()),
        "routes_positive_corrupted_gap": int((route_gaps > 0).sum()),
        "route_count": len(route_rows),
        "route_rows": route_rows,
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    split_lock, split_lock_sha = load_split_lock(args.split_lock)
    root_report = shared.validate_dual_latent_root(
        args.latent_root,
        dataset=args.dataset,
        split_lock=split_lock,
        split_lock_sha256=split_lock_sha,
    )
    dataset = HighResDualLatentWindowDataset(
        args.latent_root,
        "val",
        history_len=4,
        horizon=4,
        stream_cache_shards=2,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    device = torch.device("cuda")
    models: list[tuple[str, Path, dict[str, object], torch.nn.Module]] = []
    seen_names: set[str] = set()
    for checkpoint_path in args.checkpoints:
        payload = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        if payload.get("split_lock_sha256") != split_lock_sha:
            raise ValueError(f"checkpoint split lock mismatch: {checkpoint_path}")
        if (
            payload["representation_contract"]["geometry_projection_sha256"]
            != root_report["representation"]["geometry_projection_sha256"]
        ):
            raise ValueError(f"representation mismatch: {checkpoint_path}")
        name = checkpoint_path.stem
        if name in seen_names:
            name = f"{checkpoint_path.parent.name}_{name}"
        seen_names.add(name)
        config = HighResDualLatentWMConfig(**payload["config"])
        model = HighResMultiLayerDualLatentWM(config).to(device).eval()
        model.load_state_dict(payload["model"], strict=True)
        models.append((name, checkpoint_path, payload, model))

    baseline_record = new_record()
    model_records = {name: new_record() for name, _, _, _ in models}
    started = time.perf_counter()
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            semantic_history = batch["semantic_history"].to(
                device,
                non_blocking=True,
            )
            geometry_history = batch["geometry_history"].to(
                device,
                non_blocking=True,
            )
            semantic_future = batch["semantic_future"].to(
                device,
                non_blocking=True,
            )
            geometry_future = batch["geometry_future"].to(
                device,
                non_blocking=True,
            )
            actions = batch["actions"].to(device, non_blocking=True)
            routes = [str(value) for value in batch["trajectory_id"]]
            semantic_current = semantic_history[:, -1]
            geometry_current = geometry_history[:, -1]
            baseline_semantic = semantic_current[:, None].expand_as(
                semantic_future
            )
            baseline_geometry = geometry_current[:, None].expand_as(
                geometry_future
            )
            append_metrics(
                baseline_record,
                semantic_prediction=baseline_semantic,
                geometry_prediction=baseline_geometry,
                semantic_future=semantic_future,
                geometry_future=geometry_future,
                semantic_current=semantic_current,
                geometry_current=geometry_current,
                routes=routes,
            )
            corrupted_actions = shared.corrupt_actions(actions, batch_index)
            for name, _, _, model in models:
                with amp_context(args.amp):
                    output = model(
                        semantic_history,
                        geometry_history,
                        actions,
                    )
                    corrupted = model(
                        semantic_history,
                        geometry_history,
                        corrupted_actions,
                    )
                record = model_records[name]
                append_metrics(
                    record,
                    semantic_prediction=output[
                        "predicted_semantic_tokens"
                    ][:, 0],
                    geometry_prediction=output[
                        "predicted_geometry_tokens"
                    ][:, 0],
                    semantic_future=semantic_future,
                    geometry_future=geometry_future,
                    semantic_current=semantic_current,
                    geometry_current=geometry_current,
                    routes=routes,
                )
                append_metrics(
                    record,
                    semantic_prediction=corrupted[
                        "predicted_semantic_tokens"
                    ][:, 0],
                    geometry_prediction=corrupted[
                        "predicted_geometry_tokens"
                    ][:, 0],
                    semantic_future=semantic_future,
                    geometry_future=geometry_future,
                    semantic_current=semantic_current,
                    geometry_current=geometry_current,
                    routes=routes,
                    prefix="corrupt_",
                )
            if (batch_index + 1) % 200 == 0:
                print(
                    json.dumps(
                        {
                            "batches": batch_index + 1,
                            "windows": min(
                                (batch_index + 1) * args.batch_size,
                                len(dataset),
                            ),
                        }
                    ),
                    flush=True,
                )

    baseline = concatenate_record(baseline_record)
    baseline_dual = 0.5 * (
        baseline["semantic_nmse"][:, -1]
        + baseline["geometry_nmse"][:, -1]
    )
    checkpoint_rows = []
    for name, path, payload, _ in models:
        summary = summarize(
            concatenate_record(model_records[name]),
            baseline,
        )
        checkpoint_rows.append(
            {
                "name": name,
                "path": str(path),
                "sha256": file_sha256(path),
                "training_epoch": len(payload.get("history", ())),
                **summary,
            }
        )
    report = {
        "status": "PASS",
        "protocol": "fixed_sequential_full_development_validation",
        "dataset": args.dataset,
        "latent_root": root_report,
        "split_lock_sha256": split_lock_sha,
        "sealed_test_opened": False,
        "outcome_metrics_used": False,
        "camera_conditioning": False,
        "seed": args.seed,
        "amp": args.amp,
        "batch_size": args.batch_size,
        "validation_windows": len(dataset),
        "validation_routes": len(dataset.trajectories),
        "baseline": {
            "semantic_nmse_by_horizon": baseline[
                "semantic_nmse"
            ].mean(axis=0).tolist(),
            "geometry_nmse_by_horizon": baseline[
                "geometry_nmse"
            ].mean(axis=0).tolist(),
            "dual_h4_nmse": float(baseline_dual.mean()),
        },
        "checkpoints": checkpoint_rows,
        "best_true_prediction": min(
            checkpoint_rows,
            key=lambda row: row["dual_h4_nmse"],
        )["name"],
        "best_action_discrimination": max(
            checkpoint_rows,
            key=lambda row: row["corrupted_nmse_gap_h4"],
        )["name"],
        "runtime_seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "status": "PASS",
                "output": str(args.output),
                "output_sha256": file_sha256(args.output),
                "best_true_prediction": report["best_true_prediction"],
                "best_action_discrimination": report[
                    "best_action_discrimination"
                ],
                "runtime_seconds": report["runtime_seconds"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
