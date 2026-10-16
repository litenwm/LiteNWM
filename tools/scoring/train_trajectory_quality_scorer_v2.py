#!/usr/bin/env python3
"""Train one absolute ATE/RTP scorer-v2 seed on balanced RECON+SCAND."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from scoring.trajectory_quality_objective_v2 import (  # noqa: E402
    TrajectoryQualityLossConfig,
    trajectory_quality_loss,
)
from scoring.trajectory_quality_scorer_v2 import (  # noqa: E402
    TrajectoryQualityPrediction,
    TrajectoryQualityScorerConfig,
    TrajectoryQualityScorerV2,
)


K = 16
HORIZON = 4
ACTION_DIM = 3
EPS = 1.0e-8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recon-train", type=Path, required=True)
    parser.add_argument("--recon-calibration", type=Path, required=True)
    parser.add_argument("--scand-train", type=Path, required=True)
    parser.add_argument("--scand-calibration", type=Path, required=True)
    parser.add_argument(
        "--calibration-partition",
        choices=("train-calibration", "scorer-cal"),
        default="scorer-cal",
        help="Partition used only for scorer early stopping/model selection.",
    )
    parser.add_argument("--split-lock", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=640)
    parser.add_argument("--hidden-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_cache(
    path: Path,
    *,
    dataset: str,
    partition: str,
    split_lock_sha: str,
) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        output = {key: archive[key].copy() for key in archive.files}
    required = {
        "window_uid",
        "dataset",
        "partition",
        "trajectory_id",
        "route_id",
        "features",
        "target",
        "actions_h4",
        "valid",
        "metadata",
    }
    if missing := required.difference(output):
        raise RuntimeError(f"{path} missing arrays: {sorted(missing)}")
    if str(output["dataset"].item()) != dataset:
        raise RuntimeError(f"dataset mismatch: {path}")
    if str(output["partition"].item()) != partition:
        raise RuntimeError(f"partition mismatch: {path}")
    metadata = json.loads(str(output["metadata"].item()))
    if metadata["split_lock_sha256"] != split_lock_sha:
        raise RuntimeError(f"split-lock mismatch: {path}")
    count = len(output["window_uid"])
    if output["features"].shape[:2] != (count, K):
        raise RuntimeError(f"invalid feature shape: {path}")
    if output["target"].shape != (count, K, 2):
        raise RuntimeError(f"invalid target shape: {path}")
    if output["actions_h4"].shape != (count, K, HORIZON, ACTION_DIM):
        raise RuntimeError(f"invalid action shape: {path}")
    if output["valid"].shape != (count, K):
        raise RuntimeError(f"invalid valid shape: {path}")
    return output


def domain_balanced_moments(
    recon: np.ndarray,
    scand: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    recon_values = recon.astype(np.float64).reshape(-1, recon.shape[-1])
    scand_values = scand.astype(np.float64).reshape(-1, scand.shape[-1])
    mean = 0.5 * (recon_values.mean(axis=0) + scand_values.mean(axis=0))
    second = 0.5 * (
        np.square(recon_values).mean(axis=0)
        + np.square(scand_values).mean(axis=0)
    )
    scale = np.sqrt(np.maximum(second - np.square(mean), 1.0e-12))
    scale[scale < 1.0e-5] = 1.0
    return mean.astype(np.float32), scale.astype(np.float32)


def domain_balanced_target_scales(
    recon: np.ndarray,
    scand: np.ndarray,
) -> tuple[float, float]:
    recon_values = recon.astype(np.float64).reshape(-1, 2)
    scand_values = scand.astype(np.float64).reshape(-1, 2)
    scale = 0.5 * (
        recon_values.std(axis=0) + scand_values.std(axis=0)
    )
    scale = np.maximum(scale, np.asarray((0.02, 0.02)))
    return float(scale[0]), float(scale[1])


def balanced_action_sample(
    recon: np.ndarray,
    scand: np.ndarray,
    *,
    seed: int,
    maximum_per_domain: int = 40000,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    values = []
    for domain in (recon, scand):
        flat = domain.astype(np.float32).reshape(-1, HORIZON, ACTION_DIM)
        count = min(maximum_per_domain, len(flat))
        indices = rng.choice(len(flat), size=count, replace=False)
        values.append(flat[indices])
    return np.concatenate(values)


def action_support_statistics(
    recon: np.ndarray,
    scand: np.ndarray,
    *,
    seed: int,
) -> dict[str, Any]:
    actions = balanced_action_sample(recon, scand, seed=seed)
    lower = np.quantile(actions, 0.01, axis=0).astype(np.float32)
    upper = np.quantile(actions, 0.99, axis=0).astype(np.float32)
    width = np.maximum(upper - lower, 1.0e-5)
    flat = actions.reshape(len(actions), -1).astype(np.float64)
    mean = flat.mean(axis=0)
    covariance = np.cov(flat, rowvar=False)
    ridge = max(float(np.trace(covariance)) / covariance.shape[0], 1.0e-8) * 1.0e-3
    inverse = np.linalg.inv(covariance + ridge * np.eye(covariance.shape[0]))
    centered = flat - mean
    mahalanobis = np.einsum("ni,ij,nj->n", centered, inverse, centered)

    first = np.diff(actions, axis=1)
    second = np.diff(first, axis=1)
    roughness = np.square(first).sum(axis=-1).mean(axis=-1)
    jerk = np.square(second).sum(axis=-1).mean(axis=-1)
    path_length = np.linalg.norm(actions[..., :2], axis=-1).sum(axis=-1)
    curvature = np.abs(actions[..., 2]).sum(axis=-1) / np.maximum(
        path_length, EPS
    )
    rng = np.random.default_rng(seed + 1)
    support_count = min(2048, len(flat))
    support_indices = rng.choice(len(flat), size=support_count, replace=False)
    support = flat[support_indices].astype(np.float32)
    probe_count = min(4096, len(flat))
    probe_indices = rng.choice(len(flat), size=probe_count, replace=False)
    normalized_support = support / width.reshape(-1)
    normalized_probe = flat[probe_indices] / width.reshape(-1)
    # Chunked NumPy distance avoids an optional sklearn/scipy dependency.
    kth = []
    neighbors = min(8, support_count)
    for start in range(0, len(normalized_probe), 256):
        probe = normalized_probe[start : start + 256]
        squared = (
            np.square(probe).sum(axis=1, keepdims=True)
            + np.square(normalized_support).sum(axis=1)[None]
            - 2.0 * probe @ normalized_support.T
        )
        distance = np.sqrt(np.maximum(squared, 0.0)) / math.sqrt(
            HORIZON * ACTION_DIM
        )
        kth.append(
            np.partition(distance, neighbors - 1, axis=1)[:, neighbors - 1]
        )
    kth_distance = np.concatenate(kth)
    component_std = actions.std(axis=0)
    return {
        "source": "equal-domain outer-train candidates only",
        "lower_p01": lower.tolist(),
        "upper_p99": upper.tolist(),
        "mean": mean.astype(np.float32).tolist(),
        "covariance_inverse": inverse.astype(np.float32).tolist(),
        "mahalanobis_squared_p99": float(np.quantile(mahalanobis, 0.99)),
        "support_actions": support.tolist(),
        "knn_k": neighbors,
        "knn_distance_p99": float(np.quantile(kth_distance, 0.99)),
        "roughness_p99": float(np.quantile(roughness, 0.99)),
        "jerk_p99": float(np.quantile(jerk, 0.99)),
        "curvature_p99": float(np.quantile(curvature, 0.99)),
        "parent_trust_radius": 0.35,
        "cem_std_floor": float(
            max(0.01, np.quantile(component_std, 0.10) * 0.25)
        ),
        "cem_std_ceiling": float(
            max(0.10, np.quantile(component_std, 0.90) * 2.0)
        ),
        "fit_candidates": int(len(actions)),
    }


def batch_indices(
    count: int,
    batch_count: int,
    generator: np.random.Generator,
) -> np.ndarray:
    return generator.choice(count, size=batch_count, replace=count < batch_count)


def tensor_batch(
    data: dict[str, np.ndarray],
    indices: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    features = torch.from_numpy(
        data["features"][indices].astype(np.float32)
    ).to(device)
    target = torch.from_numpy(data["target"][indices].astype(np.float32)).to(device)
    valid = torch.from_numpy(data["valid"][indices].astype(bool)).to(device)
    return features, target, valid


def forward_features(
    model: TrajectoryQualityScorerV2,
    features: torch.Tensor,
) -> TrajectoryQualityPrediction:
    standardized = (features - model.feature_mean) / model.feature_scale
    raw = model.head(model.backbone(standardized))
    return TrajectoryQualityPrediction(
        ate=torch.nn.functional.softplus(raw[..., 0]),
        rtp=1.0 - torch.nn.functional.softplus(-raw[..., 1]),
    )


@torch.inference_mode()
def evaluate_domain(
    model: TrajectoryQualityScorerV2,
    data: dict[str, np.ndarray],
    loss_config: TrajectoryQualityLossConfig,
    device: torch.device,
    *,
    batch_size: int,
) -> dict[str, float]:
    loss_values = []
    selected_ate = []
    selected_rtp = []
    baseline_ate = []
    baseline_rtp = []
    joint = []
    for start in range(0, len(data["window_uid"]), batch_size):
        indices = np.arange(start, min(start + batch_size, len(data["window_uid"])))
        features, target, valid = tensor_batch(data, indices, device)
        prediction = forward_features(model, features)
        loss = trajectory_quality_loss(prediction, target, valid, loss_config)
        loss_values.append((float(loss.total), len(indices)))
        ate_gain = prediction.ate[:, :1] - prediction.ate
        rtp_gain = prediction.rtp - prediction.rtp[:, :1]
        utility = torch.minimum(
            ate_gain / loss_config.ate_scale,
            rtp_gain / loss_config.rtp_scale,
        )
        utility = torch.where(valid, utility, torch.full_like(utility, -torch.inf))
        utility[:, 0] = 0.0
        selected = utility.argmax(dim=1)
        row = torch.arange(len(indices), device=device)
        actual_ate = target[row, selected, 0]
        actual_rtp = target[row, selected, 1]
        selected_ate.extend(actual_ate.cpu().tolist())
        selected_rtp.extend(actual_rtp.cpu().tolist())
        baseline_ate.extend(target[:, 0, 0].cpu().tolist())
        baseline_rtp.extend(target[:, 0, 1].cpu().tolist())
        joint.extend(
            ((actual_ate < target[:, 0, 0]) & (actual_rtp > target[:, 0, 1]))
            .cpu()
            .tolist()
        )
    total_weight = sum(weight for _, weight in loss_values)
    return {
        "loss": sum(value * weight for value, weight in loss_values) / total_weight,
        "selected_ate": float(np.mean(selected_ate)),
        "selected_rtp": float(np.mean(selected_rtp)),
        "candidate0_ate": float(np.mean(baseline_ate)),
        "candidate0_rtp": float(np.mean(baseline_rtp)),
        "delta_ate": float(np.mean(selected_ate) - np.mean(baseline_ate)),
        "delta_rtp": float(np.mean(selected_rtp) - np.mean(baseline_rtp)),
        "joint_improve_fraction": float(np.mean(joint)),
    }


def save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.batch_size < 2 or args.batch_size % 2:
        raise ValueError("batch-size must be even and at least 2")
    if args.epochs <= 0 or args.patience <= 0:
        raise ValueError("epochs/patience must be positive")
    split_lock_sha = file_sha256(args.split_lock)
    train_recon = load_cache(
        args.recon_train,
        dataset="recon",
        partition="train-fit",
        split_lock_sha=split_lock_sha,
    )
    cal_recon = load_cache(
        args.recon_calibration,
        dataset="recon",
        partition=args.calibration_partition,
        split_lock_sha=split_lock_sha,
    )
    train_scand = load_cache(
        args.scand_train,
        dataset="scand",
        partition="train-fit",
        split_lock_sha=split_lock_sha,
    )
    cal_scand = load_cache(
        args.scand_calibration,
        dataset="scand",
        partition=args.calibration_partition,
        split_lock_sha=split_lock_sha,
    )
    for train, calibration, dataset in (
        (train_recon, cal_recon, "recon"),
        (train_scand, cal_scand, "scand"),
    ):
        overlap = set(train["route_id"].astype(str)) & set(
            calibration["route_id"].astype(str)
        )
        if overlap:
            raise RuntimeError(f"{dataset} train/cal route leakage")
    feature_dim = train_recon["features"].shape[-1]
    if any(
        data["features"].shape[-1] != feature_dim
        for data in (cal_recon, train_scand, cal_scand)
    ):
        raise RuntimeError("feature dimensions differ")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = torch.device(args.device)
    config = TrajectoryQualityScorerConfig(
        hidden_dim=args.hidden_dim,
        hidden_layers=args.hidden_layers,
        dropout=args.dropout,
        use_predicted_future=True,
    )
    model = TrajectoryQualityScorerV2(config).to(device)
    feature_mean, feature_scale = domain_balanced_moments(
        train_recon["features"],
        train_scand["features"],
    )
    model.set_normalization(feature_mean, feature_scale)
    ate_scale, rtp_scale = domain_balanced_target_scales(
        train_recon["target"],
        train_scand["target"],
    )
    loss_config = TrajectoryQualityLossConfig(
        ate_scale=ate_scale,
        rtp_scale=rtp_scale,
        absolute_weight=1.0,
        pairwise_weight=0.5,
        dominance_weight=0.25,
        huber_delta=1.0,
        ate_dominance_epsilon=0.01 * ate_scale,
        rtp_dominance_epsilon=0.01 * rtp_scale,
        dominance_margin=0.05,
        dominance_temperature=0.10,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    support = action_support_statistics(
        train_recon["actions_h4"],
        train_scand["actions_h4"],
        seed=args.seed,
    )
    args.output_dir.mkdir(parents=True)
    log_path = args.output_dir / "metrics.jsonl"
    best_path = args.output_dir / "best.pt"
    started = time.perf_counter()
    best_loss = math.inf
    best_epoch = -1
    stale = 0
    half = args.batch_size // 2
    steps_per_epoch = max(
        math.ceil(len(train_recon["window_uid"]) / half),
        math.ceil(len(train_scand["window_uid"]) / half),
    )
    for epoch in range(1, args.epochs + 1):
        model.train()
        generator = np.random.default_rng(args.seed * 1000 + epoch)
        running = []
        for _ in range(steps_per_epoch):
            recon_indices = batch_indices(
                len(train_recon["window_uid"]), half, generator
            )
            scand_indices = batch_indices(
                len(train_scand["window_uid"]), half, generator
            )
            recon_batch = tensor_batch(train_recon, recon_indices, device)
            scand_batch = tensor_batch(train_scand, scand_indices, device)
            features = torch.cat((recon_batch[0], scand_batch[0]), dim=0)
            target = torch.cat((recon_batch[1], scand_batch[1]), dim=0)
            valid = torch.cat((recon_batch[2], scand_batch[2]), dim=0)
            optimizer.zero_grad(set_to_none=True)
            prediction = forward_features(model, features)
            loss = trajectory_quality_loss(
                prediction,
                target,
                valid,
                loss_config,
            )
            loss.total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running.append(float(loss.total.detach()))
        model.eval()
        recon_metrics = evaluate_domain(
            model,
            cal_recon,
            loss_config,
            device,
            batch_size=args.batch_size,
        )
        scand_metrics = evaluate_domain(
            model,
            cal_scand,
            loss_config,
            device,
            batch_size=args.batch_size,
        )
        macro_loss = 0.5 * (recon_metrics["loss"] + scand_metrics["loss"])
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(running)),
            "calibration_domain_macro_loss": macro_loss,
            "recon": recon_metrics,
            "scand": scand_metrics,
            "elapsed_sec": time.perf_counter() - started,
        }
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        print(json.dumps(row, sort_keys=True), flush=True)
        if macro_loss < best_loss - 1.0e-6:
            best_loss = macro_loss
            best_epoch = epoch
            stale = 0
            save_checkpoint(
                best_path,
                {
                    "schema": "trajectory_quality_scorer_v2_checkpoint",
                    "created_at": utc_now(),
                    "seed": args.seed,
                    "epoch": epoch,
                    "calibration_domain_macro_loss": macro_loss,
                    "model_config": asdict(config),
                    "model": model.state_dict(),
                    "loss_config": asdict(loss_config),
                    "action_support": support,
                    "split_lock_sha256": split_lock_sha,
                    "feature_caches": {
                        "recon_train": {
                            "path": str(args.recon_train),
                            "sha256": file_sha256(args.recon_train),
                        },
                        "recon_calibration": {
                            "path": str(args.recon_calibration),
                            "sha256": file_sha256(args.recon_calibration),
                        },
                        "scand_train": {
                            "path": str(args.scand_train),
                            "sha256": file_sha256(args.scand_train),
                        },
                        "scand_calibration": {
                            "path": str(args.scand_calibration),
                            "sha256": file_sha256(args.scand_calibration),
                        },
                    },
                    "world_model_checkpoint_sha256": json.loads(
                        str(train_recon["metadata"].item())
                    )["world_model_checkpoint_sha256"],
                    "parameter_count": model.parameter_count(),
                    "candidate_identity_inputs": False,
                    "candidate0_anchor_target": False,
                    "cross_candidate_model_operations": False,
                    "outer_val_accessed": False,
                    "test_accessed": False,
                    "calibration_partition": args.calibration_partition,
                },
            )
        else:
            stale += 1
            if stale >= args.patience:
                break
    summary = {
        "created_at": utc_now(),
        "decision": "TRAJECTORY_QUALITY_SCORER_V2_TRAINING_COMPLETE",
        "seed": args.seed,
        "best_epoch": best_epoch,
        "best_calibration_domain_macro_loss": best_loss,
        "epochs_completed": epoch,
        "best_checkpoint": str(best_path),
        "best_checkpoint_sha256": file_sha256(best_path),
        "parameter_count": model.parameter_count(),
        "train_windows": {
            "recon": len(train_recon["window_uid"]),
            "scand": len(train_scand["window_uid"]),
        },
        "calibration_windows": {
            "recon": len(cal_recon["window_uid"]),
            "scand": len(cal_scand["window_uid"]),
        },
        "balanced_batch_domains": True,
        "calibration_partition": args.calibration_partition,
        "optimizer": {
            "name": "AdamW",
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "batch_size": args.batch_size,
        },
        "loss_config": asdict(loss_config),
        "split_lock_sha256": split_lock_sha,
        "outer_val_accessed": False,
        "test_accessed": False,
        "elapsed_sec": time.perf_counter() - started,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
