#!/usr/bin/env python3
"""Train an ATE-primary v70 scorer-v2 on RECON, SCAND and SACSoN.

RTP is a bounded secondary target so near-zero start/goal distances cannot
dominate the target scale. Target scales use per-domain winsorization, batches
traverse shuffled permutations, and checkpoint selection prioritizes inner
calibration ATE. The outer test split is never loaded.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scoring.trajectory_quality_objective_v2 import (  # noqa: E402
    TrajectoryQualityLossConfig,
    trajectory_quality_loss,
)
from scoring.trajectory_quality_scorer_v2 import (  # noqa: E402
    TrajectoryQualityScorerConfig,
    TrajectoryQualityScorerV2,
)
from tools.scoring import train_trajectory_quality_scorer_v2 as base  # noqa: E402


DOMAINS = ("recon", "scand", "sacson")
RTP_TARGET_LOWER = -1.0
RTP_TARGET_UPPER = 1.0
ROBUST_QUANTILE_LOWER = 0.01
ROBUST_QUANTILE_UPPER = 0.99


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for domain in DOMAINS:
        parser.add_argument(f"--{domain}-train", type=Path, required=True)
        parser.add_argument(f"--{domain}-calibration", type=Path, required=True)
    parser.add_argument("--split-lock", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--calibration-partition", default="scorer-cal")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--learning-rate", type=float, default=2.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=640)
    parser.add_argument("--hidden-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--rtp-target-lower", type=float, default=RTP_TARGET_LOWER)
    parser.add_argument("--rtp-target-upper", type=float, default=RTP_TARGET_UPPER)
    parser.add_argument(
        "--robust-quantile-lower",
        type=float,
        default=ROBUST_QUANTILE_LOWER,
    )
    parser.add_argument(
        "--robust-quantile-upper",
        type=float,
        default=ROBUST_QUANTILE_UPPER,
    )
    parser.add_argument(
        "--rtp-selection-weight",
        type=float,
        default=0.15,
        help="Secondary RTP weight in calibration candidate selection.",
    )
    parser.add_argument("--rtp-regression-penalty", type=float, default=0.05)
    parser.add_argument("--worst-domain-ate-penalty", type=float, default=2.0)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def moments(*arrays: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = [array.astype(np.float64).reshape(-1, array.shape[-1]) for array in arrays]
    mean = np.mean([value.mean(axis=0) for value in values], axis=0)
    second = np.mean([np.square(value).mean(axis=0) for value in values], axis=0)
    scale = np.sqrt(np.maximum(second - np.square(mean), 1.0e-12))
    scale[scale < 1.0e-5] = 1.0
    return mean.astype(np.float32), scale.astype(np.float32)


def bound_rtp_targets(
    data: dict[str, np.ndarray],
    *,
    lower: float,
    upper: float,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Return a cache view with RTP clipped to a bounded progress contract."""

    target = data["target"].astype(np.float32, copy=True)
    raw_rtp = target[..., 1].astype(np.float64)
    if not np.isfinite(raw_rtp).all():
        raise RuntimeError("RTP target contains NaN or Inf")
    target[..., 1] = np.clip(raw_rtp, lower, upper).astype(np.float32)
    result = dict(data)
    result["target"] = target
    audit = {
        "contract": "bounded_relative_target_progress",
        "lower": float(lower),
        "upper": float(upper),
        "values": int(raw_rtp.size),
        "clipped_low": int(np.count_nonzero(raw_rtp < lower)),
        "clipped_high": int(np.count_nonzero(raw_rtp > upper)),
        "raw_min": float(raw_rtp.min()),
        "raw_max": float(raw_rtp.max()),
        "bounded_min": float(target[..., 1].min()),
        "bounded_max": float(target[..., 1].max()),
    }
    return result, audit


def robust_target_scales(
    arrays: list[np.ndarray],
    valid_masks: list[np.ndarray],
    *,
    quantile_lower: float,
    quantile_upper: float,
) -> tuple[float, float, dict[str, Any]]:
    """Estimate equal-domain scales from valid, winsorized targets."""

    domain_scales = []
    audit: dict[str, Any] = {"domains": {}}
    for domain, array, valid in zip(DOMAINS, arrays, valid_masks, strict=True):
        values = array.astype(np.float64).reshape(-1, 2)
        mask = valid.astype(bool).reshape(-1)
        if len(values) != len(mask):
            raise RuntimeError(f"{domain} target/valid shapes differ")
        selected = values[mask & np.isfinite(values).all(axis=1)]
        if len(selected) < 2:
            raise RuntimeError(f"{domain} has fewer than two valid targets")
        low = np.quantile(selected, quantile_lower, axis=0)
        high = np.quantile(selected, quantile_upper, axis=0)
        winsorized = np.clip(selected, low, high)
        scale = winsorized.std(axis=0)
        domain_scales.append(scale)
        audit["domains"][domain] = {
            "valid_candidates": int(len(selected)),
            "quantile_lower": low.tolist(),
            "quantile_upper": high.tolist(),
            "winsorized_std": scale.tolist(),
        }
    scale = np.mean(domain_scales, axis=0)
    scale = np.maximum(scale, np.asarray((0.02, 0.02)))
    audit.update(
        {
            "estimator": "equal-domain mean of valid per-domain winsorized std",
            "quantile_lower": float(quantile_lower),
            "quantile_upper": float(quantile_upper),
            "ate_scale": float(scale[0]),
            "rtp_scale": float(scale[1]),
        }
    )
    return float(scale[0]), float(scale[1]), audit


class PermutationBatchStream:
    """Cycle through shuffled permutations without within-cycle repeats."""

    def __init__(
        self,
        count: int,
        batch_count: int,
        rng: np.random.Generator,
    ) -> None:
        if count <= 0 or batch_count <= 0:
            raise ValueError("sampler counts must be positive")
        self.count = count
        self.batch_count = batch_count
        self.rng = rng
        self.order = self.rng.permutation(count)
        self.offset = 0
        self.cycles = 0
        self.draws = 0
        self.seen = np.zeros(count, dtype=bool)

    def take(self) -> np.ndarray:
        chunks = []
        remaining = self.batch_count
        while remaining:
            available = self.count - self.offset
            take = min(remaining, available)
            chunk = self.order[self.offset : self.offset + take]
            chunks.append(chunk)
            self.seen[chunk] = True
            self.offset += take
            self.draws += take
            remaining -= take
            if self.offset == self.count:
                self.cycles += 1
                self.order = self.rng.permutation(self.count)
                self.offset = 0
        return np.concatenate(chunks)

    def audit(self) -> dict[str, Any]:
        return {
            "windows": self.count,
            "draws": self.draws,
            "complete_cycles": self.cycles,
            "unique_windows_seen": int(self.seen.sum()),
            "unique_fraction": float(self.seen.mean()),
        }


@torch.inference_mode()
def evaluate_domain_ate_primary(
    model: TrajectoryQualityScorerV2,
    data: dict[str, np.ndarray],
    loss_config: TrajectoryQualityLossConfig,
    device: torch.device,
    *,
    batch_size: int,
    rtp_weight: float,
) -> dict[str, float]:
    loss_values = []
    selected_ate = []
    selected_rtp = []
    baseline_ate = []
    baseline_rtp = []
    joint = []
    for start in range(0, len(data["window_uid"]), batch_size):
        indices = np.arange(start, min(start + batch_size, len(data["window_uid"])))
        features, target, valid = base.tensor_batch(data, indices, device)
        prediction = base.forward_features(model, features)
        loss = trajectory_quality_loss(prediction, target, valid, loss_config)
        loss_values.append((float(loss.total), len(indices)))
        ate_gain = (prediction.ate[:, :1] - prediction.ate) / loss_config.ate_scale
        rtp_gain = (prediction.rtp - prediction.rtp[:, :1]) / loss_config.rtp_scale
        utility = ate_gain + rtp_weight * rtp_gain
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
    candidate0_ate = float(np.mean(baseline_ate))
    candidate0_rtp = float(np.mean(baseline_rtp))
    selected_ate_mean = float(np.mean(selected_ate))
    selected_rtp_mean = float(np.mean(selected_rtp))
    return {
        "loss": sum(value * weight for value, weight in loss_values) / total_weight,
        "selected_ate": selected_ate_mean,
        "selected_rtp": selected_rtp_mean,
        "candidate0_ate": candidate0_ate,
        "candidate0_rtp": candidate0_rtp,
        "delta_ate": selected_ate_mean - candidate0_ate,
        "delta_rtp": selected_rtp_mean - candidate0_rtp,
        "ate_ratio": selected_ate_mean / max(candidate0_ate, 1.0e-8),
        "rtp_regression": max(0.0, candidate0_rtp - selected_rtp_mean),
        "joint_improve_fraction": float(np.mean(joint)),
    }


def action_support(*arrays: np.ndarray, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    samples = []
    for array in arrays:
        flat = array.astype(np.float32).reshape(-1, 4, 3)
        count = min(40000, len(flat))
        samples.append(flat[rng.choice(len(flat), size=count, replace=False)])
    actions = np.concatenate(samples, axis=0)
    lower = np.quantile(actions, 0.01, axis=0).astype(np.float32)
    upper = np.quantile(actions, 0.99, axis=0).astype(np.float32)
    return {
        "source": "equal-domain observed H4 action support",
        "domains": list(DOMAINS),
        "seed": seed,
        "fit_candidates": int(len(actions)),
        "lower": lower.tolist(),
        "upper": upper.tolist(),
    }


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.batch_size < 3 or args.batch_size % 3:
        raise ValueError("batch-size must be divisible by three")
    if args.epochs <= 0 or args.patience <= 0:
        raise ValueError("epochs/patience must be positive")
    if not args.rtp_target_lower < args.rtp_target_upper:
        raise ValueError("RTP target bounds are invalid")
    if not 0 <= args.robust_quantile_lower < args.robust_quantile_upper <= 1:
        raise ValueError("robust quantiles are invalid")
    if args.rtp_selection_weight < 0 or args.rtp_regression_penalty < 0:
        raise ValueError("RTP selection weights must be non-negative")
    if args.worst_domain_ate_penalty < 0:
        raise ValueError("worst-domain ATE penalty must be non-negative")

    seed_everything(args.seed)
    split_lock_sha = base.file_sha256(args.split_lock)
    train: dict[str, dict[str, np.ndarray]] = {}
    calibration: dict[str, dict[str, np.ndarray]] = {}
    for domain in DOMAINS:
        train[domain] = base.load_cache(
            getattr(args, f"{domain}_train"),
            dataset=domain,
            partition="train-fit",
            split_lock_sha=split_lock_sha,
        )
        calibration[domain] = base.load_cache(
            getattr(args, f"{domain}_calibration"),
            dataset=domain,
            partition=args.calibration_partition,
            split_lock_sha=split_lock_sha,
        )
        overlap = set(train[domain]["route_id"].astype(str)) & set(
            calibration[domain]["route_id"].astype(str)
        )
        if overlap:
            raise RuntimeError(f"{domain} train/calibration route overlap")

    target_audit: dict[str, Any] = {"train": {}, "calibration": {}}
    for partition, datasets in (("train", train), ("calibration", calibration)):
        for domain in DOMAINS:
            datasets[domain], target_audit[partition][domain] = bound_rtp_targets(
                datasets[domain],
                lower=args.rtp_target_lower,
                upper=args.rtp_target_upper,
            )

    feature_dim = train["recon"]["features"].shape[-1]
    all_data = [train[domain] for domain in DOMAINS] + [
        calibration[domain] for domain in DOMAINS
    ]
    if any(data["features"].shape[-1] != feature_dim for data in all_data):
        raise RuntimeError("feature dimensions differ across domains")

    device = torch.device(args.device)
    config = TrajectoryQualityScorerConfig(
        hidden_dim=args.hidden_dim,
        hidden_layers=args.hidden_layers,
        dropout=args.dropout,
        use_predicted_future=True,
    )
    model = TrajectoryQualityScorerV2(config).to(device)
    mean, scale = moments(*(train[domain]["features"] for domain in DOMAINS))
    model.set_normalization(mean, scale)
    ate_scale, rtp_scale, target_scale_audit = robust_target_scales(
        [train[domain]["target"] for domain in DOMAINS],
        [train[domain]["valid"] for domain in DOMAINS],
        quantile_lower=args.robust_quantile_lower,
        quantile_upper=args.robust_quantile_upper,
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

    args.output_dir.mkdir(parents=True)
    log_path = args.output_dir / "metrics.jsonl"
    best_path = args.output_dir / "best.pt"
    per_domain_batch = args.batch_size // 3
    steps_per_epoch = max(
        math.ceil(len(train[domain]["window_uid"]) / per_domain_batch)
        for domain in DOMAINS
    )
    best_selection_score = math.inf
    best_macro_loss = math.inf
    best_epoch = -1
    stale = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        samplers = {
            domain: PermutationBatchStream(
                len(train[domain]["window_uid"]),
                per_domain_batch,
                np.random.default_rng(args.seed * 100_000 + epoch * 10 + index),
            )
            for index, domain in enumerate(DOMAINS)
        }
        running: list[float] = []
        for _ in range(steps_per_epoch):
            batches = []
            for domain in DOMAINS:
                indices = samplers[domain].take()
                batches.append(base.tensor_batch(train[domain], indices, device))
            features = torch.cat([batch[0] for batch in batches], dim=0)
            target = torch.cat([batch[1] for batch in batches], dim=0)
            valid = torch.cat([batch[2] for batch in batches], dim=0)
            optimizer.zero_grad(set_to_none=True)
            prediction = base.forward_features(model, features)
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
        domain_metrics = {
            domain: evaluate_domain_ate_primary(
                model,
                calibration[domain],
                loss_config,
                device,
                batch_size=args.batch_size,
                rtp_weight=args.rtp_selection_weight,
            )
            for domain in DOMAINS
        }
        macro_loss = float(np.mean([domain_metrics[domain]["loss"] for domain in DOMAINS]))
        macro_ate_ratio = float(
            np.mean([domain_metrics[domain]["ate_ratio"] for domain in DOMAINS])
        )
        macro_rtp_regression = float(
            np.mean([domain_metrics[domain]["rtp_regression"] for domain in DOMAINS])
        )
        worst_domain_ate_regression = max(
            0.0,
            max(domain_metrics[domain]["ate_ratio"] for domain in DOMAINS) - 1.0,
        )
        selection_score = (
            macro_ate_ratio
            + args.rtp_regression_penalty * macro_rtp_regression / rtp_scale
            + args.worst_domain_ate_penalty * worst_domain_ate_regression
        )
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(running)),
            "calibration_domain_macro_loss": macro_loss,
            "calibration_ate_primary_score": selection_score,
            "calibration_macro_ate_ratio": macro_ate_ratio,
            "calibration_macro_rtp_regression": macro_rtp_regression,
            "calibration_worst_domain_ate_regression": worst_domain_ate_regression,
            "sampler": {domain: samplers[domain].audit() for domain in DOMAINS},
            "domains": domain_metrics,
        }
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        print(json.dumps(row, sort_keys=True), flush=True)

        if selection_score < best_selection_score - 1.0e-6:
            best_selection_score = selection_score
            best_macro_loss = macro_loss
            best_epoch = epoch
            stale = 0
            payload = {
                "schema": "trajectory_quality_scorer_v2_tridomain_ate_primary_checkpoint",
                "seed": args.seed,
                "epoch": epoch,
                "calibration_domain_macro_loss": macro_loss,
                "calibration_ate_primary_score": selection_score,
                "calibration_domains": domain_metrics,
                "model_config": asdict(config),
                "model": model.state_dict(),
                "loss_config": asdict(loss_config),
                "target_contract": target_audit,
                "target_scale_audit": target_scale_audit,
                "selection_config": {
                    "mode": "ate_primary_weighted_rtp",
                    "rtp_weight": args.rtp_selection_weight,
                    "rtp_regression_penalty": args.rtp_regression_penalty,
                    "worst_domain_ate_penalty": args.worst_domain_ate_penalty,
                    "primary_metric": "ATE",
                    "outer_test_metrics_used": False,
                },
                "action_support": action_support(
                    *(train[domain]["actions_h4"] for domain in DOMAINS),
                    seed=args.seed,
                ),
                "split_lock_sha256": split_lock_sha,
                "feature_caches": {
                    f"{domain}_{partition}": {
                        "path": str(getattr(args, f"{domain}_{partition}")),
                        "sha256": base.file_sha256(
                            getattr(args, f"{domain}_{partition}")
                        ),
                    }
                    for domain in DOMAINS
                    for partition in ("train", "calibration")
                },
                "world_model_checkpoint_sha256": json.loads(
                    str(train["recon"]["metadata"].item())
                )["world_model_checkpoint_sha256"],
                "parameter_count": model.parameter_count(),
                "outer_val_accessed": False,
                "test_accessed": False,
                "balanced_domains": list(DOMAINS),
            }
            base.save_checkpoint(best_path, payload)
        else:
            stale += 1
            if stale >= args.patience:
                break

    summary = {
        "schema": "trajectory_quality_scorer_v2_tridomain_ate_primary_summary",
        "decision": "TRIDOMAIN_ATE_PRIMARY_SCORER_V2_TRAINING_COMPLETE",
        "seed": args.seed,
        "best_epoch": best_epoch,
        "best_calibration_ate_primary_score": best_selection_score,
        "best_calibration_domain_macro_loss": best_macro_loss,
        "epochs_completed": epoch,
        "best_checkpoint": str(best_path),
        "best_checkpoint_sha256": base.file_sha256(best_path),
        "train_windows": {
            domain: len(train[domain]["window_uid"]) for domain in DOMAINS
        },
        "calibration_windows": {
            domain: len(calibration[domain]["window_uid"]) for domain in DOMAINS
        },
        "target_contract": target_audit,
        "target_scale_audit": target_scale_audit,
        "checkpoint_primary_metric": "ATE",
        "balanced_batch_domains": list(DOMAINS),
        "split_lock_sha256": split_lock_sha,
        "outer_val_accessed": False,
        "test_accessed": False,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
