#!/usr/bin/env python3
"""Train one frozen-base action-identifiable dual-latent residual.

The run uses only SCAND train trajectories for optimization and the locked
SCAND validation trajectories for label-free future-latent/action-retrieval
evaluation.  ATE, navigation objective, union65 outcomes, and sealed test data
are never loaded.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [
    str(ROOT),
    str(ROOT / "src"),
    str(Path(__file__).resolve().parent),
]

from latent_window_utils import ShardAwareBatchSampler  # noqa: E402
from recon_latents.dataset_highres_dual_v1 import (  # noqa: E402
    HighResDualLatentWindowDataset,
)
from recon_latents.losses_highres_dual_action_residual_stable_v1 import (  # noqa: E402
    ActionResidualLossWeights,
    compute_action_residual_loss,
    per_candidate_dual_nmse,
)
from recon_latents.models_highres_dual_action_residual_stable_v1 import (  # noqa: E402
    ActionResidualConfig,
    FrozenDualLTWMActionResidual,
)
from recon_latents.models_highres_dual_v1 import (  # noqa: E402
    HighResDualLatentWMConfig,
    HighResMultiLayerDualLatentWM,
)
from recon_latents.split_lock import file_sha256, load_split_lock  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def canonical_json(value: Any) -> str:
    return json.dumps(
        jsonable(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(canonical_json(value) + "\n", encoding="utf-8")
    temporary.replace(path)


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def amp_context(amp: str):
    if amp == "off":
        return nullcontext()
    dtype = torch.bfloat16 if amp == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def load_contract(path: Path, *, smoke: bool) -> dict[str, Any]:
    contract = json.loads(path.read_text(encoding="utf-8"))
    require(
        contract.get("status") == "frozen_before_single_action_residual_run",
        "contract is not frozen",
    )
    for key in ("latent_root", "base_checkpoint", "split_lock"):
        entry = contract["inputs"][key]
        target = resolve(entry["path"])
        require(target.exists(), f"missing input: {target}")
        if target.is_file():
            require(
                file_sha256(target) == entry["sha256"],
                f"input hash changed: {target}",
            )
    require(contract["data"]["ate_or_objective_loaded"] is False, "outcome leakage")
    require(contract["data"]["sealed_test_opened"] is False, "sealed test leakage")
    if smoke:
        contract = json.loads(json.dumps(contract))
        contract["training"]["steps"] = 2
        contract["evaluation"]["maximum_windows"] = 16
    return contract


def make_negative_actions(actions: torch.Tensor) -> torch.Tensor:
    """Use same-trajectory observed actions plus four structured negatives."""

    batch = actions.shape[0]
    rows = []
    for offset in range(1, batch):
        rows.append(torch.roll(actions, shifts=offset, dims=0))
    rows.append(actions.flip(1))
    yaw_flip = actions.clone()
    yaw_flip[..., 2].neg_()
    rows.append(yaw_flip)
    lateral_flip = actions.clone()
    lateral_flip[..., 1].neg_()
    rows.append(lateral_flip)
    magnitude = actions.clone()
    magnitude[..., :2] *= 0.5
    magnitude[..., 2] *= 1.5
    rows.append(magnitude)
    return torch.stack(rows, dim=1)


def load_model(
    checkpoint_path: Path,
    residual_config: ActionResidualConfig,
    device: torch.device,
) -> tuple[FrozenDualLTWMActionResidual, dict[str, Any]]:
    payload = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    require(payload.get("outcome_gradients") is False, "base outcome leakage")
    require(payload.get("sealed_test_opened") is False, "base test leakage")
    base = HighResMultiLayerDualLatentWM(
        HighResDualLatentWMConfig(**payload["config"])
    )
    base.load_state_dict(payload["model"], strict=True)
    model = FrozenDualLTWMActionResidual(base, residual_config).to(device)
    model.base.eval()
    return model, payload


def loaders(
    latent_root: Path,
    contract: dict[str, Any],
) -> tuple[
    HighResDualLatentWindowDataset,
    HighResDualLatentWindowDataset,
    DataLoader,
    DataLoader,
]:
    data = contract["data"]
    training = contract["training"]
    evaluation = contract["evaluation"]
    train = HighResDualLatentWindowDataset(
        latent_root,
        "train",
        history_len=4,
        horizon=4,
        stream_cache_shards=int(data["stream_cache_shards"]),
    )
    val = HighResDualLatentWindowDataset(
        latent_root,
        "val",
        history_len=4,
        horizon=4,
        stream_cache_shards=int(data["stream_cache_shards"]),
    )
    train_loader = DataLoader(
        train,
        batch_sampler=ShardAwareBatchSampler(
            train,
            int(training["batch_size"]),
            drop_last=True,
            seed=int(training["seed"]),
        ),
        num_workers=0,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val,
        batch_sampler=ShardAwareBatchSampler(
            val,
            int(evaluation["batch_size"]),
            drop_last=False,
            seed=int(training["seed"]) + 10000,
        ),
        num_workers=0,
        pin_memory=True,
    )
    return train, val, train_loader, val_loader


def batch_to_device(
    batch: dict[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    return (
        batch["semantic_history"].to(device, non_blocking=True),
        batch["geometry_history"].to(device, non_blocking=True),
        batch["semantic_future"].to(device, non_blocking=True),
        batch["geometry_future"].to(device, non_blocking=True),
        batch["actions"].to(device, non_blocking=True),
    )


def reconstruction_h4(
    output: dict[str, torch.Tensor],
    semantic_target: torch.Tensor,
    geometry_target: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    semantic = output["predicted_semantic_tokens"][:, 0, -1].float()
    geometry = output["predicted_geometry_tokens"][:, 0, -1].float()
    semantic_target = semantic_target[:, -1].float()
    geometry_target = geometry_target[:, -1].float()
    semantic_nmse = (
        (semantic - semantic_target).square().mean(dim=(-1, -2))
        / semantic_target.square().mean(dim=(-1, -2)).clamp_min(1.0e-8)
    )
    geometry_nmse = (
        (geometry - geometry_target).square().mean(dim=(-1, -2))
        / geometry_target.square().mean(dim=(-1, -2)).clamp_min(1.0e-8)
    )
    dual = 0.5 * (semantic_nmse + geometry_nmse)
    return (
        semantic_nmse.cpu().numpy(),
        geometry_nmse.cpu().numpy(),
        dual.cpu().numpy(),
    )


@torch.no_grad()
def evaluate(
    model: FrozenDualLTWMActionResidual,
    loader: DataLoader,
    *,
    device: torch.device,
    amp: str,
    maximum_windows: int,
) -> dict[str, Any]:
    model.eval()
    correct_errors = []
    negative_errors = []
    semantic_h4 = []
    geometry_h4 = []
    dual_h4 = []
    decode_error = []
    route_rows: dict[str, list[tuple[float, float]]] = {}
    windows = 0
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    for batch in loader:
        semantic_history, geometry_history, semantic_future, geometry_future, actions = (
            batch_to_device(batch, device)
        )
        if maximum_windows > 0 and windows + len(actions) > maximum_windows:
            keep = maximum_windows - windows
            if keep <= 0:
                break
            semantic_history = semantic_history[:keep]
            geometry_history = geometry_history[:keep]
            semantic_future = semantic_future[:keep]
            geometry_future = geometry_future[:keep]
            actions = actions[:keep]
            trajectories = [str(value) for value in batch["trajectory_id"][:keep]]
        else:
            trajectories = [str(value) for value in batch["trajectory_id"]]
        negatives = make_negative_actions(actions)
        with amp_context(amp):
            correct = model(semantic_history, geometry_history, actions)
            negative = model(
                semantic_history,
                geometry_history,
                negatives,
            )
        correct_error = per_candidate_dual_nmse(
            correct,
            semantic_future,
            geometry_future,
        ).mean(dim=-1)[:, 0]
        negative_error = per_candidate_dual_nmse(
            negative,
            semantic_future,
            geometry_future,
        ).mean(dim=-1)
        sem, geo, dual = reconstruction_h4(
            correct,
            semantic_future,
            geometry_future,
        )
        decoded = correct["decoded_action_from_predicted_latent"][:, 0]
        decode = (decoded.float() - actions.float()).abs().mean(dim=(1, 2))
        correct_errors.append(correct_error.cpu().numpy())
        negative_errors.append(negative_error.cpu().numpy())
        semantic_h4.append(sem)
        geometry_h4.append(geo)
        dual_h4.append(dual)
        decode_error.append(decode.cpu().numpy())
        negative_mean = negative_error.mean(dim=1).cpu().numpy()
        correct_numpy = correct_error.cpu().numpy()
        for trajectory, correct_value, negative_value in zip(
            trajectories,
            correct_numpy.tolist(),
            negative_mean.tolist(),
        ):
            route_rows.setdefault(trajectory, []).append(
                (float(correct_value), float(negative_value))
            )
        windows += len(actions)
        if windows % 512 < len(actions):
            print(
                canonical_json(
                    {
                        "stage": "validation",
                        "windows": windows,
                        "elapsed_seconds": time.perf_counter() - started,
                    }
                ),
                flush=True,
            )
        if maximum_windows > 0 and windows >= maximum_windows:
            break
    require(windows > 0, "evaluation produced no windows")
    correct_array = np.concatenate(correct_errors)
    negative_array = np.concatenate(negative_errors)
    rank = 1 + (negative_array < correct_array[:, None]).sum(axis=1)
    pairwise = (correct_array[:, None] < negative_array).mean()
    gap = negative_array.mean() - correct_array.mean()
    route_gap = {
        trajectory: float(
            np.mean([negative - correct for correct, negative in values])
        )
        for trajectory, values in route_rows.items()
    }
    return {
        "windows": windows,
        "candidate_count_including_true": int(1 + negative_array.shape[1]),
        "semantic_h4_nmse": float(np.concatenate(semantic_h4).mean()),
        "geometry_h4_nmse": float(np.concatenate(geometry_h4).mean()),
        "dual_h4_nmse": float(np.concatenate(dual_h4).mean()),
        "correct_action_dual_nmse": float(correct_array.mean()),
        "negative_action_dual_nmse": float(negative_array.mean()),
        "negative_action_gap": float(gap),
        "negative_gap_fraction_of_true_error": float(
            gap / max(float(correct_array.mean()), 1.0e-8)
        ),
        "pairwise_action_win_rate": float(pairwise),
        "retrieval_top1": float(np.mean(rank == 1)),
        "mean_reciprocal_rank": float(np.mean(1.0 / rank)),
        "mean_rank": float(rank.mean()),
        "latent_action_decode_mae": float(np.concatenate(decode_error).mean()),
        "positive_gap_trajectory_fraction": float(
            np.mean(np.asarray(list(route_gap.values())) > 0.0)
        ),
        "trajectory_gap": dict(sorted(route_gap.items())),
        "trajectory_count": len(route_gap),
        "runtime_seconds": time.perf_counter() - started,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
    }


def train(
    model: FrozenDualLTWMActionResidual,
    loader: DataLoader,
    contract: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    training = contract["training"]
    amp = str(training["amp"])
    weights = ActionResidualLossWeights(**contract["loss"])
    parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    steps = int(training["steps"])
    rows = []
    started = time.perf_counter()
    model.train()
    completed = 0
    epoch = 0
    while completed < steps:
        epoch += 1
        for batch in loader:
            semantic_history, geometry_history, semantic_future, geometry_future, actions = (
                batch_to_device(batch, device)
            )
            negative_actions = make_negative_actions(actions)
            optimizer.zero_grad(set_to_none=True)
            with amp_context(amp):
                correct = model(
                    semantic_history,
                    geometry_history,
                    actions,
                )
                negatives = model(
                    semantic_history,
                    geometry_history,
                    negative_actions,
                )
                loss, components = compute_action_residual_loss(
                    correct,
                    negatives,
                    semantic_future,
                    geometry_future,
                    semantic_history,
                    geometry_history,
                    actions,
                    negative_actions,
                    weights=weights,
                )
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                parameters,
                float(training["gradient_clip"]),
            )
            optimizer.step()
            rows.append(
                {
                    "loss": float(loss.detach()),
                    "gradient_norm": float(gradient_norm.detach()),
                    **{
                        name: float(value.detach())
                        for name, value in components.items()
                    },
                }
            )
            completed += 1
            if completed % 100 == 0 or completed == steps:
                recent = rows[-100:]
                print(
                    canonical_json(
                        {
                            "stage": "training",
                            "step": completed,
                            "steps": steps,
                            "loss": float(
                                np.mean([row["loss"] for row in recent])
                            ),
                            "contrastive": float(
                                np.mean(
                                    [
                                        row["action_contrastive"]
                                        for row in recent
                                    ]
                                )
                            ),
                            "elapsed_seconds": time.perf_counter() - started,
                        }
                    ),
                    flush=True,
                )
            if completed >= steps:
                break
    summary = {
        name: float(np.mean([row[name] for row in rows]))
        for name in rows[0]
    }
    summary.update(
        {
            "steps": completed,
            "epochs_touched": epoch,
            "runtime_seconds": time.perf_counter() - started,
            "trainable_parameters": int(
                sum(parameter.numel() for parameter in parameters)
            ),
            "base_trainable_parameters": int(
                sum(
                    parameter.numel()
                    for parameter in model.base.parameters()
                    if parameter.requires_grad
                )
            ),
        }
    )
    return summary


def gate_result(
    baseline: dict[str, Any],
    final: dict[str, Any],
    contract: dict[str, Any],
) -> tuple[bool, dict[str, bool], dict[str, float]]:
    gate = contract["go_gate"]
    degradation = (
        final["dual_h4_nmse"] - baseline["dual_h4_nmse"]
    ) / baseline["dual_h4_nmse"]
    gap_improvement = (
        final["negative_gap_fraction_of_true_error"]
        - baseline["negative_gap_fraction_of_true_error"]
    )
    checks = {
        "reconstruction_preserved": degradation
        <= float(gate["maximum_relative_reconstruction_degradation"]),
        "absolute_action_gap": final["negative_gap_fraction_of_true_error"]
        >= float(gate["minimum_negative_gap_fraction"]),
        "action_gap_improved": gap_improvement
        >= float(gate["minimum_gap_fraction_improvement"]),
        "pairwise_action_retrieval": final["pairwise_action_win_rate"]
        >= float(gate["minimum_pairwise_win_rate"]),
        "top1_action_retrieval": final["retrieval_top1"]
        >= float(gate["minimum_retrieval_top1"]),
        "trajectory_consistency": final["positive_gap_trajectory_fraction"]
        >= float(gate["minimum_positive_gap_trajectory_fraction"]),
    }
    return all(checks.values()), checks, {
        "relative_reconstruction_degradation": float(degradation),
        "gap_fraction_improvement": float(gap_improvement),
    }


def main() -> None:
    args = parse_args()
    require(torch.cuda.is_available(), "CUDA unavailable")
    output = args.output_dir.resolve()
    require(not output.exists(), f"refusing to overwrite: {output}")
    contract_path = args.contract.resolve()
    contract = load_contract(contract_path, smoke=args.smoke)
    seed = int(contract["training"]["seed"])
    set_seed(seed)
    latent_root = resolve(contract["inputs"]["latent_root"]["path"])
    checkpoint_path = resolve(contract["inputs"]["base_checkpoint"]["path"])
    split_lock_path = resolve(contract["inputs"]["split_lock"]["path"])
    split_lock, split_lock_sha = load_split_lock(split_lock_path)
    require(
        split_lock_sha == contract["inputs"]["split_lock"]["sha256"],
        "split lock hash changed",
    )
    splits = json.loads((latent_root / "splits.json").read_text())
    require(not splits.get("test"), "sealed test trajectories exposed")
    for split in ("train", "val"):
        require(
            set(splits[split])
            == set(split_lock["scand"]["splits"][split]),
            f"{split} split changed",
        )

    residual_config = ActionResidualConfig(**contract["model"])
    device = torch.device("cuda")
    model, base_payload = load_model(
        checkpoint_path,
        residual_config,
        device,
    )
    train_dataset, val_dataset, train_loader, val_loader = loaders(
        latent_root,
        contract,
    )
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    baseline = evaluate(
        model,
        val_loader,
        device=device,
        amp=str(contract["training"]["amp"]),
        maximum_windows=int(contract["evaluation"]["maximum_windows"]),
    )
    training = train(model, train_loader, contract, device)
    final = evaluate(
        model,
        val_loader,
        device=device,
        amp=str(contract["training"]["amp"]),
        maximum_windows=int(contract["evaluation"]["maximum_windows"]),
    )
    go, checks, deltas = gate_result(baseline, final, contract)
    checkpoint = {
        "residual_model": model.residual_state_dict(),
        "contract": model.contract_dict(),
        "base_checkpoint": {
            "path": str(checkpoint_path),
            "sha256": file_sha256(checkpoint_path),
        },
        "experiment_contract": {
            "path": str(contract_path),
            "sha256": file_sha256(contract_path),
        },
        "loss_weights": asdict(
            ActionResidualLossWeights(**contract["loss"])
        ),
        "training": training,
        "baseline_validation": baseline,
        "final_validation": final,
        "go": go,
        "outcome_gradients": False,
        "ate_or_objective_loaded": False,
        "sealed_test_opened": False,
    }
    checkpoint_path_out = output / "action_residual_final.pt"
    torch.save(checkpoint, checkpoint_path_out)
    result = {
        "schema": "highres_dual_action_residual_result_v1",
        "status": (
            "smoke_complete"
            if args.smoke
            else "complete_single_frozen_base_action_residual_run"
        ),
        "decision": (
            "go_action_identifiability_improved_without_losing_reconstruction"
            if go
            else "no_go_action_residual_did_not_meet_label_free_held_route_gate"
        ),
        "go": go,
        "checks": checks,
        "derived_deltas": deltas,
        "baseline_validation": baseline,
        "final_validation": final,
        "training": training,
        "data": {
            "train_windows": len(train_dataset),
            "validation_windows": len(val_dataset),
            "train_trajectories": len(train_dataset.trajectories),
            "validation_trajectories": len(val_dataset.trajectories),
            "split_lock_sha256": split_lock_sha,
        },
        "inputs": {
            "contract": {
                "path": str(contract_path),
                "sha256": file_sha256(contract_path),
            },
            "base_checkpoint": {
                "path": str(checkpoint_path),
                "sha256": file_sha256(checkpoint_path),
            },
            "latent_metadata": {
                "path": str(latent_root / "metadata.json"),
                "sha256": file_sha256(latent_root / "metadata.json"),
            },
            "runner": {
                "path": str(Path(__file__).resolve()),
                "sha256": file_sha256(Path(__file__).resolve()),
            },
        },
        "artifacts": {
            "checkpoint": {
                "path": str(checkpoint_path_out),
                "sha256": file_sha256(checkpoint_path_out),
            }
        },
        "base_parameter_count": int(base_payload["parameter_count"]),
        "base_frozen": True,
        "ate_or_objective_loaded": False,
        "outcome_gradients": False,
        "sealed_test_opened": False,
        "runtime_seconds": time.perf_counter() - started,
    }
    result_path = output / "result.json"
    write_json(result_path, result)
    print(
        canonical_json(
            {
                "status": result["status"],
                "decision": result["decision"],
                "go": go,
                "baseline_dual_h4_nmse": baseline["dual_h4_nmse"],
                "final_dual_h4_nmse": final["dual_h4_nmse"],
                "baseline_gap_fraction": baseline[
                    "negative_gap_fraction_of_true_error"
                ],
                "final_gap_fraction": final[
                    "negative_gap_fraction_of_true_error"
                ],
                "final_pairwise_win_rate": final[
                    "pairwise_action_win_rate"
                ],
                "final_retrieval_top1": final["retrieval_top1"],
                "result": str(result_path),
                "result_sha256": file_sha256(result_path),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(
            canonical_json(
                {
                    "status": "fail_closed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "ate_or_objective_loaded": False,
                    "sealed_test_opened": False,
                }
            ),
            file=sys.stderr,
        )
        raise SystemExit(2) from error
