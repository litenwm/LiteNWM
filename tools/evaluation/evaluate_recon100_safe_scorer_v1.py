#!/usr/bin/env python3
"""Compare frozen DINOv3 scorers with the 019f70d5 protocol on RECON-100.

The old candidate pools and split definitions are reused, but no old scorer
weight is migrated.  ``freeze`` calibrates a relative safe-promotion wrapper
on cal32.  ``test`` requires that frozen policy, performs label-free DINOv3
inference on strict_external100, and only then opens the old outcome artifact.

This is a fixed-pool scorer experiment.  It never invokes or enables CEM.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import pickle
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from planning.nomad_action_adapter import (  # noqa: E402
    integrate_local_actions,
    nomad_waypoints_to_local_actions,
)
from planning.trajectory_constraints import (  # noqa: E402
    ActionSupportStats,
    evaluate_trajectory_constraints,
)
from recon_latents.dinov3_adapter import (  # noqa: E402
    DinoV3AdapterConfig,
    DinoV3TokenAdapter,
    extract_dinov3_patch_tokens,
)
from recon_latents.models_v2 import (  # noqa: E402
    LiteTokenFormerConfig,
    LiteTokenFormerWMV2,
)
from scoring.safe_promotion_v1 import (  # noqa: E402
    SafePromotionPolicy,
    apply_safe_promotion,
    calibrate_safe_promotion,
    greedy_fixed_pool_selection,
    selection_metrics,
    wilson_interval,
)
from scoring.trajectory_quality_scorer_v2 import (  # noqa: E402
    TrajectoryQualityScorerConfig,
    TrajectoryQualityScorerV2,
)


K = 16
HORIZON = 4
INCUMBENT = 0
DEFAULT_LEGACY_ROOT = Path("/home/rainbow/ndwm/DWM/reports")
DEFAULT_TRAJECTORY_ROOT = Path("/home/rainbow/nwm/data/recon")
DEFAULT_MODEL_DIR = Path(
    "/home/rainbow/models/facebook/dinov3-vitl16-pretrain-lvd1689m"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("freeze", "test", "all"), required=True)
    parser.add_argument("--legacy-root", type=Path, default=DEFAULT_LEGACY_ROOT)
    parser.add_argument("--trajectory-root", type=Path, default=DEFAULT_TRAJECTORY_ROOT)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument(
        "--adapter-checkpoint",
        type=Path,
        default=REPO_ROOT
        / "checkpoints/project/dinov3_adapter/adapter_pilot_v1_final.pt",
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=REPO_ROOT / "checkpoints/project",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs/recon100_dinov3_safe_scorer_v1",
    )
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--dinov3-batch-size", type=int, default=8)
    parser.add_argument("--wm-batch-size", type=int, default=2)
    parser.add_argument("--candidate-chunk", type=int, default=4)
    parser.add_argument("--uncertainty-beta", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_npz_atomic(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
    temporary.replace(path)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def pool_paths(legacy_root: Path, split: str) -> dict[str, Path]:
    candidate_root = legacy_root / "recon_h4_scorer_candidates_v1"
    return {
        "manifest": candidate_root / f"{split}.json",
        "candidates": candidate_root / f"{split}.npz",
        "labels": legacy_root / "recon_h4_scorer_labels_v2" / split / "labels.npz",
    }


def load_pool(legacy_root: Path, split: str) -> dict[str, Any]:
    paths = pool_paths(legacy_root, split)
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    records = manifest.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError(f"invalid legacy manifest: {paths['manifest']}")
    with np.load(paths["candidates"], allow_pickle=False) as archive:
        waypoints = archive["candidates"].astype(np.float32)
    if waypoints.shape != (len(records), K, HORIZON, 2):
        raise ValueError(f"invalid candidate pool shape: {waypoints.shape}")
    for index, record in enumerate(records):
        current = int(record["curr_time"])
        if int(record["sample_idx"]) != index:
            raise ValueError("legacy sample_idx/order mismatch")
        if int(record["goal_time"]) != current + HORIZON:
            raise ValueError("legacy goal is not current+4")
        if [int(value) for value in record["context_times"]] != list(
            range(current - 3, current + 1)
        ):
            raise ValueError("legacy context is not [t-3,t]")
    actions = nomad_waypoints_to_local_actions(waypoints, dataset="recon")
    reconstructed = integrate_local_actions(actions)
    if not np.allclose(reconstructed, waypoints * 0.25, atol=1.0e-6, rtol=1.0e-6):
        raise RuntimeError("NoMaD waypoint-to-SE2 conversion is not lossless")
    return {
        "split": split,
        "records": records,
        "waypoints": waypoints,
        "actions": actions.astype(np.float32),
        "paths": paths,
        "candidate_sha256": file_sha256(paths["candidates"]),
        "manifest_sha256": file_sha256(paths["manifest"]),
    }


def image_path(root: Path, trajectory_id: str, timestep: int) -> Path:
    if not trajectory_id or trajectory_id in {".", ".."} or "/" in trajectory_id or "\\" in trajectory_id:
        raise ValueError(f"unsafe trajectory id: {trajectory_id!r}")
    resolved_root = root.resolve(strict=True)
    path = (resolved_root / trajectory_id / f"{timestep}.jpg").resolve(strict=True)
    try:
        path.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError(f"image path escapes trajectory root: {path}") from error
    return path


def preprocess(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        image = image.convert("RGB").resize((224, 224), Image.Resampling.BICUBIC)
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = (
        array - np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
    ) / np.asarray((0.229, 0.224, 0.225), dtype=np.float32)
    return torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1)))


def latent_binding(
    pool: dict[str, Any],
    *,
    trajectory_root: Path,
    model_dir: Path,
    adapter_checkpoint: Path,
) -> dict[str, Any]:
    return {
        "schema": "recon100_targeted_dinov3_latents_v1",
        "split": pool["split"],
        "states": len(pool["records"]),
        "candidate_sha256": pool["candidate_sha256"],
        "manifest_sha256": pool["manifest_sha256"],
        "trajectory_root": str(trajectory_root.resolve(strict=True)),
        "dinov3_weight_sha256": file_sha256(model_dir / "model.safetensors"),
        "adapter_checkpoint_sha256": file_sha256(adapter_checkpoint),
        "preprocess": "RGB_bicubic_224_imagenet_norm",
        "history": 4,
        "goal_offset": 4,
    }


def validate_cached_binding(path: Path, expected: dict[str, Any]) -> dict[str, np.ndarray]:
    arrays = load_npz(path)
    if "metadata" not in arrays or np.asarray(arrays["metadata"]).ndim != 0:
        raise RuntimeError(f"cache metadata is missing: {path}")
    actual = json.loads(str(arrays["metadata"].item()))
    if actual != expected:
        raise RuntimeError(f"cache provenance mismatch: {path}")
    return arrays


def encode_targeted_latents(
    pool: dict[str, Any],
    *,
    trajectory_root: Path,
    model_dir: Path,
    adapter_checkpoint: Path,
    output: Path,
    device: torch.device,
    batch_size: int,
) -> dict[str, np.ndarray]:
    binding = latent_binding(
        pool,
        trajectory_root=trajectory_root,
        model_dir=model_dir,
        adapter_checkpoint=adapter_checkpoint,
    )
    if output.is_file():
        arrays = validate_cached_binding(output, binding)
        expected_history = (len(pool["records"]), 4, 256, 384)
        if arrays.get("history", np.empty(0)).shape != expected_history:
            raise RuntimeError("cached history latent shape mismatch")
        if arrays.get("goal", np.empty(0)).shape != (len(pool["records"]), 256, 384):
            raise RuntimeError("cached goal latent shape mismatch")
        return arrays

    adapter_payload = torch.load(adapter_checkpoint, map_location="cpu", weights_only=False)
    if adapter_payload.get("dinov3_weight_sha256") != binding["dinov3_weight_sha256"]:
        raise ValueError("adapter/DINOv3 weight mismatch")
    adapter = DinoV3TokenAdapter(DinoV3AdapterConfig(**adapter_payload["config"]))
    adapter.load_state_dict(adapter_payload["model"], strict=True)
    from transformers import AutoModel

    backbone = AutoModel.from_pretrained(
        model_dir,
        local_files_only=True,
        dtype=torch.float32,
    ).eval().to(device).requires_grad_(False)
    adapter = adapter.eval().to(device).requires_grad_(False)

    paths: list[Path] = []
    key_to_index: dict[str, int] = {}
    record_keys: list[list[str]] = []
    for record in pool["records"]:
        trajectory_id = str(record["traj_name"])
        times = [int(value) for value in record["context_times"]] + [
            int(record["goal_time"])
        ]
        keys = []
        for timestep in times:
            path = image_path(trajectory_root, trajectory_id, timestep)
            key = str(path)
            if key not in key_to_index:
                key_to_index[key] = len(paths)
                paths.append(path)
            keys.append(key)
        record_keys.append(keys)

    encoded_batches = []
    started = time.perf_counter()
    for start in range(0, len(paths), batch_size):
        images = torch.stack([preprocess(path) for path in paths[start : start + batch_size]]).to(device)
        with torch.inference_mode():
            hidden = backbone(pixel_values=images).last_hidden_state
            adapted = adapter(extract_dinov3_patch_tokens(hidden))
        encoded_batches.append(adapted.cpu().numpy().astype(np.float16))
        print(
            json.dumps(
                {
                    "stage": "targeted_dinov3_encode",
                    "split": pool["split"],
                    "completed_images": min(start + batch_size, len(paths)),
                    "total_images": len(paths),
                    "elapsed_sec": time.perf_counter() - started,
                }
            ),
            flush=True,
        )
    encoded = np.concatenate(encoded_batches)
    history = []
    goal = []
    for keys in record_keys:
        values = encoded[[key_to_index[key] for key in keys]]
        history.append(values[:4].reshape(4, 256, 384))
        goal.append(values[4].reshape(256, 384))
    arrays = {
        "history": np.stack(history).astype(np.float16),
        "goal": np.stack(goal).astype(np.float16),
        "metadata": np.asarray(canonical_json(binding)),
    }
    write_npz_atomic(output, **arrays)
    del backbone, adapter, encoded, encoded_batches
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return arrays


def checkpoint_pairs(checkpoint_root: Path) -> list[dict[str, Any]]:
    pairs = []
    for seed in (1701, 1702, 1703):
        wm = checkpoint_root / "ltwm_expansion" / f"seed{seed}" / "best.pt"
        scorer = checkpoint_root / "scorer_v2" / f"seed{seed}" / "best.pt"
        if not wm.is_file() or not scorer.is_file():
            raise FileNotFoundError(f"missing seed{seed} WM/scorer checkpoint")
        pairs.append(
            {
                "seed": seed,
                "wm": wm,
                "scorer": scorer,
                "wm_sha256": file_sha256(wm),
                "scorer_sha256": file_sha256(scorer),
            }
        )
    return pairs


def support_stats(value: dict[str, Any]) -> ActionSupportStats:
    return ActionSupportStats(
        lower=torch.as_tensor(value["lower_p01"], dtype=torch.float32),
        upper=torch.as_tensor(value["upper_p99"], dtype=torch.float32),
        mean=torch.as_tensor(value["mean"], dtype=torch.float32),
        covariance_inverse=torch.as_tensor(
            value["covariance_inverse"], dtype=torch.float32
        ),
        mahalanobis_squared_max=float(value["mahalanobis_squared_p99"]),
        support_actions=torch.as_tensor(value["support_actions"], dtype=torch.float32),
        knn_k=int(value["knn_k"]),
        knn_distance_max=float(value["knn_distance_p99"]),
        roughness_max=float(value["roughness_p99"]),
        jerk_max=float(value["jerk_p99"]),
        curvature_max=float(value["curvature_p99"]),
        parent_trust_radius=float(value["parent_trust_radius"]),
    )


def prediction_binding(
    pool: dict[str, Any],
    latent_path: Path,
    pairs: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema": "recon100_dinov3_scorer_predictions_v1",
        "split": pool["split"],
        "states": len(pool["records"]),
        "candidates": K,
        "candidate_sha256": pool["candidate_sha256"],
        "manifest_sha256": pool["manifest_sha256"],
        "latent_cache_sha256": file_sha256(latent_path),
        "seeds": [
            {
                "seed": pair["seed"],
                "wm_sha256": pair["wm_sha256"],
                "scorer_sha256": pair["scorer_sha256"],
            }
            for pair in pairs
        ],
        "fixed_pool_only": True,
        "cem": False,
        "labels_opened_during_inference": False,
    }


def predict_pool(
    pool: dict[str, Any],
    latents: dict[str, np.ndarray],
    *,
    latent_path: Path,
    pairs: list[dict[str, Any]],
    output: Path,
    device: torch.device,
    batch_size: int,
    candidate_chunk: int,
) -> dict[str, np.ndarray]:
    binding = prediction_binding(pool, latent_path, pairs)
    if output.is_file():
        arrays = validate_cached_binding(output, binding)
        expected = (len(pairs), len(pool["records"]), K)
        if arrays.get("ate_by_seed", np.empty(0)).shape != expected:
            raise RuntimeError("cached scorer prediction shape mismatch")
        if arrays.get("rtp_by_seed", np.empty(0)).shape != expected:
            raise RuntimeError("cached scorer prediction shape mismatch")
        return arrays

    actions_cpu = torch.from_numpy(pool["actions"].astype(np.float32))
    support_by_seed = []
    ate_by_seed = []
    rtp_by_seed = []
    loss_scales = []
    started = time.perf_counter()
    for pair_index, pair in enumerate(pairs):
        wm_checkpoint = torch.load(pair["wm"], map_location="cpu", weights_only=False)
        scorer_checkpoint = torch.load(
            pair["scorer"], map_location="cpu", weights_only=False
        )
        if scorer_checkpoint.get("world_model_checkpoint_sha256") != pair["wm_sha256"]:
            raise RuntimeError(f"seed{pair['seed']} scorer/WM provenance mismatch")
        if wm_checkpoint.get("temporal_protocol") != {
            "history_frames": 4,
            "future_frames": 4,
        }:
            raise RuntimeError("world-model H4 protocol mismatch")
        wm = LiteTokenFormerWMV2(
            LiteTokenFormerConfig(**wm_checkpoint["config"])
        ).to(device)
        wm.load_state_dict(wm_checkpoint["model"], strict=True)
        wm.eval().requires_grad_(False)
        scorer = TrajectoryQualityScorerV2(
            TrajectoryQualityScorerConfig(**scorer_checkpoint["model_config"])
        ).to(device)
        scorer.load_state_dict(scorer_checkpoint["model"], strict=True)
        scorer.eval().requires_grad_(False)
        loss_config = scorer_checkpoint["loss_config"]
        loss_scales.append((float(loss_config["ate_scale"]), float(loss_config["rtp_scale"])))

        support = evaluate_trajectory_constraints(
            actions_cpu,
            support_stats(scorer_checkpoint["action_support"]),
            parent_actions=actions_cpu,
        ).valid.numpy()
        support[:, INCUMBENT] = True
        support_by_seed.append(support)
        seed_ate = []
        seed_rtp = []
        for start in range(0, len(pool["records"]), batch_size):
            end = min(start + batch_size, len(pool["records"]))
            history = torch.from_numpy(latents["history"][start:end].astype(np.float32)).to(device)
            goal = torch.from_numpy(latents["goal"][start:end].astype(np.float32)).to(device)
            actions = actions_cpu[start:end].to(device)
            ate_chunks = []
            rtp_chunks = []
            for candidate_start in range(0, K, candidate_chunk):
                candidate_end = min(candidate_start + candidate_chunk, K)
                chunk = actions[:, candidate_start:candidate_end]
                context = (
                    torch.autocast("cuda", dtype=torch.bfloat16)
                    if device.type == "cuda"
                    else torch.autocast("cpu", enabled=False)
                )
                with torch.inference_mode(), context:
                    future = wm(history, chunk)["predicted_tokens"]
                with torch.inference_mode():
                    prediction = scorer(
                        current_latent=history[:, -1],
                        goal_latent=goal,
                        candidate_actions=chunk,
                        predicted_future=future.float(),
                    )
                ate_chunks.append(prediction.ate.cpu())
                rtp_chunks.append(prediction.rtp.cpu())
            seed_ate.append(torch.cat(ate_chunks, dim=1).numpy().astype(np.float32))
            seed_rtp.append(torch.cat(rtp_chunks, dim=1).numpy().astype(np.float32))
            print(
                json.dumps(
                    {
                        "stage": "dinov3_ltwm_scorer_inference",
                        "split": pool["split"],
                        "seed": pair["seed"],
                        "completed_states": end,
                        "total_states": len(pool["records"]),
                        "elapsed_sec": time.perf_counter() - started,
                    }
                ),
                flush=True,
            )
        ate_by_seed.append(np.concatenate(seed_ate))
        rtp_by_seed.append(np.concatenate(seed_rtp))
        del wm, scorer, wm_checkpoint, scorer_checkpoint
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if not np.allclose(loss_scales, loss_scales[0], atol=0.0, rtol=0.0):
        raise RuntimeError("scorer seeds have different metric scales")
    support_array = np.stack(support_by_seed)
    # A fixed NoMaD candidate is admitted when at least two independently
    # fitted support estimators accept it.  This avoids a single seed's random
    # KNN support subsample vetoing an otherwise in-distribution trajectory.
    majority_support = support_array.sum(axis=0) >= math.ceil(len(pairs) / 2)
    majority_support[:, INCUMBENT] = True
    arrays = {
        "ate_by_seed": np.stack(ate_by_seed).astype(np.float32),
        "rtp_by_seed": np.stack(rtp_by_seed).astype(np.float32),
        "support_by_seed": support_array,
        "majority_support": majority_support,
        "ate_scale": np.asarray(loss_scales[0][0], dtype=np.float64),
        "rtp_scale": np.asarray(loss_scales[0][1], dtype=np.float64),
        "metadata": np.asarray(canonical_json(binding)),
    }
    write_npz_atomic(output, **arrays)
    return arrays


def current_outcomes(labels: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    candidates = labels["candidates_local_m"].astype(np.float64)
    ground_truth = labels["ground_truth_local_m"].astype(np.float64)
    error = np.linalg.norm(candidates - ground_truth[:, None], axis=-1)
    mean_ate = error.mean(axis=-1)
    start_distance = np.linalg.norm(ground_truth[:, -1], axis=-1)
    final_distance = np.linalg.norm(
        candidates[:, :, -1] - ground_truth[:, None, -1], axis=-1
    )
    rtp = (start_distance[:, None] - final_distance) / np.maximum(
        start_distance[:, None], 1.0e-8
    )
    return mean_ate.astype(np.float32), rtp.astype(np.float32)


def validate_labels(pool: dict[str, Any], labels: dict[str, np.ndarray]) -> None:
    expected = (len(pool["records"]), K)
    if labels.get("candidate_ate_m", np.empty(0)).shape != expected:
        raise RuntimeError("legacy label ATE shape mismatch")
    if labels.get("candidates_waypoint_units", np.empty(0)).shape != (
        len(pool["records"]), K, HORIZON, 2
    ):
        raise RuntimeError("legacy label candidate shape mismatch")
    if not np.array_equal(
        labels["candidates_waypoint_units"].astype(np.float32),
        pool["waypoints"],
    ):
        raise RuntimeError("legacy labels are not bound to this candidate pool")
    expected_keys = np.asarray(
        [f"{row['traj_name']}:{int(row['curr_time'])}" for row in pool["records"]]
    )
    if not np.array_equal(labels["record_key"].astype(str), expected_keys):
        raise RuntimeError("legacy label record order mismatch")


def load_labels_after_inference(pool: dict[str, Any]) -> dict[str, np.ndarray]:
    labels = load_npz(pool["paths"]["labels"])
    validate_labels(pool, labels)
    return labels


def legacy_selection_metrics(
    selected_index: np.ndarray,
    labels: dict[str, np.ndarray],
) -> dict[str, Any]:
    selected = np.asarray(selected_index, dtype=np.int64)
    row = np.arange(len(selected))
    promoted = selected != INCUMBENT
    output: dict[str, Any] = {
        "states": int(len(selected)),
        "promotions": int(promoted.sum()),
        "promotion_fraction": float(promoted.mean()),
    }
    success_masks = {}
    for name, key in (
        ("ate", "candidate_ate_m"),
        ("endpoint", "candidate_endpoint_m"),
        ("objective", "candidate_objective_m"),
    ):
        values = labels[key].astype(np.float64)
        delta = values[row, selected] - values[:, INCUMBENT]
        success = promoted & (delta < 0.0)
        success_masks[name] = success
        output[f"selected_{name}_mean_m"] = float(values[row, selected].mean())
        output[f"incumbent_{name}_mean_m"] = float(values[:, INCUMBENT].mean())
        output[f"net_{name}_delta_per_state_m"] = float(delta.mean())
        output[f"{name}_precision"] = (
            None if not promoted.any() else float(success.sum() / promoted.sum())
        )
        output[f"{name}_delta_by_state_m"] = delta.tolist()
    dual = success_masks["ate"] & success_masks["objective"]
    promotions = int(promoted.sum())
    dual_count = int(dual.sum())
    wilson = wilson_interval(dual_count, promotions)
    output["dual_precision"] = (
        None if promotions == 0 else float(dual_count / promotions)
    )
    output["dual_precision_wilson95"] = [float(wilson[0]), float(wilson[1])]
    return output


def paired_bootstrap(
    left: np.ndarray,
    right: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    left_value = np.asarray(left, dtype=np.float64)
    right_value = np.asarray(right, dtype=np.float64)
    if left_value.shape != right_value.shape or left_value.ndim != 1:
        raise ValueError("paired bootstrap inputs must be matching vectors")
    difference = left_value - right_value
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(difference), size=(samples, len(difference)))
    draws = difference[indices].mean(axis=1)
    return {
        "mean": float(difference.mean()),
        "ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
        "p_left_not_better_for_lower_is_better": float(np.mean(draws >= 0.0)),
        "samples": int(samples),
    }


def freeze_policy(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    pool = load_pool(args.legacy_root, "cal32")
    pairs = checkpoint_pairs(args.checkpoint_root)
    latent_path = args.output_dir / "cal32_latents.npz"
    prediction_path = args.output_dir / "cal32_predictions.npz"
    latents = encode_targeted_latents(
        pool,
        trajectory_root=args.trajectory_root,
        model_dir=args.model_dir,
        adapter_checkpoint=args.adapter_checkpoint,
        output=latent_path,
        device=device,
        batch_size=args.dinov3_batch_size,
    )
    predictions = predict_pool(
        pool,
        latents,
        latent_path=latent_path,
        pairs=pairs,
        output=prediction_path,
        device=device,
        batch_size=args.wm_batch_size,
        candidate_chunk=args.candidate_chunk,
    )
    # Calibration outcomes are opened only after label-free predictions exist.
    labels = load_labels_after_inference(pool)
    actual_ate, actual_rtp = current_outcomes(labels)
    safe_policy = calibrate_safe_promotion(
        predictions["ate_by_seed"],
        predictions["rtp_by_seed"],
        actual_ate,
        actual_rtp,
        ate_scale=float(predictions["ate_scale"].item()),
        rtp_scale=float(predictions["rtp_scale"].item()),
        uncertainty_beta=args.uncertainty_beta,
        alpha=args.alpha,
        incumbent_index=INCUMBENT,
        valid=predictions["majority_support"],
    )
    cal_selection = apply_safe_promotion(
        predictions["ate_by_seed"],
        predictions["rtp_by_seed"],
        safe_policy,
        valid=predictions["majority_support"],
    )
    strict_paths = pool_paths(args.legacy_root, "strict_external100")
    policy_payload = {
        "schema": "recon100_dinov3_019f_safe_policy_v1",
        "created_at": utc_now(),
        "status": "frozen_before_recon100_inference",
        "method": "019f70d5_relative_paired_ensemble_conformal_promotion_adapted_to_dinov3",
        "safe_policy": safe_policy.to_dict(),
        "calibration": {
            "split": "cal32",
            "candidate_sha256": pool["candidate_sha256"],
            "manifest_sha256": pool["manifest_sha256"],
            "labels_sha256": file_sha256(pool["paths"]["labels"]),
            "prediction_sha256": file_sha256(prediction_path),
            "support_acceptance_fraction": float(predictions["majority_support"].mean()),
            "metrics_current_contract": selection_metrics(
                cal_selection.selected_index,
                actual_ate,
                actual_rtp,
            ),
            "metrics_legacy_contract": legacy_selection_metrics(
                cal_selection.selected_index,
                labels,
            ),
        },
        "strict_external100": {
            "candidate_sha256": file_sha256(strict_paths["candidates"]),
            "manifest_sha256": file_sha256(strict_paths["manifest"]),
            "outcomes_read_before_freeze": False,
            "inference_pending": True,
        },
        "weights": {
            "old_scorer_weights_migrated": False,
            "dinov3_adapter_sha256": file_sha256(args.adapter_checkpoint),
            "dinov3_weight_sha256": file_sha256(args.model_dir / "model.safetensors"),
            "seeds": [
                {
                    "seed": pair["seed"],
                    "wm_sha256": pair["wm_sha256"],
                    "scorer_sha256": pair["scorer_sha256"],
                }
                for pair in pairs
            ],
        },
        "candidate_contract": {
            "incumbent_index": 0,
            "candidates": 16,
            "horizon": 4,
            "legacy_waypoints_to_current_actions": "lossless_path_heading_se2",
        },
        "fixed_pool_only": True,
        "cem": False,
        "outer_dinov3_val_accessed": False,
        "outer_dinov3_test_accessed": False,
    }
    policy_path = args.output_dir / "frozen_policy.json"
    if policy_path.exists():
        existing = json.loads(policy_path.read_text(encoding="utf-8"))
        comparable_existing = dict(existing)
        comparable_new = dict(policy_payload)
        comparable_existing.pop("created_at", None)
        comparable_new.pop("created_at", None)
        if comparable_existing != comparable_new:
            raise RuntimeError("existing frozen policy differs; refusing to overwrite")
        return existing
    write_json_atomic(policy_path, policy_payload)
    return policy_payload


def load_frozen_policy(args: argparse.Namespace) -> tuple[dict[str, Any], SafePromotionPolicy]:
    path = args.output_dir / "frozen_policy.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "recon100_dinov3_019f_safe_policy_v1":
        raise RuntimeError("frozen policy schema mismatch")
    if payload.get("status") != "frozen_before_recon100_inference":
        raise RuntimeError("policy was not frozen before RECON100 inference")
    if payload.get("cem") is not False or payload.get("fixed_pool_only") is not True:
        raise RuntimeError("policy is not fixed-pool/CEM-off")
    return payload, SafePromotionPolicy.from_dict(payload["safe_policy"])


def verify_old_replay_metrics(
    metrics: dict[str, Any],
    legacy_audit: dict[str, Any],
) -> None:
    expected = legacy_audit["selectors"]["full"]
    checks = {
        "promotions": expected["promotions"],
        "ate_precision": expected["ate_precision"],
        "objective_precision": expected["objective_precision"],
        "net_ate_delta_per_state_m": expected["mean_ate_delta_m"],
        "net_objective_delta_per_state_m": expected["mean_objective_delta_m"],
    }
    for key, value in checks.items():
        actual = metrics[key]
        if isinstance(value, int):
            if actual != value:
                raise RuntimeError(f"old replay {key} mismatch")
        elif not math.isclose(float(actual), float(value), abs_tol=1.0e-8, rel_tol=1.0e-7):
            raise RuntimeError(f"old replay {key} mismatch: {actual} != {value}")


def test_policy(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    frozen_payload, safe_policy = load_frozen_policy(args)
    pool = load_pool(args.legacy_root, "strict_external100")
    frozen_external = frozen_payload["strict_external100"]
    if pool["candidate_sha256"] != frozen_external["candidate_sha256"]:
        raise RuntimeError("RECON100 candidate pool differs from frozen policy")
    if pool["manifest_sha256"] != frozen_external["manifest_sha256"]:
        raise RuntimeError("RECON100 manifest differs from frozen policy")
    pairs = checkpoint_pairs(args.checkpoint_root)
    latent_path = args.output_dir / "strict_external100_latents.npz"
    prediction_path = args.output_dir / "strict_external100_predictions.npz"
    latents = encode_targeted_latents(
        pool,
        trajectory_root=args.trajectory_root,
        model_dir=args.model_dir,
        adapter_checkpoint=args.adapter_checkpoint,
        output=latent_path,
        device=device,
        batch_size=args.dinov3_batch_size,
    )
    predictions = predict_pool(
        pool,
        latents,
        latent_path=latent_path,
        pairs=pairs,
        output=prediction_path,
        device=device,
        batch_size=args.wm_batch_size,
        candidate_chunk=args.candidate_chunk,
    )

    # Test outcomes and the performance-bearing old audit are opened only now,
    # after the policy and all DINOv3 predictions are frozen on disk.
    labels = load_labels_after_inference(pool)
    actual_ate, actual_rtp = current_outcomes(labels)
    replay_path = (
        args.legacy_root
        / "recon_h4_scorer_bundle_v1/strict_external_frozen_selection_replay_v2.json"
    )
    audit_path = args.legacy_root / "recon_h4_scorer_bundle_v1/strict_external_audit_v1.json"
    replay = json.loads(replay_path.read_text(encoding="utf-8"))
    legacy_audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if replay.get("labels_read") is not False or replay.get("cem") is not False:
        raise RuntimeError("old frozen replay is not label-free/CEM-off")

    old_full = np.asarray(replay["selected_index"]["full"], dtype=np.int64)
    current_seed1701 = greedy_fixed_pool_selection(
        predictions["ate_by_seed"][:1],
        predictions["rtp_by_seed"][:1],
        ate_scale=float(predictions["ate_scale"].item()),
        rtp_scale=float(predictions["rtp_scale"].item()),
    )
    current_ensemble = greedy_fixed_pool_selection(
        predictions["ate_by_seed"],
        predictions["rtp_by_seed"],
        ate_scale=float(predictions["ate_scale"].item()),
        rtp_scale=float(predictions["rtp_scale"].item()),
    )
    safe = apply_safe_promotion(
        predictions["ate_by_seed"],
        predictions["rtp_by_seed"],
        safe_policy,
        valid=predictions["majority_support"],
    )
    selections = {
        "019f70d5_old_full_frozen": old_full,
        "dinov3_current_seed1701_greedy": current_seed1701,
        "dinov3_current_ensemble_greedy": current_ensemble,
        "dinov3_019f70d5_adapted_safe": safe.selected_index,
    }
    arms = {}
    for name, selected in selections.items():
        arms[name] = {
            "current_mean_ate_rtp_contract": selection_metrics(
                selected,
                actual_ate,
                actual_rtp,
            ),
            "legacy_rms_ate_endpoint_objective_contract": legacy_selection_metrics(
                selected,
                labels,
            ),
            "selected_index": selected.tolist(),
        }
    verify_old_replay_metrics(
        arms["019f70d5_old_full_frozen"][
            "legacy_rms_ate_endpoint_objective_contract"
        ],
        legacy_audit,
    )

    old_delta = np.asarray(
        arms["019f70d5_old_full_frozen"][
            "legacy_rms_ate_endpoint_objective_contract"
        ]["ate_delta_by_state_m"]
    )
    current_delta = np.asarray(
        arms["dinov3_current_ensemble_greedy"][
            "legacy_rms_ate_endpoint_objective_contract"
        ]["ate_delta_by_state_m"]
    )
    safe_delta = np.asarray(
        arms["dinov3_019f70d5_adapted_safe"][
            "legacy_rms_ate_endpoint_objective_contract"
        ]["ate_delta_by_state_m"]
    )
    comparison = {
        "schema": "recon100_dinov3_safe_scorer_comparison_v1",
        "created_at": utc_now(),
        "status": "complete_recon_dev_evidence_only",
        "candidate_pool": "strict_external100",
        "states": 100,
        "candidates_per_state": 16,
        "horizon": 4,
        "arms": arms,
        "paired_bootstrap_legacy_ate": {
            "dinov3_ensemble_minus_old_full": paired_bootstrap(
                current_delta,
                old_delta,
                samples=args.bootstrap_samples,
                seed=1701,
            ),
            "dinov3_adapted_safe_minus_old_full": paired_bootstrap(
                safe_delta,
                old_delta,
                samples=args.bootstrap_samples,
                seed=1702,
            ),
            "dinov3_adapted_safe_minus_dinov3_ensemble": paired_bootstrap(
                safe_delta,
                current_delta,
                samples=args.bootstrap_samples,
                seed=1703,
            ),
        },
        "adapted_gate": {
            "policy": safe_policy.to_dict(),
            "challenger_index": safe.challenger_index.tolist(),
            "promoted": safe.promoted.tolist(),
            "ate_gate_margin": safe.ate_gate_margin.tolist(),
            "rtp_gate_margin": safe.rtp_gate_margin.tolist(),
            "support_acceptance_fraction": float(predictions["majority_support"].mean()),
        },
        "protocol": {
            "old_weights_migrated": False,
            "old_method_and_split_protocol_migrated": True,
            "calibration_split": "cal32",
            "test_split": "strict_external100",
            "same_fixed_candidate_pool_for_all_arms": True,
            "test_labels_opened_after_policy_and_predictions_frozen": True,
            "metric_contracts_reported_separately": True,
            "outer_dinov3_val_accessed": False,
            "outer_dinov3_test_accessed": False,
            "recon_dev_only": True,
        },
        "cem": {
            "enabled": False,
            "iterations": 0,
            "generated_candidates": 0,
            "all_arms_fixed_pool_only": True,
        },
        "provenance": {
            "frozen_policy_sha256": file_sha256(args.output_dir / "frozen_policy.json"),
            "prediction_sha256": file_sha256(prediction_path),
            "candidate_sha256": pool["candidate_sha256"],
            "labels_sha256": file_sha256(pool["paths"]["labels"]),
            "old_replay_sha256": file_sha256(replay_path),
            "old_audit_sha256": file_sha256(audit_path),
        },
    }
    write_json_atomic(args.output_dir / "comparison.json", comparison)
    write_npz_atomic(
        args.output_dir / "strict_external100_selections.npz",
        **{name: selected.astype(np.int64) for name, selected in selections.items()},
        metadata=np.asarray(
            canonical_json(
                {
                    "schema": comparison["schema"],
                    "candidate_sha256": pool["candidate_sha256"],
                    "cem": False,
                }
            )
        ),
    )
    return comparison


def main() -> None:
    args = parse_args()
    if args.dinov3_batch_size <= 0 or args.wm_batch_size <= 0 or args.candidate_chunk <= 0:
        raise ValueError("batch and chunk sizes must be positive")
    if args.bootstrap_samples <= 0:
        raise ValueError("bootstrap-samples must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if args.stage in {"freeze", "all"}:
        policy = freeze_policy(args, device)
        print(json.dumps(policy, indent=2, sort_keys=True), flush=True)
    if args.stage in {"test", "all"}:
        comparison = test_policy(args, device)
        print(json.dumps(comparison, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
