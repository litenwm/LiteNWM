#!/usr/bin/env python3
"""Evaluate LTWM checkpoints on sealed Recon/SCAND/SACSoN H4-100 sets."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
import time
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from recon_latents.actions import derive_continuous_actions  # noqa: E402
from recon_latents.dinov3_adapter import (  # noqa: E402
    DinoV3AdapterConfig,
    DinoV3TokenAdapter,
    extract_dinov3_patch_tokens,
)
from recon_latents.models_v2 import (  # noqa: E402
    LiteTokenFormerConfig,
    LiteTokenFormerWMV2,
)


DOMAINS = ("recon", "scand", "sacson")
SEEDS = (1701, 1702, 1703)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path(
            "/root/autodl-tmp/dwm_runs/"
            "v70_tridomain_strict_full_window_bs56"
        ),
    )
    parser.add_argument(
        "--previous-run-root",
        type=Path,
        default=Path(
            "/root/autodl-tmp/dwm_runs/v70_tridomain_new_weights"
        ),
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/root/autodl-tmp/dwm_datasets"),
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path(
            "/root/autodl-tmp/dwm_datasets/deps/"
            "dinov3-vitl16-pretrain-lvd1689m"
        ),
    )
    parser.add_argument(
        "--adapter-checkpoint",
        type=Path,
        default=REPO_ROOT
        / "checkpoints/project/dinov3_adapter/adapter_pilot_v1_final.pt",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dino-batch-size", type=int, default=12)
    parser.add_argument("--wm-batch-size", type=int, default=4)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument(
        "--amp",
        choices=("bf16", "fp16", "off"),
        default="bf16",
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def load_pickle(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        try:
            value = pickle.load(handle)
        except ModuleNotFoundError as error:
            if not str(error.name).startswith("numpy._core"):
                raise

            class NumpyCompatUnpickler(pickle.Unpickler):
                def find_class(self, module: str, name: str) -> Any:
                    return super().find_class(
                        module.replace("numpy._core", "numpy.core"),
                        name,
                    )

            handle.seek(0)
            value = NumpyCompatUnpickler(handle).load()
    if not isinstance(value, dict) or not {"position", "yaw"}.issubset(value):
        raise ValueError(f"invalid trajectory pose file: {path}")
    return value


def preprocess(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        image = image.convert("RGB").resize(
            (224, 224),
            Image.Resampling.BICUBIC,
        )
    value = np.asarray(image, dtype=np.float32) / 255.0
    value = (
        value - np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
    ) / np.asarray((0.229, 0.224, 0.225), dtype=np.float32)
    return torch.from_numpy(
        np.ascontiguousarray(value.transpose(2, 0, 1))
    )


def manifest_and_root(
    args: argparse.Namespace,
    domain: str,
) -> tuple[Path, Path]:
    if domain == "recon":
        return (
            REPO_ROOT
            / "data_splits/recon/server_full_nwm_unseen_h4_100/"
            "manifest.json",
            args.data_root / "raw/recon_full_v71",
        )
    if domain == "scand":
        return (
            REPO_ROOT
            / "data_splits/scand/project_strict_test_h4_100_seed17/"
            "manifest.json",
            args.data_root / "raw/scand_full_v71",
        )
    return (
        args.data_root
        / "evaluation/sacson/strict_test_h4_100_seed17/manifest.json",
        args.data_root / "processed/sacson_processed",
    )


def training_latent_root(args: argparse.Namespace, domain: str) -> Path:
    return (
        args.data_root
        / "processed_v70_tridomain"
        / f"{domain}_dinov3_ltwm_h4"
    )


def validate_manifest(
    args: argparse.Namespace,
    domain: str,
    manifest_path: Path,
    manifest: dict[str, Any],
    split_lock: dict[str, Any],
) -> dict[str, Any]:
    records = manifest["records"]
    if len(records) != 100:
        raise RuntimeError(f"{domain} manifest is not H4-100")
    if int(manifest["context_frames"]) != 4 or int(manifest["horizon"]) != 4:
        raise RuntimeError(f"{domain} temporal contract is not 4ctx/H4")
    ids = {str(record["trajectory_id"]) for record in records}
    test_ids = set(map(str, split_lock[domain]["splits"]["test"]))
    outside_test = sorted(ids - test_ids)
    if outside_test:
        raise RuntimeError(
            f"{domain} manifest contains non-test trajectories: "
            f"{outside_test[:10]}"
        )
    latent_root = training_latent_root(args, domain)
    latent_splits = json.loads(
        (latent_root / "splits.json").read_text(encoding="utf-8")
    )
    exposed = {
        str(item)
        for split in ("train", "val")
        for item in latent_splits.get(split, ())
    }
    overlap = sorted(ids & exposed)
    if overlap:
        raise RuntimeError(
            f"{domain} strict test overlaps world-model training view: "
            f"{overlap[:10]}"
        )
    if latent_splits.get("test"):
        raise RuntimeError(
            f"{domain} world-model latent root exposes test trajectories"
        )
    return {
        "status": "PASS",
        "manifest": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "records": len(records),
        "unique_trajectories": len(ids),
        "all_records_in_active_outer_test": True,
        "training_view_overlap": 0,
        "training_latent_test_ids": 0,
    }


def image_index(trajectory_root: Path) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for path in trajectory_root.glob("*.jpg"):
        try:
            result[int(path.stem)] = path
        except ValueError:
            continue
    if not result:
        raise FileNotFoundError(f"no numeric JPEG frames: {trajectory_root}")
    return result


def prepare_domain_inputs(
    args: argparse.Namespace,
    domain: str,
    backbone: torch.nn.Module,
    adapter: DinoV3TokenAdapter,
    device: torch.device,
    model_sha: str,
    adapter_sha: str,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    manifest_path, trajectory_root = manifest_and_root(args, domain)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_sha = sha256(manifest_path)
    cache_path = args.output_dir / "encoded_inputs" / f"{domain}_h4_100.pt"
    binding = {
        "schema": "ltwm_strict_h4_100_encoded_inputs_v1",
        "domain": domain,
        "manifest_sha256": manifest_sha,
        "dinov3_model_sha256": model_sha,
        "adapter_checkpoint_sha256": adapter_sha,
        "image_preprocess": "RGB_BICUBIC_224_IMAGENET_NORMALIZATION",
        "action_contract": "metric_local_SE2_dx_dy_dyaw",
    }
    if cache_path.exists():
        payload = torch.load(
            cache_path,
            map_location="cpu",
            weights_only=False,
        )
        if payload.get("binding") != binding:
            raise RuntimeError(f"encoded input cache binding mismatch: {cache_path}")
        return payload["inputs"], {
            **binding,
            "cache": str(cache_path),
            "cache_status": "reused",
        }

    path_for_key: dict[tuple[str, int], Path] = {}
    actions_by_record: list[np.ndarray] = []
    record_uids: list[str] = []
    source_resolutions: set[tuple[int, int]] = set()
    trajectory_cache: dict[
        str,
        tuple[dict[int, Path], np.ndarray, np.ndarray],
    ] = {}
    for record in manifest["records"]:
        trajectory_id = str(record["trajectory_id"])
        if trajectory_id not in trajectory_cache:
            current_root = trajectory_root / trajectory_id
            images = image_index(current_root)
            pose = load_pickle(current_root / "traj_data.pkl")
            position = np.asarray(pose["position"], dtype=np.float32)
            yaw = np.asarray(pose["yaw"], dtype=np.float32).reshape(-1)
            if position.ndim != 2 or position.shape[1] != 2:
                raise ValueError(f"invalid position shape: {trajectory_id}")
            if len(position) != len(yaw):
                raise ValueError(f"pose/yaw length mismatch: {trajectory_id}")
            trajectory_cache[trajectory_id] = (images, position, yaw)
        images, position, yaw = trajectory_cache[trajectory_id]
        context_times = list(map(int, record["context_times"]))
        target_times = list(map(int, record["gt_times"]))
        curr_time = int(record["curr_time"])
        if (
            context_times != list(range(curr_time - 3, curr_time + 1))
            or target_times != list(range(curr_time + 1, curr_time + 5))
        ):
            raise RuntimeError(
                f"non-contiguous 4ctx/H4 record: {domain}/{trajectory_id}"
            )
        for timestep in context_times + target_times:
            if timestep not in images:
                raise FileNotFoundError(
                    f"missing frame {timestep}: {domain}/{trajectory_id}"
                )
            path_for_key[(trajectory_id, timestep)] = images[timestep]
        continuous = derive_continuous_actions(position, yaw)
        actions = continuous[curr_time : curr_time + 4]
        if actions.shape != (4, 3):
            raise RuntimeError(
                f"invalid H4 actions: {domain}/{trajectory_id}/{curr_time}"
            )
        actions_by_record.append(actions)
        record_uids.append(f"{domain}|{trajectory_id}|{curr_time}|4|4")

    ordered_keys = sorted(path_for_key)
    tokens_by_key: dict[tuple[str, int], torch.Tensor] = {}
    for start in range(0, len(ordered_keys), args.dino_batch_size):
        keys = ordered_keys[start : start + args.dino_batch_size]
        images = []
        for key in keys:
            path = path_for_key[key]
            with Image.open(path) as image:
                source_resolutions.add(tuple(map(int, image.size)))
            images.append(preprocess(path))
        image_batch = torch.stack(images).to(device)
        with torch.inference_mode():
            hidden = backbone(pixel_values=image_batch).last_hidden_state
            adapted = adapter(extract_dinov3_patch_tokens(hidden))
        encoded = adapted.reshape(len(keys), 256, 384).to(
            dtype=torch.float16,
            device="cpu",
        )
        for key, value in zip(keys, encoded, strict=True):
            tokens_by_key[key] = value.clone()
        print(
            json.dumps(
                {
                    "stage": "encode_strict_h4_100",
                    "domain": domain,
                    "completed_unique_images": min(
                        start + len(keys),
                        len(ordered_keys),
                    ),
                    "total_unique_images": len(ordered_keys),
                }
            ),
            flush=True,
        )

    history_rows = []
    target_rows = []
    for record in manifest["records"]:
        trajectory_id = str(record["trajectory_id"])
        history_rows.append(
            torch.stack(
                [
                    tokens_by_key[(trajectory_id, int(timestep))]
                    for timestep in record["context_times"]
                ]
            )
        )
        target_rows.append(
            torch.stack(
                [
                    tokens_by_key[(trajectory_id, int(timestep))]
                    for timestep in record["gt_times"]
                ]
            )
        )
    inputs = {
        "history": torch.stack(history_rows),
        "target": torch.stack(target_rows),
        "actions": torch.from_numpy(
            np.stack(actions_by_record).astype(np.float32)
        ),
        "record_uids": record_uids,
    }
    payload = {"binding": binding, "inputs": inputs}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(".pt.tmp")
    torch.save(payload, temporary)
    temporary.replace(cache_path)
    return inputs, {
        **binding,
        "cache": str(cache_path),
        "cache_status": "created",
        "unique_images": len(ordered_keys),
        "source_resolutions": [
            list(value) for value in sorted(source_resolutions)
        ],
    }


def amp_context(args: argparse.Namespace):
    if args.amp == "off":
        return torch.autocast(device_type="cuda", enabled=False)
    dtype = torch.bfloat16 if args.amp == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def sample_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    history: torch.Tensor,
) -> dict[str, np.ndarray]:
    prediction = prediction.float()
    target = target.float()
    current = history[:, -1].float()
    error = (prediction - target).square().mean(dim=(-1, -2))
    denominator = target.square().mean(dim=(-1, -2)).clamp_min(1e-8)
    nmse = error / denominator
    cosine = 1.0 - torch.nn.functional.cosine_similarity(
        prediction.flatten(-2),
        target.flatten(-2),
        dim=-1,
    )
    prediction_delta = prediction - current[:, None]
    target_delta = target - current[:, None]
    norm_ratio = torch.linalg.vector_norm(
        prediction_delta.flatten(-2),
        dim=-1,
    ) / torch.linalg.vector_norm(
        target_delta.flatten(-2),
        dim=-1,
    ).clamp_min(1e-8)
    return {
        "nmse": nmse.cpu().numpy(),
        "cosine": cosine.cpu().numpy(),
        "norm_ratio": norm_ratio.cpu().numpy(),
    }


def summarize(metric_rows: dict[str, np.ndarray]) -> dict[str, Any]:
    return {
        name: {
            "mean_by_horizon": value.mean(axis=0).tolist(),
            "std_by_horizon": value.std(axis=0).tolist(),
        }
        for name, value in metric_rows.items()
    }


def evaluate_checkpoint(
    args: argparse.Namespace,
    checkpoint_path: Path,
    inputs_by_domain: dict[str, dict[str, torch.Tensor]],
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, dict[str, np.ndarray]]]:
    payload = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    config_payload = dict(payload.get("config", {}))
    config_fields = {item.name for item in fields(LiteTokenFormerConfig)}
    config = LiteTokenFormerConfig(
        **{
            key: value
            for key, value in config_payload.items()
            if key in config_fields
        }
    )
    model = LiteTokenFormerWMV2(config)
    load_result = model.load_state_dict(
        payload.get("model", payload),
        strict=True,
    )
    if load_result.missing_keys or load_result.unexpected_keys:
        raise RuntimeError(
            f"checkpoint state mismatch: {checkpoint_path}: {load_result}"
        )
    model = model.eval().requires_grad_(False).to(device)
    report: dict[str, Any] = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "checkpoint_split_lock_sha256": payload.get("split_lock_sha256"),
        "config": config_payload,
        "domains": {},
    }
    raw_metrics: dict[str, dict[str, np.ndarray]] = {}
    for domain_index, domain in enumerate(DOMAINS):
        inputs = inputs_by_domain[domain]
        history_cpu = inputs["history"]
        target_cpu = inputs["target"]
        actions_cpu = inputs["actions"]
        generator = torch.Generator(device="cpu").manual_seed(
            730000 + domain_index
        )
        noise_scale = actions_cpu.std(dim=(0, 1), keepdim=True).clamp_min(
            1e-4
        )
        noisy_cpu = actions_cpu + 0.5 * noise_scale * torch.randn(
            actions_cpu.shape,
            generator=generator,
            dtype=actions_cpu.dtype,
        )
        true_parts: dict[str, list[np.ndarray]] = {
            "nmse": [],
            "cosine": [],
            "norm_ratio": [],
        }
        reverse_nmse: list[np.ndarray] = []
        noisy_nmse: list[np.ndarray] = []
        finite = 0
        started = time.perf_counter()
        for start in range(0, len(history_cpu), args.wm_batch_size):
            stop = min(start + args.wm_batch_size, len(history_cpu))
            history = history_cpu[start:stop].to(
                device=device,
                dtype=torch.float32,
            )
            target = target_cpu[start:stop].to(
                device=device,
                dtype=torch.float32,
            )
            actions = actions_cpu[start:stop].to(device)
            reverse_actions = actions.flip(1)
            noisy_actions = noisy_cpu[start:stop].to(device)
            with torch.inference_mode(), amp_context(args):
                true_prediction = model(
                    history,
                    actions,
                )["predicted_tokens"][:, 0]
                reverse_prediction = model(
                    history,
                    reverse_actions,
                )["predicted_tokens"][:, 0]
                noisy_prediction = model(
                    history,
                    noisy_actions,
                )["predicted_tokens"][:, 0]
            finite += int(
                torch.isfinite(true_prediction).all(dim=(-1, -2, -3)).sum()
            )
            current = sample_metrics(true_prediction, target, history)
            reverse = sample_metrics(reverse_prediction, target, history)
            noisy = sample_metrics(noisy_prediction, target, history)
            for name in true_parts:
                true_parts[name].append(current[name])
            reverse_nmse.append(reverse["nmse"])
            noisy_nmse.append(noisy["nmse"])
        metrics = {
            name: np.concatenate(parts, axis=0)
            for name, parts in true_parts.items()
        }
        reverse_value = np.concatenate(reverse_nmse, axis=0)
        noisy_value = np.concatenate(noisy_nmse, axis=0)
        raw_metrics[domain] = metrics
        report["domains"][domain] = {
            "samples": len(history_cpu),
            "finite_fraction": finite / len(history_cpu),
            "true": summarize(metrics),
            "reverse_nmse_gap_h4": float(
                (reverse_value[:, 3] - metrics["nmse"][:, 3]).mean()
            ),
            "noisy_nmse_gap_h4": float(
                (noisy_value[:, 3] - metrics["nmse"][:, 3]).mean()
            ),
            "runtime_seconds": time.perf_counter() - started,
        }
        print(
            json.dumps(
                {
                    "stage": "evaluate_checkpoint",
                    "checkpoint": checkpoint_path.name,
                    "domain": domain,
                    "h4_nmse": report["domains"][domain]["true"]["nmse"][
                        "mean_by_horizon"
                    ][3],
                }
            ),
            flush=True,
        )
    del model
    torch.cuda.empty_cache()
    return report, raw_metrics


def checkpoint_map(args: argparse.Namespace) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for seed in SEEDS:
        candidates = {
            f"original_v70_seed{seed}": (
                REPO_ROOT
                / f"checkpoints/project/ltwm_expansion/seed{seed}/best.pt"
            ),
            f"previous_seed{seed}": (
                args.previous_run_root
                / f"ltwm/seed{seed}/checkpoints/best_tridomain.pt"
            ),
            f"current_seed{seed}": (
                args.run_root / f"ltwm/seed{seed}/checkpoints/best.pt"
            ),
        }
        for label, path in candidates.items():
            if path.is_file():
                result[label] = path
            elif label.startswith("original_v70_"):
                print(
                    json.dumps(
                        {
                            "warning": "optional original V70 checkpoint absent",
                            "checkpoint": str(path),
                        }
                    ),
                    flush=True,
                )
            else:
                raise FileNotFoundError(path)
    return result


def paired_bootstrap(
    baseline: np.ndarray,
    current: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    if baseline.shape != current.shape or baseline.ndim != 1:
        raise ValueError("paired bootstrap expects aligned vectors")
    delta = current - baseline
    generator = np.random.default_rng(seed)
    indices = generator.integers(
        0,
        len(delta),
        size=(replicates, len(delta)),
    )
    bootstrap_means = delta[indices].mean(axis=1)
    return {
        "baseline_mean": float(baseline.mean()),
        "current_mean": float(current.mean()),
        "mean_delta_current_minus_baseline": float(delta.mean()),
        "relative_improvement_percent": float(
            100.0 * (baseline.mean() - current.mean()) / baseline.mean()
        ),
        "paired_95_percent_ci_delta": np.quantile(
            bootstrap_means,
            (0.025, 0.975),
        ).tolist(),
        "per_record_current_win_rate": float((delta < 0).mean()),
        "bootstrap_probability_mean_delta_below_zero": float(
            (bootstrap_means < 0).mean()
        ),
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA requested but unavailable")
    if args.dino_batch_size < 1 or args.wm_batch_size < 1:
        raise ValueError("batch sizes must be positive")
    if args.bootstrap_replicates < 1000:
        raise ValueError("bootstrap-replicates must be at least 1000")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    split_lock_path = args.run_root / "split_lock.json"
    split_lock = json.loads(split_lock_path.read_text(encoding="utf-8"))
    input_audit: dict[str, Any] = {
        "split_lock": str(split_lock_path),
        "split_lock_sha256": sha256(split_lock_path),
        "domains": {},
    }
    manifests: dict[str, dict[str, Any]] = {}
    for domain in DOMAINS:
        manifest_path, _ = manifest_and_root(args, domain)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifests[domain] = manifest
        input_audit["domains"][domain] = validate_manifest(
            args,
            domain,
            manifest_path,
            manifest,
            split_lock,
        )

    adapter_payload = torch.load(
        args.adapter_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    model_sha = sha256(args.model_dir / "model.safetensors")
    adapter_sha = sha256(args.adapter_checkpoint)
    if adapter_payload.get("dinov3_weight_sha256") != model_sha:
        raise RuntimeError("adapter and DINOv3 checkpoint hashes disagree")
    adapter = DinoV3TokenAdapter(
        DinoV3AdapterConfig(**adapter_payload["config"])
    )
    adapter.load_state_dict(adapter_payload["model"], strict=True)

    from transformers import AutoModel

    backbone = AutoModel.from_pretrained(
        args.model_dir,
        local_files_only=True,
        dtype=torch.float32,
    ).eval().requires_grad_(False).to(device)
    adapter = adapter.eval().requires_grad_(False).to(device)
    inputs_by_domain: dict[str, dict[str, torch.Tensor]] = {}
    encoding_report: dict[str, Any] = {}
    for domain in DOMAINS:
        inputs, encoding = prepare_domain_inputs(
            args,
            domain,
            backbone,
            adapter,
            device,
            model_sha,
            adapter_sha,
        )
        inputs_by_domain[domain] = inputs
        encoding_report[domain] = encoding
    del backbone, adapter
    torch.cuda.empty_cache()

    checkpoints = checkpoint_map(args)
    checkpoint_reports: dict[str, Any] = {}
    raw_by_label: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for label, checkpoint_path in checkpoints.items():
        report, raw = evaluate_checkpoint(
            args,
            checkpoint_path,
            inputs_by_domain,
            device,
        )
        checkpoint_reports[label] = report
        raw_by_label[label] = raw

    comparisons: dict[str, Any] = {}
    for seed in SEEDS:
        current_label = f"current_seed{seed}"
        for baseline_kind in ("previous", "original_v70"):
            baseline_label = f"{baseline_kind}_seed{seed}"
            if baseline_label not in raw_by_label:
                continue
            comparison_name = (
                f"{current_label}_vs_{baseline_label}"
            )
            domain_rows: dict[str, Any] = {}
            for domain_index, domain in enumerate(DOMAINS):
                baseline_h4 = raw_by_label[baseline_label][domain]["nmse"][:, 3]
                current_h4 = raw_by_label[current_label][domain]["nmse"][:, 3]
                domain_rows[domain] = paired_bootstrap(
                    baseline_h4,
                    current_h4,
                    replicates=args.bootstrap_replicates,
                    seed=910000
                    + seed * 10
                    + domain_index
                    + (0 if baseline_kind == "previous" else 100),
                )
            comparisons[comparison_name] = {
                "baseline": baseline_label,
                "current": current_label,
                "domains": domain_rows,
                "macro_h4_nmse_baseline": float(
                    np.mean(
                        [
                            row["baseline_mean"]
                            for row in domain_rows.values()
                        ]
                    )
                ),
                "macro_h4_nmse_current": float(
                    np.mean(
                        [
                            row["current_mean"]
                            for row in domain_rows.values()
                        ]
                    )
                ),
            }
            macro = comparisons[comparison_name]
            macro["macro_relative_improvement_percent"] = float(
                100.0
                * (
                    macro["macro_h4_nmse_baseline"]
                    - macro["macro_h4_nmse_current"]
                )
                / macro["macro_h4_nmse_baseline"]
            )

    ranking = []
    for label, report in checkpoint_reports.items():
        domain_h4 = {
            domain: float(
                report["domains"][domain]["true"]["nmse"][
                    "mean_by_horizon"
                ][3]
            )
            for domain in DOMAINS
        }
        ranking.append(
            {
                "label": label,
                "macro_h4_nmse": float(np.mean(list(domain_h4.values()))),
                "domain_h4_nmse": domain_h4,
            }
        )
    ranking.sort(key=lambda row: row["macro_h4_nmse"])
    result = {
        "schema": "ltwm_strict_tridomain_h4_100_comparison_v1",
        "status": "complete",
        "protocol": {
            "history_frames": 4,
            "future_frames": 4,
            "samples_per_domain": 100,
            "metric_primary": "DINOv3_adapter_token_H4_NMSE",
            "comparison": "paired identical records and encoded inputs",
            "checkpoint_selection_uses_test_metrics": False,
            "test_metrics_used_for_post_training_evaluation_only": True,
            "bootstrap_replicates": args.bootstrap_replicates,
        },
        "input_audit": input_audit,
        "encoding": encoding_report,
        "checkpoints": checkpoint_reports,
        "comparisons": comparisons,
        "ranking": ranking,
        "best_checkpoint_label": ranking[0]["label"],
        "created_at_unix": time.time(),
    }
    output = args.output_dir / "result.json"
    write_json_atomic(output, result)
    print(
        json.dumps(
            {
                "status": "complete",
                "result": str(output),
                "best_checkpoint_label": result["best_checkpoint_label"],
                "ranking": ranking,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
