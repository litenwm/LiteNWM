#!/usr/bin/env python3
"""Fine-tune the original proposal-aware V70 risk ensemble on three domains.

This preserves the original 815-D proposal-aware V70 trunk, setwise context,
heteroscedastic regression, dual/harm heads, and ranking losses. The domain-
conditioned portion is extended to three independent domains: RECON=0,
SCAND=1, and SACSoN=2. Normalization, heads, support statistics, temperatures,
and candidate-level policies are fitted separately for all three domains.

Checkpoint selection is based only on internal scorer-cal partitions.  The
primary criterion is macro ATE ratio with a worst-domain ATE regression
penalty.  External test labels are not accepted by this program.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import random
import sys
import time
from typing import Any

import numpy as np
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scoring.spacing_support_v1 import (  # noqa: E402
    score_spacing_envelope,
    threshold_for_quantile,
)
from scoring.universal_dualdomain_raw_v2 import (  # noqa: E402
    UniversalDualDomainRawV2,
    UniversalModelConfig,
    build_universal_features,
    build_universal_targets,
    feature_normalization,
    normalize_features,
)
from tools.planning.run_cem_safe_elite_v55 import (  # noqa: E402
    candidate_safety_mask,
    select_lowest_energy_safe,
)
from tools.scoring.finetune_universal_dualdomain_safe_rank_v2 import (  # noqa: E402
    load_envelopes,
)
from tools.scoring.run_train_dualdomain_cem_tail_v66_candidate_level import (  # noqa: E402
    candidate_energy,
)
from tools.scoring.train_recon_dense_meta_dinov3_three_arm_v1 import (  # noqa: E402
    load_features,
)
from tools.scoring.train_universal_dualdomain_raw_v2 import (  # noqa: E402
    binary_nll,
    policy_feasible,
    policy_grid,
    positive_weight,
    safety_metrics,
    save_support_bundle,
    sigmoid,
)


DATASETS = ("recon", "scand", "sacson")
MODEL_DOMAIN_INDEX = {"recon": 0, "scand": 1, "sacson": 2}
TEMP_GRID = np.asarray(
    (0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0), dtype=np.float64
)
LOSS_WEIGHTS = {
    "regression": 1.0,
    "helpful_bce": 0.3,
    "dual_bce": 1.5,
    "harm_bce": 2.5,
    "incumbent_pairwise": 1.0,
    "listwise": 1.0,
    "harmful_pair_multiplier": 3.0,
}
CALIBRATION_CONTRACT = {
    "minimum_coverage": 0.4,
    "target_dual_precision": 0.85,
    "target_wilson_lower_95": 0.80,
    "maximum_harm_rate": 0.15,
}
RELAXED_INTERNAL_CALIBRATION_V1 = {
    "minimum_coverage": 0.38,
    "target_dual_precision": 0.80,
    "target_wilson_lower_95": 0.78,
    "maximum_harm_rate": 0.20,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument(
        "--initial-checkpoint",
        type=Path,
        action="append",
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-states", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--gradient-clip-norm", type=float, default=5.0)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--hard-negative-multiplier", type=float, default=6.0)
    parser.add_argument(
        "--sacson-risk-hard-negative-multiplier",
        type=float,
        default=12.0,
    )
    parser.add_argument("--resume-state", type=Path)
    parser.add_argument(
        "--relaxed-finalize-v1",
        action="store_true",
        help=(
            "Finalize an already-completed strict-infeasible run only when "
            "its preregistered best state satisfies "
            "RELAXED_INTERNAL_CALIBRATION_V1. This never trains another epoch."
        ),
    )
    parser.add_argument("--support-quantile", type=float, default=0.999)
    parser.add_argument(
        "--result-status",
        choices=(
            "trained_and_frozen_before_A_to_H_replay",
            "frozen_after_v69_spent_failure_before_v70_spent_replay",
        ),
        default="trained_and_frozen_before_A_to_H_replay",
    )
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_target(path: Path, expected_record: np.ndarray) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        required = {
            "record_key",
            "family",
            "targets",
            "absolute_metrics",
        }
        missing = required.difference(archive.files)
        if missing:
            raise RuntimeError(f"{path} missing target arrays: {sorted(missing)}")
        result = {key: archive[key].copy() for key in required}
    if not np.array_equal(result["record_key"].astype(str), expected_record):
        raise RuntimeError(f"target/feature identity mismatch: {path}")
    if result["targets"].shape != (len(expected_record), 17, 8):
        raise RuntimeError(f"target shape mismatch: {path}")
    if result["absolute_metrics"].shape != (len(expected_record), 17, 4):
        raise RuntimeError(f"absolute metric shape mismatch: {path}")
    return result


def load_hard_mask(
    path: str | None, expected_record: np.ndarray
) -> np.ndarray:
    if path is None:
        return np.zeros((len(expected_record), 17), dtype=bool)
    with np.load(path, allow_pickle=False) as archive:
        if "record_key" not in archive.files or "hard_negative" not in archive.files:
            raise RuntimeError(f"hard-negative artifact is incomplete: {path}")
        record = archive["record_key"].astype(str)
        hard = archive["hard_negative"].astype(bool)
    if not np.array_equal(record, expected_record):
        raise RuntimeError(f"hard-negative identity mismatch: {path}")
    if hard.shape != (len(record), 17):
        raise RuntimeError(f"hard-negative shape mismatch: {path}")
    hard[:, 0] = False
    return hard


def entry_envelope(
    support_path: Path, dataset: str
) -> dict[str, Any]:
    values = load_envelopes(support_path)
    if dataset not in values:
        raise RuntimeError(f"{support_path} has no {dataset} envelope")
    return values[dataset]


def load_entry(
    record: dict[str, Any],
    *,
    support_quantile: float,
) -> dict[str, Any]:
    dataset = str(record["dataset"])
    if dataset not in DATASETS:
        raise RuntimeError(f"unknown dataset in manifest: {dataset}")
    feature_path = Path(record["features"])
    target_path = Path(record["targets"])
    support_path = Path(record["support"])
    data = load_features(feature_path, labels_expected=False)
    target = load_target(target_path, data["record_key"].astype(str))
    envelope = entry_envelope(support_path, dataset)
    spacing = float(record["spacing_m"])
    feature, support_score, physical_guard = build_universal_features(
        data,
        spacing_m=spacing,
        support_envelope=envelope,
    )
    threshold = threshold_for_quantile(envelope, support_quantile)
    valid = physical_guard & (support_score <= threshold)
    valid[:, 0] = True
    hard = load_hard_mask(record.get("hard_negative"), data["record_key"].astype(str))
    return {
        "dataset": dataset,
        "model_domain": np.full(
            len(feature), MODEL_DOMAIN_INDEX[dataset], dtype=np.int64
        ),
        "feature": feature.astype(np.float32),
        "targets_raw": target["targets"].astype(np.float32),
        "targets": build_universal_targets(target["targets"]),
        "absolute_metrics": target["absolute_metrics"].astype(np.float32),
        "valid": valid,
        "hard_negative": hard,
        "record_key": data["record_key"].astype(str),
        "family": target["family"].astype(str),
        "set_type": str(record.get("set_type", "fixed")),
        "support_path": support_path,
        "support_envelope": envelope,
        "spacing_m": spacing,
        "feature_path": feature_path,
        "target_path": target_path,
    }


def concatenate_entries(entries: list[dict[str, Any]]) -> dict[str, Any]:
    if not entries:
        raise RuntimeError("no entries to concatenate")
    target_keys = entries[0]["targets"].keys()
    return {
        "feature": np.concatenate([row["feature"] for row in entries]),
        "targets_raw": np.concatenate([row["targets_raw"] for row in entries]),
        "targets": {
            key: np.concatenate([row["targets"][key] for row in entries])
            for key in target_keys
        },
        "absolute_metrics": np.concatenate(
            [row["absolute_metrics"] for row in entries]
        ),
        "valid": np.concatenate([row["valid"] for row in entries]),
        "hard_negative": np.concatenate(
            [row["hard_negative"] for row in entries]
        ),
        "model_domain": np.concatenate(
            [row["model_domain"] for row in entries]
        ),
        "dataset": np.concatenate(
            [
                np.full(len(row["feature"]), row["dataset"])
                for row in entries
            ]
        ),
        "family": np.concatenate([row["family"] for row in entries]),
        "set_type": np.concatenate(
            [
                np.full(len(row["feature"]), row["set_type"])
                for row in entries
            ]
        ),
        "record_key": np.concatenate([row["record_key"] for row in entries]),
    }


def manifest_entry_state_count(record: dict[str, Any]) -> int:
    path = Path(record["features"])
    with np.load(path, allow_pickle=False) as archive:
        if "record_key" not in archive.files:
            raise RuntimeError(f"{path} has no record_key")
        return int(len(archive["record_key"]))


def load_records_out_of_core(
    records: list[dict[str, Any]],
    cache_dir: Path,
    *,
    support_quantile: float,
    cache_name: str,
) -> dict[str, Any]:
    """Materialize a split into NPY memmaps one source entry at a time."""
    if not records:
        raise RuntimeError(f"{cache_name} has no manifest records")
    if cache_dir.exists():
        raise FileExistsError(f"refusing existing stream cache: {cache_dir}")
    counts = [manifest_entry_state_count(record) for record in records]
    total = int(sum(counts))
    cache_dir.mkdir(parents=True)
    arrays: dict[str, np.memmap] | None = None
    target_arrays: dict[str, np.memmap] | None = None
    metadata: dict[str, list[np.ndarray]] = {
        "dataset": [], "family": [], "set_type": [], "record_key": [],
    }
    cursor = 0
    for entry_index, (record, count) in enumerate(zip(records, counts)):
        entry = load_entry(record, support_quantile=support_quantile)
        if len(entry["feature"]) != count:
            raise RuntimeError(
                f"manifest state count changed for {record['features']}"
            )
        if arrays is None:
            def allocate(name: str, value: np.ndarray) -> np.memmap:
                return np.lib.format.open_memmap(
                    cache_dir / f"{name}.npy", mode="w+",
                    dtype=value.dtype, shape=(total, *value.shape[1:]),
                )
            arrays = {
                "feature": allocate("feature", entry["feature"]),
                "targets_raw": allocate("targets_raw", entry["targets_raw"]),
                "absolute_metrics": allocate("absolute_metrics", entry["absolute_metrics"]),
                "valid": allocate("valid", entry["valid"]),
                "hard_negative": allocate("hard_negative", entry["hard_negative"]),
                "model_domain": allocate("model_domain", entry["model_domain"]),
            }
            target_arrays = {
                key: allocate(f"target_{key}", value)
                for key, value in entry["targets"].items()
            }
        assert arrays is not None and target_arrays is not None
        end = cursor + count
        for key, target in arrays.items():
            target[cursor:end] = entry[key]
        for key, target in target_arrays.items():
            target[cursor:end] = entry["targets"][key]
        metadata["dataset"].append(np.full(count, entry["dataset"]))
        metadata["family"].append(entry["family"])
        metadata["set_type"].append(np.full(count, entry["set_type"]))
        metadata["record_key"].append(entry["record_key"])
        cursor = end
        for value in arrays.values():
            value.flush()
        for value in target_arrays.values():
            value.flush()
        print(json.dumps({
            "stage": "out_of_core_feature_materialization",
            "cache": cache_name, "entry": entry_index + 1,
            "entries": len(records), "completed_states": cursor,
            "total_states": total, "dataset": entry["dataset"],
            "set_type": entry["set_type"],
        }, ensure_ascii=False), flush=True)
        del entry
        gc.collect()
    if cursor != total or arrays is None or target_arrays is None:
        raise RuntimeError(f"{cache_name} stream cache is incomplete")
    return {
        **arrays, "targets": target_arrays,
        **{key: np.concatenate(value) for key, value in metadata.items()},
    }


def dataset_route_set_weights(
    dataset: np.ndarray,
    family: np.ndarray,
    set_type: np.ndarray,
) -> np.ndarray:
    result = np.zeros(len(dataset), dtype=np.float64)
    for name in DATASETS:
        dataset_mask = dataset == name
        local_sets = np.unique(set_type[dataset_mask])
        if not len(local_sets):
            raise RuntimeError(f"training has no {name} states")
        for local_set in local_sets:
            mask = dataset_mask & (set_type == local_set)
            routes, counts = np.unique(family[mask], return_counts=True)
            if not len(routes):
                raise RuntimeError(f"{name}/{local_set} has no routes")
            count_by_route = dict(zip(routes.tolist(), counts.tolist()))
            mass = 1.0 / (len(DATASETS) * len(local_sets) * len(routes))
            for index in np.flatnonzero(mask):
                result[index] = mass / count_by_route[str(family[index])]
    per_domain = {
        name: float(result[dataset == name].sum()) for name in DATASETS
    }
    if not all(np.isclose(value, 1.0 / 3.0) for value in per_domain.values()):
        raise RuntimeError(f"dataset weights are not balanced: {per_domain}")
    return result


def tridomain_target_normalization(
    delta: np.ndarray, domain: np.ndarray
) -> dict[str, np.ndarray]:
    domains = np.asarray(domain, dtype=np.int64)
    mean = []
    scale = []
    for domain_index in range(len(DATASETS)):
        indices = np.flatnonzero(domains == domain_index)
        if not len(indices):
            raise RuntimeError(
                f'target normalization has no domain {domain_index} states'
            )
        total = np.zeros(4, dtype=np.float64)
        square = np.zeros(4, dtype=np.float64)
        count = 0
        for start in range(0, len(indices), 4096):
            local = np.asarray(
                delta[indices[start:start + 4096]], dtype=np.float32
            ).reshape(-1, 4)
            total += local.sum(axis=0, dtype=np.float64)
            square += np.einsum("ij,ij->j", local, local, dtype=np.float64)
            count += len(local)
        local_mean = total / count
        variance = np.maximum(square / count - np.square(local_mean), 0.0)
        mean.append(local_mean)
        scale.append(np.maximum(np.sqrt(variance), 0.005))
    return {
        'target_mean': np.stack(mean).astype(np.float32),
        'target_scale': np.stack(scale).astype(np.float32),
    }


def feature_normalization_out_of_core(
    feature: np.ndarray, domain: np.ndarray
) -> dict[str, np.ndarray]:
    domains = np.asarray(domain, dtype=np.int64)
    means = []
    scales = []
    for domain_index in range(len(DATASETS)):
        indices = np.flatnonzero(domains == domain_index)
        if not len(indices):
            raise RuntimeError(
                f"feature normalization has no domain {domain_index} states"
            )
        total = np.zeros(feature.shape[-1], dtype=np.float64)
        square = np.zeros(feature.shape[-1], dtype=np.float64)
        count = 0
        for start in range(0, len(indices), 1024):
            local = np.asarray(
                feature[indices[start:start + 1024]], dtype=np.float32
            ).reshape(-1, feature.shape[-1])
            total += local.sum(axis=0, dtype=np.float64)
            square += np.einsum("ij,ij->j", local, local, dtype=np.float64)
            count += len(local)
        local_mean = total / count
        variance = np.maximum(square / count - np.square(local_mean), 0.0)
        means.append(local_mean)
        scales.append(np.maximum(np.sqrt(variance), 1.0e-5))
    return {
        "feature_mean": np.stack(means).astype(np.float32),
        "feature_scale": np.stack(scales).astype(np.float32),
    }


def load_initial_model(
    path: Path, device: str, initialization_seed: int
) -> tuple[UniversalDualDomainRawV2, dict[str, Any]]:
    payload = torch.load(path, map_location='cpu', weights_only=False)
    source_config = UniversalModelConfig(**payload['model_config'])
    dual_config = UniversalModelConfig(
        input_dim=815, hidden_dim=256, domain_embedding_dim=16,
        dropout=0.1, domain_conditioned=True, setwise_context=True,
        heteroscedastic=True, num_domains=2,
    )
    tridomain_config = UniversalModelConfig(
        input_dim=815, hidden_dim=256, domain_embedding_dim=16,
        dropout=0.1, domain_conditioned=True, setwise_context=True,
        heteroscedastic=True, num_domains=3,
    )
    if source_config not in (dual_config, tridomain_config):
        raise RuntimeError(f'initial checkpoint is not a compatible V70 member: {path}')
    set_seed(initialization_seed)
    model = UniversalDualDomainRawV2(tridomain_config)
    source_state = payload['state_dict']
    if source_config.num_domains == 3:
        model.load_state_dict(source_state, strict=True)
        initialization = {
            'source_num_domains': 3,
            'target_num_domains': 3,
            'continued_domains': ['recon', 'scand', 'sacson'],
            'initialization_contract': (
                'continued from the independently conditioned three-domain '
                'stage-one V70 member'
            ),
        }
    else:
        destination = model.state_dict()
        with torch.no_grad():
            for key, value in source_state.items():
                if key == 'domain_embedding.weight':
                    if value.shape != (2, tridomain_config.domain_embedding_dim):
                        raise RuntimeError(
                            f'unexpected dual-domain embedding shape: {value.shape}'
                        )
                    destination[key][:2].copy_(value)
                elif key in destination and destination[key].shape == value.shape:
                    destination[key].copy_(value)
                else:
                    raise RuntimeError(
                        f'cannot transplant old V70 parameter {key}: {value.shape}'
                    )
        model.load_state_dict(destination, strict=True)
        initialization = {
            'source_num_domains': 2,
            'target_num_domains': 3,
            'copied_teacher_domains': ['recon', 'scand'],
            'fresh_domain': 'sacson',
            'fresh_domain_initialization_seed': initialization_seed,
            'initialization_contract': (
                'shared trunk plus RECON/SCAND embeddings and heads copied '
                'from the old dual-domain V70 member; SACSoN embedding and '
                'head are freshly initialized with no SCAND teacher parameters'
            ),
        }
    model = model.to(device)
    return model, {
        'path': str(path),
        'sha256': file_sha256(path),
        'schema': payload.get('schema'),
        'seed': payload.get('seed'),
        **initialization,
    }


def ensemble_predict_live(
    models: list[UniversalDualDomainRawV2],
    feature: np.ndarray,
    model_domain: np.ndarray,
    normalization: dict[str, np.ndarray],
    target_norm: dict[str, np.ndarray],
    *,
    device: str,
    batch_states: int,
) -> dict[str, np.ndarray]:
    per_model = []
    for model in models:
        model.eval()
        values: dict[str, list[np.ndarray]] = {
            key: []
            for key in (
                "mean",
                "variance",
                "helpful_logit",
                "dual_logit",
                "harm_logit",
                "rank_logit",
            )
        }
        for start in range(0, len(feature), batch_states):
            end = min(start + batch_states, len(feature))
            local_domain = model_domain[start:end]
            normalized = normalize_features(
                feature[start:end], local_domain, normalization
            )
            with torch.inference_mode():
                output = model(
                    torch.from_numpy(normalized).to(device),
                    torch.from_numpy(local_domain).to(device),
                )
            scale = target_norm["target_scale"][local_domain, None]
            mean = (
                output["mean"].cpu().numpy() * scale
                + target_norm["target_mean"][local_domain, None]
            )
            variance = (
                np.exp(output["logvar"].cpu().numpy()) * np.square(scale)
            )
            values["mean"].append(mean.astype(np.float32))
            values["variance"].append(variance.astype(np.float32))
            for key in (
                "helpful_logit",
                "dual_logit",
                "harm_logit",
                "rank_logit",
            ):
                values[key].append(output[key].cpu().numpy().astype(np.float32))
        per_model.append(
            {key: np.concatenate(value) for key, value in values.items()}
        )
    means = np.stack([value["mean"] for value in per_model])
    variances = np.stack([value["variance"] for value in per_model])
    ensemble_mean = means.mean(0)
    total_variance = (
        variances.mean(0)
        + np.square(means).mean(0)
        - np.square(ensemble_mean)
    )
    return {
        "mean": ensemble_mean.astype(np.float32),
        "std": np.sqrt(np.maximum(total_variance, 1.0e-8)).astype(np.float32),
        **{
            key: np.stack([value[key] for value in per_model]).mean(0)
            for key in (
                "helpful_logit",
                "dual_logit",
                "harm_logit",
                "rank_logit",
            )
        },
    }


def fit_temperature(logit: np.ndarray, target: np.ndarray) -> float:
    return float(
        min(
            TEMP_GRID,
            key=lambda value: binary_nll(logit, target, float(value)),
        )
    )


def normalized_constraint_violation(metrics: dict[str, Any]) -> float:
    """Measure preregistered safety-contract shortfall without relaxing it."""

    coverage = float(metrics.get("coverage") or 0.0)
    dual_precision = float(metrics.get("dual_precision") or 0.0)
    wilson = float(metrics.get("dual_wilson_lower_95") or 0.0)
    harm_rate = float(metrics.get("harm_rate") or 1.0)
    return float(
        max(0.0, CALIBRATION_CONTRACT["minimum_coverage"] - coverage)
        / CALIBRATION_CONTRACT["minimum_coverage"]
        + max(0.0, CALIBRATION_CONTRACT["target_dual_precision"] - dual_precision)
        / CALIBRATION_CONTRACT["target_dual_precision"]
        + max(0.0, CALIBRATION_CONTRACT["target_wilson_lower_95"] - wilson)
        / CALIBRATION_CONTRACT["target_wilson_lower_95"]
        + max(0.0, harm_rate - CALIBRATION_CONTRACT["maximum_harm_rate"])
        / CALIBRATION_CONTRACT["maximum_harm_rate"]
    )


def ate_primary_frontier(
    prediction: dict[str, np.ndarray],
    targets_raw: np.ndarray,
    targets: dict[str, np.ndarray],
    absolute_metrics: np.ndarray,
    valid: np.ndarray,
) -> dict[str, Any]:
    dual_temperature = fit_temperature(
        prediction["dual_logit"], targets["dual"]
    )
    harm_temperature = fit_temperature(
        prediction["harm_logit"], targets["harm"]
    )
    p_dual = sigmoid(prediction["dual_logit"] / dual_temperature)
    p_harm = sigmoid(prediction["harm_logit"] / harm_temperature)
    rows = np.arange(len(targets_raw))
    frontier = []
    for policy in policy_grid():
        energy = candidate_energy(
            prediction, p_dual, p_harm, valid, policy
        )
        safe = candidate_safety_mask(
            prediction, p_dual, p_harm, valid, policy
        )
        selected = select_lowest_energy_safe(energy, safe)
        metrics = safety_metrics(selected, targets_raw)
        chosen = absolute_metrics[rows, selected]
        feasible = policy_feasible(
            {"domain": metrics}, CALIBRATION_CONTRACT
        )
        violation = normalized_constraint_violation(metrics)
        frontier.append(
            {
                "policy": policy,
                "metrics": metrics,
                "selected": selected,
                "feasible": feasible,
                "constraint_violation": violation,
                "mean_ate": float(chosen[:, 0].mean()),
                "mean_rpe": float(chosen[:, 1].mean()),
                "mean_fde": float(chosen[:, 2].mean()),
                "mean_objective": float(chosen[:, 3].mean()),
            }
        )
    feasible_values = [row for row in frontier if row["feasible"]]
    if feasible_values:
        best = min(
            feasible_values,
            key=lambda row: (
                row["mean_ate"],
                row["mean_fde"],
                row["mean_rpe"],
                row["mean_objective"],
                -row["metrics"]["coverage"],
            ),
        )
    else:
        best = min(
            frontier,
            key=lambda row: (
                row["constraint_violation"],
                row["mean_ate"],
                row["mean_fde"],
                row["mean_rpe"],
                row["mean_objective"],
                -row["metrics"]["coverage"],
            ),
        )
    candidate0 = absolute_metrics[:, 0].mean(axis=0)
    return {
        "dual_temperature": dual_temperature,
        "harm_temperature": harm_temperature,
        "policy": best["policy"],
        "metrics": best["metrics"],
        "feasible": bool(feasible_values),
        "feasible_policy_count": len(feasible_values),
        "constraint_violation": best["constraint_violation"],
        "selected": best["selected"],
        "selected_mean": {
            "ate": best["mean_ate"],
            "rpe": best["mean_rpe"],
            "fde": best["mean_fde"],
            "objective": best["mean_objective"],
        },
        "candidate0_mean": {
            "ate": float(candidate0[0]),
            "rpe": float(candidate0[1]),
            "fde": float(candidate0[2]),
            "objective": float(candidate0[3]),
        },
        "selection_rule": (
            "minimum internal-calibration ATE among old-V70-feasible "
            "candidate-level policies; FDE/RPE/objective tie-breakers"
        ),
    }


def calibration_metrics(
    models: list[UniversalDualDomainRawV2],
    calibration: dict[str, dict[str, Any]],
    normalization: dict[str, np.ndarray],
    target_norm: dict[str, np.ndarray],
    *,
    device: str,
    batch_states: int,
) -> tuple[tuple[float, float, float, float], dict[str, Any]]:
    domains = {}
    ratios = []
    fde_ratios = []
    regressions = []
    constraint_violations = []
    all_feasible = True
    for dataset in DATASETS:
        data = calibration[dataset]
        prediction = ensemble_predict_live(
            models,
            data["feature"],
            data["model_domain"],
            normalization,
            target_norm,
            device=device,
            batch_states=batch_states,
        )
        value = ate_primary_frontier(
            prediction,
            data["targets_raw"],
            data["targets"],
            data["absolute_metrics"],
            data["valid"],
        )
        value.pop("selected")
        ate_ratio = (
            value["selected_mean"]["ate"] / value["candidate0_mean"]["ate"]
        )
        fde_ratio = (
            value["selected_mean"]["fde"] / value["candidate0_mean"]["fde"]
        )
        value["ate_ratio"] = ate_ratio
        value["fde_ratio"] = fde_ratio
        domains[dataset] = value
        ratios.append(ate_ratio)
        fde_ratios.append(fde_ratio)
        regressions.append(max(0.0, ate_ratio - 1.0))
        constraint_violations.append(float(value["constraint_violation"]))
        all_feasible &= value["feasible"]
    macro_ate = float(np.mean(ratios))
    macro_fde = float(np.mean(fde_ratios))
    worst_regression = float(max(regressions))
    worst_constraint_violation = float(max(constraint_violations))
    ate_score = macro_ate + 2.0 * worst_regression
    if all_feasible:
        selection_key = (0.0, ate_score, macro_fde, 0.0)
    else:
        selection_key = (
            1.0,
            worst_constraint_violation,
            ate_score,
            macro_fde,
        )
    return selection_key, {
        "score": ate_score,
        "selection_key": list(selection_key),
        "worst_constraint_violation": worst_constraint_violation,
        "macro_ate_ratio": macro_ate,
        "macro_fde_ratio": macro_fde,
        "worst_domain_ate_regression": worst_regression,
        "all_domains_feasible": all_feasible,
        "domains": domains,
    }


def train_one_epoch(
    model: UniversalDualDomainRawV2,
    optimizer: torch.optim.Optimizer,
    train: dict[str, Any],
    order: np.ndarray,
    state_weight: np.ndarray,
    normalization: dict[str, np.ndarray],
    target_norm: dict[str, np.ndarray],
    *,
    batch_states: int,
    hard_negative_multiplier: float,
    sacson_risk_hard_negative_multiplier: float,
    gradient_clip_norm: float,
    pos_weight: dict[str, torch.Tensor],
    device: str,
) -> float:
    model.train()
    accumulated = 0.0
    batches = 0
    for start in range(0, len(order), batch_states):
        index = order[start : start + batch_states]
        local_domain = train["model_domain"][index]
        normalized = normalize_features(
            train["feature"][index], local_domain, normalization
        )
        output = model(
            torch.from_numpy(normalized).to(device),
            torch.from_numpy(local_domain).to(device),
        )
        target_mean = target_norm["target_mean"][local_domain]
        target_scale = target_norm["target_scale"][local_domain]
        expected_delta = torch.from_numpy(
            (
                train["targets"]["delta"][index] - target_mean[:, None]
            )
            / target_scale[:, None]
        ).to(device)
        hard = torch.from_numpy(train["hard_negative"][index]).to(device)
        candidate_weight = (
            1.0 + (hard_negative_multiplier - 1.0) * hard.float()
        )
        denominator = candidate_weight.sum(dim=1).clamp_min(1.0)
        risk_multiplier = torch.full(
            local_domain.shape,
            hard_negative_multiplier,
            dtype=output["mean"].dtype,
            device=device,
        )
        risk_multiplier = torch.where(
            torch.from_numpy(local_domain).to(device) == MODEL_DOMAIN_INDEX["sacson"],
            torch.full_like(
                risk_multiplier,
                sacson_risk_hard_negative_multiplier,
            ),
            risk_multiplier,
        )
        risk_candidate_weight = (
            1.0 + (risk_multiplier[:, None] - 1.0) * hard.float()
        )
        risk_denominator = risk_candidate_weight.sum(dim=1).clamp_min(1.0)
        inverse_variance = torch.exp(-output["logvar"])
        regression = 0.5 * (
            inverse_variance
            * (output["mean"] - expected_delta).square()
            + output["logvar"]
        ).mean(dim=-1)
        state_loss = LOSS_WEIGHTS["regression"] * (
            regression * candidate_weight
        ).sum(dim=1) / denominator
        for name, logit_name, loss_name in (
            ("helpful", "helpful_logit", "helpful_bce"),
            ("dual", "dual_logit", "dual_bce"),
            ("harm", "harm_logit", "harm_bce"),
        ):
            expected = torch.from_numpy(train["targets"][name][index]).to(device)
            value = nn.functional.binary_cross_entropy_with_logits(
                output[logit_name],
                expected,
                reduction="none",
            )
            domain_positive_weight = pos_weight[name][
                torch.from_numpy(local_domain).to(device)
            ][:, None]
            value *= 1.0 + (
                domain_positive_weight - 1.0
            ) * expected
            state_loss += LOSS_WEIGHTS[loss_name] * (
                value * risk_candidate_weight
            ).sum(dim=1) / risk_denominator
        pair_score = (
            output["rank_logit"][:, 1:] - output["rank_logit"][:, :1]
        )
        pair_dual = torch.from_numpy(
            train["targets"]["dual"][index, 1:]
        ).to(device)
        pair_harm = torch.from_numpy(
            train["targets"]["harm"][index, 1:]
        ).to(device)
        pair_loss = nn.functional.softplus(
            -(pair_dual * 2.0 - 1.0) * pair_score
        )
        pair_multiplier = 1.0 + pair_harm * (
            LOSS_WEIGHTS["harmful_pair_multiplier"] - 1.0
        )
        hard_pair = risk_candidate_weight[:, 1:]
        state_loss += LOSS_WEIGHTS["incumbent_pairwise"] * (
            pair_loss * pair_multiplier * hard_pair
        ).sum(dim=1) / hard_pair.sum(dim=1).clamp_min(1.0)
        distribution = torch.from_numpy(
            train["targets"]["list_distribution"][index]
        ).to(device)
        state_loss += LOSS_WEIGHTS["listwise"] * (
            -(
                distribution
                * torch.log_softmax(output["rank_logit"][:, 1:], dim=1)
            ).sum(dim=1)
        )
        local_weight = torch.from_numpy(
            state_weight[index] * len(train["feature"])
        ).to(device=device, dtype=state_loss.dtype)
        loss = (state_loss * local_weight).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), gradient_clip_norm
        )
        optimizer.step()
        accumulated += float(loss.detach().cpu())
        batches += 1
    return accumulated / max(batches, 1)


def save_restart_state(
    path: Path,
    *,
    epoch: int,
    models: list[UniversalDualDomainRawV2],
    optimizers: list[torch.optim.Optimizer],
    rng: list[np.random.Generator],
    best_key: tuple[float, float, float, float],
    best_epoch: int,
    best_calibration: dict[str, Any] | None,
    best_states: list[dict[str, torch.Tensor]] | None,
    stale: int,
    data_manifest_sha256: str,
    training_config: dict[str, Any],
) -> None:
    payload = {
        "schema": "proposal_aware_v70_tridomain_restart_v1",
        "epoch": epoch,
        "model_states": [
            {
                name: value.detach().cpu()
                for name, value in model.state_dict().items()
            }
            for model in models
        ],
        "optimizer_states": [
            optimizer.state_dict() for optimizer in optimizers
        ],
        "rng_states": [value.bit_generator.state for value in rng],
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_random_state": torch.get_rng_state(),
        "cuda_random_states": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        ),
        "best_key": tuple(best_key),
        "best_epoch": best_epoch,
        "best_calibration": best_calibration,
        "best_states": best_states,
        "stale": stale,
        "data_manifest_sha256": data_manifest_sha256,
        "training_config": training_config,
        "outer_val_accessed": False,
        "test_accessed": False,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def save_checkpoint(
    path: Path,
    model: UniversalDualDomainRawV2,
    *,
    seed: int,
    member_role: str,
    initial: dict[str, Any],
    normalization: dict[str, np.ndarray],
    target_norm: dict[str, np.ndarray],
    train_states: int,
    train_states_by_dataset: dict[str, int],
    feature_cache_bindings: list[dict[str, Any]],
    world_models: dict[str, dict[str, Any]],
    paired_world_model: dict[str, Any],
    best_epoch: int,
    calibration: dict[str, Any],
    data_manifest_sha256: str,
    hard_negative_multiplier: float,
    sacson_risk_hard_negative_multiplier: float,
) -> dict[str, Any]:
    payload = {
        "schema": "proposal_aware_v70_tridomain_member_v1",
        "seed": seed,
        "member_role": member_role,
        "model_config": asdict(model.config),
        "normalization": normalization,
        "target_normalization": target_norm,
        "state_dict": {
            key: value.detach().cpu()
            for key, value in model.state_dict().items()
        },
        "primary_excludes_scorer_v2_predictions": True,
        "generic_ate_rtp_scorer_used_as_model_input": False,
        "initial_checkpoint": initial,
        "train_states": train_states,
        'train_states_by_dataset': train_states_by_dataset,
        'feature_cache_bindings': feature_cache_bindings,
        'world_models': world_models,
        'paired_world_model': paired_world_model,
        'model_domain_index': {'recon': 0, 'scand': 1, 'sacson': 2},
        "best_epoch": best_epoch,
        "checkpoint_primary_metric": "internal-calibration ATE",
        "calibration": calibration,
        "hard_negative_multiplier": hard_negative_multiplier,
        "sacson_risk_hard_negative_multiplier": (
            sacson_risk_hard_negative_multiplier
        ),
        "data_manifest_sha256": data_manifest_sha256,
        "outer_val_accessed": False,
        "test_accessed": False,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    return {
        "path": str(path.resolve()),
        "sha256": file_sha256(path),
        "seed": seed,
        "member_role": member_role,
        "train_states": train_states,
    }


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing overwrite: {args.output_dir}")
    if len(args.initial_checkpoint) != 3:
        raise ValueError("exactly three initial V70 members are required")
    if min(args.epochs, args.batch_states, args.patience) <= 0:
        raise ValueError("epoch/batch/patience values must be positive")
    if (
        args.sacson_risk_hard_negative_multiplier
        < args.hard_negative_multiplier
    ):
        raise ValueError(
            "SACSoN risk hard-negative multiplier cannot be smaller "
            "than the shared multiplier"
        )
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    manifest = json.loads(args.data_manifest.read_text(encoding="utf-8"))
    if manifest.get("external_test_accessed"):
        raise RuntimeError("training manifest reports external test access")
    expected_domain_index = {'recon': 0, 'scand': 1, 'sacson': 2}
    if manifest.get('model_domain_index') != expected_domain_index:
        raise RuntimeError('training manifest does not use three independent domains')
    world_models = manifest.get('world_models', {})
    if set(world_models) != {'1701', '1702', '1703'}:
        raise RuntimeError('training manifest must bind three new LTWM seeds')
    for seed, record in world_models.items():
        wm_path = Path(record['path']).resolve(strict=True)
        if file_sha256(wm_path) != record['sha256']:
            raise RuntimeError(f'LTWM checkpoint hash changed for seed {seed}')
    feature_cache_bindings = []
    for split in ('train', 'calibration'):
        for record in manifest[split]:
            feature_path = Path(record['features']).resolve(strict=True)
            feature_sha = file_sha256(feature_path)
            if feature_sha != record.get('features_sha256'):
                raise RuntimeError(f'feature cache hash mismatch: {feature_path}')
            hard_path_value = record.get('hard_negative')
            if hard_path_value is not None:
                hard_path = Path(hard_path_value).resolve(strict=True)
                if file_sha256(hard_path) != record.get('hard_negative_sha256'):
                    raise RuntimeError(f'hard-negative hash mismatch: {hard_path}')
            feature_cache_bindings.append({
                'split': split,
                'dataset': record['dataset'],
                'set_type': record['set_type'],
                'path': str(feature_path),
                'sha256': feature_sha,
            })
    stream_cache_root = (
        args.output_dir.parent / f"{args.output_dir.name}_stream_cache"
    )
    if stream_cache_root.exists():
        raise FileExistsError(
            f"refusing existing stream cache: {stream_cache_root}"
        )
    stream_cache_root.mkdir(parents=True)
    train = load_records_out_of_core(
        manifest["train"], stream_cache_root / "train",
        support_quantile=args.support_quantile, cache_name="train",
    )
    calibration = {
        dataset: load_records_out_of_core(
            [row for row in manifest["calibration"] if row["dataset"] == dataset],
            stream_cache_root / f"calibration_{dataset}",
            support_quantile=args.support_quantile,
            cache_name=f"calibration_{dataset}",
        )
        for dataset in DATASETS
    }
    state_weight = dataset_route_set_weights(
        train["dataset"], train["family"], train["set_type"]
    )
    normalization = feature_normalization_out_of_core(
        train["feature"],
        train["model_domain"],
    )
    target_norm = tridomain_target_normalization(
        train["targets"]["delta"], train["model_domain"]
    )
    models = []
    initial_audit = []
    member_seeds = (1701, 1702, 1703)
    for path, member_seed in zip(args.initial_checkpoint, member_seeds):
        model, audit = load_initial_model(
            path, args.device, initialization_seed=member_seed
        )
        models.append(model)
        initial_audit.append(audit)
    optimizers = [
        torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        for model in models
    ]
    pos_weight = {
        name: torch.tensor(
            [
                positive_weight(
                    train["targets"][name][
                        train["model_domain"] == domain_index
                    ]
                )
                for domain_index in range(len(DATASETS))
            ],
            device=args.device,
        )
        for name in ("helpful", "dual", "harm")
    }
    rng = [np.random.default_rng(seed) for seed in member_seeds]
    args.output_dir.mkdir(parents=True)
    metrics_path = args.output_dir / "metrics.jsonl"
    data_manifest_sha = file_sha256(args.data_manifest)
    best_key = (
        float("inf"),
        float("inf"),
        float("inf"),
        float("inf"),
    )
    best_epoch = 0
    best_calibration: dict[str, Any] | None = None
    best_states: list[dict[str, torch.Tensor]] | None = None
    stale = 0
    start_epoch = 1
    if args.resume_state is not None:
        restart = torch.load(
            args.resume_state,
            map_location="cpu",
            weights_only=False,
        )
        if restart.get("schema") != "proposal_aware_v70_tridomain_restart_v1":
            raise RuntimeError("unsupported restart-state schema")
        if restart.get("data_manifest_sha256") != data_manifest_sha:
            raise RuntimeError("restart-state data manifest SHA changed")
        if len(restart["model_states"]) != len(models):
            raise RuntimeError("restart-state ensemble size changed")
        for model, state in zip(models, restart["model_states"]):
            model.load_state_dict(state, strict=True)
        for optimizer, state in zip(
            optimizers, restart["optimizer_states"]
        ):
            optimizer.load_state_dict(state)
        for generator, state in zip(rng, restart["rng_states"]):
            generator.bit_generator.state = state
        random.setstate(restart["python_random_state"])
        np.random.set_state(restart["numpy_random_state"])
        torch.set_rng_state(restart["torch_random_state"])
        if torch.cuda.is_available() and restart["cuda_random_states"]:
            torch.cuda.set_rng_state_all(restart["cuda_random_states"])
        best_key = tuple(restart["best_key"])
        best_epoch = int(restart["best_epoch"])
        best_calibration = restart["best_calibration"]
        best_states = restart["best_states"]
        stale = int(restart["stale"])
        start_epoch = int(restart["epoch"]) + 1
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({
                "event": "resumed",
                "restart_state": str(args.resume_state.resolve()),
                "restart_epoch": start_epoch - 1,
                "test_accessed": False,
            }, ensure_ascii=False) + "\n")
    finalize_only = (
        args.relaxed_finalize_v1
        and args.resume_state is not None
        and start_epoch == args.epochs + 1
    )
    if start_epoch > args.epochs and not finalize_only:
        raise RuntimeError("restart epoch already exceeds requested epochs")
    started = time.perf_counter()
    for epoch in range(start_epoch, args.epochs + 1):
        losses = []
        for index, (model, optimizer) in enumerate(zip(models, optimizers)):
            order = rng[index].permutation(len(train["feature"]))
            if len(np.unique(order)) != len(order):
                raise RuntimeError("epoch traversal repeated a training state")
            losses.append(
                train_one_epoch(
                    model,
                    optimizer,
                    train,
                    order,
                    state_weight,
                    normalization,
                    target_norm,
                    batch_states=args.batch_states,
                    hard_negative_multiplier=args.hard_negative_multiplier,
                    sacson_risk_hard_negative_multiplier=(
                        args.sacson_risk_hard_negative_multiplier
                    ),
                    gradient_clip_norm=args.gradient_clip_norm,
                    pos_weight=pos_weight,
                    device=args.device,
                )
            )
        key, cal = calibration_metrics(
            models,
            calibration,
            normalization,
            target_norm,
            device=args.device,
            batch_states=args.batch_states,
        )
        row = {
            "epoch": epoch,
            "member_train_loss": losses,
            "calibration": cal,
            "elapsed_sec": time.perf_counter() - started,
            "train_state_coverage": "100% exactly once per member",
        }
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if key < best_key:
            best_key = key
            best_epoch = epoch
            best_calibration = cal
            best_states = [
                {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                }
                for model in models
            ]
            stale = 0
        else:
            stale += 1
        save_restart_state(
            args.output_dir / "restart_state.pt",
            epoch=epoch,
            models=models,
            optimizers=optimizers,
            rng=rng,
            best_key=best_key,
            best_epoch=best_epoch,
            best_calibration=best_calibration,
            best_states=best_states,
            stale=stale,
            data_manifest_sha256=data_manifest_sha,
            training_config={
                "epochs": args.epochs,
                "batch_states": args.batch_states,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "hard_negative_multiplier": args.hard_negative_multiplier,
                "sacson_risk_hard_negative_multiplier": (
                    args.sacson_risk_hard_negative_multiplier
                ),
            },
        )
        if (
            best_calibration is not None
            and best_calibration["all_domains_feasible"]
            and stale >= args.patience
        ):
            break
    if best_states is None or best_calibration is None:
        raise RuntimeError("training never produced a checkpoint")
    final_cem = (
        args.result_status
        == 'frozen_after_v69_spent_failure_before_v70_spent_replay'
    )
    strict_feasible = bool(best_calibration["all_domains_feasible"])
    relaxed_domain_feasible = {
        dataset: (
            best_calibration["domains"][dataset]["metrics"]["coverage"]
            >= RELAXED_INTERNAL_CALIBRATION_V1["minimum_coverage"]
            and best_calibration["domains"][dataset]["metrics"]["dual_precision"]
            >= RELAXED_INTERNAL_CALIBRATION_V1["target_dual_precision"]
            and best_calibration["domains"][dataset]["metrics"][
                "dual_wilson_lower_95"
            ]
            >= RELAXED_INTERNAL_CALIBRATION_V1["target_wilson_lower_95"]
            and best_calibration["domains"][dataset]["metrics"]["harm_rate"]
            <= RELAXED_INTERNAL_CALIBRATION_V1["maximum_harm_rate"]
        )
        for dataset in DATASETS
    }
    relaxed_feasible = all(relaxed_domain_feasible.values())
    if args.relaxed_finalize_v1:
        if not final_cem:
            raise RuntimeError(
                "relaxed fallback is valid only for the final CEM V70 stage"
            )
        if not finalize_only:
            raise RuntimeError(
                "relaxed fallback must finalize epoch-complete restart state "
                "without additional training"
            )
        if strict_feasible:
            raise RuntimeError(
                "strict calibration is already feasible; relaxed fallback "
                "must not replace it"
            )
        if not relaxed_feasible:
            write_json(
                args.output_dir / "RELAXED_CALIBRATION_INFEASIBLE.json",
                {
                    "contract": RELAXED_INTERNAL_CALIBRATION_V1,
                    "domain_feasible": relaxed_domain_feasible,
                    "best_epoch": best_epoch,
                    "best_key": list(best_key),
                    "best_calibration": best_calibration,
                    "source_restart_state": str(args.resume_state.resolve()),
                    "test_accessed": False,
                },
            )
            raise RuntimeError(
                "final V70 calibration does not satisfy the fixed relaxed "
                "contract"
            )
    elif final_cem and not strict_feasible:
        write_json(
            args.output_dir / "CALIBRATION_INFEASIBLE.json",
            {
                "best_epoch": best_epoch,
                "best_key": list(best_key),
                "best_calibration": best_calibration,
                "restart_state": str(
                    (args.output_dir / "restart_state.pt").resolve()
                ),
                "test_accessed": False,
            },
        )
        raise RuntimeError('final V70 calibration is not feasible in all domains')
    for model, state in zip(models, best_states):
        model.load_state_dict(state, strict=True)

    train_states_by_dataset = {
        dataset: int((train['dataset'] == dataset).sum())
        for dataset in DATASETS
    }
    checkpoint_records = []
    for index, model in enumerate(models):
        checkpoint_records.append(
            save_checkpoint(
                args.output_dir
                / "checkpoints"
                / f"proposal_aware_full_seed{index}.pt",
                model,
                seed=member_seeds[index],
                member_role=f"full_seed_{index}",
                initial=initial_audit[index],
                normalization=normalization,
                target_norm=target_norm,
                train_states=len(train["feature"]),
                train_states_by_dataset=train_states_by_dataset,
                feature_cache_bindings=feature_cache_bindings,
                world_models=world_models,
                paired_world_model=world_models[str(member_seeds[index])],
                best_epoch=best_epoch,
                calibration=best_calibration,
                data_manifest_sha256=data_manifest_sha,
                hard_negative_multiplier=args.hard_negative_multiplier,
                sacson_risk_hard_negative_multiplier=(
                    args.sacson_risk_hard_negative_multiplier
                ),
            )
        )
    envelopes = {}
    support_sources = {}
    for dataset in DATASETS:
        source = Path(next(
            row["support"]
            for row in manifest["train"]
            if row["dataset"] == dataset
        ))
        envelopes[dataset] = entry_envelope(source, dataset)
        support_sources[dataset] = {
            "path": str(source),
            "sha256": file_sha256(source),
        }
    support_path = args.output_dir / "support/tridomain_spacing_envelopes.npz"
    save_support_bundle(support_path, envelopes)
    calibration_result = {
        dataset: {
            key: value
            for key, value in best_calibration["domains"][dataset].items()
            if key != "selected"
        }
        for dataset in DATASETS
    }
    for dataset in DATASETS:
        calibration_result[dataset]["strict_feasible"] = bool(
            best_calibration["domains"][dataset]["feasible"]
        )
        calibration_result[dataset]["relaxed_feasible_v1"] = bool(
            relaxed_domain_feasible[dataset]
        )
    result = {
        "schema": "proposal_aware_v70_tridomain_training_result_v1",
        "status": args.result_status,
        "created_at": utc_now(),
        "cem": final_cem,
        "checkpoints": {"full": checkpoint_records},
        "calibration": calibration_result,
        "support": str(support_path.resolve()),
        "support_sha256": file_sha256(support_path),
        "support_sources": support_sources,
        'world_models': world_models,
        'feature_cache_bindings': feature_cache_bindings,
        'model_domain_index': {
            'recon': 0,
            'scand': 1,
            'sacson': 2,
        },
        "train_states": int(len(train["feature"])),
        'train_states_by_dataset': train_states_by_dataset,
        "train_state_coverage": "100% exactly once per member and epoch",
        "dataset_loss_mass": {
            dataset: float(state_weight[train["dataset"] == dataset].sum())
            for dataset in DATASETS
        },
        "best_epoch": best_epoch,
        "best_checkpoint_key": {
            "selection_key": list(best_key),
            "feasible_first": True,
            "ATE_primary_after_all_domains_feasible": True,
            "constraint_violation_primary_before_feasibility": True,
        },
        "checkpoint_primary_metric": "ATE",
        "checkpoint_secondary_metrics": ["FDE", "RPE", "objective"],
        "hard_negative_multiplier": args.hard_negative_multiplier,
        "sacson_risk_hard_negative_multiplier": (
            args.sacson_risk_hard_negative_multiplier
        ),
        "domain_positive_weights": {
            name: value.detach().cpu().tolist()
            for name, value in pos_weight.items()
        },
        "restart_state": str(
            (
                args.resume_state
                if finalize_only
                else args.output_dir / "restart_state.pt"
            ).resolve()
        ),
        "calibration_contract": CALIBRATION_CONTRACT,
        "strict_all_domains_feasible": strict_feasible,
        "relaxed_calibration_contract_v1": (
            RELAXED_INTERNAL_CALIBRATION_V1
            if args.relaxed_finalize_v1
            else None
        ),
        "relaxed_domain_feasible_v1": (
            relaxed_domain_feasible
            if args.relaxed_finalize_v1
            else None
        ),
        "relaxed_all_domains_feasible_v1": (
            relaxed_feasible
            if args.relaxed_finalize_v1
            else None
        ),
        "seal_status": (
            "RELAXED_FEASIBLE_V1"
            if args.relaxed_finalize_v1
            else "STRICT_FEASIBLE"
        ),
        "calibration_selection": (
            "ATE-primary among feasible old-V70 candidate-level policies"
        ),
        "data_manifest": str(args.data_manifest.resolve()),
        "data_manifest_sha256": data_manifest_sha,
        "initial_checkpoints": initial_audit,
        "generic_absolute_ate_rtp_scorer_used_by_model": False,
        "RTP_used_by_dual_or_harm_supervision": False,
        "A_to_H_accessed": False,
        "A_to_H_used_by_v70_training_or_calibration": False,
        "fresh_scand_used_by_v70_training_or_calibration": False,
        "reserve_accessed": False,
        "outer_val_accessed": False,
        "test_accessed": False,
        "decision": (
            "PROPOSAL_AWARE_V70_TRIDOMAIN_ENSEMBLE_"
            "FROZEN_RELAXED_FEASIBLE_V1"
            if args.relaxed_finalize_v1
            else "PROPOSAL_AWARE_V70_TRIDOMAIN_ENSEMBLE_FROZEN"
        ),
    }
    write_json(args.output_dir / "result.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
