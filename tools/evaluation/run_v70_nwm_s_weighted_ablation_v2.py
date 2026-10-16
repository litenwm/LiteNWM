#!/usr/bin/env python3
"""Matched NWM-S weight / project scorer / v70 selector ablation.

The original NWM-S EMA checkpoint is audited and bound to its completed
250-step LPIPS H4 outputs.  Every fixed-pool method is evaluated on the same
trimmed-mean + NoMaD-x16 K17 pool.  Search ablations then add the frozen v70
CEM rounds and cross-round archive.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]


METRICS = ("ate", "rpe", "endpoint", "objective")
DOMAINS = ("recon", "scand")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def route_group(domain: str, manifest_path: Path) -> np.ndarray:
    rows = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != 100:
        raise RuntimeError(f"{domain} manifest is not official-100")
    return np.asarray(
        [f"{domain}::{row['route_group']}" for row in rows]
    )


def select_lowest_energy_safe(
    energy: np.ndarray,
    safe: np.ndarray,
) -> np.ndarray:
    masked = np.where(safe, energy, np.inf)
    selected = masked.argmin(axis=1).astype(np.int64)
    return np.where(safe.any(axis=1), selected, 0).astype(np.int64)


def metric_values(
    selected_waypoint_units: np.ndarray,
    ground_truth_m: np.ndarray,
) -> np.ndarray:
    value = np.asarray(selected_waypoint_units, dtype=np.float64) * 0.25
    truth = np.asarray(ground_truth_m, dtype=np.float64)
    error = value - truth
    squared = np.sum(error * error, axis=-1)
    ate = np.sqrt(np.mean(squared, axis=-1))
    endpoint = np.sqrt(squared[:, -1])
    relative = np.diff(value, axis=1) - np.diff(truth, axis=1)
    rpe = np.sqrt(np.mean(np.sum(relative * relative, axis=-1), axis=-1))
    objective = 0.4 * ate + 0.6 * endpoint
    return np.stack((ate, rpe, endpoint, objective), axis=-1)


def summarize(
    metrics: np.ndarray,
    incumbent_metrics: np.ndarray,
    promoted: np.ndarray,
) -> dict[str, Any]:
    delta = incumbent_metrics - metrics
    dual = promoted & (delta[:, 0] > 1.0e-7) & (delta[:, 3] > 1.0e-7)
    harm = promoted & ((delta[:, 0] < -1.0e-7) | (delta[:, 3] < -1.0e-7))
    count = int(promoted.sum())
    return {
        "states": int(len(metrics)),
        "absolute_mean_m": {
            name: float(metrics[:, index].mean())
            for index, name in enumerate(METRICS)
        },
        "absolute_p90_m": {
            name: float(np.quantile(metrics[:, index], 0.9))
            for index, name in enumerate(METRICS)
        },
        "gain_vs_incumbent_mean_m": {
            name: float(delta[:, index].mean())
            for index, name in enumerate(METRICS)
        },
        "coverage": float(promoted.mean()),
        "promotions": count,
        "dual_count": int(dual.sum()),
        "dual_precision": float(dual.sum() / count) if count else None,
        "harm_count": int(harm.sum()),
        "harm_rate": float(harm.sum() / count) if count else None,
    }


def route_bootstrap_pair(
    first: np.ndarray,
    second: np.ndarray,
    route: np.ndarray,
    *,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    unique = np.asarray(sorted(np.unique(route).tolist()))
    rng = np.random.default_rng(seed)
    output: dict[str, Any] = {"routes": int(len(unique)), "draws": int(draws)}
    for index, name in enumerate(METRICS):
        improvement = second[:, index] - first[:, index]
        grouped = [improvement[route == value] for value in unique]
        sampled = np.empty(draws, dtype=np.float64)
        for draw in range(draws):
            choice = rng.integers(0, len(unique), size=len(unique))
            sampled[draw] = np.concatenate(
                [grouped[item] for item in choice]
            ).mean()
        output[f"{name}_improvement_m"] = {
            "mean": float(improvement.mean()),
            "lcb95": float(np.quantile(sampled, 0.025)),
            "ucb95": float(np.quantile(sampled, 0.975)),
            "probability_positive": float((sampled > 0.0).mean()),
        }
    return output


def load_nwm_selection(
    domain: str,
    output_dir: Path,
    expected_record: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any], dict[str, str]]:
    summary_path = output_dir / "summary.json"
    per_sample_path = output_dir / "per_sample.csv"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    with per_sample_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 100 or int(summary["num_samples"]) != 100:
        raise RuntimeError(f"{domain} NWM-S output is incomplete")
    if (
        summary["nwm_config"] != "config/nwm_cdit_s.yaml"
        or not str(summary["nwm_ckpt"]).endswith(
            "logs/nwm_cdit_s/checkpoints/100000.pth.tar"
        )
        or int(summary["diffusion_steps"]) != 250
        or int(summary["num_repeat_eval"]) != 1
        or summary["score_mode"] != "lpips"
        or int(summary["target_horizon"]) != 4
        or int(summary["num_candidates"]) != 16
    ):
        raise RuntimeError(f"{domain} NWM-S protocol changed")
    observed = np.asarray(
        [f"{row['traj_name']}:{int(row['curr_time'])}" for row in rows]
    )
    if not np.array_equal(observed, expected_record.astype(str)):
        raise RuntimeError(f"{domain} NWM-S record identity mismatch")
    selected = np.asarray(
        [int(row["nwm_selected_index"]) + 1 for row in rows],
        dtype=np.int64,
    )
    if np.any((selected < 1) | (selected > 16)):
        raise RuntimeError(f"{domain} NWM-S index outside NoMaD x16")
    return selected, summary, {
        "summary": sha256_file(summary_path),
        "per_sample": sha256_file(per_sample_path),
    }


def load_domain(
    domain: str,
    benchmark_root: Path,
    nwm_output: Path,
) -> dict[str, Any]:
    directory = benchmark_root / domain
    features_path = directory / "features" / f"official100_{domain}_features.npz"
    selected_path = directory / "v70_replay" / "selected_actions.npz"
    selected_metrics_path = directory / "selected_metrics.npz"
    manifest_path = directory / "manifest.windows.jsonl"
    result_path = directory / "result.json"
    feature = load_npz(features_path)
    selected = load_npz(selected_path)
    selected_metrics = load_npz(selected_metrics_path)
    existing_result = json.loads(result_path.read_text(encoding="utf-8"))
    record = feature["record_key"].astype(str)
    if not np.array_equal(record, selected["record_key"].astype(str)):
        raise RuntimeError(f"{domain} feature/selection identity mismatch")
    if not np.array_equal(record, selected_metrics["record_key"].astype(str)):
        raise RuntimeError(f"{domain} feature/metric identity mismatch")
    base = feature["candidates"].astype(np.float32)
    if not np.array_equal(
        base,
        selected["base_k17_feature_waypoint_units"].astype(np.float32),
    ):
        raise RuntimeError(f"{domain} K17 candidate mismatch")
    nwm_index, nwm_summary, nwm_hashes = load_nwm_selection(
        domain, nwm_output, record
    )
    row = np.arange(len(record))
    incumbent = base[:, 0]
    methods: dict[str, np.ndarray] = {
        "incumbent": incumbent,
        "dino_static_only": incumbent,
        "nwm_s_lpips_raw": base[row, nwm_index],
    }
    valid = selected["base_valid"].astype(bool)
    safe = selected["base_safe"].astype(bool)
    valid_index = np.where(valid[row, nwm_index], nwm_index, 0)
    safe_index = np.where(safe[row, nwm_index], nwm_index, 0)
    methods["nwm_s_valid_fallback"] = base[row, valid_index]
    methods["nwm_s_v70_safe_fallback"] = base[row, safe_index]
    scorer = feature["scorer_v2_aux_features"].astype(np.float32)
    raw_energy = 0.4 * scorer[..., 6] + 0.6 * scorer[..., 8]
    raw_index = np.argmin(raw_energy, axis=1).astype(np.int64)
    methods["project_scorer_v2_raw_argmin"] = base[row, raw_index]
    fixed_index = selected["fixed_pool_selected_index"].astype(np.int64)
    methods["v70_fixed_k17"] = base[row, fixed_index]

    iteration = directory / "v70_replay" / "iteration_02"
    population = load_npz(iteration / "population.npz")
    score = load_npz(iteration / "frozen_scores.npz")
    if not np.array_equal(record, population["record_key"].astype(str)):
        raise RuntimeError(f"{domain} final-round population identity mismatch")
    final_round_candidates = population[
        "candidates_feature_waypoint_units"
    ].astype(np.float32)
    final_round_index = select_lowest_energy_safe(
        score["energy"].astype(np.float32),
        score["safe"].astype(bool),
    )
    methods["v70_final_round_only"] = final_round_candidates[
        row, final_round_index
    ]
    formal_candidates = selected[
        "final_cem_feature_waypoint_units"
    ].astype(np.float32)
    formal_index = selected["final_cem_selected_index"].astype(np.int64)
    methods["v70_cross_round_archive"] = formal_candidates[row, formal_index]

    pools = [base]
    for iteration_index in range(3):
        population_path = (
            directory
            / "v70_replay"
            / f"iteration_{iteration_index:02d}"
            / "population.npz"
        )
        pool = load_npz(population_path)[
            "candidates_feature_waypoint_units"
        ].astype(np.float32)
        pools.append(pool[:, 1:])
    union = np.concatenate(pools, axis=1)
    ground_truth = selected_metrics["ground_truth_local_m"].astype(np.float32)
    union_flat = union.reshape(-1, 4, 2)
    repeated_ground = np.repeat(ground_truth, union.shape[1], axis=0)
    union_values = metric_values(union_flat, repeated_ground).reshape(
        len(record), union.shape[1], 4
    )
    oracle_index = np.argmin(union_values[..., 3], axis=1)
    methods["union65_objective_oracle"] = union[row, oracle_index]

    metric = {
        name: metric_values(value, ground_truth)
        for name, value in methods.items()
    }
    incumbent_metrics = metric["incumbent"]
    promoted = {
        name: np.any(value != incumbent, axis=(1, 2))
        for name, value in methods.items()
    }
    summary = {
        name: summarize(metric[name], incumbent_metrics, promoted[name])
        for name in methods
    }
    expected = existing_result["methods"]
    consistency = {
        "nwm_s_ate_abs_diff_m": abs(
            summary["nwm_s_lpips_raw"]["absolute_mean_m"]["ate"]
            - float(expected["nomad_plus_nwm_rep1_lpips"]["absolute_mean_m"]["ate"])
        ),
        "project_raw_ate_abs_diff_m": abs(
            summary["project_scorer_v2_raw_argmin"]["absolute_mean_m"]["ate"]
            - float(
                expected["dwm_plus_scorer_v2_raw_argmin_k17"][
                    "absolute_mean_m"
                ]["ate"]
            )
        ),
        "v70_archive_ate_abs_diff_m": abs(
            summary["v70_cross_round_archive"]["absolute_mean_m"]["ate"]
            - float(
                expected["formal_v70_three_round_cem"]["absolute_mean_m"]["ate"]
            )
        ),
    }
    if max(consistency.values()) > 1.0e-7:
        raise RuntimeError(f"{domain} existing benchmark was not reproduced")
    hashes = {
        "features": sha256_file(features_path),
        "selected_actions": sha256_file(selected_path),
        "selected_metrics": sha256_file(selected_metrics_path),
        "manifest": sha256_file(manifest_path),
        "existing_result": sha256_file(result_path),
        "nwm_summary": nwm_hashes["summary"],
        "nwm_per_sample": nwm_hashes["per_sample"],
    }
    return {
        "record": record,
        "route": route_group(domain, manifest_path),
        "methods": methods,
        "metrics": metric,
        "summary": summary,
        "nwm_summary": nwm_summary,
        "consistency": consistency,
        "hashes": hashes,
        "diagnostics": {
            "nwm_valid_acceptance": float((valid_index != 0).mean()),
            "nwm_v70_safe_acceptance": float((safe_index != 0).mean()),
            "nwm_vs_v70_fixed_same_action": float(
                np.all(base[row, nwm_index] == base[row, fixed_index], axis=(1, 2)).mean()
            ),
            "archive_source_counts": {
                "incumbent": int(
                    (selected["archive_source_round"].astype(np.int64) == -2).sum()
                ),
                "fixed": int(
                    (selected["archive_source_round"].astype(np.int64) == -1).sum()
                ),
                "round0": int(
                    (selected["archive_source_round"].astype(np.int64) == 0).sum()
                ),
                "round1": int(
                    (selected["archive_source_round"].astype(np.int64) == 1).sum()
                ),
                "round2": int(
                    (selected["archive_source_round"].astype(np.int64) == 2).sum()
                ),
            },
        },
    }


def format_method_row(name: str, value: dict[str, Any]) -> str:
    mean = value["absolute_mean_m"]
    p90 = value["absolute_p90_m"]
    dual = value["dual_precision"]
    harm = value["harm_rate"]
    return (
        f"| {name} | {1000 * mean['ate']:.2f} | {1000 * p90['ate']:.2f} | "
        f"{1000 * mean['rpe']:.2f} | {1000 * mean['objective']:.2f} | "
        f"{100 * value['coverage']:.1f}% | "
        f"{'—' if dual is None else f'{100 * dual:.1f}%'} | "
        f"{'—' if harm is None else f'{100 * harm:.1f}%'} |"
    )


def format_pair_row(name: str, value: dict[str, Any]) -> str:
    ate = value["ate_improvement_m"]
    objective = value["objective_improvement_m"]
    return (
        f"| {name} | {1000 * ate['mean']:.2f} | "
        f"[{1000 * ate['lcb95']:.2f}, {1000 * ate['ucb95']:.2f}] | "
        f"{1000 * objective['mean']:.2f} | "
        f"[{1000 * objective['lcb95']:.2f}, "
        f"{1000 * objective['ucb95']:.2f}] |"
    )


def build_report(result: dict[str, Any]) -> str:
    method_order = [
        "incumbent",
        "dino_static_only",
        "nwm_s_lpips_raw",
        "nwm_s_valid_fallback",
        "nwm_s_v70_safe_fallback",
        "project_scorer_v2_raw_argmin",
        "v70_fixed_k17",
        "v70_final_round_only",
        "v70_cross_round_archive",
        "union65_objective_oracle",
    ]
    lines = [
        "# v70 × 原版 NWM-S 权重消融",
        "",
        "> 本实验明确加载并审计原版 NWM-S `CDiT-S/2` 100k EMA checkpoint，"
        "并绑定其 250-step、LPIPS、NoMaD-x16 完整 H4-100 输出。"
        "官方 100 标签已经打开，因此这里是探索性归因，不是新的盲测。",
        "",
        "## 权重与协议",
        "",
        f"- NWM-S checkpoint SHA256：`{result['nwm_s_weight_audit']['sha256']}`",
        f"- 模型：`{result['nwm_s_weight_audit']['architecture']}`；"
        f"train steps：`{result['nwm_s_weight_audit']['train_steps']}`；"
        f"EMA tensors：`{result['nwm_s_weight_audit']['ema_tensors']}`。",
        "- 固定池比较均使用完全相同的 K17：trimmed-mean incumbent + NoMaD x16。",
        "- NWM-S：250-step CDiT rollout，解码最终帧，用 LPIPS-Alex 对目标图排序。",
        "- v70：项目 scorer 的 ATE/Objective/dual/harm/rank 与不确定性门控，"
        "再比较固定 K17、末轮 CEM 和跨轮安全归档。",
        "",
    ]
    for domain in DOMAINS:
        lines.extend(
            [
                f"## {domain.upper()}-100",
                "",
                "| 方法 | ATE mean mm | ATE p90 mm | RPE mean mm | Objective mean mm | coverage | dual | harm |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for name in method_order:
            lines.append(
                format_method_row(name, result["domains"][domain]["summary"][name])
            )
        lines.extend(
            [
                "",
                "配对比较（正数表示前者误差更低）：",
                "",
                "| 对比 | ΔATE mm | route-bootstrap 95% CI | ΔObjective mm | 95% CI |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for name, value in result["domains"][domain]["pairwise"].items():
            lines.append(format_pair_row(name, value))
        lines.append("")
    lines.extend(
        [
            "## Pooled matched-200",
            "",
            "| 对比 | ΔATE mm | route-bootstrap 95% CI | ΔObjective mm | 95% CI |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name, value in result["pooled_pairwise"].items():
        lines.append(format_pair_row(name, value))
    lines.extend(
        [
            "",
            "## 结论",
            "",
            f"- NWM-S 权重信号：**{result['decision']['nwm_weight_signal']}**。",
            f"- 项目 scorer/ranker 信号：**{result['decision']['project_ranker_signal']}**。",
            f"- v70 选择器/CEM/归档信号：**{result['decision']['selector_signal']}**。",
            "",
            "DINO static-only 与 incumbent 相同，因为它在同一状态内没有候选动作差异。"
            "本实验比较的是完整动作选择链，而不是声称 NWM-S、DINO 和项目 LTWM "
            "在参数层面可以互换。",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("protocol_id") != "v70_nwm_s_weighted_ablation_v2":
        raise RuntimeError("unexpected protocol")
    if config.get("status") != "frozen_before_new_attribution_run":
        raise RuntimeError("protocol is not frozen")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "result.json"
    if result_path.exists():
        raise FileExistsError(f"refusing to overwrite: {result_path}")

    checkpoint_path = Path(config["nwm_s"]["checkpoint"]).resolve()
    checkpoint_hash = sha256_file(checkpoint_path)
    if checkpoint_hash != config["nwm_s"]["checkpoint_sha256"]:
        raise RuntimeError("NWM-S checkpoint hash changed")
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    nwm_config_path = Path(config["nwm_s"]["config"]).resolve()
    nwm_config = yaml.safe_load(nwm_config_path.read_text(encoding="utf-8"))
    if nwm_config["model"] != "CDiT-S/2":
        raise RuntimeError("NWM-S architecture changed")
    if int(checkpoint["train_steps"]) != 100000:
        raise RuntimeError("NWM-S train step changed")
    weight_audit = {
        "path": str(checkpoint_path),
        "sha256": checkpoint_hash,
        "architecture": nwm_config["model"],
        "checkpoint_state": "ema",
        "train_steps": int(checkpoint["train_steps"]),
        "epoch": int(checkpoint["epoch"]),
        "model_tensors": int(len(checkpoint["model"])),
        "ema_tensors": int(len(checkpoint["ema"])),
        "config": str(nwm_config_path),
        "config_sha256": sha256_file(nwm_config_path),
    }
    del checkpoint

    benchmark_root = Path(config["inputs"]["benchmark_root"])
    if not benchmark_root.is_absolute():
        benchmark_root = (ROOT / benchmark_root).resolve()
    nwm_outputs = {
        "recon": Path(config["inputs"]["recon_nwm_output"]).resolve(),
        "scand": Path(config["inputs"]["scand_nwm_output"]).resolve(),
    }
    domains = {
        domain: load_domain(
            domain, benchmark_root, nwm_outputs[domain]
        )
        for domain in DOMAINS
    }
    pair_specs = {
        "nwm_safe_fallback_minus_nwm_raw": (
            "nwm_s_v70_safe_fallback",
            "nwm_s_lpips_raw",
        ),
        "v70_fixed_minus_nwm_safe_fallback": (
            "v70_fixed_k17",
            "nwm_s_v70_safe_fallback",
        ),
        "v70_archive_minus_v70_fixed": (
            "v70_cross_round_archive",
            "v70_fixed_k17",
        ),
        "v70_archive_minus_final_round_only": (
            "v70_cross_round_archive",
            "v70_final_round_only",
        ),
        "v70_archive_minus_nwm_raw": (
            "v70_cross_round_archive",
            "nwm_s_lpips_raw",
        ),
    }
    draws = int(config["statistics"]["bootstrap_draws"])
    seed = int(config["statistics"]["bootstrap_seed"])
    for domain_index, domain in enumerate(DOMAINS):
        domains[domain]["pairwise"] = {
            name: route_bootstrap_pair(
                domains[domain]["metrics"][first],
                domains[domain]["metrics"][second],
                domains[domain]["route"],
                draws=draws,
                seed=seed + 100 * domain_index + pair_index,
            )
            for pair_index, (name, (first, second)) in enumerate(
                pair_specs.items()
            )
        }

    pooled_metrics = {
        name: np.concatenate(
            [domains[domain]["metrics"][name] for domain in DOMAINS]
        )
        for name in domains["recon"]["metrics"]
    }
    pooled_route = np.concatenate(
        [domains[domain]["route"] for domain in DOMAINS]
    )
    pooled_pairwise = {
        name: route_bootstrap_pair(
            pooled_metrics[first],
            pooled_metrics[second],
            pooled_route,
            draws=draws,
            seed=seed + 1000 + pair_index,
        )
        for pair_index, (name, (first, second)) in enumerate(pair_specs.items())
    }

    nwm_vs_incumbent = route_bootstrap_pair(
        pooled_metrics["nwm_s_lpips_raw"],
        pooled_metrics["incumbent"],
        pooled_route,
        draws=draws,
        seed=seed + 2000,
    )["ate_improvement_m"]
    ranker = pooled_pairwise["v70_fixed_minus_nwm_safe_fallback"][
        "ate_improvement_m"
    ]
    archive = pooled_pairwise["v70_archive_minus_v70_fixed"][
        "ate_improvement_m"
    ]
    final_round = pooled_pairwise[
        "v70_archive_minus_final_round_only"
    ]["ate_improvement_m"]
    raw_harm = np.mean(
        [
            domains[domain]["summary"]["nwm_s_lpips_raw"]["harm_rate"]
            for domain in DOMAINS
        ]
    )
    safe_harm = np.mean(
        [
            domains[domain]["summary"]["nwm_s_v70_safe_fallback"]["harm_rate"]
            for domain in DOMAINS
        ]
    )
    if nwm_vs_incumbent["lcb95"] > 0.0:
        nwm_signal = "supported: NWM-S raw improves paired ATE over incumbent"
    elif nwm_vs_incumbent["mean"] > 0.0:
        nwm_signal = "directionally positive, but the route interval crosses zero"
    else:
        nwm_signal = "not supported: raw NWM-S does not beat the common incumbent"
    if ranker["lcb95"] > 0.0:
        project_signal = (
            "supported: under the same K17 and v70 fallback, the project ranker "
            "beats the NWM-S-selected challenger"
        )
    elif ranker["mean"] > 0.0:
        project_signal = "directionally positive, but not route-robust"
    else:
        project_signal = "not supported by the common-pool ranking comparison"
    if (
        safe_harm < raw_harm
        and archive["lcb95"] > 0.0
        and final_round["lcb95"] > 0.0
    ):
        selector_signal = (
            "supported: safety fallback reduces NWM harm and cross-round archive "
            "beats both fixed-K17 and final-round-only"
        )
    elif safe_harm < raw_harm and archive["mean"] > 0.0:
        selector_signal = (
            "partially supported: safety improves and archive is directionally "
            "better, but one route interval crosses zero"
        )
    else:
        selector_signal = "not supported by all predeclared criteria"

    serializable_domains = {
        domain: {
            key: value
            for key, value in domains[domain].items()
            if key not in {"methods", "metrics", "record", "route"}
        }
        for domain in DOMAINS
    }
    arrays_path = output_dir / "selected_metrics.npz"
    with arrays_path.open("wb") as handle:
        np.savez_compressed(
            handle,
            **{
                f"{domain}__{name}__metrics_m": domains[domain]["metrics"][name]
                for domain in DOMAINS
                for name in domains[domain]["metrics"]
            },
            **{
                f"{domain}__route_group": domains[domain]["route"]
                for domain in DOMAINS
            },
            **{
                f"{domain}__record_key": domains[domain]["record"]
                for domain in DOMAINS
            },
        )
    result = {
        "schema": "v70_nwm_s_weighted_ablation_v2",
        "status": "complete_exploratory_matched_official100",
        "confirmatory_claim_allowed": False,
        "nwm_s_weight_audit": weight_audit,
        "domains": serializable_domains,
        "pooled_pairwise": pooled_pairwise,
        "nwm_s_raw_minus_incumbent_pooled": nwm_vs_incumbent,
        "decision": {
            "nwm_weight_signal": nwm_signal,
            "project_ranker_signal": project_signal,
            "selector_signal": selector_signal,
        },
        "claim_boundary": (
            "The NWM-S outputs and v70 selections were originally frozen before "
            "their corresponding outcome reads, but this new hybrid attribution "
            "was designed after official-100 labels were opened."
        ),
        "inputs": {
            "config": {
                "path": str(config_path),
                "sha256": sha256_file(config_path),
            },
            "runner": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
        },
        "artifacts": {
            "selected_metrics": {
                "path": str(arrays_path),
                "sha256": sha256_file(arrays_path),
            }
        },
    }
    result_path.write_text(
        json.dumps(jsonable(result), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report_path = output_dir / "report.md"
    report_path.write_text(build_report(result), encoding="utf-8")
    print(
        json.dumps(
            {
                "result": str(result_path),
                "report": str(report_path),
                "nwm_s_raw_minus_incumbent_ate_m": nwm_vs_incumbent["mean"],
                "v70_fixed_minus_nwm_safe_ate_m": ranker["mean"],
                "archive_minus_fixed_ate_m": archive["mean"],
                "decision": result["decision"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
