#!/usr/bin/env python3
"""Run the frozen v70 DINO/model/selector attribution matrix.

This is an exploratory analysis on spent development outcomes.  Representation
arms share the same route-OOF ridge calibration and differ only in their frozen
input signal.  Selector arms share the exact current residual-ridge predictions
and vary one policy component at a time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from tools.evaluation import (  # noqa: E402
    run_scand_union65_densemeta_residual_ridge_v1 as residual,
)


MODE_LABELS = {
    "action": "Action-only pretrained scorer",
    "static": "Action + DINO static context",
    "latent": "Project LTWM rollout",
    "full": "Action + DINO + project LTWM",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


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


def resolve_input(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def mode_feature(
    transfer: dict[str, np.ndarray],
    formal: np.ndarray,
    mode: str,
) -> np.ndarray:
    states = len(formal)
    row = np.arange(states)
    parts: list[np.ndarray] = []
    for key in ("outcome_mean", "outcome_std"):
        value = transfer[f"{mode}_{key}"].astype(np.float32)
        parts.extend((value, value - value[row, formal, None, :]))
    for key in ("p_help_mean", "p_safe_mean", "p_harm_mean", "p_select_mean"):
        value = transfer[f"{mode}_{key}"].astype(np.float32)[..., None]
        parts.extend((value, value - value[row, formal, None, :]))
    feature = np.concatenate(parts, axis=-1).astype(np.float32)
    if feature.shape[-1] != 24:
        raise RuntimeError(f"unexpected {mode} calibrator dimension: {feature.shape}")
    if not np.isfinite(feature).all():
        raise RuntimeError(f"{mode} calibrator feature contains NaN or Inf")
    return feature


def geometry_feature(
    table: dict[str, np.ndarray],
    formal: np.ndarray,
) -> np.ndarray:
    states = len(formal)
    row = np.arange(states)
    geometry = table["candidates_feature_waypoint_units"].astype(np.float32)
    geometry = geometry.reshape(*geometry.shape[:2], -1)
    feature = np.concatenate(
        (
            geometry,
            geometry - geometry[row, formal, None, :],
            table["source_round"].astype(np.float32)[..., None],
        ),
        axis=-1,
    )
    if not np.isfinite(feature).all():
        raise RuntimeError("geometry feature contains NaN or Inf")
    return feature.astype(np.float32)


def make_targets(
    table: dict[str, np.ndarray],
) -> np.ndarray:
    ate_gain = table["ate_gain_vs_formal_v70_m"].astype(np.float32)
    objective_gain = table["objective_gain_vs_formal_v70_m"].astype(np.float32)
    downside = np.maximum(np.maximum(-ate_gain, -objective_gain), 0.0)
    return np.stack(
        (
            ate_gain,
            objective_gain,
            table["harm_vs_formal_v70"].astype(np.float32),
            downside,
        ),
        axis=-1,
    )


def select_energy(
    prediction: np.ndarray,
    valid: np.ndarray,
    formal: np.ndarray,
    *,
    gate: np.ndarray | None = None,
    pool: np.ndarray | None = None,
    fallback: bool,
) -> np.ndarray:
    states, candidates = valid.shape
    row = np.arange(states)
    eligible = valid.copy()
    if gate is not None:
        eligible &= gate
    if pool is not None:
        eligible &= pool
    score = 0.5 * (prediction[..., 0] + prediction[..., 1])
    gain = score - score[row, formal, None]
    if fallback:
        candidate_score = np.where(eligible, gain, -np.inf)
        candidate_score[row, formal] = 0.0
        selected = np.argmax(candidate_score, axis=1).astype(np.int64)
        return np.where(
            candidate_score[row, selected] > 0.0,
            selected,
            formal,
        ).astype(np.int64)
    challenger = eligible.copy()
    challenger[row, formal] = False
    candidate_score = np.where(challenger, gain, -np.inf)
    selected = np.argmax(candidate_score, axis=1).astype(np.int64)
    has_challenger = challenger.any(axis=1)
    return np.where(has_challenger, selected, formal).astype(np.int64)


def summarize_policy(
    selected: np.ndarray,
    table: dict[str, np.ndarray],
    *,
    draws: int,
    seed: int,
    outer_fold: np.ndarray,
) -> dict[str, Any]:
    return {
        "overall": residual.policy_metrics(selected, table),
        "route_cluster_bootstrap": residual.route_bootstrap(
            selected, table, draws, seed
        ),
        "outer_folds": residual.grouped_metrics(
            selected,
            {**table, "outer_fold": outer_fold},
            "outer_fold",
        ),
        "routes": {
            key: value
            for key, value in residual.grouped_metrics(
                selected, table, "route_group"
            ).items()
            if key != "groups"
        },
    }


def metric_gain(
    selected: np.ndarray,
    table: dict[str, np.ndarray],
    metric_index: int,
) -> np.ndarray:
    row = np.arange(len(selected))
    formal = table["formal_v70_index"].astype(np.int64)
    metrics = table["true_metrics_m"].astype(np.float64)
    return metrics[row, formal, metric_index] - metrics[row, selected, metric_index]


def paired_route_bootstrap(
    first: np.ndarray,
    second: np.ndarray,
    table: dict[str, np.ndarray],
    *,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    route = table["route_group"].astype(str)
    unique = np.asarray(sorted(np.unique(route).tolist()))
    rng = np.random.default_rng(seed)
    output: dict[str, Any] = {"routes": int(len(unique)), "draws": int(draws)}
    for metric_index, name in ((0, "ate"), (3, "objective")):
        delta = metric_gain(first, table, metric_index) - metric_gain(
            second, table, metric_index
        )
        route_delta = [delta[route == value] for value in unique]
        sampled_mean = np.empty(draws, dtype=np.float64)
        for draw in range(draws):
            sampled = rng.integers(0, len(unique), size=len(unique))
            sampled_mean[draw] = np.concatenate(
                [route_delta[index] for index in sampled]
            ).mean()
        output[f"{name}_gain_delta_m"] = {
            "mean": float(delta.mean()),
            "lcb95": float(np.quantile(sampled_mean, 0.025)),
            "ucb95": float(np.quantile(sampled_mean, 0.975)),
            "probability_positive": float((sampled_mean > 0.0).mean()),
        }
    return output


def format_policy_row(name: str, value: dict[str, Any]) -> str:
    overall = value["overall"]
    bootstrap = value["route_cluster_bootstrap"]["ate_gain_m"]
    return (
        f"| {name} | {1000.0 * overall['ate_gain_m']:.2f} | "
        f"[{1000.0 * bootstrap['lcb95']:.2f}, "
        f"{1000.0 * bootstrap['ucb95']:.2f}] | "
        f"{1000.0 * overall['objective_gain_m']:.2f} | "
        f"{1000.0 * overall['ate_p90_improvement_m']:.2f} | "
        f"{100.0 * overall['coverage']:.1f}% | "
        f"{100.0 * overall['dual_rate']:.1f}% | "
        f"{100.0 * overall['harm_rate']:.1f}% |"
    )


def format_pair_row(name: str, value: dict[str, Any]) -> str:
    ate = value["ate_gain_delta_m"]
    objective = value["objective_gain_delta_m"]
    return (
        f"| {name} | {1000.0 * ate['mean']:.2f} | "
        f"[{1000.0 * ate['lcb95']:.2f}, {1000.0 * ate['ucb95']:.2f}] | "
        f"{1000.0 * objective['mean']:.2f} | "
        f"[{1000.0 * objective['lcb95']:.2f}, "
        f"{1000.0 * objective['ucb95']:.2f}] |"
    )


def build_report(result: dict[str, Any]) -> str:
    lines = [
        "# v70：DINO / 项目模型 / 选择器消融",
        "",
        "> 证据等级：**spent-development exploratory**。800 个 SCAND union65 "
        "状态的标签此前已被查看，不能当作 fresh test 或部署结论。",
        "",
        "## 实验口径",
        "",
        "- NWM-faithful H4：SCAND 间距 0.38 m，不做轨迹对齐，不补当前位姿。",
        "- 所有学习型读出均按 route 做 5-fold OOF；同一路线不会同时出现在拟合和评估侧。",
        "- 表中 ATE / Objective / p90 均为相对 formal-v70 的改善，正数越大越好。",
        "- `Action + DINO` 的 DINO 是候选无关的场景/目标上下文；`LTWM` 是本项目学习到的动作条件未来表征。",
        "",
        "## 表征消融：只看排序信号",
        "",
        "| 表征 | ATE增益 mm | route-bootstrap 95% CI mm | Objective增益 mm | ATE p90改善 mm | coverage | dual | harm |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in (
        "dino_only",
        "geometry",
        "action",
        "action_plus_dino",
        "ltwm",
        "full",
    ):
        lines.append(
            format_policy_row(
                name, result["representation"]["rank_only"][name]
            )
        )
    lines.extend(
        [
            "",
            "## 表征消融：统一套用 v4 harm-UCB 门",
            "",
            "| 表征 | ATE增益 mm | route-bootstrap 95% CI mm | Objective增益 mm | ATE p90改善 mm | coverage | dual | harm |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name in (
        "dino_only",
        "geometry",
        "action",
        "action_plus_dino",
        "ltwm",
        "full",
    ):
        lines.append(
            format_policy_row(
                name, result["representation"]["common_safe"][name]
            )
        )
    lines.extend(
        [
            "",
            "## 选择器消融：固定同一份当前模型能量",
            "",
            "| 选择策略 | ATE增益 mm | route-bootstrap 95% CI mm | Objective增益 mm | ATE p90改善 mm | coverage | dual | harm |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    selector_order = [
        "formal_v70",
        "v4_safe_selector",
        "archive_no_gate_no_fallback",
        "archive_no_gate_fallback",
        "archive_mean_gate_fallback",
        "archive_ucb_gate_no_fallback",
        "initial_k17_ucb_gate_fallback",
        "cumulative_round0_ucb_gate_fallback",
        "cumulative_round1_ucb_gate_fallback",
        "final_round_ucb_gate_fallback",
        "archive_ucb_gate_fallback_current",
    ]
    for name in selector_order:
        lines.append(format_policy_row(name, result["selector"][name]))
    lines.extend(
        [
            "",
            "## 预先声明的配对归因",
            "",
            "| 对比（前者减后者） | ΔATE增益 mm | 95% CI mm | ΔObjective增益 mm | 95% CI mm |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name, value in result["pairwise"].items():
        lines.append(format_pair_row(name, value))

    dino = result["dino_static_candidate_invariance"]
    decision = result["attribution_decision"]
    lines.extend(
        [
            "",
            "## 结论",
            "",
            f"- DINO static 在同一状态的 65 个候选间最大绝对差为 "
            f"`{dino['maximum_absolute_delta']:.3e}`，超过 1e-6 的状态数为 "
            f"`{dino['states_above_1e_6']}`。因此纯 DINO 场景特征没有候选级动作排序自由度。",
            f"- 项目 LTWM/scorer 的增量证据：**{decision['project_model_signal']}**。",
            f"- v70 选择器的增量证据：**{decision['selector_signal']}**。",
            f"- 综合判断：**{decision['overall']}**。",
            "",
            "## 边界",
            "",
            "LTWM 的输入仍来自冻结 DINO token，因此这里能回答的是“项目学习到的未来模型和选择逻辑是否提供了超出静态 DINO 的增量”，不能回答“系统是否完全不依赖 DINO”。真正的最终确认仍需把这套冻结策略一次性跑在全新 route/bag 上。",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("protocol_id") != "v70_dino_model_selector_ablation_v1":
        raise RuntimeError("unexpected ablation config")
    if config.get("status") != "frozen_before_ablation_execution":
        raise RuntimeError("ablation config is not frozen")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "result.json"
    if result_path.exists():
        raise FileExistsError(f"refusing to overwrite: {result_path}")

    paths = {
        key: resolve_input(value)
        for key, value in config["inputs"].items()
    }
    table = residual.load_npz(paths["table"])
    transfer = residual.load_npz(paths["all_mode_transfer"])
    v4 = residual.load_npz(paths["v4_oof"])
    current = residual.load_npz(paths["current_oof"])
    current_result = json.loads(
        paths["current_result"].read_text(encoding="utf-8")
    )
    record = table["record_key"].astype(str)
    for name, artifact in (
        ("transfer", transfer),
        ("v4", v4),
        ("current", current),
    ):
        if not np.array_equal(record, artifact["record_key"].astype(str)):
            raise RuntimeError(f"table/{name} record orders differ")

    states, candidates = table["physical_support_pool"].shape
    row = np.arange(states)
    formal = table["formal_v70_index"].astype(np.int64)
    valid = table["physical_support_pool"].astype(bool).copy()
    valid[row, formal] = True
    outer_fold = v4["outer_fold"].astype(np.int64)
    target = make_targets(table)
    alpha = float(config["common_calibrator"]["alpha"])
    draws = int(config["statistics"]["bootstrap_draws"])
    seed = int(config["statistics"]["bootstrap_seed"])
    threshold = float(config["selector_ablation"]["harm_threshold"])
    multiplier = float(
        config["selector_ablation"]["uncertainty_multiplier"]
    )
    harm_mean = v4["heads_mean"][..., 2].astype(np.float32)
    harm_ucb = harm_mean + multiplier * v4["heads_std"][..., 2]
    mean_gate = harm_mean <= threshold
    ucb_gate = harm_ucb <= threshold

    predictions: dict[str, np.ndarray] = {}
    model_shapes: dict[str, Any] = {}
    for name, feature in [("geometry", geometry_feature(table, formal))] + [
        (mode, mode_feature(transfer, formal, mode))
        for mode in ("action", "static", "latent", "full")
    ]:
        prediction, mean, scale, coefficient = residual.fit_ridge(
            feature, target, valid, outer_fold, alpha
        )
        predictions[name] = prediction
        model_shapes[name] = {
            "feature_dim": int(feature.shape[-1]),
            "folds": int(len(mean)),
            "coefficient_shape": list(coefficient.shape),
            "normalization_shape": {
                "mean": list(mean.shape),
                "scale": list(scale.shape),
            },
        }

    rank_only_selected: dict[str, np.ndarray] = {"dino_only": formal.copy()}
    common_safe_selected: dict[str, np.ndarray] = {"dino_only": formal.copy()}
    representation_prediction_name = {
        "geometry": "geometry",
        "action": "action",
        "action_plus_dino": "static",
        "ltwm": "latent",
        "full": "full",
    }
    for label, prediction_name in representation_prediction_name.items():
        prediction = predictions[prediction_name]
        rank_only_selected[label] = select_energy(
            prediction, valid, formal, fallback=True
        )
        common_safe_selected[label] = select_energy(
            prediction,
            valid,
            formal,
            gate=ucb_gate,
            fallback=True,
        )

    current_prediction = current["prediction"].astype(np.float32)
    source_round = table["source_round"].astype(np.int8)
    pool_masks = {
        "initial": source_round == -1,
        "through_round0": source_round <= 0,
        "through_round1": source_round <= 1,
        "final_round": source_round == 2,
        "archive": np.ones_like(valid, dtype=bool),
    }
    selector_selected = {
        "formal_v70": formal.copy(),
        "v4_safe_selector": v4["safe_selected_index"].astype(np.int64),
        "archive_no_gate_no_fallback": select_energy(
            current_prediction, valid, formal, fallback=False
        ),
        "archive_no_gate_fallback": select_energy(
            current_prediction, valid, formal, fallback=True
        ),
        "archive_mean_gate_fallback": select_energy(
            current_prediction,
            valid,
            formal,
            gate=mean_gate,
            fallback=True,
        ),
        "archive_ucb_gate_no_fallback": select_energy(
            current_prediction,
            valid,
            formal,
            gate=ucb_gate,
            fallback=False,
        ),
        "initial_k17_ucb_gate_fallback": select_energy(
            current_prediction,
            valid,
            formal,
            gate=ucb_gate,
            pool=pool_masks["initial"],
            fallback=True,
        ),
        "cumulative_round0_ucb_gate_fallback": select_energy(
            current_prediction,
            valid,
            formal,
            gate=ucb_gate,
            pool=pool_masks["through_round0"],
            fallback=True,
        ),
        "cumulative_round1_ucb_gate_fallback": select_energy(
            current_prediction,
            valid,
            formal,
            gate=ucb_gate,
            pool=pool_masks["through_round1"],
            fallback=True,
        ),
        "final_round_ucb_gate_fallback": select_energy(
            current_prediction,
            valid,
            formal,
            gate=ucb_gate,
            pool=pool_masks["final_round"],
            fallback=True,
        ),
        "archive_ucb_gate_fallback_current": select_energy(
            current_prediction,
            valid,
            formal,
            gate=ucb_gate,
            pool=pool_masks["archive"],
            fallback=True,
        ),
    }
    if not np.array_equal(
        selector_selected["archive_ucb_gate_fallback_current"],
        current["selected_index"].astype(np.int64),
    ):
        raise RuntimeError("current selector could not be reproduced exactly")

    representation = {
        "rank_only": {
            name: summarize_policy(
                selected,
                table,
                draws=draws,
                seed=seed,
                outer_fold=outer_fold,
            )
            for name, selected in rank_only_selected.items()
        },
        "common_safe": {
            name: summarize_policy(
                selected,
                table,
                draws=draws,
                seed=seed,
                outer_fold=outer_fold,
            )
            for name, selected in common_safe_selected.items()
        },
    }
    selector = {
        name: summarize_policy(
            selected,
            table,
            draws=draws,
            seed=seed,
            outer_fold=outer_fold,
        )
        for name, selected in selector_selected.items()
    }

    pair_specs = {
        "action_plus_dino_minus_action_rank_only": (
            rank_only_selected["action_plus_dino"],
            rank_only_selected["action"],
        ),
        "full_minus_action_plus_dino_rank_only": (
            rank_only_selected["full"],
            rank_only_selected["action_plus_dino"],
        ),
        "full_minus_ltwm_rank_only": (
            rank_only_selected["full"],
            rank_only_selected["ltwm"],
        ),
        "current_minus_v4_selector": (
            selector_selected["archive_ucb_gate_fallback_current"],
            selector_selected["v4_safe_selector"],
        ),
        "archive_minus_initial_k17": (
            selector_selected["archive_ucb_gate_fallback_current"],
            selector_selected["initial_k17_ucb_gate_fallback"],
        ),
        "archive_minus_final_round_only": (
            selector_selected["archive_ucb_gate_fallback_current"],
            selector_selected["final_round_ucb_gate_fallback"],
        ),
        "harm_ucb_gate_minus_no_gate": (
            selector_selected["archive_ucb_gate_fallback_current"],
            selector_selected["archive_no_gate_fallback"],
        ),
        "incumbent_fallback_minus_no_fallback": (
            selector_selected["archive_ucb_gate_fallback_current"],
            selector_selected["archive_ucb_gate_no_fallback"],
        ),
    }
    pairwise = {
        name: paired_route_bootstrap(
            first,
            second,
            table,
            draws=draws,
            seed=seed + index,
        )
        for index, (name, (first, second)) in enumerate(pair_specs.items())
    }

    dino_audit = {
        "maximum_absolute_delta": float(
            transfer["static_within_state_max_abs_delta"].max()
        ),
        "states_above_1e_6": int(
            (
                transfer["static_within_state_max_abs_delta"]
                > 1.0e-6
            ).sum()
        ),
        "candidate_specific_ranking_available": bool(
            (
                transfer["static_within_state_max_abs_delta"]
                > 1.0e-6
            ).any()
        ),
    }
    model_pair = pairwise["full_minus_action_plus_dino_rank_only"][
        "ate_gain_delta_m"
    ]
    selector_pair = pairwise["current_minus_v4_selector"][
        "ate_gain_delta_m"
    ]
    archive_initial = pairwise["archive_minus_initial_k17"][
        "ate_gain_delta_m"
    ]
    current_metrics = selector[
        "archive_ucb_gate_fallback_current"
    ]["overall"]
    v4_metrics = selector["v4_safe_selector"]["overall"]
    if model_pair["lcb95"] > 0.0:
        project_model_signal = (
            "supported in this spent replay: full adds route-robust ATE gain "
            "beyond action+DINO"
        )
    elif model_pair["mean"] > 0.0:
        project_model_signal = (
            "directionally positive but route-bootstrap interval crosses zero"
        )
    else:
        project_model_signal = (
            "not supported: full does not beat action+DINO"
        )
    selector_quality = selector_pair["mean"] > 0.0
    selector_harm_nonincrease = (
        current_metrics["harm_rate"] <= v4_metrics["harm_rate"]
    )
    if (
        selector_pair["lcb95"] > 0.0
        and archive_initial["lcb95"] > 0.0
        and selector_harm_nonincrease
    ):
        selector_signal = (
            "supported for both quality and safety versus v4/initial-pool controls"
        )
    elif selector_quality:
        selector_signal = (
            "quality contribution supported, but the strict non-increasing-harm "
            "criterion is not fully met"
        )
    else:
        selector_signal = "not supported by the predeclared comparisons"
    if (
        not dino_audit["candidate_specific_ranking_available"]
        and model_pair["mean"] > 0.0
        and selector_quality
    ):
        overall = (
            "the observed action-selection gain is not explainable by static "
            "DINO alone; it requires candidate-conditioned project signals and "
            "the selector, with safety evidence still development-only"
        )
    else:
        overall = (
            "the current replay does not cleanly separate the proposed sources; "
            "do not make an attribution claim"
        )

    prediction_path = output_dir / "ablation_predictions.npz"
    with prediction_path.open("wb") as handle:
        np.savez_compressed(
            handle,
            record_key=record,
            route_group=table["route_group"],
            outer_fold=outer_fold.astype(np.int8),
            formal_v70_index=formal.astype(np.int16),
            geometry_prediction=predictions["geometry"],
            action_prediction=predictions["action"],
            action_plus_dino_prediction=predictions["static"],
            ltwm_prediction=predictions["latent"],
            full_prediction=predictions["full"],
            **{
                f"representation_rank_only_{name}_selected_index": selected
                for name, selected in rank_only_selected.items()
            },
            **{
                f"representation_common_safe_{name}_selected_index": selected
                for name, selected in common_safe_selected.items()
            },
            **{
                f"selector_{name}_selected_index": selected
                for name, selected in selector_selected.items()
            },
        )

    result = {
        "schema": "v70_dino_model_selector_ablation_v1",
        "status": "complete_spent_development_route_oof",
        "confirmatory_claim_allowed": False,
        "nwm_metric_contract": config["nwm_contract"],
        "states": int(states),
        "candidates": int(candidates),
        "routes": int(len(np.unique(table["route_group"].astype(str)))),
        "outer_folds": int(len(np.unique(outer_fold))),
        "calibrator": {
            "alpha": alpha,
            "target_names": list(residual.TARGET_NAMES),
            "model_shapes": model_shapes,
        },
        "dino_static_candidate_invariance": dino_audit,
        "representation": representation,
        "selector": selector,
        "pairwise": pairwise,
        "attribution_decision": {
            "project_model_signal": project_model_signal,
            "selector_signal": selector_signal,
            "selector_harm_nonincrease_vs_v4": selector_harm_nonincrease,
            "overall": overall,
        },
        "current_result_consistency": {
            "selected_index_exact_match": True,
            "reported_ate_gain_m": float(
                current_result["primary_policy"]["ate_gain_m"]
            ),
            "recomputed_ate_gain_m": float(current_metrics["ate_gain_m"]),
            "absolute_difference_m": float(
                abs(
                    current_result["primary_policy"]["ate_gain_m"]
                    - current_metrics["ate_gain_m"]
                )
            ),
        },
        "claim_boundary": (
            "LTWM consumes frozen DINO tokens.  These ablations test incremental "
            "value beyond static DINO, not independence from DINO.  All union65 "
            "outcomes are spent development."
        ),
        "inputs": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in paths.items()
        },
        "artifacts": {
            "predictions": {
                "path": str(prediction_path),
                "sha256": sha256_file(prediction_path),
            },
            "config": {
                "path": str(config_path),
                "sha256": sha256_file(config_path),
            },
            "runner": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
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
                "current_ate_gain_m": current_metrics["ate_gain_m"],
                "model_increment_lcb95_m": model_pair["lcb95"],
                "selector_increment_lcb95_m": selector_pair["lcb95"],
                "overall": overall,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
