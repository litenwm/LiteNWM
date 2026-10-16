#!/usr/bin/env python3
"""Strict hard/easy representation screen with the existing v42 ranker.

``freeze`` opens only preregistered fit/cal development banks, trains the
unchanged v42 Ranker, fixes a calibration-derived threshold, and writes outer
selected indices. ``evaluate`` validates that freeze and only then opens the
single outer development bank.  No final confirmation artifact is read.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scoring.dense_meta_v1 import wilson_lower  # noqa: E402
from tools.scoring.train_recon_dense_meta_dinov3_v1 import (  # noqa: E402
    file_sha256,
    write_json,
    write_npz,
)
from tools.scoring.train_recon_target_full_listwise_v42 import (  # noqa: E402
    Config,
    METRICS,
    predict_model,
    train_model,
)


PARTITIONS = {
    "hard_t": {
        "fit": list("abcdefghijklmn"),
        "cal": list("opqrs"),
        "outer": "t",
    },
    "easy_d": {
        "fit": [x for x in "abcdefghijklmno" if x != "d"],
        "cal": list("pqrst"),
        "outer": "d",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("freeze", "evaluate"), required=True)
    parser.add_argument("--rollout", type=Path, required=True)
    parser.add_argument("--fold", choices=tuple(PARTITIONS), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


def bank_roots() -> dict[str, Path]:
    return {
        **{x: ROOT / "outputs/cem_development_v1/recon_target_calibration_v20" / f"bank_{x}" for x in "abcdef"},
        "g": ROOT / "outputs/cem_development_v1/recon_target_confirmation_v29",
        "h": ROOT / "outputs/cem_development_v1/recon_target_cem_confirmation_v35",
        **{x: ROOT / "outputs/cem_development_v1/recon_target_development_v40" / f"bank_{x}" for x in "ijk"},
        **{x: ROOT / "outputs/cem_development_v1/recon_target_development_v44" / f"bank_{x}" for x in "lmnop"},
        **{x: ROOT / "outputs/cem_development_v1/recon_target_development_v46" / f"bank_{x}" for x in "qrst"},
    }


def sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -30.0, 30.0)))


def load_rollout(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        metadata = json.loads(str(payload["metadata"]))
        if metadata.get("outcomes_accessed") or metadata.get("labels_in_output"):
            raise RuntimeError("rollout artifact is not outcome-free")
        data = {key: payload[key].copy() for key in payload.files if key != "metadata"}
    if data["record_key"].shape != (2000,) or data["bank"].shape != (2000,):
        raise RuntimeError("A-T rollout identity contract changed")
    if not np.array_equal(data["bank"].astype(str), np.repeat(list("abcdefghijklmnopqrst"), 100)):
        raise RuntimeError("A-T bank order changed")
    static = data["static_features"][:, 1:].astype(np.float32)
    raw = data["raw_ltwm_features"].astype(np.float32)
    action = data["actions_h4"].astype(np.float32)
    candidate = data["candidates"].astype(np.float32)
    universal = np.concatenate(
        (
            data["universal_mean"][:, 1:].astype(np.float32),
            data["universal_std"][:, 1:].astype(np.float32),
            data["universal_dual_probability"][:, 1:, None].astype(np.float32),
            data["universal_harm_probability"][:, 1:, None].astype(np.float32),
            data["universal_rank_logit"][:, 1:, None].astype(np.float32),
        ), axis=2,
    )
    action_candidate = action[:, 1:]
    path_candidate = candidate[:, 1:]
    support = data["majority_support"][:, 1:].astype(bool)
    compact = np.concatenate(
        (
            static,
            raw[:, 1:], raw[:, 1:] - raw[:, :1],
            action_candidate.reshape(2000, 16, -1),
            (action_candidate - action[:, :1]).reshape(2000, 16, -1),
            path_candidate.reshape(2000, 16, -1),
            (path_candidate - candidate[:, :1]).reshape(2000, 16, -1),
            universal,
            support[..., None].astype(np.float32),
        ), axis=2,
    ).astype(np.float32)
    spatial = data["spatial_rollout_features"][:, 1:].astype(np.float32)
    x = np.concatenate((compact, spatial), axis=2).astype(np.float32)
    if compact.shape != (2000, 16, 486) or x.shape != (2000, 16, 744):
        raise RuntimeError(f"existing v42 feature contract changed: {compact.shape}, {x.shape}")
    if not np.isfinite(x).all():
        raise RuntimeError("non-finite representation feature")
    return {**data, "x": x, "valid": support}


def load_outcome(letter: str, rollout: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    path = bank_roots()[letter] / "source_labeled.npz"
    with np.load(path, allow_pickle=False) as payload:
        record = payload["record_key"].astype(str)
        route = payload["route_group"].astype(str)
        targets = payload["targets"].astype(np.float32)
        absolute = payload["absolute_metrics"].astype(np.float32)
    index = np.flatnonzero(rollout["bank"].astype(str) == letter)
    if len(index) != 100 or not np.array_equal(record, rollout["record_key"][index].astype(str)):
        raise RuntimeError(f"outer/outcome identity mismatch for bank {letter}")
    delta = targets[:, 1:, :4]
    dual = ((delta[..., 0] < -1e-7) & (delta[..., 3] < -1e-7)).astype(np.float32)
    harm = ((delta[..., 0] > 1e-7) | (delta[..., 3] > 1e-7)).astype(np.float32)
    return {
        "letter": np.asarray(letter), "path": np.asarray(str(path)),
        "index": index, "record": record, "route": route,
        "delta": delta, "dual": dual, "harm": harm, "absolute": absolute,
    }


def selection_metrics(selected: np.ndarray, outcome: dict[str, np.ndarray]) -> dict[str, object]:
    promote = selected != 0
    count = int(promote.sum())
    row = np.arange(100)
    chosen_delta = np.zeros((100, 4), np.float32)
    chosen_delta[promote] = outcome["delta"][row[promote], selected[promote] - 1]
    dual = int(((chosen_delta[promote, 0] < -1e-7) & (chosen_delta[promote, 3] < -1e-7)).sum())
    harm = int(((chosen_delta[promote, 0] > 1e-7) | (chosen_delta[promote, 3] > 1e-7)).sum())
    chosen_absolute = outcome["absolute"][row, selected]
    quality = {
        name: {
            "mean": float(chosen_absolute[:, i].mean()),
            "p90": float(np.quantile(chosen_absolute[:, i], 0.90)),
        }
        for i, name in enumerate(METRICS)
    }
    return {
        "promotions": count, "coverage": count / 100.0,
        "dual": dual / count if count else None,
        "wilson": wilson_lower(dual, count),
        "harm": harm / count if count else None,
        "quality": quality,
    }


def safety_pass(value: dict[str, object]) -> bool:
    return bool(
        value["coverage"] >= 0.40
        and value["dual"] is not None and value["dual"] >= 0.90
        and value["wilson"] is not None and value["wilson"] >= 0.80
        and value["harm"] is not None and value["harm"] <= 0.10
    )


def quality_pass(value: dict[str, object], incumbent: dict[str, object]) -> bool:
    return all(
        value["quality"][name][stat] <= incumbent["quality"][name][stat] + 1e-12
        for name in METRICS for stat in ("mean", "p90")
    )


def probabilities(members: np.ndarray) -> dict[str, np.ndarray]:
    mean = members.mean(0)
    std = members.std(0)
    dual_members = sigmoid(members[..., 2])
    harm_members = sigmoid(members[..., 3])
    return {
        "pred_delta": mean[..., :2],
        "pred_delta_std": std[..., :2],
        "p_dual": dual_members.mean(0), "p_harm": harm_members.mean(0),
        "p_dual_std": dual_members.std(0), "p_harm_std": harm_members.std(0),
    }


def score_policy(pred: dict[str, np.ndarray], valid: np.ndarray, policy: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    score = (
        pred["p_dual"] - policy["harm_weight"] * pred["p_harm"]
        + policy["utility_weight"] * (-pred["pred_delta"][..., 0] - pred["pred_delta"][..., 1])
        - policy["uncertainty_scale"] * (
            pred["p_dual_std"] + pred["p_harm_std"] + pred["pred_delta_std"].sum(-1)
        )
    )
    score = np.where(valid, score, -np.inf)
    best = score.argmax(1)
    return best, score[np.arange(len(score)), best]


def choose_gate(
    prediction: dict[str, np.ndarray], valid: np.ndarray,
    cal_outcomes: list[dict[str, np.ndarray]],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    feasible, diagnostics = [], []
    for harm_weight in (0.25, 0.5, 1.0, 2.0, 4.0):
        for utility_weight in (0.0, 0.5, 1.0, 2.0, 4.0):
            for uncertainty_scale in (0.0, 0.5, 1.0, 2.0):
                policy = {"harm_weight": harm_weight, "utility_weight": utility_weight, "uncertainty_scale": uncertainty_scale}
                best, confidence = score_policy(prediction, valid, policy)
                for threshold in np.unique(confidence):
                    per, coverage_ok = {}, True
                    for bank_index, outcome in enumerate(cal_outcomes):
                        local = slice(bank_index * 100, (bank_index + 1) * 100)
                        selected = np.where(confidence[local] > threshold, best[local] + 1, 0)
                        value = selection_metrics(selected, outcome)
                        incumbent = selection_metrics(np.zeros(100, np.int64), outcome)
                        value["quality_non_degradation"] = quality_pass(value, incumbent)
                        per[str(outcome["letter"])] = value
                        coverage_ok = coverage_ok and value["coverage"] >= 0.40
                    if not coverage_ok:
                        continue
                    candidate = {
                        **policy, "threshold": float(threshold), "cal_banks": per,
                        "all_safety_pass": all(safety_pass(v) for v in per.values()),
                        "all_quality_pass": all(v["quality_non_degradation"] for v in per.values()),
                        "coverage_sum": float(sum(v["coverage"] for v in per.values())),
                        "min_wilson": float(min(v["wilson"] or 0.0 for v in per.values())),
                        "min_dual": float(min(v["dual"] or 0.0 for v in per.values())),
                        "max_harm": float(max(v["harm"] if v["harm"] is not None else 1.0 for v in per.values())),
                    }
                    diagnostics.append(candidate)
                    if candidate["all_safety_pass"] and candidate["all_quality_pass"]:
                        feasible.append(candidate)
    pool = feasible if feasible else diagnostics
    if not pool:
        raise RuntimeError("no calibration threshold reaches 0.40 coverage on every cal bank")
    best = max(pool, key=lambda v: (v["all_safety_pass"], v["all_quality_pass"], v["min_wilson"], v["min_dual"], -v["max_harm"], v["coverage_sum"]))
    return best, feasible


def freeze(args: argparse.Namespace) -> None:
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    rollout = load_rollout(args.rollout)
    partition = PARTITIONS[args.fold]
    # Only non-outer labels are opened in this stage.
    fit_outcomes = [load_outcome(letter, rollout) for letter in partition["fit"]]
    cal_outcomes = [load_outcome(letter, rollout) for letter in partition["cal"]]
    fit_index = np.concatenate([value["index"] for value in fit_outcomes])
    cal_index = np.concatenate([value["index"] for value in cal_outcomes])
    outer_index = np.flatnonzero(rollout["bank"].astype(str) == partition["outer"])
    delta = np.zeros((2000, 16, 4), np.float32)
    dual = np.zeros((2000, 16), np.float32)
    harm = np.zeros((2000, 16), np.float32)
    route = np.asarray([f"unopened::{i}" for i in range(2000)])
    for value in fit_outcomes:
        delta[value["index"]] = value["delta"]
        dual[value["index"]] = value["dual"]
        harm[value["index"]] = value["harm"]
        route[value["index"]] = np.asarray([f"{value['letter']}::{x}" for x in value["route"]])
    cohort = rollout["bank"].astype(str)
    config_args = argparse.Namespace(
        hidden_dim=160, dropout=0.30, epochs=180,
        learning_rate=0.0005, weight_decay=0.002,
        device=args.device,
    )
    cal_members, outer_members, checkpoints = [], [], []
    for member in range(3):
        seed = 20261101 + member * 1009
        model, mean, scale, loss = train_model(
            rollout["x"], rollout["valid"], delta, dual, harm,
            cohort, route, fit_index, config_args, seed,
        )
        cal_members.append(predict_model(model, mean, scale, rollout["x"][cal_index], args.device))
        outer_members.append(predict_model(model, mean, scale, rollout["x"][outer_index], args.device))
        path = args.output_dir / f"member_{member}.pt"
        torch.save({
            "schema": "recon_target_representation_v51_v42_member",
            "fold": args.fold, "outer_bank": partition["outer"],
            "config": asdict(Config(744, 160, 0.30)),
            "feature_mean": mean, "feature_scale": scale,
            "model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "seed": seed, "loss": loss,
        }, path)
        checkpoints.append({"path": str(path), "sha256": file_sha256(path), "seed": seed})
        del model
    cal_prediction = probabilities(np.stack(cal_members))
    outer_prediction = probabilities(np.stack(outer_members))
    policy, feasible = choose_gate(cal_prediction, rollout["valid"][cal_index], cal_outcomes)
    outer_best, outer_confidence = score_policy(outer_prediction, rollout["valid"][outer_index], policy)
    selected = np.where(outer_confidence > policy["threshold"], outer_best + 1, 0).astype(np.int64)
    selection_path = args.output_dir / "selected_index.npz"
    write_npz(selection_path, record_key=rollout["record_key"][outer_index], selected_index=selected, confidence=outer_confidence)
    freeze_payload = {
        "schema": "recon_target_representation_v51_freeze",
        "status": "outer_selection_frozen_before_outcome_open",
        "fold": args.fold, "partition": partition,
        "rollout": str(args.rollout), "rollout_sha256": file_sha256(args.rollout),
        "ranker": "unchanged existing v42 Ranker/train_model",
        "architecture_changes": False,
        "input_dim": 744,
        "config": vars(config_args) | {"ensemble_members": 3, "seed": 20261101},
        "checkpoints": checkpoints,
        "policy": policy,
        "cal_feasible_count": len(feasible),
        "selected_index": str(selection_path),
        "selected_index_sha256": file_sha256(selection_path),
        "outer_promotions_frozen": int((selected != 0).sum()),
        "outer_outcomes_accessed": False,
        "final_confirmation_reserve_accessed": False,
        "cem": False,
    }
    write_json(args.output_dir / "freeze.json", freeze_payload)
    print(json.dumps(freeze_payload, indent=2, sort_keys=True))


def evaluate(args: argparse.Namespace) -> None:
    freeze_path = args.output_dir / "freeze.json"
    result_path = args.output_dir / "result.json"
    if result_path.exists():
        raise FileExistsError(result_path)
    frozen = json.loads(freeze_path.read_text(encoding="utf-8"))
    if frozen["rollout_sha256"] != file_sha256(args.rollout):
        raise RuntimeError("rollout changed after freeze")
    selection_path = Path(frozen["selected_index"])
    if frozen["selected_index_sha256"] != file_sha256(selection_path):
        raise RuntimeError("selected_index changed after freeze")
    rollout = load_rollout(args.rollout)
    with np.load(selection_path, allow_pickle=False) as payload:
        selected = payload["selected_index"].astype(np.int64)
        selected_record = payload["record_key"].astype(str)
    outer = load_outcome(PARTITIONS[args.fold]["outer"], rollout)
    if not np.array_equal(selected_record, outer["record"]):
        raise RuntimeError("frozen selection/outcome identity mismatch")
    selected_metrics = selection_metrics(selected, outer)
    incumbent_metrics = selection_metrics(np.zeros(100, np.int64), outer)
    selected_metrics["quality_non_degradation"] = quality_pass(selected_metrics, incumbent_metrics)
    result = {
        "schema": "recon_target_representation_v51_result",
        "evidence": "spent-development hard/easy outer screen",
        "fold": args.fold, "outer_bank": PARTITIONS[args.fold]["outer"],
        "freeze": str(freeze_path), "freeze_sha256": file_sha256(freeze_path),
        "selected": selected_metrics,
        "incumbent": incumbent_metrics,
        "all_gates_pass": safety_pass(selected_metrics) and selected_metrics["quality_non_degradation"],
        "outer_outcomes_opened_after_freeze": True,
        "final_confirmation_reserve_accessed": False,
        "formal_claim_allowed": False,
        "cem": False,
    }
    write_json(result_path, result)
    print(json.dumps(result, indent=2, sort_keys=True))


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.stage == "freeze":
        freeze(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
