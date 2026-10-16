#!/usr/bin/env python3
"""Strict nested whole-bank OOF training for the eleven RECON development banks.

For every outer bank, both the deployed ensemble and every inner calibration
prediction exclude the complete outer bank.  The risk gate is chosen only from
the remaining banks' inner whole-bank OOF predictions, then applied once to the
outer bank.  No confirmation/reserve artifact is read.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from scoring.dense_meta_v1 import wilson_lower  # noqa: E402
from tools.scoring.run_screen_recon_target_state_gate_v24 import (  # noqa: E402
    frozen_challenger,
)
from tools.scoring.screen_recon_target_state_gate_v24 import predict  # noqa: E402
from tools.scoring.train_recon_target_state_gate_hashfold_v29 import (  # noqa: E402
    train_fast,
)
from tools.scoring.train_recon_cem_latent_base_risk_v16 import (  # noqa: E402
    compact_base_feature,
)
from tools.scoring.train_recon_dense_meta_dinov3_v1 import (  # noqa: E402
    file_sha256,
    write_json,
    write_npz,
)


VARIANTS = ("state_universal_support", "state_motion_universal_support")
METRICS = ("ATE", "RPE", "Endpoint", "Objective")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True,
                        help="Root containing banks A-F")
    parser.add_argument("--bank-g", type=Path, required=True)
    parser.add_argument("--bank-h", type=Path, required=True)
    parser.add_argument("--bank-ijk-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--hidden-dim", type=int, default=48)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--ensemble-members", type=int, default=3)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--seed", type=int, default=20261013)
    return parser.parse_args()


def load_bank(name: str, feature: Path, score: Path, source: Path) -> dict:
    compact, valid = compact_base_feature(
        feature, score, np.empty((100, 0), np.float32)
    )
    with np.load(source, allow_pickle=False) as payload:
        record = payload["record_key"].astype(str)
        route = payload["route_group"].astype(str)
        targets = payload["targets"].astype(np.float32)
        absolute = payload["absolute_metrics"].astype(np.float32)
    with np.load(score, allow_pickle=False) as payload:
        score_record = payload["record_key"].astype(str)
        scores = {
            "std": payload["std"][:, 1:].astype(np.float32),
            "dual": payload["dual_probability"][:, 1:].astype(np.float32),
            "harm": payload["harm_probability"][:, 1:].astype(np.float32),
        }
    if len(record) != 100 or not np.array_equal(record, score_record):
        raise RuntimeError(f"identity/size mismatch for {name}")
    challenger = frozen_challenger(scores, valid)
    row = np.arange(100)
    delta = targets[:, 1:, :4]
    chosen_delta = delta[row, challenger]
    dual = ((chosen_delta[:, 0] < -1e-7) &
            (chosen_delta[:, 3] < -1e-7)).astype(np.float32)
    harm = ((chosen_delta[:, 0] > 1e-7) |
            (chosen_delta[:, 3] > 1e-7)).astype(np.float32)
    return {
        "name": name,
        "record": record,
        "route": np.asarray([f"{name}::{value}" for value in route]),
        "feature": {
            "state_universal_support": compact[row, challenger][:, 474:486],
            "state_motion_universal_support": compact[row, challenger][:, 434:486],
        },
        "challenger": challenger + 1,
        "dual": dual,
        "harm": harm,
        "delta": delta,
        "absolute": absolute,
        "feature_path": str(feature),
        "feature_sha256": file_sha256(feature),
        "score_path": str(score),
        "score_sha256": file_sha256(score),
        "source_path": str(source),
        "source_sha256": file_sha256(source),
    }


def load_banks(args: argparse.Namespace) -> list[dict]:
    banks: list[dict] = []
    for letter in "abcdef":
        root = args.root / f"bank_{letter}"
        feature = next((root / "features").glob(
            f"recon_target_bank_{letter}_v*_features.npz"
        ))
        banks.append(load_bank(
            f"bank_{letter}", feature, root / "universal_scores.npz",
            root / "source_labeled.npz"
        ))
    banks.append(load_bank(
        "bank_g",
        args.bank_g / "features/recon_target_confirmation_v29_features.npz",
        args.bank_g / "universal_scores.npz",
        args.bank_g / "source_labeled.npz",
    ))
    banks.append(load_bank(
        "bank_h",
        args.bank_h / "features/recon_target_cem_confirmation_v35_features.npz",
        args.bank_h / "universal_scores.npz",
        args.bank_h / "source_labeled.npz",
    ))
    for letter in "ijk":
        root = args.bank_ijk_root / f"bank_{letter}"
        banks.append(load_bank(
            f"bank_{letter}",
            root / "features" / f"recon_target_bank_{letter}_v40_features.npz",
            root / "universal_scores.npz",
            root / "source_labeled.npz",
        ))
    return banks


def safety_quality(selected: np.ndarray, bank: dict) -> dict:
    promote = selected != 0
    count = int(promote.sum())
    row = np.arange(100)
    chosen_delta = np.zeros((100, 4), np.float32)
    chosen_delta[promote] = bank["delta"][
        row[promote], selected[promote] - 1
    ]
    dual = int(((chosen_delta[promote, 0] < -1e-7) &
                (chosen_delta[promote, 3] < -1e-7)).sum())
    harm = int(((chosen_delta[promote, 0] > 1e-7) |
                (chosen_delta[promote, 3] > 1e-7)).sum())
    chosen_absolute = bank["absolute"][row, selected]
    quality = {
        metric: {
            "mean": float(np.mean(chosen_absolute[:, index])),
            "p90": float(np.quantile(chosen_absolute[:, index], 0.9)),
        }
        for index, metric in enumerate(METRICS)
    }
    return {
        "promotions": count,
        "coverage": count / 100.0,
        "dual": dual / count if count else None,
        "wilson": wilson_lower(dual, count),
        "harm": harm / count if count else None,
        "quality": quality,
    }


def passes(metrics: dict) -> bool:
    return bool(
        metrics["coverage"] >= 0.40
        and metrics["dual"] is not None and metrics["dual"] >= 0.90
        and metrics["wilson"] is not None and metrics["wilson"] >= 0.80
        and metrics["harm"] is not None and metrics["harm"] <= 0.10
    )


def train_predict(
    x: np.ndarray,
    dual: np.ndarray,
    harm: np.ndarray,
    groups: np.ndarray,
    train_index: np.ndarray,
    predict_index: np.ndarray,
    args: argparse.Namespace,
    seed_base: int,
    checkpoint_prefix: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    dual_members, harm_members, saved = [], [], []
    for member in range(args.ensemble_members):
        seed = seed_base + member * 1009
        model, mean, scale = train_fast(
            x, dual, harm, groups, train_index, args, seed
        )
        pred_dual, pred_harm = predict(
            model, mean, scale, x[predict_index], args.device
        )
        dual_members.append(pred_dual)
        harm_members.append(pred_harm)
        if checkpoint_prefix is not None:
            path = Path(f"{checkpoint_prefix}_member_{member}.pt")
            torch.save({
                "schema": "recon_target_nested_hashfold_v40_member",
                "input_dim": x.shape[1],
                "hidden_dim": args.hidden_dim,
                "dropout": args.dropout,
                "feature_mean": mean,
                "feature_scale": scale,
                "model": {
                    key: value.detach().cpu()
                    for key, value in model.state_dict().items()
                },
                "seed": seed,
            }, path)
            saved.append({
                "path": str(path), "sha256": file_sha256(path), "seed": seed
            })
        del model
    dual_stack = np.stack(dual_members)
    harm_stack = np.stack(harm_members)
    return (
        dual_stack.mean(0), harm_stack.mean(0),
        dual_stack.std(0), harm_stack.std(0), saved,
    )


def choose_inner_gate(
    inner_prediction: dict[str, dict[str, np.ndarray]],
    banks_by_name: dict[str, dict],
) -> tuple[dict, list[dict]]:
    feasible, diagnostics = [], []
    inner_names = list(inner_prediction)
    for blend in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        for harm_weight in (0.25, 0.5, 1.0, 2.0, 4.0):
            risks = {}
            for name in inner_names:
                value = inner_prediction[name]
                risks[name] = (
                    blend * value["state_universal_support_dual"]
                    + (1.0 - blend) * value["state_motion_universal_support_dual"]
                    - harm_weight * (
                        blend * value["state_universal_support_harm"]
                        + (1.0 - blend) * value["state_motion_universal_support_harm"]
                    )
                )
            thresholds = np.unique(np.concatenate(list(risks.values())))
            for threshold in thresholds:
                per_bank = {}
                for name in inner_names:
                    bank = banks_by_name[name]
                    selected = np.where(
                        risks[name] > threshold, bank["challenger"], 0
                    )
                    per_bank[name] = safety_quality(selected, bank)
                coverage_ok = all(
                    value["coverage"] >= 0.40 for value in per_bank.values()
                )
                all_safe = coverage_ok and all(
                    passes(value) for value in per_bank.values()
                )
                candidate = {
                    "blend": blend,
                    "harm_weight": harm_weight,
                    "threshold": float(threshold),
                    "inner_banks": per_bank,
                    "coverage_sum": float(sum(
                        value["coverage"] for value in per_bank.values()
                    )),
                    "min_dual": float(min(
                        value["dual"] or 0.0 for value in per_bank.values()
                    )),
                    "min_wilson": float(min(
                        value["wilson"] or 0.0 for value in per_bank.values()
                    )),
                    "max_harm": float(max(
                        value["harm"] if value["harm"] is not None else 1.0
                        for value in per_bank.values()
                    )),
                    "inner_feasible": all_safe,
                }
                if all_safe:
                    feasible.append(candidate)
                if coverage_ok:
                    diagnostics.append(candidate)
    if feasible:
        best = max(
            feasible,
            key=lambda value: (
                value["coverage_sum"], value["min_wilson"],
                value["min_dual"], -value["max_harm"]
            ),
        )
    elif diagnostics:
        best = max(
            diagnostics,
            key=lambda value: (
                value["min_wilson"], value["min_dual"],
                -value["max_harm"], value["coverage_sum"]
            ),
        )
    else:
        raise RuntimeError("no inner policy reaches 0.40 coverage on every bank")
    return best, feasible


def main() -> None:
    args = parse_args()
    result_path = args.output_dir / "result.json"
    if result_path.exists():
        raise FileExistsError(result_path)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    banks = load_banks(args)
    names = [bank["name"] for bank in banks]
    banks_by_name = {bank["name"]: bank for bank in banks}
    cohort = np.concatenate([
        np.full(100, name, dtype="<U16") for name in names
    ])
    groups = np.concatenate([bank["route"] for bank in banks])
    dual = np.concatenate([bank["dual"] for bank in banks])
    harm = np.concatenate([bank["harm"] for bank in banks])
    record = np.concatenate([bank["record"] for bank in banks])
    features = {
        variant: np.concatenate([bank["feature"][variant] for bank in banks])
        for variant in VARIANTS
    }
    selected_oof = np.zeros(len(record), np.int64)
    outer_reports = {}
    outer_checkpoints = {}
    inner_prediction_artifacts = {}
    for outer_index, outer_name in enumerate(names):
        outer_mask = cohort == outer_name
        outer_ids = np.flatnonzero(outer_mask)
        train_outer = np.flatnonzero(~outer_mask)
        outer_prediction: dict[str, np.ndarray] = {}
        inner_prediction = {
            name: {} for name in names if name != outer_name
        }
        outer_checkpoints[outer_name] = {}
        for variant_index, variant in enumerate(VARIANTS):
            x = features[variant]
            seed_root = args.seed + outer_index * 10_000_000 + variant_index * 1_000_000
            d, h, ds, hs, saved = train_predict(
                x, dual, harm, groups, train_outer, outer_ids, args,
                seed_root,
                args.output_dir / f"{variant}_exclude_{outer_name}",
            )
            outer_prediction[f"{variant}_dual"] = d
            outer_prediction[f"{variant}_harm"] = h
            outer_prediction[f"{variant}_dual_std"] = ds
            outer_prediction[f"{variant}_harm_std"] = hs
            outer_checkpoints[outer_name][variant] = saved
            inner_names = [name for name in names if name != outer_name]
            for inner_index, inner_name in enumerate(inner_names):
                predict_ids = np.flatnonzero(cohort == inner_name)
                train_ids = np.flatnonzero(
                    (cohort != outer_name) & (cohort != inner_name)
                )
                inner_seed = seed_root + (inner_index + 1) * 100_000
                idual, iharm, _, _, _ = train_predict(
                    x, dual, harm, groups, train_ids, predict_ids, args,
                    inner_seed,
                )
                inner_prediction[inner_name][f"{variant}_dual"] = idual
                inner_prediction[inner_name][f"{variant}_harm"] = iharm
        prediction_path = args.output_dir / f"inner_exclude_{outer_name}.npz"
        arrays = {"outer_excluded": np.asarray(outer_name)}
        for inner_name, values in inner_prediction.items():
            for key, value in values.items():
                arrays[f"{inner_name}__{key}"] = value
        write_npz(prediction_path, **arrays)
        inner_prediction_artifacts[outer_name] = {
            "path": str(prediction_path),
            "sha256": file_sha256(prediction_path),
        }
        policy, feasible = choose_inner_gate(inner_prediction, banks_by_name)
        blend = policy["blend"]
        harm_weight = policy["harm_weight"]
        outer_risk = (
            blend * outer_prediction["state_universal_support_dual"]
            + (1.0 - blend) * outer_prediction["state_motion_universal_support_dual"]
            - harm_weight * (
                blend * outer_prediction["state_universal_support_harm"]
                + (1.0 - blend) * outer_prediction["state_motion_universal_support_harm"]
            )
        )
        bank = banks_by_name[outer_name]
        selected = np.where(
            outer_risk > policy["threshold"], bank["challenger"], 0
        )
        selected_oof[outer_ids] = selected
        outer_metrics = safety_quality(selected, bank)
        outer_reports[outer_name] = {
            "inner_feasible_policy_count": len(feasible),
            "policy": policy,
            "outer": outer_metrics,
            "outer_pass": passes(outer_metrics),
        }
        print(json.dumps({
            "outer": outer_name,
            "inner_feasible": policy["inner_feasible"],
            "inner_feasible_policy_count": len(feasible),
            "coverage": outer_metrics["coverage"],
            "dual": outer_metrics["dual"],
            "wilson": outer_metrics["wilson"],
            "harm": outer_metrics["harm"],
        }, sort_keys=True), flush=True)
    selected_path = args.output_dir / "selected_index_oof.npz"
    write_npz(
        selected_path, record_key=record, cohort=cohort,
        selected_index=selected_oof,
    )
    worst = {
        "coverage": min(
            value["outer"]["coverage"] for value in outer_reports.values()
        ),
        "dual": min(
            value["outer"]["dual"] or 0.0 for value in outer_reports.values()
        ),
        "wilson": min(
            value["outer"]["wilson"] or 0.0 for value in outer_reports.values()
        ),
        "harm": max(
            value["outer"]["harm"] if value["outer"]["harm"] is not None else 1.0
            for value in outer_reports.values()
        ),
    }
    report = {
        "schema": "recon_target_nested_hashfold_v40",
        "status": "eleven_bank_strict_nested_whole_bank_oof_complete",
        "protocol": (
            "outer bank excluded from model and gate; inner whole-bank OOF "
            "among remaining ten banks selects fixed causal threshold"
        ),
        "development_only": True,
        "formal_claim_allowed": False,
        "cem": False,
        "banks": names,
        "states": len(record),
        "routes": len(set(groups.tolist())),
        "gates": {
            "coverage": 0.40, "dual": 0.90,
            "wilson": 0.80, "harm": 0.10,
        },
        "inputs": {
            bank["name"]: {
                key: bank[key] for key in (
                    "feature_path", "feature_sha256", "score_path",
                    "score_sha256", "source_path", "source_sha256"
                )
            }
            for bank in banks
        },
        "outer_checkpoints": outer_checkpoints,
        "inner_prediction_artifacts": inner_prediction_artifacts,
        "outer_folds": outer_reports,
        "worst_fold": worst,
        "all_outer_pass": all(
            value["outer_pass"] and value["policy"]["inner_feasible"]
            for value in outer_reports.values()
        ),
        "selected_index": {
            "path": str(selected_path),
            "sha256": file_sha256(selected_path),
        },
        "reserve_accessed": False,
    }
    write_json(result_path, report)
    print(json.dumps({
        "all_outer_pass": report["all_outer_pass"],
        "worst_fold": worst,
        "selected_index_sha256": report["selected_index"]["sha256"],
        "result": str(result_path),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
