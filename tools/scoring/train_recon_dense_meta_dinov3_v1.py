#!/usr/bin/env python3
"""Train/freeze the DINOv3 port of the 019f70d5 dense meta selector.

The runner uses family-grouped OOF predictions on dense-train to choose every
base-head residual-fusion rule, promotion threshold, ridge regularization, and
static/full meta threshold.  Dense-cal labels are opened only after the full
base policy passes the frozen OOF safety gate.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from scoring.dense_meta_v1 import (  # noqa: E402
    DenseHeadConfig,
    MODES,
    build_meta_features,
    build_targets,
    choose_threshold,
    empty_prediction,
    ensemble_predict,
    family_fold,
    feature_by_mode,
    fit_fold,
    fit_ridge,
    predict,
    predict_ridge,
    residual_rows,
    selected_indices,
    selection_metrics,
)


DEFAULT_ROOT = REPO_ROOT / "outputs/recon_dense_meta_dinov3_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-features",
        type=Path,
        default=DEFAULT_ROOT / "features/dense_train_features.npz",
    )
    parser.add_argument(
        "--cal-features",
        type=Path,
        default=DEFAULT_ROOT / "features/dense_cal_features.npz",
    )
    parser.add_argument(
        "--cal-label-source",
        type=Path,
        default=Path(
            "/home/rainbow/ndwm/DWM/reports/recon_h4_dense_aggregate_incumbent_features_v1/"
            "dense_cal/aggregate_incumbent_features.npz"
        ),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_ROOT / "training"
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--hidden", type=int, default=192)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--batch-groups", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=4.0e-4)
    parser.add_argument("--weight-decay", type=float, default=3.0e-3)
    parser.add_argument("--target-temperature", type=float, default=0.05)
    parser.add_argument("--min-promotions", type=int, default=64)
    parser.add_argument("--target-dual-precision", type=float, default=0.85)
    parser.add_argument("--target-wilson-lower", type=float, default=0.80)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def load_feature_artifact(path: Path, *, labels_expected: bool) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as payload:
        required = {
            "candidates",
            "actions_h4",
            "action_features",
            "static_features",
            "latent_features",
            "ate_by_seed",
            "rtp_by_seed",
            "majority_support",
            "record_key",
            "family",
            "metadata",
        }
        if labels_expected:
            required.add("targets")
        missing = required.difference(payload.files)
        if missing:
            raise RuntimeError(f"{path} missing arrays: {sorted(missing)}")
        data = {key: payload[key].copy() for key in required}
    metadata = json.loads(str(data.pop("metadata").item()))
    if metadata.get("schema") != "recon_dense_dinov3_feature_artifact_v1":
        raise RuntimeError("DINOv3 dense feature schema mismatch")
    if bool(metadata.get("labels_in_output")) != labels_expected:
        raise RuntimeError("feature label-presence contract mismatch")
    count, candidates = data["candidates"].shape[:2]
    if candidates != 17 or data["majority_support"].shape != (count, candidates):
        raise RuntimeError("dense feature candidate contract mismatch")
    return {**data, "metadata": metadata, "path": path}


def load_cal_targets(path: Path, expected: dict[str, Any]) -> np.ndarray:
    if file_sha256(path) != expected["metadata"]["source_sha256"]:
        raise RuntimeError("cal label source differs from prelabel feature source")
    with np.load(path, allow_pickle=False) as payload:
        record_key = payload["record_key"].astype(str)
        family = payload["family"].astype(str)
        targets = payload["targets"].astype(np.float32)
    if not np.array_equal(record_key, expected["record_key"].astype(str)):
        raise RuntimeError("cal label record order mismatch")
    if not np.array_equal(family, expected["family"].astype(str)):
        raise RuntimeError("cal label family order mismatch")
    if targets.shape != (*expected["candidates"].shape[:2], 8):
        raise RuntimeError("cal target shape mismatch")
    return targets


def store_prediction(target: dict[str, np.ndarray], index: np.ndarray, value: dict[str, np.ndarray]) -> None:
    for key, array in value.items():
        target[key][index] = array


def checkpoint_payload(
    *,
    mode: str,
    fold: int,
    seed: int,
    feature_dim: int,
    config: DenseHeadConfig,
    stats: dict[str, np.ndarray],
    model: torch.nn.Module,
    train_sha256: str,
) -> dict[str, Any]:
    return {
        "schema": "recon_dense_meta_dinov3_groupfirst_member_v1",
        "mode": mode,
        "heldout_family_fold": fold,
        "seed": seed,
        "candidate_pool": "trimmed_mean75_incumbent_plus_nomad_x16",
        "input_dim": feature_dim,
        "config": asdict(config),
        "normalization": stats,
        "state_dict": model.state_dict(),
        "train_feature_sha256": train_sha256,
        "old_weights_migrated": False,
        "dinov3_features_retrained": True,
        "fixed_pool_only": True,
        "cem": False,
    }


def policy_rows(
    predictions: dict[str, dict[str, np.ndarray]],
    targets: np.ndarray,
    valid: np.ndarray,
    policy: dict[str, dict[str, Any]],
) -> dict[str, dict[str, np.ndarray]]:
    return {
        mode: residual_rows(
            predictions["action"],
            predictions[mode],
            alpha=float(policy[mode]["alpha"]),
            gate=str(policy[mode]["gate"]),
            risk=str(policy[mode]["risk"]),
            true_target=targets,
            valid=valid,
        )
        for mode in MODES
    }


def choose_residual_policy(
    action: dict[str, np.ndarray],
    residual: dict[str, np.ndarray],
    targets: np.ndarray,
    valid: np.ndarray,
    args: argparse.Namespace,
    *,
    mode: str,
) -> tuple[dict[str, Any], dict[str, np.ndarray], list[dict[str, Any]]]:
    configs = (
        [(0.0, "linear", "action")]
        if mode == "action"
        else [
            (alpha, gate, risk)
            for alpha in (0.25, 0.50, 0.75, 1.0)
            for gate in ("linear", "min_confirm", "geometric")
            for risk in ("action", "residual", "both", "blend")
        ]
    )
    candidates = []
    for alpha, gate, risk in configs:
        rows = residual_rows(
            action,
            residual,
            alpha=alpha,
            gate=gate,
            risk=risk,
            true_target=targets,
            valid=valid,
        )
        chosen, frontier = choose_threshold(
            rows,
            min_promotions=args.min_promotions,
            target_dual_precision=args.target_dual_precision,
            target_wilson_lower=args.target_wilson_lower,
        )
        candidates.append(
            {
                "alpha": alpha,
                "gate": gate,
                "risk": risk,
                "metrics": chosen,
                "frontier_points": len(frontier),
                "rows": rows,
            }
        )
    best = min(
        candidates,
        key=lambda value: (
            0 if value["metrics"]["feasible_gate"] else 1,
            value["metrics"]["net_ate_delta_per_state_m"]
            + value["metrics"]["net_objective_delta_per_state_m"],
            -value["metrics"]["promotions"],
            value["alpha"],
            value["gate"],
            value["risk"],
        ),
    )
    policy = {
        "alpha": float(best["alpha"]),
        "gate": str(best["gate"]),
        "risk": str(best["risk"]),
        "threshold": float(best["metrics"]["threshold"]),
        "train_oof_metrics": best["metrics"],
    }
    summary = [
        {
            "alpha": float(value["alpha"]),
            "gate": value["gate"],
            "risk": value["risk"],
            "metrics": value["metrics"],
            "frontier_points": value["frontier_points"],
        }
        for value in candidates
    ]
    return policy, best["rows"], summary


def policy_selections(
    rows: dict[str, dict[str, np.ndarray]], policy: dict[str, dict[str, Any]]
) -> dict[str, np.ndarray]:
    return {
        mode: selected_indices(rows[mode], float(policy[mode]["threshold"]))
        for mode in MODES
    }


def meta_target(
    selections: dict[str, np.ndarray], targets: np.ndarray
) -> np.ndarray:
    row = np.arange(len(targets))
    static = targets[row, selections["static"], :4]
    full = targets[row, selections["full"], :4]
    static_cost = static[:, 0] + 0.5 * static[:, 3]
    full_cost = full[:, 0] + 0.5 * full[:, 3]
    return (static_cost - full_cost).astype(np.float32)


def meta_selection(
    score: np.ndarray,
    threshold: float,
    selections: dict[str, np.ndarray],
) -> np.ndarray:
    return np.where(
        np.asarray(score) >= threshold,
        selections["full"],
        selections["static"],
    ).astype(np.int64)


def choose_meta_model(
    feature: np.ndarray,
    target: np.ndarray,
    fold: np.ndarray,
    selections: dict[str, np.ndarray],
    targets: np.ndarray,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], np.ndarray, dict[str, Any]]:
    candidates = []
    for regularization in (1.0, 10.0, 100.0, 1000.0, 10000.0):
        oof = np.empty(len(feature), dtype=np.float32)
        for heldout in range(args.folds):
            train_index = np.where(fold != heldout)[0]
            test_index = np.where(fold == heldout)[0]
            model = fit_ridge(
                feature[train_index],
                target[train_index],
                regularization=regularization,
            )
            oof[test_index] = predict_ridge(model, feature[test_index])
        thresholds = np.unique(
            np.concatenate(
                (
                    np.asarray([-np.inf, np.inf], dtype=np.float32),
                    oof,
                    np.asarray([0.0], dtype=np.float32),
                )
            )
        )
        rows = []
        for threshold in thresholds:
            selected = meta_selection(oof, float(threshold), selections)
            metrics = selection_metrics(selected, targets)
            feasible = bool(
                metrics["promotions"] >= args.min_promotions
                and metrics["dual_precision"] is not None
                and metrics["dual_precision"] >= args.target_dual_precision
                and metrics["dual_wilson_lower_95"] is not None
                and metrics["dual_wilson_lower_95"] >= args.target_wilson_lower
            )
            rows.append(
                {
                    "threshold": float(threshold),
                    "full_choice_fraction": float((oof >= threshold).mean()),
                    "feasible_gate": feasible,
                    "metrics": {
                        key: value
                        for key, value in metrics.items()
                        if key != "selected_delta"
                    },
                }
            )
        best = min(
            rows,
            key=lambda value: (
                0 if value["feasible_gate"] else 1,
                value["metrics"]["net_ate_delta_per_state_m"]
                + value["metrics"]["net_objective_delta_per_state_m"],
                -value["metrics"]["promotions"],
            ),
        )
        candidates.append(
            {
                "regularization": regularization,
                "best": best,
                "oof_score": oof,
                "threshold_points": len(rows),
            }
        )
    best_candidate = min(
        candidates,
        key=lambda value: (
            0 if value["best"]["feasible_gate"] else 1,
            value["best"]["metrics"]["net_ate_delta_per_state_m"]
            + value["best"]["metrics"]["net_objective_delta_per_state_m"],
            -value["best"]["metrics"]["promotions"],
            value["regularization"],
        ),
    )
    final_model = fit_ridge(
        feature,
        target,
        regularization=float(best_candidate["regularization"]),
    )
    final_model["threshold"] = float(best_candidate["best"]["threshold"])
    summary = {
        "selected": {
            "regularization": float(best_candidate["regularization"]),
            **best_candidate["best"],
        },
        "regularization_candidates": [
            {
                "regularization": float(value["regularization"]),
                "best": value["best"],
                "threshold_points": int(value["threshold_points"]),
            }
            for value in candidates
        ],
    }
    return final_model, best_candidate["oof_score"], summary


def serializable_ridge(model: dict[str, Any]) -> dict[str, Any]:
    return {
        "feature_mean": np.asarray(model["feature_mean"]).tolist(),
        "feature_scale": np.asarray(model["feature_scale"]).tolist(),
        "coefficient": np.asarray(model["coefficient"]).tolist(),
        "intercept": float(model["intercept"]),
        "regularization": float(model["regularization"]),
        "threshold": float(model["threshold"]),
    }


def metrics_json(value: dict[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if key != "selected_delta"}


def legacy_main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.folds < 3:
        raise ValueError("at least three family folds are required")
    config = DenseHeadConfig(
        hidden=args.hidden,
        dropout=args.dropout,
        epochs=args.epochs,
        batch_groups=args.batch_groups,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        target_temperature=args.target_temperature,
    )
    config.validate()
    train = load_feature_artifact(args.train_features, labels_expected=True)
    cal = load_feature_artifact(args.cal_features, labels_expected=False)
    if train["metadata"]["split"] != "dense_train" or cal["metadata"]["split"] != "dense_cal":
        raise RuntimeError("train/cal split names changed")
    train_families = train["family"].astype(str)
    cal_families = cal["family"].astype(str)
    if set(train_families) & set(cal_families):
        raise RuntimeError("dense train/cal family overlap")
    unique_families = np.unique(train_families)
    if len(unique_families) != 192:
        raise RuntimeError(f"expected 192 dense-train families, got {len(unique_families)}")
    family_to_fold = {
        family: family_fold(str(family), args.folds) for family in unique_families
    }
    fold = np.asarray([family_to_fold[value] for value in train_families], dtype=np.int64)
    fold_family_counts = [
        int(sum(value == index for value in family_to_fold.values()))
        for index in range(args.folds)
    ]
    if min(fold_family_counts) < 24:
        raise RuntimeError(f"family fold too small: {fold_family_counts}")
    target = build_targets(train["targets"], config.target_temperature)
    states, candidates = train["candidates"].shape[:2]
    args.output_dir.mkdir(parents=True)
    train_sha = file_sha256(args.train_features)

    oof_predictions: dict[str, dict[str, np.ndarray]] = {}
    members: dict[str, list[tuple[torch.nn.Module, dict[str, np.ndarray]]]] = {}
    checkpoint_records: dict[str, list[dict[str, Any]]] = {}
    artifact_arrays: dict[str, np.ndarray] = {
        "record_key": train["record_key"].astype(str),
        "family": train_families,
        "fold": fold,
    }
    for mode_index, mode in enumerate(MODES):
        feature = feature_by_mode(train, mode)
        oof = empty_prediction(states, candidates)
        members[mode] = []
        checkpoint_records[mode] = []
        for heldout in range(args.folds):
            train_index = np.where(fold != heldout)[0]
            test_index = np.where(fold == heldout)[0]
            seed = args.seed + mode_index * 1000 + heldout
            model, stats = fit_fold(
                feature,
                target,
                train_index,
                config,
                seed=seed,
                device=args.device,
            )
            prediction = predict(
                model, stats, feature, test_index, device=args.device
            )
            store_prediction(oof, test_index, prediction)
            path = args.output_dir / f"{mode}_fold{heldout}.pt"
            torch.save(
                checkpoint_payload(
                    mode=mode,
                    fold=heldout,
                    seed=seed,
                    feature_dim=feature.shape[-1],
                    config=config,
                    stats=stats,
                    model=model,
                    train_sha256=train_sha,
                ),
                path,
            )
            members[mode].append((model, stats))
            checkpoint_records[mode].append(
                {
                    "fold": heldout,
                    "path": str(path),
                    "sha256": file_sha256(path),
                }
            )
            print(canonical_json({"stage": "fit_groupfirst", "mode": mode, "fold": heldout}), flush=True)
        oof_predictions[mode] = oof
        for key, value in oof.items():
            artifact_arrays[f"{mode}_oof_{key}"] = value

    policies: dict[str, dict[str, Any]] = {}
    train_rows: dict[str, dict[str, np.ndarray]] = {}
    search: dict[str, Any] = {}
    for mode in MODES:
        policy, rows, summary = choose_residual_policy(
            oof_predictions["action"],
            oof_predictions[mode],
            train["targets"],
            train["majority_support"],
            args,
            mode=mode,
        )
        policies[mode] = policy
        train_rows[mode] = rows
        search[mode] = summary
        for key, value in rows.items():
            artifact_arrays[f"{mode}_policy_{key}"] = value

    train_selections = policy_selections(train_rows, policies)
    train_metrics = {
        mode: metrics_json(selection_metrics(train_selections[mode], train["targets"]))
        for mode in MODES
    }
    meta_feature = build_meta_features(oof_predictions, train_rows, train)
    meta_outcome = meta_target(train_selections, train["targets"])
    meta_model, meta_oof_score, meta_summary = choose_meta_model(
        meta_feature,
        meta_outcome,
        fold,
        train_selections,
        train["targets"],
        args,
    )
    train_meta_selected = meta_selection(
        meta_oof_score,
        float(meta_model["threshold"]),
        train_selections,
    )
    train_meta_metrics = metrics_json(
        selection_metrics(train_meta_selected, train["targets"])
    )
    artifact_arrays["meta_feature"] = meta_feature
    artifact_arrays["meta_target"] = meta_outcome
    artifact_arrays["meta_oof_score"] = meta_oof_score
    artifact_arrays["meta_selected_index"] = train_meta_selected
    for mode, selected in train_selections.items():
        artifact_arrays[f"{mode}_selected_index"] = selected
    oof_path = args.output_dir / "train_oof_predictions.npz"
    write_npz(oof_path, **artifact_arrays)

    full_pass = bool(policies["full"]["train_oof_metrics"]["feasible_gate"])
    pre_cal = {
        "schema": "recon_dense_meta_dinov3_pre_cal_freeze_v1",
        "status": "frozen_before_dense_cal_labels_open",
        "full_train_oof_gate_pass": full_pass,
        "policies": policies,
        "meta": {
            "model": serializable_ridge(meta_model),
            "selection": meta_summary,
            "train_oof_metrics": train_meta_metrics,
        },
        "base_train_oof_metrics": train_metrics,
        "head_config": asdict(config),
        "folds": args.folds,
        "fold_family_counts": fold_family_counts,
        "checkpoints": checkpoint_records,
        "train_feature_sha256": train_sha,
        "cal_feature_sha256": file_sha256(args.cal_features),
        "cal_labels_opened": False,
        "old_weights_migrated": False,
        "fixed_pool_only": True,
        "cem": False,
    }
    pre_cal_path = args.output_dir / "pre_cal_freeze.json"
    write_json(pre_cal_path, pre_cal)

    if not full_pass:
        audit = {
            **pre_cal,
            "status": "rejected_on_train_oof_before_dense_cal_labels",
            "cal_labels_opened": False,
            "pre_cal_freeze": str(pre_cal_path),
            "pre_cal_freeze_sha256": file_sha256(pre_cal_path),
            "oof_artifact": str(oof_path),
            "oof_artifact_sha256": file_sha256(oof_path),
        }
        write_json(args.output_dir / "audit.json", audit)
        print(json.dumps(audit, indent=2, sort_keys=True))
        return

    # First access to dense-cal outcomes occurs only after the pre-cal freeze.
    cal_targets = load_cal_targets(args.cal_label_source, cal)
    cal_predictions = {
        mode: ensemble_predict(
            members[mode], feature_by_mode(cal, mode), device=args.device
        )
        for mode in MODES
    }
    cal_rows = policy_rows(
        cal_predictions,
        cal_targets,
        cal["majority_support"],
        policies,
    )
    cal_selections = policy_selections(cal_rows, policies)
    cal_meta_feature = build_meta_features(cal_predictions, cal_rows, cal)
    cal_meta_score = predict_ridge(meta_model, cal_meta_feature)
    cal_meta_selected = meta_selection(
        cal_meta_score,
        float(meta_model["threshold"]),
        cal_selections,
    )
    cal_metrics = {
        mode: metrics_json(selection_metrics(cal_selections[mode], cal_targets))
        for mode in MODES
    }
    cal_meta_metrics = metrics_json(selection_metrics(cal_meta_selected, cal_targets))
    cal_path = args.output_dir / "dense_cal_frozen_inference.npz"
    cal_arrays: dict[str, np.ndarray] = {
        "record_key": cal["record_key"].astype(str),
        "family": cal_families,
        "meta_feature": cal_meta_feature,
        "meta_score": cal_meta_score,
        "meta_selected_index": cal_meta_selected,
    }
    for mode in MODES:
        cal_arrays[f"{mode}_selected_index"] = cal_selections[mode]
        for key, value in cal_predictions[mode].items():
            cal_arrays[f"{mode}_prediction_{key}"] = value
    write_npz(cal_path, **cal_arrays)

    cal_pass = bool(
        cal_meta_metrics["promotions"] >= args.min_promotions
        and cal_meta_metrics["dual_precision"] is not None
        and cal_meta_metrics["dual_precision"] >= args.target_dual_precision
        and cal_meta_metrics["dual_wilson_lower_95"] is not None
        and cal_meta_metrics["dual_wilson_lower_95"] >= args.target_wilson_lower
    )
    frozen = {
        "schema": "recon_dense_meta_dinov3_frozen_policy_v1",
        "status": "frozen_after_family_disjoint_dense_cal",
        "test3_inference_authorized": cal_pass,
        "train_oof": {
            "base": train_metrics,
            "meta": train_meta_metrics,
        },
        "dense_cal": {
            "base": cal_metrics,
            "meta": cal_meta_metrics,
            "gate_pass": cal_pass,
        },
        "policies": policies,
        "meta": {
            "model": serializable_ridge(meta_model),
            "selection": meta_summary,
        },
        "head_config": asdict(config),
        "folds": args.folds,
        "checkpoints": checkpoint_records,
        "provenance": {
            "train_features": str(args.train_features),
            "train_features_sha256": train_sha,
            "cal_features": str(args.cal_features),
            "cal_features_sha256": file_sha256(args.cal_features),
            "cal_label_source": str(args.cal_label_source),
            "cal_label_source_sha256": file_sha256(args.cal_label_source),
            "pre_cal_freeze": str(pre_cal_path),
            "pre_cal_freeze_sha256": file_sha256(pre_cal_path),
            "oof_artifact": str(oof_path),
            "oof_artifact_sha256": file_sha256(oof_path),
            "cal_inference": str(cal_path),
            "cal_inference_sha256": file_sha256(cal_path),
        },
        "old_weights_migrated": False,
        "dinov3_features_retrained": True,
        "fixed_pool_only": True,
        "cem": False,
        "deployment_accepted": False,
    }
    frozen_path = args.output_dir / "frozen_policy.json"
    write_json(frozen_path, frozen)
    audit = {
        **frozen,
        "frozen_policy": str(frozen_path),
        "frozen_policy_sha256": file_sha256(frozen_path),
        "residual_search": search,
    }
    write_json(args.output_dir / "audit.json", audit)
    print(json.dumps(frozen, indent=2, sort_keys=True))


def main() -> None:
    """Route the public entry point to the audited three-arm design."""
    from tools.scoring.train_recon_dense_meta_dinov3_three_arm_v1 import main as three_arm_main

    three_arm_main()


if __name__ == "__main__":
    main()
