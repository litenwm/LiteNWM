#!/usr/bin/env python3
"""Replay frozen SCAND-domain-adapted policies on the exact RECON H4-100 pool.

The command is intentionally split into ``freeze`` and ``evaluate`` stages.
The freeze stage does not open the already-spent RECON outcome labels.  The
evaluate stage verifies the frozen artifact hash before reading labels.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from scoring.dense_meta_v1 import selection_metrics  # noqa: E402
from scoring.spacing_support_v1 import support_mask  # noqa: E402
from tools.evaluation.freeze_scand_internal_holdout_selections_v1 import (  # noqa: E402
    load_members,
    load_support,
    native_selection,
    predict_modes,
    raw_selection,
    selection_audit,
)
from tools.scoring.train_recon_dense_meta_dinov3_three_arm_v1 import (  # noqa: E402
    load_features,
)
from tools.scoring.train_recon_dense_meta_dinov3_v1 import (  # noqa: E402
    file_sha256,
    write_json,
    write_npz,
)


DEFAULT_SCAND_ROOT = REPO_ROOT / "outputs/scand_domain_adaptation_v1"
DEFAULT_RECON_ROOT = REPO_ROOT / "outputs/recon_dense_meta_dinov3_v1"
METRIC_NAMES = ("ate", "rpe", "endpoint", "objective")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("freeze", "evaluate"), required=True)
    parser.add_argument(
        "--features",
        type=Path,
        default=DEFAULT_RECON_ROOT
        / "features/strict_external100_k17_exact_features.npz",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=DEFAULT_RECON_ROOT / "recon100_k17_exact/labels.npz",
    )
    parser.add_argument(
        "--old-recon-result",
        type=Path,
        default=DEFAULT_RECON_ROOT / "recon100_k17_exact/result.json",
    )
    parser.add_argument(
        "--old-recon-inference",
        type=Path,
        default=DEFAULT_RECON_ROOT / "recon100_k17_exact/frozen_inference.npz",
    )
    parser.add_argument(
        "--training-result",
        type=Path,
        default=DEFAULT_SCAND_ROOT / "training_v1/result.json",
    )
    parser.add_argument(
        "--scand-result",
        type=Path,
        default=DEFAULT_SCAND_ROOT / "fresh_outer_v1/outcomes/result.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_SCAND_ROOT / "recon100_replay_v1",
    )
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260720)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def state_bootstrap_ci(
    delta: np.ndarray, *, replicates: int, seed: int
) -> dict[str, dict[str, float]]:
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, len(delta), size=(replicates, len(delta)))
    sampled = delta[draw].mean(axis=1)
    return {
        name: {
            "mean_delta_m": float(delta[:, metric].mean()),
            "ci95_low_m": float(np.quantile(sampled[:, metric], 0.025)),
            "ci95_high_m": float(np.quantile(sampled[:, metric], 0.975)),
        }
        for metric, name in enumerate(METRIC_NAMES)
    }


def replay_metrics(
    selected: np.ndarray,
    targets: np.ndarray,
    absolute_metrics: np.ndarray,
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    selected = np.asarray(selected, dtype=np.int64)
    if selected.shape != (len(targets),):
        raise RuntimeError("selection/state count mismatch")
    if np.any(selected < 0) or np.any(selected >= targets.shape[1]):
        raise RuntimeError("selection index outside K17 pool")
    base = selection_metrics(selected, targets)
    base.pop("selected_delta", None)
    row = np.arange(len(selected))
    chosen = absolute_metrics[row, selected]
    incumbent = absolute_metrics[:, 0]
    delta = chosen - incumbent
    promoted = selected != 0
    harmed = promoted & (
        (delta[:, 0] > 1.0e-7) | (delta[:, 3] > 1.0e-7)
    )
    return {
        **base,
        "promotion_fraction": float(promoted.mean()),
        "harm_count": int(harmed.sum()),
        "harm_rate_per_promotion": (
            float(harmed.sum() / promoted.sum()) if promoted.any() else None
        ),
        "mean_harm_magnitude_m": (
            float(np.maximum(delta[harmed, 0], delta[harmed, 3]).mean())
            if harmed.any()
            else 0.0
        ),
        "max_harm_magnitude_m": (
            float(np.maximum(delta[harmed, 0], delta[harmed, 3]).max())
            if harmed.any()
            else 0.0
        ),
        "absolute": {
            name: {
                "mean_m": float(chosen[:, metric].mean()),
                "p90_m": float(np.quantile(chosen[:, metric], 0.90)),
            }
            for metric, name in enumerate(METRIC_NAMES)
        },
        "state_paired_bootstrap_ci": state_bootstrap_ci(
            delta,
            replicates=bootstrap_replicates,
            seed=bootstrap_seed,
        ),
    }


def freeze(args: argparse.Namespace) -> None:
    freeze_path = args.output_dir / "pre_label_freeze.json"
    if freeze_path.exists():
        raise FileExistsError(freeze_path)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    old_result = read_json(args.old_recon_result)
    if old_result.get("status") != "completed_frozen_recon100_k17_three_arm":
        raise RuntimeError("old RECON exact result is not the frozen three-arm result")
    if file_sha256(args.features) != old_result["features_sha256"]:
        raise RuntimeError("RECON exact feature artifact changed")
    if file_sha256(args.old_recon_inference) != old_result["inference_sha256"]:
        raise RuntimeError("old RECON frozen inference changed")

    data = load_features(args.features, labels_expected=False)
    metadata = data["metadata"]
    if not metadata["split"].startswith("strict_external100_k17_exact"):
        raise RuntimeError("not the exact RECON H4-100 feature split")
    if metadata["labels_accessed"] or metadata["labels_in_output"]:
        raise RuntimeError("freeze-stage features unexpectedly contain labels")
    if len(data["record_key"]) != 100:
        raise RuntimeError("RECON exact state count changed")

    training = read_json(args.training_result)
    if training.get("status") != "calibrated_six_arm_fixed_pool_cem_false":
        raise RuntimeError("SCAND adaptation is not frozen/calibrated")
    if training.get("cem") or training.get("cem_cal_accessed"):
        raise RuntimeError("CEM provenance is invalid")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    spacing = np.full(100, 0.25, dtype=np.float32)
    arrays: dict[str, np.ndarray] = {
        "record_key": data["record_key"].astype(str),
        "family": data["family"].astype(str),
    }
    selection_info: dict[str, Any] = {}
    ate_scale = float(training["scorer_v2"]["ate_scale"])
    rtp_scale = float(training["scorer_v2"]["rtp_scale"])

    for regime_name in ("scand_only", "domain_balanced"):
        regime = training["regimes"][regime_name]
        members = load_members(regime["training"], args.device)
        predictions = predict_modes(members, data, args.device)
        calibration = regime["calibration"]
        support = load_support(
            Path(calibration["support_checkpoint"]),
            calibration["support_checkpoint_sha256"],
        )
        regime_info: dict[str, Any] = {}
        for arm_name in ("support_only", "raw_only", "support_plus_raw"):
            policy = calibration["arms"][arm_name]
            if arm_name == "raw_only":
                valid = data["majority_support"].astype(bool)
                support_info = {
                    "source": "frozen_scorer_v2_majority_support",
                    "candidate_acceptance": float(valid[:, 1:].mean()),
                    "no_supported_proposal_windows": int(
                        (~valid[:, 1:].any(axis=1)).sum()
                    ),
                }
            else:
                valid, support_info = support_mask(
                    support,
                    data["actions_h4"],
                    spacing,
                    quantile=float(policy["support_quantile"]),
                )
            if arm_name == "support_only":
                selected = native_selection(
                    data, valid, policy, ate_scale, rtp_scale
                )
                meta_score = np.zeros(100, dtype=np.float32)
            else:
                selected, meta_score = raw_selection(
                    predictions, data, valid, policy
                )
            label = f"new_{regime_name}__{arm_name}"
            arrays[label] = selected
            arrays[f"{label}__meta_score"] = meta_score
            regime_info[arm_name] = {
                **selection_audit(selected, valid),
                "support": support_info,
            }
        selection_info[regime_name] = regime_info
        del members, predictions
        if args.device == "cuda":
            torch.cuda.empty_cache()

    with np.load(args.old_recon_inference, allow_pickle=False) as old:
        if not np.array_equal(
            old["record_key"].astype(str), arrays["record_key"]
        ):
            raise RuntimeError("old/new RECON inference order mismatch")
        arrays["old_recon_full_meta_raw"] = old[
            "dinov3_full_meta_raw_selected_index"
        ].astype(np.int64)
        arrays["old_recon_full_meta_v2_stacked"] = old[
            "dinov3_full_meta_v2_stacked_selected_index"
        ].astype(np.int64)
        arrays["native_scorer_v2"] = old[
            "dinov3_scorer_v2_native_selected_index"
        ].astype(np.int64)

    inference_path = args.output_dir / "frozen_inference.npz"
    write_npz(inference_path, **arrays)
    checkpoint_hashes = sorted(
        {
            record["sha256"]
            for regime in training["regimes"].values()
            for records in regime["training"]["checkpoints"].values()
            for record in records
        }
    )
    freeze_record = {
        "schema": "recon100_scand_adaptation_replay_prelabel_v1",
        "status": "cross_domain_selections_frozen_before_current_label_read",
        "evidence_role": "RECON100 spent developmental replay",
        "states": 100,
        "candidates": 17,
        "waypoint_spacing_m": 0.25,
        "candidate_pool": "same exact trimmed-mean75 incumbent plus frozen NoMaD x16",
        "features": str(args.features),
        "features_sha256": file_sha256(args.features),
        "training_result": str(args.training_result),
        "training_result_sha256": file_sha256(args.training_result),
        "old_recon_result": str(args.old_recon_result),
        "old_recon_result_sha256": file_sha256(args.old_recon_result),
        "old_recon_inference_sha256": file_sha256(args.old_recon_inference),
        "model_checkpoint_sha256": checkpoint_hashes,
        "inference": str(inference_path),
        "inference_sha256": file_sha256(inference_path),
        "selection_audit": selection_info,
        "scand_calibration_reused_without_recon_retuning": True,
        "recon_labels_historically_spent": True,
        "recon_labels_opened_in_freeze_stage": False,
        "fixed_pool_only": True,
        "cem": False,
    }
    write_json(freeze_path, freeze_record)
    print(json.dumps(freeze_record, ensure_ascii=False, indent=2, sort_keys=True))


def metric_pair(value: dict[str, Any], name: str) -> str:
    metric = value["absolute"][name]
    return f"{metric['mean_m']:.4f}/{metric['p90_m']:.4f}"


def summary_rows(
    recon_methods: dict[str, dict[str, Any]],
    scand_methods: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    recon_names = (
        "incumbent",
        "native_scorer_v2",
        "old_recon_full_meta_raw",
        "old_recon_full_meta_v2_stacked",
        "new_scand_only__raw_only",
        "new_scand_only__support_only",
        "new_scand_only__support_plus_raw",
        "new_domain_balanced__raw_only",
        "new_domain_balanced__support_only",
        "new_domain_balanced__support_plus_raw",
        "posthoc_objective_pool_oracle",
    )
    scand_names = (
        "incumbent",
        "native_scorer_v2",
        "scand_only__raw_only",
        "scand_only__support_only",
        "scand_only__support_plus_raw",
        "domain_balanced__raw_only",
        "domain_balanced__support_only",
        "domain_balanced__support_plus_raw",
        "posthoc_objective_pool_oracle",
    )
    rows = []
    for dataset, methods, names in (
        ("RECON H4-100 exact", recon_methods, recon_names),
        ("SCAND fresh H4-100", scand_methods, scand_names),
    ):
        for name in names:
            value = methods[name]
            rows.append(
                {
                    "dataset": dataset,
                    "method": name,
                    "promotions": value["promotions"],
                    "ate_mean_m": value["absolute"]["ate"]["mean_m"],
                    "ate_p90_m": value["absolute"]["ate"]["p90_m"],
                    "rpe_mean_m": value["absolute"]["rpe"]["mean_m"],
                    "rpe_p90_m": value["absolute"]["rpe"]["p90_m"],
                    "endpoint_mean_m": value["absolute"]["endpoint"]["mean_m"],
                    "endpoint_p90_m": value["absolute"]["endpoint"]["p90_m"],
                    "objective_mean_m": value["absolute"]["objective"]["mean_m"],
                    "objective_p90_m": value["absolute"]["objective"]["p90_m"],
                    "ate_precision": value["ate_precision"],
                    "objective_precision": value["objective_precision"],
                    "dual_precision": value["dual_precision"],
                    "dual_wilson_lower_95": value["dual_wilson_lower_95"],
                    "harm_count": value["harm_count"],
                }
            )
    return rows


def write_summary(
    output_dir: Path,
    recon_methods: dict[str, dict[str, Any]],
    scand_result: dict[str, Any],
) -> None:
    rows = summary_rows(recon_methods, scand_result["methods"])
    csv_path = output_dir / "total_results.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    aliases = {
        "incumbent": "Incumbent",
        "native_scorer_v2": "Native scorer-v2",
        "old_recon_full_meta_raw": "Old RECON full/meta-raw",
        "old_recon_full_meta_v2_stacked": "Old RECON v2-stacked",
        "new_scand_only__raw_only": "New SCAND-only raw-only",
        "new_scand_only__support_only": "New SCAND-only support-only",
        "new_scand_only__support_plus_raw": "New SCAND-only support+raw",
        "new_domain_balanced__raw_only": "New balanced raw-only",
        "new_domain_balanced__support_only": "New balanced support-only",
        "new_domain_balanced__support_plus_raw": "New balanced support+raw",
        "scand_only__raw_only": "New SCAND-only raw-only",
        "scand_only__support_only": "New SCAND-only support-only",
        "scand_only__support_plus_raw": "New SCAND-only support+raw",
        "domain_balanced__raw_only": "New balanced raw-only",
        "domain_balanced__support_only": "New balanced support-only",
        "domain_balanced__support_plus_raw": "New balanced support+raw",
        "posthoc_objective_pool_oracle": "Post-hoc pool oracle",
    }
    lines = [
        "# DINOv3/LTWM fixed-pool total results",
        "",
        "All cells are mean/p90 metres. CEM is disabled.",
    ]
    for dataset in ("RECON H4-100 exact", "SCAND fresh H4-100"):
        lines.extend(
            (
                "",
                f"## {dataset}",
                "",
                "| Method | Promotions | ATE | RPE | Endpoint | Objective | Dual/Wilson | Harm |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            )
        )
        source = recon_methods if dataset.startswith("RECON") else scand_result["methods"]
        for row in [value for value in rows if value["dataset"] == dataset]:
            value = source[row["method"]]
            dual = (
                "—"
                if value["dual_precision"] is None
                else f"{value['dual_precision']:.3f}/{value['dual_wilson_lower_95']:.3f}"
            )
            lines.append(
                "| "
                + " | ".join(
                    (
                        aliases[row["method"]],
                        str(value["promotions"]),
                        metric_pair(value, "ate"),
                        metric_pair(value, "rpe"),
                        metric_pair(value, "endpoint"),
                        metric_pair(value, "objective"),
                        dual,
                        str(value["harm_count"]),
                    )
                )
                + " |"
            )
    lines.extend(
        (
            "",
            "RECON is a historically spent developmental replay. The new SCAND-calibrated "
            "models and thresholds were applied without RECON retuning. SCAND fresh H4-100 "
            "was frozen before its first label open.",
            "",
        )
    )
    (output_dir / "TOTAL_RESULTS.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def evaluate(args: argparse.Namespace) -> None:
    if args.bootstrap_replicates < 1000:
        raise ValueError("bootstrap_replicates must be at least 1000")
    freeze_path = args.output_dir / "pre_label_freeze.json"
    frozen = read_json(freeze_path)
    if (
        frozen.get("status")
        != "cross_domain_selections_frozen_before_current_label_read"
    ):
        raise RuntimeError("RECON cross-domain selections were not frozen")
    inference_path = Path(frozen["inference"])
    if file_sha256(inference_path) != frozen["inference_sha256"]:
        raise RuntimeError("frozen inference changed before evaluation")

    with np.load(inference_path, allow_pickle=False) as inference:
        record_key = inference["record_key"].astype(str)
        selections = {
            key: inference[key].astype(np.int64)
            for key in inference.files
            if key not in {"record_key", "family"} and not key.endswith("__meta_score")
        }
    with np.load(args.labels, allow_pickle=False) as labels:
        if not np.array_equal(record_key, labels["record_key"].astype(str)):
            raise RuntimeError("RECON label order mismatch")
        targets = labels["targets"].astype(np.float32)
        absolute = labels["absolute_metrics"].astype(np.float32)

    selections = {
        "incumbent": np.zeros(100, dtype=np.int64),
        **selections,
        "posthoc_objective_pool_oracle": absolute[..., 3]
        .argmin(axis=1)
        .astype(np.int64),
    }
    methods = {
        name: replay_metrics(
            selected,
            targets,
            absolute,
            bootstrap_replicates=args.bootstrap_replicates,
            bootstrap_seed=args.bootstrap_seed + index * 10,
        )
        for index, (name, selected) in enumerate(selections.items())
    }
    for regime_name in ("scand_only", "domain_balanced"):
        for arm_name in ("support_only", "raw_only", "support_plus_raw"):
            methods[f"new_{regime_name}__{arm_name}"]["support"] = frozen[
                "selection_audit"
            ][regime_name][arm_name]["support"]

    result = {
        **frozen,
        "schema": "recon100_scand_adaptation_replay_result_v1",
        "status": "completed_frozen_cross_domain_recon100_replay",
        "labels": str(args.labels),
        "labels_sha256": file_sha256(args.labels),
        "labels_opened_in_evaluate_stage": True,
        "methods": methods,
        "bootstrap_replicates": args.bootstrap_replicates,
        "bootstrap_seed": args.bootstrap_seed,
        "deployment_evidence": False,
        "cem": False,
    }
    result_path = args.output_dir / "result.json"
    write_json(result_path, result)

    scand_result = read_json(args.scand_result)
    if scand_result.get("status") != "completed_first_label_open_fixed_pool":
        raise RuntimeError("SCAND fresh result status changed")
    write_summary(args.output_dir, methods, scand_result)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


def main() -> None:
    args = parse_args()
    if args.stage == "freeze":
        freeze(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
