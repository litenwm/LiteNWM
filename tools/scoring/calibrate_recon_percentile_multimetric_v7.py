#!/usr/bin/env python3
"""Calibrate a percentile-normalized four-metric gate on spent RECON data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from scoring.percentile_multimetric_gate_v7 import (  # noqa: E402
    PercentileMultimetricPolicyV7,
    empirical_cdf,
    policy_dict,
    select_percentile_multimetric,
)
from scoring.route_multimodal_calibration_v5 import (  # noqa: E402
    calibrated_probabilities_v5,
    multimodal_risk_features,
)
from scoring.route_robust_calibration_v3 import (  # noqa: E402
    risk_features,
    risk_nonconformity,
)
from scoring.spacing_support_v1 import threshold_for_quantile  # noqa: E402
from scoring.universal_dualdomain_raw_v2 import build_universal_features  # noqa: E402
from tools.scoring.calibrate_recon_route_expanded_v4 import (  # noqa: E402
    load_development_data,
)
from tools.scoring.calibrate_universal_route_robust_v3 import (  # noqa: E402
    ensemble_with_members,
)
from tools.scoring.finetune_universal_dualdomain_safe_rank_v2 import (  # noqa: E402
    load_envelopes,
)
from tools.scoring.freeze_recon_route_multimodal_v5_mapping import (  # noqa: E402
    load_calibrator,
)
from tools.scoring.train_recon_dense_meta_dinov3_three_arm_v1 import (  # noqa: E402
    load_features,
)
from tools.scoring.train_recon_dense_meta_dinov3_v1 import (  # noqa: E402
    file_sha256,
    write_json,
    write_npz,
)
from tools.scoring.train_universal_dualdomain_raw_v2 import (  # noqa: E402
    safety_metrics,
)


ROOT = REPO_ROOT / "outputs/universal_dualdomain_raw_v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--training-result",
        type=Path,
        default=ROOT / "training_v2/result.json",
    )
    parser.add_argument(
        "--v5-result",
        type=Path,
        default=ROOT / "recon_route_multimodal_v5/result.json",
    )
    parser.add_argument(
        "--qualification-features",
        type=Path,
        default=ROOT
        / "recon_fresh_outer_qualification_v4/features/"
        "recon_fresh_outer_qualification_h4_100_seed20260725_features.npz",
    )
    parser.add_argument(
        "--qualification-outcomes",
        type=Path,
        default=ROOT
        / "recon_fresh_outer_qualification_v4/fresh_outer_outcomes.npz",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "recon_percentile_multimetric_v7",
    )
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument(
        "--scorer-cal-features",
        type=Path,
        default=ROOT
        / "features/recon_scorer_cal/recon_scorer_cal_universal_v2_features.npz",
    )
    parser.add_argument(
        "--scorer-cal-source",
        type=Path,
        default=ROOT / "sources/recon_scorer_cal_source.npz",
    )
    parser.add_argument(
        "--spent-features",
        type=Path,
        nargs=2,
        default=[
            ROOT
            / "recon_fresh_outer_v2/features/"
            "recon_fresh_outer_h4_100_seed20260721_features.npz",
            ROOT
            / "recon_fresh_outer_confirm_v2/features/"
            "recon_fresh_outer_confirm_h4_100_seed20260722_features.npz",
        ],
    )
    parser.add_argument(
        "--spent-outcomes",
        type=Path,
        nargs=2,
        default=[
            ROOT / "recon_fresh_outer_v2/fresh_outer_outcomes.npz",
            ROOT / "recon_fresh_outer_confirm_v2/fresh_outer_outcomes.npz",
        ],
    )
    return parser.parse_args()


def append_qualification_spent(
    datasets: list[dict[str, np.ndarray]],
    feature_path: Path,
    outcome_path: Path,
) -> dict[str, Any]:
    data = load_features(feature_path, labels_expected=False)
    with np.load(outcome_path, allow_pickle=False) as payload:
        if not np.array_equal(
            data["record_key"].astype(str),
            payload["record_key"].astype(str),
        ):
            raise RuntimeError("qualification-v4 feature/outcome mismatch")
        absolute = payload["absolute_metrics"].astype(np.float32)
        group = payload["route_group"].astype(str)
    if absolute.shape != (100, 17, 4):
        raise RuntimeError("qualification-v4 outcome shape changed")
    data["targets"] = absolute - absolute[:, :1]
    data["route_group"] = group
    data["cohort"] = np.full(100, "spent_outer_3_v4", dtype="<U32")
    existing_record = set(
        np.concatenate(
            [value["record_key"].astype(str) for value in datasets]
        )
    )
    existing_group = set(
        np.concatenate(
            [value["route_group"].astype(str) for value in datasets]
        )
    )
    if existing_record & set(data["record_key"].astype(str)):
        raise RuntimeError("qualification-v4 record overlaps development")
    if existing_group & set(group):
        raise RuntimeError("qualification-v4 route overlaps development")
    datasets.append(data)
    return {
        "features": str(feature_path),
        "features_sha256": file_sha256(feature_path),
        "outcomes": str(outcome_path),
        "outcomes_sha256": file_sha256(outcome_path),
        "states": 100,
        "groups": int(len(np.unique(group))),
        "labels_role": "spent_failed_qualification_development",
    }


def cohort_metrics(
    selected: np.ndarray,
    targets: np.ndarray,
    cohort: np.ndarray,
) -> dict[str, dict[str, Any]]:
    values = {
        str(name): safety_metrics(
            selected[cohort == name], targets[cohort == name]
        )
        for name in np.unique(cohort)
    }
    outer = np.char.startswith(cohort.astype(str), "spent_outer_")
    values["spent_outer_pooled_300"] = safety_metrics(
        selected[outer], targets[outer]
    )
    return values


def target_feasible(value: dict[str, Any]) -> bool:
    return bool(
        value["coverage"] >= 0.40
        and value["dual_precision"] is not None
        and value["dual_precision"] >= 0.90
        and value["dual_wilson_lower_95"] is not None
        and value["dual_wilson_lower_95"] >= 0.80
        and value["harm_rate"] is not None
        and value["harm_rate"] <= 0.10
    )


def policy_penalty(
    metrics: dict[str, Any],
    cohorts: dict[str, dict[str, Any]],
) -> float:
    def deficit(value: dict[str, Any]) -> float:
        return float(
            4.0 * max(0.0, 0.90 - (value["dual_precision"] or 0.0))
            + 4.0
            * max(0.0, 0.80 - (value["dual_wilson_lower_95"] or 0.0))
            + 4.0 * max(0.0, (value["harm_rate"] or 1.0) - 0.10)
            + 3.0 * max(0.0, 0.40 - value["coverage"])
        )

    outer_names = (
        "spent_outer_1",
        "spent_outer_2",
        "spent_outer_3_v4",
        "spent_outer_pooled_300",
    )
    tail = sum(
        max(
            0.0,
            cohorts["spent_outer_pooled_300"]["net_delta_per_state_m"][
                name
            ],
        )
        for name in ("rpe", "endpoint")
    )
    return float(
        deficit(metrics)
        + 2.0 * deficit(cohorts["spent_outer_pooled_300"])
        + sum(deficit(cohorts[name]) for name in outer_names[:-1])
        + 10.0 * tail
    )


def search_policy(
    prediction: dict[str, np.ndarray],
    q_dual: np.ndarray,
    q_harm: np.ndarray,
    q_ood: np.ndarray,
    q_disagreement: np.ndarray,
    valid: np.ndarray,
    targets: np.ndarray,
    cohort: np.ndarray,
) -> dict[str, Any]:
    records = []
    for harm_rank_weight in (1.0, 2.0, 4.0):
        for uncertainty_rank_weight in (0.0, 0.25):
            for rpe_rank_weight in (0.0, 5.0):
                for endpoint_rank_weight in (0.0, 2.0):
                    for z in (0.5, 1.0):
                        for margin_ate, margin_objective in (
                            (0.0, 0.0),
                            (0.0, 0.005),
                            (0.005, 0.0),
                            (0.005, 0.005),
                        ):
                            for rpe_cap in (0.0, 0.0025, 0.005, 0.010):
                                for endpoint_cap in (
                                    0.0,
                                    0.005,
                                    0.010,
                                    0.020,
                                ):
                                    for dual_threshold in (
                                        0.50,
                                        0.60,
                                        0.70,
                                        0.80,
                                        0.85,
                                    ):
                                        for harm_threshold in (
                                            0.10,
                                            0.20,
                                            0.30,
                                            0.40,
                                        ):
                                            policy = (
                                                PercentileMultimetricPolicyV7(
                                                    harm_rank_weight=(
                                                        harm_rank_weight
                                                    ),
                                                    uncertainty_rank_weight=(
                                                        uncertainty_rank_weight
                                                    ),
                                                    rpe_rank_weight=(
                                                        rpe_rank_weight
                                                    ),
                                                    endpoint_rank_weight=(
                                                        endpoint_rank_weight
                                                    ),
                                                    z=z,
                                                    margin_ate=margin_ate,
                                                    margin_objective=(
                                                        margin_objective
                                                    ),
                                                    rpe_ucb_cap=rpe_cap,
                                                    endpoint_ucb_cap=(
                                                        endpoint_cap
                                                    ),
                                                    dual_quantile_threshold=(
                                                        dual_threshold
                                                    ),
                                                    harm_quantile_threshold=(
                                                        harm_threshold
                                                    ),
                                                )
                                            )
                                            selected = (
                                                select_percentile_multimetric(
                                                    prediction,
                                                    q_dual,
                                                    q_harm,
                                                    q_ood,
                                                    q_disagreement,
                                                    valid,
                                                    policy,
                                                )
                                            )
                                            metrics = safety_metrics(
                                                selected, targets
                                            )
                                            cohorts = cohort_metrics(
                                                selected, targets, cohort
                                            )
                                            if metrics["coverage"] < 0.40:
                                                continue
                                            if any(
                                                cohorts[name]["coverage"]
                                                < 0.40
                                                for name in (
                                                    "spent_outer_1",
                                                    "spent_outer_2",
                                                    "spent_outer_3_v4",
                                                )
                                            ):
                                                continue
                                            strict = all(
                                                target_feasible(
                                                    cohorts[name]
                                                )
                                                for name in (
                                                    "spent_outer_1",
                                                    "spent_outer_2",
                                                    "spent_outer_3_v4",
                                                    "spent_outer_pooled_300",
                                                )
                                            )
                                            records.append(
                                                {
                                                    "policy": policy,
                                                    "selected": selected,
                                                    "metrics": metrics,
                                                    "cohort_metrics": cohorts,
                                                    "strict_feasible": strict,
                                                    "penalty": policy_penalty(
                                                        metrics, cohorts
                                                    ),
                                                }
                                            )
    if not records:
        raise RuntimeError("no v7 policy reaches per-cohort coverage")
    strict = [value for value in records if value["strict_feasible"]]
    if strict:
        pool = strict
        tier = "strict_all_spent_outer"
        key = lambda value: (
            -value["cohort_metrics"]["spent_outer_pooled_300"][
                "coverage"
            ],
            value["metrics"]["net_delta_per_state_m"]["rpe"],
            value["metrics"]["net_delta_per_state_m"]["endpoint"],
        )
    else:
        pool = records
        tier = "best_available"
        key = lambda value: (
            value["penalty"],
            -value["cohort_metrics"]["spent_outer_pooled_300"][
                "coverage"
            ],
        )
    best = min(pool, key=key)
    return {
        **best,
        "selection_tier": tier,
        "strict_frontier_feasible": bool(strict),
        "searched_policies": len(records),
    }


def main() -> None:
    args = parse_args()
    result_path = args.output_dir / "result.json"
    if result_path.exists():
        raise FileExistsError(result_path)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    training = json.loads(args.training_result.read_text(encoding="utf-8"))
    v5 = json.loads(args.v5_result.read_text(encoding="utf-8"))
    if (
        training.get("cem") is not False
        or v5.get("schema") != "recon_route_multimodal_calibration_v5"
        or v5.get("qualification_v4_accessed") is not False
    ):
        raise RuntimeError("v7 upstream provenance is invalid")
    datasets, leakage_audit = load_development_data(args)
    leakage_audit["spent_outer_3_v4"] = append_qualification_spent(
        datasets,
        args.qualification_features,
        args.qualification_outcomes,
    )
    envelopes = load_envelopes(Path(training["support"]))

    universal_parts = []
    support_parts = []
    valid_parts = []
    target_parts = []
    cohort_parts = []
    record_parts = []
    for data in datasets:
        universal, support, guard = build_universal_features(
            data,
            spacing_m=0.25,
            support_envelope=envelopes["recon"],
        )
        valid = guard & (
            support
            <= threshold_for_quantile(envelopes["recon"], 0.999)
        )
        valid[:, 0] = True
        universal_parts.append(universal)
        support_parts.append(support)
        valid_parts.append(valid)
        target_parts.append(data["targets"][..., :4].astype(np.float32))
        cohort_parts.append(data["cohort"].astype(str))
        record_parts.append(data["record_key"].astype(str))
    universal = np.concatenate(universal_parts)
    support = np.concatenate(support_parts)
    valid = np.concatenate(valid_parts)
    targets = np.concatenate(target_parts)
    cohort = np.concatenate(cohort_parts)
    record_key = np.concatenate(record_parts)
    prediction, base_members = ensemble_with_members(
        training["arms"]["D"]["checkpoints"]["full"],
        universal,
        np.zeros(len(universal), dtype=np.int64),
        device=args.device,
    )
    base_risk = risk_features(
        prediction,
        base_members,
        support,
        robot_context=np.zeros((len(universal), 2), dtype=np.float32),
    )
    risk = multimodal_risk_features(base_risk, universal)
    del base_members

    full_dual_members = []
    full_harm_members = []
    full_ood_members = []
    checkpoint_audit = []
    for record in v5["checkpoints"]["full"]:
        path = Path(record["path"])
        model, stats = load_calibrator(path, args.device)
        dual, harm = calibrated_probabilities_v5(
            model, risk, stats, device=args.device
        )
        full_dual_members.append(dual)
        full_harm_members.append(harm)
        full_ood_members.append(risk_nonconformity(risk, stats))
        checkpoint_audit.append(
            {"path": str(path), "sha256": file_sha256(path)}
        )
        del model
        if args.device == "cuda":
            torch.cuda.empty_cache()
    full_dual_stack = np.stack(full_dual_members)
    full_harm_stack = np.stack(full_harm_members)
    full_dual = np.clip(
        full_dual_stack.mean(0) - full_dual_stack.std(0), 0.0, 1.0
    )
    full_harm = np.clip(
        full_harm_stack.mean(0) + full_harm_stack.std(0), 0.0, 1.0
    )
    full_ood = np.stack(full_ood_members).max(0)
    full_disagreement = (
        full_dual_stack.std(0) + full_harm_stack.std(0)
    )

    with np.load(v5["calibration_predictions"], allow_pickle=False) as payload:
        oof_dual = payload["oof_p_dual"].astype(np.float32)
        oof_harm = payload["oof_p_harm"].astype(np.float32)
        oof_ood = payload["oof_risk_ood"].astype(np.float32)
        oof_disagreement = payload[
            "oof_calibrator_disagreement"
        ].astype(np.float32)
        if not np.array_equal(
            payload["record_key"].astype(str), record_key[:610]
        ):
            raise RuntimeError("v5 OOF/development identity mismatch")
    reference_mask = valid[:610].copy()
    reference_mask[:, 0] = False
    references = {
        "oof_dual": np.sort(oof_dual[reference_mask]),
        "oof_harm": np.sort(oof_harm[reference_mask]),
        "oof_ood": np.sort(oof_ood[reference_mask]),
        "oof_disagreement": np.sort(oof_disagreement[reference_mask]),
        "full_dual": np.sort(full_dual[:610][reference_mask]),
        "full_harm": np.sort(full_harm[:610][reference_mask]),
        "full_ood": np.sort(full_ood[:610][reference_mask]),
        "full_disagreement": np.sort(
            full_disagreement[:610][reference_mask]
        ),
    }
    q_dual = np.empty(valid.shape, dtype=np.float32)
    q_harm = np.empty(valid.shape, dtype=np.float32)
    q_ood = np.empty(valid.shape, dtype=np.float32)
    q_disagreement = np.empty(valid.shape, dtype=np.float32)
    q_dual[:610] = empirical_cdf(oof_dual, references["oof_dual"])
    q_harm[:610] = empirical_cdf(oof_harm, references["oof_harm"])
    q_ood[:610] = empirical_cdf(oof_ood, references["oof_ood"])
    q_disagreement[:610] = empirical_cdf(
        oof_disagreement, references["oof_disagreement"]
    )
    q_dual[610:] = empirical_cdf(
        full_dual[610:], references["full_dual"]
    )
    q_harm[610:] = empirical_cdf(
        full_harm[610:], references["full_harm"]
    )
    q_ood[610:] = empirical_cdf(full_ood[610:], references["full_ood"])
    q_disagreement[610:] = empirical_cdf(
        full_disagreement[610:], references["full_disagreement"]
    )

    search = search_policy(
        prediction,
        q_dual,
        q_harm,
        q_ood,
        q_disagreement,
        valid,
        targets,
        cohort,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reference_path = args.output_dir / "deployment_reference.npz"
    write_npz(reference_path, **{
        key: value
        for key, value in references.items()
        if key.startswith("full_")
    })
    selection_path = args.output_dir / "development_selection.npz"
    write_npz(
        selection_path,
        record_key=record_key,
        cohort=cohort,
        selected_index=search["selected"],
    )
    serializable = {
        "policy": policy_dict(search["policy"]),
        "metrics": search["metrics"],
        "cohort_metrics": search["cohort_metrics"],
        "strict_feasible": search["strict_feasible"],
        "strict_frontier_feasible": search[
            "strict_frontier_feasible"
        ],
        "selection_tier": search["selection_tier"],
        "searched_policies": search["searched_policies"],
        "penalty": search["penalty"],
    }
    report = {
        "schema": "recon_percentile_multimetric_gate_v7",
        "status": "spent_development_calibrated",
        "evidence_role": "development_only_consumes_three_spent_outer_cohorts",
        "states": 710,
        "cohorts": {
            str(name): int((cohort == name).sum())
            for name in np.unique(cohort)
        },
        "training_result": str(args.training_result),
        "training_result_sha256": file_sha256(args.training_result),
        "v5_result": str(args.v5_result),
        "v5_result_sha256": file_sha256(args.v5_result),
        "v5_full_checkpoints": checkpoint_audit,
        "data_and_leakage_audit": leakage_audit,
        "policy_search": serializable,
        "deployment_reference": str(reference_path),
        "deployment_reference_sha256": file_sha256(reference_path),
        "development_selection": str(selection_path),
        "development_selection_sha256": file_sha256(selection_path),
        "percentile_contract": (
            "full calibrator candidate score mapped against frozen full "
            "scorer-cal+spent1+spent2 reference; no route batch ranking"
        ),
        "feature_contract": (
            "current-window action+DINOv3 static+raw LTWM+universal-D "
            "predictions; no scorer-v2 prediction; no route ID/future window"
        ),
        "next_outer_accessed": False,
        "next_outer_covariates_used": False,
        "fixed_pool_only": True,
        "cem": False,
    }
    write_json(result_path, report)
    print(
        json.dumps(
            {
                "result": str(result_path),
                "result_sha256": file_sha256(result_path),
                "policy_search": serializable,
                "next_outer_accessed": False,
                "cem": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
