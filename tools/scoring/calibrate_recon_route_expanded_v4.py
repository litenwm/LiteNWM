#!/usr/bin/env python3
"""Fit a RECON route-risk gate on scorer-cal plus spent outer cohorts.

This is development-only calibration.  The two former outer cohorts are
explicitly consumed as labelled development data; qualification_v4 is never
loaded by this program.
"""

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

from scoring.route_robust_calibration_v3 import (  # noqa: E402
    calibrated_probabilities,
    policy_dict,
    risk_features,
    risk_nonconformity,
)
from scoring.spacing_support_v1 import threshold_for_quantile  # noqa: E402
from scoring.universal_dualdomain_raw_v2 import (  # noqa: E402
    build_universal_features,
    build_universal_targets,
)
from tools.scoring.calibrate_universal_route_robust_v3 import (  # noqa: E402
    challenger_for_rank,
    ensemble_with_members,
    fold_safety,
    select_challenger,
    serializable_search,
    target_feasible,
    train_calibrator,
)
from tools.scoring.finetune_universal_dualdomain_safe_rank_v2 import (  # noqa: E402
    load_envelopes,
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
    load_group,
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
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "recon_route_expanded_v4",
    )
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--calibrator-members", type=int, default=3)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


def stratified_group_folds(
    group: np.ndarray, cohort: np.ndarray, folds: int, seed: int
) -> np.ndarray:
    """Hash whole groups to folds separately inside each provenance cohort."""

    import hashlib

    result = np.full(len(group), -1, dtype=np.int64)
    for cohort_name in np.unique(cohort):
        in_cohort = cohort == cohort_name
        names = np.unique(group[in_cohort]).astype(str).tolist()
        ordered = sorted(
            names,
            key=lambda value: hashlib.sha256(
                f"{seed}|{cohort_name}|{value}".encode()
            ).hexdigest(),
        )
        mapping = {value: index % folds for index, value in enumerate(ordered)}
        result[in_cohort] = np.asarray(
            [mapping[str(value)] for value in group[in_cohort]],
            dtype=np.int64,
        )
    if (result < 0).any() or len(np.unique(result)) != folds:
        raise RuntimeError("stratified route fold assignment failed")
    for name in np.unique(group):
        if len(np.unique(result[group == name])) != 1:
            raise RuntimeError(f"route group split across folds: {name}")
    return result


def load_development_data(
    args: argparse.Namespace,
) -> tuple[list[dict[str, np.ndarray]], dict[str, Any]]:
    scorer = load_features(args.scorer_cal_features, labels_expected=True)
    scorer_group = load_group(
        args.scorer_cal_source, scorer["record_key"], "recon"
    )
    scorer["route_group"] = scorer_group
    scorer["cohort"] = np.full(len(scorer_group), "scorer_cal", dtype="<U32")

    datasets = [scorer]
    audit: dict[str, Any] = {
        "scorer_cal": {
            "features": str(args.scorer_cal_features),
            "features_sha256": file_sha256(args.scorer_cal_features),
            "source": str(args.scorer_cal_source),
            "source_sha256": file_sha256(args.scorer_cal_source),
            "states": int(len(scorer_group)),
            "groups": int(len(np.unique(scorer_group))),
            "labels_role": "nested_scorer_cal_development",
        }
    }
    for index, (feature_path, outcome_path) in enumerate(
        zip(args.spent_features, args.spent_outcomes), start=1
    ):
        name = f"spent_outer_{index}"
        data = load_features(feature_path, labels_expected=False)
        with np.load(outcome_path, allow_pickle=False) as payload:
            record_key = payload["record_key"].astype(str)
            if not np.array_equal(data["record_key"].astype(str), record_key):
                raise RuntimeError(f"{name} feature/outcome identity mismatch")
            absolute = payload["absolute_metrics"].astype(np.float32)
            group = payload["route_group"].astype(str)
        if absolute.shape != (len(record_key), 17, 4):
            raise RuntimeError(f"{name} outcome shape changed: {absolute.shape}")
        data["targets"] = (absolute - absolute[:, :1]).astype(np.float32)
        data["route_group"] = group
        data["cohort"] = np.full(len(group), name, dtype="<U32")
        datasets.append(data)
        audit[name] = {
            "features": str(feature_path),
            "features_sha256": file_sha256(feature_path),
            "outcomes": str(outcome_path),
            "outcomes_sha256": file_sha256(outcome_path),
            "states": int(len(group)),
            "groups": int(len(np.unique(group))),
            "labels_role": "spent_outer_development",
        }

    for left in range(len(datasets)):
        for right in range(left + 1, len(datasets)):
            left_name = str(np.unique(datasets[left]["cohort"])[0])
            right_name = str(np.unique(datasets[right]["cohort"])[0])
            record_overlap = set(
                datasets[left]["record_key"].astype(str)
            ) & set(datasets[right]["record_key"].astype(str))
            route_overlap = set(
                datasets[left]["route_group"].astype(str)
            ) & set(datasets[right]["route_group"].astype(str))
            if record_overlap or route_overlap:
                raise RuntimeError(
                    f"development leakage {left_name}/{right_name}: "
                    f"records={len(record_overlap)} routes={len(route_overlap)}"
                )
            audit[f"{left_name}__{right_name}"] = {
                "record_overlap": 0,
                "route_group_overlap": 0,
            }
    return datasets, audit


def cohort_metrics(
    selected: np.ndarray,
    targets: np.ndarray,
    cohort: np.ndarray,
) -> dict[str, dict[str, Any]]:
    result = {
        str(name): safety_metrics(
            selected[cohort == name], targets[cohort == name]
        )
        for name in np.unique(cohort)
    }
    spent = np.char.startswith(cohort.astype(str), "spent_outer_")
    result["spent_pooled_200"] = safety_metrics(
        selected[spent], targets[spent]
    )
    return result


def cohort_target_feasible(values: dict[str, dict[str, Any]]) -> bool:
    required = ("spent_outer_1", "spent_outer_2", "spent_pooled_200")
    return all(
        name in values and target_feasible(values[name]) for name in required
    )


def expanded_policy_penalty(
    metrics: dict[str, Any],
    folds: list[dict[str, Any]],
    cohorts: dict[str, dict[str, Any]],
) -> float:
    def deficit(value: dict[str, Any]) -> float:
        return float(
            3.0 * max(0.0, 0.90 - (value["dual_precision"] or 0.0))
            + 3.0
            * max(0.0, 0.80 - (value["dual_wilson_lower_95"] or 0.0))
            + 3.0 * max(0.0, (value["harm_rate"] or 1.0) - 0.10)
            + 2.0 * max(0.0, 0.40 - value["coverage"])
        )

    fold_dual = [value["dual_precision"] or 0.0 for value in folds]
    return float(
        deficit(metrics)
        + 2.0 * deficit(cohorts["spent_pooled_200"])
        + sum(deficit(cohorts[name]) for name in ("spent_outer_1", "spent_outer_2"))
        + 0.20 * float(np.std(fold_dual))
    )


def expanded_policy_search(
    prediction: dict[str, np.ndarray],
    p_dual: np.ndarray,
    p_harm: np.ndarray,
    risk_ood: np.ndarray,
    disagreement: np.ndarray,
    valid: np.ndarray,
    targets: np.ndarray,
    fold_id: np.ndarray,
    cohort: np.ndarray,
    *,
    require_each_spent_cohort: bool,
) -> dict[str, Any]:
    records = []
    row = np.arange(len(valid))
    for harm_weight in (1.0, 2.0, 4.0):
        for uncertainty_weight in (0.0, 0.25):
            challenger, exists = challenger_for_rank(
                prediction,
                p_dual,
                p_harm,
                valid,
                harm_weight=harm_weight,
                uncertainty_weight=uncertainty_weight,
            )
            ood_reference = risk_ood[row, challenger][exists]
            disagreement_reference = disagreement[row, challenger][exists]
            if not np.isfinite(ood_reference).all():
                raise RuntimeError("non-finite risk OOD reference")
            if not np.isfinite(disagreement_reference).all():
                raise RuntimeError("non-finite disagreement reference")
            ood_thresholds = [
                float(np.quantile(ood_reference, q))
                for q in (0.95, 0.975)
            ]
            disagreement_thresholds = [
                float(np.quantile(disagreement_reference, q))
                for q in (0.80, 0.90)
            ]
            for ood_threshold in ood_thresholds:
                for disagreement_threshold in disagreement_thresholds:
                    for z in (0.5, 1.0):
                        for margin in (0.0, 0.005):
                            for dual_threshold in (
                                0.30,
                                0.40,
                                0.50,
                                0.60,
                                0.70,
                                0.80,
                                0.90,
                            ):
                                for harm_threshold in (
                                    0.05,
                                    0.10,
                                    0.20,
                                    0.30,
                                    0.40,
                                    0.50,
                                    0.60,
                                ):
                                    from scoring.route_robust_calibration_v3 import (
                                        RouteRiskPolicyV3,
                                    )

                                    policy = RouteRiskPolicyV3(
                                        harm_rank_weight=harm_weight,
                                        uncertainty_rank_weight=uncertainty_weight,
                                        z=z,
                                        margin=margin,
                                        dual_threshold=dual_threshold,
                                        harm_threshold=harm_threshold,
                                        risk_ood_threshold=ood_threshold,
                                        calibrator_disagreement_threshold=(
                                            disagreement_threshold
                                        ),
                                    )
                                    selected = select_challenger(
                                        prediction,
                                        p_dual,
                                        p_harm,
                                        risk_ood,
                                        disagreement,
                                        challenger,
                                        exists,
                                        policy,
                                    )
                                    metrics = safety_metrics(selected, targets)
                                    cohorts = cohort_metrics(
                                        selected, targets, cohort
                                    )
                                    if metrics["coverage"] < 0.40:
                                        continue
                                    if (
                                        cohorts["spent_pooled_200"]["coverage"]
                                        < 0.40
                                    ):
                                        continue
                                    if require_each_spent_cohort and any(
                                        cohorts[name]["coverage"] < 0.40
                                        for name in (
                                            "spent_outer_1",
                                            "spent_outer_2",
                                        )
                                    ):
                                        continue
                                    folds = fold_safety(
                                        selected, targets, fold_id
                                    )
                                    strict = cohort_target_feasible(cohorts)
                                    records.append(
                                        {
                                            "policy": policy,
                                            "selected": selected,
                                            "metrics": metrics,
                                            "cohort_metrics": cohorts,
                                            "fold_metrics": folds,
                                            "target_feasible": target_feasible(
                                                metrics
                                            ),
                                            "strict_spent_feasible": strict,
                                            "penalty": expanded_policy_penalty(
                                                metrics, folds, cohorts
                                            ),
                                        }
                                    )
    if not records:
        raise RuntimeError("no expanded policy reaches coverage constraints")
    strict = [value for value in records if value["strict_spent_feasible"]]
    if strict:
        pool = strict
        tier = "strict_each_spent_target"
        key = lambda value: (
            -value["cohort_metrics"]["spent_pooled_200"]["coverage"],
            -value["metrics"]["coverage"],
            value["metrics"]["net_delta_per_state_m"]["objective"],
        )
    else:
        pool = records
        tier = "best_available"
        key = lambda value: (
            value["penalty"],
            -value["cohort_metrics"]["spent_pooled_200"]["coverage"],
            -value["metrics"]["coverage"],
        )
    best = min(pool, key=key)
    return {
        **best,
        "selection_tier": tier,
        "strict_spent_frontier_feasible": bool(strict),
        "searched_policies": len(records),
    }


def serializable_expanded(value: dict[str, Any]) -> dict[str, Any]:
    return {
        "policy": policy_dict(value["policy"]),
        "metrics": value["metrics"],
        "cohort_metrics": value["cohort_metrics"],
        "fold_metrics": value["fold_metrics"],
        "target_feasible": value["target_feasible"],
        "strict_spent_feasible": value["strict_spent_feasible"],
        "strict_spent_frontier_feasible": value[
            "strict_spent_frontier_feasible"
        ],
        "selection_tier": value["selection_tier"],
        "searched_policies": value["searched_policies"],
        "penalty": value["penalty"],
    }


def main() -> None:
    args = parse_args()
    result_path = args.output_dir / "result.json"
    if result_path.exists():
        raise FileExistsError(result_path)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    training = json.loads(args.training_result.read_text(encoding="utf-8"))
    if (
        training.get("cem") is not False
        or not training.get("primary_excludes_scorer_v2_predictions")
    ):
        raise RuntimeError("base universal D provenance is invalid")

    datasets, leakage_audit = load_development_data(args)
    envelopes = load_envelopes(Path(training["support"]))
    universal_parts = []
    support_parts = []
    valid_parts = []
    target_parts = []
    group_parts = []
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
        group_parts.append(data["route_group"].astype(str))
        cohort_parts.append(data["cohort"].astype(str))
        record_parts.append(data["record_key"].astype(str))

    universal = np.concatenate(universal_parts)
    support = np.concatenate(support_parts)
    valid = np.concatenate(valid_parts)
    targets = np.concatenate(target_parts)
    group = np.concatenate(group_parts)
    cohort = np.concatenate(cohort_parts)
    record_key = np.concatenate(record_parts)
    fold_id = stratified_group_folds(group, cohort, 5, 20260725)
    prediction, member_prediction = ensemble_with_members(
        training["arms"]["D"]["checkpoints"]["full"],
        universal,
        np.zeros(len(universal), dtype=np.int64),
        device=args.device,
    )
    risk = risk_features(
        prediction,
        member_prediction,
        support,
        robot_context=np.zeros((len(group), 2), dtype=np.float32),
    )
    target = build_universal_targets(targets)

    oof_dual = np.empty(valid.shape, dtype=np.float32)
    oof_harm = np.empty(valid.shape, dtype=np.float32)
    oof_risk_ood = np.empty(valid.shape, dtype=np.float32)
    oof_disagreement = np.empty(valid.shape, dtype=np.float32)
    checkpoints = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for heldout in range(5):
        train_state = fold_id != heldout
        test = fold_id == heldout
        dual_members = []
        harm_members = []
        ood_members = []
        member_records = []
        for member_index in range(args.calibrator_members):
            record = train_calibrator(
                risk,
                valid,
                target,
                group,
                train_state,
                epochs=args.epochs,
                seed=20260725 + heldout * 10 + member_index,
                device=args.device,
                checkpoint=args.output_dir
                / (
                    f"checkpoints/recon_fold{heldout}"
                    f"_member{member_index}.pt"
                ),
                role=(
                    f"expanded_group_oof_fold_{heldout}"
                    f"_member_{member_index}"
                ),
                domain="recon",
            )
            dual, harm = calibrated_probabilities(
                record["model"],
                risk[test],
                record["normalization"],
                device=args.device,
            )
            dual_members.append(dual)
            harm_members.append(harm)
            ood_members.append(
                risk_nonconformity(risk[test], record["normalization"])
            )
            record.pop("model")
            record.pop("normalization")
            member_records.append(record)
        dual_stack = np.stack(dual_members)
        harm_stack = np.stack(harm_members)
        oof_dual[test] = np.clip(
            dual_stack.mean(0) - dual_stack.std(0), 0.0, 1.0
        )
        oof_harm[test] = np.clip(
            harm_stack.mean(0) + harm_stack.std(0), 0.0, 1.0
        )
        oof_disagreement[test] = (
            dual_stack.std(0) + harm_stack.std(0)
        )
        oof_risk_ood[test] = np.stack(ood_members).max(0)
        checkpoints.append(
            {
                "heldout_fold": heldout,
                "aggregation": (
                    "p_dual=mean-1std;p_harm=mean+1std;risk_ood=max"
                ),
                "members": member_records,
            }
        )

    nested_selected = np.zeros(len(valid), dtype=np.int64)
    nested_policies = []
    for heldout in range(5):
        fit = fold_id != heldout
        search = expanded_policy_search(
            {key: value[fit] for key, value in prediction.items()},
            oof_dual[fit],
            oof_harm[fit],
            oof_risk_ood[fit],
            oof_disagreement[fit],
            valid[fit],
            targets[fit],
            fold_id[fit],
            cohort[fit],
            require_each_spent_cohort=False,
        )
        test = fold_id == heldout
        challenger, exists = challenger_for_rank(
            {key: value[test] for key, value in prediction.items()},
            oof_dual[test],
            oof_harm[test],
            valid[test],
            harm_weight=search["policy"].harm_rank_weight,
            uncertainty_weight=search["policy"].uncertainty_rank_weight,
        )
        nested_selected[test] = select_challenger(
            {key: value[test] for key, value in prediction.items()},
            oof_dual[test],
            oof_harm[test],
            oof_risk_ood[test],
            oof_disagreement[test],
            challenger,
            exists,
            search["policy"],
        )
        nested_policies.append(
            {
                "heldout_fold": heldout,
                "fit": serializable_expanded(search),
            }
        )

    nested_metrics = safety_metrics(nested_selected, targets)
    nested_cohorts = cohort_metrics(nested_selected, targets, cohort)
    deployment = expanded_policy_search(
        prediction,
        oof_dual,
        oof_harm,
        oof_risk_ood,
        oof_disagreement,
        valid,
        targets,
        fold_id,
        cohort,
        require_each_spent_cohort=True,
    )

    full_records = []
    for member_index in range(args.calibrator_members):
        record = train_calibrator(
            risk,
            valid,
            target,
            group,
            np.ones(len(valid), dtype=bool),
            epochs=args.epochs,
            seed=20261025 + member_index,
            device=args.device,
            checkpoint=args.output_dir
            / f"checkpoints/recon_full_member{member_index}.pt",
            role=f"expanded_deployment_full_member_{member_index}",
            domain="recon",
        )
        record.pop("model")
        record.pop("normalization")
        full_records.append(record)

    arrays_path = args.output_dir / "calibration_predictions.npz"
    write_npz(
        arrays_path,
        record_key=record_key.astype(str),
        route_group=group.astype(str),
        cohort=cohort.astype(str),
        fold_id=fold_id,
        nested_selected=nested_selected,
        deployment_oof_selected=deployment["selected"],
        oof_p_dual=oof_dual,
        oof_p_harm=oof_harm,
        oof_risk_ood=oof_risk_ood,
        oof_calibrator_disagreement=oof_disagreement,
    )
    report = {
        "schema": "recon_route_expanded_calibration_v4",
        "status": "expanded_spent_development_complete",
        "evidence_role": "development_only_consumes_two_spent_outer_cohorts",
        "qualification_v4_accessed": False,
        "base_training_result": str(args.training_result),
        "base_training_result_sha256": file_sha256(args.training_result),
        "data_and_leakage_audit": leakage_audit,
        "states": int(len(group)),
        "groups": int(len(np.unique(group))),
        "cohorts": {
            str(name): {
                "states": int((cohort == name).sum()),
                "groups": int(len(np.unique(group[cohort == name]))),
            }
            for name in np.unique(cohort)
        },
        "folds": 5,
        "fold_group_counts": {
            str(fold): int(len(np.unique(group[fold_id == fold])))
            for fold in range(5)
        },
        "checkpoints": {"oof": checkpoints, "full": full_records},
        "nested_group_oof": {
            "metrics": nested_metrics,
            "cohort_metrics": nested_cohorts,
            "target_feasible": target_feasible(nested_metrics),
            "strict_spent_feasible": cohort_target_feasible(nested_cohorts),
            "policies": nested_policies,
        },
        "deployment_policy": serializable_expanded(deployment),
        "feature_contract": (
            "universal-D raw predictions/disagreement/support only; "
            "no scorer-v2 predictions; no route/bag identity"
        ),
        "causal_deployment_audit": {
            "decision_scope": "current_window_only",
            "full_route_future_windows_used": False,
            "route_quota_used": False,
            "qualification_v4_covariates_or_outcomes_used": False,
        },
        "historical_evidence": {
            "universal_D_pooled_200": {
                "dual_precision": 0.8426,
                "dual_wilson_lower_95": 0.7623,
                "harm_rate": 0.1574,
                "qualification": "failed",
            }
        },
        "calibration_predictions": str(arrays_path),
        "calibration_predictions_sha256": file_sha256(arrays_path),
        "fixed_pool_only": True,
        "cem": False,
    }
    write_json(result_path, report)
    print(
        json.dumps(
            {
                "result": str(result_path),
                "result_sha256": file_sha256(result_path),
                "nested_group_oof": report["nested_group_oof"],
                "deployment_policy": report["deployment_policy"],
                "qualification_v4_accessed": False,
                "cem": False,
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
