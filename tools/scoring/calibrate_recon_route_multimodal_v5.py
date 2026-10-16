#!/usr/bin/env python3
"""Train a cohort-balanced multimodal RECON safety calibration head."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from scoring.route_multimodal_calibration_v5 import (  # noqa: E402
    MULTIMODAL_RISK_FEATURE_DIM,
    RouteMultimodalCalibratorV5,
    calibrated_probabilities_v5,
    multimodal_risk_features,
)
from scoring.route_robust_calibration_v3 import (  # noqa: E402
    normalization,
    risk_features,
    risk_nonconformity,
)
from scoring.spacing_support_v1 import threshold_for_quantile  # noqa: E402
from scoring.universal_dualdomain_raw_v2 import (  # noqa: E402
    build_universal_features,
    build_universal_targets,
    set_seed,
)
from tools.scoring.calibrate_recon_route_expanded_v4 import (  # noqa: E402
    cohort_metrics,
    cohort_target_feasible,
    expanded_policy_search,
    load_development_data,
    serializable_expanded,
    stratified_group_folds,
)
from tools.scoring.calibrate_universal_route_robust_v3 import (  # noqa: E402
    challenger_for_rank,
    ensemble_with_members,
    select_challenger,
)
from tools.scoring.finetune_universal_dualdomain_safe_rank_v2 import (  # noqa: E402
    load_envelopes,
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
    base = argparse.ArgumentParser(add_help=False)
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
        default=ROOT / "recon_route_multimodal_v6_listwise",
    )
    parser.add_argument("--epochs", type=int, default=350)
    parser.add_argument("--calibrator-members", type=int, default=3)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


def save_calibrator(
    path: Path,
    model: RouteMultimodalCalibratorV5,
    stats: dict[str, np.ndarray],
    metadata: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "schema": "route_multimodal_risk_calibrator_v5",
            "input_dim": MULTIMODAL_RISK_FEATURE_DIM,
            "hidden_dim": 96,
            "dropout": 0.10,
            "state_dict": model.state_dict(),
            "normalization": {
                key: value.tolist() for key, value in stats.items()
            },
            **metadata,
        },
        temporary,
    )
    temporary.replace(path)


def train_calibrator(
    feature: np.ndarray,
    valid: np.ndarray,
    target: dict[str, np.ndarray],
    groups: np.ndarray,
    cohorts: np.ndarray,
    train_state: np.ndarray,
    *,
    epochs: int,
    seed: int,
    device: str,
    checkpoint: Path,
    role: str,
) -> dict[str, Any]:
    set_seed(seed)
    in_state = np.asarray(train_state, dtype=bool)
    train_valid = np.asarray(valid, dtype=bool) & in_state[:, None]
    train_valid[:, 0] = False
    stats = normalization(feature[in_state], valid[in_state])
    normalized = np.clip(
        (
            feature[in_state] - stats["feature_center"]
        ) / stats["feature_scale"],
        -12.0,
        12.0,
    ).astype(np.float32)
    x_all = torch.from_numpy(normalized).to(device)
    local_valid = np.asarray(valid[in_state], dtype=bool).copy()
    local_valid[:, 0] = False
    valid_tensor = torch.from_numpy(local_valid).to(device)
    dual_all = torch.from_numpy(
        target["dual"][in_state].astype(np.float32)
    ).to(device)
    harm_all = torch.from_numpy(
        target["harm"][in_state].astype(np.float32)
    ).to(device)
    flat_group = np.repeat(
        np.asarray(groups).astype(str)[:, None], valid.shape[1], axis=1
    )[train_valid]
    flat_cohort = np.repeat(
        np.asarray(cohorts).astype(str)[:, None], valid.shape[1], axis=1
    )[train_valid]
    unique_groups, group_index_np = np.unique(
        flat_group, return_inverse=True
    )
    group_cohort = np.empty(len(unique_groups), dtype="<U32")
    for index, name in enumerate(unique_groups):
        values = np.unique(flat_cohort[flat_group == name])
        if len(values) != 1:
            raise RuntimeError(f"group spans provenance cohorts: {name}")
        group_cohort[index] = values[0]
    group_index = torch.from_numpy(group_index_np.astype(np.int64)).to(device)
    count = torch.bincount(group_index, minlength=len(unique_groups)).float()

    model = RouteMultimodalCalibratorV5().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=7.5e-4, weight_decay=1.0e-3
    )
    last_loss = 0.0
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        logits_all = model(
            x_all.reshape(-1, x_all.shape[-1])
        ).reshape(*x_all.shape[:2], 2)
        logits = logits_all[valid_tensor]
        y = torch.stack(
            (dual_all[valid_tensor], harm_all[valid_tensor]), dim=-1
        )
        element = F.binary_cross_entropy_with_logits(
            logits, y, reduction="none"
        )
        dual_weight = torch.where(y[:, 0] > 0.5, 1.0, 4.0)
        harm_weight = torch.where(y[:, 1] > 0.5, 5.0, 1.0)
        candidate_loss = (
            element[:, 0] * dual_weight
            + 2.0 * element[:, 1] * harm_weight
        )
        group_sum = torch.zeros(len(unique_groups), device=device)
        group_sum.scatter_add_(0, group_index, candidate_loss)
        group_loss = group_sum / count
        cohort_losses = []
        for cohort_name in np.unique(group_cohort):
            local_np = np.flatnonzero(group_cohort == cohort_name)
            local = torch.from_numpy(local_np.astype(np.int64)).to(device)
            local_loss = group_loss[local]
            centered = local_loss.detach() - local_loss.detach().mean()
            adversarial = torch.softmax(2.0 * centered, dim=0)
            weights = 0.5 * adversarial + 0.5 / len(local_np)
            cohort_losses.append((weights * local_loss).sum())
        cohort_stack = torch.stack(cohort_losses)
        cohort_centered = (
            cohort_stack.detach() - cohort_stack.detach().mean()
        )
        cohort_adversarial = torch.softmax(
            3.0 * cohort_centered, dim=0
        )
        cohort_weight = (
            0.75 * cohort_adversarial
            + 0.25 / len(cohort_losses)
        )
        classification_loss = (cohort_weight * cohort_stack).sum()
        rank_score = logits_all[..., 0] - logits_all[..., 1]
        masked_rank = rank_score.masked_fill(~valid_tensor, -1.0e9)
        has_dual = (valid_tensor & (dual_all > 0.5)).any(dim=1)
        delta = target["delta"][in_state]
        utility = -(delta[..., 0] + delta[..., 3])
        safe_utility = np.where(
            local_valid & (target["dual"][in_state] > 0.5),
            utility,
            -np.inf,
        )
        best = torch.from_numpy(
            safe_utility.argmax(axis=1).astype(np.int64)
        ).to(device)
        listwise_loss = F.cross_entropy(
            masked_rank[has_dual], best[has_dual]
        )
        selection_probability = torch.softmax(
            masked_rank / 0.5, dim=1
        )
        expected_harm = (
            selection_probability * harm_all
        ).sum(dim=1).mean()
        loss = (
            classification_loss
            + 0.75 * listwise_loss
            + 1.50 * expected_harm
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        last_loss = float(loss.detach().cpu())
    metadata = {
        "domain": "recon",
        "role": role,
        "seed": seed,
        "epochs": epochs,
        "train_states": int(in_state.sum()),
        "train_groups": int(len(unique_groups)),
        "train_cohorts": np.unique(cohorts[in_state]).astype(str).tolist(),
        "train_candidates": int(train_valid.sum()),
        "route_group_dro": True,
        "cohort_DRO": True,
        "listwise_safe_ranking": True,
        "scorer_v2_predictions_in_features": False,
    }
    save_calibrator(checkpoint, model, stats, metadata)
    return {
        "path": str(checkpoint),
        "sha256": file_sha256(checkpoint),
        **metadata,
        "last_loss": last_loss,
        "model": model,
        "normalization": stats,
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
    fold_id = stratified_group_folds(group, cohort, 5, 20260726)
    prediction, member_prediction = ensemble_with_members(
        training["arms"]["D"]["checkpoints"]["full"],
        universal,
        np.zeros(len(universal), dtype=np.int64),
        device=args.device,
    )
    base_risk = risk_features(
        prediction,
        member_prediction,
        support,
        robot_context=np.zeros((len(group), 2), dtype=np.float32),
    )
    risk = multimodal_risk_features(base_risk, universal)
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
        records = []
        for member_index in range(args.calibrator_members):
            record = train_calibrator(
                risk,
                valid,
                target,
                group,
                cohort,
                train_state,
                epochs=args.epochs,
                seed=20260726 + heldout * 10 + member_index,
                device=args.device,
                checkpoint=args.output_dir
                / (
                    f"checkpoints/recon_fold{heldout}"
                    f"_member{member_index}.pt"
                ),
                role=(
                    f"multimodal_group_oof_fold_{heldout}"
                    f"_member_{member_index}"
                ),
            )
            dual, harm = calibrated_probabilities_v5(
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
            records.append(record)
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
            {"heldout_fold": heldout, "members": records}
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
            cohort,
            np.ones(len(valid), dtype=bool),
            epochs=args.epochs,
            seed=20261126 + member_index,
            device=args.device,
            checkpoint=args.output_dir
            / f"checkpoints/recon_full_member{member_index}.pt",
            role=f"multimodal_deployment_full_member_{member_index}",
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
        "schema": "recon_route_multimodal_calibration_v6_listwise",
        "status": "expanded_spent_multimodal_listwise_development_complete",
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
        "checkpoints": {"oof": checkpoints, "full": full_records},
        "nested_group_oof": {
            "metrics": nested_metrics,
            "cohort_metrics": nested_cohorts,
            "target_feasible": (
                nested_metrics["coverage"] >= 0.40
                and nested_metrics["dual_precision"] >= 0.90
                and nested_metrics["dual_wilson_lower_95"] >= 0.80
                and nested_metrics["harm_rate"] <= 0.10
            ),
            "strict_spent_feasible": cohort_target_feasible(nested_cohorts),
            "policies": nested_policies,
        },
        "deployment_policy": serializable_expanded(deployment),
        "feature_contract": (
            "current-window action + DINOv3 static + raw LTWM + "
            "universal-D predictions/disagreement/support; no scorer-v2 "
            "predictions; no route/bag ID; no future-window feature"
        ),
        "optimization": {
            "cohort_weighting": "worst_cohort_DRO",
            "listwise_safe_ranking": True,
            "differentiable_expected_harm": True,
            "within_cohort": "route_group_DRO",
            "dual_false_positive_cost": 4.0,
            "harm_false_negative_cost": 5.0,
        },
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
