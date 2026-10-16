#!/usr/bin/env python3
"""Fit nested group-cross-fitted route-robust risk gates for universal D."""

from __future__ import annotations

import argparse
import hashlib
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

from scoring.route_robust_calibration_v3 import (  # noqa: E402
    RISK_FEATURE_DIM,
    RouteRiskCalibratorV3,
    RouteRiskPolicyV3,
    calibrated_probabilities,
    normalization,
    policy_dict,
    risk_features,
    risk_nonconformity,
)
from scoring.spacing_support_v1 import threshold_for_quantile  # noqa: E402
from scoring.universal_dualdomain_raw_v2 import (  # noqa: E402
    build_universal_features,
    build_universal_targets,
    set_seed,
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
    load_member,
    predict_member,
    safety_metrics,
)


ROOT = REPO_ROOT / "outputs/universal_dualdomain_raw_v2"
DOMAINS = ("recon", "scand")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--training-result",
        type=Path,
        default=ROOT / "training_v2/result.json",
    )
    parser.add_argument(
        "--recon-cal",
        type=Path,
        default=ROOT
        / "features/recon_scorer_cal/recon_scorer_cal_universal_v2_features.npz",
    )
    parser.add_argument(
        "--recon-source",
        type=Path,
        default=ROOT / "sources/recon_scorer_cal_source.npz",
    )
    parser.add_argument(
        "--scand-cal",
        type=Path,
        default=REPO_ROOT
        / "outputs/scand_domain_adaptation_v1/features/scorer_cal/"
        "scand_inner_scorer_cal_spacing038_features.npz",
    )
    parser.add_argument(
        "--scand-source",
        type=Path,
        default=REPO_ROOT
        / "outputs/scand_domain_adaptation_v1/sources/scorer_cal_source.npz",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "route_robust_v3"
    )
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--calibrator-members", type=int, default=3)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


def deterministic_group_folds(
    groups: np.ndarray, folds: int, seed: int
) -> np.ndarray:
    unique = np.unique(np.asarray(groups).astype(str))
    ordered = sorted(
        unique.tolist(),
        key=lambda value: hashlib.sha256(
            f"{seed}|{value}".encode()
        ).hexdigest(),
    )
    mapping = {value: index % folds for index, value in enumerate(ordered)}
    result = np.asarray([mapping[str(value)] for value in groups], dtype=np.int64)
    if len(np.unique(result)) != folds:
        raise RuntimeError("route fold assignment lost a fold")
    return result



def robot_feature_context(groups: np.ndarray) -> np.ndarray:
    names = np.char.lower(np.asarray(groups).astype(str))
    jackal = np.char.find(names, "jackal") >= 0
    spot = np.char.find(names, "spot") >= 0
    return np.stack((jackal, spot), axis=-1).astype(np.float32)


def ensemble_with_members(
    records: list[dict[str, Any]],
    feature: np.ndarray,
    domain: np.ndarray,
    *,
    device: str,
) -> tuple[dict[str, np.ndarray], list[dict[str, np.ndarray]]]:
    members = []
    for record in records:
        model = load_member(record, device)
        members.append(predict_member(model, feature, domain, device=device))
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
    means = np.stack([value["mean"] for value in members])
    variances = np.stack([value["variance"] for value in members])
    mean = means.mean(axis=0)
    prediction = {
        "mean": mean.astype(np.float32),
        "std": np.sqrt(
            np.maximum(
                variances.mean(axis=0)
                + np.square(means).mean(axis=0)
                - np.square(mean),
                1.0e-8,
            )
        ).astype(np.float32),
    }
    for key in (
        "helpful_logit",
        "dual_logit",
        "harm_logit",
        "rank_logit",
    ):
        prediction[key] = np.stack(
            [value[key] for value in members]
        ).mean(axis=0).astype(np.float32)
    return prediction, members


def save_calibrator(
    path: Path,
    model: RouteRiskCalibratorV3,
    stats: dict[str, np.ndarray],
    metadata: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "schema": "route_robust_risk_calibrator_v3",
            "input_dim": RISK_FEATURE_DIM,
            "hidden_dim": 48,
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
    train_state: np.ndarray,
    *,
    epochs: int,
    seed: int,
    device: str,
    checkpoint: Path,
    role: str,
    domain: str,
) -> dict[str, Any]:
    set_seed(seed)
    in_state = np.asarray(train_state, dtype=bool)
    train_valid = np.asarray(valid, dtype=bool) & in_state[:, None]
    train_valid[:, 0] = False
    stats = normalization(feature[in_state], valid[in_state])
    x = torch.from_numpy(
        (
            (feature - stats["feature_center"])
            / stats["feature_scale"]
        )[train_valid]
        .clip(-12.0, 12.0)
        .astype(np.float32)
    ).to(device)
    y = torch.from_numpy(
        np.stack(
            (target["dual"][train_valid], target["harm"][train_valid]),
            axis=-1,
        ).astype(np.float32)
    ).to(device)
    flat_groups = np.repeat(np.asarray(groups).astype(str)[:, None], valid.shape[1], axis=1)[
        train_valid
    ]
    _, inverse = np.unique(flat_groups, return_inverse=True)
    group_index = torch.from_numpy(inverse.astype(np.int64)).to(device)
    group_count = int(group_index.max().item()) + 1
    count = torch.bincount(group_index, minlength=group_count).float()

    model = RouteRiskCalibratorV3().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1.0e-3, weight_decay=5.0e-4
    )
    last_loss = 0.0
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        element = F.binary_cross_entropy_with_logits(
            logits, y, reduction="none"
        )
        dual_weight = torch.where(y[:, 0] > 0.5, 1.0, 2.0)
        harm_weight = torch.where(y[:, 1] > 0.5, 3.0, 1.0)
        candidate_loss = (
            element[:, 0] * dual_weight
            + 1.5 * element[:, 1] * harm_weight
        )
        group_sum = torch.zeros(group_count, device=device)
        group_sum.scatter_add_(0, group_index, candidate_loss)
        group_loss = group_sum / count
        centered = group_loss.detach() - group_loss.detach().mean()
        adversarial = torch.softmax(2.0 * centered, dim=0)
        weights = 0.5 * adversarial + 0.5 / group_count
        loss = (weights * group_loss).sum()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        last_loss = float(loss.detach().cpu())
    save_calibrator(
        checkpoint,
        model,
        stats,
        {
            "domain": domain,
            "role": role,
            "seed": seed,
            "epochs": epochs,
            "train_states": int(in_state.sum()),
            "train_groups": int(len(np.unique(groups[in_state]))),
            "train_candidates": int(train_valid.sum()),
            "group_dro": True,
            "scorer_v2_predictions_in_features": False,
        },
    )
    return {
        "path": str(checkpoint),
        "sha256": file_sha256(checkpoint),
        "role": role,
        "seed": seed,
        "epochs": epochs,
        "train_states": int(in_state.sum()),
        "train_groups": int(len(np.unique(groups[in_state]))),
        "train_candidates": int(train_valid.sum()),
        "last_loss": last_loss,
        "model": model,
        "normalization": stats,
    }


def challenger_for_rank(
    prediction: dict[str, np.ndarray],
    p_dual: np.ndarray,
    p_harm: np.ndarray,
    valid: np.ndarray,
    *,
    harm_weight: float,
    uncertainty_weight: float,
) -> tuple[np.ndarray, np.ndarray]:
    allowed = np.asarray(valid, dtype=bool).copy()
    allowed[:, 0] = False
    uncertainty = prediction["std"][..., 0] + prediction["std"][..., 3]
    rank = p_dual - harm_weight * p_harm - uncertainty_weight * uncertainty
    challenger = np.where(allowed, rank, -np.inf).argmax(axis=1)
    exists = allowed.any(axis=1)
    challenger = np.where(exists, challenger, 0).astype(np.int64)
    return challenger, exists


def select_challenger(
    prediction: dict[str, np.ndarray],
    p_dual: np.ndarray,
    p_harm: np.ndarray,
    risk_ood: np.ndarray,
    calibrator_disagreement: np.ndarray,
    challenger: np.ndarray,
    exists: np.ndarray,
    policy: RouteRiskPolicyV3,
) -> np.ndarray:
    row = np.arange(len(challenger))
    ucb_ate = (
        prediction["mean"][row, challenger, 0]
        + policy.z * prediction["std"][row, challenger, 0]
    )
    ucb_objective = (
        prediction["mean"][row, challenger, 3]
        + policy.z * prediction["std"][row, challenger, 3]
    )
    promote = (
        exists
        & (p_dual[row, challenger] >= policy.dual_threshold)
        & (p_harm[row, challenger] <= policy.harm_threshold)
        & (risk_ood[row, challenger] <= policy.risk_ood_threshold)
        & (
            calibrator_disagreement[row, challenger]
            <= policy.calibrator_disagreement_threshold
        )
        & (ucb_ate < -policy.margin)
        & (ucb_objective < -policy.margin)
    )
    return np.where(promote, challenger, 0).astype(np.int64)


def fold_safety(
    selected: np.ndarray,
    targets: np.ndarray,
    fold_id: np.ndarray,
) -> list[dict[str, Any]]:
    return [
        safety_metrics(selected[fold_id == fold], targets[fold_id == fold])
        for fold in np.unique(fold_id)
    ]


def reserve_feasible(
    metrics: dict[str, Any], folds: list[dict[str, Any]]
) -> bool:
    return (
        metrics["coverage"] >= 0.40
        and metrics["dual_precision"] is not None
        and metrics["dual_precision"] >= 0.92
        and metrics["dual_wilson_lower_95"] is not None
        and metrics["dual_wilson_lower_95"] >= 0.82
        and metrics["harm_rate"] is not None
        and metrics["harm_rate"] <= 0.08
        and min(value["coverage"] for value in folds) >= 0.25
        and min(value["dual_precision"] or 0.0 for value in folds) >= 0.85
        and max(value["harm_rate"] or 1.0 for value in folds) <= 0.15
    )


def target_feasible(metrics: dict[str, Any]) -> bool:
    return (
        metrics["coverage"] >= 0.40
        and metrics["dual_precision"] is not None
        and metrics["dual_precision"] >= 0.90
        and metrics["dual_wilson_lower_95"] is not None
        and metrics["dual_wilson_lower_95"] >= 0.80
        and metrics["harm_rate"] is not None
        and metrics["harm_rate"] <= 0.10
    )


def robust_policy_penalty(
    metrics: dict[str, Any], folds: list[dict[str, Any]]
) -> float:
    fold_coverage = [value["coverage"] for value in folds]
    fold_dual = [value["dual_precision"] or 0.0 for value in folds]
    fold_harm = [value["harm_rate"] if value["harm_rate"] is not None else 1.0 for value in folds]
    return float(
        2.0 * max(0.0, 0.92 - (metrics["dual_precision"] or 0.0))
        + 2.0 * max(
            0.0, 0.82 - (metrics["dual_wilson_lower_95"] or 0.0)
        )
        + 2.0 * max(0.0, (metrics["harm_rate"] or 1.0) - 0.08)
        + max(0.0, 0.25 - min(fold_coverage))
        + 2.0 * max(0.0, 0.85 - min(fold_dual))
        + 2.0 * max(0.0, max(fold_harm) - 0.15)
        + 0.10 * float(np.std(fold_dual))
    )


def policy_search(
    prediction: dict[str, np.ndarray],
    p_dual: np.ndarray,
    p_harm: np.ndarray,
    risk_ood: np.ndarray,
    calibrator_disagreement: np.ndarray,
    valid: np.ndarray,
    targets: np.ndarray,
    fold_id: np.ndarray,
) -> dict[str, Any]:
    records = []
    minimum_coverage = 0.40
    row = np.arange(len(valid))
    for harm_weight in (0.5, 1.0, 2.0):
        for uncertainty_weight in (0.0, 0.10, 0.25):
            challenger, exists = challenger_for_rank(
                prediction,
                p_dual,
                p_harm,
                valid,
                harm_weight=harm_weight,
                uncertainty_weight=uncertainty_weight,
            )
            challenger_ood = risk_ood[row, challenger]
            ood_reference = challenger_ood[
                exists & np.isfinite(challenger_ood)
            ]
            challenger_disagreement = calibrator_disagreement[
                row, challenger
            ]
            disagreement_reference = challenger_disagreement[
                exists & np.isfinite(challenger_disagreement)
            ]
            if not len(ood_reference) or not len(disagreement_reference):
                raise RuntimeError("no finite calibration-reference risk score")
            ood_thresholds = [
                float(np.quantile(ood_reference, 0.975))
            ]
            disagreement_thresholds = [
                float(np.quantile(disagreement_reference, 0.90))
            ]
            for risk_ood_threshold in ood_thresholds:
                for disagreement_threshold in disagreement_thresholds:
                    for z in (0.0, 0.5, 1.0, 1.5):
                        for margin in (0.0, 0.0025, 0.005, 0.010):
                            for dual_threshold in (
                                0.30,
                                0.40,
                                0.50,
                                0.60,
                                0.70,
                                0.80,
                                0.85,
                                0.90,
                                0.95,
                            ):
                                for harm_threshold in (
                                    0.05,
                                    0.10,
                                    0.15,
                                    0.20,
                                    0.30,
                                    0.40,
                                    0.50,
                                    0.60,
                                ):
                                    policy = RouteRiskPolicyV3(
                                        harm_rank_weight=harm_weight,
                                        uncertainty_rank_weight=(
                                            uncertainty_weight
                                        ),
                                        z=z,
                                        margin=margin,
                                        dual_threshold=dual_threshold,
                                        harm_threshold=harm_threshold,
                                        risk_ood_threshold=(
                                            risk_ood_threshold
                                        ),
                                        calibrator_disagreement_threshold=(
                                            disagreement_threshold
                                        ),
                                    )
                                    selected = select_challenger(
                                        prediction,
                                        p_dual,
                                        p_harm,
                                        risk_ood,
                                        calibrator_disagreement,
                                        challenger,
                                        exists,
                                        policy,
                                    )
                                    metrics = safety_metrics(
                                        selected, targets
                                    )
                                    if metrics["coverage"] < minimum_coverage:
                                        continue
                                    folds = fold_safety(
                                        selected, targets, fold_id
                                    )
                                    reserve = reserve_feasible(
                                        metrics, folds
                                    )
                                    target = target_feasible(metrics)
                                    records.append(
                                        {
                                            "policy": policy,
                                            "selected": selected,
                                            "metrics": metrics,
                                            "fold_metrics": folds,
                                            "reserve_feasible": reserve,
                                            "target_feasible": target,
                                            "penalty": robust_policy_penalty(
                                                metrics, folds
                                            ),
                                        }
                                    )
    if not records:
        raise RuntimeError("no route-risk policy reaches coverage 0.40")
    reserve_pool = [
        value for value in records if value["reserve_feasible"]
    ]
    target_pool = [value for value in records if value["target_feasible"]]
    if reserve_pool:
        pool = reserve_pool
        selection_tier = "reserve"
        key = lambda value: (
            -value["metrics"]["coverage"],
            value["metrics"]["net_delta_per_state_m"]["objective"],
            value["penalty"],
        )
    elif target_pool:
        pool = target_pool
        selection_tier = "deployment_target"
        key = lambda value: (
            value["penalty"],
            -value["metrics"]["coverage"],
            value["metrics"]["net_delta_per_state_m"]["objective"],
        )
    else:
        pool = records
        selection_tier = "best_available"
        key = lambda value: (
            value["penalty"],
            -value["metrics"]["coverage"],
            value["metrics"]["net_delta_per_state_m"]["objective"],
        )
    best = min(pool, key=key)
    return {
        **best,
        "selection_tier": selection_tier,
        "reserve_frontier_feasible": bool(reserve_pool),
        "target_frontier_feasible": bool(target_pool),
        "searched_policies": len(records),
    }


def serializable_search(value: dict[str, Any]) -> dict[str, Any]:
    return {
        "policy": policy_dict(value["policy"]),
        "metrics": value["metrics"],
        "fold_metrics": value["fold_metrics"],
        "reserve_feasible": value["reserve_feasible"],
        "target_feasible": value["target_feasible"],
        "reserve_frontier_feasible": value["reserve_frontier_feasible"],
        "target_frontier_feasible": value["target_frontier_feasible"],
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
    if args.epochs < 50:
        raise ValueError("route-risk calibrator needs at least 50 epochs")
    if args.calibrator_members < 2:
        raise ValueError("route-risk ensemble needs at least two members")
    training = json.loads(args.training_result.read_text(encoding="utf-8"))
    if (
        training.get("cem") is not False
        or not training.get("primary_excludes_scorer_v2_predictions")
    ):
        raise RuntimeError("base universal D provenance is invalid")
    envelopes = load_envelopes(Path(training["support"]))
    inputs = {
        "recon": (args.recon_cal, args.recon_source, 0.25, 0),
        "scand": (args.scand_cal, args.scand_source, 0.38, 1),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    results = {}
    for domain_offset, domain in enumerate(DOMAINS):
        feature_path, source_path, spacing, domain_index = inputs[domain]
        data = load_features(feature_path, labels_expected=True)
        group = load_group(source_path, data["record_key"], domain)
        folds = min(5, len(np.unique(group)))
        fold_id = deterministic_group_folds(
            group, folds, 20260724 + domain_offset
        )
        universal, support_score, guard = build_universal_features(
            data,
            spacing_m=spacing,
            support_envelope=envelopes[domain],
        )
        valid = guard & (
            support_score
            <= threshold_for_quantile(envelopes[domain], 0.999)
        )
        valid[:, 0] = True
        prediction, member_prediction = ensemble_with_members(
            training["arms"]["D"]["checkpoints"]["full"],
            universal,
            np.full(len(universal), domain_index, dtype=np.int64),
            device=args.device,
        )
        risk = risk_features(
            prediction,
            member_prediction,
            support_score,
            robot_context=robot_feature_context(group),
        )
        target = build_universal_targets(data["targets"])
        oof_dual = np.empty(valid.shape, dtype=np.float32)
        oof_harm = np.empty(valid.shape, dtype=np.float32)
        oof_risk_ood = np.empty(valid.shape, dtype=np.float32)
        oof_calibrator_disagreement = np.empty(
            valid.shape, dtype=np.float32
        )
        checkpoints = []
        for heldout in range(folds):
            train_state = fold_id != heldout
            test = fold_id == heldout
            member_records = []
            dual_members = []
            harm_members = []
            risk_ood_members = []
            for member_index in range(args.calibrator_members):
                record = train_calibrator(
                    risk,
                    valid,
                    target,
                    group,
                    train_state,
                    epochs=args.epochs,
                    seed=(
                        20260724
                        + domain_offset * 1000
                        + heldout * 10
                        + member_index
                    ),
                    device=args.device,
                    checkpoint=args.output_dir
                    / (
                        f"checkpoints/{domain}_fold{heldout}"
                        f"_member{member_index}.pt"
                    ),
                    role=(
                        f"group_oof_fold_{heldout}"
                        f"_member_{member_index}"
                    ),
                    domain=domain,
                )
                dual, harm = calibrated_probabilities(
                    record["model"],
                    risk[test],
                    record["normalization"],
                    device=args.device,
                )
                dual_members.append(dual)
                harm_members.append(harm)
                risk_ood_members.append(
                    risk_nonconformity(
                        risk[test], record["normalization"]
                    )
                )
                record.pop("model")
                record.pop("normalization")
                member_records.append(record)
            dual_stack = np.stack(dual_members)
            harm_stack = np.stack(harm_members)
            oof_dual[test] = np.clip(
                dual_stack.mean(axis=0) - dual_stack.std(axis=0),
                0.0,
                1.0,
            )
            oof_harm[test] = np.clip(
                harm_stack.mean(axis=0) + harm_stack.std(axis=0),
                0.0,
                1.0,
            )
            oof_calibrator_disagreement[test] = (
                dual_stack.std(axis=0) + harm_stack.std(axis=0)
            )
            oof_risk_ood[test] = np.stack(
                risk_ood_members
            ).max(axis=0)
            checkpoints.append(
                {
                    "heldout_fold": heldout,
                    "aggregation": (
                        "p_dual=mean-1std;p_harm=mean+1std;"
                        "risk_ood=max"
                    ),
                    "members": member_records,
                }
            )

        nested_selected = np.zeros(len(valid), dtype=np.int64)
        nested_policies = []
        for heldout in range(folds):
            fit = fold_id != heldout
            search = policy_search(
                {key: value[fit] for key, value in prediction.items()},
                oof_dual[fit],
                oof_harm[fit],
                oof_risk_ood[fit],
                oof_calibrator_disagreement[fit],
                valid[fit],
                data["targets"][fit],
                fold_id[fit],
            )
            test = fold_id == heldout
            challenger, exists = challenger_for_rank(
                {key: value[test] for key, value in prediction.items()},
                oof_dual[test],
                oof_harm[test],
                valid[test],
                harm_weight=search["policy"].harm_rank_weight,
                uncertainty_weight=search[
                    "policy"
                ].uncertainty_rank_weight,
            )
            nested_selected[test] = select_challenger(
                {key: value[test] for key, value in prediction.items()},
                oof_dual[test],
                oof_harm[test],
                oof_risk_ood[test],
                oof_calibrator_disagreement[test],
                challenger,
                exists,
                search["policy"],
            )
            nested_policies.append(
                {
                    "heldout_fold": heldout,
                    "fit": serializable_search(search),
                }
            )
        nested_metrics = safety_metrics(
            nested_selected, data["targets"]
        )
        deployment_search = policy_search(
            prediction,
            oof_dual,
            oof_harm,
            oof_risk_ood,
            oof_calibrator_disagreement,
            valid,
            data["targets"],
            fold_id,
        )
        full_records = []
        for member_index in range(args.calibrator_members):
            full_record = train_calibrator(
                risk,
                valid,
                target,
                group,
                np.ones(len(valid), dtype=bool),
                epochs=args.epochs,
                seed=20260924 + domain_offset * 100 + member_index,
                device=args.device,
                checkpoint=args.output_dir
                / f"checkpoints/{domain}_full_member{member_index}.pt",
                role=f"deployment_full_scorer_cal_member_{member_index}",
                domain=domain,
            )
            full_record.pop("model")
            full_record.pop("normalization")
            full_records.append(full_record)
        arrays[f"{domain}__record_key"] = data["record_key"].astype(str)
        arrays[f"{domain}__fold_id"] = fold_id
        arrays[f"{domain}__nested_selected"] = nested_selected
        arrays[f"{domain}__deployment_oof_selected"] = deployment_search[
            "selected"
        ]
        arrays[f"{domain}__oof_p_dual"] = oof_dual
        arrays[f"{domain}__oof_p_harm"] = oof_harm
        arrays[f"{domain}__oof_risk_ood"] = oof_risk_ood
        arrays[
            f"{domain}__oof_calibrator_disagreement"
        ] = oof_calibrator_disagreement
        results[domain] = {
            "states": len(valid),
            "groups": int(len(np.unique(group))),
            "folds": folds,
            "checkpoints": {
                "oof": checkpoints,
                "full": full_records,
            },
            "calibrator_ensemble": {
                "members": args.calibrator_members,
                "p_dual": "mean_minus_one_std",
                "p_harm": "mean_plus_one_std",
                "risk_ood": "member_max",
            },
            "nested_group_oof": {
                "metrics": nested_metrics,
                "target_feasible": target_feasible(nested_metrics),
                "policies": nested_policies,
            },
            "deployment_policy": serializable_search(deployment_search),
            "support_candidate_acceptance": float(valid[:, 1:].mean()),
        }
        print(
            json.dumps(
                {
                    "domain": domain,
                    "nested_group_oof": nested_metrics,
                    "deployment_policy": serializable_search(
                        deployment_search
                    ),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    selected_path = args.output_dir / "calibration_predictions.npz"
    write_npz(selected_path, **arrays)
    report = {
        "schema": "universal_route_robust_calibration_v3",
        "status": "nested_group_crossfit_route_risk_calibrated",
        "base_training_result": str(args.training_result),
        "base_training_result_sha256": file_sha256(args.training_result),
        "data_sha256": {
            "recon_cal": file_sha256(args.recon_cal),
            "scand_cal": file_sha256(args.scand_cal),
            "recon_source": file_sha256(args.recon_source),
            "scand_source": file_sha256(args.scand_source),
        },
        "feature_contract": (
            "universal D predictions + three-member disagreement + "
            "support nonconformity + robot token + fold-reference "
            "risk nonconformity; no scorer-v2 predictions"
        ),
        "calibrator_ensemble": {
            "members": args.calibrator_members,
            "conservative_probability_rule": (
                "p_dual=mean-1std;p_harm=mean+1std"
            ),
            "cost_weighting": (
                "dual_false_positive_x2;harm_false_negative_x3"
            ),
        },
        "causal_deployment_audit": {
            "decision_scope": "current_window_only",
            "risk_reference": "scorer-cal training-fold robust normalization",
            "threshold_source": (
                "fixed calibration-reference OOD q0.975 and "
                "calibrator-disagreement q0.90"
            ),
            "heldout_route_outcomes_used_for_threshold": False,
            "outer_route_future_windows_used": False,
            "full_route_quota_or_ranking": False,
        },
        "calibration_reserve": {
            "dual_precision": 0.92,
            "dual_wilson_lower_95": 0.82,
            "coverage": 0.40,
            "harm_rate": 0.08,
            "minimum_fold_dual_precision": 0.85,
            "maximum_fold_harm_rate": 0.15,
            "minimum_fold_coverage": 0.25,
        },
        "deployment_target": {
            "dual_precision": 0.90,
            "dual_wilson_lower_95": 0.80,
            "coverage": 0.40,
            "harm_rate": 0.10,
        },
        "domains": results,
        "calibration_predictions": str(selected_path),
        "calibration_predictions_sha256": file_sha256(selected_path),
        "outer_labels_accessed": False,
        "cem_cal_accessed": False,
        "fixed_pool_only": True,
        "cem": False,
    }
    write_json(result_path, report)
    print(
        json.dumps(
            {
                "result": str(result_path),
                "result_sha256": file_sha256(result_path),
                "nested_target_feasible": {
                    domain: results[domain]["nested_group_oof"][
                        "target_feasible"
                    ]
                    for domain in DOMAINS
                },
                "deployment_reserve_feasible": {
                    domain: results[domain]["deployment_policy"][
                        "reserve_frontier_feasible"
                    ]
                    for domain in DOMAINS
                },
                "cem": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
