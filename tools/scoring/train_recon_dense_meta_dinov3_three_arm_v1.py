#!/usr/bin/env python3
"""Train and freeze raw, scorer-v2-stacked, and native scorer-v2 arms."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from scoring.dense_meta_v1 import (  # noqa: E402
    DenseHeadConfig,
    MODES,
    build_meta_features,
    build_targets,
    empty_prediction,
    ensemble_predict,
    family_fold,
    feature_by_mode,
    fit_fold,
    predict,
    predict_ridge,
    residual_rows,
    selection_metrics,
)
from scoring.safe_promotion_v1 import greedy_fixed_pool_selection  # noqa: E402
from tools.scoring.train_recon_dense_meta_dinov3_v1 import (  # noqa: E402
    choose_meta_model,
    choose_residual_policy,
    file_sha256,
    meta_selection,
    meta_target,
    metrics_json,
    policy_selections,
    serializable_ridge,
    store_prediction,
    write_json,
    write_npz,
)


DEFAULT_ROOT = REPO_ROOT / "outputs/recon_dense_meta_dinov3_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-features", type=Path, default=DEFAULT_ROOT / "features/dense_train_features.npz")
    parser.add_argument("--cal-features", type=Path, default=DEFAULT_ROOT / "features/dense_cal_features.npz")
    parser.add_argument("--test-features", type=Path, default=DEFAULT_ROOT / "features/test3_features.npz")
    parser.add_argument("--cal-label-source", type=Path, default=Path("/home/rainbow/ndwm/DWM/reports/recon_h4_dense_aggregate_incumbent_features_v1/dense_cal/aggregate_incumbent_features.npz"))
    parser.add_argument("--test-label-source", type=Path, default=Path("/home/rainbow/ndwm/DWM/reports/recon_h4_dense_meta_test3_features_v1/test3_aggregate_incumbent_features.npz"))
    parser.add_argument("--scorer-train-manifest", type=Path, default=REPO_ROOT / "data/inner_manifests/recon/train-fit.windows.jsonl")
    parser.add_argument("--checkpoint-root", type=Path, default=REPO_ROOT / "checkpoints/project/scorer_v2")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ROOT / "three_arm_training")
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


def load_features(path: Path, *, labels_expected: bool) -> dict[str, Any]:
    required = {
        "candidates", "actions_h4", "action_features", "static_features",
        "raw_ltwm_features", "scorer_v2_aux_features", "support_features",
        "ate_by_seed", "rtp_by_seed", "support_by_seed", "majority_support",
        "record_key", "family", "metadata",
    }
    if labels_expected:
        required.add("targets")
    with np.load(path, allow_pickle=False) as payload:
        missing = required.difference(payload.files)
        if missing:
            raise RuntimeError(f"{path} missing arrays: {sorted(missing)}")
        data = {key: payload[key].copy() for key in required}
    metadata = json.loads(str(data.pop("metadata").item()))
    if metadata.get("schema") != "recon_dense_dinov3_feature_artifact_v1":
        raise RuntimeError("DINOv3 dense feature schema mismatch")
    if bool(metadata.get("labels_in_output")) != labels_expected:
        raise RuntimeError("feature label-presence contract mismatch")
    states, candidates = data["candidates"].shape[:2]
    if candidates != 17 or data["majority_support"].shape != (states, candidates):
        raise RuntimeError("dense feature candidate contract mismatch")
    expected_support = data["support_by_seed"].mean(axis=0)[..., None]
    if not np.allclose(data["support_features"], expected_support):
        raise RuntimeError("support feature/source mismatch")
    return {**data, "metadata": metadata, "path": path}


def load_targets(path: Path, expected: dict[str, Any]) -> np.ndarray:
    if file_sha256(path) != expected["metadata"]["source_sha256"]:
        raise RuntimeError("label source differs from prelabel feature source")
    with np.load(path, allow_pickle=False) as payload:
        record_key = payload["record_key"].astype(str)
        family = payload["family"].astype(str)
        targets = payload["targets"].astype(np.float32)
    if not np.array_equal(record_key, expected["record_key"].astype(str)):
        raise RuntimeError("label record order mismatch")
    if not np.array_equal(family, expected["family"].astype(str)):
        raise RuntimeError("label family order mismatch")
    if targets.shape != (*expected["candidates"].shape[:2], 8):
        raise RuntimeError("label target shape mismatch")
    return targets


def trajectory_and_timestep(record_key: str) -> tuple[str, int]:
    trajectory, separator, timestep = str(record_key).rpartition(":")
    if not separator or not timestep.isdigit():
        raise ValueError(f"invalid record key: {record_key}")
    return trajectory, int(timestep)


def scorer_overlap_audit(data: dict[str, Any], manifest: Path) -> tuple[np.ndarray, dict[str, Any]]:
    fit_trajectories: set[str] = set()
    fit_windows: set[tuple[str, int]] = set()
    lines = 0
    with manifest.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            value = json.loads(line)
            trajectory = str(value["trajectory_id"])
            fit_trajectories.add(trajectory)
            fit_windows.add((trajectory, int(value["current_t"])))
            lines += 1
    records = [trajectory_and_timestep(value) for value in data["record_key"].astype(str)]
    trajectory_overlap = np.asarray([value[0] in fit_trajectories for value in records])
    exact_overlap = np.asarray([value in fit_windows for value in records])
    families = data["family"].astype(str)
    contaminated_families = set(families[trajectory_overlap])
    clean = np.asarray([value not in contaminated_families for value in families])
    audit = {
        "manifest": str(manifest),
        "manifest_sha256": file_sha256(manifest),
        "manifest_windows": lines,
        "manifest_trajectories": len(fit_trajectories),
        "states": int(len(families)),
        "families": int(len(np.unique(families))),
        "exact_window_overlap_states": int(exact_overlap.sum()),
        "trajectory_overlap_states": int(trajectory_overlap.sum()),
        "trajectory_overlap_families": int(len(contaminated_families)),
        "strict_clean_states": int(clean.sum()),
        "strict_clean_families": int(len(np.unique(families[clean]))),
        "stacked_training_rule": "exclude every family containing any scorer-v2 train-fit trajectory",
    }
    return clean, audit


def subset_state_data(data: dict[str, Any], index: np.ndarray) -> dict[str, np.ndarray]:
    return {
        key: np.asarray(data[key])[index]
        for key in ("action_features", "static_features", "raw_ltwm_features", "scorer_v2_aux_features")
    }


def checkpoint_payload(*, label: str, mode: str, variant: str, fold: int, seed: int,
                       feature_dim: int, config: DenseHeadConfig, stats: dict[str, np.ndarray],
                       model: torch.nn.Module, train_sha256: str, train_states: int) -> dict[str, Any]:
    return {
        "schema": "recon_dense_meta_dinov3_groupfirst_member_v2",
        "label": label,
        "mode": mode,
        "feature_variant": variant,
        "heldout_family_fold": fold,
        "seed": seed,
        "input_dim": feature_dim,
        "config": asdict(config),
        "normalization": stats,
        "state_dict": model.state_dict(),
        "train_feature_sha256": train_sha256,
        "training_universe_states": train_states,
        "scorer_v2_predictions_in_features": variant == "v2_stacked" and mode in {"latent", "full"},
        "support_used_only_as_candidate_validity_mask": True,
        "fixed_pool_only": True,
        "cem": False,
    }


def train_head(*, label: str, mode: str, variant: str, universe: np.ndarray,
               train: dict[str, Any], target: dict[str, np.ndarray], fold_all: np.ndarray,
               config: DenseHeadConfig, args: argparse.Namespace, seed_offset: int,
               train_sha: str) -> tuple[dict[str, np.ndarray], list[tuple[torch.nn.Module, dict[str, np.ndarray]]], list[dict[str, Any]]]:
    feature_all = feature_by_mode(train, mode, variant=variant)
    feature = feature_all[universe]
    target_subset = {key: value[universe] for key, value in target.items()}
    fold = fold_all[universe]
    states, candidates = feature.shape[:2]
    oof = empty_prediction(states, candidates)
    members = []
    records = []
    for heldout in range(args.folds):
        fit_index = np.where(fold != heldout)[0]
        heldout_index = np.where(fold == heldout)[0]
        if not len(fit_index) or not len(heldout_index):
            raise RuntimeError(f"empty {label} fold {heldout}")
        seed = args.seed + seed_offset + heldout
        model, stats = fit_fold(feature, target_subset, fit_index, config, seed=seed, device=args.device)
        store_prediction(oof, heldout_index, predict(model, stats, feature, heldout_index, device=args.device))
        path = args.output_dir / "checkpoints" / f"{label}_fold{heldout}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint_payload(label=label, mode=mode, variant=variant, fold=heldout,
                                      seed=seed, feature_dim=feature.shape[-1], config=config,
                                      stats=stats, model=model, train_sha256=train_sha,
                                      train_states=len(universe)), path)
        members.append((model, stats))
        records.append({"fold": heldout, "path": str(path), "sha256": file_sha256(path)})
        print(json.dumps({"stage": "fit_groupfirst", "label": label, "fold": heldout}), flush=True)
    return oof, members, records


def rows_without_labels(predictions: dict[str, dict[str, np.ndarray]], valid: np.ndarray,
                        policies: dict[str, dict[str, Any]]) -> dict[str, dict[str, np.ndarray]]:
    return {
        mode: residual_rows(predictions["action"], predictions[mode],
                            alpha=float(policies[mode]["alpha"]), gate=str(policies[mode]["gate"]),
                            risk=str(policies[mode]["risk"]), valid=valid)
        for mode in MODES
    }


def fit_branch(*, name: str, variant: str, universe: np.ndarray,
               predictions: dict[str, dict[str, np.ndarray]], train: dict[str, Any],
               fold_all: np.ndarray, args: argparse.Namespace) -> dict[str, Any]:
    targets = train["targets"][universe]
    valid = train["majority_support"][universe]
    policies = {}
    rows = {}
    search = {}
    for mode in MODES:
        policy, selected_rows, summary = choose_residual_policy(
            predictions["action"], predictions[mode], targets, valid, args, mode=mode)
        policies[mode] = policy
        rows[mode] = selected_rows
        search[mode] = summary
    selections = policy_selections(rows, policies)
    data_subset = subset_state_data(train, universe)
    meta_feature = build_meta_features(predictions, rows, data_subset, variant=variant)
    meta_outcome = meta_target(selections, targets)
    meta_model, meta_oof_score, meta_summary = choose_meta_model(
        meta_feature, meta_outcome, fold_all[universe], selections, targets, args)
    meta_selected = meta_selection(meta_oof_score, float(meta_model["threshold"]), selections)
    return {
        "name": name,
        "variant": variant,
        "universe": universe,
        "policies": policies,
        "search": search,
        "meta_model": meta_model,
        "meta_summary": meta_summary,
        "meta_feature": meta_feature,
        "meta_target": meta_outcome,
        "meta_oof_score": meta_oof_score,
        "meta_selected": meta_selected,
        "base_metrics": {mode: metrics_json(selection_metrics(selections[mode], targets)) for mode in MODES},
        "meta_metrics": metrics_json(selection_metrics(meta_selected, targets)),
    }


def predict_branch(*, branch: dict[str, Any], predictions: dict[str, dict[str, np.ndarray]],
                   data: dict[str, Any]) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
    rows = rows_without_labels(predictions, data["majority_support"], branch["policies"])
    selections = policy_selections(rows, branch["policies"])
    feature = build_meta_features(predictions, rows, subset_state_data(data, np.arange(len(data["family"]))),
                                  variant=branch["variant"])
    score = predict_ridge(branch["meta_model"], feature)
    selected = meta_selection(score, float(branch["meta_model"]["threshold"]), selections)
    return selected, selections, score


def scorer_scales(checkpoint_root: Path) -> tuple[float, float, list[dict[str, Any]]]:
    values = []
    records = []
    for seed in (1701, 1702, 1703):
        path = checkpoint_root / f"seed{seed}/best.pt"
        payload = torch.load(path, map_location="cpu", weights_only=False)
        loss = payload["loss_config"]
        values.append((float(loss["ate_scale"]), float(loss["rtp_scale"])))
        records.append({"seed": seed, "path": str(path), "sha256": file_sha256(path),
                        "action_support": payload.get("action_support")})
    if len(set(values)) != 1:
        raise RuntimeError(f"scorer-v2 scales differ by seed: {values}")
    return values[0][0], values[0][1], records


def support_audit(data: dict[str, Any]) -> dict[str, Any]:
    return {
        "states": int(len(data["family"])),
        "candidate_acceptance_fraction": float(data["support_by_seed"][:, :, 1:].mean()),
        "majority_acceptance_fraction": float(data["majority_support"][:, 1:].mean()),
        "source": "deterministic action_support constraints stored in the three scorer-v2 checkpoints",
        "role": "candidate validity mask only; excluded from raw and stacked learned feature matrices",
    }


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    config = DenseHeadConfig(hidden=args.hidden, dropout=args.dropout, epochs=args.epochs,
                             batch_groups=args.batch_groups, learning_rate=args.learning_rate,
                             weight_decay=args.weight_decay, target_temperature=args.target_temperature)
    config.validate()
    train = load_features(args.train_features, labels_expected=True)
    cal = load_features(args.cal_features, labels_expected=False)
    test = load_features(args.test_features, labels_expected=False)
    split_names = (train["metadata"]["split"], cal["metadata"]["split"], test["metadata"]["split"])
    if split_names != ("dense_train", "dense_cal", "test3"):
        raise RuntimeError(f"split names changed: {split_names}")
    family_sets = [set(value["family"].astype(str)) for value in (train, cal, test)]
    if (family_sets[0] & family_sets[1]) or (family_sets[0] & family_sets[2]) or (family_sets[1] & family_sets[2]):
        raise RuntimeError("train/cal/test families overlap")
    train_clean, train_overlap = scorer_overlap_audit(train, args.scorer_train_manifest)
    _, cal_overlap = scorer_overlap_audit(cal, args.scorer_train_manifest)
    _, test_overlap = scorer_overlap_audit(test, args.scorer_train_manifest)
    if cal_overlap["trajectory_overlap_states"] or test_overlap["trajectory_overlap_states"]:
        raise RuntimeError("scorer-v2 train-fit overlaps cal/test trajectories")
    all_index = np.arange(len(train["family"]), dtype=np.int64)
    stacked_index = np.flatnonzero(train_clean)
    if len(np.unique(train["family"][stacked_index])) < 80:
        raise RuntimeError("too few leakage-clean families for stacking")
    family_to_fold = {family: family_fold(str(family), args.folds) for family in np.unique(train["family"].astype(str))}
    fold_all = np.asarray([family_to_fold[value] for value in train["family"].astype(str)], dtype=np.int64)
    target = build_targets(train["targets"], config.target_temperature)
    args.output_dir.mkdir(parents=True)
    train_sha = file_sha256(args.train_features)

    oof: dict[str, dict[str, np.ndarray]] = {}
    members: dict[str, list[tuple[torch.nn.Module, dict[str, np.ndarray]]]] = {}
    checkpoints: dict[str, list[dict[str, Any]]] = {}
    specifications = (
        ("shared_action", "action", "raw", all_index, 0),
        ("shared_static", "static", "raw", all_index, 1000),
        ("raw_latent", "latent", "raw", all_index, 2000),
        ("raw_full", "full", "raw", all_index, 3000),
        ("v2_stacked_latent", "latent", "v2_stacked", stacked_index, 4000),
        ("v2_stacked_full", "full", "v2_stacked", stacked_index, 5000),
    )
    for label, mode, variant, universe, offset in specifications:
        oof[label], members[label], checkpoints[label] = train_head(
            label=label, mode=mode, variant=variant, universe=universe, train=train,
            target=target, fold_all=fold_all, config=config, args=args,
            seed_offset=offset, train_sha=train_sha)

    raw_predictions = {
        "action": oof["shared_action"], "static": oof["shared_static"],
        "latent": oof["raw_latent"], "full": oof["raw_full"],
    }
    stacked_predictions = {
        "action": {key: value[stacked_index] for key, value in oof["shared_action"].items()},
        "static": {key: value[stacked_index] for key, value in oof["shared_static"].items()},
        "latent": oof["v2_stacked_latent"], "full": oof["v2_stacked_full"],
    }
    raw = fit_branch(name="dinov3_full_meta_raw", variant="raw", universe=all_index,
                     predictions=raw_predictions, train=train, fold_all=fold_all, args=args)
    stacked = fit_branch(name="dinov3_full_meta_v2_stacked", variant="v2_stacked", universe=stacked_index,
                         predictions=stacked_predictions, train=train, fold_all=fold_all, args=args)
    ate_scale, rtp_scale, scorer_records = scorer_scales(args.checkpoint_root)
    native_name = "dinov3_scorer_v2_native"

    pre_cal = {
        "schema": "recon_dense_meta_dinov3_three_arm_pre_cal_v1",
        "status": "frozen_before_dense_cal_labels_open",
        "arms": {
            raw["name"]: {"feature_contract": "action + DINOv3 static + raw LTWM rollout/three-seed epistemic only",
                          "scorer_v2_predictions_in_features": False, "train_oof": raw["meta_metrics"],
                          "policies": raw["policies"], "meta": serializable_ridge(raw["meta_model"])},
            stacked["name"]: {"feature_contract": "raw arm plus frozen scorer-v2 ATE/RTP/uncertainty",
                              "scorer_v2_predictions_in_features": True, "training_states": len(stacked_index),
                              "train_oof": stacked["meta_metrics"], "policies": stacked["policies"],
                              "meta": serializable_ridge(stacked["meta_model"])},
            native_name: {"feature_contract": "native frozen scorer-v2 three-seed greedy fixed-pool readout",
                          "ate_scale": ate_scale, "rtp_scale": rtp_scale},
        },
        "leakage_audit": {"dense_train": train_overlap, "dense_cal": cal_overlap, "test3": test_overlap},
        "support_audit": {"dense_train": support_audit(train), "dense_cal": support_audit(cal), "test3": support_audit(test)},
        "checkpoints": checkpoints,
        "scorer_v2_checkpoints": scorer_records,
        "feature_sha256": {"dense_train": train_sha, "dense_cal": file_sha256(args.cal_features),
                            "test3": file_sha256(args.test_features)},
        "cal_labels_opened": False,
        "test_labels_opened": False,
        "fixed_pool_only": True,
        "cem": False,
    }
    pre_cal_path = args.output_dir / "pre_cal_freeze.json"
    write_json(pre_cal_path, pre_cal)

    def ensemble_for(data: dict[str, Any]) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, dict[str, np.ndarray]]]:
        shared_action = ensemble_predict(members["shared_action"], feature_by_mode(data, "action", variant="raw"), device=args.device)
        shared_static = ensemble_predict(members["shared_static"], feature_by_mode(data, "static", variant="raw"), device=args.device)
        raw_prediction = {"action": shared_action, "static": shared_static,
                          "latent": ensemble_predict(members["raw_latent"], feature_by_mode(data, "latent", variant="raw"), device=args.device),
                          "full": ensemble_predict(members["raw_full"], feature_by_mode(data, "full", variant="raw"), device=args.device)}
        stacked_prediction = {"action": shared_action, "static": shared_static,
                              "latent": ensemble_predict(members["v2_stacked_latent"], feature_by_mode(data, "latent", variant="v2_stacked"), device=args.device),
                              "full": ensemble_predict(members["v2_stacked_full"], feature_by_mode(data, "full", variant="v2_stacked"), device=args.device)}
        return raw_prediction, stacked_prediction

    cal_raw_prediction, cal_stacked_prediction = ensemble_for(cal)
    cal_raw_selected, _, cal_raw_score = predict_branch(branch=raw, predictions=cal_raw_prediction, data=cal)
    cal_stacked_selected, _, cal_stacked_score = predict_branch(branch=stacked, predictions=cal_stacked_prediction, data=cal)
    cal_native_selected = greedy_fixed_pool_selection(cal["ate_by_seed"], cal["rtp_by_seed"],
                                                       ate_scale=ate_scale, rtp_scale=rtp_scale,
                                                       valid=cal["majority_support"])
    cal_inference = args.output_dir / "dense_cal_frozen_inference.npz"
    write_npz(cal_inference, record_key=cal["record_key"].astype(str), family=cal["family"].astype(str),
              dinov3_full_meta_raw_selected_index=cal_raw_selected,
              dinov3_full_meta_v2_stacked_selected_index=cal_stacked_selected,
              dinov3_scorer_v2_native_selected_index=cal_native_selected,
              raw_meta_score=cal_raw_score, stacked_meta_score=cal_stacked_score)
    cal_targets = load_targets(args.cal_label_source, cal)
    cal_metrics = {
        raw["name"]: metrics_json(selection_metrics(cal_raw_selected, cal_targets)),
        stacked["name"]: metrics_json(selection_metrics(cal_stacked_selected, cal_targets)),
        native_name: metrics_json(selection_metrics(cal_native_selected, cal_targets)),
    }
    raw_cal_pass = bool(cal_metrics[raw["name"]]["promotions"] >= args.min_promotions and
                        cal_metrics[raw["name"]]["dual_precision"] is not None and
                        cal_metrics[raw["name"]]["dual_precision"] >= args.target_dual_precision and
                        cal_metrics[raw["name"]]["dual_wilson_lower_95"] is not None and
                        cal_metrics[raw["name"]]["dual_wilson_lower_95"] >= args.target_wilson_lower)
    frozen = {**pre_cal, "schema": "recon_dense_meta_dinov3_three_arm_frozen_v1",
              "status": "frozen_after_family_disjoint_dense_cal",
              "cal_labels_opened": True, "test_labels_opened": False,
              "primary_raw_test3_authorized": raw_cal_pass, "dense_cal": cal_metrics,
              "pre_cal_freeze_sha256": file_sha256(pre_cal_path),
              "cal_inference_sha256": file_sha256(cal_inference)}
    frozen_path = args.output_dir / "frozen_policy.json"
    write_json(frozen_path, frozen)
    if not raw_cal_pass:
        write_json(args.output_dir / "audit.json", {**frozen, "status": "primary_raw_rejected_on_dense_cal"})
        print(json.dumps(frozen, indent=2, sort_keys=True))
        return

    test_raw_prediction, test_stacked_prediction = ensemble_for(test)
    test_raw_selected, _, test_raw_score = predict_branch(branch=raw, predictions=test_raw_prediction, data=test)
    test_stacked_selected, _, test_stacked_score = predict_branch(branch=stacked, predictions=test_stacked_prediction, data=test)
    test_native_selected = greedy_fixed_pool_selection(test["ate_by_seed"], test["rtp_by_seed"],
                                                        ate_scale=ate_scale, rtp_scale=rtp_scale,
                                                        valid=test["majority_support"])
    test_inference = args.output_dir / "test3_frozen_inference.npz"
    write_npz(test_inference, record_key=test["record_key"].astype(str), family=test["family"].astype(str),
              dinov3_full_meta_raw_selected_index=test_raw_selected,
              dinov3_full_meta_v2_stacked_selected_index=test_stacked_selected,
              dinov3_scorer_v2_native_selected_index=test_native_selected,
              raw_meta_score=test_raw_score, stacked_meta_score=test_stacked_score)
    pre_test = {**frozen, "status": "test3_selections_frozen_before_test3_labels_open",
                "test_inference": str(test_inference), "test_inference_sha256": file_sha256(test_inference)}
    pre_test_path = args.output_dir / "pre_test_label_freeze.json"
    write_json(pre_test_path, pre_test)
    test_targets = load_targets(args.test_label_source, test)
    test_metrics = {
        raw["name"]: metrics_json(selection_metrics(test_raw_selected, test_targets)),
        stacked["name"]: metrics_json(selection_metrics(test_stacked_selected, test_targets)),
        native_name: metrics_json(selection_metrics(test_native_selected, test_targets)),
    }
    report = {**pre_test, "schema": "recon_dense_meta_dinov3_three_arm_result_v1",
              "status": "completed_fixed_pool_three_arm_test3",
              "test_labels_opened": True, "test3": test_metrics,
              "pre_test_label_freeze_sha256": file_sha256(pre_test_path),
              "test_label_source_sha256": file_sha256(args.test_label_source),
              "naming_note": "full/meta-raw is independently retrained; full/meta-v2-stacked depends on frozen scorer-v2 predictions",
              "evidence_note": "RECON test3 was already used by the prior 019f70d5 study and is developmental replay, not fresh deployment evidence",
              "deployment_accepted": False, "cem": False}
    write_json(args.output_dir / "result.json", report)
    write_json(args.output_dir / "audit.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
