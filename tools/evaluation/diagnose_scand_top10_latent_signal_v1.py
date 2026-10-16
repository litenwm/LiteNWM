#!/usr/bin/env python3
"""Read-only seen-route versus held-route diagnostic for the frozen reranker."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from tools.evaluation import run_scand_top10_latent_reranker_v1 as rerank  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


def ranks(value: np.ndarray) -> np.ndarray:
    order = np.argsort(value, kind="stable")
    result = np.empty(len(value), dtype=np.float64)
    result[order] = np.arange(len(value), dtype=np.float64)
    return result


def score_diagnostic(
    scores: list[np.ndarray],
    targets: list[np.ndarray],
) -> dict[str, Any]:
    correlations = []
    correct = 0
    compared = 0
    top1 = 0
    states = 0
    for local_scores, local_targets in zip(scores, targets):
        for score, target in zip(local_scores, local_targets):
            score_rank = ranks(score)
            target_rank = ranks(target)
            correlation = np.corrcoef(score_rank, target_rank)[0, 1]
            if np.isfinite(correlation):
                correlations.append(float(correlation))
            left, right = np.triu_indices(len(score), 1)
            target_difference = target[left] - target[right]
            valid = np.abs(target_difference) > 1.0e-8
            score_difference = score[left] - score[right]
            correct += int(
                (
                    np.sign(score_difference[valid])
                    == np.sign(target_difference[valid])
                ).sum()
            )
            compared += int(valid.sum())
            top1 += int(np.argmax(score) == np.argmax(target))
            states += 1
    return {
        "states": states,
        "mean_within_state_spearman": float(np.mean(correlations)),
        "median_within_state_spearman": float(np.median(correlations)),
        "pairwise_order_accuracy": correct / compared,
        "shortlist_target_top1_recall": top1 / states,
    }


def selection_summary(
    data: dict[str, np.ndarray],
    states: np.ndarray,
    selected: np.ndarray,
) -> dict[str, Any]:
    stats = rerank.coarse.selection_stats(data, states, selected)
    mask = rerank.coarse.choice_mask(data, states)
    objective = data["objective_gain_vs_formal_v70_m"][states].astype(np.float64)
    masked = np.where(mask, objective, -np.inf)
    oracle = np.argmax(masked, axis=1)
    opportunity = masked[np.arange(len(states)), oracle] > 0.0
    chosen_gain = objective[np.arange(len(states)), selected]
    oracle_gain = objective[np.arange(len(states)), oracle]
    return {
        key: stats[key]
        for key in (
            "replacements",
            "coverage",
            "objective_gain_m",
            "ate_gain_m",
            "dual_rate",
            "harm_rate",
            "positive_route_fraction",
        )
    } | {
        "objective_oracle_opportunities": int(opportunity.sum()),
        "exact_objective_oracle_top1_recall": float(
            np.mean(selected[opportunity] == oracle[opportunity])
        ),
        "mean_realized_objective_oracle_fraction": float(
            np.mean(
                np.maximum(chosen_gain[opportunity], 0.0)
                / oracle_gain[opportunity]
            )
        ),
    }


def state_dict(payload: np.lib.npyio.NpzFile, prefix: str) -> dict[str, torch.Tensor]:
    return {
        name[len(prefix) :]: torch.from_numpy(np.asarray(payload[name]))
        for name in payload.files
        if name.startswith(prefix)
    }


def main() -> None:
    args = parse_args()
    result_dir = args.result_dir.resolve()
    output = args.output.resolve()
    if output.exists():
        raise RuntimeError(f"refusing to overwrite: {output}")
    result = json.loads((result_dir / "result.json").read_text(encoding="utf-8"))
    contract = json.loads(
        Path(result["inputs"]["contract"]["path"]).read_text(encoding="utf-8")
    )
    data = rerank.coarse.load_npz(Path(result["inputs"]["runtime_table"]["path"]))
    rich = rerank.load_rich(
        result_dir / "rich_ltwm_runtime_features.npz",
        data["record_key"],
    )
    actions = rerank.actions_from_data(data)
    routes = data["route_group"].astype(str)
    with np.load(
        result_dir / "outer_route_oof_predictions.npz",
        allow_pickle=False,
    ) as payload:
        folds = np.asarray(payload["outer_fold"]).copy()
        held_scores = np.asarray(payload["reranker_score"]).copy()
        coarse_oof = np.asarray(payload["coarse_score"]).copy()
        held_selected = np.asarray(payload["pure_selected_index"]).copy()
    device = torch.device(args.device)
    all_states = np.arange(len(routes))
    held_candidates = rerank.shortlist(coarse_oof, data, all_states)
    held_local_scores = [
        held_scores[index, held_candidates[index]]
        for index in range(len(all_states))
    ]
    held_local_targets = rerank.selected_target(
        data,
        all_states,
        held_candidates,
        contract,
    )
    train_states_all = []
    train_selected_all = []
    train_scores_all = []
    train_targets_all = []
    fold_records = {}
    with np.load(
        result_dir / "frozen_outer_pipeline.npz",
        allow_pickle=False,
    ) as parameters:
        for outer in range(5):
            states = all_states[folds != outer]
            coarse_model = rerank.coarse.UtilityNetwork(351).to(device)
            coarse_model.load_state_dict(
                state_dict(parameters, f"outer_{outer}__coarse__"),
                strict=True,
            )
            coarse_scores = rerank.coarse.predict(
                coarse_model,
                np.asarray(parameters[f"outer_{outer}__coarse_mean"]),
                np.asarray(parameters[f"outer_{outer}__coarse_std"]),
                data["runtime_pair_feature"][states],
                device,
            )
            candidates = rerank.shortlist(coarse_scores, data, states)
            model = rerank.LatentReranker(
                len(parameters[f"outer_{outer}__reranker_mean"]),
                int(contract["model"]["token_width"]),
                int(contract["model"]["horizon_width"]),
            ).to(device)
            model.load_state_dict(
                state_dict(parameters, f"outer_{outer}__reranker__"),
                strict=True,
            )
            scores = rerank.predict_reranker(
                model,
                np.asarray(parameters[f"outer_{outer}__reranker_mean"]),
                np.asarray(parameters[f"outer_{outer}__reranker_std"]),
                rich,
                actions,
                data,
                states,
                candidates,
                device,
                int(contract["model"]["state_batch_size"]),
            )
            targets = rerank.selected_target(data, states, candidates, contract)
            selected = candidates[np.arange(len(states)), np.argmax(scores, axis=1)]
            train_states_all.append(states)
            train_selected_all.append(selected)
            train_scores_all.append(scores)
            train_targets_all.append(targets)
            fold_records[str(outer)] = selection_summary(data, states, selected)
            del coarse_model, model
            if device.type == "cuda":
                torch.cuda.empty_cache()
    train_states = np.concatenate(train_states_all)
    train_selected = np.concatenate(train_selected_all)

    progress_score = rich["latent_scalar"][..., 3].astype(np.float32).mean(axis=(2, 3))
    progress_local = np.take_along_axis(progress_score, held_candidates, axis=1)
    progress_selected = held_candidates[
        np.arange(len(all_states)),
        np.argmax(progress_local, axis=1),
    ]
    progress_targets = rerank.selected_target(
        data,
        all_states,
        held_candidates,
        contract,
    )
    seed_progress = rich["latent_scalar"][..., 3].astype(np.float32).mean(axis=3)
    seed_agreement = []
    for state in all_states:
        candidate = held_candidates[state]
        local = seed_progress[state, candidate]
        for left in range(3):
            for right in range(left + 1, 3):
                seed_agreement.append(
                    float(np.corrcoef(ranks(local[:, left]), ranks(local[:, right]))[0, 1])
                )

    probe = {
        "schema": "scand_top10_latent_signal_seen_vs_held_diagnostic_v1",
        "status": "complete_read_only_diagnostic",
        "question": "Does the frozen reranker extract useful LTWM signal on seen routes but fail across held routes?",
        "seen_route_training_instances": {
            "selection": selection_summary(data, train_states, train_selected),
            "score_vs_target": score_diagnostic(train_scores_all, train_targets_all),
            "per_outer_model": fold_records,
            "note": "Every state appears in four seen-route models; no metric is an unbiased performance estimate.",
        },
        "held_route_oof": {
            "selection": selection_summary(data, all_states, held_selected),
            "score_vs_target": score_diagnostic(
                [np.stack(held_local_scores)],
                [held_local_targets],
            ),
        },
        "transparent_ltwm_goal_progress_on_held_shortlist": {
            "selection": selection_summary(data, all_states, progress_selected),
            "score_vs_target": score_diagnostic(
                [progress_local],
                [progress_targets],
            ),
            "mean_pairwise_seed_rank_agreement": float(np.mean(seed_agreement)),
        },
        "interpretation_rule": {
            "seen_good_held_bad": "The model can extract route-specific signal, but LTWM-derived ordering is not route-invariant; prediction/calibration is the primary bottleneck.",
            "seen_bad_held_bad": "The frozen model/loss cannot extract the signal even on seen routes; selector underfit or representation insufficiency remains unresolved.",
        },
        "new_model_trained": False,
        "sealed_validation_opened": False,
        "source_result_sha256": rerank.sha256_file(result_dir / "result.json"),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    rerank.write_json_atomic(output, probe)
    print(rerank.canonical_json(probe))


if __name__ == "__main__":
    main()
