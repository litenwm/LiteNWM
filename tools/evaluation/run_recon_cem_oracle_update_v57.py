#!/usr/bin/env python3
"""Nondeployable oracle-update CEM diagnostic on spent RECON cohorts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from planning.nomad_action_adapter import nomad_waypoints_to_local_actions  # noqa: E402
from scoring.spacing_support_v1 import (  # noqa: E402
    score_spacing_envelope,
    threshold_for_quantile,
)
from tools.evaluation.build_recon100_k17_source_v1 import metric_labels  # noqa: E402
from tools.evaluation.build_recon_inner_k17_source_v2 import (  # noqa: E402
    local_ground_truth,
)
from tools.evaluation.export_recon_dense_dinov3_features_v1 import (  # noqa: E402
    write_json_atomic,
    write_npz_atomic,
)
from tools.planning.run_cem_development_v1 import (  # noqa: E402
    sample_population,
)
from tools.planning.run_cem_safe_elite_v55 import (  # noqa: E402
    initial_proposal,
    proposal_summary,
    safe_elite_update,
)
from tools.scoring.finetune_universal_dualdomain_safe_rank_v2 import (  # noqa: E402
    load_envelopes,
)
from tools.scoring.train_recon_dense_meta_dinov3_v1 import (  # noqa: E402
    file_sha256,
)


METRICS = ("ATE", "RPE", "Endpoint", "Objective")
TRAINING_RESULT = ROOT / "outputs/universal_dualdomain_raw_v2/training_v2/result.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank-root", type=Path, action="append", required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--trajectory-root", type=Path, required=True)
    parser.add_argument("--training-result", type=Path, default=TRAINING_RESULT)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def quality(value: np.ndarray) -> dict[str, dict[str, float]]:
    return {
        name: {
            "mean": float(value[:, index].mean()),
            "p90": float(np.quantile(value[:, index], 0.9)),
        }
        for index, name in enumerate(METRICS)
    }


def action_distribution(candidates: np.ndarray) -> dict[str, float]:
    path = candidates[:, 1:].astype(np.float64)
    origin = np.zeros((*path.shape[:2], 1, 2), dtype=np.float64)
    step = np.diff(np.concatenate((origin, path), axis=2), axis=2)
    step_norm = np.linalg.norm(step, axis=-1)
    dot = (step[..., 1:, :] * step[..., :-1, :]).sum(axis=-1)
    cross = (
        step[..., :-1, 0] * step[..., 1:, 1]
        - step[..., :-1, 1] * step[..., 1:, 0]
    )
    turn = np.abs(np.arctan2(cross, dot))
    coordinate_variance = path.reshape(len(path), path.shape[1], -1).var(axis=1)
    entropy_proxy = 0.5 * np.log(
        2.0 * np.pi * np.e * np.maximum(coordinate_variance, 1.0e-9)
    ).sum(axis=1)
    return {
        "coordinate_variance_mean": float(coordinate_variance.mean()),
        "sample_entropy_proxy_mean": float(entropy_proxy.mean()),
        "path_length_mean": float(step_norm.sum(axis=-1).mean()),
        "endpoint_radius_mean": float(np.linalg.norm(path[..., -1, :], axis=-1).mean()),
        "endpoint_lateral_abs_mean": float(np.abs(path[..., -1, 1]).mean()),
        "mean_abs_turn_rad": float(turn.mean()),
        "p90_abs_turn_rad": float(np.quantile(turn, 0.9)),
    }


def pool_report(
    *,
    absolute: np.ndarray,
    valid: np.ndarray,
    oracle_safe: np.ndarray,
    elite: np.ndarray,
    updated: np.ndarray,
    candidates: np.ndarray,
) -> dict[str, object]:
    delta = absolute - absolute[:, :1]
    dual = (delta[..., 0] < -1.0e-7) & (delta[..., 3] < -1.0e-7)
    harm = (delta[..., 0] > 1.0e-7) | (delta[..., 3] > 1.0e-7)
    challenger_valid = valid[:, 1:]
    challenger_dual = dual[:, 1:]
    challenger_harm = harm[:, 1:]
    oracle_energy = 0.5 * (delta[..., 0] + delta[..., 3])
    masked = np.where(oracle_safe, oracle_energy, np.inf)
    selected = masked.argmin(axis=1)
    selected = np.where(oracle_safe.any(axis=1), selected, 0)
    row = np.arange(len(selected))
    selected_absolute = absolute[row, selected]
    elite_state = np.repeat(np.flatnonzero(updated), elite.shape[1])
    elite_index = elite[updated].reshape(-1)
    elite_delta = delta[elite_state, elite_index]
    elite_utility = -0.5 * (elite_delta[:, 0] + elite_delta[:, 3])
    return {
        "valid_candidates": int(challenger_valid.sum()),
        "dual_rate_among_valid": float(
            challenger_dual[challenger_valid].mean()
        ),
        "harm_rate_among_valid": float(
            challenger_harm[challenger_valid].mean()
        ),
        "states_with_any_dual_candidate": float(
            (oracle_safe[:, 1:].any(axis=1)).mean()
        ),
        "oracle_selected_quality": quality(selected_absolute),
        "oracle_selected_promotions": int((selected != 0).sum()),
        "elite_true_utility_mean": float(elite_utility.mean()),
        "elite_true_utility_p10": float(np.quantile(elite_utility, 0.1)),
        "elite_dual_rate": float(
            ((elite_delta[:, 0] < 0.0) & (elite_delta[:, 3] < 0.0)).mean()
        ),
        "action_distribution": action_distribution(candidates),
    }


def main() -> None:
    args = parse_args()
    if len(args.bank_root) != 2:
        raise ValueError("oracle v57 expects exactly two spent cohorts")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(args.output_dir)
    preregistration = json.loads(
        args.preregistration.read_text(encoding="utf-8")
    )
    if (
        preregistration.get("evidence_role")
        != "nondeployable_oracle_upper_bound_on_spent_B_C"
        or not preregistration.get("outcomes_already_opened")
        or preregistration.get("reserve_access_allowed")
    ):
        raise RuntimeError("oracle preregistration contract changed")
    training = json.loads(args.training_result.read_text(encoding="utf-8"))
    envelope = load_envelopes(Path(training["support"]))["recon"]
    threshold = threshold_for_quantile(envelope, 0.999)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    bank_reports = {}
    pooled_iteration = [[] for _ in range(3)]
    route_sets = []
    for bank_root in args.bank_root:
        bank = bank_root.name
        manifest = bank_root / "manifest.windows.jsonl"
        candidate_path = bank_root / "k17.npz"
        rows = read_rows(manifest)
        route_sets.append({str(row["route_id"]) for row in rows})
        ground_truth = np.stack(
            [
                local_ground_truth(
                    args.trajectory_root
                    / str(row["trajectory_id"])
                    / "traj_data.pkl",
                    int(row["current_t"]),
                )
                for row in rows
            ]
        ).astype(np.float32)
        with np.load(candidate_path, allow_pickle=False) as payload:
            incumbent = payload["candidates_waypoint_units"][:, 0].astype(
                np.float32
            )
            reference = payload["nomad_x16_waypoint_units"].astype(np.float32)
            record = payload["record_key"].astype(str)
        mean, covariance = initial_proposal(
            reference, floor=0.015, scale=0.70
        )
        seed = int(preregistration["cem"]["seeds"][bank])
        rng = np.random.default_rng(seed)
        iteration_reports = []
        bank_dir = args.output_dir / bank
        bank_dir.mkdir()
        for iteration in range(3):
            proposal_before = proposal_summary(mean, covariance)
            candidates, sampling = sample_population(
                incumbent=incumbent,
                reference=reference,
                mean=mean,
                covariance=covariance,
                population=16,
                rng=rng,
                support_envelope=envelope,
                support_threshold=threshold,
                spacing_m=0.25,
            )
            actions = nomad_waypoints_to_local_actions(candidates, dataset="recon")
            support_score, physical_guard = score_spacing_envelope(
                envelope,
                actions,
                np.full(len(rows), 0.25, dtype=np.float32),
            )
            valid = physical_guard & (support_score <= threshold)
            valid[:, 0] = True
            absolute = metric_labels(
                candidates, ground_truth, 0.25, 0.4, 0.6
            )["values"].astype(np.float32)
            delta = absolute - absolute[:, :1]
            oracle_safe = (
                valid
                & (delta[..., 0] < -1.0e-7)
                & (delta[..., 3] < -1.0e-7)
            )
            oracle_safe[:, 0] = False
            energy = 0.5 * (delta[..., 0] + delta[..., 3])
            next_mean, next_covariance, elite, updated, safe_count = (
                safe_elite_update(
                    candidates,
                    energy,
                    oracle_safe,
                    previous_mean=mean,
                    previous_covariance=covariance,
                    elite_count=4,
                    floor=0.015,
                    scale=0.70,
                )
            )
            report = {
                "iteration": iteration,
                "population": 16,
                "elite": 4,
                "updated_windows": int(updated.sum()),
                "fallback_windows": int((~updated).sum()),
                "safe_candidates": int(oracle_safe[:, 1:].sum()),
                "safe_count_mean": float(safe_count.mean()),
                "support_acceptance": sampling["support_acceptance"],
                "support_fallback_samples": sampling["fallback_samples"],
                "proposal_before": proposal_before,
                "proposal_after": proposal_summary(
                    next_mean, next_covariance
                ),
                **pool_report(
                    absolute=absolute,
                    valid=valid,
                    oracle_safe=oracle_safe,
                    elite=elite,
                    updated=updated,
                    candidates=candidates,
                ),
            }
            iteration_path = bank_dir / f"iteration_{iteration:02d}.npz"
            write_npz_atomic(
                iteration_path,
                record_key=record,
                candidates=candidates,
                absolute_metrics=absolute,
                valid=valid,
                oracle_safe=oracle_safe,
                oracle_energy=energy,
                elite_index=elite,
                proposal_updated=updated,
                proposal_mean_before=mean,
                proposal_covariance_before=covariance,
                proposal_mean_after=next_mean,
                proposal_covariance_after=next_covariance,
            )
            report["artifact"] = str(iteration_path)
            report["artifact_sha256"] = file_sha256(iteration_path)
            iteration_reports.append(report)
            pooled_iteration[iteration].append(
                {
                    "absolute": absolute,
                    "valid": valid,
                    "oracle_safe": oracle_safe,
                    "elite": elite,
                    "updated": updated,
                    "candidates": candidates,
                }
            )
            mean, covariance = next_mean, next_covariance
        first = iteration_reports[0]
        last = iteration_reports[-1]
        bank_reports[bank] = {
            "states": len(rows),
            "routes": len(route_sets[-1]),
            "seed": seed,
            "iterations": iteration_reports,
            "iteration_2_minus_0": {
                "dual_rate_among_valid": float(
                    last["dual_rate_among_valid"]
                    - first["dual_rate_among_valid"]
                ),
                "harm_rate_among_valid": float(
                    last["harm_rate_among_valid"]
                    - first["harm_rate_among_valid"]
                ),
                "elite_true_utility_mean": float(
                    last["elite_true_utility_mean"]
                    - first["elite_true_utility_mean"]
                ),
                "coordinate_variance_mean": float(
                    last["action_distribution"]["coordinate_variance_mean"]
                    - first["action_distribution"]["coordinate_variance_mean"]
                ),
                "sample_entropy_proxy_mean": float(
                    last["action_distribution"]["sample_entropy_proxy_mean"]
                    - first["action_distribution"]["sample_entropy_proxy_mean"]
                ),
            },
        }
    if route_sets[0] & route_sets[1]:
        raise RuntimeError("B/C route overlap")

    pooled_reports = []
    for iteration, parts in enumerate(pooled_iteration):
        absolute = np.concatenate([part["absolute"] for part in parts])
        valid = np.concatenate([part["valid"] for part in parts])
        oracle_safe = np.concatenate([part["oracle_safe"] for part in parts])
        candidates = np.concatenate([part["candidates"] for part in parts])
        elite = np.concatenate([part["elite"] for part in parts])
        updated = np.concatenate([part["updated"] for part in parts])
        pooled_reports.append(
            {
                "iteration": iteration,
                **pool_report(
                    absolute=absolute,
                    valid=valid,
                    oracle_safe=oracle_safe,
                    elite=elite,
                    updated=updated,
                    candidates=candidates,
                ),
            }
        )
    first, last = pooled_reports[0], pooled_reports[-1]
    trend = {
        "dual_rate_among_valid_delta": float(
            last["dual_rate_among_valid"] - first["dual_rate_among_valid"]
        ),
        "harm_rate_among_valid_delta": float(
            last["harm_rate_among_valid"] - first["harm_rate_among_valid"]
        ),
        "elite_true_utility_mean_delta": float(
            last["elite_true_utility_mean"] - first["elite_true_utility_mean"]
        ),
        "coordinate_variance_mean_delta": float(
            last["action_distribution"]["coordinate_variance_mean"]
            - first["action_distribution"]["coordinate_variance_mean"]
        ),
        "sample_entropy_proxy_mean_delta": float(
            last["action_distribution"]["sample_entropy_proxy_mean"]
            - first["action_distribution"]["sample_entropy_proxy_mean"]
        ),
        "oracle_selected_quality_delta": {
            name: {
                summary: float(
                    last["oracle_selected_quality"][name][summary]
                    - first["oracle_selected_quality"][name][summary]
                )
                for summary in ("mean", "p90")
            }
            for name in METRICS
        },
    }
    improved = bool(
        trend["dual_rate_among_valid_delta"] > 0.0
        and trend["harm_rate_among_valid_delta"] < 0.0
        and trend["elite_true_utility_mean_delta"] > 0.0
        and all(
            trend["oracle_selected_quality_delta"][name][summary] <= 0.0
            for name in METRICS
            for summary in ("mean", "p90")
        )
    )
    result = {
        "schema": "recon_cem_oracle_update_v57_result",
        "status": (
            "oracle_updates_improve_action_distribution_and_pool"
            if improved
            else "oracle_updates_do_not_uniformly_improve_all_diagnostics"
        ),
        "evidence_role": "nondeployable_oracle_upper_bound_on_spent_B_C",
        "future_outcome_used_inside_cem": True,
        "deployable": False,
        "production_claim_allowed": False,
        "banks": bank_reports,
        "pooled_iterations": pooled_reports,
        "pooled_iteration_2_minus_0": trend,
        "uniform_improvement": improved,
        "preregistration": str(args.preregistration),
        "preregistration_sha256": file_sha256(args.preregistration),
        "training_result_sha256": file_sha256(args.training_result),
        "support_checkpoint_sha256": file_sha256(Path(training["support"])),
        "route_overlap": 0,
        "reserve_accessed": False,
    }
    output = args.output_dir / "result.json"
    write_json_atomic(output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
