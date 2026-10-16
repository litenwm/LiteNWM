#!/usr/bin/env python3
"""Freeze v60 G/H fixed-pool and unified CEM selections before outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from tools.planning.run_cem_safe_elite_v55 import (  # noqa: E402
    proposal_summary,
    safe_elite_update,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def write_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
    temporary.replace(path)


def rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def best_safe(energy: np.ndarray, safe: np.ndarray) -> np.ndarray:
    masked = np.where(safe, energy, np.inf)
    best = np.argmin(masked, axis=1).astype(np.int64)
    return np.where(safe.any(axis=1), best, 0).astype(np.int64)


def freeze_bank(
    bank: str,
    root: Path,
    manifest_path: Path,
    boundary_sha: str,
    margin: float,
) -> dict[str, Any]:
    bank_root = root / bank
    run = bank_root / "cem_run_v60"
    manifest_rows = rows(manifest_path)
    expected_record = np.asarray(
        [f"{row['trajectory_id']}:{int(row['current_t'])}" for row in manifest_rows]
    )
    if len(manifest_rows) != 44 or len({row["route_id"] for row in manifest_rows}) != 11:
        raise RuntimeError(f"{bank}: manifest contract changed")

    manifest_audit_path = bank_root / "manifest.windows.audit.json"
    manifest_audit = load_json(manifest_audit_path)
    source_audit = load_json(bank_root / "source_label_free.audit.json")
    prelabel_path = run / "prelabel_freeze.json"
    prelabel = load_json(prelabel_path)
    runtime_path = run / "v60_runtime_binding.json"
    runtime = load_json(runtime_path)
    for name, audit in (
        ("manifest", manifest_audit),
        ("source", source_audit),
        ("prelabel", prelabel),
        ("runtime", runtime),
    ):
        if (
            audit.get("labels_opened")
            or audit.get("future_pose_opened")
            or audit.get("future_pose_outcomes_opened")
            or audit.get("outcomes_accessed")
            or audit.get("reserve_accessed")
        ):
            raise RuntimeError(f"{bank}: {name} audit indicates outcome/reserve access")
    if runtime.get("boundary_freeze_sha256") != boundary_sha:
        raise RuntimeError(f"{bank}: runtime was not bound to the frozen boundary")
    if prelabel.get("states") != 44 or prelabel.get("cem") is not True:
        raise RuntimeError(f"{bank}: CEM prelabel contract changed")

    selected_path = run / "selected_actions.npz"
    with np.load(selected_path, allow_pickle=False) as payload:
        observed_record = payload["record_key"].astype(str)
        base_candidates = payload["base_k17_feature_waypoint_units"].astype(np.float32)
        base_energy = payload["base_energy"].astype(np.float32)
        base_safe = payload["base_safe"].astype(bool)
    if (
        not np.array_equal(observed_record, expected_record)
        or base_candidates.shape != (44, 17, 4, 2)
        or base_energy.shape != (44, 17)
        or base_safe.shape != (44, 17)
    ):
        raise RuntimeError(f"{bank}: base archive identity/shape mismatch")
    base_safe[:, 0] = False
    fixed_index = best_safe(base_energy, base_safe)

    candidate_blocks = [base_candidates]
    energy_blocks = [base_energy]
    safe_blocks = [base_safe]
    iteration_audit: list[dict[str, Any]] = []
    for iteration in range(3):
        iteration_root = run / f"iteration_{iteration:02d}"
        population_path = iteration_root / "population.npz"
        score_path = iteration_root / "frozen_scores.npz"
        with np.load(population_path, allow_pickle=False) as population:
            candidates = population["candidates_feature_waypoint_units"].astype(np.float32)
            proposal_mean = population["proposal_mean"].astype(np.float32)
            proposal_covariance = population["proposal_covariance"].astype(np.float32)
            population_record = population["record_key"].astype(str)
        with np.load(score_path, allow_pickle=False) as scores:
            energy = scores["energy"].astype(np.float32)
            safe = scores["safe"].astype(bool)
            valid = scores["valid"].astype(bool)
            elite = scores["elite_index_for_update"].astype(np.int64)
            updated = scores["proposal_updated"].astype(bool)
            safe_count = scores["safe_count"].astype(np.int64)
        if (
            candidates.shape != (44, 17, 4, 2)
            or energy.shape != (44, 17)
            or safe.shape != (44, 17)
            or elite.shape != (44, 4)
            or not np.array_equal(population_record, expected_record)
        ):
            raise RuntimeError(f"{bank}: iteration {iteration} contract changed")
        safe[:, 0] = False
        elite_safe = True
        for state in range(44):
            if updated[state]:
                if np.any(elite[state] < 0) or not np.all(safe[state, elite[state]]):
                    elite_safe = False
            elif np.any(elite[state] != -1):
                elite_safe = False
        if not elite_safe:
            raise RuntimeError(f"{bank}: iteration {iteration} unsafe elite detected")
        if not np.array_equal(updated, safe_count >= 4):
            raise RuntimeError(f"{bank}: iteration {iteration} safe<4 fallback changed")

        next_mean, next_covariance, check_elite, check_updated, check_count = (
            safe_elite_update(
                candidates,
                energy,
                safe,
                previous_mean=proposal_mean,
                previous_covariance=proposal_covariance,
                elite_count=4,
                floor=0.015,
                scale=0.70,
            )
        )
        if not (
            np.array_equal(check_elite, elite)
            and np.array_equal(check_updated, updated)
            and np.array_equal(check_count, safe_count)
        ):
            raise RuntimeError(f"{bank}: iteration {iteration} update replay mismatch")
        fallback = ~updated
        fallback_unchanged = bool(
            np.array_equal(next_mean[fallback], proposal_mean[fallback])
            and np.array_equal(next_covariance[fallback], proposal_covariance[fallback])
        )
        if not fallback_unchanged:
            raise RuntimeError(f"{bank}: iteration {iteration} fallback proposal moved")
        if iteration < 2:
            next_population_path = run / f"iteration_{iteration + 1:02d}" / "population.npz"
            with np.load(next_population_path, allow_pickle=False) as next_population:
                persisted_mean = next_population["proposal_mean"].astype(np.float32)
                persisted_covariance = next_population["proposal_covariance"].astype(np.float32)
            if not (
                np.allclose(next_mean, persisted_mean, rtol=0.0, atol=1.0e-7)
                and np.allclose(next_covariance, persisted_covariance, rtol=0.0, atol=1.0e-7)
            ):
                raise RuntimeError(f"{bank}: iteration {iteration} persisted q mismatch")

        candidate_blocks.append(candidates[:, 1:])
        energy_blocks.append(energy[:, 1:])
        safe_blocks.append(safe[:, 1:])
        iteration_audit.append(
            {
                "iteration": iteration,
                "population_challengers": 16,
                "valid_candidates": int(valid[:, 1:].sum()),
                "safe_candidates": int(safe[:, 1:].sum()),
                "elite_candidates": int(updated.sum() * 4),
                "updated_windows": int(updated.sum()),
                "fallback_windows": int((~updated).sum()),
                "safe_count_min": int(safe_count.min()),
                "safe_count_mean": float(safe_count.mean()),
                "elite_only_from_safe_mask_verified": elite_safe,
                "safe_lt_4_proposal_unchanged_verified": fallback_unchanged,
                "proposal_before": proposal_summary(proposal_mean, proposal_covariance),
                "proposal_after": proposal_summary(next_mean, next_covariance),
                "population_sha256": sha256(population_path),
                "frozen_scores_sha256": sha256(score_path),
            }
        )

    candidates_all = np.concatenate(candidate_blocks, axis=1).astype(np.float32)
    energy_all = np.concatenate(energy_blocks, axis=1).astype(np.float32)
    safe_all = np.concatenate(safe_blocks, axis=1).astype(bool)
    if candidates_all.shape != (44, 65, 4, 2):
        raise RuntimeError(f"{bank}: combined archive is not K65")
    cem_energy = energy_all[:, 17:]
    cem_safe = safe_all[:, 17:]
    cem_local = best_safe(cem_energy, cem_safe)
    has_cem = cem_safe.any(axis=1)
    cem_index = np.where(has_cem, cem_local + 17, 0).astype(np.int64)
    combined_index = fixed_index.copy()
    fixed_has = fixed_index != 0
    for state in range(44):
        if not has_cem[state]:
            continue
        if not fixed_has[state] or (
            energy_all[state, cem_index[state]] + margin
            < energy_all[state, fixed_index[state]]
        ):
            combined_index[state] = cem_index[state]
    source_code = np.where(combined_index == 0, 0, np.where(combined_index < 17, 1, 2))

    archive_path = bank_root / "combined_archive_preoutcome_v60.npz"
    if archive_path.exists():
        raise FileExistsError(archive_path)
    write_npz(
        archive_path,
        candidates_waypoint_units=candidates_all,
        energy=energy_all,
        safe=safe_all,
        record_key=expected_record,
        window_uid=np.asarray([str(row["window_uid"]) for row in manifest_rows]),
        route_id=np.asarray([str(row["route_id"]) for row in manifest_rows]),
        fixed_selected_index=fixed_index,
        cem_best_selected_index=cem_index,
        combined_selected_index=combined_index,
        combined_source_code=source_code.astype(np.int8),
    )
    return {
        "bank": bank,
        "states": 44,
        "routes": 11,
        "manifest": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "prelabel_freeze": str(prelabel_path),
        "prelabel_freeze_sha256": sha256(prelabel_path),
        "runtime_binding": str(runtime_path),
        "runtime_binding_sha256": sha256(runtime_path),
        "archive": str(archive_path),
        "archive_sha256": sha256(archive_path),
        "candidate_array_sha256": array_sha256(candidates_all),
        "fixed_selected_index_sha256": array_sha256(fixed_index),
        "combined_selected_index_sha256": array_sha256(combined_index),
        "fixed_promotions_before_outcomes": int((fixed_index != 0).sum()),
        "combined_promotions_before_outcomes": int((combined_index != 0).sum()),
        "cem_replacements_or_additions_before_outcomes": int(
            (combined_index != fixed_index).sum()
        ),
        "iteration_audit": iteration_audit,
        "labels_opened": False,
        "outcomes_accessed": False,
        "reserve_accessed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--reserve-freeze", type=Path, required=True)
    args = parser.parse_args()
    output = args.root / "GH_preoutcome_selection_freeze_v60.json"
    if output.exists():
        raise FileExistsError(output)

    boundary_path = args.root / "boundary_freeze.json"
    allocation_path = args.root / "GH_manifest_freeze.json"
    boundary = load_json(boundary_path)
    allocation = load_json(allocation_path)
    reserve = load_json(args.reserve_freeze)
    if (
        boundary.get("G_H_outcomes_accessed")
        or boundary.get("reserve_accessed")
        or allocation.get("labels_opened")
        or allocation.get("outcomes_accessed")
        or allocation.get("sealed_25route_reserve_accessed")
        or reserve.get("labels_opened")
        or reserve.get("outcomes_accessed")
    ):
        raise RuntimeError("pre-outcome or sealed-reserve contract failed")
    if allocation.get("combined_routes") != 22 or allocation.get("combined_states") != 88:
        raise RuntimeError("G/H allocation changed")
    boundary_sha = sha256(boundary_path)
    if boundary_sha != allocation.get("boundary_freeze_sha256"):
        raise RuntimeError("boundary/allocation binding mismatch")
    rule = boundary["selected_rule"]
    margin = float(rule["dominance_margin"])

    bank_results = {}
    for bank in ("bank_g_v60", "bank_h_v60"):
        manifest = Path(allocation["cohorts"][bank]["manifest"])
        bank_results[bank] = freeze_bank(
            bank, args.root, manifest, boundary_sha, margin
        )
    g_routes = set(allocation["cohorts"]["bank_g_v60"]["routes"])
    h_routes = set(allocation["cohorts"]["bank_h_v60"]["routes"])
    if g_routes & h_routes:
        raise RuntimeError("G/H route overlap appeared")

    freeze = {
        "schema": "recon_small_fresh_GH_preoutcome_selection_v60",
        "status": "policy_and_GH_selected_indices_frozen_before_any_outcome",
        "evidence_role": "small_fresh_development_confirmation_not_H4_100",
        "protocol_is_original_25route_H4_100": False,
        "states": 88,
        "routes": 22,
        "boundary_freeze": str(boundary_path),
        "boundary_freeze_sha256": boundary_sha,
        "allocation_freeze": str(allocation_path),
        "allocation_freeze_sha256": sha256(allocation_path),
        "selected_rule": rule,
        "selection_formula": boundary["formula"],
        "banks": bank_results,
        "success_gate": {
            "combined_added_windows_min": 9,
            "combined_coverage_gain_min": 0.10,
            "pooled_dual_min": 0.85,
            "pooled_harm_max": 0.15,
            "pooled_all_four_mean_and_p90_non_degraded_vs_fixed": True,
            "each_bank_coverage_gain_positive": True,
            "each_bank_dual_min": 0.75,
            "each_bank_harm_max": 0.25,
            "each_bank_all_four_mean_and_p90_non_degraded_vs_fixed": True,
        },
        "post_outcome_tuning_or_regrouping_allowed": False,
        "all_22_routes_become_spent_after_evaluation": True,
        "labels_opened": False,
        "outcomes_accessed": False,
        "sealed_25route_reserve": str(args.reserve_freeze),
        "sealed_25route_reserve_sha256": sha256(args.reserve_freeze),
        "sealed_25route_reserve_labels_opened": False,
        "sealed_25route_reserve_outcomes_accessed": False,
        "reserve_accessed": False,
    }
    write_json(output, freeze)
    print(json.dumps(freeze, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
