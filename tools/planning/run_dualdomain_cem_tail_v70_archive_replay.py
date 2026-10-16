#!/usr/bin/env python3
"""Run v70 with a cross-round safe archive over the frozen v66 CEM loop."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from tools.planning import run_dualdomain_cem_tail_v66_replay as replay  # noqa: E402
from tools.scoring.train_dualdomain_cem_tail_v69_scand_route_crossfit import (  # noqa: E402
    predict_crossfit_ensemble,
)
from tools.scoring.train_recon_dense_meta_dinov3_v1 import file_sha256  # noqa: E402


V70_RESULT = (
    ROOT
    / "outputs/cem_development_v1/"
    "dualdomain_cem_tail_v70_broad_hashfold_archive/result.json"
)
_standard_frozen_model_contract = replay.frozen_model_contract
_standard_ensemble_predict = replay.ensemble_predict
_standard_safe_elite_update = replay.safe_elite_update
_standard_select = replay.select_lowest_energy_safe
_standard_write_npz_atomic = replay.write_npz_atomic
_standard_write_json_atomic = replay.write_json_atomic

_archive_energy = None
_archive_candidate = None
_archive_source_round = None
_archive_source_candidate = None
_safe_update_calls = 0
_selection_calls = 0


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def frozen_model_contract(*, role, domain, base_result_path, v66_result_path):
    if role == "original_arm_d":
        return _standard_frozen_model_contract(
            role=role,
            domain=domain,
            base_result_path=base_result_path,
            v66_result_path=v66_result_path,
        )
    value = read_json(v66_result_path)
    if value.get("status") != "frozen_after_v69_spent_failure_before_v70_spent_replay":
        raise RuntimeError("v70 model/gate is not frozen before its spent replay")
    if value.get("A_to_H_used_by_v70_training_or_calibration"):
        raise RuntimeError("v70 reports forbidden A-H calibration use")
    if value.get("fresh_scand_used_by_v70_training_or_calibration"):
        raise RuntimeError("v70 reports forbidden fresh-SCAND calibration use")
    if value.get("reserve_accessed"):
        raise RuntimeError("v70 reports reserve access")
    calibration = value["calibration"][domain]
    if not calibration.get("feasible"):
        raise RuntimeError(f"v70 {domain} calibration is infeasible")
    return (
        value["checkpoints"]["full"],
        calibration["policy"],
        float(calibration["dual_temperature"]),
        float(calibration["harm_temperature"]),
        {
            "model_role": role,
            "model_version": "v70_broad_hashfold_archive",
            "model_result": str(v66_result_path),
            "model_result_sha256": file_sha256(v66_result_path),
            "model_trained_with_cem": True,
            "calibration_feasible": True,
            "A_to_H_used_by_calibration": False,
            "fresh_scand_used_by_calibration": False,
        },
    )


def compatible_ensemble_predict(records, feature, domain, *, device):
    hashes = {str(record.get("sha256", "")) for record in records}
    value = read_json(V70_RESULT)
    v70_hashes = {
        str(record["sha256"]) for record in value["checkpoints"]["full"]
    }
    if hashes and hashes <= v70_hashes:
        return predict_crossfit_ensemble(records, feature, domain, device=device)
    if hashes & v70_hashes:
        raise RuntimeError("mixed original/v70 checkpoint ensemble is forbidden")
    return _standard_ensemble_predict(records, feature, domain, device=device)


def archived_safe_elite_update(candidates, energy, safe, **kwargs):
    global _archive_energy
    global _archive_candidate
    global _archive_source_round
    global _archive_source_candidate
    global _safe_update_calls
    result = _standard_safe_elite_update(
        candidates, energy, safe, **kwargs
    )
    selected = _standard_select(energy, safe)
    if _archive_energy is None:
        _archive_energy = np.full(len(candidates), np.inf, dtype=np.float32)
        _archive_candidate = candidates[:, 0].copy()
        _archive_source_round = np.full(len(candidates), -2, dtype=np.int64)
        _archive_source_candidate = np.zeros(len(candidates), dtype=np.int64)
    row = np.arange(len(candidates))
    update = (
        (selected != 0)
        & (energy[row, selected] < _archive_energy)
    )
    local = np.flatnonzero(update)
    _archive_energy[local] = energy[local, selected[local]]
    _archive_candidate[local] = candidates[local, selected[local]]
    _archive_source_round[local] = _safe_update_calls - 1
    _archive_source_candidate[local] = selected[local]
    _safe_update_calls += 1
    return result


def archived_select(energy, safe):
    global _selection_calls
    _selection_calls += 1
    if _selection_calls == 1:
        return _standard_select(energy, safe)
    if _archive_energy is None:
        raise RuntimeError("v70 archive was not initialized")
    return np.isfinite(_archive_energy).astype(np.int64)


def write_npz_with_archive(path: Path, **arrays):
    if path.name == "selected_actions.npz":
        candidates = arrays["final_cem_feature_waypoint_units"].copy()
        energy = arrays["final_cem_energy"].copy()
        valid = arrays["final_cem_valid"].copy()
        safe = arrays["final_cem_safe"].copy()
        archived = np.isfinite(_archive_energy)
        candidates[:, 1] = _archive_candidate
        energy[:, 1] = _archive_energy
        valid[:, 1] = archived
        safe[:, 1] = archived
        arrays = {
            **arrays,
            "final_cem_feature_waypoint_units": candidates,
            "final_cem_selected_index": archived.astype(np.int64),
            "final_cem_energy": energy,
            "final_cem_valid": valid,
            "final_cem_safe": safe,
            "archive_energy": _archive_energy,
            "archive_source_round": _archive_source_round,
            "archive_source_candidate": _archive_source_candidate,
        }
    _standard_write_npz_atomic(path, **arrays)


def write_json_with_archive(path: Path, value):
    if value.get("schema") == "dualdomain_cem_tail_v66_prelabel_replay":
        config = {**value["config"], "cross_round_safe_archive": True}
        value = {
            **value,
            "schema": "dualdomain_cem_tail_v70_archive_prelabel_replay",
            "config": config,
            "cross_round_safe_archive": True,
            "archive_source_counts": {
                "none": int((_archive_source_round == -2).sum()),
                "fixed_pool": int((_archive_source_round == -1).sum()),
                **{
                    f"iteration_{iteration:02d}": int(
                        (_archive_source_round == iteration).sum()
                    )
                    for iteration in range(3)
                },
            },
            "v70_result": str(V70_RESULT),
            "v70_result_sha256": file_sha256(V70_RESULT),
            "prior_A_to_H_spent_by_v69": True,
            "prior_fresh_scand_spent_by_v69": True,
            "A_to_H_used_by_v70_training_or_calibration": False,
            "fresh_scand_used_by_v70_training_or_calibration": False,
            "outcomes_used_by_cem": False,
            "reserve_accessed": False,
        }
    _standard_write_json_atomic(path, value)


def main() -> None:
    replay.frozen_model_contract = frozen_model_contract
    replay.ensemble_predict = compatible_ensemble_predict
    replay.safe_elite_update = archived_safe_elite_update
    replay.select_lowest_energy_safe = archived_select
    replay.write_npz_atomic = write_npz_with_archive
    replay.write_json_atomic = write_json_with_archive
    replay.main()


if __name__ == "__main__":
    main()
