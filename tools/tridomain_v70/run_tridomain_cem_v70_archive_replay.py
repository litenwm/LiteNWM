#!/usr/bin/env python3
"""Run the original V70 safe-elite CEM/archive policy with new tri-domain weights.

This is intentionally a thin compatibility wrapper around the frozen
``run_dualdomain_cem_tail_v66_replay`` implementation.  It replaces only the
proposal-aware checkpoint/calibration/support bindings and the LTWM checkpoint
root.  Candidate-level safety elites, three CEM rounds, archive semantics, and
the no-safe-candidate incumbent fallback are unchanged.

Required environment variables:

* ``TRIDOMAIN_V70_RESULT``: final proposal-aware training ``result.json``;
* ``TRIDOMAIN_V70_CHECKPOINT_ROOT``: paired new LTWM/scorer-v2 feature root;
* ``TRIDOMAIN_RUNTIME_DATASET``: recon, scand, or sacson.

For SACSoN, the base replay is invoked with its legacy ``--domain scand`` CLI
token, while this wrapper binds SACSoN calibration and its 0.255 m spacing.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from tools.evaluation.export_recon_dense_dinov3_features_v1 import (  # noqa: E402
    checkpoint_pairs as feature_checkpoint_pairs,
)
from tools.planning import run_dualdomain_cem_tail_v66_replay as replay  # noqa: E402
from tools.scoring.train_recon_dense_meta_dinov3_v1 import (  # noqa: E402
    file_sha256,
)


RESULT = Path(os.environ["TRIDOMAIN_V70_RESULT"]).resolve(strict=True)
PROTOCOL_PREREG = Path(
    os.environ["TRIDOMAIN_V70_PROTOCOL_PREREG"]
).resolve(strict=True)
CHECKPOINT_ROOT = Path(
    os.environ["TRIDOMAIN_V70_CHECKPOINT_ROOT"]
).resolve(strict=True)
RUNTIME_DATASET = os.environ.get(
    "TRIDOMAIN_RUNTIME_DATASET", ""
).strip().lower()
if RUNTIME_DATASET not in {"recon", "scand", "sacson"}:
    raise RuntimeError(
        "TRIDOMAIN_RUNTIME_DATASET must be recon, scand, or sacson"
    )

_standard_build_universal_features = replay.build_universal_features
_standard_ensemble_predict = replay.ensemble_predict
_standard_load_envelopes = replay.load_envelopes
_standard_checkpoint_pairs = replay.checkpoint_pairs
_standard_sample_population = replay.sample_population
_standard_safe_elite_update = replay.safe_elite_update
_standard_select = replay.select_lowest_energy_safe
_standard_write_npz_atomic = replay.write_npz_atomic
_standard_write_json_atomic = replay.write_json_atomic

_archive_energy: np.ndarray | None = None
_archive_candidate: np.ndarray | None = None
_archive_source_round: np.ndarray | None = None
_archive_source_candidate: np.ndarray | None = None
_safe_update_calls = 0
_selection_calls = 0
DOMAIN_INDEX = {'recon': 0, 'scand': 1, 'sacson': 2}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def runtime_protocol_preregistration(args) -> Path:
    value = read_json(PROTOCOL_PREREG)
    if value.get("status") != "frozen_before_external_fair_selection_v1":
        raise RuntimeError("tri-domain fair-eval preregistration status changed")
    if value.get("external_test_labels_accessed_before_freeze"):
        raise RuntimeError("fair-eval preregistration reports pre-freeze label access")
    if value.get("selection_outputs_exist_at_freeze"):
        raise RuntimeError("fair-eval selection existed before protocol freeze")
    if value.get("frozen_result_sha256") != file_sha256(RESULT):
        raise RuntimeError("tri-domain result/preregistration SHA binding changed")
    seal_path = Path(value["seal_record"]).resolve(strict=True)
    if value.get("seal_record_sha256") != file_sha256(seal_path):
        raise RuntimeError("relaxed seal/preregistration SHA binding changed")
    cem = value.get("cem_contract", {})
    expected = {
        "iterations": args.iterations,
        "population": args.population,
        "elite_fraction": args.elite_fraction,
        "proposal_scale": args.proposal_scale,
        "support_quantile": args.support_quantile,
    }
    if any(
        abs(float(cem.get(key, float("nan"))) - float(expected_value))
        > 1.0e-12
        for key, expected_value in expected.items()
    ):
        raise RuntimeError("tri-domain preregistered CEM protocol changed")
    required = (
        "candidate_level_safety_before_elite",
        "cross_round_safe_archive",
        "retain_incumbent_when_no_safe_candidate",
    )
    if not all(bool(cem.get(key)) for key in required):
        raise RuntimeError("tri-domain preregistration lacks V70 safety semantics")
    return PROTOCOL_PREREG


def frozen_model_contract(
    *,
    role: str,
    domain: str,
    base_result_path: Path,
    v66_result_path: Path,
):
    del role, domain, base_result_path, v66_result_path
    value = read_json(RESULT)
    decision = value.get("decision")
    strict_seal = decision == "PROPOSAL_AWARE_V70_TRIDOMAIN_ENSEMBLE_FROZEN"
    relaxed_seal = decision == (
        "PROPOSAL_AWARE_V70_TRIDOMAIN_ENSEMBLE_"
        "FROZEN_RELAXED_FEASIBLE_V1"
    )
    if not (strict_seal or relaxed_seal):
        raise RuntimeError("tri-domain V70 result is not frozen")
    if relaxed_seal and (
        value.get("seal_status") != "RELAXED_FEASIBLE_V1"
        or value.get("strict_all_domains_feasible")
        or not value.get("relaxed_all_domains_feasible_v1")
    ):
        raise RuntimeError("tri-domain relaxed V70 seal is inconsistent")
    if not value.get("cem"):
        raise RuntimeError("tri-domain V70 result was not trained with CEM")
    if (
        value.get("A_to_H_used_by_v70_training_or_calibration")
        or value.get("outer_val_accessed")
        or value.get("test_accessed")
    ):
        raise RuntimeError("tri-domain V70 seal reports evaluation leakage")
    if value.get('model_domain_index') != DOMAIN_INDEX:
        raise RuntimeError(
            'frozen policy does not use independent three-domain conditioning'
        )
    calibration = value["calibration"][RUNTIME_DATASET]
    calibration_feasible = bool(
        calibration.get("feasible")
        if strict_seal
        else calibration.get("relaxed_feasible_v1")
    )
    if not calibration_feasible:
        raise RuntimeError(
            f"{RUNTIME_DATASET} final calibration is not feasible"
        )
    return (
        value["checkpoints"]["full"],
        calibration["policy"],
        float(calibration["dual_temperature"]),
        float(calibration["harm_temperature"]),
        {
            "model_role": "proposal_aware_v70_tridomain",
            "model_version": "v70_broad_hashfold_archive",
            "runtime_dataset": RUNTIME_DATASET,
            "model_result": str(RESULT),
            "model_result_sha256": file_sha256(RESULT),
            "model_trained_with_cem": True,
            "calibration_feasible": True,
            "calibration_contract_status": (
                "STRICT_FEASIBLE"
                if strict_seal
                else "RELAXED_FEASIBLE_V1"
            ),
            "strict_calibration_feasible": bool(calibration.get("feasible")),
            "external_test_used_by_training_or_calibration": False,
        },
    )


def runtime_checkpoint_pairs(_ignored: Path):
    return feature_checkpoint_pairs(CHECKPOINT_ROOT)


def runtime_ensemble_predict(checkpoints, feature, _legacy_domain, *, device):
    domain = np.full(
        len(feature), DOMAIN_INDEX[RUNTIME_DATASET], dtype=np.int64
    )
    return _standard_ensemble_predict(
        checkpoints, feature, domain, device=device
    )


def runtime_load_envelopes(_legacy_path):
    value = read_json(RESULT)
    envelopes = _standard_load_envelopes(Path(value['support']))
    selected = envelopes[RUNTIME_DATASET]
    return {
        'recon': selected,
        'scand': selected,
        'sacson': selected,
    }


def runtime_build_universal_features(data, *, spacing_m, support_envelope):
    if RUNTIME_DATASET == "sacson":
        spacing_m = 0.255
    return _standard_build_universal_features(
        data,
        spacing_m=spacing_m,
        support_envelope=support_envelope,
    )


def runtime_sample_population(**kwargs):
    if RUNTIME_DATASET == "sacson":
        kwargs["spacing_m"] = 0.255
    return _standard_sample_population(**kwargs)


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
        _archive_source_round = np.full(
            len(candidates), -2, dtype=np.int64
        )
        _archive_source_candidate = np.zeros(
            len(candidates), dtype=np.int64
        )
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
        raise RuntimeError("V70 archive was not initialized")
    return np.isfinite(_archive_energy).astype(np.int64)


def write_npz_with_archive(path: Path, **arrays):
    if path.name == "selected_actions.npz":
        if (
            _archive_energy is None
            or _archive_candidate is None
            or _archive_source_round is None
            or _archive_source_candidate is None
        ):
            raise RuntimeError("V70 archive is incomplete")
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
        if _archive_source_round is None:
            raise RuntimeError("V70 archive source audit is unavailable")
        config = {
            **value["config"],
            "cross_round_safe_archive": True,
            "spacing_m": (
                0.255
                if RUNTIME_DATASET == "sacson"
                else value["config"]["spacing_m"]
            ),
        }
        value = {
            **value,
            "schema": "proposal_aware_v70_tridomain_archive_prelabel_replay",
            "domain": RUNTIME_DATASET,
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
            "tridomain_v70_result": str(RESULT),
            "tridomain_v70_result_sha256": file_sha256(RESULT),
            "runtime_dataset": RUNTIME_DATASET,
            "implementation_contract": (
                "original V70 candidate-level safe elite, three-round CEM, "
                "cross-round safe archive, incumbent fallback"
            ),
            "outcomes_used_by_cem": False,
            "outer_val_accessed": False,
            "test_labels_accessed_before_selection": False,
        }
    _standard_write_json_atomic(path, value)


def main() -> None:
    replay.protocol_preregistration = runtime_protocol_preregistration
    replay.frozen_model_contract = frozen_model_contract
    replay.checkpoint_pairs = runtime_checkpoint_pairs
    replay.build_universal_features = runtime_build_universal_features
    replay.ensemble_predict = runtime_ensemble_predict
    replay.load_envelopes = runtime_load_envelopes
    replay.sample_population = runtime_sample_population
    replay.safe_elite_update = archived_safe_elite_update
    replay.select_lowest_energy_safe = archived_select
    replay.write_npz_atomic = write_npz_with_archive
    replay.write_json_atomic = write_json_with_archive
    replay.main()


if __name__ == "__main__":
    main()
