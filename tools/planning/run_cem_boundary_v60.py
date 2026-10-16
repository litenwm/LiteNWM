#!/usr/bin/env python3
"""Run the frozen v60 candidate gate in the audited true-CEM v55 engine."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from tools.scoring.train_recon_dense_meta_dinov3_v1 import file_sha256  # noqa: E402
from tools.scoring.train_universal_dualdomain_raw_v2 import sigmoid  # noqa: E402


def load_dynamic_engine() -> SimpleNamespace:
    """Load v55 with only its 100-state assertion generalized to N states.

    The base energy/valid arrays are also persisted for the pre-outcome union
    selector.  The authoritative v55 source file is never modified.
    """

    path = ROOT / "tools/planning/run_cem_safe_elite_v55.py"
    source = path.read_text(encoding="utf-8")
    old_identity = "len(rows) != 100\n        or not np.array_equal(record_key, expected_record)"
    new_identity = "not rows\n        or not np.array_equal(record_key, expected_record)"
    old_output = "base_safe= safe,\n        final_cem_energy=final_energy,"
    new_output = (
        "base_energy=energy,\n        base_valid=valid,\n        base_safe=safe,\n"
        "        final_cem_energy=final_energy,"
    )
    if source.count(old_identity) != 1 or source.count(old_output) != 1:
        raise RuntimeError("v55 engine source contract changed")
    source = source.replace(old_identity, new_identity).replace(old_output, new_output)
    namespace: dict[str, object] = {
        "__file__": str(path),
        "__name__": "tools.planning._run_cem_boundary_v60_dynamic",
        "__package__": "tools.planning",
    }
    exec(compile(source, str(path), "exec"), namespace)
    return SimpleNamespace(**namespace)


def extract_boundary() -> Path:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--boundary-freeze", type=Path, required=True)
    known, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    return known.boundary_freeze


def main() -> None:
    boundary_path = extract_boundary()
    boundary = json.loads(boundary_path.read_text(encoding="utf-8"))
    if (
        boundary.get("status")
        != "unique_boundary_rule_frozen_before_new_GH_manifests"
        or boundary.get("G_H_outcomes_accessed")
        or boundary.get("reserve_accessed")
        or boundary.get("grid_expanded")
        or boundary.get("post_GH_tuning_allowed")
    ):
        raise RuntimeError("v60 boundary freeze is invalid")
    config = boundary["selected_rule"]
    expected = {
        "id": 29,
        "beta": 1.5,
        "benefit_threshold": 0.0,
        "dual_threshold": 0.75,
        "harm_threshold": 0.25,
        "dominance_margin": 0.01,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise RuntimeError(f"frozen v60 {key} changed")

    engine = load_dynamic_engine()

    def conservative_energy(
        prediction: dict[str, np.ndarray],
        *,
        policy: dict[str, float],
        dual_temperature: float,
        harm_temperature: float,
        valid: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        del policy
        p_dual = sigmoid(prediction["dual_logit"] / dual_temperature)
        p_harm = sigmoid(prediction["harm_logit"] / harm_temperature)
        mean = prediction["mean"]
        std = prediction["std"]
        benefit_lcb = -0.5 * (mean[..., 0] + mean[..., 3]) - float(
            config["beta"]
        ) * np.sqrt(std[..., 0] ** 2 + std[..., 3] ** 2)
        energy = -benefit_lcb + 0.25 * p_harm - 0.10 * p_dual
        energy = np.where(valid, energy, np.inf).astype(np.float32)
        energy[:, 0] = np.inf
        return energy, p_dual.astype(np.float32), p_harm.astype(np.float32)

    def candidate_safety_mask(
        prediction: dict[str, np.ndarray],
        p_dual: np.ndarray,
        p_harm: np.ndarray,
        valid: np.ndarray,
        policy: dict[str, float],
    ) -> np.ndarray:
        del policy
        mean = prediction["mean"]
        std = prediction["std"]
        benefit_lcb = -0.5 * (mean[..., 0] + mean[..., 3]) - float(
            config["beta"]
        ) * np.sqrt(std[..., 0] ** 2 + std[..., 3] ** 2)
        safe = (
            np.asarray(valid, dtype=bool)
            & (benefit_lcb >= float(config["benefit_threshold"]))
            & (p_dual >= float(config["dual_threshold"]))
            & (p_harm <= float(config["harm_threshold"]))
        )
        safe[:, 0] = False
        return safe

    engine.conservative_energy = conservative_energy
    engine.candidate_safety_mask = candidate_safety_mask
    engine.main()

    output_index = sys.argv.index("--output-dir") + 1
    output_dir = Path(sys.argv[output_index])
    binding = {
        "schema": "cem_boundary_v60_runtime_binding",
        "status": "true_cem_completed_without_outcomes",
        "boundary_freeze": str(boundary_path),
        "boundary_freeze_sha256": file_sha256(boundary_path),
        "selected_rule": config,
        "formula": boundary["formula"],
        "dynamic_engine_change": (
            "only generalize N=100 identity assertion to N>0 and persist "
            "base energy/valid arrays"
        ),
        "labels_opened": False,
        "outcomes_accessed": False,
        "reserve_accessed": False,
        "state_gate_used": False,
    }
    (output_dir / "v60_runtime_binding.json").write_text(
        json.dumps(binding, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
