#!/usr/bin/env python3
"""Run v24 while preserving no-support states as ineligible fallbacks."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tools.scoring import screen_recon_target_state_gate_v24 as screen
from tools.scoring.train_recon_cem_state_gate_v6 import method_summary


def frozen_challenger(score: dict, valid: np.ndarray) -> np.ndarray:
    uncertainty = score["std"][..., 0] + score["std"][..., 3]
    raw = score["dual"] - 0.5 * score["harm"] - 0.5 * uncertainty
    masked = np.where(valid, raw, -np.inf)
    candidate = masked.argmax(1).astype(np.int64)
    eligible = valid.any(1)
    # -16 indexes the first challenger for feature-shape continuity while the
    # negative sign remains an explicit, auditable no-promotion marker.
    candidate[~eligible] = -16
    return candidate


def top40(
    p_dual: np.ndarray, p_harm: np.ndarray,
    p_dual_std: np.ndarray, p_harm_std: np.ndarray,
    challenger: np.ndarray, delta: np.ndarray, absolute: np.ndarray,
) -> dict:
    best = None
    eligible = challenger >= 0
    for z in (0.0, 0.5, 1.0, 1.5, 2.0):
        for weight in (0.5, 1.0, 2.0, 4.0):
            risk = p_dual - z * p_dual_std - weight * (p_harm + z * p_harm_std)
            risk = np.where(eligible, risk, -np.inf)
            order = np.argsort(-risk, kind="stable")[:40]
            order = order[np.isfinite(risk[order])]
            selected = np.zeros(len(challenger), dtype=np.int64)
            selected[order] = challenger[order] + 1
            metrics = method_summary(selected, delta, absolute)
            item = {"z": z, "risk_weight": weight, "metrics": metrics}
            key = (
                metrics["dual_precision"] or 0.0,
                -(metrics["harm_rate"] if metrics["harm_rate"] is not None else 1.0),
            )
            if best is None or key > best[0]:
                best = (key, item)
    assert best is not None
    return best[1]


if __name__ == "__main__":
    screen.frozen_challenger = frozen_challenger
    screen.top40 = top40
    screen.main()
