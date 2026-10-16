#!/usr/bin/env python3
"""Read-only post-run overlap audit for v76 eligible candidates and frozen energy."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs/cem_development_v1/v76_scand_tail_safety_calibration"
PREDICTIONS = OUTPUT / "outer_oof_predictions.npz"
RECORDS = ROOT / "outputs/cem_development_v1/scand_filter_attribution_v1/candidate_records.csv"


def main() -> None:
    with np.load(PREDICTIONS, allow_pickle=False) as payload:
        prediction = {name: payload[name].copy() for name in payload.files}
    with RECORDS.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    original = np.asarray([row["safe_pass"] == "True" for row in rows]).reshape(100, 65)
    energy = np.asarray([float(row["energy"]) for row in rows]).reshape(100, 65)
    formal = np.asarray([row["is_formal_v70"] == "True" for row in rows]).reshape(100, 65)
    formal_energy = np.sum(np.where(formal, energy, 0.0), axis=1)
    names = (
        "location_scale__balanced__safe",
        "location_scale__conservative__safe",
        "conformal__balanced__safe",
        "conformal__conservative__safe",
    )
    result = {}
    for name in names:
        safe = prediction[name].astype(bool)
        energy_better = safe & (energy < formal_energy[:, None])
        result[name] = {
            "safe": int(safe.sum()),
            "new_vs_original": int((safe & ~original).sum()),
            "overlap_original": int((safe & original).sum()),
            "energy_better": int(energy_better.sum()),
            "states_energy_better": int(energy_better.any(axis=1).sum()),
        }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
