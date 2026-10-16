#!/usr/bin/env python3
"""Read-only component gate counts for completed v77 OOF predictions."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / "outputs/cem_development_v1/v77_scand_tail_joint_selector/outer_oof_predictions.npz"
PROFILES = {
    "balanced": (1.0, 0.80, 0.08, 0.002),
    "conservative": (1.5, 0.90, 0.03, 0.004),
}


def main() -> None:
    with np.load(PATH, allow_pickle=False) as payload:
        data = {key: payload[key] for key in payload.files}
    pool = data["challenger_pool"].astype(bool)
    result = {}
    for model in ("linear", "listwise"):
        gain_mean, gain_std = data[f"{model}__gain_mean"], data[f"{model}__gain_std"]
        dual_mean, dual_std = data[f"{model}__p_dual_mean"], data[f"{model}__p_dual_std"]
        harm_mean, harm_std = data[f"{model}__p_harm_mean"], data[f"{model}__p_harm_std"]
        result[model] = {}
        for profile, (sigma, dual_min, harm_max, gain_min) in PROFILES.items():
            gain = gain_mean - sigma * gain_std
            dual = dual_mean - sigma * dual_std
            harm = harm_mean + sigma * harm_std
            result[model][profile] = {
                "pool": int(pool.sum()),
                "gain_pass": int((pool & (gain >= gain_min)).sum()),
                "dual_pass": int((pool & (dual >= dual_min)).sum()),
                "harm_pass": int((pool & (harm <= harm_max)).sum()),
                "dual_harm_pass": int((pool & (dual >= dual_min) & (harm <= harm_max)).sum()),
                "all_pass": int((pool & (gain >= gain_min) & (dual >= dual_min) & (harm <= harm_max)).sum()),
                "dual_lcb_quantiles": np.quantile(dual[pool], (0.5, 0.9, 0.99)).tolist(),
                "harm_ucb_quantiles": np.quantile(harm[pool], (0.01, 0.1, 0.5)).tolist(),
                "gain_lcb_m_quantiles": np.quantile(gain[pool], (0.5, 0.9, 0.99)).tolist(),
            }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
