#!/usr/bin/env python3
"""Tri-domain V70 runtime loader with NumPy-1 checkpoint compatibility."""

from __future__ import annotations

from deploy.v70_go2_orin.numpy_pickle_compat import (
    install_numpy_pickle_compat,
)
from tools.planning.run_tridomain_cem_tail_v70_runtime import (
    frozen_model_contract,
)
from tools.planning import run_dualdomain_cem_tail_v66_replay as replay


def compatible_ensemble_predict(
    records,
    feature,
    domain,
    *,
    device,
):
    install_numpy_pickle_compat()
    return replay.ensemble_predict(
        records,
        feature,
        domain,
        device=device,
    )
