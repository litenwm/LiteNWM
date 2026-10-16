#!/usr/bin/env python3
"""Run the frozen v66 replay protocol with compatible v69 checkpoints.

The v69 SCAND-head checkpoints intentionally retain the original Arm-D model
state but use the sealed head-training checkpoint schema.  This adapter only
selects the matching loader; sampling, validity, safety, elite selection, and
all replay configuration remain owned by the frozen v66 implementation.
"""

from __future__ import annotations

import json
from pathlib import Path

from tools.planning import run_dualdomain_cem_tail_v66_replay as replay
from tools.scoring.train_dualdomain_cem_tail_v69_scand_route_crossfit import (
    predict_crossfit_ensemble,
)


V69_RESULT = (
    replay.ROOT
    / "outputs/cem_development_v1/"
    "dualdomain_cem_tail_v69_scand_route_crossfit/training/result.json"
)
_standard_ensemble_predict = replay.ensemble_predict


def _v69_checkpoint_hashes() -> set[str]:
    value = json.loads(V69_RESULT.read_text(encoding="utf-8"))
    if value.get("status") != "trained_and_frozen_before_A_to_H_replay":
        raise RuntimeError("v69 model/gate is not frozen before A-H replay")
    if value.get("A_to_H_accessed") or value.get("reserve_accessed"):
        raise RuntimeError("v69 training seal reports forbidden evaluation access")
    return {str(record["sha256"]) for record in value["checkpoints"]["full"]}


V69_CHECKPOINT_HASHES = _v69_checkpoint_hashes()


def compatible_ensemble_predict(records, feature, domain, *, device):
    hashes = {str(record.get("sha256", "")) for record in records}
    if hashes and hashes <= V69_CHECKPOINT_HASHES:
        return predict_crossfit_ensemble(records, feature, domain, device=device)
    if hashes & V69_CHECKPOINT_HASHES:
        raise RuntimeError("mixed original/v69 checkpoint ensemble is forbidden")
    return _standard_ensemble_predict(records, feature, domain, device=device)


def main() -> None:
    replay.ensemble_predict = compatible_ensemble_predict
    replay.main()


if __name__ == "__main__":
    main()
