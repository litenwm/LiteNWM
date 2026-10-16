from pathlib import Path

import numpy as np

from tools.planning.run_dualdomain_cem_tail_v66_replay import (
    BASE_RESULT,
    frozen_model_contract,
)


def test_original_arm_d_contract_loads_without_v66_result() -> None:
    checkpoints, policy, dual_temperature, harm_temperature, audit = (
        frozen_model_contract(
            role="original_arm_d",
            domain="scand",
            base_result_path=BASE_RESULT,
            v66_result_path=Path("does-not-need-to-exist.json"),
        )
    )
    assert len(checkpoints) == 3
    assert set(policy) == {
        "z",
        "margin_ate",
        "margin_objective",
        "dual_threshold",
        "harm_threshold",
    }
    assert np.isfinite(dual_temperature)
    assert np.isfinite(harm_temperature)
    assert audit["model_role"] == "original_arm_d"
    assert audit["model_trained_with_cem"] is False
