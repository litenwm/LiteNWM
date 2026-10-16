import numpy as np

from tools.planning.run_cem_development_v1 import conservative_energy
from tools.scoring.run_train_dualdomain_cem_tail_v66_candidate_level import (
    candidate_energy,
)
from tools.scoring.train_universal_dualdomain_raw_v2 import sigmoid


def test_candidate_level_calibration_energy_matches_cem_runtime() -> None:
    rng = np.random.default_rng(7)
    prediction = {
        "mean": rng.normal(size=(3, 17, 4)).astype(np.float32),
        "std": np.abs(rng.normal(size=(3, 17, 4))).astype(np.float32),
        "dual_logit": rng.normal(size=(3, 17)).astype(np.float32),
        "harm_logit": rng.normal(size=(3, 17)).astype(np.float32),
        "rank_logit": rng.normal(size=(3, 17)).astype(np.float32),
    }
    valid = np.ones((3, 17), dtype=bool)
    valid[1, 5] = False
    policy = {
        "z": 1.0,
        "margin_ate": 0.005,
        "margin_objective": 0.005,
        "dual_threshold": 0.7,
        "harm_threshold": 0.1,
    }
    p_dual = sigmoid(prediction["dual_logit"])
    p_harm = sigmoid(prediction["harm_logit"])
    expected, _, _ = conservative_energy(
        prediction,
        policy=policy,
        dual_temperature=1.0,
        harm_temperature=1.0,
        valid=valid,
    )
    actual = candidate_energy(prediction, p_dual, p_harm, valid, policy)
    np.testing.assert_allclose(actual, expected)
