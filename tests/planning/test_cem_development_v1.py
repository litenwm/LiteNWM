from __future__ import annotations

import numpy as np

from scoring.spacing_support_v1 import fit_spacing_envelope, threshold_for_quantile
from tools.planning.run_cem_development_v1 import (
    conservative_energy,
    fit_proposal,
    sample_population,
)


def simple_paths(states: int = 3, candidates: int = 17) -> np.ndarray:
    x = np.linspace(0.2, 0.8, 4, dtype=np.float32)
    result = np.zeros((states, candidates, 4, 2), dtype=np.float32)
    for state in range(states):
        for candidate in range(candidates):
            result[state, candidate, :, 0] = x * (1.0 + 0.01 * candidate)
            result[state, candidate, :, 1] = 0.002 * (candidate - 8)
    return result


def test_conservative_energy_respects_validity() -> None:
    shape = (2, 17)
    prediction = {
        "mean": np.zeros((*shape, 4), dtype=np.float32),
        "std": np.full((*shape, 4), 0.01, dtype=np.float32),
        "dual_logit": np.zeros(shape, dtype=np.float32),
        "harm_logit": np.zeros(shape, dtype=np.float32),
        "rank_logit": np.tile(np.arange(17, dtype=np.float32), (2, 1)),
    }
    valid = np.ones(shape, dtype=bool)
    valid[:, 8] = False
    energy, p_dual, p_harm = conservative_energy(
        prediction,
        policy={"z": 1.0},
        dual_temperature=1.0,
        harm_temperature=1.0,
        valid=valid,
    )
    assert np.isinf(energy[:, 0]).all()
    assert np.isinf(energy[:, 8]).all()
    assert np.isfinite(energy[:, 1:8]).all()
    assert np.allclose(p_dual, 0.5)
    assert np.allclose(p_harm, 0.5)


def test_fit_and_sample_population_preserve_k17_and_support() -> None:
    candidates = simple_paths()
    actions = np.zeros((3, 17, 4, 3), dtype=np.float32)
    actions[..., 0] = np.diff(
        np.concatenate(
            (
                np.zeros((3, 17, 1), dtype=np.float32),
                candidates[..., 0] * 0.25,
            ),
            axis=-1,
        ),
        axis=-1,
    )
    spacing = np.full(3, 0.25, dtype=np.float32)
    envelope = fit_spacing_envelope(actions, spacing)
    energy = np.tile(np.linspace(1.0, 0.0, 17, dtype=np.float32), (3, 1))
    energy[:, 0] = np.inf
    mean, covariance, elite = fit_proposal(
        candidates,
        energy,
        elite_count=4,
        floor=0.005,
        scale=0.5,
    )
    sampled, audit = sample_population(
        incumbent=candidates[:, 0],
        reference=candidates[:, 1:],
        mean=mean,
        covariance=covariance,
        population=16,
        rng=np.random.default_rng(17),
        support_envelope=envelope,
        support_threshold=threshold_for_quantile(envelope, 0.999),
        spacing_m=0.25,
    )
    assert sampled.shape == (3, 17, 4, 2)
    assert np.array_equal(sampled[:, 0], candidates[:, 0])
    assert mean.shape == (3, 8)
    assert covariance.shape == (3, 8, 8)
    assert elite.shape == (3, 4)
    assert audit["online_causal"] is True
    assert audit["future_route_windows_used"] is False
