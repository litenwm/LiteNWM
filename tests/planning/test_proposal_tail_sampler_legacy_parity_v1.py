from __future__ import annotations

import copy

import numpy as np

from tests.planning.test_proposal_tail_sidecar_v1 import (
    nomad,
    sample_population,
    spacing,
    support_envelope,
)


def legacy_sample_population(
    *, incumbent, reference, mean, covariance, population, rng,
    support_envelope, support_threshold, spacing_m,
):
    """Exact pre-observer sampler body retained only for regression parity."""

    states = len(incumbent)
    dim = mean.shape[-1]
    result = np.empty((states, population + 1, 4, 2), dtype=np.float32)
    result[:, 0] = incumbent
    reference_flat = reference.reshape(states, reference.shape[1], dim)
    lower = reference_flat.min(axis=1) - 0.35 * reference_flat.std(axis=1)
    upper = reference_flat.max(axis=1) + 0.35 * reference_flat.std(axis=1)
    accepted_total = 0
    attempted_total = 0
    fallback_total = 0
    for state in range(states):
        accepted = []
        attempts = 0
        maximum_attempts = population * 24
        while len(accepted) < population and attempts < maximum_attempts:
            draw_count = min(64, maximum_attempts - attempts)
            draw = rng.multivariate_normal(
                mean[state].astype(np.float64),
                covariance[state].astype(np.float64),
                size=draw_count,
                check_valid="raise",
            )
            draw = np.clip(draw, lower[state], upper[state]).astype(np.float32)
            paths = draw.reshape(draw_count, 4, 2)
            pool = np.concatenate((incumbent[state : state + 1], paths), axis=0)[None]
            actions = nomad.nomad_waypoints_to_local_actions(pool, dataset="recon")
            score, guard = spacing.score_spacing_envelope(
                support_envelope,
                actions,
                np.asarray([spacing_m], dtype=np.float32),
            )
            valid = guard[0, 1:] & (score[0, 1:] <= support_threshold)
            accepted.extend(paths[valid])
            attempts += draw_count
            attempted_total += draw_count
            accepted_total += int(valid.sum())
        if len(accepted) < population:
            missing = population - len(accepted)
            fallback_total += missing
            accepted.extend(reference[state, :missing])
        result[state, 1:] = np.asarray(accepted[:population], dtype=np.float32)
    return result, {
        "attempted_samples": int(attempted_total),
        "support_accepted_samples": int(accepted_total),
        "support_acceptance": float(accepted_total / attempted_total) if attempted_total else 0.0,
        "fallback_samples": int(fallback_total),
        "online_causal": True,
        "future_route_windows_used": False,
    }


def fixture():
    incumbent = np.zeros((2, 4, 2), dtype=np.float32)
    reference = np.asarray(
        [
            [[[0.1 * step, 0.01 * candidate] for step in range(1, 5)] for candidate in range(16)],
            [[[0.08 * step, -0.01 * candidate] for step in range(1, 5)] for candidate in range(16)],
        ],
        dtype=np.float32,
    )
    mean = reference.reshape(2, 16, 8).mean(axis=1)
    covariance = np.broadcast_to(np.eye(8, dtype=np.float32) * 0.01, (2, 8, 8)).copy()
    return incumbent, reference, mean, covariance


def test_logger_off_matches_preobserver_sampler_bitwise() -> None:
    incumbent, reference, mean, covariance = fixture()
    legacy_rng = np.random.default_rng(20260730)
    current_rng = np.random.default_rng(20260730)
    kwargs = {
        "incumbent": incumbent,
        "reference": reference,
        "mean": mean,
        "covariance": covariance,
        "population": 16,
        "support_envelope": support_envelope(),
        "support_threshold": 100.0,
        "spacing_m": 0.38,
    }
    legacy, legacy_audit = legacy_sample_population(rng=legacy_rng, **kwargs)
    current, current_audit = sample_population(rng=current_rng, observer=None, **kwargs)
    assert legacy.tobytes() == current.tobytes()
    assert legacy_audit == current_audit
    assert copy.deepcopy(legacy_rng.bit_generator.state) == copy.deepcopy(current_rng.bit_generator.state)
