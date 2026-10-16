from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from planning.nomad_action_adapter import (
    integrate_local_actions,
    nomad_waypoints_to_local_actions,
    window_seed,
)
from tools.planning.build_nomad_h4_proposals_v2 import load_partial, save_partial


def test_nomad_conversion_round_trips_curved_waypoints() -> None:
    waypoints = np.asarray(
        [
            [
                (1.0, 0.0),
                (2.0, 0.0),
                (2.0, 1.0),
                (1.5, 1.5),
            ]
        ],
        dtype=np.float32,
    )
    actions = nomad_waypoints_to_local_actions(waypoints, dataset="recon")
    reconstructed = integrate_local_actions(actions)
    np.testing.assert_allclose(reconstructed, waypoints * 0.25, atol=1.0e-6)
    assert actions.shape == (1, 4, 3)
    assert float(actions[0, 2, 2]) == pytest.approx(math.pi / 2.0)


def test_dataset_spacing_is_explicit() -> None:
    waypoints = np.asarray([[(1.0, 0.0)] * 4], dtype=np.float32)
    recon = integrate_local_actions(
        nomad_waypoints_to_local_actions(waypoints, dataset="recon")
    )
    scand = integrate_local_actions(
        nomad_waypoints_to_local_actions(waypoints, dataset="scand")
    )
    sacson = integrate_local_actions(
        nomad_waypoints_to_local_actions(waypoints, dataset="sacson")
    )
    assert float(recon[0, -1, 0]) == pytest.approx(0.25)
    assert float(scand[0, -1, 0]) == pytest.approx(0.38)
    assert float(sacson[0, -1, 0]) == pytest.approx(0.255)


def test_zero_length_segment_retains_previous_heading() -> None:
    waypoints = np.asarray(
        [[(1.0, 1.0), (1.0, 1.0), (2.0, 2.0), (2.0, 2.0)]],
        dtype=np.float32,
    )
    actions = nomad_waypoints_to_local_actions(waypoints, dataset="recon")
    reconstructed = integrate_local_actions(actions)
    np.testing.assert_allclose(reconstructed, waypoints * 0.25, atol=1.0e-6)
    assert float(actions[0, 1, 2]) == pytest.approx(0.0)


def test_window_seed_is_stable_and_order_independent() -> None:
    assert window_seed(17, "abc") == window_seed(17, "abc")
    assert window_seed(17, "abc") != window_seed(17, "def")
    assert window_seed(17, "abc") != window_seed(18, "abc")


def test_conversion_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError, match="at least H=4"):
        nomad_waypoints_to_local_actions(
            np.zeros((16, 3, 2), dtype=np.float32), dataset="recon"
        )
    invalid = np.zeros((16, 4, 2), dtype=np.float32)
    invalid[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN or Inf"):
        nomad_waypoints_to_local_actions(invalid, dataset="recon")


def test_partial_resume_is_bound_to_exact_provenance(tmp_path: Path) -> None:
    rows: list[dict[str, object]] = [{"window_uid": "uid0"}]
    actions = [np.zeros((16, 4, 3), dtype=np.float32)]
    binding: dict[str, object] = {
        "schema": "h4_nomad_partial_v2",
        "dataset": "recon",
        "partition": "train-fit",
        "inner_manifest_sha256": "a" * 64,
        "split_lock_sha256": "b" * 64,
        "nomad_checkpoint_sha256": "c" * 64,
        "base_seed": 17,
    }
    partial_path = tmp_path / "proposals.partial.npz"
    save_partial(partial_path, rows, actions, binding)

    restored = load_partial(partial_path, rows, binding)
    assert len(restored) == 1
    np.testing.assert_array_equal(restored[0], actions[0])

    mismatched = dict(binding)
    mismatched["base_seed"] = 18
    with pytest.raises(RuntimeError, match="provenance"):
        load_partial(partial_path, rows, mismatched)
