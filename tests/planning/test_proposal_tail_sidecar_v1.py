from __future__ import annotations

import ast
import copy
import importlib.util
from pathlib import Path
from typing import Any, Callable

import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def load_source_module(name: str, relative: str):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


nomad = load_source_module(
    "proposal_tail_nomad_action_adapter",
    "src/planning/nomad_action_adapter.py",
)
spacing = load_source_module(
    "proposal_tail_spacing_support",
    "src/scoring/spacing_support_v1.py",
)


def load_sample_population():
    """Load the actual sampler body without importing its unavailable torch CLI stack."""

    path = ROOT / "tools/planning/run_cem_development_v1.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "sample_population"
    )
    module = ast.Module(body=[function], type_ignores=[])
    namespace = {
        "np": np,
        "nomad_waypoints_to_local_actions": nomad.nomad_waypoints_to_local_actions,
        "normalized_action_vector": spacing.normalized_action_vector,
        "score_spacing_envelope": spacing.score_spacing_envelope,
        "Any": Any,
        "Callable": Callable,
    }
    exec(compile(module, str(path), "exec"), namespace)
    return namespace["sample_population"]


sample_population = load_sample_population()


def support_envelope() -> dict:
    return {
        "schema": "domain_aware_spacing_normalized_envelope_v1",
        "mean": np.zeros(12, dtype=np.float32),
        "covariance_inverse": np.eye(12, dtype=np.float32),
        "robust_center": np.zeros(12, dtype=np.float32),
        "robust_scale": np.ones(12, dtype=np.float32),
        "coordinate_abs_limit": np.full(12, 100.0, dtype=np.float32),
    }


def call(observer):
    incumbent = np.zeros((2, 4, 2), dtype=np.float32)
    reference = np.asarray(
        [
            [
                [[0.1 * step, 0.01 * candidate] for step in range(1, 5)]
                for candidate in range(16)
            ],
            [
                [[0.08 * step, -0.01 * candidate] for step in range(1, 5)]
                for candidate in range(16)
            ],
        ],
        dtype=np.float32,
    )
    mean = reference.reshape(2, 16, 8).mean(axis=1)
    covariance = np.broadcast_to(
        np.eye(8, dtype=np.float32) * 0.01, (2, 8, 8)
    ).copy()
    rng = np.random.default_rng(20260730)
    result, audit = sample_population(
        incumbent=incumbent,
        reference=reference,
        mean=mean,
        covariance=covariance,
        population=16,
        rng=rng,
        support_envelope=support_envelope(),
        support_threshold=100.0,
        spacing_m=0.38,
        observer=observer,
    )
    return result, audit, copy.deepcopy(rng.bit_generator.state)


def test_raw_draw_observer_is_bitwise_transparent() -> None:
    off_result, off_audit, off_state = call(None)
    events = []

    def observer(event):
        events.append(event)
        for value in event.values():
            if isinstance(value, np.ndarray):
                value[...] = 0

    on_result, on_audit, on_state = call(observer)
    assert np.array_equal(on_result, off_result)
    assert on_result.tobytes() == off_result.tobytes()
    assert on_audit == off_audit
    assert on_state == off_state
    assert any(event["event"] == "draw_batch" for event in events)
    assert sum(event["event"] == "state_complete" for event in events) == 2


def test_observer_records_prefilter_rejection_fields() -> None:
    events = []
    call(events.append)
    draw = next(event for event in events if event["event"] == "draw_batch")
    assert draw["raw_draw_unclipped"].shape[1] == 8
    assert draw["physical_coordinate_mask"].shape[1] == 12
    assert draw["support_margin"].shape == draw["support_score"].shape
    assert draw["final_candidate_index"].shape == draw["accepted_mask"].shape
