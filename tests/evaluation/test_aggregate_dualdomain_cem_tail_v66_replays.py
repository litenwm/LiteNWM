import numpy as np

from tools.evaluation.aggregate_dualdomain_cem_tail_v66_replays import (
    comparison,
    pooled_metrics,
)


def test_pooled_metrics_weight_states_and_promotions_exactly() -> None:
    policy = "proposal_aware_v66_cem"
    parts = [
        {
            f"{policy}__promoted": np.asarray([True, False]),
            f"{policy}__dual": np.asarray([True, False]),
            f"{policy}__harmed": np.asarray([False, False]),
            f"{policy}__absolute_metrics": np.asarray(
                [[0.1, 0.2, 0.3, 0.4], [0.2, 0.3, 0.4, 0.5]],
                dtype=np.float32,
            ),
            f"{policy}__delta_metrics": np.asarray(
                [[-0.1, -0.1, -0.1, -0.1], [0.0, 0.0, 0.0, 0.0]],
                dtype=np.float32,
            ),
        },
        {
            f"{policy}__promoted": np.asarray([True]),
            f"{policy}__dual": np.asarray([False]),
            f"{policy}__harmed": np.asarray([True]),
            f"{policy}__absolute_metrics": np.asarray(
                [[0.3, 0.4, 0.5, 0.6]], dtype=np.float32
            ),
            f"{policy}__delta_metrics": np.asarray(
                [[0.1, 0.1, 0.1, 0.1]], dtype=np.float32
            ),
        },
    ]
    pooled = pooled_metrics(parts, policy)
    assert pooled["states"] == 3
    assert pooled["coverage"] == 2.0 / 3.0
    assert pooled["dual_precision"] == 0.5
    assert pooled["harm_rate"] == 0.5


def test_comparison_requires_all_four_mean_and_p90_metrics() -> None:
    baseline = {
        "coverage": 0.2,
        "absolute_mean_m": {name: 1.0 for name in ("ate", "rpe", "endpoint", "objective")},
        "absolute_p90_m": {name: 1.0 for name in ("ate", "rpe", "endpoint", "objective")},
    }
    challenger = {
        "coverage": 0.4,
        "absolute_mean_m": dict(baseline["absolute_mean_m"]),
        "absolute_p90_m": dict(baseline["absolute_p90_m"]),
    }
    challenger["absolute_p90_m"]["endpoint"] = 1.01
    value = comparison(challenger, baseline)
    assert value["coverage_gain"] == 0.2
    assert value["all_four_mean_nondegraded"]
    assert not value["all_four_p90_nondegraded"]
