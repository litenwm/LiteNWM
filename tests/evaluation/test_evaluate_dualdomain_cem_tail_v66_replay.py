import numpy as np

from tools.evaluation.evaluate_dualdomain_cem_tail_v66_replay import (
    decision_metrics,
    metric_differences,
)


def test_metrics_treat_index_zero_as_abstention_and_report_quality() -> None:
    candidates = np.zeros((2, 17, 4, 2), dtype=np.float32)
    ground_truth = np.ones((2, 4, 2), dtype=np.float32)
    candidates[:, 1] = ground_truth / 0.25
    improved, arrays = decision_metrics(
        candidates, np.asarray([1, 0], dtype=np.int64), ground_truth
    )
    baseline, _ = decision_metrics(
        candidates, np.asarray([0, 0], dtype=np.int64), ground_truth
    )
    difference = metric_differences(improved, baseline)
    assert improved["coverage"] == 0.5
    assert improved["dual_precision"] == 1.0
    assert improved["harm_rate"] == 0.0
    assert arrays["promoted"].tolist() == [True, False]
    assert difference["coverage_gain"] == 0.5
    assert difference["all_four_mean_nondegraded"]
