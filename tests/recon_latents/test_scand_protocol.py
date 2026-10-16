from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tools.recon.process_scand_bags_rosbags import (  # noqa: E402
    eligible_ltwm_windows,
    forward_segments,
    select_official_segments,
)


def sample(x: float, timestamp: int):
    return (
        Image.new("RGB", (4, 3)),
        np.asarray([x, 0.0], dtype=np.float32),
        0.0,
        timestamp,
    )


def test_four_history_four_future_window_count() -> None:
    assert eligible_ltwm_windows(7) == 0
    assert eligible_ltwm_windows(8) == 1
    assert eligible_ltwm_windows(9) == 2


def test_short_official_segment_is_not_dropped() -> None:
    samples = [sample(float(index), index) for index in range(9)]
    segments = forward_segments(samples, min_segment_frames=1)
    assert [len(segment) for segment in segments] == [9]
    assert eligible_ltwm_windows(len(segments[0])) == 2


def test_official_cardinality_discards_only_shorter_extra_fragments() -> None:
    segments = [
        [sample(float(index), index) for index in range(4)],
        [sample(float(index), index) for index in range(20)],
    ]
    selected, discarded = select_official_segments(
        segments, ["random_mdps_example_0"]
    )
    assert [len(segment) for segment in selected] == [20]
    assert discarded == [{"raw_segment_index": 0, "frames": 4}]
