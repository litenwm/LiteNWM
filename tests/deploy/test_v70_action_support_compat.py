"""Tests for deployment action-support compatibility."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO_ROOT / "deploy/v70_go2_orin"), str(REPO_ROOT / "src")]

from v70_action_support_compat import (  # noqa: E402
    DEFAULT_TAIL_SUPPORT_PATH,
    coherent_turn_tail_support,
)


SPACING_SUPPORT_SHA256 = (
    "da86c3f8286da60ac96fd0c3a9cd612e"
    "cdc3ed4198a82345868fa24592a6bfd9"
)


class V70ActionSupportCompatibilityTests(unittest.TestCase):
    @unittest.skipUnless(
        Path(os.environ.get("V70_ACTION_TAIL_SUPPORT", str(DEFAULT_TAIL_SUPPORT_PATH))).is_file(),
        "External coherent-turn support array is not included in the source release",
    )
    def test_current_coherent_turn_is_supported(self) -> None:
        actions = np.asarray(
            [[[[0.08326671, 0.01920429, 0.22667220],
               [0.07485662, 0.02769784, 0.35439044],
               [0.08392824, 0.04971047, 0.53473650],
               [0.12114029, 0.05376020, 0.41767320]]]],
            dtype=np.float32,
        )
        support, distance = coherent_turn_tail_support(
            actions,
            spacing_support_sha256=SPACING_SUPPORT_SHA256,
        )
        self.assertEqual(support.tolist(), [[True]])
        self.assertAlmostEqual(float(distance[0, 0]), 0.053594, places=5)

    def test_malformed_turns_and_wrong_envelope_are_rejected(self) -> None:
        action = np.asarray(
            [[[0.08326671, 0.01920429, 0.22667220],
              [0.07485662, 0.02769784, 0.35439044],
              [0.08392824, 0.04971047, 0.53473650],
              [0.12114029, 0.05376020, 0.41767320]]],
            dtype=np.float32,
        )
        actions = np.repeat(action, 2, axis=0)
        actions[0, 2, :2] = np.asarray([0.005, 0.0], dtype=np.float32)
        actions[1, :, 2] = np.asarray(
            [0.22667220, -0.35439044, 0.53473650, -0.41767320],
            dtype=np.float32,
        )
        support, _ = coherent_turn_tail_support(
            actions[None],
            spacing_support_sha256=SPACING_SUPPORT_SHA256,
        )
        self.assertEqual(support.tolist(), [[False, False]])

        wrong_support, wrong_distance = coherent_turn_tail_support(
            action[None],
            spacing_support_sha256="0" * 64,
        )
        self.assertEqual(wrong_support.tolist(), [[False]])
        self.assertTrue(np.isinf(wrong_distance).all())


if __name__ == "__main__":
    unittest.main()
