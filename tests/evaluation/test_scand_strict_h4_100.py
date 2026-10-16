"""Static contract tests for the frozen SCAND strict-test H4-100 package."""

from __future__ import annotations

import hashlib
import json
import pickle
import unittest
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "data/evaluation/scand/strict_test_h4_100_seed17"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class ScandStrictH4BenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = json.loads((PACKAGE / "manifest.json").read_text())
        cls.records = cls.manifest["records"]
        cls.splits = json.loads(
            (ROOT / "data/split_lock/scand_strict_splits.json").read_text()
        )

    def test_exact_outer_test_membership_and_full_trajectory_coverage(self) -> None:
        selected_ids = {record["trajectory_id"] for record in self.records}
        self.assertEqual(selected_ids, set(self.splits["test"]))
        self.assertTrue(selected_ids.isdisjoint(self.splits["train"]))
        self.assertTrue(selected_ids.isdisjoint(self.splits["val"]))
        self.assertEqual(len(selected_ids), 43)

    def test_window_and_stratification_contract(self) -> None:
        self.assertEqual(len(self.records), 100)
        windows_per_trajectory = Counter(
            record["trajectory_id"] for record in self.records
        )
        self.assertEqual(Counter(windows_per_trajectory.values()), {2: 29, 3: 14})
        self.assertEqual(
            Counter(record["length_bin"] for record in self.records),
            {0: 25, 1: 25, 2: 25, 3: 25},
        )
        self.assertEqual(
            Counter(record["temporal_bin"] for record in self.records),
            {0: 25, 1: 25, 2: 25, 3: 25},
        )
        for record in self.records:
            curr_time = record["curr_time"]
            self.assertEqual(
                record["context_times"], list(range(curr_time - 3, curr_time + 1))
            )
            self.assertEqual(
                record["gt_times"], list(range(curr_time + 1, curr_time + 5))
            )
            self.assertEqual(record["goal_time"], curr_time + 4)
            self.assertGreaterEqual(curr_time, 3)
            self.assertLessEqual(curr_time, record["num_frames"] - 5)

    def test_runner_index_matches_manifest(self) -> None:
        with (PACKAGE / "navigation_eval.pkl").open("rb") as handle:
            navigation_eval = pickle.load(handle)
        expected = [
            (record["trajectory_id"], record["curr_time"], 4, 4)
            for record in self.records
        ]
        self.assertEqual(navigation_eval, expected)

    def test_package_hashes(self) -> None:
        for line in (PACKAGE / "SHA256SUMS.txt").read_text().splitlines():
            expected_hash, relative_path = line.split("  ", maxsplit=1)
            self.assertEqual(sha256(PACKAGE / relative_path), expected_hash)


if __name__ == "__main__":
    unittest.main()
