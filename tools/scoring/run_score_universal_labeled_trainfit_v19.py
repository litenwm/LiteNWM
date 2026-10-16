#!/usr/bin/env python3
"""Score a labeled train-fit artifact without exposing labels to inference."""

from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tools.scoring import score_universal_raw_feature_pool_v1 as scorer
from tools.scoring.train_recon_dense_meta_dinov3_three_arm_v1 import (
    load_features as _load_features,
)


def load_labeled_features_for_inference(
    path: Path,
    *,
    labels_expected: bool,
):
    del labels_expected
    value = _load_features(path, labels_expected=True)
    value.pop("targets", None)
    return value


if __name__ == "__main__":
    scorer.load_features = load_labeled_features_for_inference
    scorer.main()
