#!/usr/bin/env python3
"""Mechanical JSON-scalar entry for the paired action-residual evaluation."""

from pathlib import Path
import json
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "tools/recon")]

from tools.evaluation import evaluate_ltwm_action_residual_v1 as evaluation  # noqa: E402


evaluation.false = False
_ORIGINAL_DUMPS = json.dumps


def _dumps(value, *args, **kwargs):
    kwargs.setdefault(
        "default",
        lambda item: item.item() if isinstance(item, np.generic) else str(item),
    )
    return _ORIGINAL_DUMPS(value, *args, **kwargs)


evaluation.json.dumps = _dumps


if __name__ == "__main__":
    evaluation.main()
