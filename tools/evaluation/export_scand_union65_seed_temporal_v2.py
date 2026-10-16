#!/usr/bin/env python3
"""Run the v1 exact-seed exporter with a fresh autocast context per forward.

PyTorch autocast context objects are not reusable across repeated ``with``
entries.  The v1 exporter reused one object within each state batch, causing
all but the first forward to execute under a different numeric contract and
correctly fail the frozen-summary reproduction check.  This wrapper changes
only that context-manager behavior.
"""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from tools.evaluation import export_scand_union65_seed_temporal_v1 as base  # noqa: E402


_original_autocast = base.torch.autocast


class _FreshAutocast:
    def __init__(self, *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs
        self.context = None

    def __enter__(self):
        self.context = _original_autocast(*self.args, **self.kwargs)
        return self.context.__enter__()

    def __exit__(self, exc_type, exc_value, traceback):
        assert self.context is not None
        return self.context.__exit__(exc_type, exc_value, traceback)


def fresh_autocast(*args, **kwargs):
    return _FreshAutocast(*args, **kwargs)


def main() -> None:
    base.torch.autocast = fresh_autocast
    base.main()


if __name__ == "__main__":
    main()
