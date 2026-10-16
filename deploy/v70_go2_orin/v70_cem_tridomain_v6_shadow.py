#!/usr/bin/env python3
"""Archive-CEM shadow with latest support and checkpoint-root compatibility."""

from __future__ import annotations

import sys
from pathlib import Path

import v70_ltwm_fixed_pool_shadow as fixed_pool

from v70_action_support_compat import support_stats

# v4 imports the helper from fixed_pool, so patch it before importing v4.
fixed_pool.support_stats = support_stats

import v70_cem_tridomain_v4_shadow as implementation  # noqa: E402

implementation.support_stats = support_stats


if __name__ == "__main__":
    if "--checkpoint-root" not in sys.argv:
        release_root = Path(__file__).resolve().parents[2]
        sys.argv.extend(
            [
                "--checkpoint-root",
                str(release_root / "checkpoints/project"),
            ]
        )
    implementation.main()
