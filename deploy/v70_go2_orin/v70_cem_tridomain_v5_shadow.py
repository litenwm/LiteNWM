#!/usr/bin/env python3
"""Archive-CEM shadow runner with latest tri-domain support compatibility."""

from __future__ import annotations

import v70_ltwm_fixed_pool_shadow as fixed_pool

from v70_action_support_compat import support_stats

# v4 imports the helper from fixed_pool, so patch it before importing v4.
fixed_pool.support_stats = support_stats

import v70_cem_tridomain_v4_shadow as implementation  # noqa: E402

implementation.support_stats = support_stats


if __name__ == "__main__":
    implementation.main()
