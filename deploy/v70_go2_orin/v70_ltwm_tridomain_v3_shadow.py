#!/usr/bin/env python3
"""V70 LTWM shadow runner with latest tri-domain support compatibility."""

from __future__ import annotations

import v70_ltwm_tridomain_v2_shadow as implementation

from v70_action_support_compat import support_stats

# The implementation resolves this global when evaluating each scorer member.
implementation.support_stats = support_stats


if __name__ == "__main__":
    implementation.main()
