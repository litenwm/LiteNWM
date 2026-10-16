#!/usr/bin/env python3
"""Launch the NoMaD builder with legacy pure-Python dependencies appended last.

The local NWM Torch environment is GPU-compatible, while the older VisualNav
environment contains otherwise-missing packages such as
``efficientnet_pytorch``.  Appending (not prepending) its site-packages keeps
the current Torch build authoritative.
"""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path


def main() -> None:
    legacy_site = os.environ.get("NOMAD_LEGACY_SITE_PACKAGES")
    if legacy_site:
        path = Path(legacy_site).resolve(strict=True)
        if str(path) not in sys.path:
            sys.path.append(str(path))
    builder = Path(__file__).with_name("build_nomad_h4_proposals_v2.py")
    runpy.run_path(str(builder), run_name="__main__")


if __name__ == "__main__":
    main()
