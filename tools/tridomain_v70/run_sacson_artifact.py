#!/usr/bin/env python3
"""Compatibility entry point for the generic v70 SACSoN artifact tools."""

from __future__ import annotations

from pathlib import Path
import importlib
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


MODULES = {
    "proposals": "tools.planning.build_nomad_h4_proposals_v2",
    "labels": "tools.scoring.build_trajectory_quality_labels_v2",
    "cache": "tools.scoring.cache_trajectory_quality_features_v2",
}


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in MODULES:
        raise SystemExit(f"usage: {sys.argv[0]} MODE ... where MODE is {sorted(MODULES)}")
    module = importlib.import_module(MODULES[sys.argv[1]])
    sys.argv[:] = [sys.argv[0], *sys.argv[2:]]
    original_parse_args = module.parse_args

    def parse_args():
        original_argv = list(sys.argv)
        try:
            for index, value in enumerate(sys.argv[:-1]):
                if value == "--dataset" and sys.argv[index + 1] == "sacson":
                    sys.argv[index + 1] = "scand"
            args = original_parse_args()
        finally:
            sys.argv[:] = original_argv
        args.dataset = "sacson"
        return args

    module.parse_args = parse_args
    module.main()


if __name__ == "__main__":
    main()
