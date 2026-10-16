#!/usr/bin/env python3
"""Run dual-latent LTWM training with stable zero-motion cosine handling."""

from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Importing the stable loss replaces only degenerate zero-vector cosine terms
# before the trainer imports its public loss function.
import recon_latents.losses_highres_dual_stable_v1  # noqa: E402,F401
from train_highres_multilayer_dual_ltwm_v1 import main  # noqa: E402


if __name__ == "__main__":
    main()
