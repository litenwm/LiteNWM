#!/usr/bin/env python3
"""Freeze three unused bags with the same external-camera topic contract."""

from pathlib import Path

import run_rotation_transport_external_bag_validation_v1 as pilot


# Replaced before any compatible-bag image or latent was read.  The first
# preregistered set failed closed because two bags lacked /image_raw/compressed.
pilot.EXTERNAL_BAGS = (
    pilot.ExternalBag(
        "external_A_Spot_EERC_Ahg_Thu_Nov_11_80",
        Path("/mnt/d/VEDIOS/A_Spot_EERC_Ahg_Thu_Nov_11_80.bag"),
    ),
    pilot.ExternalBag(
        "external_B_Spot_GDC_AHG_Mon_Nov_15_116",
        Path("/mnt/d/VEDIOS/B_Spot_GDC_AHG_Mon_Nov_15_116.bag"),
    ),
    pilot.ExternalBag(
        "external_B_Spot_SAC_SAC_Mon_Nov_15_109",
        Path("/mnt/d/VEDIOS/B_Spot_SAC_SAC_Mon_Nov_15_109.bag"),
    ),
)

# Make the result's runner path/hash identify this frozen route entry point.
pilot.__file__ = __file__


if __name__ == "__main__":
    pilot.main()
