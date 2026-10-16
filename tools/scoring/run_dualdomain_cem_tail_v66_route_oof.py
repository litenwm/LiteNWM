#!/usr/bin/env python3
"""Run v66 harvest with the original Arm-D route-OOF fold identity.

The frozen v66 harvester used ``family`` when selecting an OOF checkpoint.  The
base Arm-D model was trained with RECON ``route_group`` and SCAND
``route_group::bag::bag_id`` identities.  This label-free wrapper restores that
exact contract without changing the frozen CEM/search protocol.
"""

from __future__ import annotations

import numpy as np

from tools.scoring import harvest_dualdomain_cem_tail_v66 as implementation


_load_npz = implementation.load_npz


def load_npz_with_route_oof(path):
    value = _load_npz(path)
    if "route_group" not in value:
        return value
    route = value["route_group"].astype(str)
    if "bag_id" in value:
        route = np.char.add(
            np.char.add(route, "::bag::"), value["bag_id"].astype(str)
        )
    value["family"] = route
    return value


implementation.load_npz = load_npz_with_route_oof


if __name__ == "__main__":
    implementation.main()
