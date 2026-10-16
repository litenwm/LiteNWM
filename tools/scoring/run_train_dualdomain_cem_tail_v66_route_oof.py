#!/usr/bin/env python3
"""Train v66 while preserving the original Arm-D route-OOF identity."""

from __future__ import annotations

import numpy as np

from scoring.universal_dualdomain_raw_v2 import group_fold
from tools.evaluation.export_recon_dense_dinov3_features_v1 import load_npz
from tools.scoring import train_dualdomain_cem_tail_v66 as implementation
from tools.scoring.harvest_dualdomain_cem_tail_v66 import paths


_load_domain_split = implementation.load_domain_split


def load_domain_split_with_route_oof(domain, split, envelopes):
    value = _load_domain_split(domain, split, envelopes)
    source = load_npz(paths(domain, split)["source"])
    route = source["route_group"].astype(str)
    if domain == "scand":
        route = np.char.add(
            np.char.add(route, "::bag::"), source["bag_id"].astype(str)
        )
    fold = np.asarray(
        [group_fold(domain, identity, 5) for identity in route],
        dtype=np.int64,
    )
    value["group"] = np.tile(route, 4)
    value["fold_id"] = np.tile(fold, 4)
    return value


implementation.load_domain_split = load_domain_split_with_route_oof


if __name__ == "__main__":
    implementation.main()
