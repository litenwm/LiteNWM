#!/usr/bin/env python3
"""Train v66 with route-OOF and the universal Arm-D valid contract."""

from __future__ import annotations

import numpy as np

from scoring.spacing_support_v1 import threshold_for_quantile
from tools.scoring import run_train_dualdomain_cem_tail_v66_route_oof as route_oof


implementation = route_oof.implementation
_load_domain_split = implementation.load_domain_split


def load_domain_split_universal_valid(domain, split, envelopes):
    value = _load_domain_split(domain, split, envelopes)
    feature = value["feature"]
    support = np.expm1(feature[..., -2])
    guard = feature[..., -1] > 0.5
    valid = guard & (
        support <= threshold_for_quantile(envelopes[domain], 0.999)
    )
    valid[:, 0] = True
    value["valid"] = valid
    return value


implementation.load_domain_split = load_domain_split_universal_valid


if __name__ == "__main__":
    implementation.main()
