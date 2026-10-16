#!/usr/bin/env python3
"""Run v66 with route-OOF and the frozen universal Arm-D valid contract.

The universal model was calibrated with spacing-envelope plus physical validity.
The legacy frozen scorer-v2 ``majority_support`` flag is provenance/audit data
and is deliberately excluded from the primary universal contract.
"""

from __future__ import annotations

import numpy as np

from tools.scoring import run_dualdomain_cem_tail_v66_route_oof as route_oof


implementation = route_oof.implementation


def score_population_universal_valid(
    data,
    *,
    domain,
    split,
    training,
    envelope,
    fold_id,
    device,
):
    spacing = 0.25 if domain == "recon" else 0.38
    feature, support, guard = implementation.build_universal_features(
        data,
        spacing_m=spacing,
        support_envelope=envelope,
    )
    valid = guard & (
        support <= implementation.threshold_for_quantile(envelope, 0.999)
    )
    valid[:, 0] = True
    domain_id = np.full(
        len(feature), 0 if domain == "recon" else 1, dtype=np.int64
    )
    prediction = (
        implementation.oof_predict(
            training, feature, domain_id, fold_id, device
        )
        if split == "train_fit"
        else implementation.ensemble_predict(
            training["arms"]["D"]["checkpoints"]["full"],
            feature,
            domain_id,
            device=device,
        )
    )
    policy = training["arms"]["D"]["calibration"]["frontier"]["domains"][
        domain
    ]["policy"]
    temperature = training["arms"]["D"]["temperature"]["domains"][domain]
    energy, p_dual, p_harm = implementation.conservative_energy(
        prediction,
        policy=policy,
        dual_temperature=float(temperature["deployment_dual_temperature"]),
        harm_temperature=float(temperature["deployment_harm_temperature"]),
        valid=valid,
    )
    safe = implementation.candidate_safety_mask(
        prediction, p_dual, p_harm, valid, policy
    )
    return {
        **prediction,
        "energy": energy,
        "p_dual": p_dual,
        "p_harm": p_harm,
        "valid": valid,
        "safe": safe,
    }


implementation.score_population = score_population_universal_valid


if __name__ == "__main__":
    implementation.main()
