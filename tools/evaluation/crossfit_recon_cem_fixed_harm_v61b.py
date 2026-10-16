#!/usr/bin/env python3
"""Run v61 using pre-outcome-frozen G/H identities as authority."""

from __future__ import annotations

from pathlib import Path

import numpy as np

import tools.evaluation.crossfit_recon_cem_fixed_harm_v61 as engine


_load_bank = engine.load_bank
_base_policy = engine.base_policy


def load_bank_with_identity_authority(bank: str, **kwargs):
    data, audit = _load_bank(bank, **kwargs)
    if bank in engine.GH_BANKS:
        gh_root = Path(kwargs["gh_root"])
        archive_path = gh_root / bank / "combined_archive_preoutcome_v60.npz"
        with np.load(archive_path, allow_pickle=False) as archive:
            fixed = archive["fixed_selected_index"].astype(np.int64)
            unified = archive["combined_selected_index"].astype(np.int64)
            record = archive["record_key"].astype(str)
        if not np.array_equal(record, data["record_key"]):
            raise RuntimeError(f"{bank}: frozen identity authority mismatch")
        addonly = np.where(fixed != 0, fixed, unified).astype(np.int64)
        data["frozen_fixed_override"] = fixed
        data["frozen_addonly_override"] = addonly
        audit["identity_authority"] = str(archive_path)
        audit["identity_authority_sha256"] = engine.sha256(archive_path)
        audit["fixed_identity_source"] = "pre-outcome frozen archive"
        audit["addonly_identity_source"] = (
            "deterministic v56-style composition of pre-outcome frozen fixed/unified"
        )
    return data, audit


def base_policy_with_identity_authority(data):
    policy = _base_policy(data)
    if "frozen_fixed_override" in data:
        fixed = data["frozen_fixed_override"].astype(np.int64)
        addonly = data["frozen_addonly_override"].astype(np.int64)
        policy["fixed"] = fixed
        policy["addonly"] = addonly
        policy["added"] = (fixed == 0) & (addonly != 0)
    return policy


def main() -> None:
    engine.load_bank = load_bank_with_identity_authority
    engine.base_policy = base_policy_with_identity_authority
    engine.main()


if __name__ == "__main__":
    main()
