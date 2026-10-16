#!/usr/bin/env python3
"""Run v47 with candidate-level NoMaD diffusion diagnostics appended."""

from __future__ import annotations

from pathlib import Path


def main() -> None:
    target = Path(__file__).with_name(
        "run_train_recon_target_candidate_spatial_v47.py"
    )
    source = target.read_text(encoding="utf-8")
    old = '''        '[:,1:].astype(np.float32)),axis=2)'
    )'''
    new = '''        '[:,1:].astype(np.float32)),axis=2)'
        '\\n with np.load(source.parent/"diffusion_v48b.npz",'
        'allow_pickle=False) as dp: '
        'x=np.concatenate((x,dp["candidate_diffusion_features"]'
        '.astype(np.float32)),axis=2)'
    )'''
    if source.count(old) != 1:
        raise RuntimeError("v49 compact feature source contract changed")
    source = source.replace(old, new, 1)
    if source.count("slice(434,744)") != 1:
        raise RuntimeError("v49 feature slice source contract changed")
    source = source.replace("slice(434,744)", "slice(434,834)", 1)
    source = source.replace(
        "recon_target_candidate_spatial_v47_member",
        "recon_target_candidate_diffusion_v49_member",
    ).replace(
        "recon_target_candidate_spatial_v47",
        "recon_target_candidate_diffusion_v49",
    ).replace(
        "twenty_bank_candidate_spatial_oof_learning_curve_complete",
        "twenty_bank_candidate_diffusion_oof_development_screen_complete",
    )
    exec(
        compile(source, str(target), "exec"),
        {"__name__": "__main__", "__file__": str(target)},
    )


if __name__ == "__main__":
    main()
