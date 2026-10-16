#!/usr/bin/env python3
"""Twenty-bank development spatial-candidate LOBO learning-curve endpoint."""

from __future__ import annotations

from pathlib import Path


def main() -> None:
    target = Path(__file__).with_name("train_recon_target_candidate_rank_v37.py")
    source = target.read_text(encoding="utf-8")
    compact_old = (
        'x,valid=compact_base_feature(feature,score,'
        'np.empty((100,0),np.float32))'
    )
    compact_new = compact_old + (
        '\n with np.load(source.parent/"spatial_rollout_v17.npz",'
        'allow_pickle=False) as rp: '
        'x=np.concatenate((x,rp["spatial_rollout_features"]'
        '[:,1:].astype(np.float32)),axis=2)'
    )
    parser_old = 'p.add_argument("--bank-h",type=Path,required=True);'
    parser_new = parser_old + (
        'p.add_argument("--bank-ijk-root",type=Path,required=True);'
        'p.add_argument("--bank-lp-root",type=Path,required=True);'
        'p.add_argument("--bank-qt-root",type=Path,required=True);'
    )
    append_old = (
        'banks.append(load("bank_h",a.bank_h/'
        '"features/recon_target_cem_confirmation_v35_features.npz",'
        'a.bank_h/"universal_scores.npz",'
        'a.bank_h/"source_labeled.npz"));names='
    )
    append_new = append_old[:-6] + (
        '\n for l in "ijk":\n'
        '  r=a.bank_ijk_root/f"bank_{l}";banks.append(load('
        'f"bank_{l}",r/"features"/'
        'f"recon_target_bank_{l}_v40_features.npz",'
        'r/"universal_scores.npz",r/"source_labeled.npz"))\n'
        ' for l in "lmnop":\n'
        '  r=a.bank_lp_root/f"bank_{l}";banks.append(load('
        'f"bank_{l}",r/"features"/'
        'f"recon_target_bank_{l}_v44_features.npz",'
        'r/"universal_scores.npz",r/"source_labeled.npz"))\n'
        ' for l in "qrst":\n'
        '  r=a.bank_qt_root/f"bank_{l}";banks.append(load('
        'f"bank_{l}",r/"features"/'
        'f"recon_target_bank_{l}_v46_features.npz",'
        'r/"universal_scores.npz",r/"source_labeled.npz"))\n'
        ' names='
    )
    variants_old = (
        '(("candidate_universal_support",slice(474,486)),'
        '("candidate_motion_universal_support",slice(434,486)))'
    )
    variants_new = (
        '(("candidate_universal_support",slice(474,486)),'
        '("candidate_motion_universal_support_spatial",slice(434,744)))'
    )
    for old, new in (
        (compact_old, compact_new), (parser_old, parser_new),
        (append_old, append_new), (variants_old, variants_new),
    ):
        if source.count(old) != 1:
            raise RuntimeError(f"v47 source contract changed: {old[:60]}")
        source = source.replace(old, new, 1)
    source = source.replace(
        'preds["candidate_motion_universal_support"]',
        'preds["candidate_motion_universal_support_spatial"]',
    ).replace(
        'reshape(800,16)', 'reshape(len(banks)*100,16)'
    ).replace(
        'np.arange(800)', 'np.arange(len(banks)*100)'
    ).replace(
        'recon_target_candidate_rank_v37_member',
        'recon_target_candidate_spatial_v47_member',
    ).replace(
        'recon_target_candidate_rank_v37',
        'recon_target_candidate_spatial_v47',
    ).replace(
        'eight_bank_candidate_oof_complete',
        'twenty_bank_candidate_spatial_oof_learning_curve_complete',
    )
    exec(
        compile(source, str(target), "exec"),
        {"__name__": "__main__", "__file__": str(target)},
    )


if __name__ == "__main__":
    main()
