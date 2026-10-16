# DINOv3 dense-meta three-arm comparison

This experiment retrains the most valuable fixed-pool part of the 019f70d5
planner/scorer protocol on current DINOv3/LTWM features. No old learned weight
is migrated, and CEM remains disabled.

## Feature contracts

- `dinov3_full_meta_raw` is the primary, independently retrained model. It uses
  action/past-motion features, DINOv3 static context, and 112 raw three-seed
  LTWM rollout/epistemic features. It does not use scorer-v2 ATE/RTP outputs.
- `dinov3_full_meta_v2_stacked` is an ablation. It adds 10 frozen scorer-v2
  ATE/RTP/ensemble-uncertainty features to the raw contract.
- `dinov3_scorer_v2_native` is the frozen three-seed greedy scorer-v2 readout.
- The one-dimensional support summary is stored separately and is never a
  learned feature. The deterministic majority-support mask is used only to
  reject unsupported candidates.

The previously exported 123-dimensional mixed block was mechanically split
without rerunning DINOv3: `112 raw + 10 scorer-v2 auxiliary + 1 support`.

## Leakage control

The scorer-v2 train-fit manifest contains 5,292 windows from 3,492
trajectories. Dense-train has 25 exact-window and 674 trajectory overlaps,
spanning 86 of 192 families. Therefore:

- the raw branch trains on all 1,512 states / 192 families;
- the stacked branch excludes every family containing any scorer-v2 train-fit
  trajectory and trains on 824 states / 106 families;
- dense-cal, test3, and RECON100 have zero scorer-v2 train trajectory overlap.

Both learned branches use the same five family-hash OOF folds, frozen
dense-cal gate, K17 pool, metrics, and test-label ordering.

## Results

### Dense-cal (719 states)

| Arm | Promotions | Dual precision | Wilson lower | ATE delta/state | Objective delta/state |
| --- | ---: | ---: | ---: | ---: | ---: |
| full/meta-raw | 447 | 86.58% | 83.10% | -0.01485 m | -0.02026 m |
| full/meta-v2-stacked | 426 | 86.85% | 83.31% | -0.01348 m | -0.01816 m |
| native scorer-v2 | 685 | 65.40% | 61.18% | -0.01380 m | -0.02015 m |

### Frozen test3 replay (256 family-disjoint states)

| Arm | Promotions | Dual precision | Wilson lower | ATE delta/state | Objective delta/state |
| --- | ---: | ---: | ---: | ---: | ---: |
| full/meta-raw | 158 | 89.87% | 84.18% | -0.01439 m | -0.02012 m |
| full/meta-v2-stacked | 152 | 90.79% | 85.13% | -0.01417 m | -0.01952 m |
| native scorer-v2 | 246 | 65.04% | 58.89% | -0.01036 m | -0.01526 m |

The old 019f70d5 frozen meta ridge on the same test3 pool made 171 promotions,
with 88.30% dual precision, 82.63% Wilson lower, -0.01501 m ATE/state, and
-0.02062 m objective/state. The DINOv3 raw port is safer but slightly less
aggressive; the stacked ablation has the highest point precision but lower net
gain than raw.

### Frozen RECON100 K17 replay

The original NoMaD x16 pool was augmented with the exact trimmed-mean75
incumbent used for training. All selections were written before the old labels
were opened.

| Arm | Promotions | Dual precision | Wilson lower | ATE delta/state | Objective delta/state |
| --- | ---: | ---: | ---: | ---: | ---: |
| full/meta-raw | 68 | 97.06% | 89.90% | -0.01900 m | -0.02561 m |
| full/meta-v2-stacked | 64 | 92.19% | 82.98% | -0.01738 m | -0.02285 m |
| native scorer-v2 | 98 | 62.24% | 52.34% | -0.01271 m | -0.01780 m |

The frozen raw-head ablation on the same RECON100 pool was:

| Raw route | Promotions | Dual precision | ATE delta/state | Objective delta/state |
| --- | ---: | ---: | ---: | ---: |
| action | 54 | 94.44% | -0.01022 m | -0.01375 m |
| static | 59 | 93.22% | -0.01341 m | -0.01778 m |
| latent | 56 | 98.21% | -0.01463 m | -0.01976 m |
| full | 67 | 98.51% | -0.01811 m | -0.02458 m |
| meta | 68 | 97.06% | -0.01900 m | -0.02561 m |

Thus raw LTWM features are not required to obtain a conservative selector, but
they materially raise the fixed-pool gain ceiling. Full/meta is the useful
combination rather than a latent-only replacement for action/static context.

RECON test3 and RECON100 outcomes were already spent by prior studies, so these
are frozen-policy developmental replays, not fresh deployment evidence. The
fixed-pool evidence makes raw dense-meta the primary candidate. Stacking does
not raise the RECON100 ceiling, and CEM requires a separate frozen proposal,
support, and recalibration experiment; it was not entered here.
