# DINOv3 safe-scorer RECON-100 comparison

This experiment reuses the frozen `strict_external100` NoMaD x16 H4 pool and
the old `cal32` split. It migrates the 019f70d5 relative paired-ensemble and
conformal promotion protocol, but does not migrate any old scorer weight.

## Result

| Selector | Promotions | Dual precision | Legacy RMS ATE delta/state | Legacy objective delta/state |
| --- | ---: | ---: | ---: | ---: |
| 019f70d5 old full, frozen | 18 | 94.44% | -0.01074 m | -0.01461 m |
| DINOv3 current seed1701 greedy | 94 | 67.02% | -0.01743 m | -0.02515 m |
| DINOv3 current 3-seed greedy | 96 | 65.62% | -0.01679 m | -0.02448 m |
| DINOv3 + adapted 95% safe gate | 0 | n/a | 0.00000 m | 0.00000 m |

The DINOv3 greedy readout has greater aggregate gain but much lower promotion
precision. The old full scorer is substantially stronger as a safe promotion
policy. The direct protocol adaptation is currently too conservative because
cal32 produces conformal corrections of 0.08221 m for mean ATE gain and
0.14513 for RTP gain.

All arms select only from the same frozen 100 x 16 pool. CEM is disabled:
zero iterations and zero generated candidates. These are RECON development
results, not deployment or DINOv3 outer-test evidence.

The machine-readable policy and complete result are written under
`outputs/recon100_dinov3_safe_scorer_v1/`.
