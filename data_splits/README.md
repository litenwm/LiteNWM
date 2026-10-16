# Frozen H4 evaluation splits

This directory is part of the repository so that evaluation indices keep the
same relative paths on another machine. It contains indices and provenance,
not RECON or SCAND image data.

## RECON

`recon/server_full_nwm_unseen_h4_100/` is the frozen 100-state H4 RECON split.
Its `navigation_eval.pkl` is the canonical runner input.

## SCAND

`scand/project_strict_test_h4_100_seed17/` is the 100-state H4 SCAND split
rebuilt from the project's pre-frozen strict test trajectories:

- 100 states from all 43 strict-test trajectories;
- context `t-3..t`, action/ground-truth horizon `t+1..t+4`;
- 25 states in each trajectory-length quartile;
- 25 states in each temporal-position quartile;
- zero record, trajectory, and route overlap with the SCAND sources used for
  v70 `train_fit` or `scorer_cal`.

The SCAND directory includes the frozen split provenance and an explicit
identity audit. It has 12 exact-state overlaps with the older external
`official_nwm_test_h4_100_balanced_seed17` benchmark, so it is a clean
train/calibration holdout for v70 but must not be described as a previously
untouched one-time cohort.

Both datasets' original images/trajectory files must be provided separately
and placed at the paths configured in `configs/local_assets.json`.
