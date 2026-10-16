# Frozen v70 training/calibration identities

Use this directory before constructing any new RECON or SCAND test cohort.

`v70_training_identities.jsonl` contains the label-free identity of every H4
window used by the final Arm-D training or calibration sources:

| Domain | Partition | Windows | Trajectories | Route groups |
|---|---:|---:|---:|---:|
| RECON | train-fit | 5,292 | 3,492 | 1,764 |
| RECON | scorer-cal | 410 | 365 | 206 |
| SCAND | train-fit | 7,489 | 138 | 29 |
| SCAND | scorer-cal | 1,213 | 20 | 4 |

Each identity records `record_key`, `trajectory_id`, `current_t`,
`route_group`, `bag_id`, and `window_uid`. No ATE/RPE/endpoint/objective label
or reserve identity is included.

## Required test-selection rule

1. Select trajectories only from the corresponding frozen `test` array under
   `strict_splits/`.
2. Require zero `trajectory_id` overlap with
   `v70_training_identities.jsonl`.
3. For the strongest route-generalization claim, also require zero
   `route_group` overlap.
4. A different time window from a training trajectory is not an independent
   test and must be rejected.

The exact source manifests are retained under `source_manifests/` for
reproducibility. Their absolute image paths are historical provenance; replace
dataset roots through `configs/local_assets.json` rather than editing identity
fields.

## Data that was evaluated but not trained on

RECON A-H, fresh SCAND-100, and the one-time reserve were evaluation-only and
did not train or calibrate v70. They nevertheless cannot be presented again as
previously untouched confirmation cohorts.

## External pretraining caveat

The DINOv3 backbone was pretrained upstream on LVD-1689M; this project cannot
enumerate all of those source images. The upstream NoMaD configuration enables
RECON training and leaves SCAND disabled. Consequently, use the official
NoMaD-unseen RECON test split when claiming unseen RECON performance.
