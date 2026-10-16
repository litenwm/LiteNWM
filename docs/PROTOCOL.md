# Frozen training and validation protocol

## Temporal contract

Every sample uses four historical frames and predicts four future frames.
The target horizon is H1–H4 and the goal offset is `t+4`. H4 results must not
be compared numerically with an H8 table.

## Data isolation

The canonical split is fixed by trajectory or source bag, never by overlapping
window. The exact assignments are under `data/split_lock/`, and the scorer's
inner partitions are under `data/inner_manifests/`.

The outer validation and test partitions are sealed. Scorer fitting and
calibration use outer-train trajectories only:

```text
outer train
  |- train-fit        model fitting
  |- scorer-cal       scorer selection/calibration
  |- cem-cal          CEM hyperparameter selection
  `- internal-holdout one-time inner comparison

outer validation      sealed until the method is frozen
outer test            sealed until final reporting
```

Do not tune a checkpoint, loss, seed choice, candidate sampler, CEM parameter,
or stopping rule after reading outer validation/test results.

## Representation and world model

- External frozen backbone: DINOv3-L/16.
- Frozen project adapter: LayerNorm + linear projection + bicubic resizing,
  converting native `14 x 14 x 1024` tokens into stored
  `16 x 16 x 384` float16 tokens.
- World model: LiteTokenFormer-WM-v2-4ctx-base, 8,001,960 parameters.
- Training scope: all LTWM parameters.
- RECON:SCAND batch ratio: 3:1.
- Warm start: same-seed parent checkpoint; optimizer state reset.

## Scorer

- Architecture: trajectory-quality scorer v2, 1,484,162 parameters.
- Seeds: 1701, 1702, 1703.
- Balanced domain batches.
- No candidate-identity input and no cross-candidate model operation.
- Training targets: mean H1–H4 trajectory error surrogate and H4 progress.

## NWM-faithful H4 metrics

With reference positions `p_h`, predictions `q_h`, spacing `s`, and H=4:

```text
ATE = sqrt(mean_h ||(q_h - p_h) / s||^2)
RPE = sqrt(mean_h ||((q_{h+1}-q_h) - (p_{h+1}-p_h)) / s||^2)
```

Use `s=0.25 m` for RECON and `s=0.38 m` for SCAND. Do not prepend the
current pose, align trajectories, or correct scale. Both metrics are
lower-is-better.
