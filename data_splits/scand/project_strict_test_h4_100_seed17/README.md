# SCAND strict-test H4-100 (seed 17)

This benchmark freezes 100 H4 windows for a paired comparison of trajectory
scorers. It uses only the 43 trajectories in the project SCAND strict outer-test
split, covers every one of those trajectories, and has zero trajectory overlap
with the project train and validation splits.

## Model-independent selection

- The selection uses seed 17 and SHA-256 ordering only. No model prediction,
  score, metric, checkpoint, proposal, or label is read during selection.
- Every strict-test trajectory contributes two windows; 14 deterministically
  selected trajectories contribute a third window.
- Each trajectory-length quartile contributes exactly 25 windows.
- Each temporal-position quartile contributes exactly 25 windows.
- Every case uses context `t-3..t`, future ground truth `t+1..t+4`, and goal
  frame `t+4`.

The local SCAND copy contains 5 trajectories whose usable local frame
count is shorter than the historical canonical count. Selection and range checks
therefore use the contiguous local JPEG/position/yaw intersection; both counts
remain recorded per case for auditability.

## Fair scorer comparison

Use `navigation_eval.pkl` unchanged for both scorers. To isolate scorer quality,
freeze one candidate/proposal pool per `eval_index` and give the exact same pool,
ordering, labels, normalization, and candidate mask to both scorers. Do not tune
thresholds, temperatures, or weights on this outer-test set. Report paired
per-case deltas and a trajectory-clustered confidence interval because multiple
windows can come from one trajectory.

If a scorer trained, calibrated, selected a checkpoint, or tuned hyperparameters
on any of these 43 trajectories, this benchmark is not strict for that scorer.
Audit both scorers' training provenance before making a strict comparison claim.

`navigation_eval.pkl` is the canonical runner input. `manifest.json` and
`manifest.csv` preserve selection provenance, and `SHA256SUMS.txt` freezes the
package bytes.
