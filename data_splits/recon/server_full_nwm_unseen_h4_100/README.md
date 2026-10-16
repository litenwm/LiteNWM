# Fixed RECON H4 NWM-unseen benchmark (100 cases)

This benchmark contains **100 windows from 100 distinct trajectories**. Every
trajectory belongs to both the server full-RECON test split and the official
NWM RECON test split. All 100 trajectories therefore remain unseen by the
official NWM checkpoint training split.

## Sampling

- Source intersection: 227 trajectories.
- H4-valid population (`num_frames >= 8`): 206 trajectories.
- Four trajectory-length quartiles: 25 selected from each quartile.
- Four temporal-position bins: 25 cases in each bin.
- Each length-by-time cell contains 6 or 7 cases.
- One window per trajectory; selection seed: `17`.
- Stable selection key: `SHA256(seed|namespace|trajectory_id)`.
- No model prediction, metric, B41 label, or planner outcome was used.

For each case, `curr_time=t`, context frames are `t-3..t`, the goal is `t+4`,
and ground-truth future frames are `t+1..t+4`. The pickle entries are
`(trajectory_id, curr_time, 4, 4)`.

## Interpretation

This is a **fixed custom H4 subset**, not the original NWM paper's published
100-case navigation index. It is suitable for a matched, reasonably fast
comparison of DWM-H4, NoMaD-H4, NWM-S-CEM-H4, and NWM-S+NoMaD-H4.

The split proves NWM training holdout. The exact training-overlap status of the
published NoMaD checkpoint is unknown because its checkpoint-specific original
trajectory list is unavailable; report that limitation explicitly.

Use `manifest.csv` for inspection, `manifest.json` for full provenance, and
`SHA256SUMS.txt` to freeze the exact benchmark files. For planner execution,
also set `--trajectory_eval_horizon 4` and use a deterministic per-evaluation
candidate seed when the runner supports it.
