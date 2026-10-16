# Checkpoint policy

Only project-produced checkpoints are tracked.

## Adapter

`checkpoints/project/dinov3_adapter/adapter_pilot_v1_final.pt` is the frozen
1024-to-384 token adapter used by every included LTWM run. The upstream DINOv3
pretrained weight is intentionally external.

## LTWM

For each seed:

- `ltwm_parent/seed*/best.pt` is the same-seed warm-start lineage checkpoint.
- `ltwm_expansion/seed*/best.pt` is the validation-selected expanded
  RECON+SCAND checkpoint and the default for evaluation.
- `ltwm_expansion/seed*/final.pt` and `epoch_001.pt` are retained for audit and
  exact training-state reconstruction; they are not the default deployment
  choice.

## Scorer v2

For each seed, `scorer_v2/seed*/best.pt` is the selected checkpoint.
`summary.json` preserves its selected-epoch training record. Historical
per-epoch comparison logs are intentionally outside this mainline-only
repository.

The authoritative hashes are stored in `manifests/checkpoints.json`. Run
`python tools/release/verify_repository.py` to verify every tracked file and
to reject forbidden third-party weight paths.
