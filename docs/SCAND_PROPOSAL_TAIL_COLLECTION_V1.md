# SCAND proposal-tail collection v1

This toolchain is collection infrastructure, not a new model or performance result. The formal v70 planner remains the action source. The sidecar is opt-in and defaults to off.

## Frozen behavior boundary

- The formal v70 checkpoint, three CEM iterations, population 16, elite count 4, proposal scale 0.7, physical guard, support q0.999, safety policy, scorer energy, archive, and fallback are unchanged.
- The raw-draw observer receives copies only, cannot return a value to the planner, and is checked not to change the NumPy generator state.
- Sidecar files are flushed only after `selected_actions.npz` and the formal prelabel freeze have been written.
- A path containing the forbidden final-cohort token is rejected before opening it.

The frozen preregistration is `configs/proposal_tail_collection/scand_proposal_tail_collection_v1_preregistration.json`; the output schema is `configs/proposal_tail_collection/scand_proposal_tail_collection_v1.schema.json`.

## Required future manifest fields

Every future state must provide route, collection day, bag, trajectory, window UID, history/current/goal frame indices, raw timestamps, and camera/image metadata. Camera metadata must include width, height, encoding, file paths, file sizes, and file SHA-256; JPEG quality may be null only with an explicit reason. Missing legacy fields are serialized as `null + unavailable_reason`, but a development or one-time-validation cohort fails validation until the required contract is complete.

All 65 candidate slots are retained: fixed K17 plus the 16 challengers from each of three CEM rounds. Live collection additionally records every unclipped draw, the clipped draw, coordinate-level physical margins, support score/margin, acceptance and draw-to-final-slot mapping. Runtime dedup is explicitly false in the current v70 path; the sidecar records source indices so offline dedup remains auditable.

The frozen feature contract exposes three-seed `ate_by_seed`, `rtp_by_seed`, and `support_by_seed`. It does not expose seed-specific metric RPE, endpoint, objective, or member-specific logits. Those fields are never fabricated and remain explicit unavailable fields until a future upstream contract produces them without changing decisions.

## Cohort isolation

1. Build the hash-only registry of all already-used SCAND manifests with `build_proposal_tail_identity_registry_v1.py`.
2. Obtain a complete hash-only forbidden-identity registry from the reserve custodian. Do not open or enumerate reserve content in this toolchain.
3. Freeze route/day/bag splits before labels. Development and one-time validation may not share route or trajectory hashes.
4. The one-time validation cohort cannot be used for training, calibration, threshold selection, or representation selection. Opening its labels creates a marker and a second opening is refused.

## Live opt-in example

Run the normal v70 archive replay with its existing arguments plus:

```text
--proposal-tail-sidecar-dir outputs/proposal_tail_collection_v1/<cohort>
--proposal-tail-cohort-role development
```

Omitting both options is the formal default-off path. The sidecar directory must be outside the formal replay output directory.

Validate before labels:

```text
python tools/planning/validate_proposal_tail_dataset_v1.py \
  --collection outputs/proposal_tail_collection_v1/<cohort> \
  --used-identity-registry outputs/proposal_tail_collection_v1/used_v70_v77_identities.json \
  --reserve-identity-registry <custodian-provided-hash-only-registry>
```

Build labels offline only after identity validation:

```text
python tools/planning/build_proposal_tail_labels_v1.py \
  --collection outputs/proposal_tail_collection_v1/<cohort> \
  --truth <non-reserve-truth.npz> \
  --output outputs/proposal_tail_collection_v1/<cohort>/offline_labels.npz
```

The label builder recomputes ATE, RPE, endpoint, and `objective=0.4*ATE+0.6*endpoint` in meters. It is not imported by the online runner.

## Smoke limitation

`dry_run_proposal_tail_sidecar_v1.py` may serialize exactly one already-spent SCAND window. Historical rejected draws and RNG state were never cached, so that dry-run marks them unobservable. It checks hashes and alignment only; it is forbidden as training or performance evidence.


## Public-index identity quarantine

The public SCAND CSV can be audited without downloading any bag. Run
`build_official_scand_identity_inventory_v1.py` against the frozen train
trajectory list and the v70-v77 used-identity registry. Route identity is
derived from canonical `FileName` tokens, never from the display-oriented
start/end columns. Rows that clear the known-used hashes remain quarantined
until the custodian supplies a complete hash-only forbidden registry; they are
not automatically independent routes.

## Temporal pair features

After a sidecar has passed hash and ordering validation,
`build_proposal_tail_temporal_pair_v1.py` creates label-free features relative
to the formal v70 archive choice for the same state. It includes H1-H4 action
deltas, H1-H4 latent-summary deltas, and three-seed ATE/RTP/support deltas.
The 112 LTWM scalars are reshaped using their frozen contract as
`[mean/std/min/max, seven latent blocks, H1-H4]`.

The frozen artifact does not retain consistent LTWM seed identity for the 28
temporal scalars. Per-feature min/middle/max can be reconstructed, but must not
be described as three seed trajectories. The spent one-state serialization
smoke is infrastructure-only and has `training_allowed=false`.


## Selective official-bag download and prelabel processing

Use `tools/planning/download_prepare_new_scand_bags_v1.py`. The command is
plan-only unless `--execute` is supplied. A complete custodian-produced
hash-only forbidden registry is mandatory; the template is deliberately
rejected. Identity filtering uses the public-index `FileName` tokens, the
v70-v77 used registry, and the custodian registry before network access.

The frozen camera contract is 4 Hz, four history frames, H4 goal offset,
center crop to 4:3, bicubic 320x240 output, and JPEG quality 95. Original
resolution and compressed format are retained in `route_metadata.json`; the
source bag and every processed image receive SHA-256 provenance. The prelabel
stage reads only a supported compressed camera topic and explicitly does not
open odometry, truth, outcomes, or labels.

Plan without network access:

```text
/home/rainbow/miniconda3/envs/nwm/bin/python \
  tools/planning/download_prepare_new_scand_bags_v1.py \
  --forbidden-registry data/proposal_tail_collection_v1/incoming/forbidden_identity_hashes.json
```

After reviewing the frozen plan, execute the same command with `--execute`.
Downloads resume through `.part` files, enforce the configured total-size and
free-disk limits, and stay under
`data/proposal_tail_collection_v1/incoming/quarantine/official_scand_selective_v1/`.
Only after final readiness passes are development and one-time-validation
manifests published. Validation labels remain closed.
