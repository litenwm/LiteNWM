# Paper and public-release correspondence

Paper: [arXiv:2610.12368v1](https://arxiv.org/abs/2610.12368v1).
Project: [litenwm.github.io](https://litenwm.github.io/).

This is a research-source and Go2 runtime release, not a claim that every
paper result can be reproduced from the downloadable runtime assets alone.

| Paper/site item | Code or documentation | Public-release boundary |
| --- | --- | --- |
| Shared visual encoding and joint H1–H4 prediction | `src/recon_latents/`, `deploy/v70_go2_orin/v70_persistent_planner_fast.py` | Frozen weights are in the companion release, not Git. |
| Incumbent-relative scoring, gate and refinement | `tools/tridomain_v70/`, `src/scoring/`, `src/planning/` | Full runtime uses the frozen three-member pipeline; the simple examples entry runs single-member scoring only. |
| Three-domain training and offline evaluation | `docs/RUNNING.md`, `docs/LATEST_VERSION.md`, `tools/tridomain_v70/`, `tools/tridomain_v71/` | Raw data, full latent caches, frozen proposal pools and the matching three-domain split lock must be supplied separately. Existing two-domain locks must not substitute for the missing three-domain lock. |
| Ablations and RTX 5090 efficiency tables | Research scripts under `tools/`; measurement definitions in paper IV-E and Appendix E | The CPU demo is not this benchmark. Reproducing tables requires the original matched cohorts, baseline assets, configurations and measurement setup. |
| MBRA transfer | Paper IV-D | A ready-to-run MBRA transfer recipe and its complete assets are not included in this release. |
| Go2 success/SPL and onboard timing | Website videos; paper IV-F and Appendix F | These are historical experimental results. The archived original execution code was subsequently modified for this public release. |
| Extended-route videos | Paper Appendix F; historical route code in `deploy/` | User-specific route images/maps are not in the runtime bundle. Start with the documented single-goal entry; do not infer automatic long-route reproduction from the videos. |

## Execution policy change

At the author's request, the public Go2 variant executes at most one H1
prefix from each fresh accepted plan. It removes old-action replay, H2/H4
execution queues, motion bridging during planning and old-action continuation
after abstention/target loss. No usable fresh plan results in a stop request.

This execution change does not mean removing the incumbent from the model's
candidate comparison: the planner still evaluates its frozen proposal set.
The planner's candidate incumbent and replaying an earlier physical action
are distinct concepts.

The runtime defaults to a CEM budget of one, configurable up to three.
The original paper's onboard configuration allowed conditional refinement
up to three rounds. Configuration differences affect performance and timing.

The modified variant has offline/simulated checks and real-weight CPU forward
execution, but no new physical navigation trials, success rate, SPL or onboard
latency measurements. Do not assign the website's original numbers to it.
See `robot/runtime_contract.json`, `robot/OPTIMIZATIONS.md` and
`robot/validation_report.json`.

## Version identities

- Paper citation: arXiv:2610.12368v1, 2026-10-08.
- Public source release: v1.0.0.
- Frozen model: sealed three-domain V70, 2026-07-29.
- Model ensemble seeds: 1701, 1702 and 1703; selector members: 0, 1 and 2.
- Asset file identities: `robot/assets_manifest.json`.
- Original model identity record: `manifests/latest_weights.json`; some old
  destination paths use `selector_final`, while public installation uses
  `selector`. Use the public asset manifest for installation paths.
- `pipeline_contract.json`: historical two-domain 2026-07-19 contract.

Historical scripts can contain original host paths and external dependencies.
The maintained user entry points are the root README, `examples/run_pipeline.py`
and `robot/`; preserving a script is not a claim of portable end-to-end replay.
