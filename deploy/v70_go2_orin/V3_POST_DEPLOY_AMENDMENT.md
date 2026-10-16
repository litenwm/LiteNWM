# V70 Go2 Orin V3 post-deploy amendment

V3 is a copy-on-write amendment of the verified
`dinov3_ltwm_v70_tridomain_go2_orin_20260729_v2` bundle. The trained LTWM,
selector, DINO adapter, and NoMaD checkpoint weights are unchanged.

## Safety state

All validation for this amendment must keep:

- `/tmp/go2_nav_stop` present;
- `/run/go2-safety/ARMED` absent;
- every bridge in its default `dry_run=true` mode;
- navigation, motion-executor, and Unitree SportClient commands stopped.

## Changes

1. `safe_bridge_v5.py` estimates the current ground height from each point
   cloud, filters floor and bounded Go2 front-foot self returns, and applies
   independent forward/rotation clearance gates. Invalid or insufficient
   ground evidence fails closed.
2. `v70_goal_stop.py` adds a terminal-state supervisor. A final topological
   node, NoMaD temporal distance, DINO cosine distance, fresh synchronized
   evidence, hysteresis, and consecutive confirmation are required. The first
   goal-candidate frame immediately requests zero velocity; successful stop is
   latched until an explicit reset.
3. NoMaD snapshot inference now records the checkpoint distance-head output
   with immutable observation/goal hashes.
4. DINO shadow inference records current-to-goal NMSE, cosine distance, and
   norm ratio as diagnostic evidence.
5. Board latency instrumentation records exact synchronized CUDA stage
   timings for DINO, NoMaD, LTWM, selector, CEM, and the complete process-level
   planning cycle.
6. The coherent-turn-tail action-support correction remains in force. No
   newly generated candidate may exceed the V70 training support contract.
7. The latency driver preserves the virtual-environment Python entry path.
   It must not resolve the venv executable symlink to `/usr/bin/python3.10`,
   which would bypass the deployed V70 environment.
8. Archive-CEM verifies and loads the three LTWM/scorer pairs and three Arm-D
   members once per process and reuses them across all three rounds. On the
   same Orin snapshot, all non-timing arrays were bit-identical to the
   load-per-round implementation while CEM planning fell from 9599.304 ms to
   4829.141 ms. This is an inference-runtime optimization only: population,
   iterations, random seed, weights, support constraints, calibration, and
   selection logic are unchanged.

## Posture caveat

Latency can be measured while the Go2 is lying down because no command is
issued. Visual goal-arrival and action-quality conclusions must not mix a
lying observation with a standing goal image: camera height and pitch create
an out-of-distribution appearance change. A formal navigation shadow result
therefore requires a standing, stationary, stabilized four-frame observation.

## Deployment status

The V3 bundle is shadow-only. `v70_goal_stop.py` is a pure decision module and
`safe_bridge_v5.py` defaults to zero output. Neither file by itself constitutes
authorization to start a live closed-loop navigation process.
