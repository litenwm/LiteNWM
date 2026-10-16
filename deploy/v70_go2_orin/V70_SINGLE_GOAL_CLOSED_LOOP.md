# V70 single-image-goal closed loop

This V4 entry point supports exactly one goal image. A manifest with zero or
multiple goals is rejected before any model or motion interface is started.
Long-range/topological routing is deliberately out of scope for this entry
point.

## Preserved V70 contract

The closed loop does not change any trained weight or planner principle:

- one float32 DINO batch containing four observations and one goal;
- the deployed float16 DINO artifact boundary followed by float32 LTWM input;
- 17 NoMaD samples, first 16 references and the K17 trimmed-mean incumbent;
- all three LTWM/scorer members and all three Arm-D selector members;
- frozen RECON and coherent-turn-tail support checks and abstention;
- three CEM rounds with population 16 and the frozen random seed.

Persistent model loading removes process startup and artifact hand-off from
each planning cycle. It does not reduce precision, candidates, members or CEM
work.

## One closed-loop step

1. Command and enable outputs are held at zero.
2. Acquire four time-spaced camera frames and four odometry samples.
3. Run the complete V70 planner against the one goal image.
4. If the calibrated terminal supervisor confirms arrival, assert the stop
   file and remain at zero.
5. Otherwise, if every independent interlock is open, execute only the first
   selected metric SE(2) transition with odometry feedback.
6. Stop, settle, acquire a fresh four-frame observation and replan.

Any stop/arm transition invalidates an in-flight or previously selected
action. It can never resume after re-arming; a fresh plan is mandatory.

## Board-specific sensor isolation

ROS 2 uses `rmw_cyclonedds_cpp` because the Go2 JP6 bare-DDS LiDAR endpoints
deliver odometry and point clouds through CycloneDDS on this board. FastDDS
discovers those endpoints but receives no samples.

The Unitree camera SDK runs in `v70_unitree_camera_worker.py`, a separate
process that imports no ROS module. It forwards the original JPEG bytes over
a local Unix-domain socket. The parent performs the same decode and V70 image
preprocessing as before. This separation avoids a CycloneDDS participant/type
collision without changing image pixels or inference.

## Shadow run (never moves the robot)

```bash
cd /home/unitree/dinov3_ltwm_v70_tridomain_go2_orin_20260730_v4
export V70_SHADOW_ONCE=1
export V70_OUTPUT_DIR=/home/unitree/v70_live_runs/single_goal_shadow
bash deploy/v70_go2_orin/run_v70_single_goal.bash \
  shadow /home/unitree/v70_routes/goal_single.json
```

Shadow mode keeps `/tmp/go2_nav_stop` present, requires the arm file absent,
runs both safety layers in dry-run mode and never initializes SportClient.

## Live run

Live startup is intentionally fail-closed. The single goal manifest must have
`arrival_calibrated: true`, based on standing-robot approach data. The current
example remains false and therefore cannot be used for motion.

The launcher always starts stopped and disarmed:

```bash
bash deploy/v70_go2_orin/run_v70_single_goal.bash \
  live /home/unitree/v70_routes/goal_single.json
```

Only after the dog is standing, the terminal thresholds are calibrated, the
V5 ground/clearance status is valid, and the ready marker reports `live`, an
operator may separately run `arm_v70_live.bash` with its explicit movement
acknowledgement. The launcher itself never arms or removes the stop file.

## 2026-07-30 stopped-board acceptance

- pure state/safety/runtime tests: 29 passed;
- persistent-vs-staged V70 comparison: 29/29 arrays bit-exact, max difference
  0.0;
- complete camera + odometry + V70 + V5 + dry-run executor shadow cycle:
  passed;
- persistent planning cycle: 8.418 s in the final run;
- selected first action: `[0.2281671, 0.0013640, 0.0059780]`;
- final state: stop present, arm absent, no navigation process.

The robot was prone during this acceptance run. Therefore the selected action
is evidence that the software chain works, not a standing-robot navigation
validation or authorization to move.
