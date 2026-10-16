# V70 Go2 persistent live closed loop

## Preserved V70 contract

The live loop changes orchestration only. It preserves:

- the sealed latest V70 DINO adapter, LTWM/scorer and Arm-D weights;
- one DINO batch containing four observations and one goal;
- the staged float16 DINO-to-LTWM boundary;
- 17 NoMaD diffusion samples and the formal trimmed-mean K17 pool;
- three LTWM/scorer members and three Arm-D members;
- RECON spacing, central and coherent-turn-tail support;
- frozen selector calibration and abstention;
- three CEM iterations, population 16 and frozen RNG seed.

Do not enable FP16/TensorRT, reduce the ensemble, change CEM population or
iterations, or alter support/calibration without a separate equivalence and
recalibration study.

## Process chain

```text
Unitree camera + ROS odometry
        |
persistent V70 planner (always zero while planning)
        |
odometry-closed-loop turn -> drive for selected first H4 transition
        |
/nav/cmd_vel_raw + /nav/motion_enable
        |
safe_bridge_v5 (ground-aware LiDAR, stop/arm/freshness gates)
        |
/nav/cmd_vel_safe
        |
motion_executor_v2 (independent stop/arm/heartbeat/watchdog)
        |
Unitree SportClient
```

Every process defaults to zero output. `motion_executor_v2.py` does not even
initialize SportClient when `dry_run=true`.

## Route manifest

Long-range navigation is an explicit ordered list of goal images. Each goal
must contain calibrated arrival thresholds. The live launcher rejects any
goal with `arrival_calibrated=false`.

The values in `routes/goal_single.example.json` are placeholders for shadow
collection only. Marking them calibrated without a standing approach
sequence is not valid.

For each route node, collect standing and stationary observations at:

1. far negative locations;
2. approach locations;
3. the intended stop location;
4. slightly past the stop location;
5. changed lighting and small lateral/yaw offsets.

Choose enter/exit thresholds from these real sequences, keep hysteresis, and
retain at least four consecutive confirmations. A goal-candidate frame
immediately commands zero while confirmation is running.

## Shadow start

Before every start:

```bash
touch /tmp/go2_nav_stop
test ! -e /run/go2-safety/ARMED
```

Copy and edit the example route, then run:

```bash
deploy/v70_go2_orin/run_v70_closed_loop.bash \
  shadow \
  /home/unitree/v70_routes/goal_single.json
```

For one far-goal planning cycle:

```bash
V70_SHADOW_ONCE=1 \
deploy/v70_go2_orin/run_v70_closed_loop.bash \
  shadow \
  /home/unitree/v70_routes/goal_single.json
```

Shadow mode keeps the V5 bridge and executor in `dry_run=true`.

## Live start

The robot must already be standing and physically supervised. Starting the
live processes still requires the stop file present and arm file absent:

```bash
deploy/v70_go2_orin/run_v70_closed_loop.bash \
  live \
  /home/unitree/v70_routes/calibrated_route.json
```

Wait for `/tmp/v70_nav_ready`. Inspect `status.json`, all three logs, LiDAR
clearance and the hardware scene. The launcher never arms the robot.

Only an operator who intends motion runs:

```bash
deploy/v70_go2_orin/arm_v70_live.bash \
  I_UNDERSTAND_THIS_WILL_MOVE_THE_ROBOT
```

To stop:

```bash
deploy/v70_go2_orin/disarm_v70_live.bash
```

The disarm script asserts the stop file before removing the arm file.

## Execution behavior

The planner is stop-and-go because the unchanged V70 3x16 computation is
currently several seconds on Orin:

1. command zero and snapshot four camera/odometry samples;
2. run the exact persistent V70 planner;
3. reject the result if the robot moved while planning;
4. apply goal-arrival/abstention/support gates;
5. turn to the selected first transition heading;
6. drive its metric distance with odometry feedback;
7. settle at zero and replan.

The executor rejects reverse motion, inconsistent heading/action triples,
oversized translation/yaw, stale odometry, no progress, excessive cross-track
error and timeouts. Any such fault asserts `/tmp/go2_nav_stop`.

## Required validation before first motion

1. `v70_persistent_parity_shadow.py` passes on a frozen board snapshot.
2. Remote unit tests pass.
3. Shadow one-cycle output contains no non-zero safe/executor command.
4. Stop-file, arm-file, command timeout, enable timeout, stale odometry,
   stale cloud and obstacle fault injections all produce zero.
5. Arrival thresholds are calibrated from standing real sequences.
6. First live trial uses one goal, open floor, physical supervisor and
   hardware emergency stop, before adding obstacles or long routes.
