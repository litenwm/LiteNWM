# V70 Go2 / Orin latency protocol

This benchmark runs only the shadow pipeline. It never imports a motion
executor, creates a velocity publisher, or calls Unitree `SportClient`.
Execution is fail-closed: `/tmp/go2_nav_stop` must exist and
`/run/go2-safety/ARMED` must be absent before every subprocess.

## Reported primary metrics

| Metric | Exact scope |
|---|---|
| `dinov3_encode_ms` | One CUDA-synchronized DINOv3 + adapter forward over four current observations and one goal image |
| `nomad_candidate_generation_ms` | NoMaD diffusion generation of 17 candidate trajectories |
| `ltwm_future_inference_ms` | Sum of three CUDA-synchronized LTWM future-token forwards |
| `selector_fixed_pool_planning_ms` | Three-member Arm-D prediction plus energy, safety, abstention, and selection for the fixed K17 pool |
| `cem_planning_ms` | Three CEM rounds plus final archive/action selection |
| `selector_and_cem_planning_ms` | Fixed-pool selector plus CEM planning |
| `complete_planning_cycle_wall_ms` | Serialized wall time from camera capture through saved CEM decision |

Every model timing uses `time.perf_counter_ns()`. CUDA work is synchronized
before the clock is read. The benchmark also reports process-wall timings,
sensor waits, model/setup costs, feature/scorer costs, artifact I/O, and the
sum of the main compute stages.

The present deployment launches each stage as a fresh process and loads
several checkpoints inside selector/CEM calls. Therefore process-wall latency
is the real latency of the current script pipeline; it is not a
model-persistent optimized-runtime estimate.

## Recommended board run

Use the same Jetson power mode and thermal state for every comparison. Record
the output of `nvpmodel -q`, `jetson_clocks --show`, and `tegrastats` alongside
the result directory. The benchmark does not change power or clock settings.

Run at least one warm-up and ten measured cycles; use three warm-ups and twenty
measured cycles for paper tables:

```bash
touch /tmp/go2_nav_stop
test ! -e /run/go2-safety/ARMED

python deploy/v70_go2_orin/v70_board_latency_benchmark.py \
  --goal-image /home/unitree/goal.jpg \
  --warmup 3 \
  --repeats 20
```

Outputs:

- `protocol.json`: host, safety, and measurement contract;
- `samples.jsonl`: one exact record per warm-up/measured cycle;
- `summary.json`: min/mean/std/p50/p90/p95/max for every millisecond metric;
- one directory per cycle containing stage JSON, artifacts, and full logs.

Warm-up cycles are retained for audit but excluded from summary statistics.
For the paper, report at least p50 and p95 together with the sample count,
Jetson power mode, clock policy, software versions, K/H, ensemble sizes, and
CEM population/iterations.
