# LiteNWM

**Efficient Latent World Models for Onboard Visual Navigation in the Wild**

[Project website](https://litenwm.github.io/) · [Paper (arXiv)](https://arxiv.org/abs/2610.12368) · [Model assets](https://github.com/litenwm/LiteNWM/releases/tag/v1.0.0) · [中文说明](README.zh-CN.md)

Linkai Liu, Yuntian Zhang, Zhenshan Bing, Chen Chen, Lingjuan Lyu, Shangguang Wang, Mengwei Xu, and Dongqi Cai.

Nanjing University · Imperial College London · Sony AI · Beijing University of Posts and Telecommunications.

arXiv preprint, 2026.

LiteNWM evaluates trajectories proposed by a frozen navigation policy. Shared DINOv3 encoding and a lightweight adapter feed a latent world model that jointly predicts four future horizons. A learned incumbent-relative scorer evaluates candidate consequences, with an acceptance gate and optional CEM refinement.

## What this release supports

- **CPU smoke tests:** run a small randomly initialized world model/scorer and the real execution state machine without model downloads, ROS, or a robot.
- **Offline inference with frozen weights:** score prepared token/action arrays, or plan from saved RGB observations, a visual goal and aligned poses using the full NoMaD/LiteNWM pipeline.
- **Unitree Go2 EDU / Jetson Orin deployment:** configurable ROS/SDK integration, shadow mode, explicit arming, stop controls and optional device-local TensorRT builds.

The default frozen model is the three-domain V70 release of 2026-07-29 (RECON, SCAND, SACSoN; ensemble seeds 1701/1702/1703). Historical scripts remain available but may require external experiment assets and path configuration.

**The public Go2 execution policy has been modified:** each fresh accepted plan may execute H1 once; old-action replay and buffered H2/H4 execution are removed. This version passed offline and simulated runtime checks, but has not been physically tested on a robot. The website's navigation results and timing measurements describe the original experiments, not a new measurement of this modified runtime. See [paper/release correspondence](docs/PAPER_RELEASE_MAP.md).

## Quick start: no weights or hardware

Python 3.10 or newer. Run commands from the repository root:

```bash
git clone https://github.com/litenwm/LiteNWM.git
cd LiteNWM
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[dev]'
python examples/run_pipeline.py --demo --output outputs/demo.json
python robot/smoke_test.py
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`. For CUDA, install a compatible PyTorch build instead of the CPU wheel. For Jetson, use the board-specific instructions below and preserve the vendor PyTorch/torchvision installation.

The network demo uses synthetic inputs and random small networks; it verifies execution, not model accuracy. The execution smoke test checks the actual H1 follower and fresh-plan gate without importing ROS or the motion SDK.

## Download and install frozen assets

Download `litenwm_go2_assets.tar` and `asset_release.json` from [v1.0.0](https://github.com/litenwm/LiteNWM/releases/tag/v1.0.0). Compare the archive SHA-256 with `asset_release.json`, then run:

```bash
python robot/install_assets.py --archive /absolute/path/litenwm_go2_assets.tar
python robot/preflight.py --assets
```

The installer checks 34 pinned runtime assets against [robot/assets_manifest.json](robot/assets_manifest.json) and writes local path configuration. The bundle contains the adapter, three world-model/scorer members, selector members, support/calibration files, DINOv3 and NoMaD pretrained assets, and a minimal external Python dependency subset. Project-trained and upstream materials have different licenses; see [third-party notices](THIRD_PARTY_NOTICES.md). Model files are excluded from Git history.

For saved-RGB planning, also install the runtime dependencies described in [robot/README.md](robot/README.md), supply four time-ordered observations, a goal image and aligned poses, and run:

```bash
python robot/offline_plan.py --history h0.jpg h1.jpg h2.jpg h3.jpg \
  --goal goal.jpg --poses poses.json --device cpu --cem-iterations 1 \
  --output outputs/robot_plan.json
```

This entry sends no motion commands. For prepared token/action arrays, use [docs/RUNNING.md](docs/RUNNING.md) and `examples/run_pipeline.py`; that simpler entry performs single-member scoring and does not run the full selector or CEM.

## Go2 deployment

The reference platform is Go2 EDU with Jetson Orin, Ubuntu 22.04, ROS 2 Humble and Python 3.10. Install board-compatible dependencies, configure camera/odometry/cloud and network settings, prepare a real goal, and complete preflight before running shadow mode:

```bash
bash robot/run_go2.bash shadow /absolute/path/goal.json
```

Follow [the full deployment guide](robot/README.md) for environment setup, goal calibration, live startup, explicit arming and stopping. The launcher does not arm automatically. Keep physical stop controls available. OpenTrackVLA is not required.

[Deployment optimizations](robot/OPTIMIZATIONS.md) documents persistent model loading, feature/condition caching, fixed-pool selection, conditional CEM and optional prefetch/TensorRT. [Offline readiness](robot/OFFLINE_READINESS.md) distinguishes simulation checks from required hardware validation.

## Validation and reproducibility

```bash
python tools/release/verify_repository.py
python -m pytest -q tests/robot tests/deploy/test_v70_closed_loop_core.py \
  tests/deploy/test_v70_goal_stop.py tests/planning/test_safe_bridge_core.py \
  tests/recon_latents/test_ltwm_horizon.py \
  tests/scoring/test_trajectory_quality_scorer_v2.py \
  tests/planning/test_trajectory_constraints.py
```

The current local curated suite has 53 passing tests. Real weights loaded and completed synthetic-RGB planning on CPU. This does not establish navigation accuracy, Jetson compatibility, DDS connectivity or physical motion. See [validation records](docs/VALIDATION.md).

Full paper-table reproduction additionally requires the original datasets, matching three-domain split lock, candidate pools, evaluation artifacts and configurations. These are not all included in this runtime bundle. Consult [docs/PAPER_RELEASE_MAP.md](docs/PAPER_RELEASE_MAP.md) before interpreting historical scripts as complete reproduction commands.

## Repository guide

| Path | Purpose |
| --- | --- |
| `src/recon_latents/` | Visual adapter and latent world model |
| `src/scoring/`, `src/planning/` | Scoring, calibration, selection, CEM and constraints |
| `examples/` | Weight-free demo and prepared-input scoring |
| `robot/`, `deploy/v70_go2_orin/` | Public deployment tools and implementation |
| `tools/`, `configs/` | Research training/evaluation scripts and configurations |
| `data/`, `data_splits/` | Split indices/provenance; no original image datasets |
| `docs/`, `manifests/` | Protocols, historical records and file/model hashes |

`pipeline_contract.json` is a historical 2026-07-19 dual-domain contract. It is not the current three-domain model specification; see [latest version](docs/LATEST_VERSION.md).

## Citation

```bibtex
@misc{liu2026litenwm,
  title={LiteNWM: Efficient Latent World Models for Onboard Visual Navigation in the Wild},
  author={Linkai Liu and Yuntian Zhang and Zhenshan Bing and Chen Chen and Lingjuan Lyu and Shangguang Wang and Mengwei Xu and Dongqi Cai},
  year={2026},
  eprint={2610.12368},
  archivePrefix={arXiv},
  primaryClass={cs.RO},
  url={https://arxiv.org/abs/2610.12368}
}
```

## License

Original project code/documentation and explicitly covered project-trained release weights are licensed under **CC BY-NC 4.0**. Noncommercial sharing and adaptation require attribution, a license notice and an indication of changes. Commercial use requires separate permission from the relevant rights holders. See [LICENSE.md](LICENSE.md) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

Upstream code, pretrained models, datasets, the paper and the independently hosted website retain their own terms. The citation request above does not add restrictions to the standard license.
