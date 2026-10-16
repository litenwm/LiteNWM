# v70 Go2 / Jetson Orin runtime bundle

This bundle contains the frozen formal-v70 model/runtime asset closure:

- DINOv3 ViT-L/16 backbone and project adapter;
- three LTWM expansion members and three scorer-v2 members;
- three full Arm-D deployment members and the support envelope;
- NoMaD checkpoint, config, and the minimal external inference source closure;
- the frozen v70 CEM/policy contract and project source.

It excludes raw RECON/SCAND data, training caches, OOF checkpoints,
reserve inputs/outcomes, and all historical experiment payloads.

The source v70 result explicitly sets `production_safe_claim_allowed=false`.
This is therefore a model/runtime bundle, not a safety-certified motor
controller. Keep the existing Go2 chain
`/nav/cmd_vel_raw -> safe_bridge -> /nav/cmd_vel_safe -> motion_executor`
and its arm/stop/heartbeat gates.

## Orin environment

Use the JetPack-provided CUDA PyTorch. Never install generic `torch` or
`torchvision` wheels from PyPI.

```bash
python3 -m venv --system-site-packages /home/unitree/venvs/v70-orin
source /home/unitree/venvs/v70-orin/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install --no-deps -r deploy/v70_go2_orin/requirements-orin.txt
source deploy/v70_go2_orin/orin_env.sh
python deploy/v70_go2_orin/verify_bundle.py --imports
```

For an exhaustive payload audit, run:

```bash
python deploy/v70_go2_orin/verify_bundle.py --all-files
```

`configs/local_assets.json` already uses paths relative to the extracted
bundle. Empty `runtime_inputs/recon` and `runtime_inputs/scand` directories
are placeholders for future route inputs; no dataset is needed merely to
verify or transfer the model.

