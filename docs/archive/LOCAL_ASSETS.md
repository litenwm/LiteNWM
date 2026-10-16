# Local external assets

## DINOv3-L/16

The Hugging Face repository is gated. First accept the DINOv3 license at
<https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m>, then run:

```bash
python3 -m pip install --upgrade huggingface_hub
hf auth login
bash tools/release/download_dinov3_vitl16.sh
```

The script downloads to
`/home/rainbow/models/facebook/dinov3-vitl16-pretrain-lvd1689m` by default. It pins
revision `dd0a398fa8e84f2a37179332f6c561d20276300b` and verifies the required
`model.safetensors` SHA-256:

```text
dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179
```

Pass a different destination as the first argument if needed:

```bash
bash tools/release/download_dinov3_vitl16.sh /absolute/model/directory
```

## Asset audit

Copy the example registry, edit paths if needed, and run the audit:

```bash
cp configs/local_assets.example.json configs/local_assets.json
python tools/release/audit_local_assets.py \
  --allow-missing-dataset-trajectories
```

Before DINOv3 has been downloaded, add `--allow-missing-dinov3`. Omitting an
allow flag makes the corresponding missing asset fail closed.


These two stages do not require DINOv3/LTWM latent shards. Proposal generation
reads RGB only; the separate label builder opens logged pose only after the
proposal artifact is frozen. `cache_trajectory_quality_features_v2.py` still
requires a frozen LTWM checkpoint and latent shards, so it can remain deferred
## NoMaD H4 proposals and labels

Proposal generation reads RGB only and can resume from an exact manifest
prefix:

```bash
NOMAD_LEGACY_SITE_PACKAGES=/home/rainbow/miniconda3/envs/vint_data/lib/python3.8/site-packages \
PYTHONPATH=src /home/rainbow/miniconda3/envs/nwm/bin/python \
  tools/planning/run_nomad_h4_proposals_v2.py \
  --dataset recon \
  --partition train-fit \
  --trajectory-root /home/rainbow/nwm/data/recon \
  --inner-manifest data/inner_manifests/recon/train-fit.windows.jsonl \
  --split-lock data/split_lock/split_lock.json \
  --visualnav-root /home/rainbow/visualnav-transformer \
  --diffusion-policy-root /home/rainbow/diffusion_policy \
  --nomad-config /home/rainbow/visualnav-transformer/train/config/nomad.yaml \
  --nomad-checkpoint /home/rainbow/visualnav-transformer/deployment/model_weights/nomad.pth \
  --output outputs/proposals/recon/train-fit.npz \
  --device cuda \
  --resume
```

Only after proposal generation is frozen should the separate label process
open trusted local `traj_data.pkl` pose files:

```bash
PYTHONPATH=src python tools/scoring/build_trajectory_quality_labels_v2.py \
  --dataset recon \
  --partition train-fit \
  --trajectory-root /home/rainbow/nwm/data/recon \
  --inner-manifest data/inner_manifests/recon/train-fit.windows.jsonl \
  --proposals outputs/proposals/recon/train-fit.npz \
  --split-lock data/split_lock/split_lock.json \
  --output outputs/labels/recon/train-fit.npz
```
