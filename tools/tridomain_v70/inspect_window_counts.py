#!/usr/bin/env python3
import json
from pathlib import Path


roots = {
    "old_recon": Path("/root/autodl-tmp/dwm_datasets/processed/recon_dinov3_vitl16_adapted_scand_expansion_v2_final_trainval_60e5c5af"),
    "old_scand": Path("/root/autodl-tmp/dwm_datasets/processed/scand_dinov3_vitl16_adapted_expansion_v2_final_trainval_60e5c5af"),
    "new_recon": Path("/root/autodl-tmp/dwm_datasets/processed_v70_tridomain/recon_dinov3_ltwm_h4"),
    "new_scand": Path("/root/autodl-tmp/dwm_datasets/processed_v70_tridomain/scand_dinov3_ltwm_h4"),
    "new_sacson": Path("/root/autodl-tmp/dwm_datasets/processed_v70_tridomain/sacson_dinov3_ltwm_h4"),
}
for name, root in roots.items():
    metadata = json.loads((root / "metadata.json").read_text())
    split = set(json.loads((root / "splits.json").read_text())["train"])
    windows = sum(max(int(row["num_frames"]) - 8 + 1, 0) for row in metadata["trajectories"] if row["trajectory_id"] in split)
    frames = sum(int(row["num_frames"]) for row in metadata["trajectories"] if row["trajectory_id"] in split)
    print(name, "train_trajectories", len(split), "frames", frames, "windows", windows)
