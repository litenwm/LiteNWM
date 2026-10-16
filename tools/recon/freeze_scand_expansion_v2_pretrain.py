#!/usr/bin/env python3
"""Freeze the exact inputs, baselines, gates, and commands before v2 training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from recon_latents.split_lock import file_sha256, load_split_lock  # noqa: E402


SEEDS = (1701, 1702, 1703)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--recon-latent-root", type=Path, required=True)
    parser.add_argument("--scand-latent-root", type=Path, required=True)
    parser.add_argument(
        "--parent-checkpoint-root",
        type=Path,
        default=REPO_ROOT / "checkpoints/project/ltwm_parent",
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=REPO_ROOT / "outputs/checkpoints",
    )
    parser.add_argument(
        "--report-root",
        type=Path,
        default=REPO_ROOT / "outputs/reports",
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    config = read_json(args.config)
    split_lock_path = Path(config["split_lock"])
    parent_lock_path = Path(config["parent_split_lock"])
    split_lock, split_sha = load_split_lock(split_lock_path)
    _, parent_sha = load_split_lock(parent_lock_path)
    if split_sha != config["split_lock_sha256"]:
        raise ValueError("config/final split-lock hash mismatch")
    if parent_sha != config["parent_split_lock_sha256"]:
        raise ValueError("config/parent split-lock hash mismatch")

    recon_metadata = read_json(args.recon_latent_root / "metadata.json")
    scand_metadata = read_json(args.scand_latent_root / "metadata.json")
    recon_splits = read_json(args.recon_latent_root / "splits.json")
    scand_splits = read_json(args.scand_latent_root / "splits.json")
    for name, metadata, splits in (
        ("recon", recon_metadata, recon_splits),
        ("scand", scand_metadata, scand_splits),
    ):
        if metadata.get("split_lock_sha256") != split_sha:
            raise ValueError(f"{name} latent root is not bound to final lock")
        if splits.get("test"):
            raise ValueError(f"{name} train/val latent root exposes test")

    baselines: dict[str, Any] = {}
    commands: dict[str, str] = {}
    checkpoint_hashes: dict[str, str] = {}
    for seed in SEEDS:
        scand_path = args.baseline_dir / f"scand_val_parent_seed{seed}.json"
        recon_path = args.baseline_dir / f"recon_val_parent_seed{seed}.json"
        scand_report = read_json(scand_path)
        recon_report = read_json(recon_path)
        for dataset, report in (("scand", scand_report), ("recon", recon_report)):
            if report.get("status") != "PASS":
                raise ValueError(f"{dataset} seed {seed} baseline did not pass")
            if report.get("split_lock_sha256") != split_sha:
                raise ValueError(f"{dataset} seed {seed} baseline lock mismatch")
            if report.get("dataset") != dataset or report.get("split") != "val":
                raise ValueError(f"{dataset} seed {seed} baseline scope mismatch")
            result = report["results"][0]
            if result.get("checkpoint_seed") != seed:
                raise ValueError(f"{dataset} seed {seed} checkpoint seed mismatch")
            if result.get("finite_fraction") != 1.0:
                raise ValueError(f"{dataset} seed {seed} is not fully finite")
            if result["checkpoint_split_lineage"].get("status") != "PASS":
                raise ValueError(f"{dataset} seed {seed} lineage failed")
        scand_h4 = float(scand_report["results"][0]["true"]["nmse"][3])
        recon_h4 = float(recon_report["results"][0]["true"]["nmse"][3])
        guard = recon_h4 * 1.05
        checkpoint = args.parent_checkpoint_root / f"seed{seed}/best.pt"
        checkpoint_sha = file_sha256(checkpoint)
        if checkpoint_sha != scand_report["results"][0]["checkpoint_sha256"]:
            raise ValueError(f"seed {seed} checkpoint changed after baseline")
        checkpoint_hashes[str(seed)] = checkpoint_sha
        baselines[str(seed)] = {
            "scand_val_h4_nmse": scand_h4,
            "recon_val_h4_nmse": recon_h4,
            "recon_val_h4_nmse_limit": guard,
            "scand_report": str(scand_path),
            "scand_report_sha256": file_sha256(scand_path),
            "recon_report": str(recon_path),
            "recon_report_sha256": file_sha256(recon_path),
            "parent_checkpoint": str(checkpoint),
            "parent_checkpoint_sha256": checkpoint_sha,
        }
        checkpoint_dir = args.checkpoint_root / f"seed{seed}"
        report_dir = args.report_root / f"seed{seed}"
        commands[str(seed)] = (
            f"{args.python} "
            "tools/recon/train_litetokenformer_wm_v2_4ctx.py "
            f"--latent-root {args.recon_latent_root} "
            f"--scand-latent-root {args.scand_latent_root} "
            f"--split-lock {split_lock_path} "
            "--recon-batches-per-scand 3 "
            "--history-frames 4 --future-frames 4 "
            f"--recon-val-h4-nmse-limit {guard:.12f} "
            f"--checkpoint-dir {checkpoint_dir} "
            f"--report-dir {report_dir} "
            f"--seed {seed} --epochs 1 --batches-per-epoch 36964 "
            "--eval-batches-per-epoch 1046 --batch-size 4 "
            "--eval-batch-size 4 --no-train-drop-last "
            "--lr 0.00005 --weight-decay 0.0001 --grad-clip 1.0 "
            "--stream-cache-shards 8 --num-workers 0 --amp bf16 "
            "--contrastive-every 4 --consistency-every 8 --train-scope all "
            f"--warm-start-checkpoint {checkpoint} "
            f"--warm-start-parent-split-lock {parent_lock_path}"
        )

    code_paths = (
        REPO_ROOT / "src/recon_latents/split_lock.py",
        REPO_ROOT / "tools/recon/build_scand_training_expansion_v2.py",
        REPO_ROOT / "tools/recon/encode_dinov3_adapted.py",
        REPO_ROOT / "tools/recon/train_litetokenformer_wm_v2_4ctx.py",
        REPO_ROOT / "tools/recon/eval_litetokenformer_wm_v2_4ctx.py",
        Path(__file__).resolve(),
    )
    payload = {
        "status": "PRETRAIN_FROZEN",
        "test_accessed": False,
        "test_latents_encoded": False,
        "split_lock": str(split_lock_path),
        "split_lock_sha256": split_sha,
        "parent_split_lock": str(parent_lock_path),
        "parent_split_lock_sha256": parent_sha,
        "config": str(args.config),
        "config_sha256": file_sha256(args.config),
        "data_summary": split_lock["scand"]["split_summary"],
        "latent_roots": {
            "recon": {
                "path": str(args.recon_latent_root),
                "metadata_sha256": file_sha256(
                    args.recon_latent_root / "metadata.json"
                ),
                "splits_sha256": file_sha256(
                    args.recon_latent_root / "splits.json"
                ),
            },
            "scand": {
                "path": str(args.scand_latent_root),
                "metadata_sha256": file_sha256(
                    args.scand_latent_root / "metadata.json"
                ),
                "splits_sha256": file_sha256(
                    args.scand_latent_root / "splits.json"
                ),
            },
        },
        "parent_checkpoint_hashes": checkpoint_hashes,
        "baseline_mean_scand_val_h4_nmse": (
            sum(item["scand_val_h4_nmse"] for item in baselines.values())
            / len(baselines)
        ),
        "baselines": baselines,
        "acceptance_gate": {
            "recon": "each seed <= its frozen parent baseline * 1.05",
            "scand_mean": "continued three-seed mean < frozen parent mean",
            "scand_seeds": "at least two of three continued seeds < parent",
        },
        "commands": commands,
        "code_sha256": {
            str(path.relative_to(REPO_ROOT)): file_sha256(path)
            for path in code_paths
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    digest = file_sha256(args.output)
    args.output.with_suffix(args.output.suffix + ".sha256").write_text(
        f"{digest}  {args.output.name}\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "output": str(args.output),
                "sha256": digest,
                "baseline_mean_scand_val_h4_nmse": payload[
                    "baseline_mean_scand_val_h4_nmse"
                ],
                "recon_limits": {
                    seed: item["recon_val_h4_nmse_limit"]
                    for seed, item in baselines.items()
                },
                "test_accessed": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
