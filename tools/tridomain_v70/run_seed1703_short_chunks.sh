#!/usr/bin/env bash
set -euo pipefail

REPO=/root/autodl-tmp/dinov3-nomad-ltwm-v70-tridomain
RUN=/root/autodl-tmp/dwm_runs/v70_tridomain_new_weights
RUN_STAGE="$REPO/tools/tridomain_v70/run_seed1703_stage.sh"

warm="$RUN/ltwm/seed1703_stage1/checkpoints/final.pt"

for spec in \
  "2a 1704" \
  "2b 1705" \
  "2c 1706" \
  "3a 1707" \
  "3b 1708" \
  "3c 1709"; do
  read -r stage seed <<< "$spec"
  summary="$RUN/ltwm/seed1703_stage${stage}/report/train_summary.json"
  if [[ ! -f "$summary" ]]; then
    bash "$RUN_STAGE" "$stage" "$seed" "$warm" 2000
  fi
  warm="$RUN/ltwm/seed1703_stage${stage}/checkpoints/final.pt"
done

mkdir -p "$RUN/ltwm/seed1703/checkpoints" "$RUN/ltwm/seed1703/report"

RUN="$RUN" /root/DWM/.venv/bin/python - <<'PY'
import json
import os
import shutil
from pathlib import Path

run = Path(os.environ["RUN"])
stage_names = ["1", "2a", "2b", "2c", "3a", "3b", "3c"]
records = []
for name in stage_names:
    root = run / "ltwm" / f"seed1703_stage{name}"
    summary = json.loads((root / "report" / "train_summary.json").read_text())
    records.append(
        {
            "stage": name,
            "best_score": float(summary["best_score"]),
            "best_checkpoint": str(root / "checkpoints" / "best.pt"),
            "final_checkpoint": str(root / "checkpoints" / "final.pt"),
            "history": summary["history"],
        }
    )

best = min(records, key=lambda row: row["best_score"])
output = run / "ltwm" / "seed1703"
shutil.copy2(best["best_checkpoint"], output / "checkpoints" / "best.pt")
shutil.copy2(records[-1]["final_checkpoint"], output / "checkpoints" / "final.pt")
aggregate = {
    "status": "complete",
    "model": "LiteTokenFormer-WM-v2-4ctx-base",
    "seed": 1703,
    "recovery_mode": "bounded_2000_batch_chunks_after_external_sigkill",
    "stages": records,
    "best_stage": best["stage"],
    "best_score": best["best_score"],
    "best_checkpoint": str(output / "checkpoints" / "best.pt"),
    "final_checkpoint": str(output / "checkpoints" / "final.pt"),
}
(output / "report" / "train_summary.json").write_text(
    json.dumps(aggregate, indent=2) + "\n", encoding="utf-8"
)
PY

touch "$RUN/LTWM_PIPELINE_COMPLETE"
