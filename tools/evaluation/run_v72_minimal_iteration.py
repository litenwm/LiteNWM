#!/usr/bin/env python3
"""Deployment-latency-correct entry point for the frozen v72 iteration."""

from __future__ import annotations

from pathlib import Path
import sys
import time

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.evaluation import run_v72_minimal_iteration_after_probe as iteration  # noqa: E402
from tools.evaluation import run_v72_proposal_tail_reranker as v72  # noqa: E402


ORIGINAL_LATENCY = v72.latency_cpu_ms


def deployment_latency(name, model_record, data_record):
    if name != "multitask_mlp_extended":
        return ORIGINAL_LATENCY(name, model_record, data_record)
    mlp_record = model_record["mlp"]
    feature = v72.normalize(
        data_record["extended"][0:1], mlp_record["normalization"]
    ).reshape(65, -1)
    model = v72.MultiTaskMLP(int(mlp_record["dimension"]))
    model.load_state_dict(mlp_record["state_dict"], strict=True)
    model.eval()
    torch.set_num_threads(1)
    tensor = torch.from_numpy(feature)
    values = []
    with torch.inference_mode():
        for run in range(230):
            started = time.perf_counter_ns()
            _ = model(tensor)
            elapsed = (time.perf_counter_ns() - started) / 1.0e6
            if run >= 30:
                values.append(elapsed)
    value = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(value.mean()),
        "p50": float(np.quantile(value, 0.5)),
        "p90": float(np.quantile(value, 0.9)),
        "runs": int(len(value)),
        "model_loading_excluded": True,
    }


if __name__ == "__main__":
    v72.latency_cpu_ms = deployment_latency
    iteration.main()
