#!/usr/bin/env python3
"""Corrected entrypoint for pairwise-student failure diagnostics."""

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from tools.evaluation import diagnose_scand_pairwise_student_failure_v1 as base


def metrics(mask, pred, table):
    dual = table["dual_vs_formal_v70"][mask]
    harm = table["harm_vs_formal_v70"][mask]
    helpful = table["helpful_vs_formal_v70"][mask]
    gain = table["objective_gain_vs_formal_v70_m"][mask]
    selected = pred[mask]
    phelp, pdual, pharm = selected[:, 2], selected[:, 3], selected[:, 4]
    pgain = selected[:, 0]
    return {
        "rows": int(mask.sum()),
        "dual_rate": float(dual.mean()),
        "harm_rate": float(harm.mean()),
        "dual_auroc": base.auc(pdual, dual),
        "harm_auroc": base.auc(pharm, harm),
        "helpful_auroc": base.auc(phelp, helpful),
        "dual_brier": float(np.mean((pdual - dual) ** 2)),
        "harm_brier": float(np.mean((pharm - harm) ** 2)),
        "dual_ece10": base.ece(pdual, dual),
        "harm_ece10": base.ece(pharm, harm),
        "objective_gain_pearson": float(np.corrcoef(pgain, gain)[0, 1]),
    }


if __name__ == "__main__":
    base.metrics = metrics
    base.main()
