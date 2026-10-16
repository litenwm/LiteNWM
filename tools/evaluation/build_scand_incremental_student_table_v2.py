#!/usr/bin/env python3
"""Corrected entrypoint for the spent incremental-student table builder."""

from __future__ import annotations

import sys

import numpy as np

from tools.evaluation import build_scand_incremental_student_table_v1 as base
from planning.proposal_tail_sidecar_v1 import action_structure_features


def safe_channels(values, formal):
    shape = values["energy"].shape
    observed = np.ones(shape, dtype=bool)
    for name in ("energy", "metric_mean", "metric_std", "dual_probability", "harm_probability", "rank_logit"):
        key = name + "_observed"
        if key in values:
            observed &= values[key].astype(bool)
    channel = np.concatenate((
        observed[..., None].astype(np.float32),
        values["energy"][..., None].astype(np.float32),
        values["metric_mean"].astype(np.float32),
        values["metric_std"].astype(np.float32),
        values["dual_probability"][..., None].astype(np.float32),
        values["harm_probability"][..., None].astype(np.float32),
        values["rank_logit"][..., None].astype(np.float32),
    ), axis=-1)
    channel[~np.isfinite(channel)] = 0.0
    return base.formal_delta(channel, formal)


def build_feature(candidates, formal, actions, raw, scorer, ate_seed, rtp_seed, support_seed, safe, source_round, source_candidate, pool, old_safe):
    geometry = candidates.astype(np.float32) * 0.25
    action = actions.astype(np.float32)
    structure = action_structure_features(action)
    seeds = base.seed_summary(ate_seed, rtp_seed, support_seed)
    blocks = []
    names = []
    for name, value in (
        ("ltwm_temporal", raw.astype(np.float32)),
        ("scorer_aux", scorer.astype(np.float32)),
        ("waypoint_m", geometry),
        ("actions_h4", action),
        ("action_structure", structure),
        ("seed_disagreement", seeds),
    ):
        delta, formal_value = base.formal_delta(value.reshape(*value.shape[:2], -1), formal)
        blocks.extend((delta, formal_value))
        names.extend([f"delta_{name}_{i}" for i in range(delta.shape[-1])])
        names.extend([f"formal_{name}_{i}" for i in range(formal_value.shape[-1])])
    safe_delta, safe_formal = safe_channels(safe, formal)
    blocks.extend((safe_delta, safe_formal))
    names.extend([f"delta_safety_{i}" for i in range(safe_delta.shape[-1])])
    names.extend([f"formal_safety_{i}" for i in range(safe_formal.shape[-1])])
    source_id = np.clip(source_round.astype(np.int64) + 1, 0, 3)
    source_onehot = np.eye(4, dtype=np.float32)[source_id]
    source_rank = np.clip(source_candidate.astype(np.float32) / 16.0, 0.0, 1.0)[..., None]
    ranks = np.concatenate((
        base.percentile(safe["energy"], pool),
        base.percentile(safe["dual_probability"], pool, descending=True),
        base.percentile(safe["harm_probability"], pool),
        base.percentile(safe["metric_std"][..., 3], pool),
        base.percentile(0.4 * scorer[..., 6] + 0.6 * scorer[..., 8], pool),
    ), axis=-1)
    is_formal = np.broadcast_to(np.arange(pool.shape[1])[None, :] == formal[:, None], pool.shape)
    flags = np.stack((pool.astype(np.float32), old_safe.astype(np.float32), is_formal.astype(np.float32)), axis=-1)
    blocks.extend((source_onehot, source_rank, ranks, flags))
    names.extend([f"source_{i}" for i in range(4)] + ["source_candidate_rank"] + [f"within_state_rank_{i}" for i in range(5)] + ["physical_support", "old_safe", "is_formal"])
    feature = np.concatenate(blocks, axis=-1).astype(np.float32)
    base.require(np.isfinite(feature).all(), "nonfinite common runtime feature")
    base.require(feature.shape[-1] == len(names), "feature name mismatch")
    return feature, names


def main():
    base.safe_channels = safe_channels
    base.build_feature = build_feature
    base.main()


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(base.canonical_json({"status": "fail_closed", "error_type": type(error).__name__, "error": str(error), "reserve_content_accessed": False}), file=sys.stderr)
        raise SystemExit(2) from error
