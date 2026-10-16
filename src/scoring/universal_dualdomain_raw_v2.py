"""Shared RECON+SCAND raw scorer with domain-conditioned risk heads.

The primary feature contract intentionally excludes every scorer-v2 prediction.
Spacing support contributes only nonconformity/physical-guard inputs and a hard
validity mask; it never supplies a helpful-action label.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import random
from typing import Any

import numpy as np
import torch
from torch import nn

from scoring.spacing_support_v1 import (
    normalized_action_vector,
    score_spacing_envelope,
)


DOMAINS = ("recon", "scand")
METRICS = ("ate", "rpe", "endpoint", "objective")


@dataclass(frozen=True)
class UniversalModelConfig:
    input_dim: int = 815
    hidden_dim: int = 256
    domain_embedding_dim: int = 16
    dropout: float = 0.10
    domain_conditioned: bool = True
    setwise_context: bool = True
    heteroscedastic: bool = True
    num_domains: int = len(DOMAINS)

    def validate(self) -> None:
        if min(self.input_dim, self.hidden_dim, self.domain_embedding_dim) <= 0:
            raise ValueError("model dimensions must be positive")
        if self.num_domains <= 0:
            raise ValueError('num_domains must be positive')
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0,1)")


class UniversalDualDomainRawV2(nn.Module):
    """One shared candidate trunk with optional FiLM and domain outcome heads."""

    OUTPUT_DIM = 12  # mean4, logvar4, helpful, dual, harm, rank

    def __init__(self, config: UniversalModelConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        hidden = config.hidden_dim
        self.encoder = nn.Sequential(
            nn.Linear(config.input_dim, hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.domain_embedding = nn.Embedding(
            config.num_domains, config.domain_embedding_dim
        )
        self.domain_projection = nn.Linear(config.domain_embedding_dim, hidden)
        if config.domain_conditioned:
            self.film = nn.Sequential(
                nn.Linear(config.domain_embedding_dim, hidden),
                nn.GELU(),
                nn.Linear(hidden, 2 * hidden),
            )
            self.heads = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(hidden, hidden),
                        nn.GELU(),
                        nn.Linear(hidden, self.OUTPUT_DIM),
                    )
                    for _ in range(config.num_domains)
                ]
            )
        else:
            self.film = None
            self.shared_head = nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.GELU(),
                nn.Linear(hidden, self.OUTPUT_DIM),
            )
        if config.setwise_context:
            self.set_context = nn.Sequential(
                nn.Linear(2 * hidden, hidden),
                nn.GELU(),
                nn.Linear(hidden, hidden),
            )
        else:
            self.set_context = None

    def forward(
        self, value: torch.Tensor, domain_id: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        if value.ndim != 3 or domain_id.shape != (len(value),):
            raise ValueError("expected value [B,K,D] and domain_id [B]")
        embedded = self.domain_embedding(domain_id)
        hidden = self.encoder(value)
        hidden = hidden + self.domain_projection(embedded)[:, None]
        if self.film is not None:
            gamma, beta = self.film(embedded).chunk(2, dim=-1)
            hidden = hidden * (1.0 + 0.1 * torch.tanh(gamma[:, None]))
            hidden = hidden + beta[:, None]
        if self.set_context is not None:
            pooled = torch.cat(
                (hidden.mean(dim=1), hidden.max(dim=1).values), dim=-1
            )
            hidden = hidden + self.set_context(pooled)[:, None]
        if self.config.domain_conditioned:
            per_domain = torch.stack(
                [head(hidden) for head in self.heads], dim=2
            )
            gather = domain_id[:, None, None, None].expand(
                -1, hidden.shape[1], 1, self.OUTPUT_DIM
            )
            output = per_domain.gather(2, gather).squeeze(2)
        else:
            output = self.shared_head(hidden)
        logvar = output[..., 4:8]
        if not self.config.heteroscedastic:
            logvar = torch.zeros_like(logvar)
        return {
            "mean": output[..., :4],
            "logvar": logvar.clamp(-7.0, 4.0),
            "helpful_logit": output[..., 8],
            "dual_logit": output[..., 9],
            "harm_logit": output[..., 10],
            "rank_logit": output[..., 11],
        }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def group_fold(domain: str, group: str, folds: int) -> int:
    if domain not in DOMAINS or folds < 2:
        raise ValueError("invalid domain/fold contract")
    digest = hashlib.sha256(
        f"universal-dualdomain-raw-v2:{domain}:{group}".encode()
    ).digest()
    return int.from_bytes(digest[:8], "big") % folds


def _relative_raw(raw: np.ndarray) -> np.ndarray:
    reference = raw[:, 1:]
    center = reference.mean(axis=1, keepdims=True)
    scale = np.maximum(reference.std(axis=1, keepdims=True), 1.0e-5)
    return np.clip((raw - center) / scale, -8.0, 8.0).astype(np.float32)


def build_universal_features(
    data: dict[str, np.ndarray],
    *,
    spacing_m: float,
    support_envelope: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the 815-D primary raw contract and return score/physical guard."""

    action = np.asarray(data["action_features"], dtype=np.float32)
    actions_h4 = np.asarray(data["actions_h4"], dtype=np.float32)
    static = np.asarray(data["static_features"], dtype=np.float32)
    raw = np.asarray(data["raw_ltwm_features"], dtype=np.float32)
    if action.ndim != 3 or action.shape[:2] != actions_h4.shape[:2]:
        raise ValueError("universal action feature shape mismatch")
    count, candidates = action.shape[:2]
    spacing = np.full(count, float(spacing_m), dtype=np.float32)
    physical = actions_h4.reshape(count, candidates, 12)
    normalized = normalized_action_vector(actions_h4, spacing).astype(np.float32)
    raw_delta = (raw - raw[:, :1]).astype(np.float32)
    raw_relative = _relative_raw(raw)
    score, guard = score_spacing_envelope(
        support_envelope, actions_h4, spacing
    )
    value = np.concatenate(
        (
            action,
            physical,
            normalized,
            static,
            raw,
            raw_delta,
            raw_relative,
            np.full((count, candidates, 1), spacing_m, dtype=np.float32),
            np.log1p(score)[..., None].astype(np.float32),
            guard[..., None].astype(np.float32),
        ),
        axis=-1,
    )
    if value.shape != (count, candidates, 815):
        raise RuntimeError(f"universal feature dimension changed: {value.shape}")
    if not np.isfinite(value).all():
        raise RuntimeError("universal feature contains NaN or Inf")
    return value.astype(np.float32), score.astype(np.float32), guard


def build_universal_targets(targets: np.ndarray) -> dict[str, np.ndarray]:
    value = np.asarray(targets, dtype=np.float32)
    if value.ndim != 3 or value.shape[1] != 17 or value.shape[-1] < 4:
        raise ValueError("expected target tensor [N,17,>=4]")
    delta = value[..., :4]
    helpful = delta[..., 0] < -1.0e-7
    dual = (delta[..., 0] < -1.0e-7) & (delta[..., 3] < -1.0e-7)
    harm = (delta[..., 0] > 1.0e-7) | (delta[..., 3] > 1.0e-7)
    utility = (
        delta[..., 0]
        + 0.25 * delta[..., 1]
        + 0.25 * np.maximum(delta[..., 2], 0.0)
        + 0.50 * delta[..., 3]
        + 0.10 * harm.astype(np.float32)
    )
    best = utility[:, 1:].argmin(axis=1) + 1
    pair_target = dual.astype(np.float32)
    shifted = -utility[:, 1:] / 0.03
    shifted -= shifted.max(axis=1, keepdims=True)
    list_distribution = np.exp(shifted)
    list_distribution /= list_distribution.sum(axis=1, keepdims=True)
    return {
        "delta": delta.astype(np.float32),
        "helpful": helpful.astype(np.float32),
        "dual": dual.astype(np.float32),
        "harm": harm.astype(np.float32),
        "utility": utility.astype(np.float32),
        "best_challenger": best.astype(np.int64),
        "pair_target": pair_target,
        "list_distribution": list_distribution.astype(np.float32),
    }


def feature_normalization(
    feature: np.ndarray,
    domain_id: np.ndarray,
    *,
    domain_conditioned: bool,
) -> dict[str, np.ndarray]:
    value = np.asarray(feature, dtype=np.float32)
    domains = np.asarray(domain_id, dtype=np.int64)
    domain_count = int(domains.max()) + 1
    if not np.array_equal(np.unique(domains), np.arange(domain_count)):
        raise ValueError('domain ids must be contiguous and all represented')
    if domain_conditioned:
        mean = np.stack(
            [value[domains == index].reshape(-1, value.shape[-1]).mean(0)
             for index in range(domain_count)]
        )
        scale = np.stack(
            [value[domains == index].reshape(-1, value.shape[-1]).std(0)
             for index in range(domain_count)]
        )
    else:
        flat = value.reshape(-1, value.shape[-1])
        mean = flat.mean(0, keepdims=True)
        scale = flat.std(0, keepdims=True)
        mean = np.repeat(mean, domain_count, axis=0)
        scale = np.repeat(scale, domain_count, axis=0)
    return {
        "feature_mean": mean.astype(np.float32),
        "feature_scale": np.maximum(scale, 1.0e-5).astype(np.float32),
    }


def normalize_features(
    feature: np.ndarray,
    domain_id: np.ndarray,
    normalization: dict[str, np.ndarray],
) -> np.ndarray:
    domains = np.asarray(domain_id, dtype=np.int64)
    mean = np.asarray(normalization["feature_mean"], dtype=np.float32)[domains]
    scale = np.asarray(normalization["feature_scale"], dtype=np.float32)[domains]
    return np.clip(
        (np.asarray(feature, dtype=np.float32) - mean[:, None])
        / scale[:, None],
        -8.0,
        8.0,
    ).astype(np.float32)


def config_dict(config: UniversalModelConfig) -> dict[str, Any]:
    return asdict(config)


__all__ = [
    "DOMAINS",
    "METRICS",
    "UniversalDualDomainRawV2",
    "UniversalModelConfig",
    "build_universal_features",
    "build_universal_targets",
    "config_dict",
    "feature_normalization",
    "group_fold",
    "normalize_features",
    "set_seed",
]
