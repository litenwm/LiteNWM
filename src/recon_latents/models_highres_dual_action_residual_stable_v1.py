"""Execution-stable action-residual model contract.

The original draft consumed the base model's already-projected spatial-action
tokens with a layer declared for raw 10D geometry.  This module applies the
single dimensional-contract correction before any model is instantiated.
"""

from __future__ import annotations

from torch import nn

from recon_latents import models_highres_dual_action_residual_v1 as _base


if not getattr(_base._BranchResidual, "_stable_spatial_contract", False):
    _original_init = _base._BranchResidual.__init__

    def _stable_init(
        self,
        *,
        state_dim: int,
        output_dim: int,
        residual_dim: int,
    ) -> None:
        _original_init(
            self,
            state_dim=state_dim,
            output_dim=output_dim,
            residual_dim=residual_dim,
        )
        self.spatial_projection = nn.Sequential(
            nn.Linear(state_dim, residual_dim),
            nn.SiLU(),
            nn.Linear(residual_dim, residual_dim),
        )

    _base._BranchResidual.__init__ = _stable_init
    _base._BranchResidual._stable_spatial_contract = True


ActionResidualConfig = _base.ActionResidualConfig
FrozenDualLTWMActionResidual = _base.FrozenDualLTWMActionResidual
LatentActionDecoder = _base.LatentActionDecoder

__all__ = [
    "ActionResidualConfig",
    "FrozenDualLTWMActionResidual",
    "LatentActionDecoder",
]
