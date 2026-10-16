from __future__ import annotations

import pytest
import torch

from recon_latents.dinov3_adapter import (
    DinoV3TokenAdapter,
    extract_dinov3_patch_tokens,
)


def test_adapter_matches_ltwm_contract() -> None:
    adapter = DinoV3TokenAdapter()
    output = adapter(torch.randn(2, 196, 1024))
    assert output.shape == (2, 16, 16, 384)
    assert torch.isfinite(output).all()


def test_patch_extraction_ignores_prefix_tokens() -> None:
    hidden = torch.randn(1, 201, 1024)
    patches = extract_dinov3_patch_tokens(hidden)
    assert patches.shape == (1, 196, 1024)
    torch.testing.assert_close(patches, hidden[:, -196:])


def test_adapter_rejects_wrong_grid() -> None:
    with pytest.raises(ValueError):
        DinoV3TokenAdapter()(torch.randn(1, 195, 1024))
