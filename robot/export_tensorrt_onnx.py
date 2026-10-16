from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import MethodType

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(ROOT / "deploy/v70_go2_orin"),
    str(ROOT),
    str(ROOT / "src"),
]

from v70_persistent_planner_fast import PersistentV70Planner  # noqa: E402


def manual_bilinear_border(images: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    n, channels, height, width = images.shape
    out_height, out_width = grid.shape[1:3]
    gx, gy = grid.unbind(dim=-1)
    # Border padding clamps the continuous source coordinate before the
    # bilinear neighbours/weights are formed.
    x = (((gx + 1.0) * width - 1.0) * 0.5).clamp(0.0, width - 1.0)
    y = (((gy + 1.0) * height - 1.0) * 0.5).clamp(0.0, height - 1.0)
    x_floor = torch.floor(x)
    y_floor = torch.floor(y)
    wx = x - x_floor
    wy = y - y_floor
    x0 = x_floor.to(torch.int64).clamp(0, width - 1)
    y0 = y_floor.to(torch.int64).clamp(0, height - 1)
    x1 = (x_floor.to(torch.int64) + 1).clamp(0, width - 1)
    y1 = (y_floor.to(torch.int64) + 1).clamp(0, height - 1)
    flat = images.reshape(n, channels, height * width)

    def take(ix: torch.Tensor, iy: torch.Tensor) -> torch.Tensor:
        index = (iy * width + ix).reshape(n, 1, out_height * out_width)
        index = index.expand(n, channels, out_height * out_width)
        return torch.gather(flat, 2, index).reshape(
            n, channels, out_height, out_width
        )

    v00 = take(x0, y0)
    v01 = take(x0, y1)
    v10 = take(x1, y0)
    v11 = take(x1, y1)
    wx = wx[:, None]
    wy = wy[:, None]
    return (
        v00 * (1.0 - wx) * (1.0 - wy)
        + v10 * wx * (1.0 - wy)
        + v01 * (1.0 - wx) * wy
        + v11 * wx * wy
    )


def manual_warp(self, tokens: torch.Tensor, cumulative_pose: torch.Tensor):
    batch, candidates, horizon = cumulative_pose.shape[:3]
    channels = tokens.shape[-1]
    theta = self.theta(cumulative_pose).reshape(
        batch * candidates * horizon, 2, 3
    )
    images = tokens.reshape(batch, 1, 1, 16, 16, channels).expand(
        batch, candidates, horizon, 16, 16, channels
    )
    images = images.reshape(-1, 16, 16, channels).permute(0, 3, 1, 2)
    axis = (torch.arange(16, device=tokens.device, dtype=tokens.dtype) + 0.5)
    axis = axis * (2.0 / 16.0) - 1.0
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    base = torch.stack((xx, yy, torch.ones_like(xx)), dim=-1).reshape(1, 256, 3)
    grid = torch.matmul(base, theta.transpose(1, 2)).reshape(-1, 16, 16, 2)
    warped = manual_bilinear_border(images, grid)
    warped = warped.permute(0, 2, 3, 1).reshape(
        batch, candidates, horizon, 256, channels
    )
    return warped, theta.reshape(batch, candidates, horizon, 2, 3)


class Wrapper(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, history: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.model(history, actions)["predicted_tokens"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=(1701, 1702, 1703), default=1701)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output or Path(
        str(ROOT / "runtime/tensorrt" / f"seed{args.seed}_manualgrid.onnx")
    )
    planner = PersistentV70Planner(enable_fixed_pool_fast_path=True)
    member = next(
        item for item in planner.ltwm_members if int(item["seed"]) == args.seed
    )
    model = member["wm"]
    history = torch.randn(1, 4, 256, 384, device=planner.device)
    actions = torch.randn(1, 17, 4, 3, device=planner.device) * 0.08
    original_forward = model.warp.forward
    with torch.inference_mode():
        original = model(history, actions)["predicted_tokens"].float()
    model.warp.forward = MethodType(manual_warp, model.warp)
    with torch.inference_mode():
        manual = model(history, actions)["predicted_tokens"].float()
    torch.cuda.synchronize()
    print(
        "PYTORCH_DIFF",
        float((original - manual).abs().mean()),
        float((original - manual).abs().max()),
        float(torch.nn.functional.cosine_similarity(original.flatten(), manual.flatten(), dim=0)),
    )
    model = model.float().cuda().eval()
    history_export = history.float().cuda()
    actions_export = actions.float().cuda()
    output.parent.mkdir(parents=True, exist_ok=True)
    # Avoid the inference-only fused Transformer op, which has no legacy
    # ONNX symbolic. The decomposed attention graph is mathematically the
    # same and is what the existing TensorRT export uses.
    torch.backends.mha.set_fastpath_enabled(False)
    with torch.inference_mode():
        torch.onnx.export(
            Wrapper(model),
            (history_export, actions_export),
            output,
            input_names=["history", "actions"],
            output_names=["predicted_tokens"],
            opset_version=20,
            do_constant_folding=True,
        )
    print("EXPORTED", output, output.stat().st_size)


if __name__ == "__main__":
    main()
