"""Static-shape TensorRT runtime for the V70 K17/H4 LTWM ensemble.

The engines retain the production contract: FP32 history/action inputs and an
FP32 predicted-token output with shape (1, 17, 4, 256, 384). TensorRT is an
optional backend; importing this module has no effect on the PyTorch path.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import tensorrt as trt
import torch


EXPECTED_INPUTS = {
    "history": (1, 4, 256, 384),
    "actions": (1, 17, 4, 3),
}
EXPECTED_OUTPUT = (1, 17, 4, 256, 384)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _torch_dtype(dtype: trt.DataType) -> torch.dtype:
    return torch.from_numpy(
        np.empty((), dtype=trt.nptype(dtype))
    ).dtype


class TensorRTLTWM:
    """Callable drop-in replacement for one ``LiteTokenFormerWMV2``."""

    def __init__(
        self,
        engine_path: str | Path,
        *,
        expected_sha256: str | None = None,
        device: torch.device | str = "cuda",
        fallback: Any | None = None,
    ) -> None:
        self.engine_path = Path(engine_path)
        if not self.engine_path.is_file():
            raise FileNotFoundError(self.engine_path)
        if expected_sha256 is not None:
            actual = _sha256(self.engine_path)
            if actual != expected_sha256:
                raise RuntimeError(
                    f"TensorRT engine SHA mismatch for {self.engine_path}: "
                    f"expected {expected_sha256}, got {actual}"
                )
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("TensorRT LTWM requires CUDA")
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.fallback = fallback
        self.enabled = True
        self.last_error: str | None = None
        self.logger = trt.Logger(trt.Logger.ERROR)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(
            self.engine_path.read_bytes()
        )
        if self.engine is None:
            raise RuntimeError(
                f"failed to deserialize TensorRT engine {self.engine_path}"
            )
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError(
                f"failed to create TensorRT context {self.engine_path}"
            )

        names = {
            self.engine.get_tensor_name(index)
            for index in range(self.engine.num_io_tensors)
        }
        if names != {"history", "actions", "predicted_tokens"}:
            raise RuntimeError(
                f"unexpected TensorRT IO contract: {sorted(names)}"
            )
        for name, shape in EXPECTED_INPUTS.items():
            actual = tuple(self.context.get_tensor_shape(name))
            if actual != shape:
                raise RuntimeError(
                    f"{name}: expected shape {shape}, got {actual}"
                )
            if self.engine.get_tensor_dtype(name) != trt.float32:
                raise RuntimeError(f"{name}: expected FP32 TensorRT input")
        output_shape = tuple(
            self.context.get_tensor_shape("predicted_tokens")
        )
        if output_shape != EXPECTED_OUTPUT:
            raise RuntimeError(
                "predicted_tokens: expected shape "
                f"{EXPECTED_OUTPUT}, got {output_shape}"
            )
        output_dtype = _torch_dtype(
            self.engine.get_tensor_dtype("predicted_tokens")
        )
        if output_dtype != torch.float32:
            raise RuntimeError("predicted_tokens: expected FP32 output")
        self.output = torch.empty(
            EXPECTED_OUTPUT,
            dtype=output_dtype,
            device=self.device,
        )

    def __call__(
        self,
        history: torch.Tensor,
        actions: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if not self.enabled:
            if self.fallback is None:
                raise RuntimeError(self.last_error or "TensorRT LTWM disabled")
            return self.fallback(history, actions)
        try:
            values = {"history": history, "actions": actions}
            for name, expected_shape in EXPECTED_INPUTS.items():
                value = values[name]
                if tuple(value.shape) != expected_shape:
                    raise RuntimeError(
                        f"{name}: expected shape {expected_shape}, "
                        f"got {tuple(value.shape)}"
                    )
                if value.device != self.device:
                    raise RuntimeError(
                        f"{name}: expected device {self.device}, got {value.device}"
                    )
                if value.dtype != torch.float32:
                    raise RuntimeError(
                        f"{name}: expected FP32, got {value.dtype}"
                    )
                if not value.is_contiguous():
                    value = value.contiguous()
                    values[name] = value
                self.context.set_tensor_address(name, value.data_ptr())

            self.context.set_tensor_address(
                "predicted_tokens", self.output.data_ptr()
            )
            stream = torch.cuda.current_stream(self.device)
            if not self.context.execute_async_v3(stream.cuda_stream):
                raise RuntimeError("TensorRT LTWM execution failed")
            # Synchronize once per ensemble member so asynchronous runtime
            # failures and non-finite values are caught before scoring. A
            # failed backend is latched off for the rest of the process.
            stream.synchronize()
            if not bool(torch.isfinite(self.output).all()):
                raise RuntimeError("TensorRT LTWM output contains NaN/Inf")
            return {"predicted_tokens": self.output}
        except Exception as error:
            self.enabled = False
            self.last_error = f"{type(error).__name__}: {error}"
            if self.fallback is None:
                raise
            return self.fallback(history, actions)


def load_manifest(path: str | Path) -> dict[int, dict[str, Any]]:
    manifest_path = Path(path)
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    if raw.get("schema") != "v70_ltwm_tensorrt_v1":
        raise RuntimeError(f"unsupported TensorRT manifest {manifest_path}")
    values: dict[int, dict[str, Any]] = {}
    for item in raw.get("engines", []):
        seed = int(item["seed"])
        values[seed] = dict(item)
        engine = Path(item["path"]).expanduser()
        values[seed]["path"] = str(engine if engine.is_absolute() else manifest_path.parent / engine)
    if set(values) != {1701, 1702, 1703}:
        raise RuntimeError(
            f"TensorRT manifest must contain seeds 1701/1702/1703: {path}"
        )
    return values
