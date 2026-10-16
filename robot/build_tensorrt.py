"""Build device-local BF16 LTWM engines. Requires TensorRT on the target GPU."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--onnx-dir", type=Path, default=ROOT / "runtime/tensorrt")
    p.add_argument("--output-dir", type=Path, default=ROOT / "runtime/tensorrt")
    p.add_argument("--workspace-gib", type=float, default=2)
    args = p.parse_args()
    if not 0 < args.workspace_gib <= 16: raise ValueError("Invalid workspace budget")
    import tensorrt as trt
    if not hasattr(trt.BuilderFlag, "BF16"):
        raise RuntimeError("This BF16 export requires a TensorRT build with BF16 support")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger = trt.Logger(trt.Logger.WARNING)
    records = []
    for seed in [1701, 1702, 1703]:
        onnx = args.onnx_dir / f"seed{seed}_manualgrid.onnx"
        checkpoint = ROOT / "checkpoints/latest/ltwm_expansion" / f"seed{seed}/best.pt"
        if not onnx.is_file() or not checkpoint.is_file(): raise FileNotFoundError(onnx)
        builder = trt.Builder(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, logger)
        if not parser.parse(onnx.read_bytes()):
            raise RuntimeError("\n".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(args.workspace_gib * 1024**3))
        config.set_flag(trt.BuilderFlag.BF16)
        serialized = builder.build_serialized_network(network, config)
        if serialized is None: raise RuntimeError("TensorRT build failed")
        engine = args.output_dir / f"seed{seed}_manualgrid_bf16.engine"
        engine.write_bytes(bytes(serialized))
        records.append({"seed": seed, "path": engine.name, "sha256": digest(engine),
                        "checkpoint_sha256": digest(checkpoint)})
    manifest = {"schema": "v70_ltwm_tensorrt_v1", "engines": records,
                "validation": {"status": "compare_current_pytorch_and_trt_plans_before_live"}}
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("Built engines; compare identical saved RGB/pose inputs before enabling live TensorRT")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
