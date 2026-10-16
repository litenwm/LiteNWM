from pathlib import PurePosixPath, PosixPath
import io
import os
import torch
from planning.checkpoint_io import load_checkpoint


class LinuxMetadata:
    def __reduce__(self):
        return PosixPath, ("/home/unitree/weights/best.pt",)


def test_linux_checkpoint_metadata_loads_without_mutating_pathlib():
    stream = io.BytesIO()
    torch.save({"path": LinuxMetadata(), "model": {"weight": torch.tensor([1.0, 2.0])}}, stream)
    stream.seek(0)
    result = load_checkpoint(stream, map_location="cpu", weights_only=False)
    assert str(result["path"]) == "/home/unitree/weights/best.pt"
    assert torch.equal(result["model"]["weight"], torch.tensor([1.0, 2.0]))
    if os.name == "nt": assert type(result["path"]) is PurePosixPath
    import pathlib
    assert pathlib.PosixPath is PosixPath
