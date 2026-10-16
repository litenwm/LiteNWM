"""Load trusted checkpoints with portable path metadata, without global patches."""
from __future__ import annotations
import os
import pathlib
import pickle
import types
import torch


class _PortableUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        if module == "pathlib":
            if name == "PosixPath" and os.name != "posix":
                return pathlib.PurePosixPath
            if name == "WindowsPath" and os.name != "nt":
                return pathlib.PureWindowsPath
        return super().find_class(module, name)


_PICKLE = types.ModuleType("litenwm_checkpoint_pickle")
_PICKLE.Unpickler = _PortableUnpickler
_PICKLE.Pickler = pickle.Pickler
_PICKLE.load = pickle.load
_PICKLE.loads = pickle.loads
_PICKLE.dump = pickle.dump
_PICKLE.dumps = pickle.dumps


def load_checkpoint(path, **kwargs):
    """Full pickle load is for SHA-verified, trusted author checkpoints only."""
    if kwargs.get("weights_only", True):
        return torch.load(path, **kwargs)
    kwargs.setdefault("pickle_module", _PICKLE)
    return torch.load(path, **kwargs)
