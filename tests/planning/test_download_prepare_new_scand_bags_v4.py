from __future__ import annotations

from pathlib import Path

from tools.planning.download_prepare_new_scand_bags_v4 import windows_unc_path


def test_windows_unc_path_targets_the_same_wsl_file(monkeypatch) -> None:
    monkeypatch.setenv("WSL_DISTRO_NAME", "Ubuntu-22.04")
    value = windows_unc_path(Path("/home/rainbow/example.part"))
    assert value == r"\\wsl.localhost\Ubuntu-22.04\home\rainbow\example.part"
