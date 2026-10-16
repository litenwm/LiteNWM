from __future__ import annotations

import json
import sys
from pathlib import Path

DEPLOY_DIR = (
    Path(__file__).resolve().parents[2]
    / "deploy"
    / "v70_go2_orin"
)
sys.path.insert(0, str(DEPLOY_DIR))

from v70_unitree_camera_worker import (  # noqa: E402
    archive_corrupt_jpeg,
    jpeg_structure_error,
)


def structurally_complete_jpeg() -> bytes:
    return (
        b"\xff\xd8"
        b"\xff\xda\x00\x02"
        b"entropy\xff\x00data"
        b"\xff\xd9"
    )


def test_complete_marker_stream_is_accepted() -> None:
    assert jpeg_structure_error(structurally_complete_jpeg()) is None


def test_truncated_and_structurally_invalid_frames_are_rejected() -> None:
    complete = structurally_complete_jpeg()
    assert jpeg_structure_error(complete[:-2]) == "missing_terminal_eoi"
    invalid = b"\xff\xd8garbage\xff\xd9"
    assert "unexpected_data_before_marker" in str(
        jpeg_structure_error(invalid)
    )


def test_corrupt_archive_records_hash_length_markers_and_limit(
    tmp_path: Path,
) -> None:
    payload = structurally_complete_jpeg()[:-2]
    first = archive_corrupt_jpeg(
        payload,
        directory=tmp_path,
        limit=1,
        reason="missing_terminal_eoi",
        interface="enP8p1s0",
    )
    assert first["archived"] is True
    metadata = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert metadata["length_bytes"] == len(payload)
    assert metadata["starts_with_soi"] is True
    assert metadata["ends_with_eoi"] is False
    assert len(metadata["sha256"]) == 64

    second = archive_corrupt_jpeg(
        payload + b"x",
        directory=tmp_path,
        limit=1,
        reason="missing_terminal_eoi",
        interface="enP8p1s0",
    )
    assert second["archived"] is False
    assert second["archive_reason"] == "archive_limit_reached"
