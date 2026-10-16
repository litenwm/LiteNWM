#!/usr/bin/env python3
"""Isolated Unitree camera client for V70 live navigation.

This process deliberately imports no rclpy module. It keeps Unitree SDK's
CycloneDDS participant separate from the ROS 2 CycloneDDS participant and
forwards the original JPEG bytes over an inherited Unix-domain socket.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import signal
import socket
import struct
import time
from pathlib import Path


def jpeg_structure_error(payload: bytes) -> str | None:
    """Return a compact reason when a JPEG marker stream is incomplete."""
    if len(payload) < 4:
        return "jpeg_too_short"
    if payload[:2] != b"\xff\xd8":
        return "missing_soi"
    if payload[-2:] != b"\xff\xd9":
        return "missing_terminal_eoi"

    position = 2
    saw_scan = False
    standalone = {0x01, 0xD8, 0xD9, *range(0xD0, 0xD8)}
    while position < len(payload):
        if payload[position] != 0xFF:
            return f"unexpected_data_before_marker_at_{position}"
        marker_start = position
        while position < len(payload) and payload[position] == 0xFF:
            position += 1
        if position >= len(payload):
            return "unterminated_marker_prefix"
        marker = payload[position]
        position += 1
        if marker == 0x00:
            return f"stuffed_zero_outside_scan_at_{marker_start}"
        if marker == 0xD9:
            if position != len(payload):
                return f"trailing_bytes_after_eoi_at_{position}"
            if not saw_scan:
                return "missing_scan"
            return None
        if marker in standalone:
            continue
        if position + 2 > len(payload):
            return f"missing_segment_length_at_{marker_start}"
        segment_length = int.from_bytes(
            payload[position : position + 2],
            byteorder="big",
        )
        if segment_length < 2:
            return f"invalid_segment_length_at_{marker_start}"
        segment_end = position + segment_length
        if segment_end > len(payload):
            return f"truncated_segment_at_{marker_start}"
        if marker != 0xDA:
            position = segment_end
            continue

        saw_scan = True
        position = segment_end
        while position < len(payload):
            marker_start = payload.find(b"\xff", position)
            if marker_start < 0:
                return "scan_has_no_terminal_marker"
            marker_position = marker_start + 1
            while (
                marker_position < len(payload)
                and payload[marker_position] == 0xFF
            ):
                marker_position += 1
            if marker_position >= len(payload):
                return "unterminated_scan_marker"
            marker = payload[marker_position]
            if marker == 0x00 or 0xD0 <= marker <= 0xD7:
                position = marker_position + 1
                continue
            position = marker_start
            break
    return "missing_eoi"


def recover_trailing_jpeg_bytes(payload: bytes) -> tuple[bytes, int] | None:
    """Trim bytes after a structurally valid EOI marker.

    The Go2 camera occasionally appends a few bytes from the transport buffer
    after an otherwise complete JPEG.  The marker parser has already proved
    where that JPEG ends, so keeping the validated prefix avoids an unnecessary
    camera-worker restart without accepting a truncated or malformed frame.
    """
    error = jpeg_structure_error(payload)
    prefix = "trailing_bytes_after_eoi_at_"
    if error is None or not error.startswith(prefix):
        return None
    try:
        end = int(error[len(prefix) :])
    except ValueError:
        return None
    if end < 4 or end >= len(payload):
        return None
    recovered = payload[:end]
    if jpeg_structure_error(recovered) is not None:
        return None
    return recovered, len(payload) - end


def archive_corrupt_jpeg(
    payload: bytes,
    *,
    directory: Path,
    limit: int,
    reason: str,
    interface: str,
) -> dict[str, object]:
    digest = hashlib.sha256(payload).hexdigest()
    metadata: dict[str, object] = {
        "schema": "v70_corrupt_camera_frame_v1",
        "time_unix": time.time(),
        "time_monotonic": time.monotonic(),
        "length_bytes": len(payload),
        "sha256": digest,
        "starts_with_soi": payload.startswith(b"\xff\xd8"),
        "ends_with_eoi": payload.endswith(b"\xff\xd9"),
        "reason": reason,
        "interface": interface,
        "archived": False,
    }
    directory.mkdir(parents=True, exist_ok=True)
    if len(list(directory.glob("*.json"))) >= limit:
        metadata["archive_reason"] = "archive_limit_reached"
        return metadata
    stem = f"bad_{time.time_ns()}_{digest[:12]}"
    raw_path = directory / f"{stem}.jpg"
    metadata_path = directory / f"{stem}.json"
    raw_path.write_bytes(payload)
    metadata["archived"] = True
    metadata["raw_path"] = str(raw_path)
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fd", required=True, type=int)
    parser.add_argument("--interface", default="enP8p1s0")
    parser.add_argument("--interval", type=float, default=0.20)
    parser.add_argument("--client-timeout", type=float, default=1.0)
    parser.add_argument("--corrupt-dir", required=True, type=Path)
    parser.add_argument("--max-corrupt-frames", type=int, default=32)
    return parser.parse_args()


def send_message(
    channel: socket.socket,
    kind: int,
    receive_time: float,
    payload: bytes,
) -> None:
    channel.sendall(
        struct.pack("!BdI", kind, receive_time, len(payload))
        + payload
    )


def main() -> int:
    args = parse_args()
    if not 0.02 <= args.interval <= 5.0:
        raise ValueError("--interval must be in [0.02, 5.0]")
    if not 0.10 <= args.client_timeout <= 3.0:
        raise ValueError("--client-timeout must be in [0.10, 3.0]")
    if args.max_corrupt_frames <= 0:
        raise ValueError("--max-corrupt-frames must be positive")
    corrupt_dir = args.corrupt_dir.expanduser().resolve()
    corrupt_dir.mkdir(parents=True, exist_ok=True)
    stop_requested = False

    def request_stop(_signum, _frame) -> None:
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    channel = socket.socket(fileno=args.fd)
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    from unitree_sdk2py.go2.video.video_client import VideoClient

    ChannelFactoryInitialize(0, args.interface)
    client = VideoClient()
    # Keep SDK receive latency below the parent/launcher shutdown deadline.
    client.SetTimeout(args.client_timeout)
    client.Init()

    while not stop_requested:
        started = time.monotonic()
        try:
            code, data = client.GetImageSample()
            if code != 0:
                raise RuntimeError(
                    f"GetImageSample failed: code={code}"
                )
            payload = bytes(data)
            if not payload:
                raise RuntimeError("camera returned empty JPEG")
            structure_error = jpeg_structure_error(payload)
            recovered = recover_trailing_jpeg_bytes(payload)
            if recovered is not None:
                payload, _trimmed_bytes = recovered
                structure_error = None
            if structure_error is not None:
                metadata = archive_corrupt_jpeg(
                    payload,
                    directory=corrupt_dir,
                    limit=args.max_corrupt_frames,
                    reason=structure_error,
                    interface=args.interface,
                )
                send_message(
                    channel,
                    2,
                    time.monotonic(),
                    (
                        "corrupt_camera_jpeg:"
                        f"{structure_error}:"
                        f"length={len(payload)}:"
                        f"sha256={metadata['sha256']}:"
                        f"archived={metadata['archived']}"
                    ).encode("utf-8"),
                )
                continue
            send_message(channel, 1, time.monotonic(), payload)
        except Exception as exc:
            try:
                send_message(
                    channel,
                    2,
                    time.monotonic(),
                    f"{type(exc).__name__}: {exc}".encode("utf-8"),
                )
            except OSError:
                break
        remaining = args.interval - (time.monotonic() - started)
        if remaining > 0.0:
            time.sleep(remaining)
    channel.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
