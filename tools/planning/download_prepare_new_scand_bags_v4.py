#!/usr/bin/env python3
"""SCAND downloader using Windows curl transport and WSL-side verification.

WSL Python TLS handshakes to both TDR and Hugging Face time out on this host,
while Windows curl can access the public mirror.  This entrypoint keeps the
v3 frozen official metadata snapshot and v2's mandatory official MD5 checks,
but transports bytes with Windows curl.exe into the WSL quarantine directory.
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from planning.proposal_tail_sidecar_v1 import canonical_json  # noqa: E402
from tools.planning import download_prepare_new_scand_bags_v2 as v2  # noqa: E402
from tools.planning import download_prepare_new_scand_bags_v3 as v3  # noqa: E402


SCHEMA = "download_prepare_new_scand_bags_v4"
WINDOWS_CURL = Path("/mnt/c/Windows/System32/curl.exe")
SMOKE_FLAG = "--transport-smoke-only"


def windows_unc_path(path: Path) -> str:
    resolved = path.resolve()
    v2.require(resolved.is_absolute(), "Windows transport path must be absolute")
    distro = os.environ.get("WSL_DISTRO_NAME", "Ubuntu-22.04").strip()
    v2.require(distro and "\\" not in distro and "/" not in distro, "invalid WSL distro name")
    return "\\\\wsl.localhost\\" + distro + "\\" + "\\".join(resolved.parts[1:])


def run_windows_curl(arguments: list[str]) -> None:
    v2.require(WINDOWS_CURL.is_file(), f"Windows curl is missing: {WINDOWS_CURL}")
    subprocess.run([str(WINDOWS_CURL), *arguments], check=True)


def windows_curl_download_one(
    url: str,
    destination: Path,
    expected_size: int,
    timeout: float,
    retries: int,
) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        v2.require(
            destination.stat().st_size == expected_size,
            f"existing bag size mismatch: {destination}",
        )
        return {"resumed": False, "already_complete": True, "transport": "windows_curl"}
    partial = destination.with_suffix(destination.suffix + ".part")
    existing = partial.stat().st_size if partial.exists() else 0
    run_windows_curl(
        [
            "--fail",
            "--location",
            "--retry",
            str(retries),
            "--retry-all-errors",
            "--connect-timeout",
            str(max(5, int(round(timeout)))),
            "--speed-time",
            "60",
            "--speed-limit",
            "1024",
            "--continue-at",
            "-",
            "--output",
            windows_unc_path(partial),
            url,
        ]
    )
    v2.require(partial.is_file(), f"Windows curl produced no partial file: {partial}")
    v2.require(
        partial.stat().st_size == expected_size,
        f"downloaded size mismatch: {partial}",
    )
    partial.replace(destination)
    return {
        "resumed": bool(existing),
        "already_complete": False,
        "transport": "windows_curl",
    }


def transport_smoke() -> None:
    catalog, audit = v3.fetch_tdr_catalog_from_snapshot(0)
    label = "A_Jackal_GDC_GDC_Fri_Oct_29_11.bag"
    v2.require(label in catalog, "smoke bag is absent from frozen official metadata")
    destination = (
        ROOT
        / "data/proposal_tail_collection_v1/incoming/quarantine/official_scand_selective_v1"
        / "windows_curl_transport_smoke.byte"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    try:
        run_windows_curl(
            [
                "--fail",
                "--location",
                "--range",
                "0-0",
                "--connect-timeout",
                "60",
                "--max-time",
                "90",
                "--output",
                windows_unc_path(destination),
                v2.mirror_url(label),
            ]
        )
        v2.require(destination.is_file() and destination.stat().st_size == 1, "transport smoke did not return one byte")
    finally:
        if destination.exists():
            destination.unlink()
    print(
        canonical_json(
            {
                "schema": SCHEMA + "_transport_smoke",
                "status": "pass",
                "bytes_requested": 1,
                "official_file_id": catalog[label]["file_id"],
                "official_metadata_snapshot_sha256": audit["metadata_snapshot_sha256"],
                "mirror_commit": v2.HF_MIRROR_COMMIT,
                "large_download_started": False,
                "reserve_content_accessed": False,
            }
        )
    )


def main() -> None:
    if SMOKE_FLAG in sys.argv:
        sys.argv.remove(SMOKE_FLAG)
        transport_smoke()
        return
    v2.fetch_tdr_catalog = v3.fetch_tdr_catalog_from_snapshot
    v2._ORIGINAL_DOWNLOAD_ONE = windows_curl_download_one
    v2.main()


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(
            canonical_json(
                {
                    "schema": SCHEMA + "_failure",
                    "status": "fail_closed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "network_may_have_started": "--execute" in sys.argv or SMOKE_FLAG in sys.argv,
                    "reserve_content_accessed": False,
                }
            ),
            file=sys.stderr,
        )
        raise SystemExit(2) from error
