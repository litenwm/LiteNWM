#!/usr/bin/env python3
"""Download the frozen SCAND bag plan through a checksum-verified mirror.

The v1 downloader's Google-Sheet-era Box URLs now return HTTP 404. The
official Texas Data Repository (TDR) remains the authority for exact file
identity, size, and MD5, but its large-file CloudFront endpoint returns HTTP
403 from the current network. This compatibility entrypoint therefore:

1. reads the public official TDR metadata API;
2. resolves every already-frozen bag by exact filename;
3. downloads bytes from one commit-pinned public mirror; and
4. rejects every file whose byte count or MD5 differs from official TDR.

All identity filtering, route splitting, disk guards, camera-only decoding,
320x240/JPEG95/4 Hz processing, readiness checks, and no-label/no-reserve
contracts continue to come from v1 unchanged.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping
import urllib.parse


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from planning.proposal_tail_sidecar_v1 import canonical_json  # noqa: E402
from tools.planning import download_prepare_new_scand_bags_v1 as v1  # noqa: E402


SCHEMA = "download_prepare_new_scand_bags_v2"
TDR_HOST = "dataverse.tdl.org"
TDR_DATASET_DOI = "doi:10.18738/T8/0PRYRH"
TDR_DATASET_API = (
    "https://dataverse.tdl.org/api/datasets/:persistentId/"
    "?persistentId=doi%3A10.18738%2FT8%2F0PRYRH"
)
HF_MIRROR_REPO = "franciszzj/SCAND"
HF_MIRROR_COMMIT = "cf417761d7d8701488cd28f86e2d99c218221e5d"
HF_MIRROR_HOST = "huggingface.co"

_CATALOG: dict[str, dict[str, Any]] = {}
_ORIGINAL_DOWNLOAD_ONE = v1.download_one
_ORIGINAL_PROCESS_CAMERA_ONLY = v1.process_camera_only
_ORIGINAL_EXECUTE = v1.execute


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def mirror_url(file_name: str) -> str:
    require(file_name.endswith(".bag"), "mirror filename must end in .bag")
    require("/" not in file_name and "\\" not in file_name, "unsafe mirror filename")
    quoted = urllib.parse.quote(file_name, safe="")
    value = (
        f"https://{HF_MIRROR_HOST}/datasets/{HF_MIRROR_REPO}/resolve/"
        f"{HF_MIRROR_COMMIT}/random_mdps/{quoted}?download=true"
    )
    parsed = urllib.parse.urlparse(value)
    expected_prefix = (
        f"/datasets/{HF_MIRROR_REPO}/resolve/{HF_MIRROR_COMMIT}/random_mdps/"
    )
    require(parsed.scheme == "https" and parsed.hostname == HF_MIRROR_HOST, "invalid mirror host")
    require(parsed.path.startswith(expected_prefix), "mirror URL is not commit-pinned")
    require(parsed.path.endswith(".bag"), "mirror URL is not a bag")
    require(urllib.parse.parse_qs(parsed.query) == {"download": ["true"]}, "invalid mirror query")
    require(not parsed.username and not parsed.password, "credentials in mirror URL are forbidden")
    return parsed.geturl()


def parse_tdr_catalog(payload: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    require(payload.get("status") == "OK", "official TDR API did not return OK")
    data = payload.get("data")
    require(isinstance(data, Mapping), "official TDR API lacks data object")
    latest = data.get("latestVersion")
    require(isinstance(latest, Mapping), "official TDR API lacks latestVersion")
    files = latest.get("files")
    require(isinstance(files, list), "official TDR API lacks file list")
    catalog: dict[str, dict[str, Any]] = {}
    for wrapper in files:
        require(isinstance(wrapper, Mapping), "invalid official TDR file wrapper")
        label = str(wrapper.get("label", ""))
        if not label.endswith(".bag"):
            continue
        require(label not in catalog, f"duplicate official TDR bag label: {label}")
        require(wrapper.get("restricted") is False, f"official TDR bag is restricted: {label}")
        data_file = wrapper.get("dataFile")
        require(isinstance(data_file, Mapping), f"official TDR bag lacks dataFile: {label}")
        checksum = data_file.get("checksum")
        require(isinstance(checksum, Mapping), f"official TDR bag lacks checksum: {label}")
        checksum_type = str(checksum.get("type", "")).upper()
        checksum_value = str(checksum.get("value", "")).lower()
        require(checksum_type == "MD5", f"official checksum is not MD5: {label}")
        require(
            len(checksum_value) == 32
            and all(character in "0123456789abcdef" for character in checksum_value),
            f"invalid official MD5: {label}",
        )
        file_id = int(data_file.get("id", 0))
        file_size = int(data_file.get("filesize", 0))
        require(file_id > 0 and file_size > 0, f"invalid official TDR metadata: {label}")
        catalog[label] = {
            "file_id": file_id,
            "filesize": file_size,
            "checksum_type": checksum_type,
            "checksum": checksum_value,
            "persistent_id": data_file.get("persistentId"),
            "directory_label": wrapper.get("directoryLabel"),
        }
    require(catalog, "official TDR API contains no bag files")
    return catalog


def fetch_tdr_catalog(timeout: float) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    parsed = urllib.parse.urlparse(TDR_DATASET_API)
    require(parsed.scheme == "https" and parsed.hostname == TDR_HOST, "invalid frozen TDR API URL")
    with v1.request(TDR_DATASET_API, timeout=timeout) as response:
        payload = json.load(response)
    require(isinstance(payload, Mapping), "official TDR API response is not an object")
    catalog = parse_tdr_catalog(payload)
    latest = payload["data"]["latestVersion"]
    return catalog, {
        "dataset_doi": TDR_DATASET_DOI,
        "dataset_api": TDR_DATASET_API,
        "dataset_version_id": latest.get("id"),
        "dataset_version": latest.get("versionNumber"),
        "dataset_version_minor": latest.get("versionMinorNumber"),
        "official_bag_count": len(catalog),
    }


def md5_file(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metadata_for_path(path: Path) -> dict[str, Any]:
    label = path.name
    require(label in _CATALOG, f"download destination is absent from official TDR: {label}")
    return _CATALOG[label]


def remote_size_from_tdr(legacy_url: str, timeout: float) -> int:
    del timeout
    matches = [
        metadata
        for metadata in _CATALOG.values()
        if metadata["legacy_box_download_url"] == legacy_url
    ]
    require(len(matches) == 1, "legacy Box URL did not resolve uniquely through official TDR")
    return int(matches[0]["filesize"])


def download_one_verified(
    legacy_url: str,
    destination: Path,
    expected_size: int,
    timeout: float,
    retries: int,
) -> dict[str, Any]:
    metadata = _metadata_for_path(destination)
    require(metadata["legacy_box_download_url"] == legacy_url, "legacy URL/file mismatch")
    require(int(metadata["filesize"]) == expected_size, "official TDR size changed")
    transport_url = mirror_url(destination.name)
    print(
        canonical_json(
            {
                "status": "downloading_bag",
                "official_file_name": destination.name,
                "bytes": expected_size,
                "transport": "commit_pinned_mirror_official_tdr_md5_required",
            }
        ),
        file=sys.stderr,
        flush=True,
    )
    result = _ORIGINAL_DOWNLOAD_ONE(
        transport_url,
        destination,
        expected_size,
        timeout,
        retries,
    )
    require(
        md5_file(destination) == metadata["checksum"],
        f"downloaded bag differs from official TDR MD5: {destination}",
    )
    return {
        **result,
        "transport": "commit_pinned_public_mirror",
        "mirror_repo": HF_MIRROR_REPO,
        "mirror_commit": HF_MIRROR_COMMIT,
        "official_tdr_file_id": metadata["file_id"],
        "official_tdr_md5": metadata["checksum"],
        "official_md5_verified": True,
    }


def process_camera_only_verified(
    bag_path: Path,
    destination: Path,
    row: Mapping[str, Any],
    forbidden: Mapping[str, set[str]],
) -> dict[str, Any]:
    metadata = _metadata_for_path(bag_path)
    patched_row = {**row, "download_url": mirror_url(bag_path.name)}
    result = _ORIGINAL_PROCESS_CAMERA_ONLY(bag_path, destination, patched_row, forbidden)
    route_metadata = dict(result["metadata"])
    route_metadata.update(
        {
            "source_download_transport": "commit_pinned_public_mirror_official_tdr_md5_verified",
            "source_legacy_box_download_url": row["download_url"],
            "source_official_tdr_file_id": metadata["file_id"],
            "source_official_tdr_persistent_id": metadata["persistent_id"],
            "source_official_md5": metadata["checksum"],
            "source_mirror_repo": HF_MIRROR_REPO,
            "source_mirror_commit": HF_MIRROR_COMMIT,
        }
    )
    v1.write_json_atomic(destination / "route_metadata.json", route_metadata)
    result["metadata"] = route_metadata
    return result


def execute_verified(
    args: Any,
    plan: Mapping[str, Any],
    forbidden: Mapping[str, set[str]],
    plan_path: Path,
) -> dict[str, Any]:
    global _CATALOG
    official_catalog, tdr_audit = fetch_tdr_catalog(args.timeout_seconds)
    selected_catalog: dict[str, dict[str, Any]] = {}
    source_rows = []
    for row in plan["selected"]:
        label = f"{row['official_file_name']}.bag"
        require(label in official_catalog, f"frozen bag missing from official TDR: {label}")
        metadata = {
            **official_catalog[label],
            "legacy_box_download_url": row["download_url"],
        }
        selected_catalog[label] = metadata
        source_rows.append(
            {
                "official_file_name": label,
                "bag_id": row["bag_id"],
                "cohort_role": row["cohort_role"],
                "official_tdr": official_catalog[label],
                "mirror_url": mirror_url(label),
            }
        )
    require(len(selected_catalog) == len(plan["selected"]), "source-resolution cardinality mismatch")
    _CATALOG = selected_catalog
    quarantine = args.incoming_root / "quarantine/official_scand_selective_v1"
    source_resolution_path = quarantine / "source_resolution.v2.json"
    source_resolution = {
        "schema": SCHEMA + "_source_resolution",
        "status": "resolved_before_large_download",
        "official_tdr": tdr_audit,
        "transport": {
            "kind": "commit_pinned_public_mirror",
            "repo": HF_MIRROR_REPO,
            "commit": HF_MIRROR_COMMIT,
            "trust_boundary": "mirror bytes are accepted only after official TDR size and MD5 match",
        },
        "selected": source_rows,
        "reserve_content_accessed": False,
    }
    v1.write_json_atomic(source_resolution_path, source_resolution)
    result = _ORIGINAL_EXECUTE(args, plan, forbidden, plan_path)
    return {
        **result,
        "source_resolution_v2": {
            "path": str(source_resolution_path),
            "sha256": v1.sha256_file(source_resolution_path),
            "official_md5_required": True,
        },
    }


def install_patches() -> None:
    v1.remote_size = remote_size_from_tdr
    v1.download_one = download_one_verified
    v1.process_camera_only = process_camera_only_verified
    v1.execute = execute_verified


def main() -> None:
    install_patches()
    v1.main()


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
                    "network_may_have_started": "--execute" in sys.argv,
                    "reserve_content_accessed": False,
                }
            ),
            file=sys.stderr,
        )
        raise SystemExit(2) from error
