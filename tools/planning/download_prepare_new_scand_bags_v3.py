#!/usr/bin/env python3
"""Run the checksum-verified SCAND downloader from a frozen TDR snapshot.

Windows can read the public TDR metadata API, while the WSL Python TLS
connection to that metadata endpoint times out on this host.  The selected 18
file records were therefore fetched once from the official API and frozen in a
versioned local JSON snapshot.  This entrypoint keeps v2's commit-pinned mirror
transport and mandatory official size/MD5 verification, while eliminating the
unreliable live TDR request from WSL.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from planning.proposal_tail_sidecar_v1 import canonical_json  # noqa: E402
from tools.planning import download_prepare_new_scand_bags_v2 as v2  # noqa: E402


SCHEMA = "download_prepare_new_scand_bags_v3"
TDR_METADATA_SNAPSHOT = (
    ROOT / "configs/proposal_tail_collection/scand_tdr_selected_files_v1.json"
)


def fetch_tdr_catalog_from_snapshot(
    timeout: float,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    del timeout
    snapshot = v2.v1.read_json(TDR_METADATA_SNAPSHOT)
    v2.require(snapshot.get("schema") == "scand_tdr_selected_files_v1", "wrong TDR snapshot schema")
    v2.require(snapshot.get("dataset_doi") == v2.TDR_DATASET_DOI, "wrong TDR snapshot DOI")
    v2.require(snapshot.get("dataset_api") == v2.TDR_DATASET_API, "wrong TDR snapshot API provenance")
    rows = snapshot.get("files")
    v2.require(isinstance(rows, list), "TDR snapshot lacks file list")
    payload = {
        "status": "OK",
        "data": {
            "latestVersion": {
                "id": snapshot.get("dataset_version_id"),
                "files": [
                    {
                        "label": row["label"],
                        "restricted": False,
                        "directoryLabel": row.get("directory_label"),
                        "dataFile": {
                            "id": row["file_id"],
                            "filesize": row["filesize"],
                            "persistentId": row.get("persistent_id"),
                            "checksum": {
                                "type": row["checksum_type"],
                                "value": row["checksum"],
                            },
                        },
                    }
                    for row in rows
                ],
            }
        },
    }
    catalog = v2.parse_tdr_catalog(payload)
    return catalog, {
        "dataset_doi": v2.TDR_DATASET_DOI,
        "dataset_api": v2.TDR_DATASET_API,
        "dataset_version_id": snapshot.get("dataset_version_id"),
        "metadata_snapshot": str(TDR_METADATA_SNAPSHOT),
        "metadata_snapshot_sha256": v2.v1.sha256_file(TDR_METADATA_SNAPSHOT),
        "official_metadata_retrieved_at": snapshot.get("official_metadata_retrieved_at"),
        "official_bag_count": len(catalog),
        "live_tdr_request_required": False,
    }


def main() -> None:
    v2.fetch_tdr_catalog = fetch_tdr_catalog_from_snapshot
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
                    "network_may_have_started": "--execute" in sys.argv,
                    "reserve_content_accessed": False,
                }
            ),
            file=sys.stderr,
        )
        raise SystemExit(2) from error
