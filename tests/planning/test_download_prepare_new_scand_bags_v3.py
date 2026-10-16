from __future__ import annotations

from tools.planning.download_prepare_new_scand_bags_v3 import (
    TDR_METADATA_SNAPSHOT,
    fetch_tdr_catalog_from_snapshot,
)


def test_frozen_official_snapshot_contains_the_frozen_18() -> None:
    catalog, audit = fetch_tdr_catalog_from_snapshot(0)
    assert TDR_METADATA_SNAPSHOT.is_file()
    assert len(catalog) == 18
    assert audit["official_bag_count"] == 18
    assert audit["live_tdr_request_required"] is False
    assert catalog["A_Jackal_GDC_GDC_Fri_Oct_29_11.bag"]["file_id"] == 133233
    assert catalog["A_Jackal_GDC_GDC_Fri_Oct_29_11.bag"]["checksum"] == (
        "9c94d9fac1ecf4dce6dba383ddac61ac"
    )
