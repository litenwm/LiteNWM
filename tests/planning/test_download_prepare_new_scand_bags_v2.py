from __future__ import annotations

import pytest

from tools.planning.download_prepare_new_scand_bags_v2 import (
    HF_MIRROR_COMMIT,
    mirror_url,
    parse_tdr_catalog,
)


def official_payload() -> dict:
    return {
        "status": "OK",
        "data": {
            "latestVersion": {
                "files": [
                    {
                        "label": "A_Spot_AHG_Library_Tue_Nov_16_118.bag",
                        "restricted": False,
                        "directoryLabel": "Bags",
                        "dataFile": {
                            "id": 138265,
                            "filesize": 2202282340,
                            "persistentId": "doi:10.18738/T8/0PRYRH/example",
                            "checksum": {
                                "type": "MD5",
                                "value": "27ce45b7b2f6e452ae74cdda1beead8d",
                            },
                        },
                    },
                    {
                        "label": "readme.pdf",
                        "restricted": False,
                        "dataFile": {"id": 1, "filesize": 10},
                    },
                ]
            }
        },
    }


def test_mirror_url_is_commit_pinned() -> None:
    url = mirror_url("A_Spot_AHG_Library_Tue_Nov_16_118.bag")
    assert HF_MIRROR_COMMIT in url
    assert "/resolve/main/" not in url
    assert url.endswith(".bag?download=true")
    with pytest.raises(RuntimeError):
        mirror_url("../unsafe.bag")


def test_tdr_catalog_is_the_file_authority() -> None:
    catalog = parse_tdr_catalog(official_payload())
    assert list(catalog) == ["A_Spot_AHG_Library_Tue_Nov_16_118.bag"]
    assert catalog["A_Spot_AHG_Library_Tue_Nov_16_118.bag"]["file_id"] == 138265
    assert catalog["A_Spot_AHG_Library_Tue_Nov_16_118.bag"]["filesize"] == 2202282340
    assert catalog["A_Spot_AHG_Library_Tue_Nov_16_118.bag"]["checksum"] == (
        "27ce45b7b2f6e452ae74cdda1beead8d"
    )


def test_tdr_catalog_rejects_restricted_or_non_md5_bags() -> None:
    restricted = official_payload()
    restricted["data"]["latestVersion"]["files"][0]["restricted"] = True
    with pytest.raises(RuntimeError):
        parse_tdr_catalog(restricted)
    wrong_checksum = official_payload()
    wrong_checksum["data"]["latestVersion"]["files"][0]["dataFile"]["checksum"]["type"] = "SHA-1"
    with pytest.raises(RuntimeError):
        parse_tdr_catalog(wrong_checksum)
