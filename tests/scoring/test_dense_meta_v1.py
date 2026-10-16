from __future__ import annotations

import numpy as np

from scoring.dense_meta_v1 import feature_by_mode


def _data() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(17)
    return {
        "action_features": rng.normal(size=(3, 5, 2)).astype(np.float32),
        "static_features": rng.normal(size=(3, 5, 3)).astype(np.float32),
        "raw_ltwm_features": rng.normal(size=(3, 5, 4)).astype(np.float32),
        "scorer_v2_aux_features": rng.normal(size=(3, 5, 2)).astype(np.float32),
        "support_features": rng.normal(size=(3, 5, 1)).astype(np.float32),
    }


def test_raw_and_stacked_feature_contracts_are_distinct() -> None:
    data = _data()
    raw = feature_by_mode(data, "full", variant="raw")
    stacked = feature_by_mode(data, "full", variant="v2_stacked")
    assert raw.shape == (3, 5, 13)
    assert stacked.shape == (3, 5, 17)
    changed = {**data, "scorer_v2_aux_features": data["scorer_v2_aux_features"] + 100.0}
    np.testing.assert_array_equal(raw, feature_by_mode(changed, "full", variant="raw"))
    assert not np.array_equal(stacked, feature_by_mode(changed, "full", variant="v2_stacked"))


def test_support_is_never_a_learned_feature() -> None:
    data = _data()
    changed = {**data, "support_features": data["support_features"] + 1000.0}
    for variant in ("raw", "v2_stacked"):
        for mode in ("action", "static", "latent", "full"):
            np.testing.assert_array_equal(
                feature_by_mode(data, mode, variant=variant),
                feature_by_mode(changed, mode, variant=variant),
            )


def test_action_and_static_heads_do_not_depend_on_scorer_auxiliary() -> None:
    data = _data()
    changed = {**data, "scorer_v2_aux_features": -data["scorer_v2_aux_features"]}
    for mode in ("action", "static"):
        np.testing.assert_array_equal(
            feature_by_mode(data, mode, variant="v2_stacked"),
            feature_by_mode(changed, mode, variant="v2_stacked"),
        )
