"""Run incoming-route assertions when pytest is unavailable."""

from pathlib import Path
import tempfile

from tests.planning.test_incoming_route_preflight_v1 import (
    test_development_validation_route_overlap_fails_closed,
    test_forbidden_path_token_fails_closed,
    test_incomplete_forbidden_registry_fails_closed,
    test_missing_timestamp_fails_closed,
    test_opened_label_or_changed_image_fails_closed,
    test_used_identity_overlap_fails_closed,
    test_valid_disjoint_prelabel_input_is_ready,
)


def run(function) -> None:
    function(Path(tempfile.mkdtemp(prefix=function.__name__ + "_")))


def main() -> None:
    checks = (
        test_valid_disjoint_prelabel_input_is_ready,
        test_forbidden_path_token_fails_closed,
        test_missing_timestamp_fails_closed,
        test_used_identity_overlap_fails_closed,
        test_incomplete_forbidden_registry_fails_closed,
        test_development_validation_route_overlap_fails_closed,
        test_opened_label_or_changed_image_fails_closed,
    )
    for check in checks:
        run(check)
    print(f"{len(checks)} passed (manual; fail-closed incoming route checks)")


if __name__ == "__main__":
    main()
