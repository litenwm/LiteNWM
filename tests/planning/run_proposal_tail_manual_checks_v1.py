"""Run proposal-tail assertions when pytest is unavailable."""

from pathlib import Path
import tempfile

from tests.planning.test_proposal_tail_sampler_legacy_parity_v1 import (
    test_logger_off_matches_preobserver_sampler_bitwise,
)
from tests.planning.test_proposal_tail_sidecar_serialization_v1 import (
    test_live_sidecar_serializes_without_mutating_formal_arrays,
)
from tests.planning.test_proposal_tail_sidecar_v1 import (
    test_observer_records_prefilter_rejection_fields,
    test_raw_draw_observer_is_bitwise_transparent,
)


def main() -> None:
    test_logger_off_matches_preobserver_sampler_bitwise()
    test_raw_draw_observer_is_bitwise_transparent()
    test_observer_records_prefilter_rejection_fields()
    test_live_sidecar_serializes_without_mutating_formal_arrays(
        Path(tempfile.mkdtemp(prefix="proposal_tail_sidecar_v1_"))
    )
    print("4 passed (manual; pytest unavailable)")


if __name__ == "__main__":
    main()
