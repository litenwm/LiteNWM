#!/usr/bin/env python3
"""Open an arbitrary number of preregistered development banks, not reserve."""

from __future__ import annotations

from pathlib import Path


def main() -> None:
    target = Path(__file__).with_name(
        "open_recon_target_calibration_bank_v20.py"
    )
    source = target.read_text(encoding="utf-8")
    if source.count("nargs=3") != 3:
        raise RuntimeError("development opener source contract changed")
    source = source.replace("nargs=3", 'nargs="+"')
    source = source.replace(
        "recon_target_calibration_bank_v20_outcome_open",
        "recon_target_development_v44_outcome_open",
    ).replace(
        "development_bank_labels_opened_reserve_sealed",
        "expanded_development_bank_labels_opened_reserve_sealed",
    )
    exec(
        compile(source, str(target), "exec"),
        {"__name__": "__main__", "__file__": str(target)},
    )


if __name__ == "__main__":
    main()
