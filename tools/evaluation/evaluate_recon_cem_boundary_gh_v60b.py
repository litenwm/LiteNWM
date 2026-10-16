#!/usr/bin/env python3
"""Execute the v60 one-time evaluator with its atomic-output mkdir fix."""

from pathlib import Path


def main() -> None:
    source_path = Path(__file__).with_name("evaluate_recon_cem_boundary_gh_v60.py")
    source = source_path.read_text(encoding="utf-8")
    old = "output_root.mkdir(parents=True, exist_ok=False)"
    new = "output_root.mkdir(parents=True, exist_ok=True)"
    if source.count(old) != 1:
        raise RuntimeError("v60 evaluator source contract changed")
    namespace = {
        "__file__": str(source_path),
        "__name__": "tools.evaluation._evaluate_recon_cem_boundary_gh_v60_fixed",
        "__package__": "tools.evaluation",
    }
    exec(compile(source.replace(old, new), str(source_path), "exec"), namespace)
    namespace["main"]()


if __name__ == "__main__":
    main()
