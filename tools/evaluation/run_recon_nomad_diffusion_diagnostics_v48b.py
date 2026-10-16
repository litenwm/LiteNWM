#!/usr/bin/env python3
"""Run v48 with probe timesteps scaled to the checkpoint diffusion horizon."""

from __future__ import annotations

from pathlib import Path


def main() -> None:
    target = Path(__file__).with_name(
        "export_recon_nomad_diffusion_diagnostics_v48.py"
    )
    source = target.read_text(encoding="utf-8")
    old = 'timesteps = np.asarray((10, 25, 50, 75, 90), dtype=np.int64)'
    new = '''maximum_timestep = int(config["num_diffusion_iters"]) - 1
    timesteps = np.asarray(
        [
            max(0, min(maximum_timestep, round(value * maximum_timestep)))
            for value in (0.10, 0.25, 0.50, 0.75, 0.90)
        ], dtype=np.int64,
    )'''
    if source.count(old) != 1:
        raise RuntimeError("v48 timestep source contract changed")
    source = source.replace(old, new, 1).replace(
        "recon_nomad_diffusion_diagnostics_v48",
        "recon_nomad_diffusion_diagnostics_v48b",
    )
    exec(
        compile(source, str(target), "exec"),
        {"__name__": "__main__", "__file__": str(target)},
    )


if __name__ == "__main__":
    main()
