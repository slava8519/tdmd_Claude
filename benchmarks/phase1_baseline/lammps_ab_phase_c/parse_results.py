#!/usr/bin/env python3
"""Parse a Phase C results_<ts>/ directory and print a comparison table.

For each cell (pot × size) the median of three TDMD runs (from
bench_pipeline_scheduler JSON) is compared against the median of three
LAMMPS runs (from the second "Loop time of ..." line in the LAMMPS log).
"""
from __future__ import annotations

import json
import re
import statistics
import sys
from pathlib import Path

LOOP_RE = re.compile(
    r"Loop time of ([\d.]+) on \d+ procs for (\d+) steps"
)

POTENTIALS = ("morse", "eam")
SIZES = ("tiny", "small", "medium")
N_ATOMS = {"tiny": 256, "small": 4000, "medium": 32000}


def tdmd_ts_per_s(run_file: Path) -> float:
    with run_file.open() as f:
        data = json.load(f)
    return float(data["timesteps_per_s"])


def lammps_ts_per_s(log_file: Path) -> float:
    """Return ts/s from the SECOND Loop time line (measured block)."""
    matches = []
    for line in log_file.read_text().splitlines():
        m = LOOP_RE.search(line)
        if m:
            loop_time = float(m.group(1))
            n_steps = int(m.group(2))
            matches.append(n_steps / loop_time)
    if len(matches) < 2:
        raise RuntimeError(
            f"{log_file}: expected >=2 Loop time lines, got {len(matches)}"
        )
    return matches[1]


def main(results_dir: str) -> int:
    root = Path(results_dir)
    if not root.is_dir():
        print(f"not a directory: {root}", file=sys.stderr)
        return 1

    rows = []
    for pot in POTENTIALS:
        for size in SIZES:
            tdmd_vals = []
            for r in (1, 2, 3):
                f = root / f"tdmd_{pot}_{size}_run{r}.json"
                tdmd_vals.append(tdmd_ts_per_s(f))

            lmp_vals = []
            for r in (1, 2, 3):
                f = root / f"lammps_{pot}_{size}_run{r}.log"
                lmp_vals.append(lammps_ts_per_s(f))

            tdmd_med = statistics.median(tdmd_vals)
            lmp_med = statistics.median(lmp_vals)
            ratio = tdmd_med / lmp_med
            rows.append(
                dict(
                    pot=pot,
                    size=size,
                    atoms=N_ATOMS[size],
                    tdmd=tdmd_med,
                    lammps=lmp_med,
                    ratio=ratio,
                    tdmd_raw=tdmd_vals,
                    lammps_raw=lmp_vals,
                )
            )

    print(f"# Phase C results from {root.name}")
    print()
    print(
        "| Potential | Size   |  Atoms | TDMD ts/s | LAMMPS ts/s | TDMD/LAMMPS |"
    )
    print(
        "|-----------|--------|-------:|----------:|------------:|------------:|"
    )
    for r in rows:
        print(
            f"| {r['pot']:<9} | {r['size']:<6} | {r['atoms']:>6} |"
            f" {r['tdmd']:>9.0f} | {r['lammps']:>11.0f} | {r['ratio']:>10.2f}  |"
        )
    print()
    print("## Raw (all three runs per cell)")
    print()
    for r in rows:
        tdmd_str = ", ".join(f"{v:.0f}" for v in r["tdmd_raw"])
        lmp_str = ", ".join(f"{v:.0f}" for v in r["lammps_raw"])
        print(f"- {r['pot']:<5} {r['size']:<6}: TDMD=[{tdmd_str}]  LAMMPS=[{lmp_str}]")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "."))
