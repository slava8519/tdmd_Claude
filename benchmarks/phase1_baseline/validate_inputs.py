#!/usr/bin/env python3
"""Validate that TDMD and LAMMPS Phase 1 inputs have identical physics.

Reads both inputs and prints key parameters side by side for manual comparison.
"""
import re
import sys
from pathlib import Path


def parse_lammps_data(path: Path) -> dict:
    """Extract box dimensions and atom count from a LAMMPS data file."""
    info = {"file": str(path), "n_atoms": 0, "box": {}}
    with open(path) as f:
        for line in f:
            line = line.strip()
            m = re.match(r"(\d+)\s+atoms", line)
            if m:
                info["n_atoms"] = int(m.group(1))
            for dim in ("xlo xhi", "ylo yhi", "zlo zhi"):
                if dim in line:
                    parts = line.split()
                    info["box"][dim] = (float(parts[0]), float(parts[1]))
    return info


def parse_lammps_input(path: Path) -> dict:
    """Extract key parameters from a LAMMPS input script."""
    info = {"file": str(path)}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or not line:
                continue
            # pair_coeff 1 1 0.3429 1.3588 2.866
            if line.startswith("pair_coeff"):
                parts = line.split()
                if len(parts) >= 5:
                    info["morse_D"] = float(parts[3])
                    info["morse_alpha"] = float(parts[4])
                    info["morse_r0"] = float(parts[5]) if len(parts) > 5 else "N/A"
            # pair_style morse 6.0
            if line.startswith("pair_style"):
                parts = line.split()
                if len(parts) >= 3:
                    info["morse_rc"] = float(parts[2])
            # timestep 0.001
            if line.startswith("timestep"):
                info["dt"] = float(line.split()[1])
            # neighbor 1.0 bin
            if line.startswith("neighbor"):
                info["skin"] = float(line.split()[1])
            # velocity all create 300.0 42
            if line.startswith("velocity"):
                parts = line.split()
                if "create" in parts:
                    idx = parts.index("create")
                    info["T_init"] = float(parts[idx + 1])
                    info["seed"] = int(parts[idx + 2])
            # run 1000
            if line.startswith("run"):
                info["n_steps"] = int(line.split()[1])
    return info


def main():
    base = Path(__file__).parent

    # TDMD data file (small).
    tdmd_data = base / "small.data"
    lammps_input = base / "lammps_small" / "in.baseline"

    if not tdmd_data.exists():
        print(f"Warning: {tdmd_data} not found. Run generate_data.sh first.")
    if not lammps_input.exists():
        print(f"Warning: {lammps_input} not found.")
        sys.exit(1)

    print("=== TDMD/LAMMPS Phase 1 Input Comparison ===\n")

    # Parse TDMD data file.
    if tdmd_data.exists():
        data = parse_lammps_data(tdmd_data)
        print(f"TDMD data file: {data['file']}")
        print(f"  Atoms: {data['n_atoms']}")
        for dim, (lo, hi) in sorted(data["box"].items()):
            axis = dim[0]
            print(f"  Box {axis}: [{lo}, {hi}] = {hi - lo:.6f} A")
    else:
        print("TDMD data file: NOT FOUND")

    print()

    # Parse LAMMPS input.
    lmp = parse_lammps_input(lammps_input)
    print(f"LAMMPS input: {lmp['file']}")
    for key in ("morse_D", "morse_alpha", "morse_r0", "morse_rc",
                "dt", "skin", "T_init", "seed", "n_steps"):
        val = lmp.get(key, "N/A")
        print(f"  {key}: {val}")

    print()

    # TDMD bench defaults for comparison.
    print("TDMD bench_pipeline_scheduler defaults:")
    tdmd_defaults = {
        "morse_D": 0.3429,
        "morse_alpha": 1.3588,
        "morse_r0": 2.866,
        "morse_rc": 6.0,
        "dt": 0.001,
        "skin": 1.0,
        "T_init": 300.0,
        "seed": 42,
        "n_steps": "1000 (--steps)",
    }
    for key, val in tdmd_defaults.items():
        lmp_val = lmp.get(key, "N/A")
        match = "OK" if str(val) == str(lmp_val) else "MISMATCH"
        print(f"  {key}: {val}  (LAMMPS: {lmp_val})  [{match}]")

    print("\n=== Manual check: verify box dimensions match between data files ===")
    print("LAMMPS reads the same small.data file via read_data ../small.data")


if __name__ == "__main__":
    main()
