#!/usr/bin/env python3
"""
Generate input/cu_fcc_4000_T100K.data for the nve-drift VerifyLab case.

Takes the 4000-atom Cu FCC lattice from benchmarks/phase1_baseline/small.data,
adds a deterministic Maxwell-Boltzmann velocity distribution at T = 100 K,
and subtracts net momentum so the center of mass is stationary.

The output file is committed so the check is reproducible without numpy at
test time. Re-run this script only if the input lattice changes or the
target temperature changes.

Usage:
  python3 generate_input.py
"""
from __future__ import annotations

import random
import math
from pathlib import Path

CASE_DIR = Path(__file__).parent
REPO_ROOT = CASE_DIR.parents[2]
SRC_DATA = REPO_ROOT / "benchmarks" / "phase1_baseline" / "small.data"
OUT_DATA = CASE_DIR / "input" / "cu_fcc_4000_T100K.data"

# Target temperature in Kelvin.
TARGET_T = 100.0
# RNG seed for reproducibility.
SEED = 42

# LAMMPS `metal` units:
#   mass  : g/mol (amu)
#   vel   : Angstrom / picosecond
#   energy: eV
#   k_B   : 8.617333262e-5 eV/K
#   conv  : to get KE in eV from 0.5*m*v^2 with m in amu and v in A/ps,
#           multiply by mvv2e = 1.0364269e-4
K_B_EV_PER_K = 8.617333262e-5
MVV2E = 1.0364269e-4  # (A/ps)^2 * amu -> eV


def read_atoms(path: Path) -> tuple[list[str], list[tuple[int, int, float, float, float]], float]:
    """Return (header_lines_up_to_and_including_atoms, atoms, mass)."""
    text = path.read_text().splitlines()
    header: list[str] = []
    atoms: list[tuple[int, int, float, float, float]] = []
    mass = None
    in_atoms = False
    in_masses = False
    i = 0
    while i < len(text):
        line = text[i]
        stripped = line.strip()
        if in_atoms:
            if not stripped:
                i += 1
                continue
            parts = stripped.split()
            if not parts[0].isdigit():
                break
            aid = int(parts[0])
            atype = int(parts[1])
            x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
            atoms.append((aid, atype, x, y, z))
            i += 1
            continue
        header.append(line)
        if in_masses and stripped and stripped[0].isdigit():
            parts = stripped.split()
            mass = float(parts[1])
            in_masses = False
        if stripped == "Masses":
            in_masses = True
        if stripped.startswith("Atoms"):
            in_atoms = True
        i += 1
    if mass is None:
        raise RuntimeError("no Masses section found")
    return header, atoms, mass


def generate_velocities(natoms: int, mass_amu: float, temperature_K: float,
                        rng: random.Random) -> list[tuple[float, float, float]]:
    """Maxwell-Boltzmann velocities in Angstrom/picosecond (LAMMPS metal units).

    Each component ~ N(0, sigma) with sigma = sqrt(kT/m) in natural units,
    then converted via mvv2e so that KE in eV matches (3/2) N kB T.
    """
    # In SI: sigma_v = sqrt(kB * T / m). In LAMMPS metal units, the KE
    # expression is (mvv2e/2)*m*v^2 in eV, so we need:
    #   <0.5 * mvv2e * m * v_i^2> = 0.5 * kB * T
    # => <v_i^2> = kB * T / (mvv2e * m)
    # => sigma   = sqrt(kB * T / (mvv2e * m))
    sigma = math.sqrt(K_B_EV_PER_K * temperature_K / (MVV2E * mass_amu))
    vel = [(rng.gauss(0, sigma), rng.gauss(0, sigma), rng.gauss(0, sigma))
           for _ in range(natoms)]

    # Remove net momentum (COM drift).
    vx_mean = sum(v[0] for v in vel) / natoms
    vy_mean = sum(v[1] for v in vel) / natoms
    vz_mean = sum(v[2] for v in vel) / natoms
    vel = [(v[0] - vx_mean, v[1] - vy_mean, v[2] - vz_mean) for v in vel]

    # Rescale to hit the target temperature exactly (accounts for the
    # momentum removal which slightly lowered KE).
    ke = 0.5 * MVV2E * mass_amu * sum(v[0]**2 + v[1]**2 + v[2]**2 for v in vel)
    target_ke = 1.5 * natoms * K_B_EV_PER_K * temperature_K
    scale = math.sqrt(target_ke / ke)
    vel = [(v[0] * scale, v[1] * scale, v[2] * scale) for v in vel]

    return vel


def write_data(path: Path, header: list[str], atoms: list, velocities: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write(f"LAMMPS data file: {len(atoms)} Cu atoms, FCC, "
                f"MB velocities at T={TARGET_T:.0f} K (seed {SEED})\n")
        for line in header[1:]:
            f.write(line + "\n")
        # LAMMPS data format: blank line after "Atoms" header before rows.
        f.write("\n")
        for aid, atype, x, y, z in atoms:
            f.write(f"{aid} {atype} {x:.10f} {y:.10f} {z:.10f}\n")
        f.write("\nVelocities\n\n")
        for (aid, _, _, _, _), (vx, vy, vz) in zip(atoms, velocities):
            f.write(f"{aid} {vx:.10e} {vy:.10e} {vz:.10e}\n")


def main() -> None:
    if not SRC_DATA.exists():
        raise SystemExit(f"source lattice not found: {SRC_DATA}")
    header, atoms, mass = read_atoms(SRC_DATA)
    print(f"source    : {SRC_DATA}")
    print(f"natoms    : {len(atoms)}")
    print(f"mass      : {mass} amu")
    print(f"target T  : {TARGET_T} K")
    print(f"seed      : {SEED}")

    rng = random.Random(SEED)
    vel = generate_velocities(len(atoms), mass, TARGET_T, rng)

    # Sanity: report the actual temperature we generated.
    ke = 0.5 * MVV2E * mass * sum(v[0]**2 + v[1]**2 + v[2]**2 for v in vel)
    temp = 2 * ke / (3 * len(atoms) * K_B_EV_PER_K)
    print(f"actual T  : {temp:.4f} K (should be {TARGET_T:.4f})")
    print(f"KE        : {ke:.6f} eV")

    write_data(OUT_DATA, header, atoms, vel)
    print(f"wrote     : {OUT_DATA}")


if __name__ == "__main__":
    main()
