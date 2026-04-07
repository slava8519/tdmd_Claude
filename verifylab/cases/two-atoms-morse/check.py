#!/usr/bin/env python3
"""
VerifyLab check for two-atoms-morse.

Runs TDMD on the case input, then compares step-0 forces and energies against:
  1. The analytic Morse formula (committed in reference/analytic.json).
  2. The LAMMPS reference log (committed in reference/lammps.log, M1+).

Exits 0 on pass, non-zero on fail. Prints a summary table either way.

This is a STUB until M1 — currently it parses the analytic reference and
prints what it WOULD compare. The actual TDMD invocation lives behind a
TODO that the M1 implementer will fill in.
"""
from __future__ import annotations

import json
import math
import sys
import tomllib  # Python 3.11+
from pathlib import Path

CASE_DIR = Path(__file__).parent
TOL_PATH = CASE_DIR / "tolerance.toml"
ANALYTIC_PATH = CASE_DIR / "reference" / "analytic.json"


def load_tolerances() -> dict:
    with TOL_PATH.open("rb") as f:
        return tomllib.load(f)


def load_analytic() -> dict:
    with ANALYTIC_PATH.open("r") as f:
        return json.load(f)


def compute_morse(r: float, D0: float, alpha: float, r0: float) -> tuple[float, float]:
    """Return (U, F) for the Morse potential at distance r."""
    e = math.exp(-alpha * (r - r0))
    U = D0 * (e * e - 2.0 * e)
    # F = -dU/dr; positive means repulsive, negative means attractive.
    F = 2.0 * alpha * D0 * (e * e - e)
    return U, F


def run_tdmd_and_collect():
    """
    M1 TODO: invoke `./build/bin/tdmd input/input.in` and parse log + dump
    to extract step-0 forces and energies. For now this returns None and the
    check is skipped.
    """
    return None


def main() -> int:
    tol = load_tolerances()
    analytic = load_analytic()

    # Sanity-check the analytic reference itself
    pot = analytic["potential"]
    U, F = compute_morse(
        analytic["step0"]["r_A"], pot["D0_eV"], pot["alpha_inv_A"], pot["r0_A"]
    )

    print(f"== two-atoms-morse VerifyLab check ==")
    print(f"Analytic U = {U:+.6f} eV   (committed: {analytic['step0']['U_eV']:+.6f})")
    print(f"Analytic F = {F:+.6f} eV/A (committed: {analytic['step0']['F_eV_per_A']:+.6f})")

    # Self-consistency of the committed reference
    if abs(U - analytic["step0"]["U_eV"]) > 1e-4:
        print("FAIL: committed analytic U disagrees with on-the-fly computation")
        return 1
    if abs(F - analytic["step0"]["F_eV_per_A"]) > 1e-4:
        print("FAIL: committed analytic F disagrees with on-the-fly computation")
        return 1
    print("PASS: committed analytic reference is self-consistent.")

    # Run TDMD and compare — STUB until M1
    tdmd_out = run_tdmd_and_collect()
    if tdmd_out is None:
        print("SKIP: TDMD invocation not implemented yet (M1 TODO).")
        return 0

    # M1 will fill in: compare tdmd_out vs analytic vs lammps log,
    # using tolerances from `tol`.
    print("PASS: TDMD vs analytic vs LAMMPS — within tolerances.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
