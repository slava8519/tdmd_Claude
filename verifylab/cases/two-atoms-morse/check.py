#!/usr/bin/env python3
"""
VerifyLab check for two-atoms-morse.

Runs tdmd_standalone on input/two_atoms.data, parses the step-0 force dump
and thermo line, and compares against the analytic Morse reference in
reference/analytic.json.

Usage:
  python3 check.py                              # uses default binary
  python3 check.py --tdmd-bin <path>            # override binary
  python3 check.py --mode {mixed,fp64}          # pick tolerance column
  TDMD_BIN=<path> python3 check.py              # env var alternative

Exit codes: 0 = PASS, 1 = FAIL, 2 = setup error.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
import tomllib
from pathlib import Path

CASE_DIR = Path(__file__).parent
REPO_ROOT = CASE_DIR.parents[2]

sys.path.insert(0, str(REPO_ROOT / "verifylab" / "runners"))
from result_schema import emit_result  # noqa: E402
TOL_PATH = CASE_DIR / "tolerance.toml"
ANALYTIC_PATH = CASE_DIR / "reference" / "analytic.json"
DATA_PATH = CASE_DIR / "input" / "two_atoms.data"

DEFAULT_BIN_MIXED = REPO_ROOT / "build-mixed" / "tdmd_standalone"
DEFAULT_BIN_FP64 = REPO_ROOT / "build-fp64" / "tdmd_standalone"

MORSE_PARAMS = "0.5,1.5,2.0,8.0"  # D0, alpha, r0, cutoff — matches analytic.json


def load_tolerances() -> dict:
    with TOL_PATH.open("rb") as f:
        return tomllib.load(f)


def load_analytic() -> dict:
    with ANALYTIC_PATH.open("r") as f:
        return json.load(f)


def parse_lammps_dump(path: Path) -> dict:
    """Parse a LAMMPS custom dump with columns: id type x y z fx fy fz.

    Returns a dict mapping atom id -> {type, pos, force}.
    """
    lines = path.read_text().splitlines()
    idx = 0
    atoms: dict[int, dict] = {}
    while idx < len(lines):
        line = lines[idx].strip()
        if line.startswith("ITEM: ATOMS"):
            columns = line.split()[2:]
            idx += 1
            while idx < len(lines) and not lines[idx].startswith("ITEM:"):
                parts = lines[idx].split()
                row = dict(zip(columns, parts))
                aid = int(row["id"])
                atoms[aid] = {
                    "type": int(row["type"]),
                    "pos": (float(row["x"]), float(row["y"]), float(row["z"])),
                    "force": (float(row["fx"]), float(row["fy"]), float(row["fz"])),
                }
                idx += 1
        else:
            idx += 1
    return atoms


THERMO_RE = re.compile(
    r"Step\s+(\d+)\s+PE\s+([-\d.eE+]+)\s+KE\s+([-\d.eE+]+)\s+TE\s+([-\d.eE+]+)\s+T\s+([-\d.eE+]+)"
)


def parse_first_thermo(stdout: str) -> dict:
    """Parse the first 'Step N  PE X  KE Y  TE Z  T T' line from stdout."""
    for line in stdout.splitlines():
        m = THERMO_RE.search(line)
        if m:
            return {
                "step": int(m.group(1)),
                "pe": float(m.group(2)),
                "ke": float(m.group(3)),
                "te": float(m.group(4)),
                "temp": float(m.group(5)),
            }
    raise RuntimeError("no thermo line found in tdmd_standalone stdout")


def run_tdmd(bin_path: Path) -> tuple[dict, dict]:
    """Invoke tdmd_standalone on the two-atoms input. Returns (thermo, atoms)."""
    if not bin_path.exists():
        raise FileNotFoundError(f"tdmd binary not found: {bin_path}")
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"input data file not found: {DATA_PATH}")
    with tempfile.TemporaryDirectory() as tmpdir:
        dump_path = Path(tmpdir) / "forces.dump"
        cmd = [
            str(bin_path),
            "--data", str(DATA_PATH),
            "--morse", MORSE_PARAMS,
            "--nsteps", "0",
            "--dump-forces", str(dump_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            raise RuntimeError(
                f"tdmd_standalone failed (rc={result.returncode})\n"
                f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
            )
        thermo = parse_first_thermo(result.stdout)
        atoms = parse_lammps_dump(dump_path)
    return thermo, atoms


def max_rel_err(a: float, b: float) -> float:
    denom = max(abs(a), abs(b), 1e-30)
    return abs(a - b) / denom


def select_thresh(section: dict, key_base: str, mode: str) -> float:
    """Pick threshold_<mode> if present, else plain 'threshold'."""
    mode_key = f"{key_base}_{mode}"
    if mode_key in section:
        return float(section[mode_key])
    return float(section[key_base])


def check_atom_forces(tdmd_atoms: dict, analytic: dict, tol: dict,
                      mode: str) -> tuple[list[str], float]:
    """Compare per-atom forces against analytic reference.

    Returns (failures, max_force_err) where max_force_err is the worst
    per-component error (relative where analytic > 1e-12, absolute otherwise).
    """
    failures: list[str] = []
    thresh = select_thresh(tol["forces"], "threshold", mode)

    expected = {
        1: tuple(analytic["step0"]["force_on_atom_1_eV_per_A"]),
        2: tuple(analytic["step0"]["force_on_atom_2_eV_per_A"]),
    }
    max_err = 0.0
    for aid, fexp in expected.items():
        if aid not in tdmd_atoms:
            failures.append(f"atom {aid} missing from TDMD dump")
            continue
        fobs = tdmd_atoms[aid]["force"]
        for comp, (e, o) in enumerate(zip(fexp, fobs)):
            # For components where the analytic expectation is zero, compare absolute;
            # for nonzero, compare relative.
            if abs(e) < 1e-12:
                err = abs(o)
                if err > max_err:
                    max_err = err
                if err > thresh:
                    failures.append(
                        f"atom {aid} F[{comp}]: expected 0, got {o:+.6e} "
                        f"(abs > {thresh:.1e})"
                    )
            else:
                rel = max_rel_err(o, e)
                if rel > max_err:
                    max_err = rel
                if rel > thresh:
                    failures.append(
                        f"atom {aid} F[{comp}]: expected {e:+.6e}, got {o:+.6e} "
                        f"(rel {rel:.3e} > {thresh:.1e})"
                    )
    return failures, max_err


def check_energy(tdmd_thermo: dict, analytic: dict, tol: dict,
                 mode: str) -> tuple[list[str], float]:
    """Compare PE against analytic. Returns (failures, pe_abs_err)."""
    failures: list[str] = []
    thresh = select_thresh(tol["energy"], "threshold", mode)
    expected = float(analytic["step0"]["U_eV"])
    observed = tdmd_thermo["pe"]
    err = abs(observed - expected)
    if err > thresh:
        failures.append(
            f"PE: expected {expected:+.8f}, got {observed:+.8f} "
            f"(abs {err:.3e} > {thresh:.1e})"
        )
    return failures, err


def resolve_binary(explicit: str | None, mode: str) -> Path:
    if explicit:
        return Path(explicit)
    env = os.environ.get("TDMD_BIN")
    if env:
        return Path(env)
    return DEFAULT_BIN_MIXED if mode == "mixed" else DEFAULT_BIN_FP64


def main() -> int:
    env_mode = os.environ.get("TDMD_MODE", "mixed")
    ap = argparse.ArgumentParser()
    ap.add_argument("--tdmd-bin", help="Path to tdmd_standalone binary")
    ap.add_argument(
        "--mode",
        choices=["mixed", "fp64"],
        default=env_mode,
        help="Precision mode — picks threshold_<mode> from tolerance.toml "
             "(default: $TDMD_MODE or 'mixed')",
    )
    args = ap.parse_args()
    if args.mode not in ("mixed", "fp64"):
        print(f"ERROR: invalid mode '{args.mode}'")
        return 2

    try:
        tol = load_tolerances()
        analytic = load_analytic()
    except Exception as e:
        print(f"ERROR loading references: {e}")
        return 2

    bin_path = resolve_binary(args.tdmd_bin, args.mode)
    print(f"== two-atoms-morse VerifyLab check ==")
    print(f"mode       : {args.mode}")
    print(f"binary     : {bin_path}")

    t0 = time.monotonic()
    try:
        thermo, atoms = run_tdmd(bin_path)
    except Exception as e:
        print(f"ERROR running TDMD: {e}")
        emit_result(
            case="two-atoms-morse",
            mode=args.mode,
            status="error",
            metrics={},
            thresholds={},
            failures=[f"run failed: {e}"],
            duration_s=time.monotonic() - t0,
        )
        return 2

    expected_pe = float(analytic["step0"]["U_eV"])
    expected_f1 = analytic["step0"]["force_on_atom_1_eV_per_A"][0]
    print(f"TDMD step-0 PE = {thermo['pe']:+.14f} eV  (analytic: {expected_pe:+.14f})")
    print(f"  atom 1 Fx    = {atoms[1]['force'][0]:+.14f} eV/A  (analytic: {expected_f1:+.14f})")
    print(f"  atom 2 Fx    = {atoms[2]['force'][0]:+.14f} eV/A  (analytic: {-expected_f1:+.14f})")

    failures: list[str] = []
    force_fails, max_force_err = check_atom_forces(atoms, analytic, tol, args.mode)
    pe_fails, pe_abs_err = check_energy(thermo, analytic, tol, args.mode)
    failures += force_fails
    failures += pe_fails

    status = "fail" if failures else "pass"
    emit_result(
        case="two-atoms-morse",
        mode=args.mode,
        status=status,
        metrics={
            "max_force_err": max_force_err,
            "pe_abs_err": pe_abs_err,
        },
        thresholds={
            "max_force_err": select_thresh(tol["forces"], "threshold", args.mode),
            "pe_abs_err": select_thresh(tol["energy"], "threshold", args.mode),
        },
        failures=failures,
        duration_s=time.monotonic() - t0,
    )

    if failures:
        print("\nFAIL:")
        for f in failures:
            print(f"  - {f}")
        return 1

    print("\nPASS: TDMD step-0 forces and energy match analytic Morse reference.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
