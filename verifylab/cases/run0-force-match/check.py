#!/usr/bin/env python3
"""
VerifyLab check for run0-force-match.

Runs tdmd_standalone on a 256-atom Cu FCC lattice with two potentials
(Morse and EAM/alloy), parses the resulting step-0 force dumps, and compares
atom-by-atom against committed LAMMPS `run 0` reference dumps.

The lattice is perfect FCC at the equilibrium spacing (a = 3.615 A), so all
per-atom forces are zero up to machine noise. LAMMPS references sit at
~1e-15; TDMD in mixed mode sits at ~1e-6 (float32 force path per ADR 0007);
TDMD in fp64 sits at ~1e-14. Relative error is meaningless here — we use an
absolute max-component tolerance instead.

Usage:
  python3 check.py                              # uses build-mixed by default
  python3 check.py --tdmd-bin <path>            # override binary
  python3 check.py --mode {mixed,fp64}          # pick tolerance column
  TDMD_BIN=<path> python3 check.py              # env var alternative

Exit codes: 0 = PASS, 1 = FAIL, 2 = setup error.
"""
from __future__ import annotations

import argparse
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
DATA_PATH = CASE_DIR / "cu_fcc_256.data"
EAM_PATH = CASE_DIR / "Cu_mishin1.eam.alloy"
LAMMPS_MORSE_DUMP = CASE_DIR / "forces_morse.dump"
LAMMPS_EAM_DUMP = CASE_DIR / "forces_eam.dump"

DEFAULT_BIN_MIXED = REPO_ROOT / "build-mixed" / "tdmd_standalone"
DEFAULT_BIN_FP64 = REPO_ROOT / "build-fp64" / "tdmd_standalone"

# Morse parameters from in.morse_run0.lmp: D0, alpha, r0, cutoff.
MORSE_PARAMS = "0.3429,1.3588,2.866,6.0"
# LAMMPS `neighbor 1.0 bin` → skin = 1.0.
SKIN = "1.0"


def load_tolerances() -> dict:
    with TOL_PATH.open("rb") as f:
        return tomllib.load(f)


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
    for line in stdout.splitlines():
        m = THERMO_RE.search(line)
        if m:
            return {
                "step": int(m.group(1)),
                "pe": float(m.group(2)),
                "ke": float(m.group(3)),
            }
    raise RuntimeError("no thermo line found in tdmd_standalone stdout")


def run_tdmd(bin_path: Path, pot: str) -> tuple[dict, dict]:
    """Invoke tdmd_standalone for one potential. Returns (thermo, atoms).

    pot is 'morse' or 'eam'.
    """
    if not bin_path.exists():
        raise FileNotFoundError(f"tdmd binary not found: {bin_path}")
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"input data file not found: {DATA_PATH}")

    with tempfile.TemporaryDirectory() as tmpdir:
        dump_path = Path(tmpdir) / f"forces_{pot}.dump"
        cmd = [
            str(bin_path),
            "--data", str(DATA_PATH),
            "--nsteps", "0",
            "--skin", SKIN,
            "--dump-forces", str(dump_path),
        ]
        if pot == "morse":
            cmd += ["--morse", MORSE_PARAMS]
        elif pot == "eam":
            if not EAM_PATH.exists():
                raise FileNotFoundError(f"EAM setfl file not found: {EAM_PATH}")
            cmd += ["--eam", str(EAM_PATH)]
        else:
            raise ValueError(f"unknown potential {pot!r}")

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            raise RuntimeError(
                f"tdmd_standalone ({pot}) failed (rc={result.returncode})\n"
                f"cmd: {' '.join(cmd)}\n"
                f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
            )
        thermo = parse_first_thermo(result.stdout)
        atoms = parse_lammps_dump(dump_path)
    return thermo, atoms


def compare_forces(tdmd_atoms: dict, lammps_atoms: dict) -> tuple[float, int, tuple]:
    """Return (max_abs_component_diff, offending_atom_id, offending_components).

    offending_components is (comp_idx, tdmd_val, lammps_val).
    """
    if set(tdmd_atoms.keys()) != set(lammps_atoms.keys()):
        missing = set(lammps_atoms.keys()) - set(tdmd_atoms.keys())
        extra = set(tdmd_atoms.keys()) - set(lammps_atoms.keys())
        raise RuntimeError(
            f"atom id sets differ; missing in TDMD: {sorted(missing)[:5]}..., "
            f"extra in TDMD: {sorted(extra)[:5]}..."
        )

    worst = 0.0
    worst_id = -1
    worst_info = (0, 0.0, 0.0)
    for aid, lammps in lammps_atoms.items():
        tdmd = tdmd_atoms[aid]
        for comp in range(3):
            t = tdmd["force"][comp]
            l = lammps["force"][comp]
            d = abs(t - l)
            if d > worst:
                worst = d
                worst_id = aid
                worst_info = (comp, t, l)
    return worst, worst_id, worst_info


def select_thresh(section: dict, mode: str) -> float:
    key = f"threshold_{mode}"
    if key in section:
        return float(section[key])
    return float(section["threshold"])


def resolve_binary(explicit: str | None, mode: str) -> Path:
    if explicit:
        return Path(explicit)
    env = os.environ.get("TDMD_BIN")
    if env:
        return Path(env)
    return DEFAULT_BIN_MIXED if mode == "mixed" else DEFAULT_BIN_FP64


def run_one(pot: str, bin_path: Path, tol: dict, mode: str,
            lammps_dump: Path) -> tuple[list[str], float]:
    """Run TDMD for one potential and compare against LAMMPS reference.

    Returns (failures, max_force_diff). max_force_diff is NaN if the run
    could not complete (missing dump, TDMD crash, etc.).
    """
    failures: list[str] = []
    print(f"\n-- {pot.upper()} --")

    if not lammps_dump.exists():
        failures.append(f"{pot}: LAMMPS reference dump missing: {lammps_dump}")
        return failures, float("nan")

    try:
        thermo, tdmd_atoms = run_tdmd(bin_path, pot)
    except Exception as e:
        failures.append(f"{pot}: TDMD run failed: {e}")
        return failures, float("nan")

    lammps_atoms = parse_lammps_dump(lammps_dump)

    print(f"   N atoms: {len(tdmd_atoms)} (TDMD) vs {len(lammps_atoms)} (LAMMPS)")
    print(f"   TDMD PE = {thermo['pe']:+.14e} eV")

    try:
        worst, wid, (comp, t, l) = compare_forces(tdmd_atoms, lammps_atoms)
    except Exception as e:
        failures.append(f"{pot}: force comparison failed: {e}")
        return failures, float("nan")

    thresh = select_thresh(tol[f"forces_{pot}"], mode)
    status = "PASS" if worst <= thresh else "FAIL"
    print(f"   max |F_tdmd - F_lammps| = {worst:.3e} eV/A  "
          f"(thresh={thresh:.1e}, {status})")
    print(f"   worst: atom {wid} comp {comp}  "
          f"TDMD={t:+.3e}  LAMMPS={l:+.3e}")

    if worst > thresh:
        failures.append(
            f"{pot}: max force diff {worst:.3e} > {thresh:.1e} "
            f"(atom {wid}, comp {comp})"
        )
    return failures, worst


def main() -> int:
    env_mode = os.environ.get("TDMD_MODE", "mixed")
    ap = argparse.ArgumentParser()
    ap.add_argument("--tdmd-bin", help="Path to tdmd_standalone binary")
    ap.add_argument(
        "--mode",
        choices=["mixed", "fp64"],
        default=env_mode,
        help="Precision mode — picks threshold_<mode> from tolerance.toml",
    )
    args = ap.parse_args()

    try:
        tol = load_tolerances()
    except Exception as e:
        print(f"ERROR loading tolerances: {e}")
        return 2

    bin_path = resolve_binary(args.tdmd_bin, args.mode)
    print(f"== run0-force-match VerifyLab check ==")
    print(f"mode       : {args.mode}")
    print(f"binary     : {bin_path}")
    print(f"input      : {DATA_PATH.name} (256 Cu atoms, perfect FCC, a=3.615 A)")

    t0 = time.monotonic()
    failures: list[str] = []
    morse_fails, morse_worst = run_one(
        "morse", bin_path, tol, args.mode, LAMMPS_MORSE_DUMP)
    eam_fails, eam_worst = run_one(
        "eam", bin_path, tol, args.mode, LAMMPS_EAM_DUMP)
    failures += morse_fails
    failures += eam_fails

    status = "fail" if failures else "pass"
    # If either run hit NaN (setup/crash), demote to error.
    if (morse_worst != morse_worst) or (eam_worst != eam_worst):
        status = "error"
    emit_result(
        case="run0-force-match",
        mode=args.mode,
        status=status,
        metrics={
            "max_force_diff_morse": morse_worst,
            "max_force_diff_eam": eam_worst,
        },
        thresholds={
            "max_force_diff_morse": select_thresh(tol["forces_morse"], args.mode),
            "max_force_diff_eam": select_thresh(tol["forces_eam"], args.mode),
        },
        failures=failures,
        duration_s=time.monotonic() - t0,
    )

    if failures:
        print("\nFAIL:")
        for f in failures:
            print(f"  - {f}")
        return 1 if status == "fail" else 2

    print("\nPASS: TDMD step-0 forces match LAMMPS reference within tolerance "
          "(both Morse and EAM).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
