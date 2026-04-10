#!/usr/bin/env python3
"""
VerifyLab check for cross-precision-ab.

Runs the *same* short NVE trajectory in both build-mixed and build-fp64,
parses each final-state dump, and asserts that positions and forces agree
to within a tolerance set by the float32 precision of the mixed-mode
force path (ADR 0007) accumulated over the short trajectory.

This is the only VerifyLab case that does NOT take `--mode` as an input
— it compares mixed against fp64 directly, so it needs both binaries.
The resolved binary pair comes from (in priority order):
  1. --tdmd-bin-mixed / --tdmd-bin-fp64 CLI flags
  2. TDMD_BIN_MIXED / TDMD_BIN_FP64 env vars
  3. Default paths build-mixed/tdmd_standalone and build-fp64/tdmd_standalone

The case runs on the nve-drift input (4000 Cu atoms, T=100 K, MB seed 42)
for 100 steps (100 fs). That's long enough for mixed-vs-fp64 divergence
to become measurable, short enough that Lyapunov-style chaotic
amplification has not yet dominated. Asserts remain in the "numerical
noise" regime, not the "trajectories have diverged macroscopically"
regime.

Usage:
  python3 check.py
  python3 check.py --tdmd-bin-mixed <path> --tdmd-bin-fp64 <path>
  python3 check.py --nsteps 50

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
# Reuse the 4000-atom Cu FCC + thermal velocities from nve-drift.
DATA_PATH = (REPO_ROOT / "verifylab" / "cases" / "nve-drift"
             / "input" / "cu_fcc_4000_T100K.data")
EAM_PATH = (REPO_ROOT / "verifylab" / "cases" / "run0-force-match"
            / "Cu_mishin1.eam.alloy")

DEFAULT_BIN_MIXED = REPO_ROOT / "build-mixed" / "tdmd_standalone"
DEFAULT_BIN_FP64 = REPO_ROOT / "build-fp64" / "tdmd_standalone"

DEFAULT_NSTEPS = 100
DT_PS = 0.001
SKIN_A = 1.0

THERMO_RE = re.compile(
    r"Step\s+(\d+)\s+PE\s+([-\d.eE+]+)\s+KE\s+([-\d.eE+]+)\s+TE\s+([-\d.eE+]+)\s+T\s+([-\d.eE+]+)"
)


def load_tolerances() -> dict:
    with TOL_PATH.open("rb") as f:
        return tomllib.load(f)


def parse_lammps_dump(path: Path) -> dict:
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
                    "pos": (float(row["x"]), float(row["y"]), float(row["z"])),
                    "force": (float(row["fx"]), float(row["fy"]), float(row["fz"])),
                }
                idx += 1
        else:
            idx += 1
    return atoms


def parse_last_thermo(stdout: str) -> dict | None:
    last = None
    for line in stdout.splitlines():
        m = THERMO_RE.search(line)
        if m:
            last = {
                "step": int(m.group(1)),
                "pe": float(m.group(2)),
                "te": float(m.group(4)),
            }
    return last


def run_tdmd(bin_path: Path, nsteps: int, dump_path: Path) -> dict:
    if not bin_path.exists():
        raise FileNotFoundError(f"tdmd binary not found: {bin_path}")
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"input data file not found: {DATA_PATH}\n"
            f"run `python3 ../nve-drift/generate_input.py` first"
        )
    if not EAM_PATH.exists():
        raise FileNotFoundError(f"EAM setfl not found: {EAM_PATH}")

    cmd = [
        str(bin_path),
        "--data", str(DATA_PATH),
        "--eam", str(EAM_PATH),
        "--nsteps", str(nsteps),
        "--dt", str(DT_PS),
        "--skin", str(SKIN_A),
        "--thermo", str(max(1, nsteps)),
        "--dump-final", str(dump_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(
            f"tdmd_standalone failed (rc={result.returncode})\n"
            f"cmd: {' '.join(cmd)}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    thermo = parse_last_thermo(result.stdout)
    if thermo is None:
        raise RuntimeError("no thermo line found in tdmd_standalone stdout")
    return thermo


def diff_dumps(a: dict, b: dict) -> tuple[float, float, int, int]:
    """Return (max |dpos|, max |dforce|, worst_pos_atom, worst_force_atom)."""
    if set(a.keys()) != set(b.keys()):
        raise RuntimeError("atom id sets differ between mixed and fp64 dumps")

    worst_pos = 0.0
    worst_pos_id = -1
    worst_force = 0.0
    worst_force_id = -1

    for aid in a:
        pa = a[aid]["pos"]
        pb = b[aid]["pos"]
        fa = a[aid]["force"]
        fb = b[aid]["force"]
        for comp in range(3):
            dp = abs(pa[comp] - pb[comp])
            if dp > worst_pos:
                worst_pos = dp
                worst_pos_id = aid
            df = abs(fa[comp] - fb[comp])
            if df > worst_force:
                worst_force = df
                worst_force_id = aid
    return worst_pos, worst_force, worst_pos_id, worst_force_id


def resolve_binaries(cli_mixed: str | None, cli_fp64: str | None) -> tuple[Path, Path]:
    mixed = Path(cli_mixed) if cli_mixed else None
    if mixed is None:
        env = os.environ.get("TDMD_BIN_MIXED")
        mixed = Path(env) if env else DEFAULT_BIN_MIXED

    fp64 = Path(cli_fp64) if cli_fp64 else None
    if fp64 is None:
        env = os.environ.get("TDMD_BIN_FP64")
        fp64 = Path(env) if env else DEFAULT_BIN_FP64

    return mixed, fp64


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tdmd-bin-mixed", help="Path to mixed-mode tdmd_standalone")
    ap.add_argument("--tdmd-bin-fp64", help="Path to fp64-mode tdmd_standalone")
    ap.add_argument("--nsteps", type=int, default=DEFAULT_NSTEPS)
    args = ap.parse_args()

    try:
        tol = load_tolerances()
    except Exception as e:
        print(f"ERROR loading tolerances: {e}")
        return 2

    bin_mixed, bin_fp64 = resolve_binaries(args.tdmd_bin_mixed, args.tdmd_bin_fp64)
    print(f"== cross-precision-ab VerifyLab check ==")
    print(f"mixed binary : {bin_mixed}")
    print(f"fp64  binary : {bin_fp64}")
    print(f"input        : {DATA_PATH.name}")
    print(f"nsteps       : {args.nsteps}  (dt={DT_PS} ps, total={args.nsteps*DT_PS:.3f} ps)")

    t0 = time.monotonic()
    pos_thresh = float(tol["position"]["threshold"])
    force_thresh = float(tol["force"]["threshold"])
    te_thresh = float(tol["energy"]["threshold_rel"])

    with tempfile.TemporaryDirectory() as tmpdir:
        dump_mixed = Path(tmpdir) / "final_mixed.dump"
        dump_fp64 = Path(tmpdir) / "final_fp64.dump"

        try:
            thermo_mixed = run_tdmd(bin_mixed, args.nsteps, dump_mixed)
            thermo_fp64 = run_tdmd(bin_fp64, args.nsteps, dump_fp64)
        except Exception as e:
            print(f"ERROR running TDMD: {e}")
            emit_result(
                case="cross-precision-ab",
                mode="mixed+fp64",
                status="error",
                metrics={},
                thresholds={
                    "max_dpos": pos_thresh,
                    "max_dforce": force_thresh,
                    "rel_dte": te_thresh,
                },
                failures=[f"TDMD run failed: {e}"],
                duration_s=time.monotonic() - t0,
            )
            return 2

        atoms_mixed = parse_lammps_dump(dump_mixed)
        atoms_fp64 = parse_lammps_dump(dump_fp64)

    print(f"\nfinal thermo:")
    print(f"  mixed TE = {thermo_mixed['te']:+.10f} eV")
    print(f"  fp64  TE = {thermo_fp64['te']:+.10f} eV")
    print(f"  |dTE|    = {abs(thermo_mixed['te']-thermo_fp64['te']):.3e} eV  "
          f"({abs(thermo_mixed['te']-thermo_fp64['te'])/abs(thermo_fp64['te']):.2e} rel)")

    worst_pos, worst_force, pos_id, force_id = diff_dumps(atoms_mixed, atoms_fp64)
    print(f"\nfinal-state divergence (max over {len(atoms_mixed)} atoms):")
    print(f"  max |dx_i|   = {worst_pos:.3e} A       (atom {pos_id})")
    print(f"  max |dF_i|   = {worst_force:.3e} eV/A  (atom {force_id})")

    failures: list[str] = []

    print(f"\nthresholds:")
    print(f"  position  : {pos_thresh:.1e} A")
    print(f"  force     : {force_thresh:.1e} eV/A")
    print(f"  |dTE|/TE  : {te_thresh:.1e}")

    if worst_pos > pos_thresh:
        failures.append(
            f"position: max |dx| {worst_pos:.3e} > {pos_thresh:.1e} A "
            f"(atom {pos_id})"
        )
    if worst_force > force_thresh:
        failures.append(
            f"force: max |dF| {worst_force:.3e} > {force_thresh:.1e} eV/A "
            f"(atom {force_id})"
        )
    rel_te = abs(thermo_mixed['te'] - thermo_fp64['te']) / abs(thermo_fp64['te'])
    if rel_te > te_thresh:
        failures.append(f"energy: |dTE|/|TE| {rel_te:.3e} > {te_thresh:.1e}")

    emit_result(
        case="cross-precision-ab",
        mode="mixed+fp64",
        status="fail" if failures else "pass",
        metrics={
            "max_dpos": worst_pos,
            "max_dforce": worst_force,
            "rel_dte": rel_te,
        },
        thresholds={
            "max_dpos": pos_thresh,
            "max_dforce": force_thresh,
            "rel_dte": te_thresh,
        },
        failures=failures,
        duration_s=time.monotonic() - t0,
    )

    if failures:
        print("\nFAIL:")
        for f in failures:
            print(f"  - {f}")
        return 1

    print("\nPASS: mixed and fp64 trajectories agree within float32-noise "
          "tolerance after short NVE run.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
