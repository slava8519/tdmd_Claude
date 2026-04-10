#!/usr/bin/env python3
"""
VerifyLab check for nve-drift.

Runs a 20 ps NVE trajectory on 4000 Cu atoms with EAM/alloy and asserts
that the total energy does not drift beyond a per-picosecond relative
threshold. This is the first long-running case in VerifyLab — it verifies
that the integrator + force kernel + neighbor list cooperate correctly
over thousands of steps, not just step 0.

The check:
  1. Run tdmd_standalone --nsteps 20000 --thermo 500 (40 thermo samples).
  2. Parse every "Step N  PE ... KE ... TE ... T ..." line.
  3. Discard the first 25% of samples as equilibration transient.
     (A perfect FCC lattice + thermal velocities is not in equilibrium —
     KE drops by half as atoms settle into potential wells. This transient
     is not drift; we only measure drift after it dies.)
  4. Linear regression of TE vs time (ps) on the remaining samples.
  5. Relative drift = |slope| / |mean TE|.
  6. Assert relative drift ≤ tolerance.toml.

This is a *slow* suite case — it runs in ~3 minutes on CPU, so it should
only be triggered in nightly CI, not on every PR.

Usage:
  python3 check.py                              # uses build-mixed by default
  python3 check.py --tdmd-bin <path>
  python3 check.py --mode {mixed,fp64}
  python3 check.py --nsteps 5000                # short-dev override
  TDMD_BIN=<path> python3 check.py

Exit codes: 0 = PASS, 1 = FAIL, 2 = setup error.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path

CASE_DIR = Path(__file__).parent
REPO_ROOT = CASE_DIR.parents[2]
TOL_PATH = CASE_DIR / "tolerance.toml"
DATA_PATH = CASE_DIR / "input" / "cu_fcc_4000_T100K.data"
EAM_PATH = REPO_ROOT / "verifylab" / "cases" / "run0-force-match" / "Cu_mishin1.eam.alloy"

DEFAULT_BIN_MIXED = REPO_ROOT / "build-mixed" / "tdmd_standalone"
DEFAULT_BIN_FP64 = REPO_ROOT / "build-fp64" / "tdmd_standalone"

DEFAULT_NSTEPS = 20000
THERMO_EVERY = 500
DT_PS = 0.001
SKIN_A = 1.0
EQUILIB_FRACTION = 0.25  # discard this fraction of samples as transient

THERMO_RE = re.compile(
    r"Step\s+(\d+)\s+PE\s+([-\d.eE+]+)\s+KE\s+([-\d.eE+]+)\s+TE\s+([-\d.eE+]+)\s+T\s+([-\d.eE+]+)"
)


def load_tolerances() -> dict:
    with TOL_PATH.open("rb") as f:
        return tomllib.load(f)


def parse_thermo(stdout: str) -> list[dict]:
    samples: list[dict] = []
    for line in stdout.splitlines():
        m = THERMO_RE.search(line)
        if m:
            samples.append({
                "step": int(m.group(1)),
                "pe": float(m.group(2)),
                "ke": float(m.group(3)),
                "te": float(m.group(4)),
                "temp": float(m.group(5)),
            })
    return samples


def run_tdmd(bin_path: Path, nsteps: int) -> list[dict]:
    if not bin_path.exists():
        raise FileNotFoundError(f"tdmd binary not found: {bin_path}")
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"input data file not found: {DATA_PATH}\n"
                                f"run `python3 generate_input.py` first")
    if not EAM_PATH.exists():
        raise FileNotFoundError(f"EAM setfl not found: {EAM_PATH}")

    cmd = [
        str(bin_path),
        "--data", str(DATA_PATH),
        "--eam", str(EAM_PATH),
        "--nsteps", str(nsteps),
        "--dt", str(DT_PS),
        "--skin", str(SKIN_A),
        "--thermo", str(THERMO_EVERY),
    ]
    print(f"running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if result.returncode != 0:
        raise RuntimeError(
            f"tdmd_standalone failed (rc={result.returncode})\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return parse_thermo(result.stdout)


def linreg_slope(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """Return (slope, intercept) of simple linear regression y = slope*x + b."""
    n = len(xs)
    if n < 2:
        raise ValueError("need at least 2 points for regression")
    sx = sum(xs)
    sy = sum(ys)
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in zip(xs, ys))
    denom = n * sxx - sx * sx
    if denom == 0:
        raise ValueError("regression: all x equal")
    slope = (n * sxy - sx * sy) / denom
    intercept = (sy - slope * sx) / n
    return slope, intercept


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
    ap.add_argument("--nsteps", type=int, default=DEFAULT_NSTEPS,
                    help=f"Number of MD steps (default {DEFAULT_NSTEPS})")
    args = ap.parse_args()

    try:
        tol = load_tolerances()
    except Exception as e:
        print(f"ERROR loading tolerances: {e}")
        return 2

    bin_path = resolve_binary(args.tdmd_bin, args.mode)
    print(f"== nve-drift VerifyLab check ==")
    print(f"mode       : {args.mode}")
    print(f"binary     : {bin_path}")
    print(f"input      : {DATA_PATH.name} (4000 Cu atoms, EAM, T=100 K)")
    print(f"nsteps     : {args.nsteps}   (dt={DT_PS} ps, total={args.nsteps*DT_PS:.1f} ps)")

    try:
        samples = run_tdmd(bin_path, args.nsteps)
    except Exception as e:
        print(f"ERROR running TDMD: {e}")
        return 2

    if len(samples) < 8:
        print(f"ERROR: too few thermo samples ({len(samples)}); expected >= 8")
        return 2

    # Discard equilibration transient.
    n_drop = max(1, int(len(samples) * EQUILIB_FRACTION))
    steady = samples[n_drop:]
    print(f"samples    : {len(samples)} total, dropped {n_drop} as transient, "
          f"{len(steady)} used")

    times_ps = [s["step"] * DT_PS for s in steady]
    te_vals = [s["te"] for s in steady]

    mean_te = sum(te_vals) / len(te_vals)
    min_te = min(te_vals)
    max_te = max(te_vals)
    fluct = max_te - min_te

    slope, intercept = linreg_slope(times_ps, te_vals)
    rel_drift = abs(slope) / abs(mean_te)

    print(f"\nsteady-state statistics (t = {times_ps[0]:.2f} .. {times_ps[-1]:.2f} ps):")
    print(f"  mean TE              : {mean_te:+.6f} eV")
    print(f"  TE fluctuation range : {fluct:.6e} eV  "
          f"({fluct/abs(mean_te):.2e} relative)")
    print(f"  slope (TE vs t)      : {slope:+.3e} eV/ps")
    print(f"  |slope| / |mean TE|  : {rel_drift:.3e} /ps")

    thresh = select_thresh(tol["drift"], args.mode)
    print(f"  threshold            : {thresh:.1e} /ps")

    if rel_drift > thresh:
        print(f"\nFAIL: relative drift {rel_drift:.3e} /ps > {thresh:.1e} /ps")
        return 1

    print(f"\nPASS: NVE total energy drift within tolerance "
          f"over {args.nsteps*DT_PS:.0f} ps of Cu EAM dynamics.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
