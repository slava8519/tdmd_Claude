# VerifyLab case: two-atoms-morse

> **The simplest possible physics test.** Two atoms in a box, Morse pair potential, no thermostat. Compare TDMD's force and energy against the analytic Morse formula and against a single-step LAMMPS reference.

## What this checks

1. **Morse force formula is implemented correctly** — to FP64 precision against the analytic derivative.
2. **Energy is consistent** — step-0 PE matches the analytic Morse potential.
3. **Precision contract holds** — mixed mode matches analytic to ~float32
   epsilon; fp64 mode matches to ~double epsilon.

LAMMPS A/B on the same input lives in `run0-force-match/` (VL-2), which
will reuse the same dump parser.

This is the **first** physics test we run. If it fails, nothing else matters.

## Setup

Two atoms of type 1 (mass 1.0 amu), placed at:

- atom 1: (0.0, 0.0, 0.0)
- atom 2: (2.5, 0.0, 0.0)

Periodic box, 20 × 20 × 20 Å, periodic in all directions but big enough that PBC images don't matter.

Morse parameters:
- D₀ = 0.5 eV
- α = 1.5 Å⁻¹
- r₀ = 2.0 Å
- cutoff = 8.0 Å

Initial velocities: zero. The atoms feel a single force pair on step 0, then evolve.

## Analytic reference

The Morse potential:

```
U(r) = D₀ · ( exp(-2α(r - r₀)) - 2·exp(-α(r - r₀)) )
F(r) = -dU/dr = 2·α·D₀ · ( exp(-2α(r - r₀)) - exp(-α(r - r₀)) )
```

At r = 2.5 Å, with the parameters above:

- ε = α·(r - r₀) = 0.75
- exp(-2ε) = exp(-1.5) ≈ 0.22313016014842982
- exp(-ε)  = exp(-0.75) ≈ 0.47236655274101469
- U_eV                    = -0.36080147266679978
- F_radial_eV_per_A       = -0.37385458888887730   (attractive, scalar along r̂)
- force on atom 1 (at 0)  = (+0.37385458888888, 0, 0)   (pulled toward atom 2)
- force on atom 2 (at 2.5)= (-0.37385458888888, 0, 0)   (pulled toward atom 1)

These numbers are committed in `reference/analytic.json` to full double
precision; `check.py` compares against them in both build modes.

## Files

- `input/two_atoms.data` — LAMMPS data file consumed by `tdmd_standalone`
- `input/input.in` — aspirational LAMMPS-script view of the same setup
  (kept as documentation; not parsed by TDMD yet)
- `lammps/in.case` — LAMMPS reference input
- `reference/analytic.json` — analytic expected values (full double precision)
- `check.py` — runs `tdmd_standalone`, parses dump, asserts tolerances
- `tolerance.toml` — numerical tolerances (per-mode)

## Running

```bash
# from project root, after build
./scripts/run-verifylab.sh --mode mixed --case two-atoms-morse
./scripts/run-verifylab.sh --mode fp64  --case two-atoms-morse

# or directly
cd verifylab/cases/two-atoms-morse
python3 check.py --mode mixed
python3 check.py --tdmd-bin /path/to/tdmd_standalone --mode fp64
```

`check.py` reads `TDMD_BIN` and `TDMD_MODE` env vars as fallbacks when the
corresponding CLI flags are not given. `./scripts/run-verifylab.sh` sets both
based on its `--mode` argument, so the runner Just Works.

## How it works

1. `check.py` calls `tdmd_standalone` with `--nsteps 0 --dump-forces <tmp>`.
2. The binary computes step-0 forces, prints thermo to stdout, and writes a
   LAMMPS-format dump to the tmp file.
3. `check.py` parses the dump (positions + forces) and the first `Step 0`
   thermo line (PE), then compares against `reference/analytic.json`.
4. Tolerances come from `tolerance.toml`: `threshold` is the fp64 bound,
   `threshold_mixed` is the mixed-precision bound.

The dump format is identical to LAMMPS `dump custom id type x y z fx fy fz`,
chosen so that the same parser can be reused for `run0-force-match` and
other cases that compare against committed LAMMPS reference dumps.

## Pass criteria

All assertions in `check.py` must pass within the tolerances declared in `tolerance.toml`.

## Status

Working end-to-end. Passes in both `build-mixed/` and `build-fp64/` builds.
Observed residuals on 2026-04-10:

| mode  | PE error (abs)   | Force error (rel) |
|-------|------------------|-------------------|
| fp64  | ~0 (14+ digits)  | ~0 (14+ digits)   |
| mixed | ~1.5e-8          | ~3e-8             |

fp64 hits machine epsilon. mixed residual is at float32 precision — the
step-0 force path evaluates `exp()` in `force_t = float` (ADR 0007).
