# VerifyLab case: two-atoms-morse

> **The simplest possible physics test.** Two atoms in a box, Morse pair potential, no thermostat. Compare TDMD's force and energy against the analytic Morse formula and against a single-step LAMMPS reference.

## What this checks

1. **Morse force formula is implemented correctly** — to FP64 precision against the analytic derivative.
2. **Velocity-Verlet integration is correct** — over a short trajectory the energy stays bit-stable.
3. **LAMMPS A/B works** — same input → same forces → bug in TDMD if not.

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
- exp(-2ε) = exp(-1.5) ≈ 0.22313
- exp(-ε) = exp(-0.75) ≈ 0.47237
- U = 0.5 · (0.22313 - 2·0.47237) ≈ -0.36080 eV
- F = 2·1.5·0.5 · (0.22313 - 0.47237) ≈ -0.37386 eV/Å (attractive)

These numbers are committed in `reference/analytic.json` and `check.py` compares against them.

## Files

- `input/input.in` — TDMD input
- `lammps/in.case` — LAMMPS reference input
- `reference/lammps.log` — LAMMPS reference log (committed, regenerated only on potential update)
- `reference/analytic.json` — analytic expected values
- `check.py` — runs both, compares, asserts tolerances
- `tolerance.toml` — numerical tolerances

## Running

```bash
# from project root, after build
./scripts/run-verifylab.sh --case two-atoms-morse

# or directly
cd verifylab/cases/two-atoms-morse
python3 check.py
```

## Pass criteria

All assertions in `check.py` must pass within the tolerances declared in `tolerance.toml`.

## When this case will be created

Milestone **M1**, alongside the first Morse pair implementation. Until then this folder contains only this README and stubs.
