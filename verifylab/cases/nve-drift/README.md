# VerifyLab case: nve-drift

> **The longest-running physics test.** 4000 Cu atoms, EAM/alloy
> potential, 20 ps of NVE dynamics. Asserts that total energy does
> not drift beyond a per-picosecond relative threshold.

## What this checks

1. **Integrator + force kernel cooperate correctly over thousands of
   steps** — not just a step-0 force match. NVE drift is the classic
   integrator sanity check: if energy is drifting, velocity-Verlet is
   broken, or the force is non-conservative, or precision is leaking.
2. **Neighbor list rebuilds are transparent** — rebuilding the list
   mid-trajectory must not introduce an energy jump.
3. **Precision contract holds under accumulated arithmetic** —
   mixed-mode float32 force errors must not compound into visible
   drift over 20 ps / 20 000 steps.

This is the first VerifyLab case that actually integrates equations of
motion. Previous cases (`two-atoms-morse`, `run0-force-match`) only
validate step-0 forces.

## Setup

- 4000-atom Cu FCC, lattice parameter a = 3.615 Å, 10×10×10 unit cells.
- Box 36.15 × 36.15 × 36.15 Å (periodic).
- Mass 63.546 amu.
- Initial velocities: Maxwell–Boltzmann at T = 100 K, deterministic
  (Python `random.Random(42)`), net momentum subtracted, then rescaled
  to hit the target temperature exactly.
- Potential: **EAM/alloy** (`Cu_mishin1.eam.alloy`, shared with
  `run0-force-match/`). More physical than Morse for Cu.
- `dt = 1 fs`, `skin = 1.0 Å`, `nsteps = 20000` → 20 ps total.
- Thermo printed every 500 steps → 41 samples.

The equilibrium temperature after the lattice relaxes is ~50 K
(equipartition between KE and PE on a cold FCC lattice). The first 25 %
of thermo samples are discarded as transient — drift is only measured
on the steady-state portion.

## Metric

```
rel_drift_per_ps = |slope(linreg(TE vs t))| / |mean(TE)|
```

where the regression is over the steady-state samples only. This is
the standard NVE drift metric: it tests the *trend*, not the noise.
Thermal fluctuations in TE are large in mixed mode (~1e-5 relative)
but that's noise; the slope averages them out.

## Files

- `input/cu_fcc_4000_T100K.data` — committed, deterministic MB
  velocities, seed 42. Regenerate with `python3 generate_input.py` if
  the source lattice or target T changes.
- `generate_input.py` — helper that produces the data file (numpy-free
  so it runs anywhere with a stock Python 3).
- `check.py` — runs `tdmd_standalone`, parses thermo, asserts drift
- `tolerance.toml` — per-mode drift thresholds + `slow = true`

## Running

```bash
# Slow suite — nightly CI only
./scripts/run-verifylab.sh --mode mixed --suite slow

# Or directly
cd verifylab/cases/nve-drift
python3 check.py --mode mixed
python3 check.py --tdmd-bin /path/to/tdmd_standalone --mode fp64

# Short development run (fewer steps, looser statistics)
python3 check.py --mode mixed --nsteps 3000
```

## How it works

1. `check.py` invokes `tdmd_standalone` with `--eam`, `--nsteps 20000`,
   `--thermo 500`, and other NVE-relevant flags.
2. Parses every `Step N  PE ... KE ... TE ... T ...` line from stdout.
3. Drops the first 25 % of samples (equilibration transient).
4. Linear regression of TE vs time (picoseconds) on the remainder.
5. Asserts `|slope| / |mean TE| ≤ threshold_<mode>`.

## Pass criteria

All assertions in `check.py` must pass within the tolerances in
`tolerance.toml`.

## Status

Working end-to-end in both `build-mixed/` and `build-fp64/`.
Observed on 2026-04-10 (20 ps / 20 000 steps, CPU-only driver):

| mode  | mean TE (eV)     | fluctuation rel | drift /ps | threshold |
|-------|------------------|-----------------|-----------|-----------|
| fp64  | -14109.162       | ~4.4e-8         | ~5.7e-10  | 5e-7      |
| mixed | -14109.031       | ~8.4e-6         | ~3.0e-8   | 5e-5      |

fp64 is essentially perfect — drift is ~1000× under threshold and
fluctuation is at double precision noise floor. Mixed is bounded by
float32 force errors (ADR 0007); drift is still ~1700× under its
(looser) threshold, because drift is a *trend* and float32 force
noise mostly cancels on long time-averaging.

Runtime: ~3 minutes per mode on CPU (`tdmd_standalone` is CPU-only).
That's why this case is marked `slow = true`: it runs nightly, not
on every PR.
