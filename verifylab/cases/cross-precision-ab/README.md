# VerifyLab case: cross-precision-ab

> **Mixed vs fp64, head-to-head.** Run the same short NVE trajectory in
> both `build-mixed/` and `build-fp64/`, compare final positions, forces,
> and total energy atom-by-atom. This is the only VerifyLab case where
> TDMD is graded against *itself* in a different precision mode.

## What this checks

1. **The precision contract from ADR 0007 is the only divergence source.**
   Mixed mode keeps `pos_t = vel_t = double`, downgrades only `force_t`
   to float. After a short NVE trajectory, mixed and fp64 should differ
   only at the float32 noise floor (plus a small deterministic bias
   from the distinct transcendental paths in the force kernel).
2. **Nothing silently runs in single precision that shouldn't be.** If a
   reduction, an integrator update, or a neighbor-list decision is
   secretly float32 in mixed mode, this test catches the cumulative
   effect within 100 steps.
3. **The two builds are reproducible against the same input.** Both
   modes read the same `.data` file, initialize velocities from the
   same `Velocities` section, and compute forces from the same EAM
   setfl file.

## Why this case must stay SHORT

MD is chaotic: identical inputs with different floating-point paths
diverge **exponentially** under Lyapunov amplification, typically with a
Lyapunov time of ~1 ps for dense metals. Over 10 ps, two trajectories
that started at round-off-level separation will be O(1 Å) apart —
that's physics, not a bug. So this check only makes sense at very short
times where the divergence is still in the "numerical noise" regime.

We use **100 steps = 0.1 ps**, which is well below the Lyapunov
amplification threshold.

## Setup

- Input: `verifylab/cases/nve-drift/input/cu_fcc_4000_T100K.data`
  (same 4000-atom Cu FCC + Maxwell-Boltzmann at T=100 K, seed 42)
- Potential: EAM/alloy (shared with `run0-force-match/` and `nve-drift/`)
- `dt = 1 fs`, `skin = 1.0 Å`, `nsteps = 100` → 100 fs total
- Both `build-mixed/tdmd_standalone` and `build-fp64/tdmd_standalone`
  are invoked, each writing a `--dump-final` to a tmp file
- No per-case input data is committed — everything is reused

## Metrics

```
max |dx|      = max over all atoms, components of |x_mixed_i - x_fp64_i|
max |dF|      = max over all atoms, components of |F_mixed_i - F_fp64_i|
|dTE| / |TE|  = |TE_mixed - TE_fp64| / |TE_fp64|
```

## Files

- `check.py` — runs both binaries, parses both dumps, compares
- `tolerance.toml` — single-valued thresholds (not per-mode)
- (no README-committed input data — reuses `../nve-drift/input/`)

## Running

```bash
# Fast suite — runs on every PR
./scripts/run-verifylab.sh --suite fast          # includes this case

# Or directly
cd verifylab/cases/cross-precision-ab
python3 check.py
python3 check.py --tdmd-bin-mixed /path/to/mixed --tdmd-bin-fp64 /path/to/fp64
python3 check.py --nsteps 50                     # shorter dev run
```

Unlike other VerifyLab cases, this one does **not** take `--mode`. It
always runs *both* modes; `--mode mixed/fp64` would be meaningless here.
Binaries are resolved in priority order:
1. `--tdmd-bin-mixed` / `--tdmd-bin-fp64` CLI flags
2. `TDMD_BIN_MIXED` / `TDMD_BIN_FP64` env vars
3. Default paths `build-mixed/tdmd_standalone` and
   `build-fp64/tdmd_standalone`

## How it works

1. `check.py` invokes `tdmd_standalone` twice (once per mode) with
   `--nsteps 100 --dump-final <tmp>`.
2. For each run, it parses the final-state dump (LAMMPS custom format
   with `id type x y z fx fy fz`) and the last thermo line from stdout.
3. Atom-by-atom, it computes per-component `|mixed - fp64|` for
   positions and forces, tracking the max.
4. Asserts all three metrics are within `tolerance.toml`.

## Pass criteria

All three thresholds in `tolerance.toml` must hold.

## Status

Working in the fast suite. Observed on 2026-04-10 (100 fs, 4000 atoms):

| metric           | observed   | threshold | margin |
|------------------|------------|-----------|--------|
| max \|dx\|       | 2.1e-5 Å   | 5e-4 Å    | ~24×   |
| max \|dF\|       | 6.8e-4 eV/Å| 2e-2 eV/Å | ~30×   |
| \|dTE\| / \|TE\| | 1.1e-5     | 1e-4      | ~9×    |

The position and force divergences are dominated by float32 noise in
the mixed-mode force path (ADR 0007). The energy difference is mostly a
*constant bias* between the two modes' equilibrium PE baselines — not a
growing drift. Tighten thresholds only if a regression lowers these.
