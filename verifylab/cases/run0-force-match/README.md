# VerifyLab case: run0-force-match

> **LAMMPS A/B on a real lattice.** 256-atom Cu FCC at equilibrium,
> `run 0` in both TDMD and LAMMPS with Morse *and* EAM/alloy potentials,
> step-0 per-atom forces compared atom-by-atom.

## What this checks

1. **Force kernel correctness on a real lattice** — not just a 2-atom toy.
   Full neighbor list, minimum image, type table.
2. **Both potentials wired end-to-end** — Morse (pair) and EAM/alloy (MEAM-
   style with density gather + embedding + force scatter).
3. **Precision contract holds on N-atom inputs** — mixed residuals match
   float32 (~1e-6), fp64 residuals match double (~1e-15).
4. **LAMMPS A/B gate** — the reference is not analytic; it is a committed
   LAMMPS dump. This is the first case where TDMD is graded against LAMMPS
   directly.

Follows on from `two-atoms-morse/` (analytic) — the same dump parser is
reused verbatim.

## Setup

- 256-atom Cu FCC, lattice parameter a = 3.615 Å, 4×4×4 unit cells.
- Box 14.46 × 14.46 × 14.46 Å (periodic).
- Single atom type (Cu), mass 63.546 amu.
- Initial velocities: zero. Step-0 forces only.

The lattice is perfect, so by symmetry every per-atom force is **zero**.
LAMMPS reports components at ~1e-15 (machine zero for double). TDMD in
mixed mode reports ~1e-6 (float32 noise floor in the force path per
ADR 0007); in fp64 mode ~1e-14.

## Potentials

**Morse** (`in.morse_run0.lmp`):
```
pair_style  morse 6.0
pair_coeff  1 1 0.3429 1.3588 2.866
```
D₀ = 0.3429 eV, α = 1.3588 Å⁻¹, r₀ = 2.866 Å, cutoff = 6.0 Å.

**EAM/alloy** (`in.eam_run0.lmp`):
```
pair_style  eam/alloy
pair_coeff  * * Cu_mishin1.eam.alloy Cu
```
The `Cu_mishin1.eam.alloy` setfl file is committed alongside the data file.

Neighbor skin is `1.0 Å` in both cases (LAMMPS `neighbor 1.0 bin`).

## Files

- `cu_fcc_256.data` — LAMMPS data file, consumed by both TDMD and LAMMPS
- `Cu_mishin1.eam.alloy` — EAM/alloy setfl potential
- `in.morse_run0.lmp`, `in.eam_run0.lmp` — LAMMPS reference inputs
- `forces_morse.dump`, `forces_eam.dump` — committed LAMMPS reference dumps
- `log.lammps` — LAMMPS run log (kept for provenance)
- `check.py` — runs `tdmd_standalone` twice, parses both dumps, asserts
- `tolerance.toml` — per-potential, per-mode absolute force tolerances

## Running

```bash
# from project root, after build
./scripts/run-verifylab.sh --mode mixed --case run0-force-match
./scripts/run-verifylab.sh --mode fp64  --case run0-force-match

# or directly
cd verifylab/cases/run0-force-match
python3 check.py --mode mixed
python3 check.py --tdmd-bin /path/to/tdmd_standalone --mode fp64
```

`check.py` reads `TDMD_BIN` and `TDMD_MODE` env vars as fallbacks; the
runner shell script sets both.

## How it works

For each potential (Morse, EAM):

1. `check.py` invokes `tdmd_standalone` with `--nsteps 0 --dump-forces <tmp>`,
   passing either `--morse D,alpha,r0,rc` or `--eam <setfl-file>`.
2. TDMD computes step-0 forces, writes a LAMMPS-format dump to the tmp file,
   prints thermo to stdout.
3. `check.py` parses the tmp dump and the committed LAMMPS reference dump.
4. For each atom, compares each force component by **absolute** difference
   (`|F_tdmd - F_lammps|`). Relative comparison is meaningless here because
   the reference is effectively zero.
5. Asserts max component diff ≤ `tolerance.toml`.

## Pass criteria

All assertions in `check.py` must pass within the tolerances in
`tolerance.toml`. Tolerances are absolute (eV/Å), selected per mode.

## Status

Working end-to-end in both `build-mixed/` and `build-fp64/`.
Observed residuals on 2026-04-10:

| potential | mode  | max \|ΔF\|     | threshold  |
|-----------|-------|----------------|------------|
| Morse     | fp64  | ~2.3e-15 eV/Å  | 1e-12      |
| Morse     | mixed | ~5.0e-6  eV/Å  | 5e-5       |
| EAM       | fp64  | ~9.1e-16 eV/Å  | 1e-12      |
| EAM       | mixed | ~2.7e-6  eV/Å  | 1e-4       |

fp64 hits machine epsilon on both potentials — TDMD and LAMMPS agree to
machine zero when both run in double. Mixed is bounded by the float32 force
path as documented in ADR 0007.
