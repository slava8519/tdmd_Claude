# Ni-Co-Cr 3-component baseline

> A simpler, smaller, more debuggable EAM benchmark. **Run this before FCC16.**

## Why Ni-Co-Cr first

- 3 atom types instead of 16 → 6 unique pair interactions instead of 136.
- Public, well-validated NIST setfl alloy file.
- Small enough to run on a CPU in seconds for cross-checking.
- A known-good system in the materials-science literature (high-entropy alloy precursor).

The rule is: if EAM is broken on Ni-Co-Cr, it will be broken on FCC16 in 50 confusing ways. Catch it on the simpler test first.

## Sizes

| Size | Atoms | Use |
|---|---|---|
| Tiny | 108 (3×3×3 conventional cells × 4 atoms) | Smoke test |
| Small | 4 000 | CI nightly |
| Medium | 32 000 | Single-GPU validation |

## Files (TBD at M2)

- `generate.py` — picks a random Ni:Co:Cr composition (e.g. 33/33/34) and writes a LAMMPS data file.
- `lammps/in.nicocr.in` — LAMMPS reference input.
- `expected/` — committed reference numbers.

## Pass criteria

Same tolerance bands as FCC16 (force match `< 1e-5` relative, energy drift `< 1e-7 / atom / ps`).
