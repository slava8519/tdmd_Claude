# Architecture: LAMMPS Compatibility

> LAMMPS is TDMD's validation oracle and partial input/output compatibility target. This document defines the contract.

## What we copy from LAMMPS

1. **Data file format** — atom positions, box, types, masses. Atomic style only at M1; charge / molecular styles much later.
2. **Potential file formats** — `.eam.alloy` (setfl), Morse parameter blocks. Read but not interpreted differently.
3. **Dump format** — `dump_style custom` with id/type/x/y/z/vx/vy/vz columns. M2+.
4. **Units** — `units metal` (Å, eV, ps, K, bar, g/mol). Hard-coded for M1–M7.
5. **Thermo breakdown columns** — same names and meanings (Step, Temp, PotEng, KinEng, TotEng, Press, etc.).
6. **Performance breakdown style** — the `Loop time / Section / min/avg/max / %total` format at end of run.

## What we do NOT copy

- LAMMPS input script syntax (LAMMPS uses a custom DSL we don't want to reimplement). TDMD has its own simple input format — see `docs/04-development/input-format.md` (TBD).
- Bond/angle/dihedral/improper styles.
- `fix` system in its full generality. We will have a small number of named modules (NVE, NVT-NoseHoover, NPT, output).
- Coulomb long-range (Ewald, PPPM, MSM).
- `kspace_style`, `compute`, `variable`, `region` — far out of M1–M7 scope.

## Where LAMMPS appears in the codebase

| Place | Why |
|---|---|
| `src/io/lammps_data_reader.{hpp,cpp}` | Reads `.lmps` data files |
| `src/io/lammps_dump_writer.{hpp,cpp}` | Writes `dump custom` outputs |
| `src/potentials/eam_alloy.cpp` | Reads setfl `.eam.alloy` files |
| `verifylab/cases/*/lammps/in.case` | LAMMPS reference input scripts (committed) |
| `verifylab/cases/*/reference/lammps.log` | LAMMPS reference logs (committed) |
| `external/lammps/` (M8+) | LAMMPS as a submodule, only if we build the plugin variant |

## VerifyLab A/B protocol

For every physics-meaningful test:

1. Generate or pick a system (e.g. FCC Ni-Co-Cr 256-atom box).
2. Write a LAMMPS input that runs it deterministically (`velocity ... loop geom`, fixed seed).
3. Run LAMMPS, capture log + dump.
4. Translate the same setup into TDMD input.
5. Run TDMD.
6. Compare with `verifylab/runners/compare_with_lammps.py` against the tolerances in the case's `tolerance.toml`.

If TDMD disagrees with LAMMPS in a meaningful way, **TDMD is wrong** until proven otherwise via an ADR.

## What "compatible" means in practice

> A user with a working LAMMPS data file and EAM potential file should be able to run TDMD on the same files with no manual conversion, and get statistically equivalent physics within the tolerances we publish.

That's the bar. Anything less and "LAMMPS-compatible" is marketing.

## Versioning

We track the LAMMPS version we validate against in `docs/05-benchmarks/lammps-baseline.md`. When LAMMPS releases a new stable version, we re-run baselines and bump the documented version. We do not silently follow LAMMPS HEAD.
