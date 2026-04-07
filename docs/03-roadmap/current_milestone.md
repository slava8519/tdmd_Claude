# Current Milestone: M1 — Reference MD (CPU, single-threaded) ✅ COMPLETE

> **Goal:** a correct, slow, single-threaded MD code that we can trust as a reference.

## Checklist

- [x] `core/math.hpp` — Vec3 ops, minimum_image
- [x] `core/constants.hpp` — physical constants
- [x] `io/LammpsDataReader` — parse LAMMPS data file (atomic style, ortho box)
- [x] `domain/CellList` — regular grid cell list
- [x] `neighbors/NeighborList` — Verlet list with skin
- [x] `potentials/MorsePair` — Morse pair potential
- [x] `potentials/EamAlloy` — EAM/alloy with setfl reading
- [x] `integrator/VelocityVerlet` — classic velocity-Verlet NVE
- [x] `drivers/tdmd_standalone` — full CLI, end-to-end run
- [x] VerifyLab: two-atoms-morse (analytic, tolerance 1e-12)
- [x] VerifyLab: run0-force-match-morse (256 atoms, vs LAMMPS < 1e-6)
- [x] VerifyLab: run0-force-match-eam (256 atoms, vs LAMMPS < 1e-6)
- [x] VerifyLab: nve-drift-morse (50k steps, |dE/E| < 1e-4)

## Exit criteria — ALL MET

- [x] Two-atom Morse: forces and energy match analytic solution to 1e-12.
- [x] Run-0 Morse on a 256-atom box: max force-component error vs LAMMPS < 1e-6 (FP64).
- [x] Run-0 EAM on a 256-atom box: same tolerance.
- [x] NVE drift over 50 000 steps at dt=1 fs: |dE/E| < 1e-4.
- [x] All 31 unit tests pass on `main`.

## Next: M2 — GPU port (single GPU)
