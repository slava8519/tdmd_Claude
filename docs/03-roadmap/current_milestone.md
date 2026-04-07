# Current Milestone: M1 — Reference MD (CPU, single-threaded)

> **Goal:** a correct, slow, single-threaded MD code that we can trust as a reference.
> No GPU. No parallelism. No zones. Just an honest implementation we can compare against LAMMPS atom-by-atom.

## Checklist

- [ ] `core/math.hpp` — Vec3 ops, minimum_image
- [ ] `core/constants.hpp` — physical constants
- [ ] `io/LammpsDataReader` — parse LAMMPS data file (atomic style, ortho box)
- [ ] `domain/CellList` — regular grid cell list
- [ ] `neighbors/NeighborList` — Verlet list with skin
- [ ] `potentials/MorsePair` — Morse pair potential
- [ ] `potentials/EamAlloy` — EAM/alloy with setfl reading
- [ ] `integrator/VelocityVerlet` — classic velocity-Verlet NVE
- [ ] `drivers/tdmd_standalone` — full CLI, end-to-end run
- [ ] VerifyLab: two-atoms-morse (analytic, tolerance 1e-12)
- [ ] VerifyLab: run0-force-match-morse (256 atoms, vs LAMMPS < 1e-6)
- [ ] VerifyLab: run0-force-match-eam (256 atoms, vs LAMMPS < 1e-6)
- [ ] VerifyLab: nve-drift-morse (50k steps, |dE/E| < 1e-4)

## Exit criteria

- [ ] Two-atom Morse: forces and energy match analytic solution to 1e-12.
- [ ] Run-0 Morse on a 256-atom box: max force-component error vs LAMMPS < 1e-6 (FP64).
- [ ] Run-0 EAM on a 256-atom box: same tolerance.
- [ ] NVE drift over 50 000 steps at dt=1 fs: |dE/E| < 1e-4.
- [ ] All VerifyLab scenarios pass on `main`.
