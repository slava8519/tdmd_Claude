# Current Milestone: M1 — Reference MD (CPU, single-threaded)

> **Goal:** a correct, slow, single-threaded MD code that we can trust as a reference.
> No GPU. No parallelism. No zones. Just an honest implementation we can compare against LAMMPS atom-by-atom.

## Checklist

- [x] `core/math.hpp` — Vec3 ops, minimum_image
- [x] `core/constants.hpp` — physical constants
- [x] `io/LammpsDataReader` — parse LAMMPS data file (atomic style, ortho box)
- [x] `domain/CellList` — regular grid cell list
- [x] `neighbors/NeighborList` — Verlet list with skin
- [x] `potentials/MorsePair` — Morse pair potential
- [ ] `potentials/EamAlloy` — EAM/alloy with setfl reading (needs setfl file + LAMMPS for A/B)
- [x] `integrator/VelocityVerlet` — classic velocity-Verlet NVE
- [x] `drivers/tdmd_standalone` — full CLI, end-to-end run
- [x] VerifyLab: two-atoms-morse (analytic, tolerance 1e-12) — via unit test
- [ ] VerifyLab: run0-force-match-morse (256 atoms, vs LAMMPS < 1e-6) — needs LAMMPS
- [ ] VerifyLab: run0-force-match-eam (256 atoms, vs LAMMPS < 1e-6) — needs EAM + LAMMPS
- [ ] VerifyLab: nve-drift-morse (50k steps, |dE/E| < 1e-4) — needs 256-atom system

## Exit criteria

- [x] Two-atom Morse: forces and energy match analytic solution to 1e-12.
- [ ] Run-0 Morse on a 256-atom box: max force-component error vs LAMMPS < 1e-6 (FP64).
- [ ] Run-0 EAM on a 256-atom box: same tolerance.
- [x] NVE drift over 10k steps on 2-atom Morse: |dE/E| < 1e-4.
- [ ] All VerifyLab scenarios pass on `main`.

## Remaining work for full M1 closure

1. **EAM/alloy potential** — setfl file reader + 3-pass force compute. Needs a Cu.eam.alloy file.
2. **LAMMPS A/B validation** — needs LAMMPS installed/submoduled for run-0 force comparison.
3. **256-atom Morse NVE drift** — generate larger FCC data file, run 50k steps.
4. **VerifyLab automation** — wire up the existing VerifyLab framework with actual runners.
