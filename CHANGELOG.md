# Changelog

All notable changes to TDMD will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-04-07

### Added
- `core/math.hpp` — Vec3 arithmetic, minimum_image, wrap_position.
- `core/constants.hpp` — Boltzmann, mvv2e in LAMMPS metal units.
- `io/LammpsDataReader` — parser for LAMMPS data files (atomic style, ortho box).
- `domain/CellList` — O(N) cell list for neighbor search acceleration.
- `neighbors/NeighborList` — Verlet list with skin (half-list, CPU).
- `potentials/MorsePair` — Morse pair potential with energy and force-over-distance.
- `potentials/force_compute` — pair force evaluation with Newton 3rd law.
- `integrator/VelocityVerlet` — velocity-Verlet NVE integrator in metal units.
- `drivers/tdmd_standalone` — full CLI for end-to-end MD simulation.
- GoogleTest (FetchContent), 25 unit tests covering all modules.
- 32-atom Cu FCC test data file.

- `potentials/EamAlloy` — EAM/alloy potential with setfl reader and 3-pass force compute.
- `io/dump_reader` — LAMMPS custom dump file reader for A/B comparison.
- LAMMPS A/B validation: run-0 force match for Morse and EAM (< 1e-6 FP64).
- NVE drift test: 256 atoms, 50k steps, |dE/E| < 1e-4.
- `tools/gen_fcc_data.py` — FCC lattice generator.
- 31 unit tests (GoogleTest).

### Notes
- **M1 complete.** All exit criteria met.
- Force convention: LAMMPS-compatible (delta = r_i - r_j).
- Next: M2 (GPU port).

## [0.0.1] - 2026-04-07

### Added
- Initial project scaffold (M0).
- CLAUDE.md with rules for AI agentic development.
- Full theory documentation (Time Decomposition method, zone state machine).
- Architecture overview, module catalogue, data structures.
- Roadmap M0–M8 with exit criteria.
- VerifyLab subsystem specification.
- Prompts and roles for Claude Code (implementer, architect, reviewer, test-engineer, physicist-validator, doc-writer).
- Workflows: start-new-milestone, add-feature, add-kernel, validate-against-lammps, fix-bug, investigate-performance, update-docs.
- Build system skeleton (CMake + Ninja), CI workflows.
- `scripts/build.sh`, `scripts/run-tests.sh`, `scripts/format.sh` working.
- Smoke test (`tests/unit/test_smoke.cpp`) passing.
- Core headers: `types.hpp`, `box.hpp`, `system_state.hpp`, `error.hpp`.
- Zone state machine header (`scheduler/zone.hpp`).
- Apache 2.0 license.

### Notes
- M0 complete. No working physics yet.
- Next: M1 (single-threaded CPU reference MD).
