# Changelog

All notable changes to TDMD will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0] - 2026-04-08

### Added
- `scheduler/zone.hpp` — Zone struct with ZoneState enum, state machine transitions.
- `domain/ZonePartition` — 1D zone partition along X-axis with atom-zone assignment.
- `scheduler/SequentialScheduler` — sequential zone walker for M3 (all zones same timestep).
- Zone neighbor mapping: precomputed per r_list for dependency checks.
- 8 new tests: zone state machine (3), zone partition (3), scheduler validation (2).

### Notes
- **M3 complete.** All exit criteria met. 54 tests passing.
- Zone-walked MD produces bit-identical results to M2 global compute.
- Next: M4 (TD scheduler with full pipeline).

## [0.2.0] - 2026-04-07

### Added
- `core/DeviceBuffer<T>` — RAII CUDA device memory wrapper (move-only).
- `core/DeviceSystemState` — GPU-resident SoA mirror of SystemState with upload/download.
- `domain/DeviceCellList` — GPU cell list via counting sort + prefix sum.
- `neighbors/DeviceNeighborList` — GPU full neighbor list (lock-free force scatter).
- `potentials/device_morse` — GPU Morse pair force kernel.
- `potentials/device_eam` — GPU EAM/alloy 3-pass force computation (density, embedding, force).
- `integrator/device_velocity_verlet` — GPU velocity-Verlet half-kick and drift kernels.
- `cmake/CUDAConfig.cmake` — CUDA build config (sm_89 native + sm_90 PTX for Blackwell JIT).
- 15 CUDA unit tests covering all GPU modules.
- GPU LAMMPS A/B validation: Morse and EAM run-0 force match < 1e-6.
- GPU NVE conservation: 50k steps, |dE/E| < 1e-4.

### Changed
- `EamAlloy` — added public accessors for spline tables (GPU upload support).
- Build system: `-DTDMD_ENABLE_CUDA=ON` activates CUDA compilation.

### Notes
- **M2 complete.** All exit criteria met. 46 tests passing (31 CPU + 15 GPU).
- Next: M3 (Time Decomposition scheduler).

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
