# Changelog

All notable changes to TDMD will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.7.0] - 2026-04-08

### Added
- `integrator/device_nose_hoover` — GPU Nosé-Hoover chain (NHC) thermostat with MTTK integration.
  - `device_compute_ke`: two-pass GPU kinetic energy reduction (per-block partial sums, host final sum).
  - `device_scale_velocities` / `device_scale_velocities_zone`: GPU velocity scaling kernels.
  - `device_compute_vmax`: GPU max atomic speed reduction for adaptive Δt.
  - `NoseHooverChain`: host-side MTTK chain integration, returns velocity scale factor.
- NVT support in all three schedulers: PipelineScheduler, DistributedPipelineScheduler, HybridPipelineScheduler.
  - Pipeline is drained each step in NVT mode for correctness (pipelined NVT is future work).
  - Multi-rank schedulers use MPI_Allreduce for global KE before NHC half-step.
- Adaptive Δt mode: `dt = min(dt_max, c2 * rc / v_max)`, opt-in via `PipelineConfig::adaptive_dt`.
- 5 new tests: NVT temperature convergence, device/host KE match, deterministic NVT reproducibility, device/host v_max match, adaptive Δt NVE stability.

### Notes
- **M7 NVT + adaptive Δt complete.** 68 tests passing (63 M0-M6 + 5 M7).
- NVT ⟨T⟩ converges to 300K target within 15% for 256-atom Cu FCC.
- Adaptive Δt NVE drift |dE/E| < 1e-2 over 1000 steps (expected: variable step size breaks symplecticity slightly).
- NPT, kernel optimizations, and roofline analysis deferred to post-M7.
- Next: M8 (ML potentials).

## [0.6.0] - 2026-04-08

### Added
- `domain/SpatialDecomp` — 1D Y-axis slab decomposition for spatial subdomain partitioning.
- `scheduler/HybridPipelineScheduler` — 2D time × space pipeline scheduler with MPI_Cart topology.
- 2D MPI Cartesian communicator: `[P_time, P_space]`, split into time_comm (TD ring) and space_comm (halo exchange).
- Ghost atom halo exchange on space_comm with deduplication by atom ID.
- 3 new M6 tests: DeterministicMatchesM5 (4-rank), PipelineNVEConservation (4-rank), ZoneTimeStepsAdvance (4-rank).

### Fixed
- PBC wrapping in GPU drift kernel: `if/else if` → `if/if` to prevent atoms at `box.hi` after wrap-from-below. This caused atoms to leave their spatial subdomain and break ghost exchange.

### Notes
- **M6 complete.** Core exit criteria met. 63 tests passing (60 M0-M5 + 3 M6 hybrid MPI).
- 4-rank hybrid (P_time=2, P_space=2) matches M4 single-rank within 1e-6.
- 4-rank hybrid NVE conservation |dE/E| < 1e-4 over 500 steps.
- Multi-GPU scaling benchmarks deferred (single-GPU dev machine).
- Next: M7 (NVT/NPT + adaptive Δt + optimizations).

## [0.5.0] - 2026-04-08

### Added
- `comm/MpiRingComm` — MPI ring communicator with async send/recv for zone data exchange.
- `comm/CMakeLists.txt` — MPI-conditional build module.
- Multi-rank zone ownership: `ZonePartition::assign_to_ranks()`, `owner_rank()`, `ghost_zones()`.
- `DeviceBuffer` offset-based `copy_from_host`/`copy_to_host` overloads.
- `scheduler/DistributedPipelineScheduler` — multi-rank TD pipeline with MPI_Sendrecv exchange.
- Boundary zone detection and ghost zone time_step tracking for cross-rank dependencies.
- Build option `-DTDMD_ENABLE_MPI=ON` activates MPI compilation.
- 3 new MPI tests: DeterministicMatchesSingleRank, PipelineNVEConservation, ZoneTimeStepsAdvance.

### Notes
- **M5 complete.** Core exit criteria met. 60 tests passing (57 M0-M4 + 3 M5 MPI).
- 2-rank distributed pipeline matches M4 single-rank within 1e-6.
- 2-rank pipeline NVE conservation |dE/E| < 1e-4 over 1000 steps.
- Full replication (all atoms on each rank). Ghost-only optimization deferred.
- Multi-GPU scaling benchmarks deferred (single-GPU dev machine).
- Next: M6 (2D time × space parallelism).

## [0.4.0] - 2026-04-08

### Added
- `scheduler/StreamPool` — CUDA stream pool with per-stream events for pipeline overlap.
- `potentials/device_morse_zone` — per-zone Morse force kernel (restricted atom range on CUDA stream).
- `integrator/device_velocity_verlet_zone` — per-zone half-kick, drift, zero-forces kernels.
- `scheduler/PipelineScheduler` — full TD pipeline scheduler with dependency DAG.
- Causal dependency check (I-2): neighbor zones must be at time_step >= T-1 and not Computing.
- Deterministic mode: single CUDA stream, sequential zone walk (bit-identical to M3).
- Pipeline mode: multi-stream, zones at different time steps, overlapping computation.
- `PipelineStats` telemetry counters (kernel launches, dep checks, ticks).
- 3 new tests: DeterministicMatchesM3, PipelineNVEConservation, ZoneTimeStepsAdvance.

### Notes
- **M4 complete.** Core exit criteria met. 57 tests passing.
- Deterministic pipeline matches M3 sequential scheduler within 1e-10.
- Pipeline NVE energy conservation |dE/E| < 1e-4 over 1000 steps.
- Performance metrics (occupancy, overhead) deferred to M7.
- Next: M5 (MPI ring parallelization).

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
