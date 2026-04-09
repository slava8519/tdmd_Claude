# Changelog

All notable changes to TDMD will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
- **NVT multi-rank atom range bug** in `DistributedPipelineScheduler`. `first_local_atom_`/`local_atom_count_` were computed in the constructor before `assign_atoms()`, resulting in a 0-atom range. The thermostat was effectively a no-op. Fix: recompute after `assign_atoms()` in `upload()`. Regression test: `DistributedPipeline.NVTTemperatureConverges`.
- `MPI_DOUBLE` hardcoded in NVT `MPI_Allreduce` calls in both `DistributedPipelineScheduler` and `HybridPipelineScheduler`. Now uses `sizeof(real) == 8 ? MPI_DOUBLE : MPI_FLOAT` for FP32 correctness.
- Version string updated from `0.1.0-m1` to `0.7.0-dev`.

### Added
- `tests/support/precision_tolerance.hpp` — FP32/FP64-aware tolerance constants and macros (`EXPECT_POSITION_NEAR`, `EXPECT_VELOCITY_NEAR`, `EXPECT_ENERGY_REL_NEAR`). For new tests only.
- `DistributedPipeline.NVTTemperatureConverges` regression test (MPI, 2 ranks).

### Performance (session 3B.7.fix)
- **Fixed mixed mode performance regression** introduced by ADR 0007 design flaw. The original ADR mandated double-precision distance computation in force kernel for "deterministic cutoff check", which produced ~178 FP64 instructions per kernel call and gave a 5-7x slowdown on consumer GPU (RTX 5080, 1:32 FP64:FP32 ratio).
- **Adopted LAMMPS-derived relative-coordinate trick.** Position storage remains `double` (Vec3D) on GPU because TDMD's GPU integrator requires it. But distance computation in force kernel uses one double-subtract followed by force_t cast: `force_t dx = (force_t)(pos_i.x - pos_j.x)`. This is one FP64 instruction per pair instead of ~10, while still being more accurate than pure float subtraction (avoids catastrophic cancellation).
- **Removed epsilon buffer on cutoff** (`* 1.0001f`). LAMMPS doesn't use one; the neighbor list skin distance already provides the needed margin.
- **PBC in force kernel kept but converted from double to force_t.** Moving PBC out of force kernel (LAMMPS's approach) would require image counter for unwrapped coordinates — deferred to potential Phase Б.
- **Same fix applied to neighbor list builder.** Distance computation there is the second major FP64 hotspot.
- **ADR 0007 updated** with revised "Force compute contract" reflecting the hybrid LAMMPS-derived approach. Includes "Why we don't follow LAMMPS exactly" explanation: GPU integrator preservation drives the difference.

**Performance results on RTX 5080 (mixed mode, fast_pipeline scheduler):**

| System | 3B.6 (double dist) | 3B.7.fix (relative trick) | Phase 2 FP32 | Recovery |
|---|---|---|---|---|
| tiny (256) | 3,348 ts/s | 14,814 ts/s | 16,413 ts/s | 90% |
| small (4,000) | 2,090 ts/s | 9,139 ts/s | 9,828 ts/s | 93% |
| medium (32,000) | 1,080 ts/s | 6,927 ts/s | 7,638 ts/s | 91% |

Energy drift in mixed mode: 2.57e-13 per step (target: < 1e-9).

Mixed mode is now usable as the production default. The remaining ~9% gap to Phase 2 FP32 baseline comes from PBC inside force kernel + 3 FP64 subtract instructions per pair from the relative-coordinate trick.

### Phase 3 series — closed

Phase 3 series of seven sub-sessions complete. TDMD now ships with mixed
precision as default build mode, recovering 91% of Phase 2 FP32 performance
while maintaining FP64-quality trajectory and energy conservation. See
`docs/03-roadmap/current_milestone.md` for full Phase 3 summary.

**Key sessions in chronological order:**
- Session 1 (hygiene): docs sync, CI compile-only safety net
- Session 2 (critical bugs): NVT multi-rank atom range fix, MPI type fix
- Session 3A + 3A.1 (precision contract): ADR 0007 design, type aliases
- Session 3B.1–3B.6 (implementation): build system, MPI helper, reductions, force kernels, integrator, test tolerance migration
- Session 3B.7 (math intrinsics hygiene): device_math.cuh helper
- Session 3B.7.fix (performance recovery): LAMMPS relative-coordinate trick
- Meta-1 (process rule): ADR 0008 "Copy LAMMPS where applicable"
- Session 3B.closing (this session): commits, backlog, roadmap update

### Process
- **ADR 0008 — Copy LAMMPS where applicable.** Adopted process rule: TDMD copies LAMMPS for general GPU MD problems (precision, force kernel layouts, integrator details, neighbor lists, MPI patterns) and only invents for TD-specific tasks (scheduler, zone state machine, kernel fusion, multi-step batching). Rationale: session 3B revealed that several architectural mistakes in ADR 0007 came from designing without checking LAMMPS first. Reading LAMMPS source takes 30-60 minutes; reinventing wrongly costs 10+ hours per mistake. The rule formalizes the practice.
- **CLAUDE.md updated** with corresponding hard rule pointing to ADR 0008.

### Changed
- CI: `fail-fast: false` in build matrix so Debug/Release failures are reported independently.

### Architecture (session 3A)
- ADR 0007 — Precision contract. Defines LAMMPS-style mixed precision strategy: positions/velocities double, forces float, reductions always double. Replaces upcoming `TDMD_FP64` flag with `TDMD_PRECISION={mixed,fp64}`. Pure FP32 mode removed in 3B.
- ADR 0006 (distributed scaffold honesty) updated with session 2 findings: silent NVT no-op was previously undetected, MPI_DOUBLE hardcoding was latent UB in FP32 distributed path.
- New role-based type aliases in `src/core/types.hpp`: `pos_t`, `vel_t`, `force_t`, `accum_t`, `PositionVec`, `VelocityVec`, `ForceVec`. In session 3A these are aliases for `real` (no behavior change). In 3B they will diverge per ADR 0007.
- `accum_t` is the only alias that diverges from `real` in 3A: it is always `double`, providing correct accumulator type for future reduction kernel migration.
- `Vec3T<T>` template + `Vec3D`/`Vec3F` explicit-precision vector types added for session 3B use.
- `tests/support/precision_tolerance.hpp` extended with placeholders: `kReductionTolerance`, `kEnergyDriftPerStepMixed`, `kEnergyDriftPerStepFP64`, `kAnalyticTolerance`, `EXPECT_ANALYTIC_NEAR`. Defined but not used until 3B.
- Roadmap updated with Phase 3 (session 3B) backlog item for mixed precision implementation.
- ADR 0007 refined in session 3A.1: tightened energy drift targets to `1e-12`/step (mixed) matching LAMMPS, EAM spline coefficients moved from `float` to `double` in both modes (constant tables don't benefit from FP32, accuracy matters near embedding curve minima), explicit "Force compute contract" section added with position-in-double / distance-in-double / force-expr-in-float pattern, and PE atomicAdd approach resolved to direct `atomicAdd(double*)` call.

### Architecture
- ADR 0005 (batched kernels) — **Implemented**. `FastPipelineScheduler` in `src/scheduler/fast_pipeline_scheduler.{cuh,cu}`.
- Stream parameter (`cudaStream_t stream = 0`) added to 5 device functions (`device_half_kick`, `device_drift`, `device_zero_forces`, `compute_morse_gpu`, `DeviceNeighborList::build`). Backward-compatible via default argument.

### Performance
- TDMD small (4000 atoms, Morse, FP32): 9,828 ts/s (vs old 566 ts/s, **17.4x improvement**).
- TDMD medium (32000 atoms, Morse, FP32): 7,638 ts/s (vs old 517 ts/s, **14.8x improvement**).
- TDMD medium is **2.38x faster** than LAMMPS-GPU on same hardware (RTX 5080, Morse potential).
- TDMD small reaches 79% of LAMMPS-GPU performance (bottleneck: neighbor list rebuild).
- Energy conservation: FP64 drift 3.93e-16 (machine epsilon), FP32 drift 3.52e-07 at 10k steps.

### Added
- `scheduler/FastPipelineScheduler` — batched single-GPU scheduler (ADR 0005 Phase 2).
  - 5 kernel launches per step (half_kick, drift, zero_forces, morse, half_kick).
  - Dedicated non-default CUDA stream, single sync at end of `run_until()`.
  - `FastPipelineStats` telemetry: ticks, kernel_launches, rebuilds.
- `benchmarks/phase1_baseline/` — measurement infrastructure with `bench_pipeline_scheduler`, LAMMPS baseline scripts, `validate_inputs.py`.
- `tests/unit/test_fast_pipeline.cu` — NVE conservation, long NVE drift (10k steps), kernel launch invariant tests.
- `docs/05-benchmarks/phase2-batched-scheduler-results.md` — full results and analysis of Phase 2.
- LAMMPS baseline inputs for small and medium systems.

### Documented
- README synced with reality: M0-M7 complete, Phase 2 results, known limitations explicit. Quickstart verified locally.
- `docs/04-development/build-and-run.md` rewritten: actual CLI (`--data`), all build modes (CPU/CUDA/FP64/MPI/benchmarks), benchmark runner guide.
- `docs/03-roadmap/current_milestone.md` replaced with honest status: closed items, prioritized backlog, known issues table.
- M5/M6 honestly described as full-replication scaffold in `milestones.md`.
- ADR 0006: distributed-scaffold-honesty — documents gap between scaffold and production distributed MD.
- `docs/04-development/ci-strategy.md` — CI coverage gaps, manual check requirements, path to GPU runner.
- ADR 0005 status updated to Implemented with full measurement tables and nsys breakdown.
- Roadmap updated: Phase 3a (neighbor list optimization), Phase 4 (EAM migration) added with estimates.
- M7 performance exit criterion marked as exceeded (2.38x LAMMPS-GPU on medium).
- Measurement exclusivity rule documented (no parallel benchmarks on the same GPU).
- `docs/02-architecture/gpu-strategy.md` — rewritten with three-level parallelism model.
- `docs/02-architecture/scheduler.md` — pseudocode rewritten for batched launch model.
- `docs/02-architecture/parallel-model.md` — added three-level parallelism table.

### CI
- Added `cuda-compile-only` job (TEMPORARY) — compiles `.cu` files on Ubuntu runner without GPU. Catches signature breakage, missing headers, kernel launch syntax errors. Does NOT run GPU tests.
- Added `mpi-compile-only` job (TEMPORARY) — compiles MPI code with OpenMPI. Does NOT run multi-rank tests.
- Both jobs marked TEMPORARY in CI display name. Permanent solution requires self-hosted GPU runner.
- `scripts/build.sh` documented as CPU-only convenience wrapper.

### Notes
- **ADR 0005 Phase 2 complete.** Per-zone PipelineScheduler preserved unchanged as baseline.
- FP32 is production-safe: linear drift extrapolates to 1e-3 threshold at ~30M steps.
- Main remaining bottleneck: `DeviceNeighborList::build` (42% GPU time, 175 us/call vs LAMMPS 60 us/call).
- **Known issues:** NVT multi-rank atom range bug (session 2), FP32 test tolerances (session 3). See `current_milestone.md`.

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
