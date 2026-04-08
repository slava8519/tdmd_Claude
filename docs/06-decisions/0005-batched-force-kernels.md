# 0005 — Batched force kernels: one launch per tick, not per zone

- **Status:** Implemented (Phase 2)
- **Date:** 2026-04-08 (proposed), 2026-04-08 (implemented)
- **Decider:** human + architect
- **Affected milestone(s):** M8+ (migration from M4-M7 per-zone launch model)

## Context

The current scheduler (M4-M7) launches **separate force and integration kernels per zone**. Each zone gets its own CUDA stream, and `launch_zone_step()` fires 5 kernels (half_kick, drift, zero_forces, morse_pair_force, half_kick) for each individual zone.

This is an anti-pattern for GPU utilization:

- A typical zone on a 256-atom system contains ~85 atoms → 1 thread block of 256 threads.
- RTX 5080 has 84 SMs. One thread block = **~1.2% occupancy**.
- On FCC16 medium (32K atoms, ~30 zones), 30 separate tiny launches vs 1 batched launch with ~128 blocks of 256 threads (full occupancy).
- Multiple streams per zone create many tiny independent command queues, adding launch overhead without improving occupancy.

The target model has always been batched launches (documented in `docs/02-architecture/gpu-strategy.md` since M3), but the M4-M7 implementation used per-zone launches for development speed and correctness focus. Now that M7 is complete and correctness is established, this deviation needs to be addressed.

### Current violations (baseline for migration)

1. `PipelineScheduler::launch_zone_step()` in `src/scheduler/pipeline_scheduler.cu` (lines 122-158): launches 5 separate kernels for each individual zone.
2. `compute_morse_gpu_zone()` in `src/potentials/device_morse_zone.cu` (lines 84-96): takes `(first_atom, atom_count)` for a single zone range.
3. `PipelineScheduler::tick()` in `src/scheduler/pipeline_scheduler.cu` (lines 179-209): acquires a separate stream for each Ready zone.

## Options considered

### Option A — Mask-based batching

All atoms reside in a single contiguous SoA buffer regardless of zone. The kernel receives a bitmask or zone-ID array indicating which atoms belong to active (Ready) zones. Inactive atoms are skipped via per-thread conditional.

- Pros: no data movement on zone state changes; simple atom layout; easy to implement.
- Cons: divergent threads (inactive atoms waste cycles); warp divergence hurts on small zone counts; extra memory for mask array.

### Option B — Pre-sorted contiguous ranges (recommended)

Atoms are sorted by zone at migration time. Each zone corresponds to a contiguous `(offset, count)` range in the SoA arrays. The kernel receives a list of active ranges and iterates over their union.

- Pros: no divergence; compact iteration; works naturally with the existing `ZonePartition::assign_atoms()` which already sorts atoms by zone and stores `atom_offset` + `natoms_in_zone` per zone.
- Cons: requires re-sorting atoms when zones gain/lose atoms (migration events, ~every 10-100 steps).

### Option C — Do nothing (keep per-zone launches)

- Pros: no code changes; correctness already validated.
- Cons: ~1% GPU occupancy on small systems; severe performance deficit on RTX 5080 (84 SMs); blocks the single-GPU performance contract (within 1.5x of LAMMPS-GPU); blocks fused multi-step kernels (K>1).

## Decision

**Option B — pre-sorted contiguous ranges.** This leverages the existing atom-zone sorting infrastructure and avoids warp divergence. The concrete choice between A and B should be confirmed by measurement in Phase 1 (below).

## Migration plan

The migration proceeds in 3 phases, each independently testable:

### Phase 1 — Measurement baseline

- [x] Benchmark current per-zone launch model on Cu FCC tiny (256), small (4000), medium (32000 atoms).
- [x] Measure: kernel launch count per tick, GPU occupancy (via `nsys`), total timesteps/s.
- [x] Decision: Option A (mask-based) skipped. Simpler approach taken — whole-system kernels without batching per zone, since single-GPU degenerate case (all zones at same time_step) means all atoms are always active.

### Phase 2 — Batched kernel implementation

- [x] `FastPipelineScheduler` created in `src/scheduler/fast_pipeline_scheduler.{cuh,cu}`.
- [x] Whole-system kernels: 5 launches per step (half_kick, drift, zero_forces, morse, half_kick).
- [x] Dedicated non-default CUDA stream; single `cudaStreamSynchronize` at end of `run_until()`.
- [x] Stream parameter (`cudaStream_t stream = 0`) added to 5 device functions (back-compatible).
- [x] All existing tests pass unchanged. 3 new tests added (NVE conservation, long NVE drift, kernel launch invariant).

### Phase 3 — Fused multi-step kernels (K>1)

- [ ] When K > 1, the batched kernel performs K consecutive steps of `force → integrate → force → integrate ...` inside a single launch.
- [ ] Requires: neighbor list valid for K steps (skin large enough), device-side force accumulation only, no host logic between steps.
- [ ] Only applicable to NVE zones or zones where thermostat is applied at K-step boundaries.
- [ ] Separate ADR for fusion details (builds on this ADR).

## Consequences

- **Positive:** GPU occupancy jumps from ~1% to near-full on medium+ systems. Single-GPU performance contract (1.5x LAMMPS-GPU) becomes achievable. Enables fused multi-step kernels. Simplifies stream pool (fewer streams needed).
- **Negative:** kernel signatures become slightly more complex (batch descriptor instead of single range). Migration requires careful testing to preserve bit-identical deterministic results.
- **Risks:** re-sorting atoms on migration adds overhead (mitigated: migration is infrequent, ~every 10-100 steps). Batched kernel may have slightly different FP accumulation order (mitigated: deterministic mode uses atom-id-sorted reduction).
- **Reversibility:** easy — the per-zone launch model still works, just slower. Both models can coexist during migration.

## Results (2026-04-08)

### Implementation

- **New class:** `FastPipelineScheduler` in `src/scheduler/fast_pipeline_scheduler.{cuh,cu}`.
- **Approach:** new class alongside existing `PipelineScheduler` (preserved unchanged as baseline and for multi-rank future work). No modifications to existing scheduler, potentials, or integrator code beyond adding `cudaStream_t stream = 0` parameter to 5 device functions.
- **Invariant:** exactly 5 kernel launches per step, verified at runtime by `FastPipelineScheduler.KernelLaunchInvariant` test.

### Performance measurements

Hardware: NVIDIA RTX 5080 (sm_120, Blackwell), CUDA 12.6, single GPU.
Potential: Morse (D=0.3429, alpha=1.3588, r0=2.866, rc=6.0), Cu FCC systems.
Protocol: 1000 steps + 100 warmup, exclusive GPU access (no parallel workloads).

| System | Atoms | Old PipelineScheduler (FP64) | FastPipelineScheduler (FP64) | FastPipelineScheduler (FP32) |
|--------|-------|------------------------------|------------------------------|------------------------------|
| tiny   | 256   | 1,487 ts/s                   | 1,897 ts/s (1.28x)          | 16,413 ts/s (11.0x)         |
| small  | 4,000 | 566 ts/s                     | 1,488 ts/s (2.63x)          | 9,828 ts/s (17.4x)          |
| medium | 32,000| 517 ts/s                     | 756 ts/s (1.46x)            | 7,638 ts/s (14.8x)          |

LAMMPS-GPU comparison (same hardware, same Morse potential, FP32):

| System | LAMMPS-GPU (ts/s) | TDMD FP32 (ts/s) | Ratio |
|--------|-------------------|-------------------|-------|
| small  | 12,508            | 9,828             | 0.79x |
| medium | 3,210             | 7,638             | **2.38x** |

### Energy conservation

| Mode | Steps | E0 (eV) | Ef (eV) | |dE/E| |
|------|-------|---------|---------|--------|
| FP64 | 1,000 | -867.645912 | -867.645912 | 3.93e-16 (machine epsilon) |
| FP64 | 10,000 | -867.645912 | -867.645912 | 3.93e-16 |
| FP32 | 1,000 | -867.638916 | -867.638977 | 7.03e-08 |
| FP32 | 10,000 | -867.638916 | -867.639221 | 3.52e-07 |

FP64 achieves perfect symplecticity (drift at machine epsilon). FP32 drift grows linearly; extrapolation to 1e-3 threshold is ~30 million steps — safe for production.

### nsys profiling breakdown (small, FP32, 1000 steps)

| Kernel | % GPU time | Avg (us) | Instances |
|--------|-----------|----------|-----------|
| morse_force | 54.2% | 45.3 | 1,101 |
| build_nlist | 41.8% | 175.0 | 220 |
| half_kick | 2.2% | 0.9 | 2,200 |
| drift | 0.9% | 0.7 | 1,100 |
| zero_forces | 0.6% | 0.5 | 1,101 |

### Remaining optimization targets

1. **`DeviceNeighborList::build`** — 42% of GPU time, 175 us/call vs LAMMPS ~60 us/call (2.9x slower). Contains `cudaStreamSynchronize` for host prefix sum. Main bottleneck. Addressed in Phase 3.
2. **Force kernel vectorization** — horizontal vector operations (LAMMPS-style). Expected 2-3x in compute-bound regime. Addressed in future optimization pass.
3. **Kernel fusion (K>1)** — TD-unique feature, unavailable to spatial-decomposition codes. Expected 1.3-3x on small systems. Addressed in Phase 3 ADR.

## Follow-ups

- [x] Phase 1: measurement baseline
- [x] Phase 2: batched kernel implementation
- [ ] Phase 3: fused multi-step kernels (separate ADR)
- [ ] Phase 3a: neighbor list rebuild optimization (175 us → target 60 us)
- [ ] Phase 4: EAM migration to FastPipelineScheduler
- [x] Update `docs/02-architecture/gpu-strategy.md` anti-patterns section
- [x] Verify single-GPU performance contract: **exceeded on medium** (2.38x LAMMPS-GPU)
