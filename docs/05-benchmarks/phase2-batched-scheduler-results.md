# Phase 2: FastPipelineScheduler — results and analysis

> Date: 2026-04-08. Hardware: NVIDIA RTX 5080 (sm_120, Blackwell). CUDA 12.6. Single GPU.

## Context

Phase 1 measurement baseline revealed a **~24x performance deficit** vs LAMMPS-GPU on the small (4000-atom) system. Root cause: the M4-M7 `PipelineScheduler` launches **5 separate CUDA kernels per zone per step**. With ~30 zones on a 4000-atom system, this means ~150 tiny kernel launches per step, each processing ~130 atoms (1 thread block, ~1.2% GPU occupancy). Additionally, each zone step used `cudaDeviceSynchronize` for dependency checks, creating serial GPU stalls.

ADR 0005 proposed a 3-phase migration. This document reports Phase 2 results.

## What changed

**New class:** `FastPipelineScheduler` in `src/scheduler/fast_pipeline_scheduler.{cuh,cu}` (see [ADR 0005](../06-decisions/0005-batched-force-kernels.md)).

Key design decisions:
- Whole-system kernels: each kernel processes all atoms in a single launch.
- Exactly 5 kernel launches per step: `half_kick -> drift -> zero_forces -> morse_force -> half_kick`.
- Dedicated non-default CUDA stream (`cudaStreamCreate`) — eliminates implicit sync from legacy default stream.
- Single `cudaStreamSynchronize` at end of `run_until()` — the only sync point in the hot path.
- Neighbor list rebuild amortized (every 10 steps by default), contains internal sync for host prefix sum.

**What was NOT changed:**
- `PipelineScheduler` — preserved unchanged as baseline and for multi-rank TD pipeline.
- Existing tests — all 68 M0-M7 tests pass unmodified.
- Integrator and potential kernels — only gained a `cudaStream_t stream = 0` parameter (back-compatible).

## Method

- **Systems:** Cu FCC lattice (a=3.615 A). Sizes: tiny (256), small (4000), medium (32000 atoms).
- **Potential:** Morse (D=0.3429, alpha=1.3588, r0=2.866, rc=6.0 A).
- **Integration:** NVE, dt=0.001 ps, Verlet skin=1.0 A, neighbor list rebuild every 10 steps.
- **Protocol:** 1000 measurement steps after 100 warmup steps. T_init=300K, seed=42.
- **Measurement exclusivity:** all benchmarks run strictly one at a time, no parallel GPU workloads. This is critical — concurrent GPU benchmarks compete for SMs, L2 cache, and memory bandwidth, producing unreliable numbers.
- **Tools:** `bench_pipeline_scheduler` executable, `nsys` 2026.2, LAMMPS 2023 with GPU package (PTX JIT from compute_90 to sm_120).

## Results

### TDMD performance comparison

| System | Atoms | Old PipelineScheduler (FP64) | FastPipelineScheduler (FP64) | FastPipelineScheduler (FP32) |
|--------|-------|------------------------------|------------------------------|------------------------------|
| tiny   | 256   | 1,487 ts/s                   | 1,897 ts/s (1.28x)          | 16,413 ts/s (11.0x)         |
| small  | 4,000 | 566 ts/s                     | 1,488 ts/s (2.63x)          | 9,828 ts/s (17.4x)          |
| medium | 32,000| 517 ts/s                     | 756 ts/s (1.46x)            | 7,638 ts/s (14.8x)          |

### LAMMPS-GPU comparison (FP32)

| System | LAMMPS-GPU (ts/s) | TDMD FP32 (ts/s) | Ratio |
|--------|-------------------|-------------------|-------|
| small  | 12,508            | 9,828             | 0.79x |
| medium | 3,210             | 7,638             | **2.38x** |

### Energy conservation

| Mode | Steps | E0 (eV)     | Ef (eV)     | |dE/E|    |
|------|-------|-------------|-------------|----------|
| FP64 | 1,000  | -867.645912 | -867.645912 | 3.93e-16 |
| FP64 | 10,000 | -867.645912 | -867.645912 | 3.93e-16 |
| FP32 | 1,000  | -867.638916 | -867.638977 | 7.03e-08 |
| FP32 | 10,000 | -867.638916 | -867.639221 | 3.52e-07 |

## Analysis

### Why FP64 -> FP32 gives 6-10x speedup

RTX 5080 (Blackwell consumer) has an FP64:FP32 throughput ratio of **1:32**. All TDMD kernels use `real` type throughout — in FP64 mode, every arithmetic operation runs at 1/32nd of peak throughput. FP32 mode unlocks the full compute pipeline.

The 6-10x (not 32x) actual speedup indicates we are not purely compute-bound: memory bandwidth, kernel launch overhead, and neighbor list rebuilds are significant contributors.

### Why small benefits more from batched scheduler

On old `PipelineScheduler`, small (4000 atoms, ~30 zones) was **scheduler-bound**: 150 tiny kernel launches per step, each with ~130 atoms. The launch overhead dominated. Batching to 5 launches per step eliminates this overhead → 2.63x improvement in FP64.

On medium (32000 atoms), zones are larger and the old scheduler was closer to compute-bound → only 1.46x improvement from batching.

### Why medium beats LAMMPS but small doesn't

On medium, TDMD's Morse force kernel (45 us) is competitive with LAMMPS, and the neighbor list rebuild cost (175 us) is amortized over 32000 atoms. Total throughput exceeds LAMMPS by 2.38x.

On small, the neighbor list rebuild is the bottleneck. TDMD's `build_nlist` takes 175 us vs LAMMPS ~60 us (2.9x slower). LAMMPS has a more optimized neighbor list builder with horizontal vector operations and a single-pass approach. This 2.9x nlist penalty cancels the force kernel gains on small systems.

### nsys profiling breakdown (small, FP32, 1000 steps)

| Kernel             | % GPU time | Avg (us) | Instances |
|--------------------|-----------|----------|-----------|
| morse_force        | 54.2%     | 45.3     | 1,101     |
| build_nlist        | 41.8%     | 175.0    | 220       |
| half_kick          | 2.2%      | 0.9      | 2,200     |
| drift              | 0.9%      | 0.7      | 1,100     |
| zero_forces        | 0.6%      | 0.5      | 1,101     |

96% of GPU time is in morse_force + build_nlist. Integrator kernels are negligible.

## Next steps

Prioritized backlog with estimates:

1. **Phase 3a: neighbor list optimization.** Target: 60 us/rebuild (from 175 us). Eliminate host prefix sum sync, move to fully GPU-resident build. Expected impact: +10-15% on small, +5% on medium. Effort: ~1 day.

2. **Phase 4: EAM migration to FastPipelineScheduler.** Reuse the same architecture, replace `compute_morse_gpu` with 3-pass EAM (density + embedding + force). Step goes from 5 to 7-8 launches. Invariant "constant N launches per step" preserved. Effort: ~1 day.

3. **Kernel fusion (K>1).** TD-unique feature unavailable to spatial-decomposition codes like LAMMPS. Multiple velocity-Verlet steps fused into a single kernel launch. Expected 1.3-3x speedup on small systems where launch overhead is significant. Effort: ~2-3 days. Separate ADR.

4. **Force kernel vectorization.** Horizontal vector operations (LAMMPS-style: process 4 neighbors per iteration using SIMD lanes). Expected 2-3x in compute-bound regime. Effort: TBD.

5. **Mixed precision mode.** FP32 compute + FP64 accumulators for energy/force sums. Extends production-safe run length beyond 30M steps. Effort: TBD.

## Discovered invariants and lessons learned

1. **Zone is a scheduling unit, not a GPU work unit.** Confirmed empirically: per-zone launches waste GPU occupancy. Zone boundaries matter for TD causal dependencies, not for kernel dispatch.

2. **Neighbor list is potential-neutral.** All nlist optimizations benefit Morse, EAM, and future ML potentials equally.

3. **Integrator kernels are potential-neutral.** half_kick, drift, zero_forces are shared across all force styles.

4. **FastPipelineScheduler architecture is potential-neutral.** Adding EAM requires changing one function call in `step()`, not redesigning the scheduler.

5. **Measurement exclusivity is non-negotiable.** Two concurrent GPU benchmarks produce random numbers. Always run one benchmark at a time with 5-10 second cooling pauses between runs.
