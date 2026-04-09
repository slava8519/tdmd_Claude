# Architecture: GPU Strategy

> How TDMD uses NVIDIA GPUs. The first target is RTX 5080 (Blackwell consumer); the design must scale to data-center GPUs (H100 / B100) without rewrites.

## Principles

1. **GPU is the primary path.** CPU code exists as a correctness reference and for tiny debug runs.
2. **One CUDA context per rank.** No multi-context juggling.
3. **SoA layout for everything atom-related.** Positions, velocities, forces, types — separate device buffers.
4. **Async by default.** Streams + events. `cudaDeviceSynchronize` is forbidden in the hot loop.
5. **Mixed precision is the default.** FP32 for forces, FP64 for accumulators where stability demands it. FP64-everywhere mode exists for VerifyLab.
6. **Kernels are small and named.** One responsibility per kernel. Easy to profile, easy to read.

---

## Three levels of parallelism

TDMD has **three independent levels of parallelism** that compose multiplicatively. Understanding this decomposition is critical for correct performance reasoning.

| Level | Scope | Mechanism | Controlled by |
|---|---|---|---|
| 1 | Between ranks | TD ring + SD halo exchange | MPI, `comm` module |
| 2 | Within one GPU, across zones | Batched kernel launches (all Ready zones combined) | `scheduler` + `potentials` |
| 3 | Within one kernel, across atoms | SIMT, one thread per atom | CUDA runtime (automatic) |

### Level 1 — inter-rank (MPI)

TD ring between time groups + SD halo exchange within a time group. Scale: 1 to thousands of GPUs. See `docs/02-architecture/parallel-model.md` for the full description of the 2D `time x space` topology.

### Level 2 — intra-GPU, across zones (batched kernels)

On each tick the scheduler collects **all** zones in state `Ready` that share a compatible `time_step` and launches **one batched kernel** over the combined set of their atoms. A zone is a unit of *scheduling*, not a unit of *GPU work*.

Why this matters: a typical zone on a 256-atom test system contains ~30-100 atoms. A single kernel launch over 85 atoms produces 1 thread block of 256 threads. On an RTX 5080 with 84 SMs, that is **~1.2% occupancy**. Batching all 3 zones into one launch (256 atoms) produces at least 1 block, still small but at least a single launch. On FCC16 medium (32K atoms, ~30 zones), batching produces ~128 blocks of 256 threads = full occupancy, vs 30 separate tiny launches.

### Level 3 — intra-kernel SIMT

One CUDA thread per atom, thousands of threads per launch. The CUDA runtime distributes thread blocks across SMs automatically. The programmer does not control this directly — it is a consequence of the grid/block dimensions chosen at launch time. The only design requirement is that kernels are written in a thread-per-atom style with coalesced memory access patterns.

### How the levels compose

```
Total parallelism = P_ranks  x  N_atoms_in_batch  x  (SIMT automatic)
                    Level 1     Level 2                Level 3
```

Level 1 scales the number of GPUs. Level 2 fills each GPU. Level 3 is free (hardware). All three are orthogonal.

**Common misconception:** "filling one GPU is the job of spatial decomposition." This is **incorrect**. SD is a Level 1 mechanism (inter-rank communication). Filling one GPU is achieved by Level 2 (batched kernels) and Level 3 (SIMT), which are orthogonal to SD.

---

## Kernel batching strategy

The scheduler groups Ready zones into batched kernel launches. Two layout strategies are possible:

**(a) Mask-based.** All atoms reside in a single contiguous SoA buffer regardless of zone. The kernel receives a bitmask or zone-ID array indicating which atoms belong to active (Ready) zones. Advantage: no data movement on zone state changes. Disadvantage: divergent threads (inactive atoms waste cycles).

**(b) Pre-sorted contiguous ranges.** Atoms are sorted by zone at migration time. Each zone corresponds to a contiguous `(offset, count)` range in the SoA arrays. The kernel receives a list of active ranges and iterates over their union. Advantage: no divergence, compact iteration. Disadvantage: requires re-sorting atoms when zones gain/lose atoms (migration events).

**Current implementation (M4-M7)** uses approach (b): `ZonePartition::assign_atoms()` sorts atoms by zone, and each zone stores `atom_offset` and `natoms_in_zone`. However, the scheduler currently launches **separate kernels per zone** rather than combining ranges into a single launch. See ADR `0005-batched-force-kernels.md` for the migration plan.

The concrete choice between (a) and (b) for the batched implementation will be made based on measurements in Phase 1 of the ADR.

---

## Fused multi-step kernels (TD K>1 bonus)

When K > 1 (multi-step batching), a kernel can perform K consecutive steps of `force -> integrate -> force -> integrate ...` inside a single launch, without returning to the host between steps.

**Why this is unique to TD.** In spatial-decomposition codes (LAMMPS, HOOMD, GROMACS), every integration step requires a halo exchange with neighboring ranks before the next force computation. This forces a kernel launch boundary after every step — the host must orchestrate communication between steps. In TD, zones that have no cross-rank dependencies can advance multiple steps locally, making fusion physically possible.

**Expected benefit.** On small systems where kernel launch overhead dominates (< 10K atoms per rank), fused kernels can provide 1.3-3x speedup by amortizing launch overhead across K steps. On large systems, the benefit is smaller (launch overhead is already a small fraction of compute).

**Requirements for fusion.**
- The integrator must be implementable entirely within a single kernel, without host-side logic between consecutive steps. This means: no host-side reductions, no host-side conditionals, no host-to-device copies between steps.
- The neighbor list must remain valid for K steps (skin must be large enough).
- Force accumulation must use device-side buffers only.

**Architectural constraint:** do not write integrator code that requires host-side logic between consecutive velocity-Verlet steps. This blocks fusion and must be treated as a design error. NVT thermostat (NHC) inherently requires host-side chain updates between steps; fusion applies only to NVE zones or zones where the thermostat can be applied at K-step boundaries rather than every step.

**Status:** not yet implemented. Planned as a separate ADR building on `0005-batched-force-kernels.md`.

---

## Single-GPU degenerate case

When `P_time=1, P_space=1` (single GPU, no MPI), TDMD must operate with **zero TD overhead**:

- All zones have the same `time_step` at all times.
- The scheduler detects this and collapses all zones into one batched launch per phase (force, integrate). No per-zone launch boundaries.
- Ring communication is disabled (no MPI neighbors).
- The dependency check trivially passes for all zones (all at the same step).
- The result is equivalent to a non-TD single-GPU MD code.

**Performance contract:** single-GPU TDMD must achieve throughput within **1.5x** of single-GPU LAMMPS-GPU on the same input and hardware. If it is slower, this indicates a Level 2 deficiency (under-utilization of the GPU due to per-zone launches or excessive launch overhead) and requires investigation.

This contract is verified by the benchmark in `benchmarks/single_gpu_vs_lammps/` and the VerifyLab case `single-gpu-collapse`.

**Current status:** the scheduler does NOT yet implement zone collapse. See ADR `0005-batched-force-kernels.md`.

---

## Memory layout

```
SystemState (device-resident):
  positions  : float3[natoms]   or double3[natoms] in FP64 mode
  velocities : float3[natoms]
  forces     : float3[natoms]
  types      : int32_t[natoms]
  ids        : int32_t[natoms]   stable global IDs

NeighborList (device-resident, CSR):
  neighbors        : int32_t[total]
  neighbor_offsets : int32_t[natoms + 1]
  neighbor_counts  : int32_t[natoms]
```

Padding to 32-element boundaries for warp coalescing where it matters.

---

## Stream model

A pool of CUDA streams (default 2-4). Streams exist to **overlap phases** (force compute, communication, neighbor rebuild, host-device copies), **not** to parallelize zone computations. Zone parallelism comes from batched kernel launches (Level 2), not from multiple streams.

Typical assignment:

```
stream 0:  [batched force compute]  [batched integrate]
stream 1:  [recv H2D]              [send D2H]
stream 2:  [neighbor rebuild]
```

Cross-stream dependencies are handled with `cudaEvent_t`. There is no `cudaStreamSynchronize` in the inner loop. A pool size of 2-4 is sufficient; increasing beyond 4 yields no benefit because there are only a few distinct phases to overlap.

**Note on current implementation:** the M4-M7 scheduler assigns one stream per zone and launches separate kernels per zone. This is a known deviation from the target model. See ADR `0005-batched-force-kernels.md`.

---

## Kernel inventory (planned)

| Kernel | Module | Notes |
|---|---|---|
| `morse_pair_force` | potentials | Per-atom, neighbor-list driven |
| `eam_density` | potentials | Stage 1 of 3 — gather neighbor distances -> rho_i |
| `eam_embed` | potentials | Stage 2 — F(rho) and dF/drho |
| `eam_force` | potentials | Stage 3 — scatter pair + embedding contributions |
| `velocity_verlet_stage1` | integrator | v(t+dt/2), x(t+dt) |
| `velocity_verlet_stage2` | integrator | v(t+dt) using new forces |
| `cell_list_build` | neighbors | Bin atoms into cells |
| `neighbor_list_build` | neighbors | Cell -> Verlet conversion |
| `skin_check` | neighbors | Compute v_max, decide rebuild |

---

## Mixed precision rules

- **Positions**: FP32 (default) or FP64.
- **Velocities**: same as positions.
- **Forces**: FP32 accumulation in default mode; FP64 atomic accumulation in deterministic mode.
- **Energy reductions**: always FP64.
- **Neighbor distances**: FP32 (default).
- **Time integration coefficients (dt, masses)**: always FP64.

VerifyLab tolerances are tuned to this rule.

---

## Profiling

Always-on NVTX ranges around: `force_compute`, `neighbor_build`, `integrate_stage1/2`, `comm_send`, `comm_recv`, `schedule_tick`. Use `nsys profile` for visual analysis. The `Schedule` line in the LAMMPS-style breakdown printer reports the time spent outside compute kernels.

---

## Anti-patterns (explicit prohibitions)

The following patterns are **explicitly prohibited** in TDMD's GPU code. Each anti-pattern includes references to any current code that violates it (known deviations tracked for migration).

### 1. `cudaDeviceSynchronize()` in the hot loop

Forces the CPU to wait for the entire GPU pipeline to drain. Kills all async overlap.

### 2. One kernel launch per zone

A zone with 30-100 atoms produces 1 thread block on a GPU with 84 SMs. This is ~1% occupancy. All Ready zones with compatible time_step must be combined into a single batched launch.

> **Current violation:** `PipelineScheduler::launch_zone_step()` in `src/scheduler/pipeline_scheduler.cu` (lines 122-158) launches 5 separate kernels (half_kick, drift, zero_forces, morse_zone, half_kick) for each individual zone. `compute_morse_gpu_zone()` in `src/potentials/device_morse_zone.cu` (lines 84-96) takes `(first_atom, atom_count)` for a single zone range. Migration: ADR `0005-batched-force-kernels.md`.

### 3. One CUDA stream per zone

The stream pool is a mechanism for overlapping distinct phases (compute / comm / memcpy), not for parallelizing zone computations. A pool of 2-4 streams is sufficient. Assigning one stream per zone does not improve occupancy — it just creates many tiny independent command queues, each with a single tiny kernel.

> **Current violation:** `PipelineScheduler::tick()` in `src/scheduler/pipeline_scheduler.cu` (lines 179-209) acquires a separate stream for each Ready zone and launches zone kernels individually on each stream. Migration: ADR `0005-batched-force-kernels.md`.

### 4. Copying zone atoms to a separate buffer before launch

The SoA layout with pre-sorted contiguous zone ranges (or mask-based indexing) allows kernels to operate in-place. Copying atoms to a temporary buffer adds a memcpy per zone per tick, which dominates on small zones.

### 5. Host-device transfers in the hot loop

All atom data must stay device-resident between ticks. The only H-D transfers in the hot loop should be ghost atom exchange buffers (pinned host staging) and small scalars (e.g., v_max for adaptive dt).

### 6. `cudaMallocAsync` in the hot loop

All device buffers must be pre-allocated at startup or at upload time. Dynamic allocation in the hot loop incurs driver overhead and fragmentation.

### 7. Host-side logic between consecutive integrator steps

Any host-side reduction, conditional, or transfer between consecutive velocity-Verlet steps blocks kernel fusion at K>1. The integrator must be expressible as a pure device-side sequence. Exception: NVT thermostat requires host-side NHC chain updates; fusion applies only to NVE zones or at K-step boundaries.

---

## What we do NOT do (for now)

- No CUDA Graphs (M8+ optimization, after we're sure the topology is stable).
- No persistent kernels.
- No CUTLASS / cuBLAS calls in the hot loop. EAM and Morse are bandwidth-bound, not GEMM-bound.
- No multi-GPU per rank.
- No CPU-GPU oversubscription games.
- No `__syncthreads` tricks until profiling proves they help.
