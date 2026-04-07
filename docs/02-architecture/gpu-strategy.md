# Architecture: GPU Strategy

> How TDMD uses NVIDIA GPUs. The first target is RTX 5080 (Blackwell consumer); the design must scale to data-center GPUs (H100 / B100) without rewrites.

## Principles

1. **GPU is the primary path.** CPU code exists as a correctness reference and for tiny debug runs.
2. **One CUDA context per rank.** No multi-context juggling.
3. **SoA layout for everything atom-related.** Positions, velocities, forces, types — separate device buffers.
4. **Async by default.** Streams + events. `cudaDeviceSynchronize` is forbidden in the hot loop.
5. **Mixed precision is the default.** FP32 for forces, FP64 for accumulators where stability demands it. FP64-everywhere mode exists for VerifyLab.
6. **Kernels are small and named.** One responsibility per kernel. Easy to profile, easy to read.

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

## Stream model

A pool of CUDA streams (default 4). Each zone, when launched, gets a stream. Cross-stream dependencies are handled with `cudaEvent_t`. There is no `cudaStreamSynchronize` in the inner loop.

```
stream 0:  [zone A force][zone A integrate]
stream 1:  [zone B force][zone B integrate]
stream 2:  [recv H2D]              [send D2H]
stream 3:  [neighbor rebuild]
```

## Kernel inventory (planned)

| Kernel | Module | Notes |
|---|---|---|
| `morse_pair_force` | potentials | Per-atom, neighbor-list driven |
| `eam_density` | potentials | Stage 1 of 3 — gather neighbor distances → ρ_i |
| `eam_embed` | potentials | Stage 2 — F(ρ) and dF/dρ |
| `eam_force` | potentials | Stage 3 — scatter pair + embedding contributions |
| `velocity_verlet_stage1` | integrator | v(t+dt/2), x(t+dt) |
| `velocity_verlet_stage2` | integrator | v(t+dt) using new forces |
| `cell_list_build` | neighbors | Bin atoms into cells |
| `neighbor_list_build` | neighbors | Cell→Verlet conversion |
| `skin_check` | neighbors | Compute v_max, decide rebuild |

## Mixed precision rules

- **Positions**: FP32 (default) or FP64.
- **Velocities**: same as positions.
- **Forces**: FP32 accumulation in default mode; FP64 atomic accumulation in deterministic mode.
- **Energy reductions**: always FP64.
- **Neighbor distances**: FP32 (default).
- **Time integration coefficients (dt, masses)**: always FP64.

VerifyLab tolerances are tuned to this rule.

## Profiling

Always-on NVTX ranges around: `force_compute`, `neighbor_build`, `integrate_stage1/2`, `comm_send`, `comm_recv`, `schedule_tick`. Use `nsys profile` for visual analysis. The `Schedule` line in the LAMMPS-style breakdown printer reports the time spent outside compute kernels.

## What we do NOT do (for now)

- No CUDA Graphs (M8+ optimization, after we're sure the topology is stable).
- No persistent kernels.
- No CUTLASS / cuBLAS calls in the hot loop. EAM and Morse are bandwidth-bound, not GEMM-bound.
- No multi-GPU per rank.
- No CPU↔GPU oversubscription games.
- No `__syncthreads` tricks until profiling proves they help.
