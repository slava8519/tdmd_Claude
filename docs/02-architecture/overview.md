# 02 — Architecture Overview

> **High-level map of how TDMD is built.**
> Read this before reading any individual module doc.

---

## 1. Layer cake

```
┌───────────────────────────────────────────────────────────┐
│                       Drivers                              │
│   tdmd_standalone    |    tdmd_lammps_plugin (later)       │
└───────────────────────────────────────────────────────────┘
┌───────────────────────────────────────────────────────────┐
│                       Scheduler                            │
│   Zone state machine, dependency DAG, work queue           │
└───────────────────────────────────────────────────────────┘
┌──────────────────┬────────────────────┬───────────────────┐
│   Integrator     │     Potentials     │  NeighborList     │
│   velocity-Verlet│   Morse / EAM / ML │  Verlet + cells   │
└──────────────────┴────────────────────┴───────────────────┘
┌──────────────────┬────────────────────┬───────────────────┐
│      Domain      │       Comm         │   Telemetry       │
│  zones, cells    │  MPI / NCCL ring   │  NVTX, counters   │
└──────────────────┴────────────────────┴───────────────────┘
┌───────────────────────────────────────────────────────────┐
│                       Core                                 │
│   SystemState (SoA), Box, Types, Errors, Logging          │
└───────────────────────────────────────────────────────────┘
┌───────────────────────────────────────────────────────────┐
│                          IO                                │
│   LAMMPS data reader, dump writer, restart                 │
└───────────────────────────────────────────────────────────┘
```

Lower layers do not depend on higher layers. **Core has no dependencies on anything else in the project.**

## 2. Module catalogue

| Module | Owns | Depends on | Purpose |
|---|---|---|---|
| `core` | `SystemState`, `Box`, `Vec3`, `Error`, `Log` | — | Fundamental data types and utilities. Header-only where possible. |
| `io` | data file parser, dump writer | `core` | LAMMPS-compatible IO. Stays single-threaded. |
| `domain` | `Cell`, `Zone`, partition | `core` | Spatial partitioning into cells and zones. |
| `neighbors` | `NeighborList` | `core`, `domain` | Verlet lists with skin, GPU build. |
| `potentials` | `IPairStyle`, `MorsePair`, `EamAlloy` | `core`, `neighbors` | Force/energy kernels. Plugin interface. |
| `integrator` | `VelocityVerlet`, `NoseHoover` | `core`, `potentials` | Time integration of equations of motion. |
| `scheduler` | `Zone` state machine, `Scheduler` | `core`, `domain`, `integrator`, `potentials`, `comm` | TD pipeline. The brain of the system. |
| `comm` | `IComm`, `MpiRingComm` | `core` | Network abstraction. Two backends: GPU-aware and host-staged. |
| `telemetry` | NVTX wrappers, counters, breakdown printer | `core` | Observability. Always on, low overhead. |
| `drivers` | `tdmd_standalone` main | everything | Entry points and CLI. |

## 3. Key data structures

### SystemState (in `src/core/system_state.hpp`)

The single canonical representation of the simulation. SoA layout, lives on the device.

```cpp
struct SystemState {
  int64_t natoms;
  Box box;

  // device arrays (cuda::std::span or raw device pointers wrapped in DeviceBuffer)
  DeviceBuffer<float3>  positions;    // shape: [natoms]
  DeviceBuffer<float3>  velocities;   // shape: [natoms]
  DeviceBuffer<float3>  forces;       // shape: [natoms]
  DeviceBuffer<int32_t> types;        // shape: [natoms]
  DeviceBuffer<int32_t> ids;          // shape: [natoms], stable global IDs

  // per-type
  std::vector<float> masses;          // host-side
  std::vector<std::string> type_names;

  // simulation clock
  int64_t step{0};
  double  time{0.0};
};
```

Notes:
- `float3` for positions/velocities/forces. Mixed precision is the default; FP64 mode keeps the same layout but uses `double3`.
- `ids` are stable across migrations; this is how we cross-check with LAMMPS.
- Forces are zeroed at the start of every force pass.

### Zone (in `src/scheduler/zone.hpp`)

See `docs/01-theory/zone-state-machine.md` for the full state machine. The struct holds:

```cpp
struct Zone {
  int32_t id;
  std::array<int32_t, 3> lattice_index;
  Aabb bbox;
  int32_t natoms_in_zone;
  int32_t atom_offset;
  int32_t time_step;
  ZoneState state;
  int32_t owner_rank;
  cudaEvent_t done_event;
  // ...
};
```

### NeighborList (in `src/neighbors/neighbor_list.hpp`)

```cpp
struct NeighborList {
  DeviceBuffer<int32_t> neighbors;        // flat list, neighbors of atom i
  DeviceBuffer<int32_t> neighbor_offsets; // CSR-style offsets
  DeviceBuffer<int32_t> neighbor_counts;
  float r_cut;
  float r_skin;
  int64_t built_at_step;
  // ...
};
```

CSR layout for memory efficiency and coalesced access on GPU.

## 4. Threading and async model

- **One CUDA context per rank.** No multi-context games.
- **Stream pool** of size N (default 4). The scheduler acquires a stream when launching work for a zone, releases it when the kernel completes.
- **CUDA events** are used for cross-stream dependencies, never `cudaDeviceSynchronize` in the hot loop.
- **One CPU host thread for IO/telemetry**, separate from the main scheduler thread.
- **MPI ranks** never share a GPU in the basic case (1 rank ↔ 1 GPU). Multi-rank-per-GPU is a tunable for performance experiments only.

## 5. Build system

CMake, single root `CMakeLists.txt`, modular subdirectories. CUDA is enabled at the root level. No fancy meta-build systems. Use `Ninja` as the generator.

```
cmake -B build -G Ninja
cmake --build build
ctest --test-dir build
```

See `docs/04-development/build-and-run.md` for the full guide.

## 6. The role of LAMMPS in this architecture

LAMMPS is **not a runtime dependency** of TDMD. It is:
1. A reference for input/output file formats (we read LAMMPS data files; we write LAMMPS-format dumps).
2. The **oracle** for VerifyLab. We run LAMMPS on the same input and compare numbers.
3. Optionally (M8+), embedded as a library by `tdmd_lammps_plugin` for direct A/B work.

We do **not** link against LAMMPS for the standalone driver. The standalone is self-sufficient.

## 7. Cross-cutting concerns

### Logging

Use `tdmd::log::info / warn / error / debug`. No `printf`, no `std::cout`. Logging is configurable per module via env var `TDMD_LOG=scheduler:debug,potentials:info`.

### Error handling

- C++ exceptions for recoverable user errors at IO/setup time.
- `TDMD_CHECK_CUDA(call)` macro for CUDA errors — turns them into exceptions in debug, into `std::abort` with a clear message in release.
- `TDMD_ASSERT(cond, msg)` for invariants that should never fail. Always on, even in release builds, for the first year of development.

### Determinism

Two modes:
- **Fast mode** (default): allow non-deterministic floating-point order, mixed precision, fastest available kernels.
- **Deterministic mode**: enforce stable summation order, FP64 accumulation, fixed CUDA stream count = 1. Used by VerifyLab and CI.

Selected via CLI `--deterministic` or env var `TDMD_DETERMINISTIC=1`.

### Telemetry

NVTX ranges around every major phase: `force_compute`, `integrate_stage1`, `integrate_stage2`, `neighbor_build`, `comm_send`, `comm_recv`, `schedule`. Use `nsys profile` to visualize.

In-process counters maintained in `src/telemetry/counters.hpp`. Printed at end of run in LAMMPS-style breakdown:

```
Loop time of 12.34 on 1 procs for 50000 steps with 4096 atoms

Performance: 4054.4 timesteps/s
99.5% CPU use with 1 MPI tasks x 1 OpenMP threads

MPI task timing breakdown:
Section    | min time  | avg time  | max time |%total
-----------+-----------+-----------+----------+------
Pair       | 7.1234    | 7.1234    | 7.1234   | 57.7
Neigh      | 1.5678    | 1.5678    | 1.5678   | 12.7
Comm       | 0.2345    | 0.2345    | 0.2345   |  1.9
Integrate  | 1.8901    | 1.8901    | 1.8901   | 15.3
Output     | 0.1234    | 0.1234    | 0.1234   |  1.0
Modify     | 0.5678    | 0.5678    | 0.5678   |  4.6
Schedule   | 0.8901    | 0.8901    | 0.8901   |  7.2
Other      | 0.0           | 0.0           | 0.0          |  0.0
```

The `Schedule` line is TDMD-specific and reports the time spent in the TD scheduler outside of compute kernels.

## 8. What this architecture is NOT

- **Not a plugin system for everything.** Potentials and comm backends are pluggable; the integrator and scheduler are not.
- **Not a generic task graph framework.** The scheduler is purpose-built for TD. Don't try to make it solve general DAGs.
- **Not header-only.** Templates are kept narrow; .cpp/.cu compilation is the default.
- **Not coroutine-based.** Standard async with streams and events. No `co_await` for the foreseeable future.

---

## Reading order for new contributors

1. `docs/00-vision.md`
2. `docs/01-theory/time-decomposition.md`
3. `docs/01-theory/zone-state-machine.md`
4. **This file.**
5. `docs/02-architecture/scheduler.md`
6. `docs/02-architecture/parallel-model.md`
7. `docs/02-architecture/gpu-strategy.md`
8. `docs/03-roadmap/milestones.md`

After this you should be able to navigate any code in `src/`.
