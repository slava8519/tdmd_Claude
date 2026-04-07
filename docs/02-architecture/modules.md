# Architecture: Module Catalogue

> Detailed list of every module in `src/`, what it owns, what it depends on, and where to look first.

| Module | Header(s) | Owns | Depends on | Status |
|---|---|---|---|---|
| `core` | `core/system_state.hpp`, `core/box.hpp`, `core/types.hpp`, `core/log.hpp`, `core/error.hpp` | `SystemState` (SoA), `Box`, `Vec3`, `DeviceBuffer`, error/log facilities | — | M0 stub, M1 grows |
| `io` | `io/lammps_data_reader.hpp`, `io/dump_writer.hpp` | LAMMPS data file reader, dump writer, restart files | `core` | M1 |
| `domain` | `domain/cell_list.hpp`, `domain/zone.hpp`, `domain/partition.hpp` | Cell list, zone partition, atom→cell→zone mapping | `core` | M1 (cells), M3 (zones) |
| `neighbors` | `neighbors/neighbor_list.hpp`, `neighbors/skin_check.hpp` | Verlet neighbor list with skin, dynamic v_max tracking | `core`, `domain` | M1 (CPU), M2 (GPU) |
| `potentials` | `potentials/ipair.hpp`, `potentials/morse.hpp`, `potentials/eam_alloy.hpp` | `IPairStyle` interface + concrete implementations | `core`, `neighbors` | M1 (Morse + EAM CPU), M2 (GPU), M8 (ML plugin) |
| `integrator` | `integrator/iintegrator.hpp`, `integrator/velocity_verlet.hpp`, `integrator/nose_hoover.hpp` | Time integration, ensemble thermostats/barostats | `core`, `potentials` | M1 (NVE), M7 (NVT/NPT) |
| `scheduler` | `scheduler/zone.hpp`, `scheduler/scheduler.hpp`, `scheduler/traversal_order.hpp`, `scheduler/dependency_check.hpp`, `scheduler/stream_pool.hpp` | Zone state machine, TD pipeline coordinator, CUDA stream pool, dependency DAG | `core`, `domain`, `integrator`, `potentials`, `comm` | M3 (M3 stub), M4 (full pipeline) |
| `comm` | `comm/icomm.hpp`, `comm/mpi_ring_comm.hpp`, `comm/mpi_cartesian_comm.hpp` | Network abstraction, async send/recv, even/odd phase alternation, GPU-aware MPI | `core` | M5 (TD ring), M6 (2D Cartesian) |
| `telemetry` | `telemetry/counters.hpp`, `telemetry/breakdown_printer.hpp`, `telemetry/nvtx.hpp` | Per-module counters, run-end breakdown printer, NVTX wrappers | `core` | M0 (NVTX), M1 (counters), M5 (per-rank aggregation) |
| `drivers` | `drivers/tdmd_main.cpp` | Standalone executable entry point, CLI parsing | everything | M0 stub, grows with milestones |

## Dependency rules

1. `core` depends on nothing project-internal.
2. Higher modules can depend on lower modules but not vice versa. Use the order above as the dependency hierarchy.
3. No circular dependencies. If you need one, refactor the shared concept into `core`.
4. `drivers/` is the only thing that depends on everything. It is not depended on by anything.

## Where to add a new module

If you need a new top-level module:

1. Justify it via an ADR.
2. Add a row to this table.
3. Create the directory and a `CMakeLists.txt`.
4. Add a stub header and a stub test file.
5. Update `docs/02-architecture/overview.md` if the layer cake changes.
