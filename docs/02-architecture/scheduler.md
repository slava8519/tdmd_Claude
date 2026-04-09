# Architecture: Scheduler

> **The brain of TDMD.** Everything in this doc translates the TD method (`docs/01-theory/time-decomposition.md`) and the zone state machine (`docs/01-theory/zone-state-machine.md`) into a concrete software design.

---

## What the scheduler does

The scheduler is the component that:

1. Owns the set of `Zone` objects belonging to this rank.
2. Walks them in a fixed order (the **traversal order**).
3. Maintains each zone's state through the state machine.
4. Checks dependencies (the cutoff-sphere rule).
5. Launches force/integration kernels on CUDA streams.
6. Posts asynchronous sends and receives via `comm`.
7. Polls completions and advances states.
8. Updates telemetry counters.

It does **not** compute forces, integrate equations of motion, or talk to the network directly. It coordinates the components that do.

---

## Where it lives

```
src/scheduler/
├── zone.hpp                # Zone struct + ZoneState enum + transitions
├── zone.cpp
├── traversal_order.hpp     # how to order zones
├── traversal_order.cpp
├── dependency_check.hpp    # cutoff-sphere causality check
├── dependency_check.cpp
├── stream_pool.hpp         # CUDA stream pool
├── stream_pool.cpp
├── scheduler.hpp           # main Scheduler class
└── scheduler.cpp
```

Max ~500 lines per file. If `scheduler.cpp` grows past that, split it.

---

## Public interface

```cpp
namespace tdmd::scheduler {

class Scheduler {
public:
  Scheduler(SystemState& state,
            const ScheduleConfig& cfg,
            potentials::IPairStyle& pair,
            integrator::IIntegrator& integ,
            comm::IComm& comm);

  /// Run one TD "tick": advance the pipeline by one wall-clock step.
  /// Different zones may end up at different time_step values.
  void tick();

  /// Run until every zone has reached at least target_step.
  void run_until(int64_t target_step);

  /// Force a neighbor list rebuild on next opportunity (skin violation).
  void request_rebuild();

  /// Telemetry snapshot for the perf-breakdown printer.
  ScheduleStats stats() const;

private:
  SystemState&            state_;
  std::vector<Zone>       zones_;
  TraversalOrder          order_;
  StreamPool              streams_;
  potentials::IPairStyle& pair_;
  integrator::IIntegrator& integ_;
  comm::IComm&            comm_;
  ScheduleConfig          cfg_;
  ScheduleStats           stats_;
};

} // namespace tdmd::scheduler
```

---

## The main loop (pseudocode)

### Target model (batched launches)

The scheduler collects **all** Ready zones with a compatible `time_step` and launches **one batched kernel** over the combined set of their atoms. A zone is a unit of *scheduling*, not a unit of *GPU work*.

```
function tick():
  poll_completions()           # advance Computing→Done, Sending→Free, etc.
  poll_comm()                  # advance Receiving→Received, Sending→Free

  for zone in order_.next_window():    # window of zones to consider
    case zone.state:
      Free:
        if should_recv(zone):
          comm_.post_recv(zone)
          zone.transition_to(Receiving)

      Received:
        if check_deps(zone, zones_):
          zone.transition_to(Ready)

      Done:
        if should_send(zone):
          comm_.post_send(zone)
          zone.transition_to(Sending)

  # --- Batch launch: combine all Ready zones into one kernel ---
  batch = collect_ready_zones(zones_)     # list of (offset, count) ranges
  if batch is not empty:
    stream = streams_.acquire(Phase::Compute)
    launch_batched_force_kernel(batch, stream)    # one kernel, all Ready atoms
    launch_batched_integrate_kernel(batch, stream) # one kernel, all Ready atoms
    record_event(batch.zones, stream)
    for zone in batch.zones:
      zone.transition_to(Computing)

  update_telemetry()
```

Key difference from a per-zone model: `collect_ready_zones()` builds a **batch descriptor** — a list of `(atom_offset, atom_count)` ranges from all Ready zones. The force and integrate kernels receive this descriptor and iterate over the union of ranges in a single launch. This is critical for GPU occupancy — see `docs/02-architecture/gpu-strategy.md` §"Three levels of parallelism".

### Current implementation (M4-M7)

> **Note:** The current implementation (`PipelineScheduler`, `DistributedPipelineScheduler`, `HybridPipelineScheduler`) launches **separate kernels per zone**, each on its own stream. This is a known deviation from the target model described above. Each zone gets a stream from the pool, and 5 separate kernels (half_kick, drift, zero_forces, morse, half_kick) are launched per zone. This pattern under-utilizes the GPU — a zone with 85 atoms produces 1 thread block on an 84-SM GPU (~1.2% occupancy). Migration to batched launches is tracked in ADR `0005-batched-force-kernels.md`.

---

## Key sub-components

### 1. Traversal order

A `TraversalOrder` is an iterator over zone indices. Multiple implementations:

- `LinearOrder1D` — straight line along the x-axis. The reference implementation. Used in M3–M6.
- `LinearOrder3D` — naive `i + nx*j + nx*ny*k`. Used in M3 for 3D, with the explicit understanding that it's slow.
- `HilbertOrder3D` — space-filling curve. Post-M7 research item, see roadmap.
- `MortonOrder3D` — Z-order curve. Same status.

The order is **fixed at startup** and **identical across ranks** (invariant I-8).

### 2. Dependency check

`bool check_deps(const Zone& z, const std::vector<Zone>& zones)`:

For zone `z` to advance from `Received` to `Ready` at time step `T`, every zone `z'` such that `dist(z, z') ≤ r_c + r_skin + r_buf` must satisfy `z'.time_step ≥ T - 1` and `z'.state ∈ {Received, Ready, Computing, Done}`.

The implementation precomputes a static neighbor list of zones (neighbors-of-zones, not neighbors-of-atoms) at startup, since the zone partition doesn't change every step.

### 3. Stream pool

A pool of 2-4 CUDA streams. Streams exist to **overlap distinct phases** (force compute, communication, neighbor rebuild, host-device copies), **not** to parallelize zone computations. Zone parallelism comes from batched kernel launches (Level 2 parallelism), not from multiple streams.

Target stream assignment:

```
stream 0:  [batched force compute]  [batched integrate]
stream 1:  [recv H2D]              [send D2H]
stream 2:  [neighbor rebuild]
```

Cross-stream dependencies are handled with `cudaEvent_t`. A pool size of 2-4 is sufficient; increasing beyond 4 yields no benefit because there are only a few distinct phases to overlap.

```cpp
class StreamPool {
public:
  explicit StreamPool(int n_streams);
  ~StreamPool();

  std::optional<cudaStream_t> try_acquire();  // nullopt if all busy
  void release(cudaStream_t stream);

  int n_busy() const;
  int n_total() const;
};
```

> **Current deviation:** The M4-M7 implementation assigns one stream per zone and launches separate kernels per zone. This creates many tiny independent command queues, each with a single tiny kernel — it does not improve occupancy. Migration to phase-based stream assignment is part of ADR `0005-batched-force-kernels.md`.

### 4. Completion polling

Each `Computing` zone has a recorded `cudaEvent_t`. We poll with `cudaEventQuery` (non-blocking). On success → `Computing → Done`, increment `time_step`, release stream.

We do **not** use `cudaStreamSynchronize` or `cudaDeviceSynchronize` in `tick()`.

---

## Configuration knobs (`ScheduleConfig`)

| Field | Default | Meaning |
|---|---|---|
| `n_streams` | 4 | CUDA stream pool size (2-4 for phase overlap) |
| `K` | 1 | Multi-step batching factor (M5+) |
| `traversal_order` | `LinearOrder1D` | Zone walk order |
| `r_skin` | 2.0 Å | Verlet skin |
| `r_buf` | auto | Dynamic buffer (per `v_max`) |
| `n_zones_x/y/z` | auto from box | Zone partition |
| `deterministic` | false | Force single stream, FP64 reductions |

Defaults are **tuned for correctness, not speed**. Performance tuning is M7+.

---

## Telemetry the scheduler emits

| Counter | Meaning |
|---|---|
| `time_in_state[state]` | Wall time spent with at least one zone in this state |
| `transitions[from][to]` | How many times each transition fired |
| `pipeline_occupancy` | Average % of zones in `Computing` over the run |
| `stream_busy_fraction` | Average % of streams busy |
| `dep_check_calls` | How often we ran dependency checks |
| `dep_check_failures` | How often deps weren't met |
| `forced_rebuilds` | Skin-violation triggered rebuilds |
| `kernel_launches` | Total force kernel launches |

These show up in the LAMMPS-style breakdown printer at end of run, in the `Schedule` section.

---

## Failure modes the scheduler must handle

| Situation | Response |
|---|---|
| Skin violation detected mid-tick | Drain pipeline, rebuild neighbor list, reset relevant zones |
| Send fails (network error) | Retry once, then crash with clear error |
| Recv times out | Same |
| Dependency cycle (should be impossible if I-8 holds) | `TDMD_ASSERT` |
| Out-of-stream-pool starvation | Block on the oldest event, log warning |
| All zones stuck in `Receiving` | `TDMD_ASSERT` deadlock |

---

## Testing strategy

- **Unit tests** (`tests/unit/test_scheduler.cpp`):
  - State machine fuzz (see `docs/01-theory/zone-state-machine.md`).
  - `LinearOrder1D` returns expected sequence.
  - Dependency check returns correct answer for hand-crafted scenarios.
  - StreamPool acquire/release semantics.
- **Integration tests** (`tests/integration/test_td_pipeline.cpp`):
  - Run a small system through the pipeline, compare against M3 zone-walked reference.
- **VerifyLab cases** (`verifylab/cases/`):
  - `td-pipeline-equivalence` — physics result matches the M3 reference within FP tolerance.
  - `1-vs-N-ranks` — same input, different rank counts, same trajectory in deterministic mode.

---

## Anti-goals

- No general task-graph framework. The scheduler is purpose-built for TD.
- No coroutines.
- No automatic load balancing in the basic design (M7+ research item).
- No adaptive K (K is set at startup).
- No special-case fast paths for `n_zones < 4`.
