# Architecture: Parallel Model

> **How TDMD scales across processors.**
> Read `docs/01-theory/time-decomposition.md` first.

---

## Two axes of parallelism

TDMD parallelizes along **two axes**:

1. **Time** — Time Decomposition. Different ranks compute different `time_step` values for the same atoms. This is the unique TD mode.
2. **Space** — classic spatial decomposition. Different ranks own different regions of the box and exchange ghost atoms on halo boundaries. This is what LAMMPS does.

These compose into a 2D `time × space` grid:

```
P_total = P_time * P_space
```

For pure TD: `P_space = 1`, `P_time = N_ranks`. For pure SD: `P_time = 1`, `P_space = N_ranks`. The interesting cases are mixed.

---

## Why both?

Pure TD is the cleanest, but it has a hard ceiling. The dissertation derives:

```
P_opt(TD) = N_zones / N_min_per_proc
```

Where `N_min_per_proc` is the minimum zones a rank must hold for the pipeline to work (2 in 1D, ~4 in 2D, more in 3D depending on traversal order). Above `P_opt`, adding more ranks doesn't help — every rank already has the minimum number of zones.

To scale past `P_opt`, we use space too. The 2D scheme says: take groups of ranks, give each group the full set of zones (TD), then give each *rank within a group* a spatial subdomain (SD). The TD pipeline runs over groups; SD halo exchange runs within a group.

```
group 0:  [ rank 0 (subdomain A) | rank 1 (subdomain B) ]   ↘
group 1:  [ rank 2 (subdomain A) | rank 3 (subdomain B) ]   →  TD ring across groups
group 2:  [ rank 4 (subdomain A) | rank 5 (subdomain B) ]   ↗
```

Rank 0 talks to rank 1 (SD halo) and rank 2 (TD ring). Two distinct neighbor types, two distinct comm patterns, two MPI communicators.

---

## MPI topology

We create three communicators at startup:

```cpp
MPI_Comm world;             // MPI_COMM_WORLD
MPI_Comm time_comm;         // ranks in the same spatial subdomain, across time groups
MPI_Comm space_comm;        // ranks in the same time group, across spatial subdomains
```

This is exactly the `MPI_Cart_create` + `MPI_Cart_sub` pattern that LAMMPS, GROMACS, and HOOMD use for spatial decomposition. We add `time_comm` on top.

---

## Comm patterns

### TD axis (time_comm) — ring topology

- Each rank has exactly 2 neighbors: previous and next in the ring.
- Even/odd phase alternation to avoid deadlock:
  - Even ranks: recv from prev, then send to next.
  - Odd ranks: send to next, then recv from prev.
- Async with `MPI_Isend`/`MPI_Irecv` and explicit `MPI_Test` polling in `tick()`.
- Payload: positions + velocities of zones being transferred. No forces (recomputed by receiver).
- Multi-step batching: K time steps worth of zone updates in a single message.

### SD axis (space_comm) — halo exchange

- Up to 6 neighbors (3D, face-only). 26 with corners and edges if needed for many-body potentials.
- Synchronous from the rank's POV (the SD halo exchange completes before forces are computed for that step).
- Standard ghost atom protocol: send positions of border atoms, receive ghost positions, compute forces, no force back-communication needed (Newton's-third-law optimization disabled by default for clarity, opt-in for speed).

---

## Why this saves bandwidth

The total bytes per second across the cluster:

- **SD only:** `t_comm ∝ S^(2/3) · N_steps` per step per rank, with `O(6)` neighbors. Hard floor — every step needs a halo exchange.
- **TD only:** `t_comm ∝ S / (K · P)` per step per rank, with `O(2)` neighbors. K reduces both frequency and per-message overhead.
- **TD + SD:** combine. The TD axis carries the bulk of the bandwidth at low frequency; the SD axis is fast but small (only ghost atoms on subdomain borders).

The win comes from two places:

1. **Fewer comm partners (2 vs 6+)** → less MPI overhead, more predictable latency, simpler topology mapping to NVLink/InfiniBand.
2. **Multi-step batching (K)** → linear reduction in message frequency. No other MD method has this knob — they all do exactly one halo exchange per step.

---

## Choosing P_time vs P_space

Tuning rule of thumb:

1. Start by maximizing `P_time` up to `P_opt(TD)` (set `P_space = 1`).
2. If `N_ranks > P_opt(TD)`, set `P_space = N_ranks / P_opt(TD)`.
3. If the cluster's interconnect is bandwidth-bound (most cheap GPU clusters), prefer larger K and larger `P_time`.
4. If the cluster has NVLink-class fabric and many ranks, you can push `P_space` higher because halo exchange is cheap.

These choices live in `ScheduleConfig` and are exposed in the input file:

```
parallel  time 8  space 4  K 3
```

`P_time × P_space` must equal the total rank count.

---

## What deadlocks look like and how we avoid them

The biggest TD-specific risk is a deadlock where every rank is waiting for a recv from the previous rank, but no one is sending. We avoid it three ways:

1. **Even/odd phase alternation** (above).
2. **Non-blocking everything**: every send and recv is `Isend`/`Irecv` with `Test`-based polling in the scheduler tick.
3. **Bounded buffer per rank**: a rank with K future steps already in its outbox blocks on its own outbox before posting more, instead of pushing data onto a peer that hasn't drained.

The full state-machine analysis lives in `docs/01-theory/zone-state-machine.md` and the scheduler tests fuzz it.

---

## Determinism in parallel

In **deterministic mode**, the parallel result is bit-identical to the serial result if and only if:

- The traversal order is identical across ranks (invariant I-8).
- Force accumulation order is fixed (per-zone, atom-id-sorted reduction).
- FP64 is used everywhere.
- Stream pool size is 1.
- Network reordering doesn't matter because we sort received data by atom id before applying.

In **fast mode**, parallel results may differ from serial in the last few bits. The VerifyLab `1-vs-N-ranks` case enforces "controlled drift" tolerances rather than equality.

---

## Out of scope

- **Dynamic rebalancing.** A future research item; see `docs/03-roadmap/milestones.md` post-M7 backlog.
- **Heterogeneous ranks.** Every rank has the same GPU.
- **Sub-rank parallelism beyond CUDA streams.** No CPU OpenMP for the GPU path. The CPU reference path is single-threaded by design.
- **Dynamic K.** K is fixed at startup.
