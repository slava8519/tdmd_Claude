# 01 — Time Decomposition Method

> **This document is the single source of truth for the TD method in TDMD.**
> It is derived from the dissertation at `docs/01-theory/dissertation-source.md`.
> If the dissertation and this document disagree, open an ADR — do not silently change either.

---

## 1. The core observation

In molecular dynamics, atoms interact through a potential with a finite cutoff radius `r_c`. Typical values for metallic EAM potentials are 5–7 Å. The simulation box is typically hundreds of Å on a side. This means the ratio `r_c / L` is small — MD is a **short-range interaction process**.

The key consequence: **atoms in regions separated by more than the interaction diameter `2·r_c` cannot influence each other within a single integration step.**

This is not a physical approximation. It is a hard mathematical fact about the integrator: during one step of velocity-Verlet, forces on atom `i` depend only on neighbors within `r_c`, and atom `i` can only move a distance `v_i · Δt`, which for sane timesteps (1 fs) and sane velocities (thermal) is `< 0.1 Å`. So the "sphere of causal influence" of atom `i` in one step is at most `r_c + v_max · Δt`, and for `Δt = 1 fs` this is essentially `r_c`.

## 2. What we can do with that

If two regions `A` and `B` are separated by more than `2·(r_c + v_max·Δt)`, then:

- The state of `A` at step `h+1` depends only on the state of `A` at step `h` and the state of regions touching `A`.
- It does **not** depend on the state of `B` at step `h`.

So we can compute `A` at step `h+1` **at the same time** as we compute `B` at step `h`. They are causally independent for this one step.

If we arrange regions in a 1D chain along one spatial axis, we get a picture like this:

```
Time step 5:  [A] [ ] [ ] [ ] [ ]
Time step 4:  [A] [B] [ ] [ ] [ ]
Time step 3:  [A] [B] [C] [ ] [ ]
Time step 2:  [A] [B] [C] [D] [ ]
Time step 1:  [A] [B] [C] [D] [E]

Processor:     1   2   3   4   5
```

Each column is one spatial region ("zone"). Each row is one moment of wall-clock time. At wall-clock time 5, processor 1 is computing step 5 of zone A while processor 5 is computing step 1 of zone E — simultaneously, on the same physical data.

This is a **pipeline in time**. Steps flow through processors the way instructions flow through a CPU pipeline. The word "decomposition" refers to decomposing the *time* axis (not space) across processors.

## 3. The zone

The unit of work in TD is the **zone** (Russian: расчётная зона, РЗ). A zone is a contiguous block of space, sized so that atoms inside it interact only with atoms in the zone itself and its immediate neighbors along the traversal order.

**Zone size rule.** In 1D, the zone thickness is `≥ r_c`. A one-cell-wide zone is the minimum; wider zones are allowed and sometimes desirable (fewer zones → less bookkeeping, but coarser pipeline granularity).

**Zone shape.** Rectangular (box-aligned) in the reference implementation. Curved or adaptive zones are a future research item.

**Zone count.** For a box of length `L` along the decomposition axis:
```
N_zones = floor(L / zone_thickness)
```

**Atoms per zone.** Not fixed. Atoms migrate between zones as they move; every `N_rebuild` steps we reassign atoms to zones and rebuild neighbor lists.

## 4. The zone state machine

Each zone has a **state** that changes over time as work and communication happen. The states are:

| State | Meaning |
|---|---|
| `Free` | Zone is empty (or has been transmitted away). Ready to receive new data. |
| `Receiving` | A non-blocking receive is posted; data is coming. |
| `Received` | Data has arrived, waiting for dependencies to be satisfied. |
| `Ready` | All dependencies satisfied, work can start. |
| `Computing` | Force calculation and integration are running (kernel launched). |
| `Done` | Computation complete, data is valid for `time_step = T`. |
| `Sending` | Non-blocking send posted, waiting for completion. |

**Transitions:**

```
Free ──recv_posted──> Receiving
Receiving ──data_arrived──> Received
Received ──deps_met──> Ready
Ready ──kernel_launched──> Computing
Computing ──kernel_done──> Done
Done ──send_posted──> Sending
Sending ──ack──> Free
```

**Critical:** a zone also has a `time_step` field. At any wall-clock moment, different zones on the same processor may have different `time_step` values. That is the whole point of TD.

## 5. The boundary problem

Atoms near the edge of a zone can move across the boundary during one step. If the zone has already been transmitted to the next processor, and an atom tries to move into it, we have a correctness problem.

The dissertation offers three solutions:

### Solution A — Keep extra zones in memory

Each processor holds at least 2 zones in 1D (≥4 in 2D, more in 3D). A zone is not released until its neighbors have been advanced enough that the edge atoms' causal future is guaranteed to be inside this processor's memory.

### Solution B — Verlet-like neighbor tables with skin

Build neighbor lists with a skin buffer `r_skin`. As long as no atom moves more than `r_skin / 2` between rebuilds, the lists are still correct. This allows a zone to remain "owned" for `N_skin` steps without rebuilding, which gives scheduling flexibility.

The condition is:
```
N_skin · v_max · Δt  <  r_skin / 2
```

This is the classic LAMMPS-style approach and it's what TDMD uses.

### Solution C — Dynamic buffer zone width

Compute `r_buf = C · v_max · Δt · N_safe` before each cycle, based on the fastest atom in the relevant zones. This is a runtime-adaptive version of the skin idea. Use it on top of B for safety.

**TDMD implements B and C together**: Verlet lists with `r_skin`, plus a dynamic check on `v_max` that triggers an early rebuild if an atom is about to escape the skin.

## 6. The traversal order

In what order do we process zones? This matters a lot.

**Three requirements from the dissertation:**

1. **Same order on all processors.** Every rank walks zones in the same sequence. Any other choice creates deadlocks or race conditions.
2. **Pipeline startup is fast.** After a few steps, the system should reach a "steady state" where every processor is doing useful work. Slow startup = wasted compute.
3. **Priority to zones in the cutoff sphere.** When processor P chooses what to compute next, it should prefer zones that are within `r_c` of its "current working front." This keeps data locality high.

### 1D case (simplest, the reference implementation starts here)

Walk zones in a straight line: zone 1, zone 2, ..., zone N. Each processor needs a minimum of **2 zones in memory** to form a stable pipeline. Optimal processor count:

```
P_opt = floor(N_zones / 2)
```

### 2D case

A sphere of radius `r_c` can touch up to 4 zones. Minimum zones per processor is ~4. Many traversal orders work; a snake or Z-curve is most common.

### 3D case

**Warning:** naive sequential numbering is catastrophic. The dissertation shows a concrete example where the sphere of influence of the "next" zone intersects ~274 already-computed zones, meaning a processor needs to receive 274 zones before starting the next step. This wrecks the pipeline.

**Better orders** use space-filling curves (Hilbert, Morton) that preserve spatial locality. **TDMD ships with naive sequential order in M1 and moves to Hilbert/Morton as a post-M7 research item** (see roadmap).

## 7. Communication pattern

**Topology:** ring. Rank `k` sends to rank `k+1` and receives from rank `k-1`. Wraparound if needed.

**Asynchronous phases** (even and odd ranks alternate to avoid deadlock):
- Even ranks: first receive, then send.
- Odd ranks: first send, then receive.

**Payload:** for each transmitted zone, we send atom positions + velocities. Forces are not sent (they are recomputed by the receiver on its time step).

**Buffering.** Each rank can batch up to K time steps locally before sending. Increasing K reduces communication frequency linearly:

```
t_comm ∝ S / (K · P)
```

where S is the total atom data. For SD/RD/FD, K is always 1. This is why TD scales better on bandwidth-limited clusters.

## 8. Integration and scaling

Each processor computes **all atoms** of the model, but for a different time step than its neighbors. So the total work across `P` processors in wall-clock time `T` is `P · W`, where `W` is the work of one processor for one step. That means wall-clock speedup is **linear in P** up to `P_opt`.

```
Speedup(P) = T_1 / T_P  ≈  P · (1 - t_comm / t_compute)
```

Efficiency can exceed 95% on well-tuned systems (this is a measured result from the dissertation).

**Beyond P_opt**, pure TD cannot use more processors. That's where the 2D `time × space` scheme kicks in (see `docs/02-architecture/parallel-model.md`).

## 9. How TD compares to other parallel methods

| Property | Atom Decomp (RD) | Force Decomp (FD) | Spatial Decomp (SD) | **Time Decomp (TD)** |
|---|---|---|---|---|
| Compute / proc | O(N/P) | O(N/sqrt(P)) | O(N/P) | O(N) |
| Communication / proc | O(N) | O(N/sqrt(P)) | O((N/P)^{2/3}) | **O(N / (K·P))** |
| Memory / proc | O(N) | O(N/sqrt(P)) | O(N/P) | O(N) |
| Neighbors per rank | P (all-to-all) | ~sqrt(P) | 6 (3D) | **2** |

TD has the **highest compute per processor** (because it doesn't divide work spatially) but the **lowest communication neighbors** (2) and the **lowest communication bandwidth** (due to K). This is a different operating point from the other methods: TD is ideal when compute is fast, memory is plentiful, and bandwidth is the bottleneck. Modern GPU clusters increasingly match this profile.

## 10. What this means for the implementation

From the theory above, the implementation must:

1. Have a **SystemState** that lives entirely in each rank (no spatial split in the basic TD case).
2. Define **Zone** as a first-class data structure with state, time_step, and bbox.
3. Define a **Scheduler** that walks zones in a fixed order, maintains the state machine, enforces dependencies, and launches work.
4. Build **neighbor lists** with skin, and rebuild them when `v_max` threatens skin violation.
5. Use **asynchronous ring communication** with even/odd phase alternation.
6. Support a **K parameter** for multi-step batching.
7. Track `time_step` per zone (not global).

The next document (`docs/02-architecture/overview.md`) translates these requirements into modules and interfaces.

---

## References

- Source dissertation (see `dissertation-source.md`): proposes the method, derives scaling, demonstrates 95%+ efficiency on FCC-lattice test cases with Morse and EAM potentials.
- Plimpton, S. J. (1995). "Fast Parallel Algorithms for Short-Range Molecular Dynamics." *J. Comp. Phys.* 117, 1-19. [RD/FD/SD comparison baseline.]
- Verlet, L. (1967). Original neighbor-list algorithm that the skin idea generalizes.
