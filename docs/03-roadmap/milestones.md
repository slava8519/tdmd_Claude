# 03 — Roadmap (Milestones)

> **The plan from zero to a usable TDMD.**
> Milestones are sequential. Each one has hard exit criteria.
> Do not start M(N+1) before M(N) is signed off.

---

## Status legend

- ⬜ Not started
- 🟨 In progress
- ✅ Done
- ❌ Blocked

---

## Overview

| ID | Name | Approx. effort | Status |
|---|---|---|---|
| M0 | Foundation & infra | 2 weeks | ✅ |
| M1 | Reference MD (CPU, single-thread) | 4 weeks | ✅ |
| M2 | GPU port (single GPU) | 4 weeks | ✅ |
| M3 | Zone decomposition (in-process) | 3 weeks | ✅ |
| M4 | TD scheduler (full pipeline, in-process) | 5 weeks | ⬜ |
| M5 | MPI ring parallelization | 6 weeks | ⬜ |
| M6 | 2D time × space parallelism | 4 weeks | ⬜ |
| M7 | NVT/NPT + adaptive Δt + optimizations | 4 weeks | ⬜ |
| M8 | ML potentials (continuous) | ongoing | ⬜ |

Total to M7: **~32 weeks**.

Post-M7 research items: dynamic load balancing, space-filling-curve traversal orders, biological MD support — all in `docs/03-roadmap/post-m7-research.md` (TBD).

---

## M0 — Foundation & infrastructure

**Goal:** the project skeleton compiles, tests run, CI is green, docs are reachable.

**Deliverables:**
- Repository structure as scaffolded (this folder).
- CMake builds an empty executable that prints `tdmd v0.0.0` and exits.
- `tests/unit/test_smoke.cpp` runs and passes.
- CI pipeline (GitHub Actions) builds + runs tests on every push.
- `scripts/build.sh`, `scripts/run-tests.sh`, `scripts/format.sh` working.
- `docs/` is complete and consistent (this scaffold).

**Exit criteria:**
- [x] `./scripts/build.sh` exits 0 on a clean Ubuntu 22.04 box with CUDA 12.
- [x] `./scripts/run-tests.sh` exits 0.
- [ ] CI green on `main`.
- [ ] All docs in this scaffold pass markdown lint.
- [x] `CHANGELOG.md` has an entry "M0 complete".

**Out of scope:** any actual physics. M0 is plumbing only.

---

## M1 — Reference MD (CPU, single-threaded)

**Goal:** a correct, slow, single-threaded MD code that we can trust as a reference. **No GPU yet. No parallelism. No zones.** Just an honest implementation that we can compare against LAMMPS atom-by-atom.

**Deliverables:**

1. `core/SystemState` on the CPU (host vectors, not device).
2. `io/LammpsDataReader` reads a LAMMPS data file (ortho box, atomic style).
3. `domain/CellList` builds a basic cell list.
4. `neighbors/NeighborList` builds Verlet lists with skin.
5. `potentials/MorsePair` — Morse pair potential, energy and forces.
6. `potentials/EamAlloy` — EAM/alloy with setfl file reading.
7. `integrator/VelocityVerlet`.
8. `drivers/tdmd_standalone` runs a fixed input file end-to-end.
9. `verifylab/scenarios/two_atoms_morse/` — analytic 2-atom test.
10. `verifylab/scenarios/run0_force_match_morse/` — small Morse system, A/B vs LAMMPS.
11. `verifylab/scenarios/nve_drift_morse/` — 50 000 steps, energy drift check.
12. `verifylab/scenarios/run0_force_match_eam/` — small EAM system, A/B vs LAMMPS.

**Exit criteria:**
- [ ] Two-atom Morse: forces and energy match analytic solution to 1e-12.
- [ ] Run-0 Morse on a 256-atom box: max force-component error vs LAMMPS < 1e-6 (FP64).
- [ ] Run-0 EAM on a 256-atom box: same tolerance.
- [ ] NVE drift over 50 000 steps at dt=1 fs: |ΔE/E| < 1e-4.
- [ ] All VerifyLab scenarios pass on `main`.

**Out of scope:** GPU, MPI, zones, ML potentials, NVT/NPT, dump format, restart files.

**Why this matters:** if we can't get this right on CPU, we cannot debug TD on GPU later. M1 is our oracle for everything that comes after.

---

## M2 — GPU port (single GPU)

**Goal:** the M1 reference, ported to CUDA, on a single GPU. Same numerical results within FP tolerance.

**Deliverables:**

1. CUDA kernels for Morse pair forces.
2. CUDA kernels for EAM (3-stage: gather ρ → F(ρ) → scatter forces).
3. GPU neighbor list builder (cell-based).
4. Device-resident SystemState.
5. Mixed-precision mode (FP32 forces, FP64 accumulators) and FP64-only mode.
6. NVTX markers around all major phases.

**Exit criteria:**
- [ ] Same VerifyLab scenarios as M1 pass on GPU.
- [ ] Run-0 force match vs M1 CPU: < 1e-5 relative in mixed precision.
- [ ] FCC16 small (256 atoms) reaches > 1000 timesteps/s on RTX 5080.
- [ ] `nsys profile` shows the expected phase breakdown.

**Out of scope:** zones, multi-GPU, MPI, scheduler.

---

## M3 — Zone decomposition (in-process)

**Goal:** introduce the `Zone` data structure and the cell→zone mapping, but keep everything on a single rank. The scheduler just walks zones in order; no time decomposition yet.

**Deliverables:**

1. `domain/Zone` struct.
2. 1D, 2D, 3D zone partitioning (configurable).
3. Cell-to-zone mapping.
4. Scheduler walks zones in fixed order, computes forces zone-by-zone.
5. Buffer/skin handling: per-zone v_max tracking, early rebuild trigger.
6. Tests for partitioning correctness (every cell maps to exactly one zone, every atom is in exactly one zone, etc.).

**Exit criteria:**
- [ ] Zone-walked simulation produces bit-identical results to M2 (in deterministic mode).
- [ ] Skin invariant test: synthesize a fast atom, assert early rebuild fires.
- [ ] FCC16 medium runs through the zone walker without performance regression > 10% vs M2.

**Out of scope:** different time_steps per zone (that's M4), MPI.

---

## M4 — TD scheduler (full pipeline, in-process)

**Goal:** implement the actual time-decomposition pipeline, but still on a single rank. Simulate the ring of "processors" in shared memory, with different `time_step` values living in different zones simultaneously. This is the hardest milestone.

**Deliverables:**

1. Full Zone state machine implementation (per `docs/01-theory/zone-state-machine.md`).
2. Dependency DAG between zones.
3. Priority queue for `Ready` zones (by sphere-of-cutoff rule).
4. CUDA stream pool with N streams (default 4).
5. Per-zone CUDA events for completion tracking.
6. K parameter (multi-step batching) with the local-only ring simulator.
7. Telemetry counters: per-state time, transitions per second, pipeline occupancy %.

**Exit criteria:**
- [ ] Full state machine test suite green (see `docs/01-theory/zone-state-machine.md` §"Tests required").
- [ ] In deterministic mode: bit-identical results to M3.
- [ ] In fast mode: < 1e-10 relative drift over 25 900 steps vs M3.
- [ ] Pipeline occupancy ≥ 80% in steady state on FCC16 medium.
- [ ] Schedule overhead < 5% of total step time.

**Out of scope:** real network communication, multi-rank.

---

## M5 — MPI ring parallelization

**Goal:** lift the in-process ring simulator from M4 to actual MPI ranks. Multi-rank, single GPU per rank.

**Deliverables:**

1. `comm/MpiRingComm` — async send/recv with even/odd phase alternation.
2. GPU-aware MPI mode + pinned-host fallback mode (auto-detect at startup).
3. Multi-rank zone ownership (each rank holds a subset of zones).
4. Wraparound and boundary conditions on the ring.
5. Linear scaling test up to P_opt.

**Exit criteria:**
- [ ] FCC16 medium runs correctly on 2, 4, 8 ranks (each with its own GPU).
- [ ] Forces and energies match the M4 single-rank reference within FP tolerance.
- [ ] Strong scaling efficiency ≥ 0.85 at P = 4 on FCC16 medium.
- [ ] No deadlocks under stress test (1M steps, multiple ranks).

**Out of scope:** spatial decomposition within a rank.

---

## M6 — 2D time × space parallelism

**Goal:** add spatial decomposition as a second axis, so we can scale beyond P_opt of pure TD.

**Deliverables:**

1. 2D MPI Cartesian communicator: `[P_time, P_space]`.
2. Within-rank spatial subdomain + halo exchange (classic SD).
3. The TD pipeline runs over groups of spatial-subdomain ranks.
4. Tests on a single big system (FCC16 large) with both axes nontrivial.

**Exit criteria:**
- [ ] Correct results vs M5 reference at the same total rank count.
- [ ] Scaling beyond P_opt works: 16 ranks gives meaningful speedup over 8 ranks even when pure TD would saturate.
- [ ] Documented in `docs/02-architecture/parallel-model.md`.

**Out of scope:** dynamic rebalancing.

---

## M7 — NVT/NPT, adaptive Δt, optimizations

**Goal:** the simulation supports the standard ensembles needed by users, plus the adaptive Δt described in the dissertation. Plus the first round of perf optimization.

**Deliverables:**

1. `integrator/NoseHoover` for NVT and NPT.
2. Adaptive Δt mode (per dissertation, with the `C2`, `C3` parameters).
3. Roofline analysis of the force kernels.
4. Top 3 kernel optimizations (likely: register usage, shared memory tiling, vectorized loads).
5. Performance regression test in CI (alerts if a benchmark slows by > 5%).

**Exit criteria:**
- [ ] NVT and NPT VerifyLab scenarios pass vs LAMMPS within statistics.
- [ ] Adaptive Δt mode produces stable trajectories on a thermal benchmark.
- [ ] FCC16 medium achieves at least 70% of LAMMPS-GPU timesteps/s on the same RTX 5080.

---

## M8 — ML potentials (continuous, post-M7)

**Goal:** enable SNAP/ACE/ML-IAP through plugin interface and/or LAMMPS bridge.

This milestone is open-ended and will be tracked as feature work, not as a hard deadline.

---

## Cross-milestone rules

1. **No milestone is "done" until VerifyLab is green**, unit tests pass, docs are updated, and the human has signed off.
2. **No skipping milestones.** If M3 isn't done, M4 doesn't start.
3. **Spike branches** (`exp-*`) are allowed for trying things, but they don't count as progress until merged via a real milestone PR.
4. **Performance work is M7+.** Before M7, we optimize for **correctness and clarity** only. Premature optimization is forbidden.

---

## Post-M7 research backlog

- **Dynamic zone re-balancing** (load imbalance from shock waves, plasticity).
- **Hilbert / Morton space-filling curves** for 3D zone traversal order.
- **Persistent kernels** with NCCL ring instead of MPI for very-low-latency clusters.
- **Long-range Coulomb** (PPPM) integration — out of metals scope but a future direction.
- **Biomolecular force fields** — far future.
