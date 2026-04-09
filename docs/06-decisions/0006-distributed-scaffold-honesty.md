# 0006 — Distributed scaffold honesty: full replication is not production distributed MD

- **Status:** Accepted (documents existing reality)
- **Date:** 2026-04-09
- **Decider:** human + architect
- **Affected milestone(s):** M5, M6

## Context

M5 (`DistributedPipelineScheduler`) and M6 (`HybridPipelineScheduler`) implement multi-rank Time Decomposition and 2D time x space pipelines respectively. Both work: tests pass, physics matches single-rank references within FP tolerance, NVT thermostat functions across ranks.

However, both use **full replication**: every rank holds all N atoms on GPU. Zone ownership determines which zones a rank advances, but all atom data is present everywhere. Boundary zone data exchange uses synchronous `MPI_Sendrecv` with explicit D2H/H2D staging through host buffers.

## Why this is OK for now

1. **Correctness scaffold.** Full replication eliminates an entire class of bugs (missing ghost atoms, incorrect halo widths, stale data at boundaries). It let us validate TD pipeline logic, zone state machine, causal dependencies, and NVT across ranks without also debugging ghost atom management.

2. **Multi-rank tests exist and pass.** M5 has 3 MPI tests (deterministic match, NVE conservation, zone time step advance). M6 has 3 hybrid MPI tests (4-rank). These validate parallel execution correctness.

3. **Single-GPU is the current optimization target.** Phase 2 (FastPipelineScheduler) focused on single-GPU performance and achieved 2.38x LAMMPS-GPU on medium systems. Multi-rank performance is not the current bottleneck.

## Why this is not production distributed MD

1. **O(N) memory per rank.** Each rank allocates device memory for all N atoms, not just owned + ghost atoms. For large systems, this limits problem size to what fits on a single GPU — defeating the purpose of multi-rank.

2. **Blocking communication.** `MPI_Sendrecv` blocks until exchange completes. No overlap between communication and computation. TD's architectural advantage (only 2 neighbors, low bandwidth) is not yet exploited.

3. **D2H/H2D staging.** Zone boundary data goes: GPU -> host buffer -> MPI -> host buffer -> GPU. With GPU-aware MPI or NCCL, this could be GPU -> GPU directly.

4. **No ghost-atom optimization.** Even with full replication, the code exchanges entire zone buffers rather than only the atoms needed by neighboring ranks.

## Path forward

The path from scaffold to production distributed MD:

1. **Ghost-atom representation.** Each rank stores: owned atoms (full data) + ghost atoms (positions + types only, updated via halo exchange). This makes memory O(N/P + N_ghost).

2. **Async halo exchange.** Replace `MPI_Sendrecv` with `MPI_Isend`/`MPI_Irecv`. Overlap zone boundary exchange with interior zone computation. TD's 2-neighbor topology makes this straightforward.

3. **GPU-direct communication.** Use GPU-aware MPI or NCCL for GPU-to-GPU transfers, eliminating D2H/H2D staging overhead.

4. **Overlap with compute.** While rank K exchanges boundary data for zone I, it computes zone J (which doesn't depend on the exchanged data). This is the core of TD pipeline parallelism.

This is significant engineering work (~2-4 weeks), scope to be defined as a separate milestone (M5.5 or post-M8).

## Consequences

- Anyone benchmarking TDMD multi-rank performance must understand that current numbers reflect full-replication overhead, not production distributed MD.
- Scaling claims from M5/M6 tests are **correctness** claims, not **performance** claims.
- The path forward is well-defined and does not require architectural changes — only implementation of the communication layer that the architecture already allows.
