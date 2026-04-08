# Current Milestone: M5 — MPI Ring Parallelization ✅ COMPLETE

> **Goal:** Lift the in-process pipeline from M4 to actual MPI ranks. Multi-rank, single GPU per rank.

## Checklist

- [x] `comm/MpiRingComm` — async MPI ring communicator with send/recv
- [x] `comm/CMakeLists.txt` — MPI-conditional build
- [x] Extend `ZonePartition` for multi-rank ownership (assign_to_ranks, owner_rank, ghost_zones)
- [x] `DeviceBuffer` offset-based copy_from_host/copy_to_host
- [x] `scheduler/DistributedPipelineScheduler` — multi-rank pipeline with synchronous MPI_Sendrecv
- [x] Boundary zone detection (send_to_next/send_to_prev zones)
- [x] Ghost zone time_step tracking for cross-rank dependency checks
- [x] Synchronized loop via min_global_time_step (MPI_Allreduce)
- [x] Synchronous boundary exchange (pack/MPI_Sendrecv/unpack) — deadlock-free
- [x] Neighbor list rebuild synchronized across ranks
- [x] Test: 2-rank deterministic matches single-rank M4 (< 1e-6)
- [x] Test: 2-rank pipeline NVE conservation (1000 steps, |dE/E| < 1e-4)
- [x] Test: all zones advance to target_step across ranks

## Exit criteria — status

- [x] 2-rank run produces results matching M4 single-rank within FP tolerance (< 1e-6).
- [x] 2-rank NVE energy conservation: 1000 steps, |dE/E| < 1e-4.
- [x] All 60 tests pass (57 M0-M4 + 3 M5 distributed MPI).
- [ ] Strong scaling efficiency >= 0.85 at P=4 (needs multi-GPU cluster — deferred).
- [ ] No deadlocks under stress test (needs multi-GPU — deferred).

## Architecture notes

- Each rank holds ALL atoms on GPU (full replication). Memory-efficient ghost-only mode is future optimization.
- Synchronous MPI_Sendrecv for boundary exchange: pack zones → exchange sizes → exchange data → unpack.
- Loop condition uses `min_global_time_step()` (MPI_Allreduce) so all ranks enter/exit together.
- Zone assignment: contiguous blocks, `base + remainder` distribution.
- 3 zones auto (from floor(Lx/r_cut)=2, min 3 for PBC), 2 ranks → rank 0 gets 2 zones, rank 1 gets 1 zone.
- Ghost zones identified per rank: non-local zones in any local zone's neighbor list.

## Next: M6 — 2D time × space parallelism
