# Current state: post-M7, ADR 0005 Phase 2 complete

> **Last updated:** 2026-04-09

## Closed

- **M0–M7:** all milestones formally complete. 71 tests passing (FP64+CUDA+MPI build).
- **ADR 0005 Phase 2:** `FastPipelineScheduler` implemented and measured. Single-GPU batched kernels, 5 launches/step. FP32 medium (32k atoms): 7,638 ts/s, **2.38x faster than LAMMPS-GPU**. See [`docs/05-benchmarks/phase2-batched-scheduler-results.md`](../05-benchmarks/phase2-batched-scheduler-results.md).

## Active backlog (prioritized)

### High priority — next sessions

1. **NVT multi-rank atom range bug.** `DistributedPipelineScheduler` computes `first_local_atom_`/`local_atom_count_` in constructor before `assign_atoms()`, but `upload()` calls `assign_atoms()` which re-sorts atoms. The KE/scaling in NVT path uses stale offsets. Fix: recompute local atom range after `assign_atoms()` in `upload()`. Severity: medium (affects MPI NVT only). Planned: session 2.

2. **FP32 test tolerance failures.** 12 of 65 tests fail in FP32 build (`TDMD_FP64=OFF`). Root cause: tests use `EXPECT_DOUBLE_EQ` and `1e-10`/`1e-12` tolerances hardcoded for FP64. Fix: make tolerances dependent on `sizeof(real)`. Planned: session 3.

3. **Phase 3a: neighbor list optimization.** `DeviceNeighborList::build` takes 175 us/call (42% GPU time on small, vs LAMMPS ~60 us). Contains host prefix sum with `cudaStreamSynchronize`. Target: 60 us/call, fully GPU-resident. Expected: +10-15% overall. Effort: ~1 day.

4. **Phase 4: EAM migration to FastPipelineScheduler.** Replace `compute_morse_gpu` with 3-pass EAM. Step goes from 5 to 7-8 launches. Architecture is potential-neutral. Effort: ~1 day.

### Medium priority — future optimization

5. **Kernel fusion K>1.** TD-unique feature. Multiple VV steps in a single kernel launch. Expected 1.3-3x on small systems. Separate ADR. Effort: ~2-3 days.

6. **Force kernel vectorization.** Horizontal vector ops (4 neighbors per SIMD lane). Expected 2-3x in compute-bound regime. Effort: TBD.

7. **Mixed precision mode.** FP32 compute + FP64 accumulators for long runs (>30M steps). Effort: TBD.

### Low priority — infrastructure debt

8. **CI GPU coverage.** Current CI is CPU-only + compile-only CUDA/MPI. Permanent solution: self-hosted GPU runner. See [`docs/04-development/ci-strategy.md`](../04-development/ci-strategy.md).

9. **Multi-rank scaffold to production distributed MD.** Current M5/M6 use full replication. Path: ghost-atom exchange, async overlap, true distributed scaling. See [ADR 0006](../06-decisions/0006-distributed-scaffold-honesty.md).

## Known issues

| Issue | Severity | Location | Planned fix |
|-------|----------|----------|-------------|
| NVT multi-rank stale atom offsets | Medium | `distributed_pipeline_scheduler.cu:83,421` | Session 2 |
| FP32 test failures (12/65) | High | various tests, hardcoded FP64 tolerances | Session 3 |
| CI does not run GPU tests | Medium | `.github/workflows/build.yml` | Self-hosted runner (backlog) |
| Docs had stale M0 claims | Fixed | README, build-and-run.md | This session |
