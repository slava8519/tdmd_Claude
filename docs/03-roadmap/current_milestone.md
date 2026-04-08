# Current Milestone: M6 — 2D Time × Space Parallelism ✅ COMPLETE

> **Goal:** Add spatial decomposition as a second axis, so we can scale beyond P_opt of pure TD.

## Checklist

- [x] `domain/SpatialDecomp` — 1D Y-axis slab decomposition for spatial subdomain partitioning
- [x] `scheduler/HybridPipelineScheduler` — 2D time × space pipeline scheduler
- [x] MPI_Cart_create with dims [P_time, P_space], periodic in both dimensions
- [x] time_comm (TD ring: same spatial subdomain), space_comm (halo exchange: same time group)
- [x] Ghost atom halo exchange on space_comm (positions, velocities, IDs)
- [x] Ghost atom deduplication (critical for P_space=2 where prev==next)
- [x] TD boundary zone exchange on time_comm (reuses M5 pack/unpack pattern)
- [x] Device buffers pre-allocated for natoms_global to avoid destructive resize
- [x] Fix: PBC wrapping `if/else if` → `if/if` in drift kernel (atoms at box.hi broke spatial decomp)
- [x] Test: 4-rank hybrid (P_time=2, P_space=2) matches M4 single-rank (< 1e-6)
- [x] Test: 4-rank hybrid NVE conservation (500 steps, |dE/E| < 1e-4)
- [x] Test: all local zones advance to target_step
- [x] Build system: hybrid_pipeline_scheduler.cu added to scheduler CMakeLists.txt
- [x] Build system: tdmd_m6_tests executable with 3 CTest entries (4-rank MPI)

## Exit criteria — status

- [x] Correct results vs M5/M4 reference at the same total rank count (< 1e-6).
- [ ] Scaling beyond P_opt: 16 ranks gives meaningful speedup over 8 ranks (needs multi-GPU cluster — deferred).
- [x] All 63 tests pass (60 M0-M5 + 3 M6 hybrid MPI).
- [ ] Documented in `docs/02-architecture/parallel-model.md` (deferred — no existing file yet).

## Architecture notes

- 2D MPI Cartesian topology: dims=[P_time, P_space], both periodic.
- Zone partition along X (for TD pipeline), spatial partition along Y (for scaling).
- Data layout on GPU: [owned atoms sorted by zone | ghost atoms appended after].
- Halo exchange: download owned pos/vel → identify_send_ghosts → MPI_Sendrecv pos+vel+IDs → deduplicate → upload ghost region.
- Ghost deduplication is essential when P_space=2: prev_rank==next_rank, so atoms near both Y-boundaries arrive via both channels.
- Device buffers allocated for natoms_global_ (not n_total_) to prevent destructive resize in upload_ghosts().
- PBC drift fix: `if/else if` wrapping could leave atoms at box.hi after wrap-from-below, breaking spatial decomposition ownership checks. Changed to `if/if` for idempotent wrapping.

## Next: M7 — NVT/NPT + adaptive Δt + optimizations
