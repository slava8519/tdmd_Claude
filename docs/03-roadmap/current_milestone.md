# Current milestone status

> **Last updated:** 2026-04-09

## Phase 3 series — COMPLETE (closed in session 3B.closing)

Phase 3 series consisted of 7 sub-sessions covering hygiene, critical bug
fixes, mixed precision implementation, and performance recovery. Final state:

### What was delivered

- **Mixed precision mode (LAMMPS-style)** as production default. Positions
  and velocities in `double` for trajectory quality and GPU integrator
  preservation. Forces in `float` for compute speed. All accumulators in
  `double`. See ADR 0007 (precision contract) for full design.
- **Performance recovery to 91% of Phase 2 FP32 baseline** on medium system.
  Tiny: 90%, small: 93%, medium: 91%. Achieved via LAMMPS-derived
  relative-coordinate trick (double-subtract + float cast) for distance
  computation in force kernel and neighbor list builder.
- **Energy drift in mixed mode: 2.57e-13/step**, well below ADR 0007 target
  of 1e-12/step. FP64 mode drift: 1.05e-20/step (machine epsilon limit).
- **Critical bug fixes:**
  - NVT multi-rank atom range bug (silent thermostat no-op since M5/M6)
  - MPI_DOUBLE hardcoded in precision-agnostic code (UB risk in FP32)
  - Both regression-tested.
- **Process rule established (ADR 0008): Copy LAMMPS where applicable.**
  Formalizes the practice of reading LAMMPS source as reference for general
  GPU MD problems before designing TDMD-specific solutions.
- **CI safety net:** compile-only jobs for CUDA and MPI. Both build modes
  (mixed and fp64) verified.

### Test status (final)

- `build-mixed/`: 26 CUDA passed + 1 skipped (DeterministicMatchesM3), 38 unit passed
- `build-fp64/`: 27 CUDA passed, 38 unit passed
- All MPI tests passing in both modes
- No known failing tests in any production build configuration

### Performance summary

Mixed mode (production default), `fast_pipeline` scheduler, RTX 5080:

| System | timesteps/s | vs Phase 2 FP32 | vs LAMMPS-GPU |
|---|---|---|---|
| tiny (256 atoms) | 14,814 | 90% | ~1.2x faster |
| small (4,000 atoms) | 9,139 | 93% | ~1.4x faster |
| medium (32,000 atoms) | 6,927 | 91% | ~2.2x faster |

FP64 mode (validation/reference), medium: 754 ts/s.

### Backlog after Phase 3

- **Phase Б (deferred):** PBC outside force kernel + image counter. ~7%
  additional performance + unwrapped trajectory output. Trigger conditions
  documented in milestones.md.
- ~~**Session 3B.8:** EAM full migration to role aliases~~ — **COMPLETE
  (session EAM-1B).** Density accumulator fixed to `accum_t` (LAMMPS
  parity). Relative-coordinate trick applied to both density and force
  kernels. Force accumulators migrated to `force_t`. ADR 0007 corrected
  (spline coefficients are `real`/float, not double as originally stated).
  All tests pass in both modes.
- **Session 3C:** Remove `using real = ...` typedef from core/types.hpp,
  full migration cleanup. Cosmetic, low priority.
- **Phase 4 backlog:** neighbor list rebuild optimization (~10-15% additional
  speedup), EAM migration to FastPipelineScheduler, kernel fusion K > 1 for
  M7 (TDMD-unique optimization).
- **Distributed scaffold honesty:** M5/M6 still use full replication, not
  ghost-only exchange. See ADR 0006.

## Next milestone — TBD

Phase 3 closure is a natural pause point. Next milestone selection awaits
project lead input. Candidates:
- M7 kernel fusion K > 1 (TDMD-unique optimization)
- Phase 4: neighbor list rebuild optimization
- ~~Session 3B.8: EAM migration~~ (done in EAM-1B)
- Multi-rank distributed work (ghost-only exchange)
- VerifyLab expansion
