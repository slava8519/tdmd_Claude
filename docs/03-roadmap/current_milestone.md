# Current milestone status

> **Last updated:** 2026-04-10

## Phase 3 series — COMPLETE (closed in session 3B.closing, hardened in session 3D)

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

### Session 3D — Phase 3 hardening (2026-04-10)

After EAM-1B closed the precision contract, session 3D shipped two
follow-on bundles that lock the migration into the project's process,
not just the code.

**Bundle 1 — Vec3 role-alias migration finished and documented:**

- **Stages 2–4 atomic refactor (commit `4a718f8`, 57 files):** all
  remaining `Vec3` storage and signatures migrated to `PositionVec` /
  `VelocityVec` / `ForceVec`. Touched: `hybrid_pipeline_scheduler.cu`
  (pack/unpack zones), `mpi_ring_comm` (split payload sizing),
  `cell_list` + `device_cell_list` (`Vec3D` cell metrics),
  `dump_reader`, `tdmd_main` (KE accumulator), and 16 test/benchmark
  files. Build clean in both modes; 73/73 unit tests pass in
  `build-mixed/` and `build-fp64/`. Stages 2–4 had to ship as one
  commit due to a dependency cycle between scheduler and pack format.
- **ADR 0009 (commit `00fd490`):** new ADR records the 5-stage typed
  Vec3 rollout — context, alternatives considered, performance
  outcome (4.0–5.3× mixed-vs-fp64 speedup, TDMD-mixed beats LAMMPS-GPU
  1.26× on 32k atoms), and the meta-lesson that documents drift while
  code does not.

**Bundle 2 — Phase 3 hardening (8 commits, lessons → process):**

The four pitfalls Phase 3 hit (silent storage gap, EAM accumulator
precision loss, default-stream serialization, distance-precision trap)
are now codified across three layers of defence:

1. `5e105a0` — CLAUDE.md §4: four new hard rules
   (relative-coord trick, float intrinsics, `accum_t` reductions,
   grep-verify type invariants).
2. `22b3049` — `docs/04-development/lessons-learned.md` (new, 202
   lines): the four pitfall stories with reproductions, fixes, and
   commit references; meta-lesson "Documents lie, code does not".
3. `5c1d5be` — `prompts/roles/implementer.md`: closing grep step
   added to "After writing code".
4. `15998b4` — `prompts/roles/architect.md`: new "Grounding: read
   the code before you recommend" section + ADR-pushback rule.
5. `b8189b0` — `prompts/roles/reviewer.md`: default-stream and
   `accum_t` items added to "almost always catch"; new "Review by
   grep, not by report" section requiring LAMMPS-citation evidence
   for ADR 0008 claims.
6. `c3a81ac` — `prompts/roles/researcher.md` (new, 120 lines):
   read-only investigation role with halt-on-surprise discipline.
   Reports go to `/tmp/`, not the repo.
7. `072caec` — `prompts/roles/user-advocate.md` (new, 155 lines):
   usability evaluation against LAMMPS baseline. Reports to
   `docs/05-benchmarks/usability/`.
8. `ddcbc24` — CLAUDE.md §12: registers the two new roles in the
   role index.

After bundle 2 the rule is: any session that touches precision,
storage types, or `DeviceBuffer` element types ends with a grep, and
that grep result goes into the user-facing report. Documents are
treated as intent, code as fact.

### Session VL — VerifyLab expansion (in progress, 2026-04-10)

5-session plan to take VerifyLab from one-stub to a physics-validation
suite that gates every PR. VL-1 and VL-2 done; VL-3, VL-4, VL-5 remain.

- **VL-1 ✅ — `two-atoms-morse` wired end-to-end.** Runs
  `tdmd_standalone --nsteps 0 --dump-forces` on a 2-atom input, parses
  the dump + step-0 thermo line, compares against analytic Morse
  reference in `reference/analytic.json`. Added `--dump-forces` flag
  and `write_lammps_force_dump()` to `tdmd_main.cpp`; bumped thermo
  print to 14 fractional digits so fp64 assertions have headroom.
  Found and fixed a sign bug in the committed `analytic.json` (atom 1
  at origin is pulled in +x toward atom 2, not -x) — VerifyLab's first
  real catch was in its own reference. Residuals on 2026-04-10: fp64
  hits machine epsilon (~0, 14+ digits); mixed sits at ~1.5e-8 PE,
  ~3e-8 force (bounded by float32 force path per ADR 0007).
  `scripts/run-verifylab.sh` rewritten to accept `--mode {mixed,fp64}`.
- **VL-2 ✅ — `run0-force-match` wired end-to-end.** 256-atom Cu FCC,
  Morse *and* EAM/alloy, compared atom-by-atom against committed
  LAMMPS `run 0` reference dumps. Added `--eam <setfl>` flag to
  `tdmd_standalone` (mutually exclusive with `--morse`, dispatched via
  lambdas `pot_cutoff()` / `compute_forces_dispatch()`). `check.py`
  runs TDMD once per potential, parses both reference dumps, compares
  by **absolute** max-component diff (relative is meaningless when the
  reference is ~1e-15 machine zero on a perfect lattice). Residuals on
  2026-04-10: fp64 Morse ~2.3e-15, fp64 EAM ~9.1e-16 (both hit machine
  epsilon — TDMD and LAMMPS agree to zero when both run in double);
  mixed Morse ~5.0e-6, mixed EAM ~2.7e-6. All 2 VerifyLab cases PASS
  in both modes; unit tests 73/73 green in both builds.
- **VL-3 ⏳ — `nve-drift` long run.** ~4k-atom Cu FCC, 50k NVE steps,
  assert `|dE/E|` bound. Slow suite (nightly).
- **VL-4 ⏳ — cross-precision A/B.** Same input run in both
  `build-mixed/` and `build-fp64/`, trajectories compared against each
  other (not against LAMMPS). Answers "how fast do mixed and fp64
  diverge under NVE?".
- **VL-5 ⏳ — CI wiring.** `./scripts/run-verifylab.sh --suite fast`
  as a required check on PRs. Fast suite runs every PR; slow suite
  nightly.

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
  full migration cleanup. Cosmetic, low priority. (Note: as of 3D the
  legacy `Vec3 = Vec3T<real>` alias still lives in `types.hpp` and
  `math.hpp` for tests that haven't been converted; safe to defer.)
- **Phase 4 backlog:** neighbor list rebuild optimization (~10-15% additional
  speedup), EAM migration to FastPipelineScheduler, kernel fusion K > 1 for
  M7 (TDMD-unique optimization).
- **Distributed scaffold honesty:** M5/M6 still use full replication, not
  ghost-only exchange. See ADR 0006.
- **VerifyLab stub:** `verifylab/cases/single-gpu-collapse/` README is
  drafted (3 test cases, all unimplemented). Blocked on ADR 0005 Phase 2
  zone-collapse implementation in the scheduler.

## Next milestone — TBD

Phase 3 closure is a natural pause point. Next milestone selection awaits
project lead input. Candidates:
- M7 kernel fusion K > 1 (TDMD-unique optimization)
- Phase 4: neighbor list rebuild optimization
- ~~Session 3B.8: EAM migration~~ (done in EAM-1B)
- Multi-rank distributed work (ghost-only exchange)
- VerifyLab expansion
