# Current milestone status

> **Last updated:** 2026-04-10 (post session VL-EXT)

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
- **VL-3 ✅ — `nve-drift` long run.** 4000-atom Cu FCC, EAM/alloy,
  20 ps NVE (20 000 steps). First case that integrates equations of
  motion rather than just grading step-0 forces. `generate_input.py`
  produces a committed data file with deterministic Maxwell-Boltzmann
  velocities (seed 42) at T=100 K; numpy-free so it runs from a stock
  Python 3. `check.py` parses every thermo sample, drops the first
  25 % as equilibration transient (KE/PE redistribute on a cold FCC
  lattice; that's not drift), then measures drift as
  `|slope(linreg(TE vs t))| / |mean(TE)|`. Marked `slow = true` (runs
  ~3 min per mode on the CPU-only driver) so it won't gate PRs — that
  wiring happens in VL-5. Observed on 2026-04-10: fp64 drift ~5.7e-10
  /ps (1000× under 5e-7 threshold), mixed drift ~3.0e-8 /ps (1700×
  under 5e-5 threshold). Float32 force noise mostly cancels on long
  time-averaging, so drift stays tiny even in mixed mode.
- **VL-4 ✅ — `cross-precision-ab` case.** Runs 100 fs of NVE on the
  shared 4000-atom Cu FCC input in both `build-mixed/` and
  `build-fp64/`, compares final positions, forces, and total energy
  atom-by-atom. Added `--dump-final <file>` flag to `tdmd_standalone`
  so end-of-run state can be dumped in the same LAMMPS-custom format
  already used for step-0 dumps. This is the only VerifyLab case that
  does not take `--mode` — it always runs both. 100 steps is
  deliberate: MD is chaotic, so identical inputs in different
  precision paths diverge exponentially with Lyapunov time ~1 ps;
  any check run for much longer than 0.1 ps would be measuring
  physics, not a bug. Observed on 2026-04-10: max `|dx|` ~2.1e-5 Å,
  max `|dF|` ~6.8e-4 eV/Å, `|dTE|/|TE|` ~1.1e-5. All within ~24×
  margin of the committed thresholds. Fast suite now at 3/3 PASS
  (two-atoms-morse, run0-force-match, cross-precision-ab).
- **VL-5 ✅ — CI wiring.** `.github/workflows/verifylab.yml`
  rewritten. Two jobs:
  - `fast-suite` runs on `pull_request` and `push` to `main`. Builds
    both `build-mixed/` and `build-fp64/` in a CPU-only configuration
    (no CUDA / no MPI — GitHub free-tier runners have no GPU), then
    runs the fast suite in each mode. The three fast cases
    (two-atoms-morse, run0-force-match, cross-precision-ab) now gate
    every PR. Runs on `ubuntu-24.04` because VerifyLab checks use
    `tomllib`, which is stdlib only from Python 3.11+ (Ubuntu 22.04
    ships 3.10). Estimated wall-clock ~2 min.
  - `slow-suite` runs only on the existing nightly cron (03:00 UTC).
    Same build, runs `--suite slow` in both modes (just nve-drift for
    now). Gated by `if: github.event_name == 'schedule'` so it does
    not fire on PRs. Estimated wall-clock ~6 min.
  - The previous `|| echo "VerifyLab not yet available at M0"`
    swallow-failure trick is gone — real failures now break the
    build as intended. `docs/04-development/ci-strategy.md` updated
    to list VerifyLab as a real CI coverage line.
  - Validated locally by spinning up fresh CPU-only `build-mixed/`
    and `build-fp64/` directories in `/tmp/`, running both
    `run-verifylab.sh --mode mixed --suite fast` and
    `--mode fp64 --suite fast`. All 3 fast cases PASS in each mode.

### Session VL-EXT — research-doc-driven expansion (2026-04-10)

Autonomous session driven by the "Comprehensive Test Suite Design for a
Time-Decomposition MD Engine" research document. Goal: fold what's
achievable given current infrastructure (CPU-only GitHub CI, LAMMPS as
local submodule, no ML yet) into the suite, and freeze the rest as a
documented backlog.

**Shipped (8 commits):**

- **ADR 0010 — Reduction determinism contract** (`42885c1`). Complete
  RD-1 audit of every reduction primitive in `src/`: 6 enumerated call
  sites (cell-list slot reservation + 4 energy accumulators + 1
  histogram), no hidden `cub`/`thrust`/`__shfl` usage anywhere, forces
  NOT atomic-accumulated (each thread owns `forces[i]` exclusively).
  Conclusion: full bit-reproducibility is bounded to those 6 sites.
  Decision: Option B — two modes via compile-time flag
  `TDMD_DETERMINISTIC_REDUCE`, default OFF, new `build-deterministic/`
  pairs with fp64. Perf budget 1.5–3× in deterministic mode;
  production mode must not regress. Guarantees scoped explicitly:
  same-GPU same-build bit-equal forces/energies/virial; NOT
  cross-GPU, NOT cross-MPI-rank, NOT cross-precision. Implementation
  (RD-3..RD-5) deferred — this session delivers the contract and
  audit, not the kernels.

- **VL-9 — Neighbor list stress tests** (`a0c85da`). 5 new tests in
  `test_neighbor_list.cpp`: random-gas brute-force parity (uneven
  cell occupancy), precise half-skin boundary (0.99× vs 1.01× on two
  skin values to guard against hardcoded 0.5 Å), multi-atom max-not-
  sum (50 atoms under threshold must not trigger; one over must),
  fast-atom rebuild + post-rebuild correctness. Caught a genuine
  off-by-one in my own first draft — the boundary test is effective
  against its intended class of bug.

- **VL-10 — EAM mapping permutation negative test + parser bug fix**
  (`703d1e7`). Synthetic 2-element A/B setfl generator with
  deliberately distinct functional forms; test asserts that
  `[1,2,1,2] vs [2,1,2,1]` on the same 4-atom config produces PE and
  force differences > 1e-3. **Found a real bug in production code
  while writing this:** the multi-element `read_setfl` loop used
  `std::getline` which consumed the empty remainder of the previous
  rho(r) array instead of the next element header. Invisible for
  months because every existing test used single-element
  `Cu_mishin1.eam.alloy`. Fix: skip-whitespace loop in
  `eam_alloy.cpp:137`. All 46/46 tests green after the fix. Exactly
  the class of regression the research doc's P4 ("permuting mapping
  must change energies") was designed to surface.

- **VL-11 — EAM embedding density consistency** (`3e62a99`). On
  256-atom Cu FCC, computes per-atom `rho_i` two ways (neighbor list
  half-list walk vs. O(N²) brute force) and asserts match within
  1e-10. Catches a class of bugs `run0-force-match` cannot: a missing
  neighbor on the density list silently corrupts `fp_i` AND — through
  the embedding-derivative coupling — `fp_j` for every atom that has
  `i` as a neighbor. Pair-additive Morse is already protected by
  run0-force-match; EAM's many-body coupling deserves its own
  assertion.

- **VL-14 — Unified VerifyLab result schema + migration** (`85b08f0`).
  New `verifylab/runners/result_schema.py` defines schema v1 (case,
  mode, status={pass,fail,error}, metrics, thresholds, failures,
  duration, timestamp) and the stable `VL_RESULT:` sentinel line. All
  4 existing cases migrated in one commit to emit the record at the
  end of `main()`. `run_all.py` gains `--jsonl <path>` to aggregate
  scraped records. Status semantics are explicit: "fail" = real
  regression, "error" = setup/crash (cannot grade); CI should weigh
  these differently. NaN/inf metrics serialize as `null` so downstream
  parsers don't choke. Fast suite 3/3 PASS, aggregated JSONL validated.

- **VL-15 — setfl parser robustness** (`70a3cb6`). 5 negative tests
  asserting that corrupt setfl files throw rather than silently
  producing garbage splines: empty file, comments-only, missing
  embedding array, truncated embedding array, missing pair block. Two
  cases deliberately NOT covered and tracked as follow-ups: negative
  `Nrho`/`Nr` (requires a guard in the reader before it can reject),
  and short element-name list on the ntypes line (reader currently
  tolerates silently).

- **ML test suite frozen as backlog** (`42885c1`, embedded in the
  ADR 0010 commit). P5/ML1/ML2/ML3/ML4 contract added to
  `current_milestone.md`: no ML milestone can close without all 5
  cases green in both build-mixed/ and build-fp64/. This is a
  commitment, not an aspiration — it gives future-us permission to
  refuse closing an ML milestone that lacks descriptor reproducibility
  or finite-difference force validation.

**Test count after session:** `build-mixed/` and `build-fp64/` both at
52/52 unit tests PASS (was 38 at start of VL session). VerifyLab fast
suite at 3/3 PASS in both modes; nve-drift slow case unchanged.

**Deferred from this session (explicitly):**

| Item | Status | Reason |
|---|---|---|
| VL-6 LAMMPS `rerun` oracle | deferred | needs LAMMPS in CI image; out of scope for a session that already shipped 8 commits |
| VL-7 NVT stability vs LAMMPS | deferred | same — needs LAMMPS in nightly |
| VL-8 MSD/diffusion vs LAMMPS | deferred | same |
| VL-12 deterministic velocity init (`velocity ... loop geom` analogue) | deferred | **blocked**: TDMD has no internal velocity generator — velocities come from LAMMPS data files. Revisit when a `velocity create` CLI flag is added |
| VL-13 sum-order sensitivity | deferred | blocked on RD-3/4 (needs `build-deterministic/`) |
| VL-16 fault injection via `#ifdef` | deferred | blocked on VL-14 (✅ now) and RD-4 (needs `build-deterministic/`) |
| RD-3 implement deterministic cell placement + ordered energy reduce | deferred | major core engine work; ADR 0010 delivers the contract, implementation is a separate arc |
| RD-4 `build-deterministic/` CMake preset + CI slow-suite job | deferred | blocked on RD-3 |
| RD-5 wire VL-13 / VL-16 to deterministic mode | deferred | blocked on RD-4 |

Follow-ups VL-14 unblocks: fault injection (VL-16) can now latch onto
the schema once `TDMD_DETERMINISTIC_REDUCE` exists. Sum-order
sensitivity (VL-13) can write strict-equality metrics against the
`status=pass` threshold `0.0`.

**Process note.** The EAM multi-element parser bug (VL-10 commit) is
the second Phase-3-era latent bug found by this session; both were
invisible in production because no test exercised the relevant code
path. `docs/04-development/lessons-learned.md` already carries the
meta-lesson "documents lie, code does not" — VL-10 and VL-11 reinforce
it from the opposite direction: **tests that don't exist lie the
loudest.** Every case added here was motivated by the observation
that a specific failure mode had no regression gate.

### Session VL closed

The 5-session VerifyLab expansion is complete. TDMD now has four
live physics-validation cases that run against the real
`tdmd_standalone` CPU driver, and three of them gate every PR:

| case                | mode(s)       | suite | what it checks                        |
|---------------------|---------------|-------|---------------------------------------|
| two-atoms-morse     | mixed + fp64  | fast  | analytic Morse force + energy         |
| run0-force-match    | mixed + fp64  | fast  | 256-atom Cu FCC LAMMPS A/B (Morse+EAM)|
| cross-precision-ab  | both required | fast  | mixed vs fp64 short-NVE divergence    |
| nve-drift           | mixed + fp64  | slow  | 20 ps NVE total-energy drift          |

CPU physics regressions will now be caught automatically. GPU
physics still needs manual validation until a self-hosted GPU runner
exists (see `docs/04-development/ci-strategy.md`).

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
- **ML potential test suite — deferred, blocked on ML support landing.**
  Reference: "Comprehensive Test Suite Design" research doc, ML-specific
  section. When ML lands, DoD of the ML milestone includes the full set:
  - **P5:** descriptor → model → force pipeline parity vs LAMMPS ML
    backend (`pair_style snap` / `mliap` / `kim`), `run 0` oracle.
  - **ML1:** descriptor reproducibility CPU vs GPU — L2/RMS ≤ 1e-10
    (double), ≤ 1e-6 (float), stable across MPI layouts.
  - **ML2:** analytic forces vs finite-difference of energy — δ-sweep,
    relative error band within tolerance as δ shrinks to noise floor.
  - **ML3:** batch vs per-atom evaluation equivalence (GPU throughput
    path must match per-atom path bitwise within reduction tolerance).
  - **ML4:** GPU vs CPU kernel parity on medium (4k–16k atom) config.
  This entry is a contract: no ML milestone can close without these 5
  cases green in both `build-mixed/` and `build-fp64/`.

## Next milestone — TBD

Phase 3 closure is a natural pause point. Next milestone selection awaits
project lead input. Candidates:
- M7 kernel fusion K > 1 (TDMD-unique optimization)
- Phase 4: neighbor list rebuild optimization
- ~~Session 3B.8: EAM migration~~ (done in EAM-1B)
- Multi-rank distributed work (ghost-only exchange)
- VerifyLab expansion
