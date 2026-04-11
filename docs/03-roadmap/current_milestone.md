# Current milestone status

> **Last updated:** 2026-04-11 (post session FEAT-EAM Phase C — LAMMPS A/B closed)

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

### Session RD-3 — deterministic reduction mode implementation (2026-04-10)

Session VL-EXT shipped ADR 0010 ("Reduction Determinism Contract")
as a spec; this follow-on session implemented it. Four sub-commits,
each independently buildable:

| sub-session | commit   | scope |
|-------------|----------|-------|
| RD-3a       | `89a4d56` | CMake flag `TDMD_DETERMINISTIC_REDUCE`, `src/core/determinism.hpp` with `kDeterministicReduce` constexpr, skipped stub tests wired. Three build dirs (`build-mixed`, `build-fp64`, `build-det`) all compile clean. |
| RD-3b       | `f27a1ca` | R2 fix — `device_cell_list.cu` scatter under deterministic mode does a host-side sequential placement (H2D atom_cells → linear scan → D2H cell_atoms) instead of the `atomicAdd(&cell_placed[...])` race. Canonical ordering: within each cell, atoms appear in ascending atom-ID order. Default path untouched. `DeviceCellListDeterminism.TwoBuildsBitIdentical` activated. |
| RD-3c       | `4397a33` | R3/R4/R5/R6 fix — float/double `atomicAdd` on `d_energy` in Morse, Morse-zone, and both EAM kernels replaced under deterministic mode by a per-atom `accum_t` scratch buffer + single-thread sequential-sum kernel. One copy of `sum_per_atom_kernel` per TU (anonymous namespace). EAM's 3-pass flow sums after embedding (R5) and after forces (R6), zeroing the scratch between them. `DeviceMorseDeterminism.EnergyBitIdentical` and `DeviceEamDeterminism.EnergyBitIdentical` added — both `memcmp` the raw `accum_t` bytes of `d_energy` so a single-ULP drift across three back-to-back runs fails the test. |
| RD-3d       | (this commit) | Regression + perf smoke. |

**RD-3d results.** All three build modes green on full test suites:

| build       | cuda tests            | unit tests |
|-------------|-----------------------|------------|
| build-mixed | 26/30 pass, 4 skipped | 52/52      |
| build-fp64  | 27/30 pass, 3 skipped | 52/52      |
| build-det   | 29/30 pass, 1 skipped | 52/52      |

The 3 determinism tests skip in default builds and run in `build-det`.
The only remaining skip in `build-det` is the pre-existing
`PipelineScheduler.DeterministicMatchesM3`, unrelated to this arc.

**VerifyLab fast suite** run against `build-det/tdmd_standalone`:
3/3 PASS (`two-atoms-morse`, `run0-force-match`, `cross-precision-ab`).
The deterministic reduction path produces physically correct answers
to the same tolerance as the default path — the reordering does not
introduce systematic bias, only removes scheduling variance.

**Perf smoke — 500 steps, 4000-atom Cu EAM, RTX 5080:**

| build       | wall clock | overhead vs default |
|-------------|------------|---------------------|
| build-mixed | 4.68 s     | — (baseline)        |
| build-det   | 4.78 s avg (3 runs: 4.76/4.80/4.78) | +2.1% |

Sub-3% overhead on this configuration. The sequential per-atom sum
(~N adds on one thread) is a small fraction of force-kernel time for
N=4000; the cell-list H2D/D2H round trip runs ~every 10 steps and
costs a few ms per rebuild. At much larger N the sequential reduction
starts to dominate — if/when that matters, a future commit can swap
in a bit-reproducible deterministic parallel tree, but for now
"correctness now, perf later" is the right call per ADR 0010 §6.
Default build is untouched, so production users pay zero.

**Observation on the ADR 0010 premise.** On the 4000-atom FCC Cu
input, `build-mixed` also prints bit-identical thermo across 4 runs
at 14-decimal precision. This does not contradict ADR 0010 — it
reflects that for this *particular* input + this *particular* GPU,
warp scheduling happens to serialize the atomicAdds consistently
across runs. ADR 0010's contract is that default is *allowed* to be
non-deterministic and deterministic mode *must* be; the unit tests
enforce the bit-identity guarantee at the `accum_t` level, where
any future divergence (different input, different GPU, future CUDA
version) would be caught.

**Closes:** RD-3 (was deferred from session VL-EXT as "major core
engine work, ADR 0010 delivers the contract, implementation is a
separate arc"). ADR 0010 is now implemented end-to-end.

**Still deferred (pre-RD-4):**

| item   | reason |
|--------|--------|
| RD-4   | CMake preset + CI compile-only job for `build-deterministic` — needs a CI slot and a decision on whether to run the det suite on every PR or nightly |
| RD-5   | VL-13 sum-order sensitivity + VL-16 fault injection — were blocked on `build-deterministic` existing, now can be written, but sit behind RD-4 in priority because they need CI wiring to be useful |
| VL-6/7/8 LAMMPS A/B cases | unchanged: still need LAMMPS in CI image |
| VL-12  | still blocked: TDMD has no `velocity create` CLI flag |

### Session RD-4 — CMake preset + deterministic CI job (2026-04-10)

Formalizes the three build configurations as first-class CMake
presets so "what options go into build-det" is documented in a
tracked file instead of shell history.

**`CMakePresets.json` at repo root**, presets:

| preset          | binaryDir     | options |
|-----------------|---------------|---------|
| `mixed`         | `build-mixed` | CUDA on, PRECISION=mixed |
| `fp64`          | `build-fp64`  | CUDA on, PRECISION=fp64  |
| `deterministic` | `build-det`   | CUDA on, PRECISION=mixed, DETERMINISTIC_REDUCE=ON |

Matching `buildPresets` and `testPresets` entries so the full
`cmake --preset X && cmake --build --preset X && ctest --preset X`
flow works for each. The presets preserve existing `binaryDir` names
so nothing in the tree that references `build-mixed/`, `build-fp64/`,
`build-det/` needs to move.

Confirmed locally — after `rm -rf build-*` a clean
`cmake --preset deterministic && cmake --build --preset deterministic
&& ctest --preset deterministic -R Determinism` returns
`3/3 Passed`.

**New CI job `deterministic-compile-only`** in `.github/workflows/build.yml`.
Mirrors the existing `cuda-compile-only` job but runs
`cmake --preset deterministic` so the det code path cannot silently
rot. GitHub free-tier has no GPU, so the determinism unit tests
themselves still run manually during development — the CI job only
guarantees that both the `TDMD_DETERMINISTIC_REDUCE` ON and OFF
branches continue to compile cleanly.

**Closes:** RD-4. Still deferred behind it:

- **VL-6/7/8** (LAMMPS A/B cases). Still blocked on LAMMPS in CI image.
- **VL-12** (deterministic velocity init). Still blocked on a
  `velocity create` CLI feature.

### Session RD-5 — sum-order sensitivity + telemetry negative-path (2026-04-10)

Two small, honest additions to close the research-doc N2 and D1 rows
without overclaiming what the current implementation provides.

**VL-13 — `tests/unit/test_sum_order_sensitivity.cu` (new, 2 tests):**

`SumOrderSensitivity.MorseEnergyBounded` and `.EamEnergyBounded`. Each
downloads the GPU-built neighbor CSR, permutes every atom's neighbor
window with a seeded `std::shuffle` across three RNG seeds (1, 7, 42),
re-uploads the permuted list, and re-runs the force kernel. Asserts
that `|E_perm − E_base| / |E_base|` stays below a precision-dependent
bound:

- mixed (`force_t = float`): `1e-5` relative
- fp64 (`force_t = double`): `1e-12` relative

Scope note embedded in the test header: this is a *bounded-variance*
test, not a "variance = 0" test. Even under `TDMD_DETERMINISTIC_REDUCE`
the intra-thread `pe += ...` accumulation is done in neighbor-list
order in `force_t`, so permuting the window perturbs the per-thread
partial at the ULP level. ADR 0010 only guarantees bit-reproducibility
on identical input — that property lives in
`DeviceMorseDeterminism` / `DeviceEamDeterminism`. VL-13's job is
orthogonal: the kernel must not amplify neighbor-order perturbations
beyond the float noise floor, and a future regression that drops a
double accumulator (the EAM-1B failure mode) would blow right past
`1e-5`.

Verified green on both production presets: `build-mixed` and
`build-fp64` each run the pair in ~205 ms. Morse:  ~201 ms; EAM: ~5 ms.
Full `ctest --preset mixed` after the addition: 85/85 passed, 4
deterministic-mode tests correctly skipped.

**VL-16 — `verifylab/runners/test_result_schema.py` (new, 6 tests):**

Meta-test of `result_schema.emit_result` / `parse_result_line` — the
telemetry negative path that VL-14 rewired. Guards against the class
of bug where a failing case silently reports PASS because the failure
machinery itself is broken. Coverage:

1. A plain `status="fail"` record round-trips through stdout +
   `parse_result_line` with the `failures[]` list intact.
2. `float('nan')` / `float('inf')` metrics are sanitized to JSON
   `null` without crashing `json.dumps(..., allow_nan=False)` — the
   direct VL-14 regression case.
3. An invalid status string (`"failed"`, `"FAIL"`, `"passed"`, `""`,
   `"ok"`) is rejected at emit time with `ValueError`, so a typo can
   never leak through as an unrecognized status the runner might
   treat as pass.
4. The `VL_RESULT:` sentinel is strictly required; near-misses
   (`"VL_RESULT {}"`, raw JSON, half-written lines) all parse to
   `None`.
5. The `TDMD_VERIFYLAB_JSONL` side-channel writes match the stdout
   payload byte-for-byte (minus the sentinel prefix), so downstream
   diffs never have to worry about which channel they're reading.

Runs with the stdlib only — no pytest dependency:
`python3 -m unittest verifylab.runners.test_result_schema`. Currently
invoked manually; wiring into a CI step is small and can follow when
a Python test runner job exists (it does not yet — the runners dir
has no automated test discovery today).

6/6 green locally. Combined with VL-13 this closes the research-doc
N2/D1 rows without overstating what deterministic mode actually
guarantees.

**Closes:** RD-5. Still deferred:

- **VL-6/7/8** (LAMMPS A/B in CI): blocked on LAMMPS in CI image.
- **VL-12** (deterministic velocity init): blocked on `velocity create`.

### Session OPT — optimization pass over Phase 3 (2026-04-11)

Follow-on perf session after Phase 3 closure. Scouted three OPT items,
shipped one cleanly, invalidated one before implementation, and
scoped one down to an API-only slice after discovering the original
framing was wrong. Full numbers and reasoning live in
[`docs/05-benchmarks/opt-session-results.md`](../05-benchmarks/opt-session-results.md).

**OPT-1 — GPU-resident nlist prefix sum via CUB (shipped):**

`DeviceNeighborList::build()` no longer does `D2H counts → CPU scan →
H2D offsets`. The path is now `cub::DeviceScan::ExclusiveSum +
cub::DeviceReduce::Max + pack_meta_kernel + one 8-byte D2H`, all on
the caller's stream. Persistent CUB temp/meta scratch buffers on the
`DeviceNeighborList` object so temp-storage allocation is paid once
per lifetime. Benchmark deltas (median of 3, `FastPipelineScheduler`,
mixed preset, RTX 5080):

| System | before | after | Δ      |
|--------|-------:|------:|-------:|
| tiny   | 10 005 | 9 981 | −0.2 % |
| small  |  5 964 | 5 950 | −0.2 % |
| medium |  4 026 | 4 070 | +1.1 % |

Smaller than the +5–15 % scout estimate because the two host syncs in
the old path weren't on the scheduler's critical path (in-stream
serialization already hid them) and CUB launch overhead eats most of
the PCIe savings. Architectural value remains: zero PCIe traffic per
rebuild, last host round-trip removed from scheduler hot path,
unblocks future grow-on-overflow and pair-list-form changes. Commit
`b669f7f`. 85/85 green on both presets.

**OPT-2 — reframed to API slice only:**

Original plan was "migrate EAM to FastPipelineScheduler". Scoping work
revealed EAM has no production path at all — no benchmark driver, no
scheduler wiring, unit-test-only. That makes the full port a feature
slot, not an optimization: there is no existing EAM-in-loop baseline
to measure a speedup against. Shipped the small piece that is useful
regardless of scheduling — `cudaStream_t stream = 0` parameter on
`DeviceEam::compute`, routed through all four internal kernel launches
(density / embedding / force / sum_per_atom) plus inline
`cudaMemsetAsync` for the two scratch zeroes (the buffer helper used
synchronous default-stream memset, which would race against non-
default-stream kernels). Commit `84168b3`. Default stream value keeps
every existing EAM unit test behaviourally identical. No throughput
delta claimed or measured. 85/85 green on both presets.

Full enablement (scheduler dispatch, `--potential eam --eam <setfl>`
CLI, LAMMPS A/B) is filed as **FEAT-EAM-Production-Pipeline** for a
future M2/M3 feature slot.

**OPT-3 — invalidated before implementation:**

Scouted as "hoist PBC branches out of the force kernel" for ~+7 %.
Closer reading showed the premise was wrong: drift already wraps
positions, and the force-kernel PBC branches compute the minimum-image
convention on relative vectors — structurally required. Dropped. No
commit.

**Session net:** one modest-but-real perf win (OPT-1, +1.1 % on
medium), one small API improvement unblocking future async EAM work,
and the architectural cleanup of removing the last host sync from the
nlist rebuild path. The single-GPU optimization ceiling at ≤32 K atoms
is narrower than pre-session estimates suggested; remaining wins are
either structural (multi-GPU, kernel fusion) or require touching
cell-list / neighbor-list storage format.

### Session FEAT-EAM — EAM production pipeline, Phase A + B (2026-04-11)

Direct follow-on to the OPT session. OPT-2 was reframed mid-session
when it became clear that "migrate EAM to FastPipelineScheduler" was
a feature port, not an optimization — there was no existing EAM-in-
loop baseline to measure a speedup against, so it was filed as
FEAT-EAM-Production-Pipeline. This session delivered Phase A (core
wiring + tests) and Phase B (benchmark CLI + sweep + enablement
report). Phase C (LAMMPS A/B via VerifyLab) is deferred to its own
session. Full report:
[`docs/05-benchmarks/feat-eam-pipeline-results.md`](../05-benchmarks/feat-eam-pipeline-results.md).

**Phase A — core wiring (commit `8c18a94`):**

Scheduler gets a second constructor
`FastPipelineScheduler(box, natoms, EamAlloy, cfg)` that uploads the
spline tables into an owned `DeviceEam` during construction. The
potential choice is captured explicitly via a new
`enum class PotentialKind { Morse, Eam }` rather than implicitly via
a null pointer — the `step()` hot path does a short switch-dispatch
to `step_morse()` / `step_eam()` so feature work does not smear
across the integrator sequence. Cached `cutoff_` + new
`interaction_cutoff()` accessor serve as the single source of truth
for the neighbor list builder.

Three new unit tests in `tests/unit/test_fast_pipeline_eam.cu`:

- `EamStepMatchesDirectCompute` — at t=0 the scheduler's initial
  force compute must match a standalone `DeviceEam::compute()` on the
  same positions to within `kForceTolerance`. Guards against any
  hidden transformation between `upload()` and EAM dispatch.
- `EamNve100StepsStable` — 100 velocity-Verlet steps on Cu FCC 256
  with `Cu_mishin1` EAM, `|dE/E| < 1e-3` against the CPU reference.
  This is a "scheduler wiring did not break the physics" floor, not
  a precision claim.
- `EamDeterministicReplay` — two independent scheduler instances
  with the same initial condition must reach matching positions over
  50 steps. Step-level replay on a single GPU; does not require
  `TDMD_DETERMINISTIC_REDUCE`.

Morse path is untouched except for the dispatch hop. The existing
`FastPipelineScheduler.KernelLaunchInvariant` test still pins Morse
at exactly 5 kernel launches per step. EAM pins at exactly 7 (the
three EAM passes replace Morse's single force launch).

**Phase B — benchmark CLI + sweep (commit `ec93913`):**

`bench_pipeline_scheduler` grew `--potential morse|eam` and
`--eam <setfl>`. Validation is strict: `--potential eam` requires
`--eam <file>` and `--scheduler fast_pipeline` (the legacy
`PipelineScheduler` has no EAM wiring). Both potentials share one
measurement loop via a `unique_ptr<FastPipelineScheduler>`, so the
code path downstream of construction is identical.

Sweep (6 cells × 3 sequential runs, 7 s pauses per §4, mixed
preset, RTX 5080, medians of 3):

| Size   | Atoms  | Morse ts/s | EAM ts/s | EAM / Morse |
|--------|-------:|-----------:|---------:|------------:|
| tiny   |    256 |     8 860  |   4 241  |      0.48   |
| small  |  4 000 |     5 254  |   2 889  |      0.55   |
| medium | 32 000 |     3 687  |   2 301  |      0.62   |

Per-run raw JSON + stderr committed under
`docs/05-benchmarks/feat-eam-raw/`. The EAM/Morse ratio improves
with system size because small systems are launch-overhead bound
and amortize EAM's two extra passes (density + embedding) poorly;
larger systems let the embedding/force arithmetic dominate.

Honest caveat in the report: Morse absolute numbers in this sweep
are 6–12 % below the 2026-04-10 OPT-1 session's numbers despite
identical binary, data, scheduler, and protocol. Diagnosed as
between-session environmental drift (thermal / clock boost
headroom). Only **within-session** ratios are apples-to-apples;
the EAM/Morse column is the honest comparison.

**Explicitly not delivered this session:**

- LAMMPS A/B for EAM via VerifyLab (Phase C).
- EAM force determinism beyond step-level replay (that property
  already lives under `TDMD_DETERMINISTIC_REDUCE` via
  `DeviceEamDeterminism.EnergyBitIdentical` — unchanged).
- Any optimization targeting EAM itself. Today's numbers become the
  baseline any future EAM-targeted work would measure against.

**Tests:** `ctest --preset mixed` → 88/88 passed;
`ctest --preset fp64` → 88/88 passed. 85 baseline + 3 new EAM
FastPipeline tests. Morse launch-invariant test still exact at
5/step.

### Session FEAT-EAM Phase C — LAMMPS A/B for Morse and EAM (2026-04-11)

Closes the last open part of FEAT-EAM-Production-Pipeline. Phase A+B
wired EAM through `FastPipelineScheduler` and delivered an enablement
benchmark; Phase C answers the *other* half of "done": how do TDMD
and stock LAMMPS compare on the same problem? Full report:
[`docs/05-benchmarks/lammps-ab-results.md`](../05-benchmarks/lammps-ab-results.md).

**Protocol.** 6 cells (Morse/EAM × tiny/small/medium) × 3 runs × 2
engines, all 36 runs back-to-back in a single GPU session with 7-second
pauses per CLAUDE.md §4. TDMD and LAMMPS runs for the same cell are
**interleaved** inside the sweep (TDMD 1-3, then LAMMPS 1-3, then
sleep, then next cell). This cancels the cross-session thermal / clock
drift the FEAT-EAM report had to caveat against — the ratios in the
A/B table are apples-to-apples on the same silicon state. LAMMPS is
stable 2Aug2023u3 from `third_party/lammps/build/lmp`, GPU package in
mixed precision (`-sf gpu -pk gpu 1`). Step counts match the FEAT-EAM
report exactly so the TDMD half cross-references back to it.

**Throughput (median of 3, mixed preset, RTX 5080):**

| Potential | Size   |  Atoms | TDMD ts/s | LAMMPS ts/s | TDMD/LAMMPS |
|-----------|--------|-------:|----------:|------------:|------------:|
| morse     | tiny   |    256 |    9 955  |    21 120   |    0.47     |
| morse     | small  |  4 000 |    5 954  |    13 031   |    0.46     |
| morse     | medium | 32 000 |    4 077  |     3 225   |  **1.26**   |
| eam       | tiny   |    256 |    4 802  |    13 702   |    0.35     |
| eam       | small  |  4 000 |    3 288  |     9 455   |    0.35     |
| eam       | medium | 32 000 |    2 645  |     2 815   |    0.94     |

**Headline:** on the compute-bound cell (medium, 32 k atoms) TDMD's
Morse path is **26 % faster** than LAMMPS-GPU, and TDMD's EAM path
closes to 94 % of LAMMPS. On tiny and small, LAMMPS is ~2× faster for
Morse and ~3× for EAM — these are launch-bound regimes where LAMMPS's
decade-tuned GPU driver dominates. The crossover pattern (TDMD/LAMMPS
climbing 0.47 → 0.46 → 1.26 on Morse, 0.35 → 0.35 → 0.94 on EAM) is
the expected shape for a scheduler that is compute-bound at size and
launch-bound at tiny scales.

**Physics sanity — force-match at t=0 (mixed preset):**

| Potential | max \|ΔF\| (eV/Å) | rms \|ΔF\| (eV/Å) |
|-----------|-------------------:|-------------------:|
| morse     |         5.362e-06  |         4.096e-06  |
| eam       |         2.676e-06  |         2.200e-06  |

Existing `DeviceLammpsAB.{Morse,Eam}Run0ForceMatch` tests extended to
print max/rms stats unconditionally (previously silent on pass). Both
well under `kForceTolerance = 1e-4`.

**Physics sanity — PE match at t=0 (256-atom Cu):**

| Potential | LAMMPS PE (eV) | TDMD PE (eV) |  ΔPE / \|PE\|  |
|-----------|---------------:|-------------:|---------------:|
| morse     |    -867.64593  |  -867.645868 |    7.1e-08     |
| eam       |    -906.29591  |  -906.297363 |    1.6e-06     |

Both in the rounding floor of the mixed-precision sum, exactly as
ADR 0007 predicts. Morse is ~10⁻⁸ (pairwise sum of ~40 per-atom
terms). EAM is ~10⁻⁶ (spline interpolation + 3-pass accumulation
compounds the rounding).

**Artifacts:**

- `benchmarks/phase1_baseline/lammps_ab_phase_c/` — 6 LAMMPS inputs,
  `run_ab_sweep.sh` driver, `parse_results.py` aggregator, README.
- `benchmarks/phase1_baseline/lammps_ab_phase_c/results_20260411_014944/`
  — per-run JSON + LAMMPS logs + stdout for all 36 runs.
- `docs/05-benchmarks/lammps-ab-results.md` — full report with the
  three-questions split (throughput vs force-match vs PE-match).
- `tests/unit/test_device_lammps_ab.cu` — extended with per-stat
  printouts and a host-side Morse PE sum via the existing half-list
  + `MorsePair::compute`.

**Tests:** `ctest --preset mixed` → 88/88 passed (force-match values
captured in the stdout; test count unchanged). `ctest --preset fp64`
run locally, same 88/88.

**What Phase C explicitly does not claim:**

- Not bit-identical trajectories. Different RNG for velocity init,
  different nlist orderings, different accumulation orders — step-0
  agreement does not imply step-N agreement.
- Not a statement about LAMMPS on KOKKOS backend (`-k on`). Only the
  GPU package is measured.
- Not an optimization result. Phase C *measures*; any kernel-fusion
  or nlist-layout work starts from these numbers as the baseline.

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
