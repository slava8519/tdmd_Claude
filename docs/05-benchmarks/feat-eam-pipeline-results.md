# FEAT-EAM-Production-Pipeline benchmark results (2026-04-11)

First-ever EAM-through-`FastPipelineScheduler` numbers. This is an
**enablement benchmark**, not an optimization benchmark: there was no
prior EAM-in-loop baseline to compare against, so there is no "X%
faster" headline to report. The value here is (a) establishing the
absolute throughput numbers EAM delivers through the scheduler async
contract, and (b) confirming the per-step kernel-launch invariant.

Scheduler: `FastPipelineScheduler`. Mixed-precision preset. RTX 5080,
CUDA 12.6. EAM potential: `tests/data/Cu_mishin1.eam.alloy` (Cu,
1-type). Data files: `tests/data/cu_fcc_256.data` (tiny),
`benchmarks/phase1_baseline/small.data` (small),
`benchmarks/phase1_baseline/medium.data` (medium).

Protocol: 3 sequential runs per cell with 7 s pauses between runs
(CLAUDE.md §4 GPU-exclusive rule). Same `bench_pipeline_scheduler`
binary for both potentials, single commit, single build. `t_init=300K`,
`seed=42`, `rebuild_every=10`, `skin=1.0`, `dt=0.001`. Median of 3
reported.

## Per-step kernel-launch invariant

Pinned before reading any throughput numbers — guards against the
scheduler silently inserting extra launches in either path:

| Potential | Launches / step | Kernels                                                                    |
|-----------|----------------:|----------------------------------------------------------------------------|
| Morse     |               5 | half_kick · drift · zero_forces · morse_force · half_kick                  |
| EAM       |               7 | half_kick · drift · zero_forces · eam_density · eam_embedding · eam_force · half_kick |

Pinned via `kernel_launches_per_step` in the JSON output of every run
in the sweep. The `FastPipelineScheduler.KernelLaunchInvariant` unit
test still holds for the Morse branch post-refactor (`tests/unit/
test_fast_pipeline.cu`).

## Throughput (median of 3)

| Size   | Atoms  | steps | warmup | Morse ts/s | EAM ts/s | EAM / Morse |
|--------|-------:|------:|-------:|-----------:|---------:|------------:|
| tiny   |    256 |  5000 |    500 |     8 860  |   4 241  |      0.48   |
| small  |  4 000 |  2000 |    200 |     5 254  |   2 889  |      0.55   |
| medium | 32 000 |  1000 |    100 |     3 687  |   2 301  |      0.62   |

Raw per-run numbers live in `feat-eam-raw/*.json`. Spread inside each
cell is ≤3 % (noise is dominated by run-to-run GPU clock jitter, not
the workload).

## Reading the numbers

**Ratios improve with system size.** EAM/Morse climbs from 0.48 (tiny)
→ 0.55 (small) → 0.62 (medium). This is the expected pattern for a
3-pass force compute vs a single-pass one:

- On `tiny` the ~2–3 µs kernel launch overhead per EAM pass is a
  non-trivial fraction of the per-step cost, so EAM pays the launch
  overhead three times against Morse's one. Small systems are launch-
  bound.
- On `medium` the per-kernel grid is large enough that launch
  overhead is amortized; EAM's extra work is dominated by the
  embedding and force passes' arithmetic + bandwidth. EAM still does
  more work per atom than Morse (density sweep, spline evaluation,
  second neighbor sweep), so it cannot reach parity — but the gap
  narrows toward the compute-bound regime.

**Note on Morse absolute numbers vs the OPT-1 session.** The OPT-1
report on 2026-04-10 measured `tiny=10 005`, `small=5 964`,
`medium=4 026` ts/s for Morse on the same binary, same scheduler,
same data files. Today's numbers are 6–12 % lower across the board.
Both sessions saw spread ≤3 % *within* a session; the difference is
between sessions (thermal state, clock boost headroom, background
GPU activity at measurement time). The OPT-1 report already flagged
a similar gap against the older Phase 2 / Phase 3 historical figures
(~40 % wider). Conclusion: compare **within-session ratios**, not
absolute ts/s across sessions. The EAM/Morse ratio column in the
table above is apples-to-apples because both potentials were measured
in the same sweep with 7 s pauses between runs.

## What this session delivered

- **EAM is now a first-class citizen of `FastPipelineScheduler`.**
  Before this session EAM only ran inside unit tests driving
  `DeviceEam::compute` directly. Now any production caller can
  construct `FastPipelineScheduler(box, natoms, EamAlloy, cfg)` and
  get the same async single-stream contract Morse has had since
  Phase 2.
- **Benchmark driver grew `--potential eam --eam <setfl>`**. Same
  binary, same JSON schema, same warmup/measurement shape as the
  Morse path. Reviewers can re-run either potential with one flag
  change. EAM is gated to `--scheduler fast_pipeline` because the
  legacy `PipelineScheduler` has no EAM wiring (and nothing in the
  current roadmap asks it to grow one).
- **Three unit tests lock the wiring**:
  `FastPipelineSchedulerEam.EamStepMatchesDirectCompute`,
  `EamNve100StepsStable`, and `EamDeterministicReplay`. Mixed + fp64
  both green.

## What this session intentionally did NOT deliver

- **No LAMMPS A/B for EAM via VerifyLab.** Phase C of the original
  FEAT plan. Deferred to its own session because golden-file
  generation, spec design, and CI integration should not be mixed
  with scheduler wiring.
- **No EAM force determinism guarantee across runs.** The
  `EamDeterministicReplay` unit test checks *step-level* replay on a
  single GPU instance (same launch order, same nlist schedule → same
  positions). It does not promise bit-identical energy reductions —
  that property lives under `TDMD_DETERMINISTIC_REDUCE` and is
  guarded by `DeviceEamDeterminism.EnergyBitIdentical`, unchanged by
  this session.
- **No "EAM is X% faster" headline.** There was nothing to be X%
  faster than. Any future optimization pass that targets EAM can use
  today's numbers as the baseline.
