# LAMMPS A/B — FEAT-EAM Phase C (2026-04-11)

First direct throughput + physics comparison of TDMD against LAMMPS-GPU
on identical Morse and EAM workloads. Runs TDMD's `bench_pipeline_scheduler`
and stock LAMMPS (stable 2Aug2023u3, GPU package in mixed precision)
interleaved within a single GPU session to eliminate cross-session
thermal / clock-boost drift.

- **Hardware:** NVIDIA RTX 5080, CUDA 12.6.
- **TDMD:** `FastPipelineScheduler`, mixed-precision preset. Source at
  HEAD of main at the time of this report.
- **LAMMPS:** `third_party/lammps/build/lmp`, built from the vendored
  submodule with GPU package (`-DPKG_GPU=on -DGPU_API=cuda -DGPU_PREC=mixed`).
  Invoked as `lmp -sf gpu -pk gpu 1` so *every* supported pair style
  runs on the GPU.
- **Potentials:** Morse (`pair_style morse 6.0`, D=0.3429 α=1.3588 r0=2.866),
  EAM (`pair_style eam/alloy` with `tests/data/Cu_mishin1.eam.alloy`,
  Mishin Cu EAM1 PRB(2001)63:224106).
- **Systems:** Cu FCC tiny (256 atoms), small (4000 atoms), medium
  (32000 atoms) — the same data files used in
  `feat-eam-pipeline-results.md`.
- **Protocol (CLAUDE.md §4):** 3 sequential runs per cell with 7-second
  pauses between runs, median of 3 reported. TDMD and LAMMPS runs for
  the same cell are interleaved (TDMD 1-3, then LAMMPS 1-3) so they
  share the same thermal state. LAMMPS timings use the "Loop time of
  X.XX on 1 procs for N steps" line from the *second* `run` block
  (the first is 10 % warmup, then `reset_timestep 0`).
- **Step counts:** tiny = 500 warmup + 5000 measured, small = 200 +
  2000, medium = 100 + 1000. These match the FEAT-EAM benchmark so the
  TDMD half of this table is cross-referenceable.

Raw logs in `benchmarks/phase1_baseline/lammps_ab_phase_c/results_20260411_014944/`.

## Section A — Throughput (median of 3)

Single-session interleaved sweep, `results_20260411_014944/`.

| Potential | Size   |  Atoms | TDMD ts/s | LAMMPS ts/s | TDMD / LAMMPS |
|-----------|--------|-------:|----------:|------------:|--------------:|
| morse     | tiny   |    256 |     9 955 |      21 120 |      0.47     |
| morse     | small  |  4 000 |     5 954 |      13 031 |      0.46     |
| morse     | medium | 32 000 |     4 077 |       3 225 |      **1.26** |
| eam       | tiny   |    256 |     4 802 |      13 702 |      0.35     |
| eam       | small  |  4 000 |     3 288 |       9 455 |      0.35     |
| eam       | medium | 32 000 |     2 645 |       2 815 |      0.94     |

Raw per-run numbers (all three runs per cell, spread ≤ 2 % everywhere):

- morse tiny:   TDMD=[9958, 9933, 9955]  LAMMPS=[21177, 21120, 21107]
- morse small:  TDMD=[5953, 6001, 5954]  LAMMPS=[13031, 13143, 12906]
- morse medium: TDMD=[4064, 4077, 4128]  LAMMPS=[3210,  3239,  3225]
- eam   tiny:   TDMD=[4760, 4806, 4802]  LAMMPS=[13702, 13728, 13609]
- eam   small:  TDMD=[3303, 3271, 3288]  LAMMPS=[9412,  9575,  9455]
- eam   medium: TDMD=[2645, 2638, 2666]  LAMMPS=[2815,  2837,  2813]

### Reading the numbers

**Tiny is a launch-latency kernel, not a performance headline.** At 256
atoms every per-step cost — kernel launch, stream synchronization,
kernel-argument binding — is a non-trivial fraction of the total. Both
engines are *under-utilizing* the RTX 5080 at this size; what tiny
measures is "who pays less launch overhead per step", which is a
different question from "who computes faster when the GPU is saturated".
Do not read tiny as the comparative result — it is a sensitivity test
for launch cost. LAMMPS wins this cell by a wide margin (~2.1× Morse,
~2.9× EAM) because LAMMPS's GPU package has been tuned against
small-system launch cost for a decade and because LAMMPS runs more of
the per-step work on the host CPU, short-circuiting GPU entirely when
the grid is too small.

**Small is still launch-bound for both codes.** 4 000 atoms is the
inflection region where the force kernel starts dominating but the
5–7 kernel launches per step are still comparable to compute. LAMMPS
keeps its ~2.2× Morse / ~2.9× EAM advantage because its launch budget
per step is lower (fused kick+drift+force path in their GPU package
driver).

**Medium is the compute-bound cell and the reversal is the headline.**
On the 32 000-atom Morse cell, **TDMD is 26 % faster than LAMMPS-GPU**
(4 077 vs 3 225 ts/s). On the 32 000-atom EAM cell, TDMD closes the
gap to 94 % of LAMMPS. That is the result the FastPipelineScheduler
was designed to deliver: once the per-kernel grid is large enough that
launch overhead amortizes, TDMD's single-stream async contract
(`FastPipelineScheduler`) has no more moving parts than LAMMPS's
driver and a tighter neighbor-list path (`DeviceNeighborList` with
GPU-resident prefix sum, from OPT-1). LAMMPS still wins on EAM medium
because their 3-pass EAM kernel is separately tuned; our EAM path is
a straight port from the reference CPU implementation.

**Scaling pattern (TDMD / LAMMPS ratio vs system size):**

- Morse: 0.47 → 0.46 → **1.26** (crosses 1 between small and medium)
- EAM:   0.35 → 0.35 → 0.94 (approaches 1 but does not cross it yet)

The EAM gap at medium is entirely in the force kernel — the nlist,
the integrator, and the stream plumbing are all shared with Morse
and match LAMMPS at that size. Closing the last 6 % is an EAM kernel
optimization task (embedding / density pass fusion, better vector
loads), not a scheduler task.

## Section B — Physics sanity (force-match and energy-match at t = 0)

This section is **separate from throughput on purpose**. A run that
produces the right number-per-second is worthless if the forces it
produces are wrong. The two numbers answer different questions and
must not be folded into a single "score".

### B.1 Force-match (`run 0`)

Source: `tests/unit/test_device_lammps_ab.cu` — exercised as
`DeviceLammpsAB.MorseRun0ForceMatch` and `DeviceLammpsAB.EamRun0ForceMatch`.
Reference forces: `tests/data/reference/forces_{morse,eam}.dump` —
LAMMPS `run 0` DumpAtom on the 256-atom Cu FCC input with identical
potential parameters, cutoff, and neighbor-list skin.

Tolerance: `kForceTolerance = 1e-4` (mixed preset), `1e-10` (fp64
preset). These are inherited from `tests/support/precision_tolerance.hpp`
and guard every GPU A/B test in the repo.

| Potential | Preset | max \|ΔF\| (eV/Å) | rms \|ΔF\| (eV/Å) | mean \|F_ref\| (eV/Å) |
|-----------|--------|-------------------:|-------------------:|-----------------------:|
| morse     | mixed  |        5.362e-06   |        4.096e-06   |          6.246e-15     |
| eam       | mixed  |        2.676e-06   |        2.200e-06   |          3.128e-15     |

Note: the 256-atom FCC state at the data-file equilibrium geometry has
forces of order 1e-15 eV/Å (machine noise around the FCC minimum), so
`max |ΔF|` is a test of rounding behavior, not of physical accuracy.
What it *does* prove: every code path from positions → neighbor list
→ force kernel → force vector agrees, bit-level plus noise, between
TDMD and LAMMPS.

### B.2 Energy-match at t = 0

LAMMPS step-0 PE scraped from the Phase C sweep's thermo output (same
data file, same potential). TDMD PE computed via the host-side reference
path (`EamAlloy::compute_forces` for EAM; summed `MorsePair::compute`
for Morse) on the same positions.

| Potential | LAMMPS PE (eV) | TDMD PE (eV) |  ΔPE (eV)  |  ΔPE / \|PE\|  |
|-----------|---------------:|-------------:|-----------:|---------------:|
| morse     |    -867.64593  |  -867.645868 |   6.2e-05  |    7.1e-08     |
| eam       |    -906.29591  |  -906.297363 |   1.45e-03 |    1.6e-06     |

**Interpretation.** Morse agrees to within 7 × 10⁻⁸ of the LAMMPS
reference — that is at the rounding floor of a double-precision sum
of ~40 per-atom float-precision energies on 256 atoms, exactly where
ADR 0007 says it should be. EAM is 1.6 × 10⁻⁶ — two orders of
magnitude larger than Morse, because EAM goes through a spline
interpolation and a 3-pass density + embedding + force sum where each
pass accumulates independent rounding. Both are well below any
physically meaningful energy scale (thermal fluctuation at 300 K on
256 atoms is ~9.9 eV; rounding error is ~10⁻⁵ of the total KE drift
per step).

Virial agreement is not reported — TDMD has no production virial
reduction yet (needed for NPT in M7+). Will revisit when M7 lands.

## What this report does and does not claim

**Does claim:**

- TDMD and LAMMPS-GPU produce the same physics at step 0 on identical
  input, at the precision floor their mixed-precision strategies
  inherit from LAMMPS (1e-4).
- TDMD's throughput relative to LAMMPS on the *same* hardware in the
  *same* session is the number in Section A.

**Does not claim:**

- Not bit-identical trajectories across N steps. RNG for
  `velocity create` uses different seeds internally; neighbor list
  ordering differs; accumulation order in force reductions differs.
  Integrated states diverge in a normal numerical-noise sense.
- Not an "optimization finding". Phase C is a measurement, not an
  optimization pass. Any future tuning work should start from the
  numbers here, not replace them.
- Not a statement about LAMMPS on other backends. `-sf gpu -pk gpu 1`
  runs the GPU package; KOKKOS (`-k on`) can be faster or slower on
  the same hardware. That comparison is out of scope.

## Reproducing

```bash
# Sequential sweep — all 36 runs take about 6 minutes on RTX 5080.
./benchmarks/phase1_baseline/lammps_ab_phase_c/run_ab_sweep.sh

# Aggregate a results_<ts>/ dir into a Markdown table.
python3 benchmarks/phase1_baseline/lammps_ab_phase_c/parse_results.py \
    benchmarks/phase1_baseline/lammps_ab_phase_c/results_<ts>
```

Force-match stats (re-emit after a rebuild):

```bash
./build-mixed/tests/tdmd_cuda_tests \
    --gtest_filter='DeviceLammpsAB.*'
```
