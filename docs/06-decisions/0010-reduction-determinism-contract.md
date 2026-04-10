# 0010 — Reduction determinism contract for GPU kernels

- **Status:** Proposed
- **Date:** 2026-04-10
- **Decider:** human + architect
- **Affected milestone(s):** Phase 4, gates VL-13 and VL-16

## Context

Time-decomposition molecular dynamics pipelines the **same spatial region
through overlapping time-slices**. The scheduler's correctness proof rests
on an invariant of the form "the state produced by zone A at slice k is
exactly the state that would be produced by recomputing zone A from the
same input". If the underlying force kernel returns **bit-different
results from run to run on the same input**, this invariant becomes
impossible to assert: a scheduler bug and a reduction-order artifact look
identical in any diff.

VerifyLab already feels this. VL-4 (`cross-precision-ab`) is zero-threshold
against itself in principle, but observed on 2026-04-10 it sits at
`|dF| ~ 6.8e-4 eV/Å` — partly from legitimate float32 propagation through
the force kernel, and partly from GPU atomic-ordering noise that we cannot
currently isolate. VL-13 (sum-order sensitivity) and VL-16 (fault injection)
cannot be written as strict-equality tests until the engine exposes a
deterministic mode. VL-16 in particular wants "this specific injected
fault always flips exactly this specific atom's force" — not
"it probably flips something within a noise envelope".

### Baseline state: current reductions in the engine (RD-1 audit)

The audit grepped every reduction primitive in `src/` on 2026-04-10.
Complete inventory:

| # | File : line | Sink | Source | Role | Deterministic? |
|---|---|---|---|---|---|
| R1 | `src/domain/device_cell_list.cu:40` | `cell_counts[cell_id]` | `1` | cell histogram (atom count per cell) | ✅ outcome deterministic (commutative integer sum) |
| R2 | `src/domain/device_cell_list.cu:52` | `cell_placed[cell_id]` | `1` | slot reservation via fetch-add | ❌ **order-dependent**: per-cell atom ordering depends on thread schedule |
| R3 | `src/potentials/device_morse.cu:96` | `d_energy` (accum_t) | per-atom `pe` (force_t) | total PE scalar | ❌ **non-associative sum** across ~natoms contributions |
| R4 | `src/potentials/device_morse_zone.cu:91` | `d_energy` (accum_t) | per-atom `pe` (force_t) | total PE, zone scope | ❌ same as R3 |
| R5 | `src/potentials/device_eam.cu:132` | `d_energy` (accum_t) | embedding energy per atom (real) | embedding PE | ❌ same pattern |
| R6 | `src/potentials/device_eam.cu:237` | `d_energy` (accum_t) | pair PE half-sum per atom (force_t) | pair PE | ❌ same pattern |

**Not found** (which is structurally important): **no `cub::BlockReduce`,
no `__shfl`, no `thrust::reduce`, no `reduce_by_key`, no shared-memory
reductions, no Kahan summation, no ordered prefix sums, no deterministic
accumulators of any kind.**

**Also not found**: forces are **not** atomic-accumulated. Every force
kernel in TDMD uses the full-list (`newton off`) pattern: one thread per
atom `i`, the thread accumulates `forces[i]` in registers over its private
neighbor loop, then writes back with a single non-atomic store
(`device_morse.cu:91-93`, `device_eam.cu:232-234`). This means **force
values are deterministic given a fixed neighbor traversal order**.

### What this implies

The entire non-determinism surface in TDMD reduces to two questions:

1. **Is the neighbor traversal order the same between two runs?** It is
   not today, because R2 (slot reservation via atomic fetch-add in the
   cell-list kernel) makes intra-cell atom ordering schedule-dependent,
   which cascades into neighbor list ordering, which cascades into the
   order in which each per-atom force loop sums pair contributions.
   Float addition is non-associative, so different summation orders
   produce different `forces[i]` values — at the ulp-to-few-ulp level,
   but non-zero.
2. **Is the energy scalar reduction order the same between two runs?**
   It is not, because R3–R6 use free-for-all `atomicAdd` on the global
   energy scalar.

Fix #1 and all per-atom forces become bitwise reproducible run-to-run.
Fix #2 and the global energy becomes bitwise reproducible run-to-run.
These are the two and only two items on the path to a deterministic
TDMD build.

### What LAMMPS does

LAMMPS has a long-standing `neigh_modify binsize` and atom-ID-based
sort: atoms within a cell are sorted by atom ID on every neighbor
rebuild, which removes the schedule dependency for free
(`lib/gpu/lal_neighbor_gpu.cu` and friends do the same on GPU). For
energies, LAMMPS uses per-atom arrays (`compute pe/atom`) that are
summed via a parallel reduce whose order is fixed by atom ID, not by
thread ordering. This is the exact pattern we should copy (see ADR
0008).

## Options considered

### Option A — Force determinism unconditionally, take the perf hit

Rewrite R2 to place atoms in cells by global atom ID (not via atomic
fetch-add), and rewrite R3–R6 to accumulate per-atom energies into a
buffer that is then reduced by a deterministic tree-reduction ordered
by atom ID.

- **Pros:** Simple model — there is one mode. Every run is reproducible.
  VerifyLab becomes strict-equality everywhere.
- **Cons:** Per-cell ID sort costs O(cell_size · log cell_size) per
  rebuild. Energy tree-reduce is a second kernel launch. Estimated cost:
  2–5% on medium, 5–10% on large, 1% on tiny. Unacceptable for the
  production path where the difference between 91% and 85% of LAMMPS
  throughput is the difference between "faster than LAMMPS" and "not
  faster than LAMMPS".

### Option B — Two modes, compile-time flag

Keep the current fast path as `TDMD_DETERMINISTIC_REDUCE=OFF` (default,
production). Add `TDMD_DETERMINISTIC_REDUCE=ON` that rewires exactly
the six sites above to their deterministic equivalents.

- **Pros:** Production keeps current performance. VerifyLab gets a
  build target it can assert bit-equality against. The set of affected
  sites is *small and enumerated* — the audit above is the whole list,
  not a starting point. No scope creep risk.
- **Cons:** Two code paths to maintain. Requires a third build
  directory in CI (`build-deterministic/`). Must hold the invariant
  that both paths compute the **same mathematical quantity**, only the
  summation order differs.

### Option C — Leave it, document the noise floor instead

Accept that reductions are noisy, document the floor per platform, and
write all VerifyLab thresholds above the floor.

- **Pros:** Zero code change.
- **Cons:** Noise floor is platform-dependent (different GPUs, different
  drivers, even different CUDA toolkit versions) — thresholds rot every
  time infra changes. VL-13 and VL-16 become impossible to write as
  assertions; they would become "trend watchers" which are cosmetic.
  Worst of all: TD scheduler correctness remains undecidable when the
  scheduler is suspected of a bug, because force diffs between expected
  and actual zone state cannot be tied to a specific cause.

## Decision

**Option B: two reduction modes via compile-time flag
`TDMD_DETERMINISTIC_REDUCE`, default `OFF`.**

Rationale: the audit showed the problem is bounded to six call sites.
Option A's perf cost is real and hits the production path that
benchmarks and shipping care about. Option C leaks cost into every
future test that needs to distinguish scheduler bugs from reduction
noise — which is most of the tests that matter for TD.

### Contract

**What `TDMD_DETERMINISTIC_REDUCE=ON` guarantees:**

1. **Bit-equal per-atom forces** between any two runs, on the same GPU,
   same CUDA toolkit, same scheduler, same input, same MPI rank count.
   Measured: `memcmp(forces_run1, forces_run2) == 0`.
2. **Bit-equal global energies** (PE, embedding energy, pair energy)
   under the same conditions.
3. **Bit-equal virial** (when implemented — currently not accumulated).
4. **Independence from kernel launch configuration:** changing block
   size or grid size must not change the bit result. This is the
   strongest version of the guarantee and forces both R2 and R3-R6
   fixes to be launch-config-independent.

**What `TDMD_DETERMINISTIC_REDUCE=ON` does NOT guarantee:**

1. **Cross-GPU reproducibility.** Different GPU architectures have
   different transcendental intrinsic implementations (`expf`, `sqrtf`);
   we do not attempt to paper over this.
2. **Cross-precision reproducibility.** Mixed-mode bit-equal to fp64 is
   not a goal — that is what VL-4 is for.
3. **Cross-MPI-rank-count reproducibility.** This is a separate and
   larger problem (the domain decomposition itself changes), tracked
   as future work.
4. **Cross-compiler reproducibility.** NVCC version changes may alter
   PTX emission; this is out of scope.

**What changes at each of the six audit sites:**

- **R1** (`cell_counts` histogram): stays atomic. Outcome is already
  deterministic.
- **R2** (`cell_placed` slot reservation): removed. Replaced by a
  two-phase kernel: phase 1 computes `cell_counts` and `cell_offsets`
  as now; phase 2 scans atoms in global-ID order and places each atom
  at `offsets[cell_id] + (local_index_by_id_within_cell)`. The local
  index is computed by counting how many atoms with smaller global ID
  share the same cell — O(cell_size) work per atom, but cell_size is
  small (typically 4–10 atoms for a metal), so the cost is bounded.
  (LAMMPS does essentially this sort on CPU and a device-side radix
  sort on GPU — see `lib/gpu/lal_neighbor_gpu.cu` for reference.)
- **R3–R6** (`d_energy` atomics): replaced by a per-atom energy buffer
  (`accum_t* d_energy_per_atom`) that is written non-atomically by the
  force kernel (each thread owns atom `i`'s slot), then reduced to the
  scalar `d_energy` by a second kernel that performs a segmented
  tree-reduction in fixed atom-ID order. Kernel launch: one block, each
  thread handles `natoms / blockDim.x` contiguous atoms, warp-shuffle
  reduction inside each thread, then a fixed binary tree across threads.
  This pattern is bit-reproducible because (a) each thread's input range
  is fixed by atom ID and (b) the intra-warp shuffle and inter-warp
  tree both have fixed order.

**Performance budget**

- Expected slowdown in deterministic mode: **1.5–3× on medium (32k
  atoms)**, dominated by the per-atom energy reduction kernel launch
  overhead on small/tiny, and by the cell-list ID sort on large.
- **Production mode must not regress.** CI includes a benchmark gate
  on `build-mixed/` against Phase 3 baseline; adding the flag must not
  change that number.

### Usage policy

**Who uses `TDMD_DETERMINISTIC_REDUCE=ON`:**

- All VerifyLab cases that assert bit-equality (VL-13, VL-16).
- Unit tests that regression-check force values against golden numbers.
- Debugging sessions trying to separate a scheduler bug from reduction
  noise.
- CI's third build target `build-deterministic/` (added in RD-4).

**Who does NOT use it:**

- `benchmarks/` — always runs production mode. Deterministic mode
  numbers are never reported as throughput.
- Production runs (the standalone driver default).
- Default CMake build (`cmake -B build`) without explicitly passing the
  flag.

### Build-target layout after RD-4

```
build-mixed/          -DTDMD_PRECISION=mixed    -DTDMD_DETERMINISTIC_REDUCE=OFF  # production fast path
build-fp64/           -DTDMD_PRECISION=fp64     -DTDMD_DETERMINISTIC_REDUCE=OFF  # validation reference
build-deterministic/  -DTDMD_PRECISION=fp64     -DTDMD_DETERMINISTIC_REDUCE=ON   # VerifyLab golden oracle
```

The deterministic build pairs with `fp64` rather than `mixed` on purpose:
if VerifyLab golden oracles used mixed precision, they would still have
a legitimate noise floor from float32 propagation, defeating the point
of strict equality.

## Consequences

- **Positive:**
  - Six enumerated sites, one compile-time flag, one new build target.
    The change is bounded and grep-verifiable.
  - VL-13 and VL-16 become writable as strict-equality assertions.
  - VL-4 noise envelope can be tightened by 1–2 orders of magnitude
    when both sides run under deterministic mode.
  - TD scheduler invariants become provable: any mismatch between
    expected and actual zone state in deterministic mode is, by
    definition, a scheduler bug.
- **Negative:**
  - Second code path for each of the six sites. Must be kept in sync.
    Mitigation: unit test `DeterministicReductionParity` asserts that
    production-mode and deterministic-mode totals agree within the
    documented noise floor on a fixed input — catches drift between
    the two paths early.
  - Third build directory in CI. Estimated CI wall-clock cost: +90
    seconds for the `build-deterministic/` configure+build, +60 seconds
    for the VerifyLab passes. Goes into `slow-suite` (nightly), not
    `fast-suite`.
- **Risks:**
  - **Intrinsic non-reproducibility despite our efforts.** CUDA
    intrinsics like `rsqrtf`, `expf`, `rcp.approx` may have different
    implementations across SM architectures. We test on RTX 5080
    (sm_120); if the ADR 0010 contract is asserted on a different GPU
    and silently breaks, the failure mode is confusing. **Mitigation:**
    the deterministic build runs a self-check at startup that computes
    a fixed reference reduction and compares to a hardcoded expected
    value for the target SM. Mismatch → warn loudly, do not assert the
    stronger guarantees.
  - **Cell-list ID sort cost may be worse than estimated on large
    systems** with dense cells. Mitigation: benchmark gate before
    accepting the implementation; if >3× on medium, fall back to a
    per-cell insertion-sort variant.
- **Reversibility:** high. The flag defaults OFF; removing the
  deterministic path is a `git revert` and one CMake variable cleanup.
  Nothing in production code takes a hard dependency on the flag.

## Follow-ups

- [ ] RD-3: implement deterministic cell placement (R2 fix) behind
      `TDMD_DETERMINISTIC_REDUCE`. Gate via unit test
      `DeterministicCellListParity`.
- [ ] RD-3: implement per-atom energy buffer + ordered reduction
      (R3–R6 fix). Gate via unit test `DeterministicEnergyReduction`.
- [ ] RD-4: add `build-deterministic/` CMake preset and CI slow-suite
      job. Self-check kernel at startup.
- [ ] RD-5: rewrite VL-13 as strict-equality assertion (remove
      envelope thresholds). Rewrite VL-16 to assert per-fault
      bit-diff at a specific atom.
- [ ] Tighten VL-4 `|dF|` threshold by 1–2 orders of magnitude once
      both `build-mixed/` and `build-fp64/` have deterministic variants
      available for the A/B comparison reference.
- [ ] Research item: what about MPI multi-rank determinism? This ADR
      deliberately leaves it out; revisit when ghost-only exchange
      lands (see ADR 0006).
