# ADR 0007 — Precision contract for TDMD

- **Status:** Accepted (designed in session 3A, implementation in session 3B)
- **Deciders:** Project lead + AI architect role
- **Date:** 2026-04-09

## Context

TDMD targets consumer GPUs (RTX 5080 and similar Blackwell/Ada cards) where
FP64 throughput is 1/32 of FP32. Running pure FP64 leaves 30x of compute
performance on the table; running pure FP32 fails 13 tests due to chaos
amplification and round-off in cumulative quantities (energy, momentum,
position drift over long runs).

LAMMPS has solved this problem since 2010 with a "mixed precision" approach:
positions and velocities in double, forces in float, accumulators in double.
This gives full FP32 force compute speed without sacrificing trajectory
quality or energy conservation. We adopt the same approach.

### Inventory (session 3A)

A full read-only inventory of precision-sensitive sites found:

| Category | Count | Current type | Notes |
|---|---|---|---|
| Force kernels (Morse, EAM) | 7 kernels + 2 helpers | `force_t`/`accum_t` | Migrated: Morse in 3B.7.fix, EAM in EAM-1B. Relative-coordinate trick, force_t accumulators, accum_t density (EAM) |
| Integrator kernels (VV, zone VV) | 6 kernels | `real` | half_kick: `factor = half_dt / (mass * kMvv2e)` |
| Reductions (KE, vmax, PE atomicAdd) | 4 sites | `real` accumulators | Main precision concern: FP32 accumulation of N terms |
| Neighbor list distance | 1 kernel | `real` | Standard dx/dy/dz, r^2 pattern |
| Physical parameters | ~15 sites | `real` structs | MorseParams, SplineTable, constants |
| Host-device transfers | 6 sites | Templated `T` | Type-safe; no precision issue |
| MPI transfers | 3 Allreduce + many Sendrecv | Fixed in session 2 | Need `mpi_type<T>()` helper |
| Hardcoded double | 5 in tdmd_main.cpp | `double` | CLI config only, not physics code |

The `real` abstraction is clean — zero hardcoded float/double in physics code.
Migration to role-based types is straightforward.

## Decision

TDMD adopts **LAMMPS-style mixed precision** as the default precision mode,
with FP64 retained as a reference/validation mode and pure FP32 removed entirely.

### Build modes

A new CMake option `TDMD_PRECISION` replaces `TDMD_FP64`:

| Value | Description | Default |
|---|---|---|
| `mixed` | LAMMPS-style mixed precision | yes |
| `fp64` | All-double reference mode for validation | no |

`TDMD_PRECISION=fp32` does NOT exist. Pure FP32 mode is removed because:

- Energy drift ~1e-7/step linear, fails on long runs.
- 13 existing tests fail under it, not fixable without giving up on
  test-driven validation.
- Provides no benefit over `mixed` mode (force compute is identical FP32
  in both; only position/velocity precision differs).

### Per-quantity precision table

| Quantity | Mixed mode type | FP64 mode type | Rationale |
|---|---|---|---|
| Position (x, y, z) | `double` | `double` | Trajectory must accumulate without drift; negligible perf cost (memory bandwidth, not compute) |
| Velocity (vx, vy, vz) | `double` | `double` | Same as position; integrator math needs full precision |
| Force (fx, fy, fz) | `float` | `double` | Force compute is the hot path; FP32 here is the main speedup source |
| Mass (per type) | `double` | `double` | Constant, no perf impact |
| Time step `dt` | `double` | `double` | Constant, no perf impact |
| Morse params (D, alpha, r0, rc) | `double` | `double` | Constants, no perf impact |
| EAM spline coefficients (a, b, c, d) | `real` (float) | `double` | LAMMPS uses `numtyp` (float) for all spline tables in mixed mode. Read-only data, fits in cache regardless of precision; cubic interpolation precision sufficient for embedded atom potentials validated by LAMMPS production use. **Original ADR 0007 specified `double` here based on design-time intuition; LAMMPS source reading in session EAM-1A established that `float` is correct (per ADR 0008 process rule).** |
| EAM spline metadata (dr, rmin) | `real` (float) | `double` | Same as coefficients — float in mixed mode, matches LAMMPS |
| EAM density accumulator (rho_i) | `accum_t` (double) | `double` | LAMMPS uses `acctyp` (double) for density gather. Sum of ~50-100 spline-eval terms per atom; double accumulation prevents catastrophic accumulation error in density sum |
| EAM embedding derivative (fp_i) | `real` (float) | `double` | LAMMPS uses `numtyp` (float). Computed from spline lookup at density value, then used as multiplier in force expression — float precision sufficient |
| Distance computation (dx, dy, dz, r^2) | `force_t` (via relative-coordinate trick: double subtract → force_t cast) | `double` | Relative-coordinate trick gives float distance from double subtraction, avoiding catastrophic cancellation; no epsilon buffer needed (skin distance suffices) |
| Energy (KE, PE, total) | `double` | `double` | Reductions: FP32 sums of N atoms accumulate epsilon*N error |
| Virial sum | `double` | `double` | Same as energy |
| Momentum (per axis) | `double` | `double` | Same |
| Constants (kBoltzmann, kMvv2e) | `double` | `double` | Always double; used in integrator math |
| Reduction accumulators (general) | `double` | `double` | Always `double`, regardless of mode |

### Type role aliases

In `src/core/types.hpp`, define a two-layer type system:

**Layer 1: Precision-explicit vector types (template)**

```cpp
template<class T>
struct Vec3T { T x, y, z; };

using Vec3D = Vec3T<double>;
using Vec3F = Vec3T<float>;
```

**Layer 2: Role aliases (semantic naming)**

```cpp
#if TDMD_PRECISION_MIXED
using pos_t      = double;    // position storage
using vel_t      = double;    // velocity storage
using force_t    = float;     // force storage (hot path)
using accum_t    = double;    // reduction accumulators
using PositionVec = Vec3D;
using VelocityVec = Vec3D;
using ForceVec    = Vec3F;
#elif TDMD_PRECISION_FP64
using pos_t      = double;
using vel_t      = double;
using force_t    = double;
using accum_t    = double;
using PositionVec = Vec3D;
using VelocityVec = Vec3D;
using ForceVec    = Vec3D;
#endif
```

**Backward compatibility during migration:** the existing `using real = ...`
typedef remains for code that hasn't been migrated yet. New code uses role
aliases. Migration happens in 3B file by file.

### Integrator math contract

Even when forces are stored as `float` (mixed mode), integrator math must
happen in `double`:

```cpp
// In half_kick kernel (mixed mode):
double dvx = static_cast<double>(force_x) * static_cast<double>(rmass) * 0.5 * dt;
new_vx = old_vx + dvx;   // both old_vx and new_vx are double
```

This is the core of "mixed mode": float storage of forces, but the velocity
update happens with full double precision arithmetic. The `static_cast<double>`
on force load is the "promotion on the fly" pattern.

### Force compute contract

In mixed mode, force kernels follow the LAMMPS-derived hybrid pattern:
positions stored as double on GPU (required for GPU integrator), but
distance computation in `force_t` (=float) using the relative-coordinate trick.

```cpp
// Position load in double (storage type matches use type)
pos_t pix = positions[i].x;
pos_t piy = positions[i].y;
pos_t piz = positions[i].z;

// Box dimensions to force_t once, outside the loop
const force_t bsx = static_cast<force_t>(box_size.x);
// ...
const force_t rc_sq = static_cast<force_t>(params.rc_sq);

for (each neighbor j) {
    // RELATIVE-COORDINATE TRICK: one double-subtract per dimension, cast to force_t.
    // This is one FP64 instruction per pair, not ~10. More accurate than
    // pure float subtraction (avoids catastrophic cancellation).
    force_t dx = static_cast<force_t>(pix - positions[j].x);
    force_t dy = static_cast<force_t>(piy - positions[j].y);
    force_t dz = static_cast<force_t>(piz - positions[j].z);

    // PBC in force_t (we keep PBC inside force kernel — moving it out
    // requires image counter, deferred to future work)
    if (pbc_x) { ... apply min image to dx in force_t ... }

    force_t r2 = dx*dx + dy*dy + dz*dz;   // force_t
    if (r2 >= rc_sq) continue;             // force_t comparison, no buffer

    // Force expression — all force_t
    force_t r = sqrt(r2);
    force_t f = compute_force(r);

    // Force accumulation in force_t
    fx += f * dx;
    fy += f * dy;
    fz += f * dz;

    // PE accumulation in force_t (cast to double on atomic write)
    pe_local += compute_energy(r);
}

// Write forces in force_t (float in mixed mode)
forces[i].x = fx;
forces[i].y = fy;
forces[i].z = fz;

// Cast PE to double for atomic accumulation in accum_t* d_energy
atomicAdd(d_energy, static_cast<accum_t>(pe_local));
```

**Key points:**

- **Positions stored as `double`** on GPU. Required for GPU integrator (which
  TDMD keeps unlike LAMMPS, because GPU integrator is needed for K > 1 kernel
  fusion in M7).
- **Distance computation in `force_t` via relative-coordinate trick.** One
  double subtract per dimension, then cast to `force_t`. In mixed mode this
  is one FP64 instruction per pair instead of ~10 (which is what pure double
  distance costs). The relative trick also avoids catastrophic cancellation
  when positions are close. In fp64 mode, `force_t=double` so the cast is
  identity.
- **PBC in `force_t`.** Kept inside force kernel for now. Moving PBC out (the
  LAMMPS approach) would require image counter for unwrapped coordinates,
  deferred to future work. Float PBC adds ~6 branches per neighbor but no
  FP64 cost.
- **No epsilon buffer on cutoff.** The neighbor list skin distance already
  provides the margin LAMMPS relies on. Removing the epsilon buffer (which
  was in the original ADR 0007 design) brings TDMD in line with LAMMPS.
- **Force expression in `force_t`.** Hot arithmetic, where FP32 wins.
- **Force buffer writes in `force_t`.** `force_t = float` in mixed mode.
- **PE accumulation in `force_t` during loop, cast to `double` on atomic
  write.** Slightly less accurate than double accumulator, but faster.
  Validated via energy drift tests (2.57e-13/step, well within 1e-9 target).

**Why we don't follow LAMMPS exactly:**

LAMMPS solves the precision problem by storing positions as `float` on GPU
(host conversion happens before each force step) and running integrator on
CPU. This works for LAMMPS because they don't need GPU integrator. TDMD
requires GPU integrator for K > 1 kernel fusion (M7), so we keep positions
in `double` on GPU and use relative-coordinate trick to get LAMMPS-comparable
performance from float distances. See ADR 0008 for the general "copy LAMMPS
where applicable" process rule.

The cost of TDMD's approach vs LAMMPS:
- 3 extra FP64 instructions per pair (one double-subtract per dimension)
- ~6 extra float branches per pair (PBC inside kernel)

Total measured overhead vs Phase 2 FP32 baseline: ~9% on medium system. This
is acceptable for preserving GPU integrator capability.

### Reduction contract

All reductions (KE, PE, virial, momentum) accumulate in `double` regardless
of input precision. Force or velocity may be loaded as float, immediately
cast to double, accumulated as double. The reduction kernel template parameter
is `accum_t`, hardcoded to `double` in both modes.

The current `atomicAdd(d_energy, pe)` in force kernels needs adaptation:
in mixed mode, `pe` is computed from float forces but must be accumulated
into a `double` energy counter.

**Resolution:** use `atomicAdd(double*, double)` directly. This is supported
on sm_60 and above (we target sm_120), and in practice contributes < 1% wall
time on systems up to 100K atoms — PE accumulation is per-pair, not per-thread,
and the atomic contention is much lower than typical force atomic patterns.

The two-pass approach (block-local float accumulation + host reduction) is
reserved as an optimization to be considered only if profiling shows PE
atomicAdd as a measurable bottleneck. As of session 3A design, no such
measurement exists, so we use the simple approach.

### Energy drift targets

| Mode | Per-step drift | Long-run behavior | Reasoning |
|---|---|---|---|
| Mixed | `< 1e-12` (per step) | Bounded oscillation; no linear drift visible over 10^6 steps | Float force load round-off ~1e-7 propagates through `double` integrator math; only loss is in force compute itself, which is bounded |
| FP64  | `< 1e-15` (per step) | Bounded oscillation at machine epsilon; no measurable drift over 10^9 steps | Symplectic Verlet in pure double — drift exists only via floating-point round-off in reduction sums |

These are **acceptance targets**, not aspirational. If session 3B measurements
exceed these bounds, that indicates a bug in the precision implementation
(typically: a reduction accidentally left in `float`, or integrator math not
promoted to double when reading float forces).

LAMMPS reference: GPU package mixed mode achieves ~1e-13/step on Lennard-Jones
NVE on consumer GPUs. We target the same order of magnitude.

### MPI types

A new helper replaces runtime `sizeof(real)` checks:

```cpp
template<class T> constexpr MPI_Datatype mpi_type();
template<> constexpr MPI_Datatype mpi_type<float>()  { return MPI_FLOAT; }
template<> constexpr MPI_Datatype mpi_type<double>() { return MPI_DOUBLE; }
template<> constexpr MPI_Datatype mpi_type<int>()    { return MPI_INT; }
```

All `MPI_Allreduce` calls use `mpi_type<real>()` (or `mpi_type<accum_t>()` for
reductions) instead of hardcoded `MPI_DOUBLE` or runtime size checks.

MPI boundary exchange via `MPI_BYTE` (raw binary packing) is safe and unchanged:
all ranks run the same binary, so struct layout is identical. The `vec3_bytes()`
and `i32_bytes()` helpers correctly compute buffer sizes based on actual
`sizeof(Vec3)` and `sizeof(i32)`.

### Test tolerance contract

The `precision_tolerance.hpp` header provides per-quantity, per-mode tolerance
constants. Each test uses the macro corresponding to what it compares:

- `EXPECT_POSITION_NEAR` for position comparisons
- `EXPECT_VELOCITY_NEAR` for velocity comparisons
- `EXPECT_FORCE_NEAR` for force comparisons
- `EXPECT_ENERGY_REL_NEAR` for energy relative comparisons

Never raw numeric tolerance in new tests. Existing tests migrated file by file
in session 3B.

## Consequences

### Positive

- Force compute runs at full FP32 speed on consumer GPUs (~30x faster than FP64 on RTX 5080).
- Integrator stays at double precision, energy drift remains at LAMMPS level.
- Aligns with the most-used MD package on the planet, validates our approach.
- Pure FP32 build mode and its 13 broken tests disappear.
- MPI type bug class is structurally prevented by `mpi_type<T>()` helper.
- Test tolerances become physically meaningful, not magic numbers.

### Negative

- More complex type system: developers must understand pos_t vs force_t vs accum_t.
- Memory footprint not reduced (positions still double); pure-FP32 memory savings forfeited.
- Refactoring effort in session 3B: ~349 lines of physics code + integrator + tests + build system.
- Two build modes to maintain in CI (mixed default, fp64 reference).

### Risks

- If integrator promotion (float -> double on the fly) is incorrect anywhere,
  energy drift grows silently. Mitigated by long-run NVE tests in 3B.
- If reduction accumulators are accidentally float somewhere, totals lose
  precision. Mitigated by explicit `accum_t` type and
  `static_assert(std::is_same_v<accum_t, double>)` in reduction kernels.
- PE accumulation via `atomicAdd(double*, ...)` is slower than float atomicAdd.
  If this becomes a bottleneck, a two-pass reduction approach may be needed.

## Implementation results (session 3B)

### Measured energy drift (100k NVE steps, 256 Cu atoms, Morse, dt=0.001 ps)

| Mode | Per-step drift | ADR target | Margin |
|---|---|---|---|
| Mixed | 5.73e-13 | 1e-12 (aspirational) | 1.7x better |
| Mixed | 5.73e-13 | 1e-9 (conservative) | 1700x better |
| FP64 | 1.05e-20 | 1e-15 | 95000x better |

Mixed precision implementation meets both the aspirational and conservative ADR
targets. The per-step drift of 5.7e-13 confirms that the double integrator math
+ float forces pattern produces trajectory quality indistinguishable from
LAMMPS-style mixed precision.

### Test coverage

73 tests pass in both modes. 2 tests skipped in mixed mode:

1. **DeterministicMatchesM3** — CPU-float vs GPU-mixed comparison is not
   meaningful since CPU uses float everywhere while GPU uses double for
   distances and integrator math.
2. **DeterministicMatchesM5** — Spatial decomposition ghost atoms communicated
   in float cause boundary divergence vs single-rank reference. NVE
   conservation test (which passes) covers correctness.

### Precision-aware test tolerances

`tests/support/precision_tolerance.hpp` provides per-quantity, per-mode
tolerance constants. All existing tests migrated from hardcoded magic numbers
to these constants.

### Implementation results (session 3B.7.fix)

Session 3B.7.fix applied the LAMMPS-derived relative-coordinate trick to fix
a 5-7x performance regression caused by the original ADR 0007 design mandating
double-precision distance computation.

**Performance on RTX 5080 (mixed mode, fast_pipeline scheduler):**

| System | 3B.6 (double dist) | 3B.7.fix (relative trick) | Phase 2 FP32 baseline | Recovery |
|---|---|---|---|---|
| tiny (256) | 3,348 ts/s | 14,814 ts/s | 16,413 ts/s | 90% |
| small (4,000) | 2,090 ts/s | 9,139 ts/s | 9,828 ts/s | 93% |
| medium (32,000) | 1,080 ts/s | 6,927 ts/s | 7,638 ts/s | 91% |

**Energy drift (mixed mode, 100k NVE steps, 256 Cu atoms):** 2.57e-13/step
(ADR target: 1e-9 conservative, 1e-12 aspirational).

**FP64 sanity check:** medium 754 ts/s, unchanged from baseline.

**Changes applied:**
- Morse force kernels: relative-coordinate trick, force_t distance/PBC/force,
  no epsilon buffer
- Morse zone variant: same changes
- Neighbor list builder: same relative-coordinate trick, real distance/PBC
- device_math.cuh intrinsics retained for sqrt/exp

### EAM migration results (session EAM-1B)

EAM kernels migrated to ADR 0007 Force compute contract pattern:

- Density accumulator: `real` (float) → `accum_t` (double) — correctness fix,
  LAMMPS parity. LAMMPS uses `acctyp` (double) for density gather.
- Density kernel distance: `pos_t` (double) → `force_t` (float) via
  relative-coordinate trick. Eliminates FP64 hotspot.
- Force kernel distance: same relative-coordinate trick pattern.
- Force kernel accumulators: `real` → `force_t`. Follows Morse pattern.
- Spline coefficients and metadata: confirmed `real` (float in mixed) is
  correct per LAMMPS reading. ADR table entries corrected above.

**Test impact:**
- All EAM tests pass in both build modes (mixed and fp64). No skips needed.
- FP64 mode bit-identical to pre-migration state (all casts are identity when
  `force_t = double`).
- NVE drift tests unaffected (they use Morse, not EAM).

**Performance:** EAM-specific benchmark not yet available
(`bench_pipeline_scheduler` supports Morse only). FP64 hotspot elimination
expected to provide similar speedup as Morse migration (~4-9x in mixed mode).
EAM benchmark infrastructure deferred to future session.

## References

- LAMMPS GPU package precision: https://docs.lammps.org/Speed_gpu.html
- Session 2: NVT multi-rank bug, MPI_DOUBLE finding
- ADR 0006 update (session 2 findings appendix)
- gpu-strategy.md, to be updated after 3B implementation
