# 0009 — Role-typed Vec3 aliases (PositionVec / VelocityVec / ForceVec)

- **Status:** Accepted
- **Date:** 2026-04-10
- **Decider:** human + architect + implementer (Claude Opus 4.6)
- **Affected milestone(s):** M2/M5 hardening (precision contract follow-up)

## Context

ADR 0007 fixed a precision contract:

- positions and velocities in `double` (geometry / integrator state),
- forces in `float` in mixed mode and `double` in fp64 mode,
- reduction accumulators always `double`.

Stage 0 (commit `332bcc4`) templatized `Vec3` as `Vec3T<T>` and made math
operators work for both `Vec3D` and `Vec3F`. Stage 1 (commit `ac64079`)
moved the `Box` to `Vec3D` storage. After those two stages every Vec3
parameter in the engine still went through a single `Vec3 = Vec3T<real>`
typedef where `real = float` in mixed mode. That left a real bug in the
hot path: the GPU integrator read `double` positions, did the drift in
`double`, and then wrote them back through a `Vec3` (= `Vec3T<float>`)
storage interface. The truncation killed the very precision that ADR 0007
was supposed to deliver.

The same problem appeared in every kernel that touched positions or
velocities (Morse, EAM, Verlet, NHC, neighbor list, cell list), in every
host scheduler that owned device buffers, and in the MPI wire format.

We needed a way to make the precision role of every Vec3 *visible at the
type level* so the compiler enforces "positions are double, forces are
mixed" instead of leaving it as a per-kernel discipline.

## Options considered

### Option A — Keep one `Vec3` typedef, fix sites case-by-case

- Pros: minimal code churn; "just one type to think about".
- Cons: this is what we already had. The truncation bug existed precisely
  because nothing in the type system distinguished a force from a position.
  The fix would have to be re-applied every time someone added a kernel.

### Option B — Per-call template parameters

- Make every kernel `template<class P, class F>` and let the caller pick.
- Pros: maximum flexibility.
- Cons: explodes compile times, inflates the API surface, and the right
  answer is *always the same trio* (PositionVec/VelocityVec/ForceVec). The
  flexibility is fictional.

### Option C — Three role-typed aliases over `Vec3T<T>`

- `PositionVec = Vec3T<pos_t>` where `pos_t = double` always.
- `VelocityVec = Vec3T<vel_t>` where `vel_t = double` always.
- `ForceVec = Vec3T<force_t>` where `force_t = float` (mixed) / `double` (fp64).
- Apply the aliases to *storage and signatures*, not to local variables —
  local arithmetic still uses the underlying `pos_t / accum_t / force_t`
  scalar typedefs.

- Pros: makes the precision role part of the type so the compiler refuses
  to silently truncate; keeps the underlying `Vec3T<T>` template; mode
  switching (mixed ↔ fp64) is a single rebuild flag flipping `force_t`.
- Cons: every API touching atom data has to be updated once. MPI wire
  format changes because `sizeof(PositionVec) + sizeof(VelocityVec)`
  is no longer `2 * sizeof(Vec3)` in mixed mode.

## Decision

We chose **Option C**. The role aliases live in `src/core/types.hpp`
alongside the precision typedefs from ADR 0007. The legacy `Vec3` alias
is retained as `Vec3T<real>` for backwards compatibility but is no longer
used inside the engine — every storage site and every public/internal
signature now uses one of the three role aliases. Local variables in
kernels continue to use the scalar typedefs (`pos_t`, `accum_t`, `force_t`)
to express the intended precision of intermediate arithmetic.

The migration was carried out in five stages, with each stage commitable
and testable on its own:

| Stage | Commit | Scope |
|-------|--------|-------|
| 0 | `332bcc4` | Template `Vec3T<T>`, templatized math ops |
| 1 | `ac64079` | Box → Vec3D storage |
| 2 | `4a718f8` | Host SystemState → typed vectors |
| 3 | `4a718f8` | DeviceSystemState + GPU integrator (kills the truncation) |
| 4 | `4a718f8` | Force, neighbor, cell kernels migrated |
| 5 | this ADR + verification | Schedulers + MPI wire format + docs |

Stages 2/3/4 ended up entangled because the host SystemState, the device
buffers, and the kernel signatures form a single dependency cycle — none
of them can compile in isolation once the type changes — so they ship as
one commit.

## Consequences

### Positive

- **The truncation bug is gone**: GPU integrator drift in mixed mode now
  preserves double-precision positions through the writeback. ADR 0007
  acceptance test (`DeviceNVEDrift.ADR0007AcceptanceTest100k`) passes.
- **Compiler-enforced discipline**: a future kernel that tries to write
  a float into a `PositionVec*` won't compile. The role of every buffer
  is visible at the function signature.
- **Mixed/fp64 mode switching is a single rebuild**: nothing in the
  engine knows whether `ForceVec` is float or double; the
  `TDMD_PRECISION_*` flag controls it from one place in `types.hpp`.
- **Performance contract delivered**:
  - small (4k atoms):  mixed = 5992 steps/s, fp64 = 1487 steps/s — **4.0× speedup**.
  - medium (32k atoms): mixed = 4025 steps/s, fp64 =  755 steps/s — **5.3× speedup**.
  - medium vs LAMMPS-GPU: TDMD-mixed (4025) is **1.26× faster** than the LAMMPS reference (3207 steps/s).

### Negative

- One large mechanical commit touching 57 files. We deliberately did the
  whole engine in one shot rather than feature-flagging because the type
  graph is interconnected and partial states don't compile.
- MPI wire format changes silently: any rank running an old binary will
  produce malformed messages on the wire. There is no version negotiation.
  Mitigation: Phase-1 distributed runs are bounded to single sessions, so
  there are no persisted snapshots. Documented as a known incompatibility.
- Tests had to be updated to thread the typed aliases through. 15 of 17
  test files in `tests/unit/` were touched (test logic unchanged).

### Risks

- **fp64 mode regression**: someone could accidentally write a `Vec3F`
  literal into a `ForceVec` and it would silently break in fp64 mode.
  Mitigated by running both build modes (`build-mixed` and `build-fp64`)
  in CI; the migration commit has both passing 73/73.
- **Wire-format size mismatch on heterogeneous-mode rollouts**: don't
  mix mixed-mode and fp64-mode binaries in the same MPI run. Out of scope
  for now (single-build deployments only).

### Reversibility

Easy in principle (the aliases are typedefs); hard in practice because
57 files would have to be undone. We don't expect to ever revert this.

## Follow-ups

- [x] All 73 unit tests pass in `build-mixed`.
- [x] All 73 unit tests pass in `build-fp64`.
- [x] LAMMPS A/B benchmark (small + medium) recorded.
- [ ] Update CHANGELOG with the precision migration entry.
- [ ] Update `docs/03_roadmap/current_milestone.md` with stage closure.
- [ ] Phase-2 sweep: re-run nsys traces to confirm the force kernel is now
      taking the FP32 issue path on Blackwell SMs.
