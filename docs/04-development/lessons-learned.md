# Lessons Learned — Phase 3 (ADR 0007 precision migration)

> This is the document a new contributor reads to understand **why** the hard
> rules in `CLAUDE.md` §4 exist. It is narrative, not prescriptive. If you
> want the rule, read `CLAUDE.md`. If you want to know what it cost to learn
> the rule, read this.

Phase 3 ran from late M5 through early M6. The goal was to implement the
mixed-precision contract laid out in ADR 0007 (positions and velocities in
double, forces in float, reductions in double) across the whole engine. On
paper it was a mechanical refactor. In practice it exposed four pitfalls —
three of them silent, one of them that cost real test failures — that are
now encoded as hard rules. Each pitfall is worth a story.

---

## Pitfall 1 — The Vec3 storage gap (months of invisible drift)

**What happened.** In Stage 0 of the migration (commit `332bcc4`) we
templatized `Vec3` as `Vec3T<T>` and added three role-typed aliases to
`src/core/types.hpp`:

```cpp
using PositionVec = Vec3T<pos_t>;    // always Vec3D
using VelocityVec = Vec3T<vel_t>;    // always Vec3D
using ForceVec    = Vec3T<force_t>;  // Vec3F in mixed, Vec3D in fp64
```

And ADR 0007 claimed the precision contract was "in place". But for several
sessions afterwards, every `DeviceBuffer` in the engine stayed typed as
`DeviceBuffer<Vec3>` — which, in mixed mode, means `DeviceBuffer<Vec3F>`,
i.e. **float** positions. The aliases existed; nothing used them.

The GPU integrator read positions as double (because it internally used
`pos_t`), did the drift in double, and then wrote back through a
`Vec3`-typed storage interface that silently truncated to float. The
truncation happened at every single timestep of every run. The NVE drift
test passed because the truncation error cancels out to first order on
short runs. But the long-run drift acceptance test (100k steps) flagged it.

Reading the ADR, we had been confident the migration was done. Reading the
code, it was not.

**What it cost.** Roughly four sessions of work spread across a week before
the gap was noticed. Two physical validation tests failed for reasons that
looked precision-related but were attributed to other causes (neighbor skin,
stream ordering). The actual fix — commit `4a718f8` — touched 57 files and
had to ship as one atomic commit because the type graph forms a dependency
cycle that doesn't permit partial states.

**The rule this produced.**
> *Type invariants are grep-verified, not document-verified.* (CLAUDE.md §4)

Any session that touches precision or storage types closes with
`grep '\bVec3\b' src/` and `grep 'DeviceBuffer<' src/`. The results go into
the session report. The ADR describes intent; the grep describes reality.
When they disagree, reality wins and the ADR is wrong.

---

## Pitfall 2 — The EAM density accumulator (silent 3-digit precision loss)

**What happened.** The EAM density kernel walks ~50 to 100 neighbors per
atom and sums up pairwise density contributions to produce a per-atom
electron density `rho_i`. In the first cut of the kernel, the accumulator
was declared as `real rho = 0` — which, in mixed mode, is `float`.

For a single neighbor pair, the precision is fine. For 100 neighbors at
alloy densities ~1.0, the Kahan-free float sum silently loses 3 to 4
significant digits. The derived EAM force involves `F'(rho)`, which amplifies
that error because `F'` is steep near equilibrium density. The force
kernel looked correct in isolation and matched LAMMPS to ~1e-4 for small
systems; it diverged to ~1e-2 on a 4000-atom FCC test.

**What it cost.** Two sessions of debugging the force kernel under a
suspicion of neighbor-list corruption, before re-reading the LAMMPS
reference (`lib/gpu/lal_eam.cu`) and noticing that LAMMPS accumulates in
`double rho` even in their single-precision mode.

**The rule this produced.**
> *Reductions always accumulate in `accum_t` (double), regardless of mode.* (CLAUDE.md §4)

KE, PE, virial, EAM density, any sum with more than ~10 terms — always
double. This is a hard rule without exceptions. The cost of a `double` on
a reduction that runs once per atom per step is unmeasurable; the cost of
a float precision loss on a force input is catastrophic and silent.

---

## Pitfall 3 — The default stream trap (synchronization you didn't ask for)

**What happened.** In an early version of the pipeline scheduler, one
branch of the code path called a kernel without specifying a stream:

```cpp
kernel<<<grid, block>>>(...);   // default stream
```

All the other branches used the stream pool:

```cpp
kernel<<<grid, block, 0, streams_.stream(sid)>>>(...);
```

CUDA's default stream synchronizes implicitly against every other stream
in the same context. So every call on the "default" branch was silently
serializing the whole pipeline. The scheduler claimed to be running 4
streams in parallel; `nsys` showed one stream in use, with the others idle.

The bug was invisible to correctness tests (the result was right, just
slow) and only surfaced when a performance investigation asked "why is
streams=4 no faster than streams=1?"

**What it cost.** Half a day of performance debugging. The fix was a
one-line change, but finding it required reading the nsys timeline
carefully because the CUDA API calls all looked identical in the log.

**The rule this produced.**
> *Никогда не запускай CUDA-kernel без явного stream в scheduler'е.*

This lives in the scheduler code style guide and in reviewer.md's
"things you almost always catch" list. The default stream is reserved for
one-shot host helpers (upload, download, initial nlist build) and is
never allowed in `tick()`, `launch_zone_step()`, or any inner loop.

---

## Pitfall 4 — Distance precision on consumer GPU (the 30× performance trap)

**What happened.** An early version of the Morse force kernel computed
pair distance in double:

```cpp
double dx = pos_i.x - pos_j.x;  // pos_t = double
double r2 = dx*dx + dy*dy + dz*dz;
if (r2 > rcut2) return;
```

On a data-center GPU (A100, H100) with 1:2 FP64:FP32 ratio, this costs
~2× for the distance compute and is tolerable. On RTX 5080 (Blackwell
consumer) with 1:32 ratio, it costs **30×** on those specific instructions.
A kernel that should have been bandwidth-bound became FP64-bound and ran
at ~3% of FP32 peak.

LAMMPS solves this with the relative-coordinate trick: one `double`
subtract to preserve precision across large absolute coordinates, then
an immediate cast to float for all downstream arithmetic. The neighbor
skin (typically 1.0 Å) absorbs the float-cast error because the cast
introduces less than 1e-5 Å of error on reasonable box sizes.

**What it cost.** A day of confusion reading `ncu` output before the
bottleneck was identified as FP64 throughput rather than memory latency.

**The rule this produced.**
> *Relative-coordinate trick for GPU distance compute.* (CLAUDE.md §4)

And a second, meta-rule worth remembering: **always check the FP64:FP32
ratio of the target GPU first.** A precision strategy that is fine on
H100 is broken on RTX 5080. This is a project-specific constraint that
should be re-checked when the target hardware changes.

---

## Meta-lesson — Documents lie, code does not

The thread running through all four pitfalls is the same: a pattern that
was true *at some point* in a document diverged silently from the code,
and nothing in the process flagged the divergence. ADR 0007 described a
precision contract that wasn't implemented. The scheduler doc described a
multi-stream design that the default-stream bug had disabled. The EAM
design note described density accumulation as "double-precision" without
specifying which variable.

The fix is not "write better documents". Documents drift. The fix is
**make verification cheap and mandatory**:

- For types: `grep` at session close.
- For kernel streams: `nsys` check on any scheduler change.
- For reductions: the scalar type of any `+=` target is visible in the
  diff; reviewers are trained to look for `real` on a loop accumulator.

These checks take seconds. They catch drift that took days to diagnose
after the fact. That asymmetry is why they are hard rules rather than
suggestions.

---

## How to add to this document

When you find a pitfall that cost more than a session to diagnose, and
the fix is a rule that should apply to future work, write it up here.
Format:

- **What happened** (the specific incident, with commits/files if possible).
- **What it cost** (in hours, or in failed tests, or in scope creep).
- **The rule this produced** (and where it lives: CLAUDE.md, an ADR, a
  role prompt, or a review checklist).

This file is not exhaustive — it's the cemetery for the traps that really
bit us. Trap avoidance through rules is the main deliverable; the stories
are the justification that keeps the rules from being deleted six months
later as "overkill".
