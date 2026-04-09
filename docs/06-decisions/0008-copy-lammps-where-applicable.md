# ADR 0008 — Copy LAMMPS where applicable

- **Status:** Accepted
- **Deciders:** Project lead + architect
- **Date:** 2026-04-09
- **Supersedes:** none (process rule, not technical)
- **Related:** ADR 0007 (precision contract), and any future ADR involving
  general MD techniques

## Context

In sessions 1 through 3B of the Phase 3 series, TDMD made a number of
architectural decisions that turned out to be wrong because we did not check
how LAMMPS solves the same problem. Examples:

1. **Default CUDA stream trap** (Phase 2). The first attempt at FastPipelineScheduler
   used legacy default stream for compute, expecting "natural serialization".
   Performance was 40% of target. LAMMPS uses explicit non-default streams from
   day one, for exactly this reason — they hit it years ago and documented it.
   We did not check; we lost a session figuring it out.

2. **Distance precision in force kernel** (session 3B). ADR 0007 mandated
   double-precision distance computation for "deterministic cutoff check".
   On consumer GPU with 1:32 FP64:FP32 ratio, this gave a 5-7x performance
   regression. LAMMPS uses pure float distance computation with the relative-
   coordinate trick, no epsilon buffer, no double-precision distance anywhere.
   We did not check; we lost another session.

3. **PBC inside force kernel** (session 3B). Our force kernels do PBC
   correction per neighbor pair. LAMMPS wraps positions on the host before
   sending them to GPU, so PBC is zero cost in force kernel. We did not check.

These mistakes share a single root cause: **we treated solved problems as
unsolved problems.** Each of these is a general GPU MD challenge that LAMMPS
has spent over a decade refining. Reinventing them takes us hours-to-days per
issue, with predictable failure modes.

## Decision

TDMD adopts the following process rule for all future architectural work:

### Rule: classify the task before designing

Before designing any architectural change, classify the task as either:

- **TD-specific**, where TDMD must invent because no other code has the same
  problem
- **General MD problem**, where TDMD must copy LAMMPS

### Examples of TD-specific tasks (TDMD invents)

- Time Decomposition scheduler architecture
- Zone state machine
- Multi-step kernel fusion (K > 1 batching)
- Dependency DAG between zones at different time steps
- Pipelined execution overlapping force compute and time integration
- Multi-rank time decomposition (when we get there)
- Adaptive K scheduling

### Examples of general MD tasks (TDMD copies LAMMPS)

- Mixed precision strategy (which types where, why, with what trade-offs)
- Force kernel internal layout (loop structure, data access patterns,
  precision boundaries)
- Integrator implementation details (Velocity-Verlet, NHC chain, integrator
  numerical recipes)
- Neighbor list algorithms (cell-based, skin distance, rebuild triggers)
- Data file formats (LAMMPS data format already adopted via ADR 0003)
- MPI patterns (datatype dispatch, ghost atom layout, sendrecv ordering)
- CMake patterns for CUDA + MPI projects
- Test data generation for benchmarks

### Rule: process before designing general MD task

When tackling a general MD task, the procedure is:

1. **Identify** the specific LAMMPS file or files most relevant to the task.
   Usually under `lib/gpu/` for GPU code, `src/` for CPU. Common targets:
   - `lib/gpu/lal_precision.h` — precision modes
   - `lib/gpu/lal_atom.{h,cpp}` — atom data layout
   - `lib/gpu/lal_pair.cpp` — common pair pattern
   - `lib/gpu/lal_<potential>.{h,cu}` — specific potential implementation
   - `src/integrate*` — integrator implementations
   - `src/neighbor*` — neighbor list management

2. **Clone** LAMMPS read-only:
   ```bash
   git clone --depth 1 --branch stable https://github.com/lammps/lammps.git /tmp/lammps-readonly
   ```

3. **Read** the relevant files. Take notes. Identify exact precision types,
   exact loop structures, exact data flow.

4. **Document** what LAMMPS does in the architectural design (ADR or design
   document). Include direct code excerpts where helpful.

5. **Decide** which parts to copy verbatim, which to adapt, which to change
   because of TD-specific requirements.

6. **For each "change because of TD requirement"**, write the explicit
   justification in the ADR. Future readers must understand why TDMD diverges
   from LAMMPS at this specific point.

### Rule: limits of LAMMPS authority

LAMMPS is the **default reference** for general MD tasks, but not the **only**
reference. Three exceptions:

1. **HOOMD-blue** is sometimes a better reference for GPU-native architecture
   (LAMMPS GPU is bolted on; HOOMD GPU is built in). Consult HOOMD when GPU
   architecture matters more than physics.

2. **GROMACS** is the reference for biomolecular force fields, not LAMMPS.
   Not relevant to TDMD's metals scope, but worth noting.

3. **When LAMMPS solution is suboptimal for our hardware target** (e.g., LAMMPS
   developed primarily for data center GPUs with 1:2 FP64:FP32 ratio; we
   target consumer GPUs with 1:32). In this case: copy LAMMPS approach
   structurally, then add a TDMD-specific adaptation for the hardware difference.
   Document the adaptation explicitly in the ADR.

### Rule: when to deliberately diverge

TDMD diverges from LAMMPS when:

1. **Time Decomposition requires it.** Example: GPU integrator (we keep,
   LAMMPS uses CPU), because GPU integrator is required for K > 1 kernel
   fusion in M7.

2. **Hardware target differs.** Example: more aggressive float-precision in
   force compute on consumer GPUs, where LAMMPS's data-center-optimized
   trade-offs don't apply.

3. **TDMD's architecture allows a cleaner solution.** Rare. Should be flagged
   in ADR explicitly with measurement-backed justification.

## Consequences

### Positive

- Future architectural work gets correct answers in 30-60 minutes (read LAMMPS)
  instead of days (invent + measure + fix + iterate).
- TDMD inherits 30 years of LAMMPS battle-testing for general MD problems.
- Scope of "what TDMD must invent" becomes much smaller and clearer — only
  TD-specific work, where we **should** be inventing.
- Reduces cognitive load on developers and AI agents working on TDMD.
- Reduces the risk of hidden bugs from non-standard precision/layout/algorithm
  choices.

### Negative

- TDMD becomes architecturally similar to LAMMPS in places where they
  overlap. This is not a bug — it is the intended outcome.
- Developers must be willing to read LAMMPS source code, which is large and
  not always self-explanatory. Mitigated by AI agents being good at this kind
  of read-and-summarize work.

### Risks

- **Risk:** LAMMPS is a moving target. Their architecture evolves. Ours may
  diverge over time as theirs improves and ours stays static.
  - **Mitigation:** every few months, designate a "LAMMPS sync session" where
    we re-read relevant LAMMPS files and check whether their approach has
    evolved meaningfully. Apply useful changes.

- **Risk:** Over-coupling to LAMMPS. If a major LAMMPS architectural decision
  turns out to be wrong, we inherit the mistake.
  - **Mitigation:** copy with understanding, not blindly. Each copied
    decision should have a documented rationale, so we can re-evaluate later.

## How this rule was discovered

This rule comes from session 3B's experience: we lost approximately 4-6 hours
of agent work to architectural mistakes that could have been avoided by
spending 30-60 minutes reading LAMMPS source first. The cost of "not reading"
is not just the wrong design — it includes the effort to discover the design
is wrong, the rework, and the cognitive load of doubting whether current
working implementation is also wrong.

The break-even is sharply in favor of "read first": ~10 minutes spent reading
LAMMPS is rarely wasted, while ~10 minutes spent inventing a wrong design
costs 10+ hours to fix.

## Related ADRs

- ADR 0003 — LAMMPS data format adoption (already a copy from LAMMPS, this
  rule formalizes the pattern)
- ADR 0007 — Precision contract (currently being revised because we did
  not follow this rule on first attempt)
- All future ADRs touching general MD techniques should reference this rule
  in their context section
