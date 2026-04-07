# Dissertation Source

> The original source for the Time Decomposition method that anchors TDMD.

## Reference

The Time Decomposition method as implemented in TDMD comes from a doctoral
dissertation in computational physics / molecular dynamics by V. V. Andreev,
under the supervision of A. M. Lipanov. The dissertation is the **single source
of truth** for the theoretical formulation of TD as we use it.

A PDF of the relevant chapters is in the project knowledge base under
`time_decomposition.docx`. A second derived planning document
(`Проект_системы_молекулярной_динамики_с_декомпозицией_по_времени...docx`)
contains the human author's earlier architectural notes for the same project.

## What we take from the dissertation

- The **definition** of Time Decomposition: regions separated by more than the
  interaction diameter cannot influence each other within one integration step,
  so they can be advanced at different time steps simultaneously.
- The **zone concept** as the unit of work, with constraints on minimum size
  (`≥ r_c`) and on the relationship between zone count and processor count
  (`P_opt ∝ N_zones / N_min_per_proc`).
- The **state-machine view** of zone lifecycle (Free → Receiving → ... → Sending → Free).
- The **ring topology** for inter-processor communication, with even/odd phase
  alternation to avoid deadlock.
- The **K parameter** for batching multiple steps before sending, which gives
  TD its bandwidth-scaling advantage `t_comm ∝ S / (K · P)`.
- The **comparison with RD/FD/SD** showing that TD has the lowest communication
  bandwidth and the lowest neighbor count per rank.
- The **boundary handling** alternatives (extra zones / Verlet skin / dynamic
  buffer width), of which TDMD uses Verlet skin + dynamic check.

## What we do NOT take literally

- **Hardware-specific tuning numbers** in the dissertation are from mid-2000s
  CPUs and clusters; we re-derive everything for modern GPUs.
- **Specific traversal orders for 3D zones** — the dissertation observes that
  naive sequential ordering is catastrophic in 3D and discusses alternatives;
  TDMD ships with naive linear order in M1 and moves to Hilbert/Morton curves
  as a post-M7 research item.
- **Code examples** in the dissertation are in older Fortran/C and serve as
  illustration only; we re-implement from the math, not from the listings.

## What we add on top of the dissertation

- **GPU acceleration** (CUDA kernels for force calculation) — not in the original.
- **2D `time × space` parallelism** for scaling beyond pure-TD's `P_opt`.
- **Modern testing infrastructure** (VerifyLab, LAMMPS A/B, sanitizers, CI).
- **EAM and ML potential support** (the dissertation focuses on pair potentials).

## How to resolve disagreements

If a question arises about "how should TDMD do X" and the dissertation has an answer,
**the dissertation wins**. If the dissertation is silent or ambiguous, we open an ADR
in `docs/06-decisions/` and decide consciously.

If you (Claude or human) think the dissertation is wrong about something, **do not
silently fix it**. Open an ADR explaining the discrepancy and the proposed alternative,
get human review, and only then implement.

## See also

- `docs/01-theory/time-decomposition.md` — the operational summary derived from this source.
- `docs/01-theory/zone-state-machine.md` — formal state machine.
