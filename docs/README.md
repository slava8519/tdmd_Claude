# TDMD Documentation Index

> Start here if you're new to the project. Read in this order.

## 0. Vision

- [00-vision.md](00-vision.md) — one-page rationale: what TDMD is, why it exists, what success looks like.

## 1. Theory

The physics and the method.

- [01-theory/dissertation-source.md](01-theory/dissertation-source.md) — the original source of the TD method.
- [01-theory/time-decomposition.md](01-theory/time-decomposition.md) — **the core TD spec.** Read this before writing any scheduler code.
- [01-theory/zone-state-machine.md](01-theory/zone-state-machine.md) — formal state machine and invariants for the `Zone` type.
- [01-theory/integrator.md](01-theory/integrator.md) — velocity-Verlet derivation.
- [01-theory/potentials.md](01-theory/potentials.md) — Morse and EAM math.
- [01-theory/neighbor-lists.md](01-theory/neighbor-lists.md) — Verlet + cell-list theory.
- [01-theory/invariants.md](01-theory/invariants.md) — system-wide physics invariants TDMD must preserve.

## 2. Architecture

How TDMD is built.

- [02-architecture/overview.md](02-architecture/overview.md) — **layer cake and module map.** Read this before reading any individual module doc.
- [02-architecture/modules.md](02-architecture/modules.md) — per-module catalogue with dependencies.
- [02-architecture/data-structures.md](02-architecture/data-structures.md) — `SystemState`, `Zone`, `NeighborList`, `Box`.
- [02-architecture/scheduler.md](02-architecture/scheduler.md) — TD scheduler design, the brain of TDMD.
- [02-architecture/parallel-model.md](02-architecture/parallel-model.md) — 2D `time × space` parallelism, MPI topology.
- [02-architecture/gpu-strategy.md](02-architecture/gpu-strategy.md) — CUDA streams, mixed precision, kernel inventory.
- [02-architecture/lammps-compatibility.md](02-architecture/lammps-compatibility.md) — what we copy from LAMMPS, what we don't.
- [02-architecture/verifylab.md](02-architecture/verifylab.md) — how VerifyLab plugs into the architecture.

## 3. Roadmap

What we're building, in what order, with what exit criteria.

- [03-roadmap/milestones.md](03-roadmap/milestones.md) — **M0–M8 plan.** The single source of truth for "what's next."

## 4. Development

How to build, test, debug, contribute.

- [04-development/build-and-run.md](04-development/build-and-run.md) — building TDMD from source.
- [04-development/code-style.md](04-development/code-style.md) — C++/CUDA style guide.
- [04-development/testing-strategy.md](04-development/testing-strategy.md) — unit / integration / VerifyLab.
- [04-development/debugging.md](04-development/debugging.md) — Nsight, gdb, sanitizers, tips.
- [04-development/git-workflow.md](04-development/git-workflow.md) — branches, commits, PRs.
- [04-development/documentation-rules.md](04-development/documentation-rules.md) — keep docs in sync with code.

## 5. Benchmarks

What we measure and how.

- [05-benchmarks/README.md](05-benchmarks/README.md) — overview.
- [05-benchmarks/fcc16-spec.md](05-benchmarks/fcc16-spec.md) — FCC16 benchmark suite spec.
- [05-benchmarks/lammps-baseline.md](05-benchmarks/lammps-baseline.md) — how to obtain and version LAMMPS baselines.
- [05-benchmarks/metrics.md](05-benchmarks/metrics.md) — what numbers we track and why.

## 6. Decisions

Architectural decision records (ADRs). Lightweight, immutable, dated.

- [06-decisions/template.md](06-decisions/template.md) — copy this to start a new ADR.
- [06-decisions/0001-cpp20-cuda.md](06-decisions/0001-cpp20-cuda.md) — language and toolchain.
- [06-decisions/0002-standalone-primary.md](06-decisions/0002-standalone-primary.md) — standalone vs LAMMPS plugin.
- [06-decisions/0003-lammps-data-format.md](06-decisions/0003-lammps-data-format.md) — input/output compatibility.
- [06-decisions/0004-verifylab-separate.md](06-decisions/0004-verifylab-separate.md) — VerifyLab as separate subsystem.

## Adjacent docs

- [/CLAUDE.md](../CLAUDE.md) — master rules for Claude Code working on TDMD.
- [/prompts/README.md](../prompts/README.md) — role prompts and workflows.
- [/verifylab/README.md](../verifylab/README.md) — physics validation subsystem.
- [/CHANGELOG.md](../CHANGELOG.md) — release notes.
- [/CONTRIBUTING.md](../CONTRIBUTING.md) — how to contribute.
