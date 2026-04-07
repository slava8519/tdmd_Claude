# 0002 — Standalone executable is the primary form, LAMMPS plugin is secondary

- **Status:** Accepted
- **Date:** 2026-04-07
- **Decider:** human + architect
- **Affected milestone(s):** M0–all

## Context

TDMD could be packaged in two ways:

1. As a standalone executable that reads its own (LAMMPS-compatible) input.
2. As a plugin loaded by LAMMPS (e.g. as a custom `fix` or `pair_style`).

Both are technically possible. We need to commit to one as the primary.

## Options considered

### Option A — Standalone primary, plugin secondary (chosen)
- Pros: full control over IO, scheduler, telemetry; clear separation from LAMMPS internals; easier to test and debug; we own the user experience end to end.
- Cons: more code to write (input parser, output writer, CLI).

### Option B — Plugin primary, standalone secondary
- Pros: instant access to LAMMPS's input parser, dump writer, fixes, computes, the entire LAMMPS ecosystem.
- Cons: TDMD's correctness becomes entangled with LAMMPS's data structures; the TD scheduler must coexist with LAMMPS's spatial decomp scheduler; debugging is much harder; no standalone product.

### Option C — Equal modes
- Pros: best of both.
- Cons: doubles the maintenance surface; both modes will rot.

## Decision

**Standalone executable is the primary form.** LAMMPS is used as:
- The validation oracle (VerifyLab runs LAMMPS as a subprocess for A/B comparison).
- A future debug bridge (M8+): a thin LAMMPS plugin that calls into TDMD's force/integrator for back-to-back testing inside a single LAMMPS run.

The plugin is **not a product** — it is a debugging tool that may or may not ship.

## Consequences

- **Positive:** clean architecture, controllable, testable, our own user experience, no LAMMPS-internal coupling.
- **Negative:** more boilerplate (LAMMPS data file reader, dump writer, CLI, config). This is acceptable.
- **Risks:** users will ask "why not just be a LAMMPS pair_style" — we must explain the answer in `docs/00-vision.md` and `docs/02-architecture/lammps-compatibility.md`.
- **Reversibility:** medium. We could later promote the plugin to primary, but it would be a significant rewrite of the IO layer.

## Follow-ups

- [ ] M1: implement minimal LAMMPS data-file reader.
- [ ] M1: implement LAMMPS-style dump output.
- [ ] M1: implement LAMMPS-style thermo output.
- [ ] M8+ (optional): explore the debug-bridge plugin.
