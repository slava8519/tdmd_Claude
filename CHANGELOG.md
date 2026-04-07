# Changelog

All notable changes to TDMD will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.0.1] - 2026-04-07

### Added
- Initial project scaffold (M0).
- CLAUDE.md with rules for AI agentic development.
- Full theory documentation (Time Decomposition method, zone state machine).
- Architecture overview, module catalogue, data structures.
- Roadmap M0–M8 with exit criteria.
- VerifyLab subsystem specification.
- Prompts and roles for Claude Code (implementer, architect, reviewer, test-engineer, physicist-validator, doc-writer).
- Workflows: start-new-milestone, add-feature, add-kernel, validate-against-lammps, fix-bug, investigate-performance, update-docs.
- Build system skeleton (CMake + Ninja), CI workflows.
- `scripts/build.sh`, `scripts/run-tests.sh`, `scripts/format.sh` working.
- Smoke test (`tests/unit/test_smoke.cpp`) passing.
- Core headers: `types.hpp`, `box.hpp`, `system_state.hpp`, `error.hpp`.
- Zone state machine header (`scheduler/zone.hpp`).
- Apache 2.0 license.

### Notes
- M0 complete. No working physics yet.
- Next: M1 (single-threaded CPU reference MD).
