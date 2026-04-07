# 0004 — VerifyLab is a separate subsystem from `tests/`

- **Status:** Accepted
- **Date:** 2026-04-07
- **Decider:** human + physicist-validator
- **Affected milestone(s):** M1+

## Context

We need to test physical correctness, not just code correctness. These two kinds of tests have different shapes:

| Property | Unit tests | Physics tests |
|---|---|---|
| Speed | ms | minutes to hours |
| Determinism | exact | statistical |
| Oracle | the spec | LAMMPS, analytic, distribution |
| Frequency | every commit | nightly |
| Failure means | code bug | code bug **or** physics misunderstanding |

Mixing them in a single `tests/` folder either makes unit tests slow (bad) or makes physics tests rare (worse).

## Decision

**VerifyLab is its own top-level subsystem.** It has its own folder (`verifylab/`), its own runner (`verifylab/runners/run_all.py`), its own CI workflow (`verifylab.yml`), and its own conventions for cases (each case is a folder with input/, lammps/, reference/, check.py, tolerance.toml). See `verifylab/README.md`.

`tests/` stays focused on fast, deterministic, code-level checks.

## Consequences

- **Positive:** fast unit tests stay fast, physics tests get the attention they deserve, clear separation in failure interpretation.
- **Negative:** two test infrastructures to maintain.
- **Reversibility:** high.

## Follow-ups

- [ ] M1: first VerifyLab cases (two-atoms-morse, run0-force-match, nve-drift).
- [ ] CI: run fast suite on PRs, full suite nightly.
