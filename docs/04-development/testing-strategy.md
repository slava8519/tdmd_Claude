# Testing Strategy

> How TDMD knows it's correct.

## Three layers of tests

```
┌──────────────────────────────────────┐
│  VerifyLab     (physics, slow)        │  ← whole simulations vs LAMMPS / analytic
├──────────────────────────────────────┤
│  Integration   (multi-module, medium) │  ← small SystemState, real interactions
├──────────────────────────────────────┤
│  Unit          (one function, fast)   │  ← isolated, deterministic, < 30 s suite
└──────────────────────────────────────┘
```

Each layer answers a different question:

| Layer | Question | Speed | Where |
|---|---|---|---|
| Unit | Does this function do what I told it to? | ms | `tests/unit/` |
| Integration | Do these modules work together? | seconds | `tests/integration/` |
| VerifyLab | Is the physics correct? | minutes/hours | `verifylab/cases/` |

## Unit tests

- One file per module: `tests/unit/test_<module>.cpp`.
- Total suite < 30 seconds.
- Deterministic. Use fixed RNG seeds.
- Cover: happy path, error paths, at least one edge case.
- Run on every commit, both locally and in CI.

Test framework: in-tree harness at M0; **Catch2 v3** from M1 onward (added via `find_package` or `FetchContent`).

## Integration tests

- Small SystemState (a few hundred atoms).
- Real modules talking to each other through real interfaces.
- < 10 seconds per test.
- Run on every commit.

Use cases:
- Read a small data file, build neighbor list, compute forces, compare to reference.
- Run 100 steps of integration on a 64-atom box, check final positions.

## VerifyLab

See `verifylab/README.md` for the full spec. Summary:

- Whole simulations.
- Compared against LAMMPS, analytic solutions, or known statistical distributions.
- Run nightly (full) and on PRs (fast subset).
- **The most trusted tests in the project.**

## Coverage targets

We do not chase line coverage numbers. We track **behavior coverage**:

- Every public function has at least one test.
- Every error path has at least one test.
- Every state machine transition has a test (M3+, see `docs/01-theory/zone-state-machine.md`).
- Every CUDA kernel has a CPU reference and a numerical comparison test.
- Every potential has a VerifyLab case against LAMMPS.

## Determinism

Two test modes:

- **Default** (fast): allows non-deterministic FP order. Tests assert "approximately equal."
- **Deterministic** (`-DTDMD_DETERMINISTIC=ON`): bit-identical mode. Tests assert "exactly equal."

VerifyLab cases run in deterministic mode by default. Unit tests run in default mode.

## What good tests look like

Good test (clear, checks one thing, named for what it tests):

```cpp
TEST_CASE("ZoneStateMachine: Free can only transition to Receiving") {
  REQUIRE(is_legal_transition(ZoneState::Free, ZoneState::Receiving));
  REQUIRE_FALSE(is_legal_transition(ZoneState::Free, ZoneState::Computing));
  REQUIRE_FALSE(is_legal_transition(ZoneState::Free, ZoneState::Done));
}
```

Bad test (multiple things, vague name, no clear failure mode):

```cpp
TEST_CASE("Zones work") {
  Zone z;
  // ... 50 lines of mixed setup, calls, assertions ...
}
```

## Flaky tests are forbidden

A flaky test is worse than no test. If a test is flaky, **fix the flakiness immediately** before doing anything else. Do not retry; do not mark as "known to flake." Find the race condition, the uninitialized memory, the wall-clock dependency, and fix it.
