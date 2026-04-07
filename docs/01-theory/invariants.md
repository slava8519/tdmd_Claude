# System Invariants

> Properties that must hold throughout a TDMD simulation. Violations indicate bugs.

## What an invariant is

An invariant is a statement that is **always true** at certain checkpoints in the program. For TDMD we identify invariants at:

- After force computation.
- After each integration step.
- After each communication round.
- At the end of each VerifyLab case.

Each invariant must be checkable, either automatically (in code, in assertions, in tests) or in post-processing.

## Code-level invariants

### CI-1 — All atoms inside the box (with PBC applied)
After every integration step, every atom is inside `box.lo ≤ r ≤ box.hi` (modulo periodic wrap). Wraps must be applied at end of every drift.

### CI-2 — Force array is initialized
Forces are zeroed before computation. After computation, no NaN or Inf in any force component.

### CI-3 — Neighbor list is fresh
The neighbor list `built_at_step` ≤ current step, and the skin invariant holds (no atom moved > `r_skin/2` since `built_at_step`).

### CI-4 — IDs are unique and stable
`SystemState.ids` contains no duplicates. The mapping from id to atom may permute (atoms can be reordered for cache), but the set of ids is constant.

### CI-5 — Type consistency
Every `types[i]` is in `[0, type_names.size())`. Every type has an entry in the masses array.

## Physics-level invariants

### PI-1 — Energy conservation in NVE
Total energy `E = K + U` is conserved up to second-order Verlet drift:

```
|ΔE / E| < tolerance(dt, system, integrator)
```

Default tolerance: `1e-7 per atom per picosecond` for metals at room temperature with `dt=1fs`.

### PI-2 — Linear momentum conservation in NVE+PBC
`|Σ m_i v_i|` should remain at its initial value (zero, if initialized correctly) within FP noise:

```
|Σ m_i v_i| < ε per atom
```

Drift indicates a bug in force symmetry or in periodic image handling.

### PI-3 — Equipartition in NVT
Kinetic energy per degree of freedom equals `(1/2) k_B T` after equilibration. All translational DOFs should have the same average kinetic energy within `O(1/√N)` statistical noise.

### PI-4 — Maxwell-Boltzmann velocity distribution
After thermalization, the per-component velocity histogram passes a Kolmogorov-Smirnov test against the analytic Gaussian with width `√(k_B T / m)`.

### PI-5 — Detailed balance (NVT/NPT)
Forward and backward statistics agree. Tested by running pairs of long simulations starting from time-reversed states.

## Scheduler invariants (TD-specific)

### SI-1 — Zone time monotonicity
For every zone, `time_step` is monotonic non-decreasing through its lifetime. The only transition that increments it is `Computing → Done`.

### SI-2 — No simultaneous Computing on conflicting zones
Two zones in `Computing` state at the same wall-clock instant must have disjoint atom sets and disjoint causal influence regions.

### SI-3 — Causal dependency holds
A zone `Z` cannot transition to `Computing` at time step `T` unless every neighbor zone has `time_step ≥ T - 1`.

### SI-4 — Same zone walking order on all ranks
Every rank walks zones in the same order. Diverging order is a bug in the partitioner.

(Full state-machine spec: `docs/01-theory/zone-state-machine.md`.)

## How invariants are checked

| Invariant | When | How |
|---|---|---|
| CI-1, CI-2, CI-5 | Every step (Debug build) | `TDMD_ASSERT` in integrator |
| CI-3 | Every neighbor use | `TDMD_ASSERT` in `compute_forces` entry |
| CI-4 | At dump/checkpoint time | hash check |
| PI-1 | VerifyLab `nve-drift` | post-run analysis script |
| PI-2 | VerifyLab `momentum-conservation` | post-run analysis script |
| PI-3, PI-4 | VerifyLab `nvt-stability` | statistical test in check.py |
| PI-5 | VerifyLab `detailed-balance` | M7+ |
| SI-1..SI-4 | Every zone transition | `TDMD_ASSERT` in `transition_to` |

## When an invariant fails

1. **Reproduce.** Get a minimal failing case.
2. **Add a regression test.** Either as a unit test (for code-level invariants) or as a VerifyLab case (for physics-level).
3. **Fix.** Find the root cause.
4. **Verify.** Run the test, run the full suite, confirm everything stays green.
5. **Document.** If the failure mode was non-obvious, write it down — either in a code comment or in `docs/06-decisions/` if the fix changes architecture.
