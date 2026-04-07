# Integrator — Velocity Verlet

> The integration scheme used by TDMD.

## Why velocity Verlet

We use **velocity Verlet** for time integration. It is the standard choice in MD because:

- **Symplectic** — bounded long-term energy drift in NVE.
- **Time-reversible** — important for correct ensemble sampling.
- **Second-order accurate** — `O(Δt²)` per step, `O(Δt²)` global.
- **Single force evaluation per step** — efficient.
- **Compatible with TD** — the only force evaluation is at the start of the step, so the dependency pattern is simple and matches the zone state machine.

## The algorithm

For each step:

1. **Half-kick:** `v(t + Δt/2) = v(t) + (Δt/2) · F(t) / m`
2. **Drift:**     `r(t + Δt) = r(t) + Δt · v(t + Δt/2)`
3. **Force:**     `F(t + Δt) = compute_forces(r(t + Δt))`
4. **Half-kick:** `v(t + Δt) = v(t + Δt/2) + (Δt/2) · F(t + Δt) / m`

That's it. Forces are computed once per step.

## Why this maps cleanly to TD

The key observation: **the force evaluation in step 3 only depends on positions at `t + Δt`**. The half-kicks (1 and 4) are local to each atom and trivially parallel.

So the dependency between zone Z and zone Z' at step `n+1` is:
- Z needs `r(t + Δt)` from atoms in `(Z ∪ neighbors(Z))`.
- It does **not** need anything from atoms outside that set.

This is exactly the locality assumption that TD relies on. The dissertation's "causal sphere of one step" is `r_c + v_max · Δt` — the cutoff plus the maximum drift in one step.

## Choice of Δt

Standard for metals: `Δt = 1 fs = 10⁻¹⁵ s` in `units metal`. This is conservative and works for most EAM systems up to a few thousand K.

For very stiff potentials (light atoms, hydrogen), `Δt = 0.5 fs` or smaller may be needed.

For high-temperature melt simulations, the dissertation's adaptive Δt can be enabled (M7+).

## Adaptive Δt (M7)

The dissertation describes an adaptive scheme where Δt is adjusted per cycle based on the maximum atomic velocity:

```
Δt = min(Δt_max, C₂ · r_c / v_max)
```

This is implemented as an opt-in mode for high-temperature or shock simulations. It is **not** the default — the default is fixed `Δt`.

## What we do NOT use

- **Leapfrog** — equivalent to velocity Verlet but velocities and positions are at staggered time steps, which complicates the TD scheduler. Avoid.
- **RESPA / multi-timestep** — the dissertation's TD method *is* a multi-timestep scheme in a different sense; mixing it with RESPA-style hierarchy is interesting future work but not for the first version.
- **Gear predictor-corrector** — historical, no advantages, more memory.

## Tests required (M1)

- **Two-atom Morse:** energy and trajectory match analytic solution to 1e-12 in FP64.
- **Harmonic oscillator:** period matches `2π√(m/k)` exactly.
- **NVE drift:** `|ΔE/E| < 1e-7` per atom per ps over 50,000 steps at `Δt=1 fs`.
- **Time reversibility:** integrate forward N steps, reverse velocities, integrate N steps, return to start within FP tolerance.
