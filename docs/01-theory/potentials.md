# Interatomic Potentials

> Math for the potentials TDMD supports.

## What TDMD supports

| Potential | Class | Milestone |
|---|---|---|
| Morse | Pair | M1 |
| Lennard-Jones | Pair | M1 (cheap to add) |
| EAM (setfl/funcfl) | Many-body | M1 (CPU) / M2 (GPU) |
| EAM/alloy | Many-body | M2 |
| ML (SNAP, ACE, MLIAP via plugin) | Many-body | M8+ |

## Pair potentials

A pair potential is a function `U(r)` of the scalar distance `r` between two atoms. Force on atom `i` from atom `j`:

```
F_ij = -dU/dr · r̂_ij
```

where `r̂_ij` is the unit vector from `i` to `j`.

### Morse

```
U(r) = D · [(1 - exp(-α(r - r₀)))² - 1]
```

Parameters: `D` (well depth), `α` (width), `r₀` (equilibrium distance).

Derivative:

```
dU/dr = 2 · D · α · (1 - exp(-α(r - r₀))) · exp(-α(r - r₀))
```

We compute force-over-r so we don't need a sqrt:

```
fpair = -(dU/dr) / r
```

Then `F_ij = fpair · (r_i - r_j)`.

### Lennard-Jones

```
U(r) = 4ε [ (σ/r)¹² - (σ/r)⁶ ]
```

Standard textbook form. Trivial to implement, useful as a sanity check.

## EAM (Embedded Atom Method)

EAM is a many-body potential of the form:

```
U_total = Σ_i F(ρ_i) + (1/2) Σ_i Σ_{j≠i} φ(r_ij)
```

where:
- `φ(r)` is a pair term (different from the simple Morse pair).
- `F(ρ)` is the embedding function — a function of the local electron density at atom `i`.
- `ρ_i = Σ_{j≠i} f(r_ij)` is the local density, computed by summing contributions `f(r)` from neighbors.

### Three-stage computation

EAM forces are computed in **three passes** over neighbors:

1. **Density gather:** for each atom `i`, sum `ρ_i = Σ f(r_ij)`.
2. **Embedding derivative:** for each atom `i`, compute `F'(ρ_i)`.
3. **Force scatter:** for each pair `(i, j)`, contribute `F'(ρ_i) · f'(r_ij) + F'(ρ_j) · f'(r_ji) + φ'(r_ij)` to the force on `i` and `j`.

This is the canonical EAM force calculation. It requires two neighbor passes (passes 1 and 3) plus a per-atom pass (pass 2).

### Why EAM is harder for spatial decomposition

Pass 2 requires a full per-atom value (`F'(ρ_i)`) that depends on all neighbors. If neighbors are split across MPI ranks, you must communicate `ρ_i` (or `F'(ρ_i)`) for ghost atoms before pass 3 can run. That's an extra synchronization per step.

In TD, every rank holds its own complete copy of the model (per zone), so EAM's three-stage computation runs entirely locally. **This is one of the main reasons TD is well-suited to many-body potentials.**

### File formats

- **funcfl** — single-element EAM. Simple format.
- **setfl** — multi-element EAM/alloy. Used for HEA and metal alloys.

We parse both at M1 (CPU) and reuse the same parser for the GPU code at M2.

## Cutoffs and skin

Every potential has a finite cutoff `r_c`. We pad it with a skin `r_skin`:

```
r_list = r_c + r_skin
```

Atoms within `r_list` are kept in the neighbor list. The list is rebuilt when any atom has moved more than `r_skin / 2` since the last build (Verlet's classic criterion).

Default values for metals (overridable per potential):
- `r_c` from the potential file.
- `r_skin = 1.0 Å`.

## Tests required (M1)

- **Morse two-atom analytic** — forces match analytic derivative to FP precision.
- **Morse run-0 vs LAMMPS** — small box, FP64, tolerance 1e-6 relative.
- **EAM run-0 vs LAMMPS** — single-element FCC Cu, FP64, tolerance 1e-6 relative.
- **EAM/alloy run-0 vs LAMMPS** — Ni-Co-Cr small box, FP64, tolerance 1e-6 relative.
- **Cutoff sanity** — pair beyond `r_c` contributes zero force.
