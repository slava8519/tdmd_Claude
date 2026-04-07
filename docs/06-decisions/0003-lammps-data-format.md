# 0003 — LAMMPS `data` file format is the canonical input

- **Status:** Accepted
- **Date:** 2026-04-07
- **Decider:** human + architect
- **Affected milestone(s):** M1+

## Context

TDMD needs an input format for atomic systems: positions, types, masses, box dimensions, optionally velocities. Many formats exist (XYZ, PDB, CIF, GROMACS gro/top, LAMMPS data, ExtXYZ, ASE Atoms JSON).

We need one canonical format that:
- Round-trips with LAMMPS for A/B testing.
- Is human-readable enough for small test cases.
- Has tooling support (ASE, OVITO, VMD all read it).
- Is well-documented and stable.

## Options considered

### Option A — LAMMPS `data` file (chosen)
- Pros: trivial round-trip with LAMMPS, well-known in materials community, ASE/OVITO/VMD all read it, stable for years.
- Cons: rigid format, hard to extend, idiosyncratic (e.g., atom-style-dependent column layout).

### Option B — ExtXYZ
- Pros: simpler, more extensible, ASE-native.
- Cons: not a first-class LAMMPS input — requires conversion in tests.

### Option C — Custom JSON / TOML format
- Pros: easy to parse, extensible.
- Cons: incompatible with everyone else; would need converters in both directions.

## Decision

**LAMMPS `data` file is the canonical input.** TDMD reads it directly with its own parser (no LAMMPS dependency). For the first milestone we support:

- `units metal` only.
- `atomic` atom_style only.
- Orthorhombic box only (triclinic post-M7).
- Optional `Velocities` section.

Other formats can be supported later via conversion utilities, but `data` is the ground truth.

## Consequences

- **Positive:** zero-friction A/B with LAMMPS, ecosystem compatibility.
- **Negative:** must implement a robust parser; LAMMPS data files have many edge cases.
- **Risks:** parser edge cases will cause subtle bugs — mitigate with golden-file tests and a fuzz test on real LAMMPS examples.
- **Reversibility:** high — adding more formats later is easy.

## Follow-ups

- [ ] M1: implement `LammpsDataReader` for the supported subset.
- [ ] M1: golden-file test against a few small LAMMPS data files.
- [ ] M7: extend to triclinic boxes.
