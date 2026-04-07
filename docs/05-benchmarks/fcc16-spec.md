# FCC16 Benchmark Specification

> The flagship validation and stress benchmark for TDMD.
> Adapted from prior research notes; see `deep-research-report.md` in the project root for the full methodology.

## What it is

A 16-component high-entropy alloy (HEA) on an FCC lattice, simulated with an EAM/alloy potential. The 16 elements, lattice parameter, and exact composition follow a published HEA reference (TBD specific paper for M5).

## Why this benchmark

1. **Many-body potential** — EAM is harder for spatial decomposition than pair potentials, which is exactly where TD has its biggest advantage.
2. **Large type count** — 16 element types stress the per-type tables and the EAM density gather kernel.
3. **High energy density** — HEAs have non-trivial dynamics, so trajectories are sensitive to bugs.
4. **LAMMPS reference exists** — we can A/B with LAMMPS-GPU on the same system.

## Sizes

| Name | atoms | purpose |
|---|---|---|
| FCC16 small | 256 | unit-test scale, quick A/B |
| FCC16 medium | ~32,000 | scaling tests, single GPU |
| FCC16 large | ~256,000 | multi-GPU scaling |

## Files

- `benchmarks/fcc16/generate.py` — generates the data file (M2+).
- `benchmarks/fcc16/lammps/in.fcc16_*.in` — LAMMPS input scripts for each size.
- `benchmarks/fcc16/expected/` — committed reference outputs (thermo logs).

## Pre-FCC16 ramp-up

Before running the FCC16 stress test, we **must** pass the simpler benchmarks first, in this order:

1. **Two-atom Morse** (analytic) — VerifyLab.
2. **Pure-pair Morse box** — small periodic box, A/B vs LAMMPS.
3. **Single-element EAM** (Cu, Al, Ni) — single-component validation.
4. **Ni-Co-Cr (3-component)** — multi-element EAM but small type count.
5. **FCC16** — full stress.

Each step proves the previous worked. Skipping steps loses information when things go wrong.

## Pass criteria

| Quantity | Tolerance |
|---|---|
| Forces (run 0, FP64, vs LAMMPS) | < 1e-6 relative |
| Forces (run 0, mixed precision) | < 1e-3 relative |
| Energy drift (NVE, 50k steps, dt=1fs) | < 1e-7 per atom per ps |
| MSD slope (vs LAMMPS, high T) | within 5% |
| timesteps/s (vs LAMMPS-GPU, single RTX 5080) | ≥ 70% (target for M7) |

## Timeline

- M2: small size on single GPU, validation only.
- M5: medium size on multiple ranks, validation + scaling.
- M7: large size, optimized, performance comparison vs LAMMPS-GPU.
