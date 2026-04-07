# Benchmarks

> What we measure to know TDMD is fast and correct.

## Two purposes

Benchmarks in TDMD serve two distinct goals:

1. **Validation** — confirm that TDMD computes the same physics as LAMMPS on representative problems. (See also `verifylab/` for the formal physics-correctness subsystem.)
2. **Performance tracking** — measure speed and scaling, detect regressions, compare against LAMMPS-GPU.

Both are important. A code that's fast but wrong is worse than no code.

## Suites

| Suite | Purpose | Status |
|---|---|---|
| `fcc16/` | Stress test on a 16-component HEA model | M5+ |
| `nicocr/` | Smaller 3-component HEA baseline (Ni-Co-Cr) | M2+ |
| `morse_simple/` | Pure-pair Morse benchmark, easy validation | M1+ |
| `eam_simple/` | Pure-EAM benchmark, single-element FCC | M1+ |

See individual `README.md` files in each subfolder.

## Metrics we track

- **timesteps/s** — primary speed metric.
- **atom·steps/(s·GPU)** — for cross-system normalization.
- **MPI rank scaling** — strong scaling efficiency at 1, 2, 4, 8, 16 ranks.
- **GPU occupancy** — from `nsys` / `ncu`.
- **Bandwidth utilization** — from `ncu`.

## How we run benchmarks

```bash
./scripts/run-benchmarks.sh                # full suite
./scripts/run-benchmarks.sh --suite morse  # one suite
./scripts/run-benchmarks.sh --compare-lammps  # also run LAMMPS for comparison
```

(`run-benchmarks.sh` is added at M2 — for M0/M1 there are no kernels to benchmark.)

## Recording results

Each benchmark run writes a JSON record to `benchmarks/results/<date>-<commit>.json`. The records are checked into git and used to track regressions over time.

`benchmarks/results/latest.json` always points to the most recent run.

## Performance regression policy

If a benchmark slows by more than **5%** on `main`, CI flags it. The author must either:
- Justify the slowdown (e.g., "added correctness check, X% slowdown is expected"), or
- Fix the regression before merging.

This policy is enabled at **M7** (the optimization milestone). Until then, we focus on correctness.
