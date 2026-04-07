# TDMD Benchmarks

> Performance and validation benchmarks. Read this alongside `docs/05-benchmarks/`.

## Suites

| Suite | Purpose | Status |
|---|---|---|
| `fcc16/` | FCC-lattice multi-type EAM stress test (the main benchmark) | M2+ |
| `nicocr/` | Ni-Co-Cr 3-component baseline (sanity check, easier than fcc16) | M2 |

## Running

```bash
./scripts/run-benchmarks.sh                  # all suites (M5+)
./scripts/run-benchmarks.sh --suite fcc16    # single suite
```

## What we measure

See `docs/05-benchmarks/metrics.md`. Headlines:

- `timesteps/s` — wall-clock throughput.
- `atom·steps/(s·GPU)` — normalized throughput, comparable across system sizes and GPU counts.
- LAMMPS-style breakdown: Pair / Neigh / Comm / Integrate / Output / Modify / Schedule / Other.
- Energy drift over the benchmark window (`|ΔE/E|/atom/ps`).
- Force comparison vs LAMMPS (run-0 forces, max relative error).

## Adding a new benchmark

1. Create `benchmarks/<n>/`.
2. Add a `README.md` describing the system, the question it answers, and the run command.
3. Add `lammps/in.<n>` for the LAMMPS reference.
4. Add `expected/` for committed baseline numbers.
5. Wire it into `scripts/run-benchmarks.sh` (M5+).
