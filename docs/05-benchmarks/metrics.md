# Benchmark Metrics

> What we measure and how we report it.

## Primary metrics

### timesteps/s
The most important number. How many integration steps the system completes per wall-clock second on a given input.

Reported per benchmark, per build mode, per hardware.

### atom·steps / (s·GPU)
Normalized throughput. Lets us compare across system sizes.

```
throughput = (atoms × timesteps) / (wall_seconds × num_gpus)
```

### Strong scaling efficiency
For a fixed problem size, run on `1, 2, 4, 8, 16` ranks. Compute:

```
efficiency(P) = (timesteps_per_sec_at_P) / (P × timesteps_per_sec_at_1)
```

Ideal = 1.0. Above 0.85 is good. Below 0.5 means we're communication-bound.

### Weak scaling efficiency
For a per-rank problem size held constant, scale total work with `P`. Compute:

```
efficiency(P) = (timesteps_per_sec_at_P) / (timesteps_per_sec_at_1)
```

For TDMD this is more representative of cluster usage.

## Secondary metrics

- **GPU occupancy** — from `ncu`. Target: > 0.5 for major kernels.
- **Memory bandwidth utilization** — from `ncu`. Target: > 50% of peak for force kernels.
- **Pipeline occupancy** (TD-specific) — fraction of wall-clock time the scheduler is in `Computing` vs `Waiting`. Target: > 80% in steady state.
- **Comm fraction** — % of step time spent in send/recv. Target: < 15% on FCC16 medium at 4 ranks.
- **Neighbor build frequency** — every N steps. Target: every 10–50 steps.

## How metrics are recorded

Each benchmark run writes a JSON record:

```json
{
  "date": "2026-04-07T18:00:00Z",
  "commit": "abc1234",
  "build_type": "Release",
  "fp64": false,
  "deterministic": false,
  "hardware": {
    "gpu": "NVIDIA RTX 5080",
    "host_cpu": "AMD Ryzen 9 7950X",
    "ranks": 1,
    "gpus": 1
  },
  "case": "fcc16-medium",
  "atoms": 32768,
  "steps": 10000,
  "wall_seconds": 12.34,
  "timesteps_per_sec": 810.4,
  "atom_steps_per_sec_per_gpu": 26554000,
  "breakdown": {
    "pair": 7.12,
    "neighbor": 1.57,
    "comm": 0.23,
    "integrate": 1.89,
    "schedule": 0.89,
    "other": 0.64
  }
}
```

Records live in `benchmarks/results/` and are checked into git.

## Comparing against LAMMPS

LAMMPS-GPU is our reference. For each benchmark we keep a baseline LAMMPS run with:

- Same atoms (same data file).
- Same potential (same EAM file).
- Same `dt`, same number of steps.
- Same hardware (RTX 5080).

```
relative_speed = tdmd_timesteps_per_sec / lammps_timesteps_per_sec
```

Targets:
- M2: > 0.3 (we don't optimize yet)
- M5: > 0.5
- M7: > 0.7 single-GPU; > 1.0 on 4+ GPUs at the bandwidth-bound regime

## What we do NOT report

- "FLOPS." Almost meaningless for MD; varies wildly with potential complexity.
- "Speedup over CPU." Useless without context about which CPU.
- "Marketing numbers" rounded for press releases. We are not selling anything.
