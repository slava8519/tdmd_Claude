# Single-GPU TDMD vs LAMMPS-GPU benchmark

> Verifies the single-GPU performance contract: TDMD must achieve throughput within **1.5x** of LAMMPS-GPU on the same input and hardware.

## Context

When `P_time=1, P_space=1`, TDMD operates as a conventional single-GPU MD code. All zones have the same `time_step`, the scheduler collapses all zones into one batched launch per phase, and there is zero TD overhead. If TDMD is slower than 1.5x LAMMPS-GPU in this mode, it indicates a Level 2 deficiency (under-utilization of the GPU due to per-zone launches or excessive launch overhead).

See ADR `0005-batched-force-kernels.md` for migration plan.

## Benchmark matrix

| System | Atoms | Potential | Steps | Hardware |
|---|---|---|---|---|
| Cu FCC small | 4,096 | Morse | 10,000 | RTX 5080 |
| Cu FCC medium | 32,768 | Morse | 10,000 | RTX 5080 |
| FCC16 medium | 32,768 | EAM | 10,000 | RTX 5080 |
| FCC16 large | 262,144 | EAM | 5,000 | RTX 5080 |

## Metrics

- **timesteps/s** — primary metric
- **GPU occupancy** — via `nsys profile`
- **kernel launch count per tick** — batched (1 per phase) vs per-zone (N_zones per phase)
- **total kernel time vs total wall time** — schedule overhead fraction

## Baselines to collect

### Phase 1 (pre-migration): current per-zone launch model
- [ ] Cu FCC small: timesteps/s, occupancy
- [ ] Cu FCC medium: timesteps/s, occupancy
- [ ] FCC16 medium: timesteps/s, occupancy
- [ ] FCC16 large: timesteps/s, occupancy

### Phase 2 (post-migration): batched launch model
- [ ] Same matrix, same hardware
- [ ] Compare against Phase 1 baselines
- [ ] Compare against LAMMPS-GPU (same inputs, `lmp -sf gpu -pk gpu 1`)

## Running

```bash
# LAMMPS reference
lmp -sf gpu -pk gpu 1 -in benchmarks/single_gpu_vs_lammps/lammps/in.cu_morse.in -var natoms 32768

# TDMD (current)
./build/bin/tdmd --input benchmarks/single_gpu_vs_lammps/tdmd/in.cu_morse_32k.in --benchmark

# Profile
nsys profile --stats=true ./build/bin/tdmd --input ... --benchmark
```

## Status

- [ ] LAMMPS inputs created
- [ ] TDMD inputs created
- [ ] Phase 1 baselines collected (pre-migration)
- [ ] Phase 2 baselines collected (post-migration)
- [ ] 1.5x contract verified or deficit documented
