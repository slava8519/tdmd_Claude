# Phase 1 / Phase 2 measurement infrastructure

> Benchmark and profiling tools for TDMD scheduler development (ADR 0005).

## What this is

Measurement infrastructure for comparing TDMD schedulers (`PipelineScheduler` vs `FastPipelineScheduler`) and establishing baselines against LAMMPS-GPU. Used for ADR 0005 Phase 1 (measurement) and Phase 2 (validation of batched kernels).

## How to use

### Build

```bash
# FP64 mode (higher precision, slower on consumer GPUs)
cmake -B build -S . -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DTDMD_BUILD_TESTS=ON -DTDMD_FP64=ON \
    -DTDMD_ENABLE_CUDA=ON -DTDMD_BUILD_BENCHMARKS=ON
cmake --build build -j

# FP32 mode (full GPU throughput on RTX 5080)
cmake -B build-fp32 -S . -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DTDMD_BUILD_TESTS=ON -DTDMD_FP64=OFF \
    -DTDMD_ENABLE_CUDA=ON -DTDMD_BUILD_BENCHMARKS=ON
cmake --build build-fp32 -j
```

### Generate data files

```bash
./benchmarks/phase1_baseline/generate_data.sh
# Creates: tiny.data (256 atoms), small.data (4000), medium.data (32000)
```

### Run benchmarks

```bash
# FastPipelineScheduler (batched, recommended)
./build/benchmarks/bench_pipeline_scheduler \
    --scheduler fast_pipeline \
    --data benchmarks/phase1_baseline/small.data \
    --steps 1000 --warmup 100

# Old PipelineScheduler (per-zone, baseline)
./build/benchmarks/bench_pipeline_scheduler \
    --scheduler pipeline \
    --data benchmarks/phase1_baseline/small.data \
    --steps 1000 --warmup 100
```

Output is JSON to stdout, human-readable summary to stderr.

### Compare with LAMMPS-GPU

```bash
# Set LAMMPS binary path (default: third_party/lammps/build/lmp)
export LMP_BIN=third_party/lammps/build/lmp

# Small (4000 atoms)
cd benchmarks/phase1_baseline/lammps_small && $LMP_BIN -in in.baseline -sf gpu -pk gpu 1

# Medium (32000 atoms)
cd benchmarks/phase1_baseline/lammps_medium && $LMP_BIN -in in.baseline -sf gpu -pk gpu 1
```

### Profile with nsys

```bash
nsys profile --output=results/trace --trace=cuda,osrt --stats=true \
    ./build/benchmarks/bench_pipeline_scheduler \
    --scheduler fast_pipeline \
    --data benchmarks/phase1_baseline/small.data \
    --steps 1000 --warmup 100
```

## IMPORTANT: measurement exclusivity rule

**Run benchmarks one at a time, never in parallel, never in background.**

Two concurrent GPU benchmarks compete for SMs, L2 cache, memory bandwidth, and CUDA context locks. Results become random noise.

Protocol:
1. Run one benchmark. Wait for it to finish completely.
2. Wait 5-10 seconds for GPU thermal stabilization and context release.
3. Run the next benchmark.

This applies equally to `bench_pipeline_scheduler`, `nsys profile`, `ncu`, and long-running unit tests that stress the GPU.

## Files

| File | Purpose |
|---|---|
| `bench_pipeline_scheduler.cu` | Benchmark executable with `--scheduler` flag |
| `generate_data.sh` | Generate Cu FCC data files (tiny/small/medium) |
| `run_tdmd_profiled.sh` | Run benchmark + nsys + ncu |
| `run_lammps_baseline.sh` | Run LAMMPS-GPU on small system |
| `validate_inputs.py` | Compare TDMD/LAMMPS parameters |
| `lammps_small/in.baseline` | LAMMPS input matching TDMD small |
| `lammps_medium/in.baseline` | LAMMPS input matching TDMD medium |
| `CHECKLIST.md` | Step-by-step measurement instructions |
| `HOW_TO_READ_TRACES.md` | nsys/ncu metric extraction guide |

## Current results

See [Phase 2 results analysis](../../docs/05-benchmarks/phase2-batched-scheduler-results.md) for full measurement data and analysis.

## Known limitations

1. **No NVTX markers in src/.** nsys traces show kernel names but no semantic phase labels.
2. **LAMMPS velocity initialization differs from TDMD.** Same temperature, different RNG — acceptable for performance comparison, not for atom-by-atom force matching.
3. **Results depend on GPU thermal state.** Lock clocks if possible (`nvidia-smi -lgc`), always respect measurement exclusivity.
