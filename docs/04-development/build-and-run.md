# Build and Run

> How to get TDMD building and running on your machine.

## Prerequisites

| Tool | Version | Required for |
|---|---|---|
| CMake | >= 3.25 | always |
| Ninja | any recent | always |
| C++ compiler | gcc-13+ or clang-17+ | always |
| CUDA Toolkit | 12.4+ | GPU builds |
| MPI | OpenMPI 4.1+ or MPICH 4.1+ | multi-rank builds |
| Python | 3.11+ | VerifyLab, benchmarks, tools |
| LAMMPS | 2023-stable+ | VerifyLab A/B comparison |

On Ubuntu 22.04+:

```bash
sudo apt install cmake ninja-build g++-13 python3 python3-pip clang-format
# CUDA: follow NVIDIA's official instructions for your distribution
# MPI: sudo apt install libopenmpi-dev openmpi-bin
```

## Quick build (CPU-only)

```bash
./scripts/build.sh           # Release build, CPU-only
./scripts/run-tests.sh       # run unit tests
```

`scripts/build.sh` is a CPU-only convenience wrapper. For CUDA/MPI builds, use cmake directly (see below).

## Build modes

### CPU-only (default)

```bash
cmake -B build -S . -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DTDMD_BUILD_TESTS=ON
cmake --build build -j
```

### GPU (CUDA)

```bash
cmake -B build -S . -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DTDMD_ENABLE_CUDA=ON \
    -DTDMD_BUILD_TESTS=ON
cmake --build build -j
```

### GPU + FP64 (double precision, reference quality, slow on consumer GPUs)

```bash
cmake -B build -S . -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DTDMD_ENABLE_CUDA=ON \
    -DTDMD_FP64=ON \
    -DTDMD_BUILD_TESTS=ON
cmake --build build -j
```

Consumer GPUs (RTX 5080) have 1:32 FP64:FP32 ratio. FP64 is ~7-10x slower than FP32, but provides machine-epsilon energy conservation. Use FP64 for validation, FP32 for production runs.

### GPU + MPI (multi-rank)

```bash
cmake -B build -S . -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DTDMD_ENABLE_CUDA=ON \
    -DTDMD_ENABLE_MPI=ON \
    -DTDMD_BUILD_TESTS=ON
cmake --build build -j
```

### GPU + benchmarks

```bash
cmake -B build -S . -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DTDMD_ENABLE_CUDA=ON \
    -DTDMD_BUILD_BENCHMARKS=ON
cmake --build build -j
```

## CMake options

| Option | Default | Notes |
|---|---|---|
| `TDMD_BUILD_TESTS` | ON | Build unit test binaries |
| `TDMD_ENABLE_CUDA` | OFF | Compile CUDA kernels and GPU tests |
| `TDMD_ENABLE_MPI` | OFF | Link against MPI, enable multi-rank schedulers |
| `TDMD_FP64` | OFF | Force double precision (`real = double`). Default is `real = float` |
| `TDMD_DETERMINISTIC` | OFF | Bit-identical mode (single stream, sequential zone walk) |
| `TDMD_BUILD_BENCHMARKS` | OFF | Build `bench_pipeline_scheduler` and other benchmark tools |

## Running the standalone driver

The CPU standalone driver runs Morse NVE MD on LAMMPS data files:

```bash
./build/tdmd_standalone --data tests/data/cu_fcc_256.data --nsteps 1000 --thermo 100
```

CLI options:

```
--data <file>          LAMMPS data file (required)
--nsteps <N>           Number of steps (default 1000)
--dt <ps>              Time step in ps (default 0.001)
--morse D,alpha,r0,rc  Morse parameters (default 0.3429,1.3588,2.866,9.5)
--skin <A>             Neighbor list skin (default 1.0)
--thermo <N>           Print thermo every N steps (default 100)
--version, -v          Print version
--help, -h             Print this help
```

Without `--data`, the binary prints a brief help message and exits.

## Running tests

```bash
# All tests (CPU or CPU+CUDA depending on build)
ctest --test-dir build --output-on-failure

# Filter by name
ctest --test-dir build -R FastPipeline

# CUDA tests only
./build/tests/tdmd_cuda_tests

# MPI tests (requires mpirun)
mpirun -np 2 ./build/tests/tdmd_mpi_tests
mpirun -np 4 ./build/tests/tdmd_hybrid_mpi_tests
```

## Running benchmarks

Requires `-DTDMD_BUILD_BENCHMARKS=ON` and `-DTDMD_ENABLE_CUDA=ON`.

```bash
# Generate data files (tiny 256, small 4000, medium 32000 atoms)
./benchmarks/phase1_baseline/generate_data.sh

# FastPipelineScheduler (batched, recommended)
./build/benchmarks/bench_pipeline_scheduler \
    --scheduler fast_pipeline \
    --data benchmarks/phase1_baseline/small.data \
    --steps 1000 --warmup 100

# Old PipelineScheduler (per-zone, baseline comparison)
./build/benchmarks/bench_pipeline_scheduler \
    --scheduler pipeline \
    --data benchmarks/phase1_baseline/small.data \
    --steps 1000 --warmup 100
```

Output is JSON to stdout, human-readable summary to stderr. See [`benchmarks/phase1_baseline/README.md`](../../benchmarks/phase1_baseline/README.md) for the full measurement guide.

**Important:** run benchmarks one at a time, never in parallel. See [measurement exclusivity rule](../../CLAUDE.md).

## IDE setup

After a build, `compile_commands.json` is symlinked to repo root. Use it with:

- **VSCode** + clangd extension — works out of the box.
- **CLion** — open the repo, point at `CMakeLists.txt`.
- **Neovim** + clangd — also works out of the box.

## Common build issues

| Symptom | Fix |
|---|---|
| `CMake error: Could not find CUDA` | install CUDA toolkit, set `CUDACXX=/usr/local/cuda/bin/nvcc` |
| `unsupported clang version for CUDA` | use gcc-13 instead, or update CUDA |
| `undefined reference to MPI_*` | configure with `-DTDMD_ENABLE_MPI=ON` |
| linker can't find `tdmd::core` | run a clean rebuild: `rm -rf build && cmake -B build ...` |

## Cleaning

```bash
rm -rf build/
```
