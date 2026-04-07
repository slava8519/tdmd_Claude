# Build and Run

> How to get TDMD building and running on your machine.

## Prerequisites

| Tool | Version | Required for |
|---|---|---|
| CMake | ≥ 3.24 | always |
| Ninja | any recent | always |
| C++ compiler | gcc-12+ or clang-15+ | always |
| CUDA Toolkit | 12.0+ | M2+ (GPU) |
| MPI | OpenMPI 4.1+ or MPICH 4.1+ | M5+ (multi-rank) |
| Python | 3.10+ | VerifyLab, benchmarks |
| LAMMPS | 2024-stable+ | VerifyLab A/B |

On Ubuntu 22.04:

```bash
sudo apt install cmake ninja-build g++-12 python3 python3-pip clang-format
# CUDA: follow NVIDIA's official instructions for your distribution
# MPI: sudo apt install libopenmpi-dev openmpi-bin
```

## Quick build

```bash
git clone <your fork>
cd tdmd
./scripts/build.sh           # Release build
./scripts/run-tests.sh       # run unit tests
./build/tdmd_standalone --version
```

## Build modes

```bash
./scripts/build.sh Release           # default, optimized
./scripts/build.sh Debug             # asserts, debuginfo, no opt
./scripts/build.sh RelWithDebInfo    # opt + debuginfo
./scripts/build.sh Release clean     # full clean rebuild
```

## CMake options

| Option | Default | Notes |
|---|---|---|
| `TDMD_BUILD_TESTS` | ON | Build the unit test binary |
| `TDMD_ENABLE_CUDA` | OFF (M0/M1) → ON (M2+) | Compile CUDA kernels |
| `TDMD_ENABLE_MPI`  | OFF (M0–M4) → ON (M5+) | Link against MPI |
| `TDMD_FP64`        | OFF | Force double precision everywhere |
| `TDMD_DETERMINISTIC` | OFF | Bit-identical mode (slower) |

Example custom configure:

```bash
cmake -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DTDMD_ENABLE_CUDA=ON \
  -DTDMD_FP64=ON
cmake --build build -j
```

## Running

At M0 the binary just prints version info. From M1 it accepts a LAMMPS-style input file:

```bash
./build/tdmd_standalone --input examples/two_atoms.in
```

## Running tests

```bash
./scripts/run-tests.sh                  # all unit tests
ctest --test-dir build -R smoke         # filter by name
ctest --test-dir build -V               # verbose
```

## Running VerifyLab (M1+)

```bash
./scripts/run-verifylab.sh              # fast suite
./scripts/run-verifylab.sh --suite full # full suite (slow)
./scripts/run-verifylab.sh --case nve-drift
```

## IDE setup

After a build, `compile_commands.json` is symlinked to repo root. Use it with:

- **VSCode** + clangd extension — works out of the box.
- **CLion** — open the repo, point at `CMakeLists.txt`.
- **Neovim** + clangd — also works out of the box.

## Common build issues

| Symptom | Fix |
|---|---|
| `CMake error: Could not find CUDA` | install CUDA toolkit, set `CUDACXX=/usr/local/cuda/bin/nvcc` |
| `unsupported clang version for CUDA` | use gcc-12 instead, or update CUDA |
| `undefined reference to MPI_*` | configure with `-DTDMD_ENABLE_MPI=ON` |
| linker can't find `tdmd::core` | run a clean rebuild: `./scripts/build.sh Release clean` |

## Cleaning

```bash
rm -rf build/                # everything
./scripts/build.sh Release clean
```
