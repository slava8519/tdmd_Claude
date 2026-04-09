# CI strategy for TDMD

> **Last updated:** 2026-04-09

## Current state

| Job | What it covers | What it misses |
|-----|---------------|----------------|
| **CPU build+test** | CPU-only code: core, io, domain, CPU potentials, CPU integrator, CPU neighbor list. Runs `ctest`. | Everything CUDA and MPI. |
| **CUDA compile-only** (TEMPORARY) | Compilation of `.cu` files, kernel launch syntax, CUDA API signatures, header includes. | Runtime correctness, race conditions, memory errors, physics, performance. |
| **MPI compile-only** (TEMPORARY) | MPI header includes, collective call signatures, linker resolution. | Deadlocks, rank logic bugs, off-by-one in partitioning. |

Both TEMPORARY jobs exist because GitHub free-tier runners have no GPU. They catch ~60% of typical regressions (broken signatures, missing headers, syntax errors) but provide **no runtime guarantees**.

## What CI does NOT catch

Be explicit about this. CI green means:

- CPU code compiles and passes tests.
- CUDA code compiles (but has NOT been run on a GPU).
- MPI code compiles (but has NOT been run with multiple ranks).

CI green does **NOT** mean:

- GPU kernels produce correct results.
- GPU race conditions are absent.
- Device memory errors are absent (use `compute-sanitizer` locally).
- Performance has not regressed.
- Physics is correct on CUDA path.
- MPI multi-rank logic is correct.
- Energy conservation holds.
- FP32 path works (only FP64 tests are reliable; FP32 has 12 known failures).

## Manual checks required

Before merging PRs that touch these areas:

| Area | Manual check | Command |
|------|-------------|---------|
| Any CUDA code (`src/**/*.cu`) | Run GPU tests | `ctest --test-dir build --output-on-failure` (CUDA build) |
| Potentials / integrator / scheduler | Run FastPipeline tests | `./build/tests/tdmd_cuda_tests --gtest_filter="FastPipelineScheduler.*"` |
| Any MPI code | Run MPI tests | `mpirun -np 2 ./build/tests/tdmd_mpi_tests` |
| Hybrid scheduler | Run hybrid tests | `mpirun -np 4 ./build/tests/tdmd_hybrid_mpi_tests` |
| Performance-sensitive changes | Run benchmark | `./build/benchmarks/bench_pipeline_scheduler --scheduler fast_pipeline --data benchmarks/phase1_baseline/small.data --steps 1000 --warmup 100` |
| Physics changes (forces, integrator) | Run NVE conservation | Check `FastPipelineScheduler.NVEConservation` and `LongNVEDrift` tests |

## Path to permanent solution

**Self-hosted runner with GPU.** This is the right answer. When the runner exists:

1. CUDA compile-only becomes CUDA compile-and-test (remove TEMPORARY tag).
2. MPI compile-only becomes MPI compile-and-test.
3. Add performance regression job (compare `bench_pipeline_scheduler` output against baseline).
4. Add FP32 test job (once tolerances are fixed in session 3).

Tracked as backlog item. Requires spare hardware with NVIDIA GPU, network access to GitHub, and `actions-runner` setup.

**Until then:** developers MUST run GPU tests manually before merging anything that touches CUDA code. "CI green" is not sufficient proof of correctness for CUDA/MPI changes.

## Why this matters

The project's value is in CUDA/MPI scheduler code. CI that doesn't cover this path gives a **false sense of security**. The TEMPORARY markers and this document exist to prevent that false sense from becoming invisible. Anyone reading "CI green" must understand what it actually means.
