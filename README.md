# TDMD — Time-Decomposition Molecular Dynamics

> Современная MD-система на C++20/CUDA, реализующая **метод декомпозиции по времени** (Time Decomposition, TD) из диссертации В. В. Андреева под руководством А. М. Липанова.
> Цель — одна из самых быстрых в мире программ молекулярной динамики для металлов и сплавов с поддержкой Morse / EAM / ML-потенциалов на NVIDIA GPU.

---

## What makes TDMD different

Almost every modern MD code (LAMMPS, GROMACS, HOOMD, NAMD) parallelizes by **splitting space** across processors. TDMD is built on a different idea: because interatomic potentials have a finite cutoff, regions of the model that are far apart cannot influence each other within a single integration step. So we can **compute different time steps simultaneously for different regions**.

This is the Time Decomposition (TD) method. Its unique properties:

- **Only two communication neighbors per rank** (ring topology) vs. 6+ for spatial decomposition.
- **Linearly reduced communication bandwidth** via multi-step batching: `t_comm ∝ S / (K · P)`.
- **Better scaling on many-body potentials** (EAM, ML) — no spatial split is needed within a single step.

On modern GPU clusters where compute outpaces interconnect bandwidth, TD has architectural advantages no other MD method offers.

---

## Status

**Milestones M0–M7 complete.** Active development phase: single-GPU performance optimization.

What works today:
- **CPU reference MD** — standalone driver with Morse and EAM potentials, NVE integration, LAMMPS data file input.
- **GPU single-rank** — `FastPipelineScheduler` with batched whole-system kernels. On RTX 5080 with FP32 Morse: **9,800 ts/s** on 4k atoms, **7,600 ts/s** on 32k atoms (**2.4x faster than LAMMPS-GPU** on medium systems).
- **GPU multi-rank (scaffold)** — `DistributedPipelineScheduler` and `HybridPipelineScheduler` implement multi-rank TD and 2D time x space pipelines. Currently uses full replication (each rank holds all atoms) — correctness scaffold, not production distributed MD. See [ADR 0006](docs/06-decisions/0006-distributed-scaffold-honesty.md).
- **NVT thermostat** — Nose-Hoover chain with MTTK integration in all schedulers.
- **Adaptive dt** — `dt = min(dt_max, c2 * rc / v_max)`, opt-in.
- **71 tests** (CPU + CUDA + MPI), LAMMPS A/B force validation.

What's next: neighbor list optimization (Phase 3), EAM migration to batched scheduler (Phase 4), kernel fusion K>1. See [`docs/03-roadmap/milestones.md`](docs/03-roadmap/milestones.md).

**Known limitations:**
- Multi-rank paths use full replication, not ghost-atom exchange. Not suitable for distributed benchmarking.
- FP32 test suite has tolerance issues (12 failures). FP64 is the validated path. Fix planned.
- CI covers CPU-only builds. CUDA and MPI have compile-only CI checks. Full GPU CI requires self-hosted runner.

---

## Quickstart

### CPU-only build (no GPU required)

```bash
git clone https://github.com/slava8519/tdmd_Claude.git
cd tdmd
./scripts/build.sh
./scripts/run-tests.sh
```

### GPU build (requires CUDA 12.x)

```bash
cmake -B build -S . -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DTDMD_ENABLE_CUDA=ON \
    -DTDMD_BUILD_TESTS=ON
cmake --build build -j

# Run tests
ctest --test-dir build --output-on-failure
```

### Run a simulation (CPU standalone driver)

```bash
./build/tdmd_standalone --data tests/data/cu_fcc_256.data --nsteps 1000 --thermo 100
```

### Run benchmarks (GPU)

```bash
cmake -B build -S . -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DTDMD_ENABLE_CUDA=ON \
    -DTDMD_BUILD_BENCHMARKS=ON
cmake --build build -j

./build/benchmarks/bench_pipeline_scheduler \
    --scheduler fast_pipeline \
    --data benchmarks/phase1_baseline/small.data \
    --steps 1000 --warmup 100
```

See [`docs/04-development/build-and-run.md`](docs/04-development/build-and-run.md) for full build options, FP64/FP32 modes, and MPI builds.

---

## Documentation (read in this order)

1. [`docs/00-vision.md`](docs/00-vision.md) — why this project exists.
2. [`docs/01-theory/`](docs/01-theory/) — Time Decomposition method, integrator, potentials, neighbor lists.
3. [`docs/02-architecture/`](docs/02-architecture/) — modules, data structures, scheduler design.
4. [`docs/03-roadmap/milestones.md`](docs/03-roadmap/milestones.md) — M0–M8 plan with exit criteria.
5. [`docs/04-development/`](docs/04-development/) — build, test, debug, code style, git workflow.
6. [`docs/05-benchmarks/`](docs/05-benchmarks/) — what we measure and against what.
7. [`docs/06-decisions/`](docs/06-decisions/) — architectural decision records (ADRs).
8. [`verifylab/README.md`](verifylab/README.md) — physics validation subsystem.
9. [`prompts/`](prompts/) — role prompts and workflows for Claude Code.

---

## For AI agents (Claude Code)

**Before any task, read [`CLAUDE.md`](CLAUDE.md) at the repo root.** It contains the rules for working on this codebase: hard rules, code style, testing, git, autonomy boundaries.

Role prompts (`prompts/roles/`) and workflow recipes (`prompts/workflows/`) overlay specific contexts on top of the base rules.

---

## Project goals (in priority order)

1. **Correctness** — physics matches LAMMPS within FP tolerance, verified by VerifyLab.
2. **Readability** — a solo developer can understand any part of the codebase in minutes.
3. **Speed** — among the fastest MD codes for metallic systems with EAM on NVIDIA GPUs.
4. **Usability** — LAMMPS-compatible inputs, sensible defaults, clear errors, real-time telemetry.

Speed never comes at the cost of correctness or readability.

---

## License

Apache 2.0 — see [`LICENSE`](LICENSE).

## Acknowledgements

The Time Decomposition method is the original contribution of the dissertation author (see [`docs/01-theory/dissertation-source.md`](docs/01-theory/dissertation-source.md)). TDMD is an independent modern implementation.

LAMMPS is used as the validation oracle. This project is not affiliated with the LAMMPS team.
