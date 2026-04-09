# Phase 1 measurement checklist

Follow these steps in order. Report results to the AI assistant after each step.

## Prerequisites

- [ ] TDMD built with CUDA and benchmarks enabled (see build command below)
- [ ] `nsys` available in PATH (`nsys --version`)
- [ ] `ncu` available in PATH (`ncu --version`) — optional, for occupancy data
- [ ] RTX 5080 visible to CUDA (`nvidia-smi`)
- [ ] No other heavy GPU processes running (`nvidia-smi` shows low utilization)

### Build command

```bash
PATH=$HOME/.local/cuda-12.6/bin:$HOME/.local/bin:$PATH \
cmake -B build -S . -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DTDMD_BUILD_TESTS=ON \
    -DTDMD_FP64=ON \
    -DTDMD_ENABLE_CUDA=ON \
    -DTDMD_BUILD_BENCHMARKS=ON

cmake --build build
```

---

## Step 0: sanity — nsys on existing test

Verify that nsys + GPU + TDMD work together before any new code.

```bash
mkdir -p benchmarks/phase1_baseline/results/sanity
nsys profile \
    --output=benchmarks/phase1_baseline/results/sanity/trace \
    --trace=cuda,osrt \
    --stats=true \
    ./build/tdmd_cuda_tests --gtest_filter="PipelineScheduler.PipelineNVEConservation"
```

Expected: test passes, nsys produces a `.nsys-rep` file. Runtime < 1 minute.

**Collect and report:**
- Did the test pass?
- Does `nsys stats` output show CUDA kernel launches?
- Rough kernel count from the summary.

---

## Step 1a: generate data files

```bash
./benchmarks/phase1_baseline/generate_data.sh
```

Verify output:
```bash
ls -lh benchmarks/phase1_baseline/*.data
```

Expected: `tiny.data` (256 atoms), `small.data` (4000 atoms), `medium.data` (32000 atoms).

---

## Step 1b: validate inputs

```bash
python3 benchmarks/phase1_baseline/validate_inputs.py
```

Check that TDMD and LAMMPS parameters match (Morse D/alpha/r0/rc, dt, skin).

---

## Step 2: TDMD tiny (smoke test)

```bash
./benchmarks/phase1_baseline/run_tdmd_profiled.sh tiny
```

Expected runtime: < 30 seconds. This verifies the benchmark tool works.

**Collect:** timesteps/s, kernel_launches_per_step from `results.json`.

---

## Step 3: TDMD small

```bash
./benchmarks/phase1_baseline/run_tdmd_profiled.sh small
```

Expected runtime: 1-5 minutes.

**Collect:** timesteps/s, kernel_launches_per_step, nsys summary.

---

## Step 4: TDMD medium

```bash
./benchmarks/phase1_baseline/run_tdmd_profiled.sh medium
```

Expected runtime: 5-30 minutes.

**If medium takes more than 1 hour, STOP** (Ctrl+C) and report. This would indicate per-zone launches are catastrophic at this size.

**Collect:** timesteps/s, kernel_launches_per_step, nsys summary.

---

## Step 5: LAMMPS small (A/B reference)

```bash
./benchmarks/phase1_baseline/run_lammps_baseline.sh
```

Expected runtime: < 5 minutes.

**Collect:** timesteps/s from LAMMPS log, nsys summary.

---

## Step 6: summary table

Fill in this table and paste it to the AI:

| Metric | TDMD tiny | TDMD small | TDMD medium | LAMMPS small |
|---|---|---|---|---|
| n_atoms | 256 | 4000 | 32000 | 4000 |
| n_zones | | | | N/A |
| timesteps/s | | | | |
| wall clock (s) | | | | |
| kernel_launches_per_step | | | | N/A |
| kernel launches total (nsys) | | | | |
| avg kernel duration (us) | | | | |
| launch overhead fraction | | | | |
| sm occupancy (ncu) | | | | N/A |

See `HOW_TO_READ_TRACES.md` for how to extract each metric from the traces.

---

## Step 7: pack and send

```bash
tar czf phase1_results.tar.gz benchmarks/phase1_baseline/results/
```

Send the archive along with the summary table.
