# VerifyLab: single-gpu-collapse

> Verifies that TDMD in single-GPU mode (P_time=1, P_space=1) produces identical results to the batched-launch path and has zero TD overhead.

## What this tests

When all zones have the same `time_step` and there is one rank, the scheduler should detect this degenerate case and collapse all zones into a single batched launch per phase. The result must be:

1. **Bit-identical** to the non-zone M2 global compute path (in deterministic FP64 mode).
2. **Zero TD overhead**: no per-zone launch boundaries, no stream-per-zone allocation, no dependency checks (all trivially pass).

## Test cases

### 1. Physics equivalence (deterministic)

- Input: Cu FCC 256 atoms, Morse potential, NVE, 1000 steps
- Mode: deterministic (single stream, FP64 reductions)
- Reference: M2 global compute (no zones)
- Criterion: max position difference < 1e-12

### 2. Physics equivalence (fast mode)

- Input: Cu FCC 256 atoms, Morse potential, NVE, 1000 steps
- Mode: fast (multi-stream)
- Reference: M2 global compute
- Criterion: energy drift |dE/E| < 1e-6 (same as M2)

### 3. Performance check

- Input: Cu FCC 32K atoms, Morse potential, NVE, 5000 steps
- Metric: timesteps/s
- Criterion: collapsed zone scheduler within 1.05x of non-zone global compute
- Purpose: detect schedule overhead in the degenerate case

## Status

- [ ] Test 1 implemented
- [ ] Test 2 implemented
- [ ] Test 3 implemented
- [ ] Prerequisite: scheduler implements zone collapse (ADR 0005-batched-force-kernels.md Phase 2)
