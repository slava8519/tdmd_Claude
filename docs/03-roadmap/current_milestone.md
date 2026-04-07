# Current Milestone: M2 — GPU Port (single GPU) ✅ COMPLETE

> **Goal:** Port all M1 physics to GPU. Same correctness, but running on CUDA.

## Checklist

- [x] `core/DeviceBuffer` — RAII CUDA memory wrapper (move-only)
- [x] `core/DeviceSystemState` — GPU-resident SoA mirror of SystemState
- [x] `domain/DeviceCellList` — counting sort + prefix sum on GPU
- [x] `neighbors/DeviceNeighborList` — full-list (no atomics for force scatter)
- [x] `potentials/device_morse` — GPU Morse pair force kernel
- [x] `potentials/device_eam` — GPU EAM 3-pass (density, embedding, force)
- [x] `integrator/device_velocity_verlet` — GPU half-kick + drift kernels
- [x] VerifyLab GPU: Morse forces match CPU < 1e-10
- [x] VerifyLab GPU: EAM forces match CPU < 1e-8
- [x] VerifyLab GPU: Morse run-0 force match vs LAMMPS < 1e-6
- [x] VerifyLab GPU: EAM run-0 force match vs LAMMPS < 1e-6
- [x] VerifyLab GPU: NVE drift 50k steps |dE/E| < 1e-4

## Exit criteria — ALL MET

- [x] GPU Morse forces match CPU to < 1e-10 (FP64).
- [x] GPU EAM forces match CPU to < 1e-8 (FP64).
- [x] Run-0 Morse: GPU vs LAMMPS max force diff < 1e-6.
- [x] Run-0 EAM: GPU vs LAMMPS max force diff < 1e-6.
- [x] GPU NVE conservation: 50k steps, |dE/E| < 1e-4.
- [x] All 46 tests pass (31 CPU + 15 GPU).

## Architecture notes

- GPU uses full neighbor list (vs CPU half-list) for lock-free force accumulation.
- Cell list prefix sum done on host (ncells is small, typically <1000).
- EAM spline tables packed flat in device global memory.
- CUDA architectures: sm_89 (Ada) native + sm_90 PTX (JIT for Blackwell sm_120).
- CUDA 12.6 with --expt-relaxed-constexpr.

## Next: M3 — Time Decomposition scheduler
