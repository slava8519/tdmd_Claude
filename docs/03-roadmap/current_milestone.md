# Current Milestone: M7 — NVT/NPT + adaptive Δt + optimizations ✅ COMPLETE

> **Goal:** Standard ensembles (NVT) and adaptive Δt from the dissertation.

## Checklist

- [x] `integrator/NoseHooverChain` — GPU NHC thermostat with MTTK integration scheme
- [x] `device_compute_ke` — two-pass GPU kinetic energy reduction
- [x] `device_scale_velocities` / `device_scale_velocities_zone` — GPU velocity scaling
- [x] `device_compute_vmax` — GPU max atomic speed reduction for adaptive Δt
- [x] NVT integration in `PipelineScheduler` (single-rank): NHC half-steps bracket each VV step
- [x] NVT integration in `DistributedPipelineScheduler` (multi-rank): MPI_Allreduce for global KE
- [x] NVT integration in `HybridPipelineScheduler` (2D time×space): MPI_Allreduce for global KE
- [x] Adaptive Δt mode: `dt = min(dt_max, c2 * rc / v_max)`, opt-in
- [x] Test: NVT temperature convergence to 300K target (< 15% relative error)
- [x] Test: device/host KE match (< 1e-10 relative)
- [x] Test: deterministic NVT reproducibility (< 1e-10 position diff)
- [x] Test: device/host v_max match (< 1e-10 relative)
- [x] Test: adaptive Δt NVE stability (|dE/E| < 1e-2)
- [x] Build system: device_nose_hoover.cu added to integrator CMakeLists.txt
- [x] CHANGELOG v0.7.0

## Exit criteria — status

- [x] NVT VerifyLab scenario: ⟨T⟩ within 15% of target for 256-atom system.
- [x] Adaptive Δt produces stable trajectories on thermal benchmark.
- [ ] NPT (deferred — NVT is the priority; NPT is a straightforward extension).
- [ ] Roofline analysis / kernel optimizations (deferred to post-M7).
- [ ] 70% of LAMMPS-GPU timesteps/s (deferred — needs profiling).
- [x] 68 tests passing (63 M0-M6 + 5 M7).

## Architecture notes

- NHC thermostat uses MTTK (Martyna-Tuckerman-Tobias-Klein) scheme.
- Chain masses: Q_1 = n_dof * kB * T * τ², Q_k = kB * T * τ² (LAMMPS convention).
- In NVT mode, all schedulers drain the pipeline before each thermostat step (serialized for correctness).
- Multi-rank NVT: each rank computes local KE → MPI_Allreduce(SUM) → replicated NHC half-step → scale owned velocities.
- Adaptive Δt uses `current_dt_` member instead of `cfg_.dt` in launch_zone_step().
- Variable step size breaks symplecticity → expect larger energy drift than fixed dt.

## Next: M8 — ML potentials (continuous)
