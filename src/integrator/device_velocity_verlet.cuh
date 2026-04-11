// SPDX-License-Identifier: Apache-2.0
// device_velocity_verlet.cuh — GPU velocity-Verlet NVE integrator.
#pragma once

#include <cuda_runtime.h>

#include "../core/box.hpp"
#include "../core/types.hpp"

namespace tdmd::integrator {

/// @brief GPU velocity-Verlet half-kick: v += (dt/2) * F / (m * mvv2e).
void device_half_kick(VelocityVec* d_velocities, const ForceVec* d_forces,
                      const i32* d_types, const real* d_masses, i32 natoms,
                      real dt, cudaStream_t stream = 0);

/// @brief GPU velocity-Verlet drift: r += dt * v, then wrap into box.
void device_drift(PositionVec* d_positions, const VelocityVec* d_velocities,
                  i32 natoms, real dt, const Box& box,
                  cudaStream_t stream = 0);

/// @brief Fused half-kick + drift in a single per-atom kernel.
///
/// Bit-exact equivalent of calling `device_half_kick` immediately followed by
/// `device_drift`: both are embarrassingly parallel per atom with no
/// cross-atom dependency, so merging them saves one launch + the intermediate
/// velocity round-trip through HBM. Math contract (ADR 0007) unchanged:
/// half-kick and drift arithmetic both run in double; storage writeback is
/// direct. Used by FastPipelineScheduler as the BOOTSTRAP step at the start
/// of a run_until window (velocities at integer time t → v(t+dt/2)+drift).
void device_fused_half_kick_drift(VelocityVec* d_velocities,
                                  PositionVec* d_positions,
                                  const ForceVec* d_forces,
                                  const i32* d_types, const real* d_masses,
                                  i32 natoms, real dt, const Box& box,
                                  cudaStream_t stream = 0);

/// @brief Fused full-kick + drift in a single per-atom kernel.
///
/// Collapses the cross-step pair (kick2 of step N) + (kick1 of step N+1)
/// into a single full-dt velocity update, then drifts positions. Used for
/// every step *after the first* inside a run_until window. Saves one
/// launch per step vs the half_kick + fused_half_kick_drift pattern
/// (OPT-FUSE-1c). Math contract unchanged: arithmetic in double, direct
/// writeback. Only valid when the incoming velocities are already
/// half-advanced — i.e., the previous step ran force compute and did NOT
/// apply a closing half-kick. The scheduler is responsible for issuing
/// `device_half_kick` exactly once at the end of each run_until window to
/// finalize velocities at integer time t+N dt before download.
void device_fused_full_kick_drift(VelocityVec* d_velocities,
                                  PositionVec* d_positions,
                                  const ForceVec* d_forces,
                                  const i32* d_types, const real* d_masses,
                                  i32 natoms, real dt, const Box& box,
                                  cudaStream_t stream = 0);

/// @brief Zero all force vectors on GPU (whole-system).
void device_zero_forces(ForceVec* d_forces, i32 natoms,
                        cudaStream_t stream = 0);

}  // namespace tdmd::integrator
