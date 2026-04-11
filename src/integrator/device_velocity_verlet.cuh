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
/// direct. Used by FastPipelineScheduler::step_{morse,eam} to trim the per-
/// step kernel graph from 5→4 (Morse) / 7→6 (EAM).
void device_fused_half_kick_drift(VelocityVec* d_velocities,
                                  PositionVec* d_positions,
                                  const ForceVec* d_forces,
                                  const i32* d_types, const real* d_masses,
                                  i32 natoms, real dt, const Box& box,
                                  cudaStream_t stream = 0);

/// @brief Zero all force vectors on GPU (whole-system).
void device_zero_forces(ForceVec* d_forces, i32 natoms,
                        cudaStream_t stream = 0);

}  // namespace tdmd::integrator
