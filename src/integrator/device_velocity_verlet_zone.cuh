// SPDX-License-Identifier: Apache-2.0
// device_velocity_verlet_zone.cuh — Per-zone GPU integration kernels.
#pragma once

#include <cuda_runtime.h>

#include "../core/box.hpp"
#include "../core/types.hpp"

namespace tdmd::integrator {

/// @brief Per-zone half-kick on GPU.
void device_half_kick_zone(VelocityVec* d_velocities, const ForceVec* d_forces,
                           const i32* d_types, const real* d_masses,
                           i32 first_atom, i32 atom_count, real dt,
                           cudaStream_t stream);

/// @brief Per-zone drift on GPU.
void device_drift_zone(PositionVec* d_positions,
                       const VelocityVec* d_velocities, i32 first_atom,
                       i32 atom_count, real dt, const Box& box,
                       cudaStream_t stream);

/// @brief Zero forces for a zone on GPU.
void device_zero_forces_zone(ForceVec* d_forces, i32 first_atom,
                             i32 atom_count, cudaStream_t stream);

}  // namespace tdmd::integrator
