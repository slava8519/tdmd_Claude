// SPDX-License-Identifier: Apache-2.0
// device_velocity_verlet.cuh — GPU velocity-Verlet NVE integrator.
#pragma once

#include "../core/box.hpp"
#include "../core/types.hpp"

namespace tdmd::integrator {

/// @brief GPU velocity-Verlet half-kick: v += (dt/2) * F / (m * mvv2e).
void device_half_kick(Vec3* d_velocities, const Vec3* d_forces,
                      const i32* d_types, const real* d_masses, i32 natoms,
                      real dt);

/// @brief GPU velocity-Verlet drift: r += dt * v, then wrap into box.
void device_drift(Vec3* d_positions, const Vec3* d_velocities, i32 natoms,
                  real dt, const Box& box);

}  // namespace tdmd::integrator
