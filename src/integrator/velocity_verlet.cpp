// SPDX-License-Identifier: Apache-2.0
// velocity_verlet.cpp — velocity-Verlet integrator (CPU).

#include "velocity_verlet.hpp"

#include "../core/constants.hpp"
#include "../core/math.hpp"

namespace tdmd::integrator {

void VelocityVerlet::half_kick(SystemState& state) const {
  const real half_dt = real{0.5} * dt_;

  for (i64 i = 0; i < state.natoms; ++i) {
    auto si = static_cast<std::size_t>(i);
    const i32 type = state.types[si];
    const real mass = state.masses[static_cast<std::size_t>(type)];

    // dt_force = dt / (2 * mass * mvv2e)
    const real factor = half_dt / (mass * kMvv2e);

    state.velocities[si] += factor * state.forces[si];
  }
}

void VelocityVerlet::drift(SystemState& state) const {
  const Vec3D box_size = state.box.size();

  for (i64 i = 0; i < state.natoms; ++i) {
    auto si = static_cast<std::size_t>(i);
    state.positions[si] += dt_ * state.velocities[si];

    // Wrap into box.
    state.positions[si] =
        wrap_position(state.positions[si], state.box.lo, box_size,
                      state.box.periodic);
  }

  state.step += 1;
  state.time += dt_;
}

}  // namespace tdmd::integrator
