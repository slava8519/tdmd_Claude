// SPDX-License-Identifier: Apache-2.0
// velocity_verlet.cpp — velocity-Verlet integrator (CPU).

#include "velocity_verlet.hpp"

#include "../core/constants.hpp"
#include "../core/math.hpp"

namespace tdmd::integrator {

void VelocityVerlet::half_kick(SystemState& state) const {
  // ADR 0007 integrator math contract: all arithmetic in double regardless
  // of build mode. Forces (float in mixed) are promoted on read.
  const double half_dt = static_cast<double>(dt_) * 0.5;

  for (i64 i = 0; i < state.natoms; ++i) {
    auto si = static_cast<std::size_t>(i);
    const i32 type = state.types[si];
    const double mass = static_cast<double>(state.masses[static_cast<std::size_t>(type)]);

    const double factor = half_dt / (mass * static_cast<double>(kMvv2e));

    state.velocities[si].x += factor * static_cast<double>(state.forces[si].x);
    state.velocities[si].y += factor * static_cast<double>(state.forces[si].y);
    state.velocities[si].z += factor * static_cast<double>(state.forces[si].z);
  }
}

void VelocityVerlet::drift(SystemState& state) const {
  // ADR 0007 integrator math contract: position update in double.
  const double d_dt = static_cast<double>(dt_);
  const Vec3D box_size = state.box.size();

  for (i64 i = 0; i < state.natoms; ++i) {
    auto si = static_cast<std::size_t>(i);
    state.positions[si].x += d_dt * state.velocities[si].x;
    state.positions[si].y += d_dt * state.velocities[si].y;
    state.positions[si].z += d_dt * state.velocities[si].z;

    // Wrap into box (positions are Vec3D, box is Vec3D).
    state.positions[si] =
        wrap_position(state.positions[si], state.box.lo, box_size,
                      state.box.periodic);
  }

  state.step += 1;
  state.time += dt_;
}

}  // namespace tdmd::integrator
