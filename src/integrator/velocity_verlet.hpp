// SPDX-License-Identifier: Apache-2.0
// velocity_verlet.hpp — velocity-Verlet NVE integrator.
//
// Algorithm per step:
//   1. half-kick:  v(t+dt/2) = v(t) + (dt/2) * F(t) / m
//   2. drift:      r(t+dt) = r(t) + dt * v(t+dt/2)
//   3. [caller computes F(t+dt)]
//   4. half-kick:  v(t+dt) = v(t+dt/2) + (dt/2) * F(t+dt) / m
#pragma once

#include "../core/system_state.hpp"
#include "../core/types.hpp"

namespace tdmd::integrator {

/// @brief Velocity-Verlet NVE integrator.
///
/// LAMMPS metal units: force in eV/A, mass in g/mol, time in ps.
/// Conversion: a = F / (m * mvv2e) where mvv2e = 1.0364269e-4.
class VelocityVerlet {
 public:
  /// @param dt Time step in picoseconds.
  explicit VelocityVerlet(real dt) : dt_(dt) {}

  /// Half-kick: update velocities using current forces (step 1 or step 4).
  void half_kick(SystemState& state) const;

  /// Drift: update positions using current velocities (step 2).
  /// Also wraps positions into box and advances step/time counters.
  void drift(SystemState& state) const;

  [[nodiscard]] real dt() const noexcept { return dt_; }

 private:
  real dt_;
};

}  // namespace tdmd::integrator
