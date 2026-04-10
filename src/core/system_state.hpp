// SPDX-License-Identifier: Apache-2.0
// system_state.hpp — the canonical simulation state.
//
// Layout: SoA (Structure of Arrays). Same shape on host and device.
// M0: host-only stub. M2: device-resident with DeviceBuffer.
#pragma once

#include <string>
#include <vector>

#include "box.hpp"
#include "types.hpp"

namespace tdmd {

/// The complete state of an MD simulation. One instance per rank.
///
/// At M0 / M1 this is host-only (std::vector). At M2 we add device mirrors and
/// a DeviceBuffer wrapper. The interface should not change.
struct SystemState {
  i64 natoms{0};
  Box box;

  // Per-atom arrays (SoA, length = natoms).
  // Per ADR 0007: positions and velocities always stored in double; forces
  // are float in mixed mode and double in fp64 mode.
  std::vector<PositionVec> positions;   // Vec3D in both modes
  std::vector<VelocityVec> velocities;  // Vec3D in both modes
  std::vector<ForceVec>    forces;      // Vec3F (mixed) / Vec3D (fp64)
  std::vector<i32>         types;
  std::vector<i32>         ids;  // stable global IDs, used for LAMMPS A/B comparisons

  // Per-type, host-side.
  std::vector<real>        masses;
  std::vector<std::string> type_names;

  // Simulation clock.
  i64  step{0};
  real time{0};

  /// Resize all per-atom arrays consistently.
  void resize(i64 n) {
    natoms = n;
    positions.resize(static_cast<std::size_t>(n));
    velocities.resize(static_cast<std::size_t>(n));
    forces.resize(static_cast<std::size_t>(n));
    types.resize(static_cast<std::size_t>(n));
    ids.resize(static_cast<std::size_t>(n));
  }
};

}  // namespace tdmd
