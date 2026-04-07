// SPDX-License-Identifier: Apache-2.0
// device_system_state.cuh — GPU-resident mirror of SystemState.
#pragma once

#include <cstddef>

#include "box.hpp"
#include "device_buffer.cuh"
#include "system_state.hpp"
#include "types.hpp"

namespace tdmd {

/// @brief GPU-resident simulation state.  SoA layout mirrors SystemState.
///
/// Owns device memory via DeviceBuffer.  Upload/download copy between
/// a host SystemState and this object.  Move-only (inherits from DeviceBuffer).
struct DeviceSystemState {
  i64 natoms{0};
  Box box;

  DeviceBuffer<Vec3> positions;
  DeviceBuffer<Vec3> velocities;
  DeviceBuffer<Vec3> forces;
  DeviceBuffer<i32>  types;
  DeviceBuffer<i32>  ids;
  DeviceBuffer<real> masses;  // per-type, length = ntypes

  /// @brief Allocate device arrays for n atoms (and ntypes types).
  void resize(i64 n, i32 ntypes = 0);

  /// @brief Upload host SystemState to device.
  void upload(const SystemState& host);

  /// @brief Download device arrays back to host SystemState.
  void download(SystemState& host) const;

  /// @brief Zero force array on device.
  void zero_forces();
};

}  // namespace tdmd
