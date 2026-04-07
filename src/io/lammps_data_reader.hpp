// SPDX-License-Identifier: Apache-2.0
// lammps_data_reader.hpp — parser for LAMMPS data files (atomic style, ortho box).
#pragma once

#include <string>

#include "../core/system_state.hpp"

namespace tdmd::io {

/// @brief Read a LAMMPS data file and populate a SystemState.
///
/// Supported subset (M1):
///   - units metal only (caller's responsibility)
///   - atom_style atomic (id type x y z)
///   - orthorhombic box
///   - optional Velocities section
///   - optional Masses section
///
/// @param filename Path to the LAMMPS data file.
/// @return Fully populated SystemState.
/// @throws tdmd::Error on parse failure.
[[nodiscard]] SystemState read_lammps_data(const std::string& filename);

}  // namespace tdmd::io
