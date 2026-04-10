// SPDX-License-Identifier: Apache-2.0
// dump_reader.hpp — reader for LAMMPS custom dump files (forces).
#pragma once

#include <string>
#include <vector>

#include "../core/types.hpp"

namespace tdmd::io {

/// Per-atom data from a LAMMPS dump: id, type, x, y, z, fx, fy, fz.
struct DumpAtom {
  i32 id;
  i32 type;
  PositionVec pos;
  ForceVec force;
};

/// @brief Read forces from a LAMMPS custom dump file.
/// Expects columns: id type x y z fx fy fz.
/// @return Vector of atoms sorted by id.
[[nodiscard]] std::vector<DumpAtom> read_lammps_dump(const std::string& filename);

}  // namespace tdmd::io
