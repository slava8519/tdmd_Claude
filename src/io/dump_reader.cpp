// SPDX-License-Identifier: Apache-2.0
// dump_reader.cpp — LAMMPS custom dump reader.

#include "dump_reader.hpp"

#include <algorithm>
#include <fstream>
#include <sstream>
#include <string>

#include "../core/error.hpp"

namespace tdmd::io {

std::vector<DumpAtom> read_lammps_dump(const std::string& filename) {
  std::ifstream file(filename);
  if (!file.is_open()) {
    TDMD_THROW("cannot open dump file: " + filename);
  }

  std::vector<DumpAtom> atoms;
  std::string line;
  i64 natoms = 0;

  // Parse header.
  while (std::getline(file, line)) {
    if (line.find("ITEM: NUMBER OF ATOMS") != std::string::npos) {
      std::getline(file, line);
      natoms = std::stoll(line);
    }
    if (line.find("ITEM: ATOMS") != std::string::npos) {
      break;
    }
  }

  if (natoms == 0) TDMD_THROW("no atoms in dump file");

  atoms.reserve(static_cast<std::size_t>(natoms));
  for (i64 i = 0; i < natoms; ++i) {
    if (!std::getline(file, line)) TDMD_THROW("premature end of dump");
    std::istringstream iss(line);
    DumpAtom a{};
    iss >> a.id >> a.type >> a.pos.x >> a.pos.y >> a.pos.z
        >> a.force.x >> a.force.y >> a.force.z;
    if (iss.fail()) TDMD_THROW("bad dump line");
    atoms.push_back(a);
  }

  // Sort by id.
  std::sort(atoms.begin(), atoms.end(),
            [](const DumpAtom& a, const DumpAtom& b) { return a.id < b.id; });

  return atoms;
}

}  // namespace tdmd::io
