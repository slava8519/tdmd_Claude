// SPDX-License-Identifier: Apache-2.0
// lammps_data_reader.cpp — LAMMPS data file parser implementation.

#include "lammps_data_reader.hpp"

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "../core/error.hpp"

namespace tdmd::io {
namespace {

/// Trim leading/trailing whitespace.
std::string trim(const std::string& s) {
  auto start = s.find_first_not_of(" \t\r\n");
  if (start == std::string::npos) return "";
  auto end = s.find_last_not_of(" \t\r\n");
  return s.substr(start, end - start + 1);
}

/// Check if line starts with the given keyword (after trimming).
bool starts_with(const std::string& line, const std::string& prefix) {
  return line.compare(0, prefix.size(), prefix) == 0;
}

}  // namespace

SystemState read_lammps_data(const std::string& filename) {
  std::ifstream file(filename);
  if (!file.is_open()) {
    TDMD_THROW("cannot open LAMMPS data file: " + filename);
  }

  SystemState state;
  i64 natoms = 0;
  i32 ntypes = 0;
  bool have_box_x = false;
  bool have_box_y = false;
  bool have_box_z = false;

  // First pass: read header lines until we hit a section keyword.
  std::string line;

  // Skip first line (comment/title).
  std::getline(file, line);

  while (std::getline(file, line)) {
    std::string trimmed = trim(line);
    if (trimmed.empty()) continue;

    // Check for section keywords.
    if (trimmed == "Atoms" || trimmed == "Velocities" ||
        trimmed == "Masses" || starts_with(trimmed, "Atoms ") ||
        starts_with(trimmed, "Pair Coeffs") ||
        starts_with(trimmed, "Bond Coeffs")) {
      break;
    }

    // Header lines: "N keyword..."
    if (trimmed.find("atoms") != std::string::npos) {
      std::istringstream iss(trimmed);
      iss >> natoms;
      if (iss.fail() || natoms <= 0) {
        TDMD_THROW("invalid atom count in header");
      }
      continue;
    }

    if (trimmed.find("atom types") != std::string::npos) {
      std::istringstream iss(trimmed);
      iss >> ntypes;
      if (iss.fail() || ntypes <= 0) {
        TDMD_THROW("invalid atom type count in header");
      }
      continue;
    }

    if (trimmed.find("xlo xhi") != std::string::npos) {
      std::istringstream iss(trimmed);
      iss >> state.box.lo.x >> state.box.hi.x;
      have_box_x = true;
      continue;
    }
    if (trimmed.find("ylo yhi") != std::string::npos) {
      std::istringstream iss(trimmed);
      iss >> state.box.lo.y >> state.box.hi.y;
      have_box_y = true;
      continue;
    }
    if (trimmed.find("zlo zhi") != std::string::npos) {
      std::istringstream iss(trimmed);
      iss >> state.box.lo.z >> state.box.hi.z;
      have_box_z = true;
      continue;
    }
  }

  if (natoms == 0) TDMD_THROW("no atoms in data file");
  if (!have_box_x || !have_box_y || !have_box_z)
    TDMD_THROW("incomplete box definition");

  state.resize(natoms);
  state.masses.resize(static_cast<std::size_t>(ntypes + 1), real{1});
  state.type_names.resize(static_cast<std::size_t>(ntypes + 1));

  // Parse sections. `line` already holds the first section keyword.
  bool atoms_read = false;
  bool velocities_read = false;

  auto parse_current_section = [&]() {
    std::string section = trim(line);

    // Strip style hint, e.g. "Atoms # atomic" -> "Atoms"
    auto hash_pos = section.find('#');
    if (hash_pos != std::string::npos) {
      section = trim(section.substr(0, hash_pos));
    }

    if (section == "Masses") {
      // Skip blank line after section keyword.
      std::getline(file, line);
      for (i32 t = 0; t < ntypes; ++t) {
        if (!std::getline(file, line)) TDMD_THROW("premature end in Masses");
        std::string trimmed_line = trim(line);
        if (trimmed_line.empty()) {
          --t;
          continue;
        }
        std::istringstream iss(trimmed_line);
        i32 type_id;
        real mass;
        iss >> type_id >> mass;
        if (iss.fail()) TDMD_THROW("bad Masses line");
        state.masses[static_cast<std::size_t>(type_id)] = mass;
      }
    } else if (section == "Atoms") {
      // Skip blank line.
      std::getline(file, line);
      for (i64 i = 0; i < natoms; ++i) {
        if (!std::getline(file, line)) TDMD_THROW("premature end in Atoms");
        std::string trimmed_line = trim(line);
        if (trimmed_line.empty()) {
          --i;
          continue;
        }
        std::istringstream iss(trimmed_line);
        i32 id, type;
        real x, y, z;
        iss >> id >> type >> x >> y >> z;
        if (iss.fail()) TDMD_THROW("bad Atoms line");

        // LAMMPS IDs are 1-based; store at index (id-1) for sorted order.
        auto idx = static_cast<std::size_t>(id - 1);
        state.ids[idx] = id;
        state.types[idx] = type;
        state.positions[idx] = {x, y, z};
      }
      atoms_read = true;
    } else if (section == "Velocities") {
      // Skip blank line.
      std::getline(file, line);
      for (i64 i = 0; i < natoms; ++i) {
        if (!std::getline(file, line)) TDMD_THROW("premature end in Velocities");
        std::string trimmed_line = trim(line);
        if (trimmed_line.empty()) {
          --i;
          continue;
        }
        std::istringstream iss(trimmed_line);
        i32 id;
        real vx, vy, vz;
        iss >> id >> vx >> vy >> vz;
        if (iss.fail()) TDMD_THROW("bad Velocities line");
        auto idx = static_cast<std::size_t>(id - 1);
        state.velocities[idx] = {vx, vy, vz};
      }
      velocities_read = true;
    }
    // Other sections are silently skipped.
  };

  // Process the section we already have in `line`.
  parse_current_section();

  // Read remaining sections.
  while (std::getline(file, line)) {
    std::string trimmed_sec = trim(line);
    if (trimmed_sec.empty()) continue;

    // Section keywords start with an uppercase letter (convention).
    if (!trimmed_sec.empty() && std::isupper(trimmed_sec[0])) {
      parse_current_section();
    }
  }

  if (!atoms_read) TDMD_THROW("no Atoms section found");
  (void)velocities_read;

  return state;
}

}  // namespace tdmd::io
