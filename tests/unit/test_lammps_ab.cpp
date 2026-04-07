// SPDX-License-Identifier: Apache-2.0
// test_lammps_ab.cpp — A/B force comparison: TDMD vs LAMMPS reference.
//
// These tests read pre-computed LAMMPS force dumps and compare against
// TDMD's force computation on the same input.
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <vector>

#include "core/math.hpp"
#include "core/types.hpp"
#include "io/dump_reader.hpp"
#include "io/lammps_data_reader.hpp"
#include "neighbors/neighbor_list.hpp"
#include "potentials/eam_alloy.hpp"
#include "potentials/force_compute.hpp"
#include "potentials/morse.hpp"

#ifndef TDMD_TEST_DATA_DIR
#error "TDMD_TEST_DATA_DIR must be defined"
#endif

static std::string data_dir() { return TDMD_TEST_DATA_DIR; }

using namespace tdmd;
using namespace tdmd::potentials;
using namespace tdmd::neighbors;

/// Compare TDMD forces against LAMMPS reference dump.
/// Returns max absolute force-component error.
static real compare_forces(const SystemState& state,
                           const std::vector<io::DumpAtom>& ref) {
  EXPECT_EQ(state.natoms, static_cast<i64>(ref.size()));

  real max_err = 0;
  for (i64 i = 0; i < state.natoms; ++i) {
    auto si = static_cast<std::size_t>(i);
    // Match by global id.
    i32 id = state.ids[si];
    // Find in reference (sorted by id, so ref[id-1]).
    auto ridx = static_cast<std::size_t>(id - 1);
    EXPECT_EQ(ref[ridx].id, id);

    real ex = std::abs(state.forces[si].x - ref[ridx].force.x);
    real ey = std::abs(state.forces[si].y - ref[ridx].force.y);
    real ez = std::abs(state.forces[si].z - ref[ridx].force.z);

    max_err = std::max({max_err, ex, ey, ez});
  }
  return max_err;
}

TEST(LammpsAB, MorseRun0ForceMatch) {
  // Read the same 256-atom Cu FCC data file.
  auto state = io::read_lammps_data(data_dir() + "/cu_fcc_256.data");

  // Compute forces with TDMD Morse. Cutoff 6.0 < L/2=7.23 for valid PBC.
  MorsePair pot(0.3429, 1.3588, 2.866, 6.0);
  NeighborList nlist;
  nlist.build(state.positions.data(), state.natoms, state.box,
              pot.cutoff(), real{1.0});
  (void)compute_pair_forces(state, nlist, pot);

  // Read LAMMPS reference.
  auto ref = io::read_lammps_dump(data_dir() + "/reference/forces_morse.dump");

  real max_err = compare_forces(state, ref);

  // On a perfect FCC lattice, all forces are ~0 (machine epsilon).
  // Both TDMD and LAMMPS should agree to < 1e-6.
  EXPECT_LT(max_err, 1e-6)
      << "Max force-component error: " << max_err;
}

TEST(LammpsAB, EamRun0ForceMatch) {
  auto state = io::read_lammps_data(data_dir() + "/cu_fcc_256.data");

  EamAlloy eam;
  eam.read_setfl(data_dir() + "/Cu_mishin1.eam.alloy");

  NeighborList nlist;
  nlist.build(state.positions.data(), state.natoms, state.box,
              eam.cutoff(), real{1.0});
  (void)eam.compute_forces(state, nlist);

  auto ref = io::read_lammps_dump(data_dir() + "/reference/forces_eam.dump");

  real max_err = compare_forces(state, ref);

  EXPECT_LT(max_err, 1e-6)
      << "Max force-component error: " << max_err;
}
