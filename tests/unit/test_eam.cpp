// SPDX-License-Identifier: Apache-2.0
// test_eam.cpp — tests for EAM/alloy potential.
#include <gtest/gtest.h>

#include <cmath>

#include "core/math.hpp"
#include "core/types.hpp"
#include "io/lammps_data_reader.hpp"
#include "neighbors/neighbor_list.hpp"
#include "potentials/eam_alloy.hpp"

#ifndef TDMD_TEST_DATA_DIR
#error "TDMD_TEST_DATA_DIR must be defined"
#endif

static std::string data_dir() { return TDMD_TEST_DATA_DIR; }

using namespace tdmd;
using namespace tdmd::potentials;

TEST(EamAlloy, ReadSetfl) {
  EamAlloy eam;
  ASSERT_NO_THROW(eam.read_setfl(data_dir() + "/Cu_mishin1.eam.alloy"));
  EXPECT_EQ(eam.ntypes(), 1);
  EXPECT_NEAR(eam.cutoff(), 5.50679, 0.001);
}

TEST(EamAlloy, TwoAtomForces) {
  // Two Cu atoms at distance 2.556 A (FCC nearest-neighbor).
  // Forces should be equal and opposite, nonzero.
  EamAlloy eam;
  eam.read_setfl(data_dir() + "/Cu_mishin1.eam.alloy");

  SystemState state;
  state.resize(2);
  state.box.lo = {0, 0, 0};
  state.box.hi = {20, 20, 20};
  state.box.periodic = {true, true, true};
  state.positions[0] = {8.0, 10.0, 10.0};
  state.positions[1] = {10.556, 10.0, 10.0};
  state.types[0] = 1;
  state.types[1] = 1;
  state.ids[0] = 1;
  state.ids[1] = 2;
  state.masses = {0, 63.546};

  neighbors::NeighborList nlist;
  nlist.build(state.positions.data(), state.natoms, state.box,
              eam.cutoff(), real{0.5});

  real pe = eam.compute_forces(state, nlist);

  // Energy should be negative (bound state).
  EXPECT_LT(pe, 0.0);

  // Newton 3rd law.
  EXPECT_NEAR(state.forces[0].x + state.forces[1].x, 0.0, 1e-10);
  EXPECT_NEAR(state.forces[0].y + state.forces[1].y, 0.0, 1e-10);
  EXPECT_NEAR(state.forces[0].z + state.forces[1].z, 0.0, 1e-10);

  // Forces should be nonzero (not at exact equilibrium).
  EXPECT_GT(std::abs(state.forces[0].x), 1e-6);
}

TEST(EamAlloy, FccBulkEnergy) {
  // 256-atom Cu FCC at a=3.615. Cohesive energy should be ~-3.54 eV/atom.
  EamAlloy eam;
  eam.read_setfl(data_dir() + "/Cu_mishin1.eam.alloy");

  auto state = io::read_lammps_data(data_dir() + "/cu_fcc_256.data");

  neighbors::NeighborList nlist;
  nlist.build(state.positions.data(), state.natoms, state.box,
              eam.cutoff(), real{0.5});

  real pe = eam.compute_forces(state, nlist);
  real pe_per_atom = pe / static_cast<real>(state.natoms);

  // Mishin Cu EAM: cohesive energy ~-3.54 eV/atom at a=3.615.
  EXPECT_NEAR(pe_per_atom, -3.54, 0.1)
      << "PE/atom = " << pe_per_atom;
}
