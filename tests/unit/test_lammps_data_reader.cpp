// SPDX-License-Identifier: Apache-2.0
// test_lammps_data_reader.cpp — unit tests for io/lammps_data_reader.
#include <gtest/gtest.h>

#include "core/error.hpp"
#include "io/lammps_data_reader.hpp"

// Path to test data, set by CMake define.
#ifndef TDMD_TEST_DATA_DIR
#error "TDMD_TEST_DATA_DIR must be defined"
#endif

static std::string data_dir() { return TDMD_TEST_DATA_DIR; }

TEST(LammpsDataReader, Basic4Atoms) {
  auto state = tdmd::io::read_lammps_data(data_dir() + "/4atoms.data");

  EXPECT_EQ(state.natoms, 4);

  // Box
  EXPECT_DOUBLE_EQ(state.box.lo.x, 0.0);
  EXPECT_DOUBLE_EQ(state.box.hi.x, 10.0);
  EXPECT_DOUBLE_EQ(state.box.lo.y, 0.0);
  EXPECT_DOUBLE_EQ(state.box.hi.y, 10.0);

  // Masses (1-indexed)
  EXPECT_NEAR(state.masses[1], 26.9815, 1e-4);
  EXPECT_NEAR(state.masses[2], 63.546, 1e-3);

  // Atom 1 (index 0): id=1, type=1, pos=(1,2,3), vel=(0.1,0.2,0.3)
  EXPECT_EQ(state.ids[0], 1);
  EXPECT_EQ(state.types[0], 1);
  EXPECT_DOUBLE_EQ(state.positions[0].x, 1.0);
  EXPECT_DOUBLE_EQ(state.positions[0].y, 2.0);
  EXPECT_DOUBLE_EQ(state.positions[0].z, 3.0);
  EXPECT_DOUBLE_EQ(state.velocities[0].x, 0.1);
  EXPECT_DOUBLE_EQ(state.velocities[0].y, 0.2);
  EXPECT_DOUBLE_EQ(state.velocities[0].z, 0.3);

  // Atom 3 (index 2): id=3, type=2, pos=(7,8,9)
  EXPECT_EQ(state.ids[2], 3);
  EXPECT_EQ(state.types[2], 2);
  EXPECT_DOUBLE_EQ(state.positions[2].x, 7.0);
  EXPECT_DOUBLE_EQ(state.positions[2].y, 8.0);
  EXPECT_DOUBLE_EQ(state.positions[2].z, 9.0);

  // Atom 4 velocity
  EXPECT_DOUBLE_EQ(state.velocities[3].x, 1.0);
  EXPECT_DOUBLE_EQ(state.velocities[3].y, 1.1);
  EXPECT_DOUBLE_EQ(state.velocities[3].z, 1.2);
}

TEST(LammpsDataReader, FileNotFound) {
  EXPECT_THROW(tdmd::io::read_lammps_data("nonexistent.data"), tdmd::Error);
}
