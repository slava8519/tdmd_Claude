// SPDX-License-Identifier: Apache-2.0
// test_zone_partition.cpp — zone partitioning tests.
#include <gtest/gtest.h>

#include <algorithm>
#include <set>
#include <string>
#include <vector>

#include "core/types.hpp"
#include "domain/zone_partition.hpp"
#include "io/lammps_data_reader.hpp"

using namespace tdmd;

TEST(ZonePartition, EveryAtomInExactlyOneZone) {
  std::string data_dir = TDMD_TEST_DATA_DIR;
  SystemState state = io::read_lammps_data(data_dir + "/cu_fcc_256.data");

  domain::ZonePartition zp;
  zp.build(state.box, real{6.0});
  zp.assign_atoms(state.positions.data(), state.velocities.data(),
                  state.forces.data(), state.types.data(), state.ids.data(),
                  state.natoms);

  // Every atom id should appear exactly once.
  std::set<i32> seen;
  i32 total = 0;
  for (auto& z : zp.zones()) {
    total += z.natoms_in_zone;
    for (i32 i = 0; i < z.natoms_in_zone; ++i) {
      auto idx = static_cast<std::size_t>(z.atom_offset + i);
      i32 id = state.ids[idx];
      EXPECT_EQ(seen.count(id), 0u) << "atom " << id << " in multiple zones";
      seen.insert(id);
    }
  }
  EXPECT_EQ(total, static_cast<i32>(state.natoms));
  EXPECT_EQ(static_cast<i32>(seen.size()), static_cast<i32>(state.natoms));
}

TEST(ZonePartition, AtomsInCorrectZone) {
  std::string data_dir = TDMD_TEST_DATA_DIR;
  SystemState state = io::read_lammps_data(data_dir + "/cu_fcc_256.data");

  domain::ZonePartition zp;
  zp.build(state.box, real{6.0});
  zp.assign_atoms(state.positions.data(), state.velocities.data(),
                  state.forces.data(), state.types.data(), state.ids.data(),
                  state.natoms);

  // Each atom's x position should be within its zone's bbox.
  for (auto& z : zp.zones()) {
    for (i32 i = 0; i < z.natoms_in_zone; ++i) {
      auto idx = static_cast<std::size_t>(z.atom_offset + i);
      real x = state.positions[idx].x;
      EXPECT_GE(x, z.bbox.lo.x)
          << "atom at x=" << x << " below zone " << z.id << " lo=" << z.bbox.lo.x;
      EXPECT_LT(x, z.bbox.hi.x + real{1e-10})
          << "atom at x=" << x << " above zone " << z.id << " hi=" << z.bbox.hi.x;
    }
  }
}

TEST(ZonePartition, ZoneNeighbors) {
  Box box;
  box.lo = {0, 0, 0};
  box.hi = {20, 20, 20};
  box.periodic = {true, true, true};

  domain::ZonePartition zp;
  zp.build(box, real{5.0}, 4);  // zone_width = 5.0
  zp.build_zone_neighbors(real{6.5});  // r_list = 6.5 → span = ceil(6.5/5.0) = 2

  // With 4 zones and span=2, each zone neighbors all 4 zones (PBC).
  for (i32 z = 0; z < 4; ++z) {
    auto& nbrs = zp.zone_neighbors(z);
    EXPECT_EQ(static_cast<i32>(nbrs.size()), 4)
        << "zone " << z << " should have all 4 zones as neighbors";
  }

  // With span=1 (r_list < zone_width), only self + 2 neighbors.
  zp.build_zone_neighbors(real{4.0});  // span = ceil(4.0/5.0) = 1
  for (i32 z = 0; z < 4; ++z) {
    auto& nbrs = zp.zone_neighbors(z);
    EXPECT_EQ(static_cast<i32>(nbrs.size()), 3)
        << "zone " << z << " should have 3 neighbors (self + 2 adjacent)";
  }
}
