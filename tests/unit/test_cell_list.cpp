// SPDX-License-Identifier: Apache-2.0
// test_cell_list.cpp — unit tests for domain/CellList.
#include <gtest/gtest.h>

#include <algorithm>
#include <set>
#include <vector>

#include "core/types.hpp"
#include "domain/cell_list.hpp"

using namespace tdmd;
using namespace tdmd::domain;

TEST(CellList, EveryAtomInExactlyOneCell) {
  // 8 atoms at FCC-like positions in a 10x10x10 box.
  Box box;
  box.lo = {0, 0, 0};
  box.hi = {10, 10, 10};

  std::vector<PositionVec> pos = {
      {1, 1, 1}, {5, 1, 1}, {1, 5, 1}, {1, 1, 5},
      {5, 5, 1}, {5, 1, 5}, {1, 5, 5}, {5, 5, 5},
  };

  CellList cl;
  cl.build(pos.data(), static_cast<i64>(pos.size()), box, real{3.0});

  // Collect all atoms from all cells. Each atom should appear exactly once.
  std::set<i32> seen;
  i32 total = 0;
  for (i32 c = 0; c < cl.ncells_total(); ++c) {
    const i32* atoms = cl.atoms_in_cell(c);
    i32 cnt = cl.count(c);
    for (i32 k = 0; k < cnt; ++k) {
      EXPECT_TRUE(seen.insert(atoms[k]).second) << "atom " << atoms[k] << " in multiple cells";
      ++total;
    }
  }
  EXPECT_EQ(total, 8);
  EXPECT_EQ(static_cast<i32>(seen.size()), 8);
}

TEST(CellList, CellSizeAtLeastRlist) {
  Box box;
  box.lo = {0, 0, 0};
  box.hi = {20, 20, 20};

  std::vector<PositionVec> pos = {{1, 1, 1}, {10, 10, 10}};

  CellList cl;
  cl.build(pos.data(), 2, box, real{5.5});

  // Cell side should be >= r_list (may be slightly larger due to integer division).
  auto cs = cl.cell_size();
  EXPECT_GE(cs.x, real{5.5} - real{0.01});
  EXPECT_GE(cs.y, real{5.5} - real{0.01});
  EXPECT_GE(cs.z, real{5.5} - real{0.01});
}

TEST(CellList, MinimumThreeCellsPerAxis) {
  // Very small r_list relative to box — but box is small.
  // With box=6 and r_list=5, floor(6/5)=1 which gets clamped to 3.
  Box box;
  box.lo = {0, 0, 0};
  box.hi = {6, 6, 6};

  std::vector<PositionVec> pos = {{1, 1, 1}, {3, 3, 3}};

  CellList cl;
  cl.build(pos.data(), 2, box, real{5.0});

  EXPECT_GE(cl.ncells_x(), 3);
  EXPECT_GE(cl.ncells_y(), 3);
  EXPECT_GE(cl.ncells_z(), 3);
}
