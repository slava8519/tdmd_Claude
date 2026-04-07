// SPDX-License-Identifier: Apache-2.0
// test_neighbor_list.cpp — unit tests for neighbors/NeighborList.
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <set>
#include <vector>

#include "core/math.hpp"
#include "core/types.hpp"
#include "neighbors/neighbor_list.hpp"

using namespace tdmd;
using namespace tdmd::neighbors;

// Build a small FCC unit cell for testing: 4 atoms per unit cell.
static void make_fcc(real a, i32 nx, i32 ny, i32 nz,
                     std::vector<Vec3>& positions, Box& box) {
  // FCC basis
  Vec3 basis[4] = {
      {0, 0, 0},
      {real{0.5} * a, real{0.5} * a, 0},
      {real{0.5} * a, 0, real{0.5} * a},
      {0, real{0.5} * a, real{0.5} * a},
  };

  positions.clear();
  for (i32 iz = 0; iz < nz; ++iz) {
    for (i32 iy = 0; iy < ny; ++iy) {
      for (i32 ix = 0; ix < nx; ++ix) {
        for (auto& b : basis) {
          positions.push_back(Vec3{
              static_cast<real>(ix) * a + b.x,
              static_cast<real>(iy) * a + b.y,
              static_cast<real>(iz) * a + b.z,
          });
        }
      }
    }
  }

  box.lo = {0, 0, 0};
  box.hi = {static_cast<real>(nx) * a, static_cast<real>(ny) * a,
            static_cast<real>(nz) * a};
  box.periodic = {true, true, true};
}

TEST(NeighborList, NoLossNoGain) {
  // FCC Cu: a=3.615 A, cutoff=5.5 A. Each atom should have 12 nearest neighbors.
  std::vector<Vec3> pos;
  Box box;
  make_fcc(real{3.615}, 3, 3, 3, pos, box);
  auto natoms = static_cast<i64>(pos.size());

  NeighborList nlist;
  nlist.build(pos.data(), natoms, box, real{5.5}, real{0.5});

  const real r_list_sq = real{6.0} * real{6.0};  // (5.5 + 0.5)^2
  const Vec3 box_size = box.size();

  // Verify: every pair in list is within r_list, and no missing pair.
  // Collect all pairs from list.
  std::set<std::pair<i32, i32>> listed_pairs;
  for (i64 i = 0; i < natoms; ++i) {
    i32 cnt = nlist.count(i);
    const i32* nbrs = nlist.neighbors_of(i);
    for (i32 k = 0; k < cnt; ++k) {
      i32 j = nbrs[k];
      EXPECT_GT(j, static_cast<i32>(i)) << "half-list violated: j <= i";
      listed_pairs.insert({static_cast<i32>(i), j});

      // Verify distance.
      Vec3 delta = pos[static_cast<std::size_t>(j)] -
                   pos[static_cast<std::size_t>(i)];
      delta = minimum_image(delta, box_size, box.periodic);
      real r2 = length_sq(delta);
      EXPECT_LT(r2, r_list_sq + real{1e-6}) << "pair beyond r_list in list";
    }
  }

  // Brute-force: find all pairs within r_list.
  i32 brute_count = 0;
  for (i64 i = 0; i < natoms; ++i) {
    for (i64 j = i + 1; j < natoms; ++j) {
      Vec3 delta = pos[static_cast<std::size_t>(j)] -
                   pos[static_cast<std::size_t>(i)];
      delta = minimum_image(delta, box_size, box.periodic);
      real r2 = length_sq(delta);
      if (r2 < r_list_sq) {
        ++brute_count;
        EXPECT_TRUE(listed_pairs.count(
            {static_cast<i32>(i), static_cast<i32>(j)}))
            << "missing pair (" << i << ", " << j << ") r=" << std::sqrt(r2);
      }
    }
  }

  EXPECT_EQ(static_cast<i32>(listed_pairs.size()), brute_count);
}

TEST(NeighborList, NeedsRebuild) {
  std::vector<Vec3> pos = {{0, 0, 0}, {3, 0, 0}};
  Box box;
  box.lo = {0, 0, 0};
  box.hi = {10, 10, 10};

  NeighborList nlist;
  nlist.build(pos.data(), 2, box, real{5.0}, real{1.0});

  // No movement -> no rebuild.
  EXPECT_FALSE(nlist.needs_rebuild(pos.data(), 2));

  // Move atom 0 by 0.4 A (< skin/2 = 0.5) -> no rebuild.
  pos[0].x = real{0.4};
  EXPECT_FALSE(nlist.needs_rebuild(pos.data(), 2));

  // Move atom 0 by 0.6 A total (> skin/2 = 0.5) -> rebuild needed.
  pos[0].x = real{0.6};
  EXPECT_TRUE(nlist.needs_rebuild(pos.data(), 2));
}
