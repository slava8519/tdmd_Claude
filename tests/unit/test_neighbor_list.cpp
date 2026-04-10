// SPDX-License-Identifier: Apache-2.0
// test_neighbor_list.cpp — unit tests for neighbors/NeighborList.
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <random>
#include <set>
#include <vector>

#include "core/math.hpp"
#include "core/types.hpp"
#include "neighbors/neighbor_list.hpp"

using namespace tdmd;
using namespace tdmd::neighbors;

// Build a small FCC unit cell for testing: 4 atoms per unit cell.
static void make_fcc(double a, i32 nx, i32 ny, i32 nz,
                     std::vector<PositionVec>& positions, Box& box) {
  // FCC basis
  PositionVec basis[4] = {
      {0, 0, 0},
      {0.5 * a, 0.5 * a, 0},
      {0.5 * a, 0, 0.5 * a},
      {0, 0.5 * a, 0.5 * a},
  };

  positions.clear();
  for (i32 iz = 0; iz < nz; ++iz) {
    for (i32 iy = 0; iy < ny; ++iy) {
      for (i32 ix = 0; ix < nx; ++ix) {
        for (auto& b : basis) {
          positions.push_back(PositionVec{
              static_cast<double>(ix) * a + b.x,
              static_cast<double>(iy) * a + b.y,
              static_cast<double>(iz) * a + b.z,
          });
        }
      }
    }
  }

  box.lo = {0, 0, 0};
  box.hi = {static_cast<double>(nx) * a, static_cast<double>(ny) * a,
            static_cast<double>(nz) * a};
  box.periodic = {true, true, true};
}

TEST(NeighborList, NoLossNoGain) {
  // FCC Cu: a=3.615 A, cutoff=5.5 A. Each atom should have 12 nearest neighbors.
  std::vector<PositionVec> pos;
  Box box;
  make_fcc(3.615, 3, 3, 3, pos, box);
  auto natoms = static_cast<i64>(pos.size());

  NeighborList nlist;
  nlist.build(pos.data(), natoms, box, real{5.5}, real{0.5});

  const real r_list_sq = real{6.0} * real{6.0};  // (5.5 + 0.5)^2
  const Vec3D box_size = box.size();

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
      PositionVec delta = pos[static_cast<std::size_t>(j)] -
                          pos[static_cast<std::size_t>(i)];
      delta = minimum_image(delta, box_size, box.periodic);
      double r2 = length_sq(delta);
      EXPECT_LT(r2, static_cast<double>(r_list_sq) + 1e-6)
          << "pair beyond r_list in list";
    }
  }

  // Brute-force: find all pairs within r_list.
  i32 brute_count = 0;
  for (i64 i = 0; i < natoms; ++i) {
    for (i64 j = i + 1; j < natoms; ++j) {
      PositionVec delta = pos[static_cast<std::size_t>(j)] -
                          pos[static_cast<std::size_t>(i)];
      delta = minimum_image(delta, box_size, box.periodic);
      double r2 = length_sq(delta);
      if (r2 < static_cast<double>(r_list_sq)) {
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
  std::vector<PositionVec> pos = {{0, 0, 0}, {3, 0, 0}};
  Box box;
  box.lo = {0, 0, 0};
  box.hi = {10, 10, 10};

  NeighborList nlist;
  nlist.build(pos.data(), 2, box, real{5.0}, real{1.0});

  // No movement -> no rebuild.
  EXPECT_FALSE(nlist.needs_rebuild(pos.data(), 2));

  // Move atom 0 by 0.4 A (< skin/2 = 0.5) -> no rebuild.
  pos[0].x = 0.4;
  EXPECT_FALSE(nlist.needs_rebuild(pos.data(), 2));

  // Move atom 0 by 0.6 A total (> skin/2 = 0.5) -> rebuild needed.
  pos[0].x = 0.6;
  EXPECT_TRUE(nlist.needs_rebuild(pos.data(), 2));
}

// VL-9: Random gas stress test.
//
// FCC is an easy case for neighbor lists — atoms live on a lattice, so cells
// are uniformly populated and no cell stores more than 4 atoms. Random gas
// stresses the opposite: uneven cell occupancy, empty cells adjacent to dense
// ones, and pair distances that sample the full [0, rlist] range uniformly.
// A parity failure here that does not appear in NoLossNoGain would point at
// cell-boundary logic or ghost-cell handling.
TEST(NeighborList, RandomGasBruteForceParity) {
  constexpr std::size_t kN = 200;
  constexpr double kBox = 20.0;
  constexpr double kCut = 4.0;
  constexpr double kSkin = 1.0;
  constexpr double kRlist = kCut + kSkin;
  constexpr double kRlistSq = kRlist * kRlist;

  std::mt19937 rng(12345);
  std::uniform_real_distribution<double> U(0.0, kBox);

  std::vector<PositionVec> pos(kN);
  for (auto& p : pos) {
    p = {U(rng), U(rng), U(rng)};
  }

  Box box;
  box.lo = {0, 0, 0};
  box.hi = {kBox, kBox, kBox};
  box.periodic = {true, true, true};

  NeighborList nlist;
  nlist.build(pos.data(), static_cast<i64>(kN), box, real{kCut}, real{kSkin});

  std::set<std::pair<i32, i32>> listed;
  for (i64 i = 0; i < static_cast<i64>(kN); ++i) {
    i32 cnt = nlist.count(i);
    const i32* nbrs = nlist.neighbors_of(i);
    for (i32 k = 0; k < cnt; ++k) {
      listed.insert({static_cast<i32>(i), nbrs[k]});
    }
  }

  // Brute force: every pair within rlist must be present.
  const Vec3D box_size = box.size();
  std::size_t brute = 0;
  for (i64 i = 0; i < static_cast<i64>(kN); ++i) {
    for (i64 j = i + 1; j < static_cast<i64>(kN); ++j) {
      PositionVec d = pos[static_cast<std::size_t>(j)] -
                      pos[static_cast<std::size_t>(i)];
      d = minimum_image(d, box_size, box.periodic);
      double r2 = length_sq(d);
      if (r2 < kRlistSq) {
        ++brute;
        EXPECT_TRUE(listed.count({static_cast<i32>(i), static_cast<i32>(j)}))
            << "missing random-gas pair (" << i << "," << j
            << ") r=" << std::sqrt(r2);
      }
    }
  }
  EXPECT_EQ(listed.size(), brute);
}

// VL-9: Precise half-skin boundary.
//
// The existing NeedsRebuild test uses 0.4 vs 0.6 A with skin=1.0, which leaves
// a 0.1 A margin on each side of the 0.5 A threshold. This test tightens the
// margin to 1% of half_skin on either side, and varies skin to catch any
// hardcoded constants. The rule being tested is LAMMPS' canonical "rebuild
// when any atom has moved > skin/2 since the last build" (see ADR 0008 and
// docs/04-development/lessons-learned.md for context on why half-skin is the
// right threshold rather than full skin).
TEST(NeighborList, HalfSkinPreciseBoundary) {
  constexpr real kSkin = 2.0;
  const double half_skin = 0.5 * static_cast<double>(kSkin);  // 1.0 A
  std::vector<PositionVec> pos = {{5, 5, 5}, {8, 5, 5}, {5, 8, 5}};
  Box box;
  box.lo = {0, 0, 0};
  box.hi = {20, 20, 20};
  box.periodic = {true, true, true};

  NeighborList nlist;
  nlist.build(pos.data(), 3, box, real{4.0}, kSkin);

  // Just under the boundary: 0.99 * half_skin from rest. Must not rebuild.
  pos[1].x = 8.0 + 0.99 * half_skin;
  EXPECT_FALSE(nlist.needs_rebuild(pos.data(), 3))
      << "spurious rebuild at 0.99 * half_skin (= " << (0.99 * half_skin) << ")";

  // Just over the boundary: 1.01 * half_skin. Must rebuild.
  pos[1].x = 8.0 + 1.01 * half_skin;
  EXPECT_TRUE(nlist.needs_rebuild(pos.data(), 3))
      << "missed rebuild at 1.01 * half_skin (= " << (1.01 * half_skin) << ")";
}

// VL-9: Boundary also holds with a different skin value.
//
// Guards against any place that might hardcode half_skin = 0.5 A rather than
// compute it from r_skin_ at runtime.
TEST(NeighborList, HalfSkinBoundaryWithDifferentSkin) {
  constexpr real kSkin = 0.8;  // unusual, not 1.0
  const double half_skin = 0.5 * static_cast<double>(kSkin);  // 0.4 A
  std::vector<PositionVec> pos = {{5, 5, 5}, {9, 5, 5}};
  Box box;
  box.lo = {0, 0, 0};
  box.hi = {20, 20, 20};
  box.periodic = {true, true, true};

  NeighborList nlist;
  nlist.build(pos.data(), 2, box, real{4.0}, kSkin);

  pos[0].y = 5.0 + 0.99 * half_skin;
  EXPECT_FALSE(nlist.needs_rebuild(pos.data(), 2));

  pos[0].y = 5.0 + 1.01 * half_skin;
  EXPECT_TRUE(nlist.needs_rebuild(pos.data(), 2));
}

// VL-9: Multi-atom concurrent motion.
//
// Single-atom drift is the easy case. This test moves MANY atoms each by less
// than half-skin and asserts no rebuild — then flips one of them just over the
// threshold and asserts the rebuild fires. The bug this catches: implementations
// that use `sum(displacements)` or `mean(displacements)` instead of
// `max(displacements)`, which would incorrectly trigger on the safe case.
TEST(NeighborList, MultiAtomMaxNotSum) {
  constexpr real kSkin = 1.0;
  constexpr double kHalfSkin = 0.5;
  constexpr std::size_t kN = 50;

  std::vector<PositionVec> pos(kN);
  for (std::size_t i = 0; i < kN; ++i) {
    pos[i] = {static_cast<double>(i) * 0.5 + 1.0, 5, 5};
  }

  Box box;
  box.lo = {0, 0, 0};
  box.hi = {kN * 0.6 + 2.0, 20, 20};
  box.periodic = {true, true, true};

  NeighborList nlist;
  nlist.build(pos.data(), static_cast<i64>(kN), box, real{2.0}, kSkin);

  // Move every atom by 0.9 * half_skin. Sum of displacements is huge
  // (50 * 0.45 = 22.5 A total), but max per-atom is 0.45 < half_skin.
  // Rebuild must NOT fire.
  for (auto& p : pos) {
    p.y += 0.9 * kHalfSkin;
  }
  EXPECT_FALSE(nlist.needs_rebuild(pos.data(), static_cast<i64>(kN)))
      << "sum-not-max bug: rebuild fired when all atoms moved < half_skin";

  // Now push exactly one atom over the threshold. Rebuild MUST fire.
  pos[kN / 2].z += 1.1 * kHalfSkin;
  EXPECT_TRUE(nlist.needs_rebuild(pos.data(), static_cast<i64>(kN)))
      << "max-over-threshold not detected";
}

// VL-9: Fast-atom catch — one atom crosses multiple cell boundaries.
//
// An atom that moves far enough to cross multiple cells in one timestep is the
// pathological case the half-skin rule is designed to catch. This test verifies
// the trigger fires for such a motion (not just the 0.51 * half_skin boundary
// case), which exercises the full displacement-tracking path rather than the
// near-threshold comparison.
TEST(NeighborList, FastAtomTriggersRebuild) {
  constexpr real kCut = 3.0;
  constexpr real kSkin = 1.0;
  std::vector<PositionVec> pos = {
      {5, 5, 5}, {7, 5, 5}, {5, 7, 5}, {5, 5, 7},
  };
  Box box;
  box.lo = {0, 0, 0};
  box.hi = {20, 20, 20};
  box.periodic = {true, true, true};

  NeighborList nlist;
  nlist.build(pos.data(), 4, box, kCut, kSkin);

  // Move atom 0 by 3 A — ten times half_skin. Must rebuild.
  pos[0] = {8, 5, 5};
  EXPECT_TRUE(nlist.needs_rebuild(pos.data(), 4))
      << "fast atom (3 A motion, half_skin=0.5) must trigger rebuild";

  // After a rebuild with the new positions, the new neighbor set must be
  // correct (no lingering references to the old positions).
  nlist.build(pos.data(), 4, box, kCut, kSkin);
  const Vec3D bs = box.size();
  std::set<std::pair<i32, i32>> listed;
  for (i64 i = 0; i < 4; ++i) {
    i32 cnt = nlist.count(i);
    const i32* nbrs = nlist.neighbors_of(i);
    for (i32 k = 0; k < cnt; ++k) {
      listed.insert({static_cast<i32>(i), nbrs[k]});
    }
  }

  // Brute force verification against new positions.
  const real rlist = kCut + kSkin;
  const double rlist_sq = static_cast<double>(rlist) * static_cast<double>(rlist);
  for (i64 i = 0; i < 4; ++i) {
    for (i64 j = i + 1; j < 4; ++j) {
      PositionVec d = pos[static_cast<std::size_t>(j)] -
                      pos[static_cast<std::size_t>(i)];
      d = minimum_image(d, bs, box.periodic);
      double r2 = length_sq(d);
      if (r2 < rlist_sq) {
        EXPECT_TRUE(listed.count({static_cast<i32>(i), static_cast<i32>(j)}))
            << "post-rebuild missing pair (" << i << "," << j << ")";
      }
    }
  }
}
