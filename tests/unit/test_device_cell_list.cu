// SPDX-License-Identifier: Apache-2.0
// test_device_cell_list.cu — GPU cell list tests.
#include <gtest/gtest.h>

#include <vector>

#include "core/determinism.hpp"
#include "core/device_buffer.cuh"
#include "core/types.hpp"
#include "domain/cell_list.hpp"
#include "domain/device_cell_list.cuh"

using namespace tdmd;

TEST(DeviceCellList, EveryAtomInExactlyOneCell) {
  // Simple 8-atom FCC-like layout in a 10x10x10 box.
  constexpr i64 N = 8;
  Box box;
  box.lo = {0, 0, 0};
  box.hi = {10, 10, 10};
  box.periodic = {true, true, true};

  std::vector<PositionVec> h_pos = {{1, 1, 1}, {3, 3, 3}, {5, 5, 5}, {7, 7, 7},
                                    {2, 8, 1}, {9, 1, 9}, {4, 6, 2}, {8, 4, 6}};

  DeviceBuffer<PositionVec> d_pos(static_cast<std::size_t>(N));
  d_pos.copy_from_host(h_pos.data(), static_cast<std::size_t>(N));

  domain::DeviceCellList dcl;
  dcl.build(d_pos.data(), N, box, real{3.5});

  // Download cell data and verify.
  i32 ncells = dcl.ncells_total();
  std::vector<i32> h_counts(static_cast<std::size_t>(ncells));
  std::vector<i32> h_offsets(static_cast<std::size_t>(ncells));
  std::vector<i32> h_atoms(static_cast<std::size_t>(N));

  DeviceBuffer<i32> tmp_counts(static_cast<std::size_t>(ncells));
  // Copy from const device pointer via cudaMemcpy directly.
  TDMD_CUDA_CHECK(cudaMemcpy(h_counts.data(), dcl.d_cell_counts(),
                             static_cast<std::size_t>(ncells) * sizeof(i32),
                             cudaMemcpyDeviceToHost));
  TDMD_CUDA_CHECK(cudaMemcpy(h_offsets.data(), dcl.d_cell_offsets(),
                             static_cast<std::size_t>(ncells) * sizeof(i32),
                             cudaMemcpyDeviceToHost));
  TDMD_CUDA_CHECK(cudaMemcpy(h_atoms.data(), dcl.d_cell_atoms(),
                             static_cast<std::size_t>(N) * sizeof(i32),
                             cudaMemcpyDeviceToHost));

  // Sum of counts should equal N.
  i32 total = 0;
  for (i32 c = 0; c < ncells; ++c) {
    total += h_counts[static_cast<std::size_t>(c)];
  }
  EXPECT_EQ(total, N);

  // Every atom index [0..N) should appear exactly once.
  std::vector<int> seen(static_cast<std::size_t>(N), 0);
  for (i64 i = 0; i < N; ++i) {
    i32 idx = h_atoms[static_cast<std::size_t>(i)];
    ASSERT_GE(idx, 0);
    ASSERT_LT(idx, N);
    seen[static_cast<std::size_t>(idx)]++;
  }
  for (i64 i = 0; i < N; ++i) {
    EXPECT_EQ(seen[static_cast<std::size_t>(i)], 1)
        << "atom " << i << " not found exactly once";
  }
}

TEST(DeviceCellList, MatchesCPUCellList) {
  // Compare GPU cell list against CPU cell list for 256 atoms.
  constexpr i64 N = 256;
  Box box;
  box.lo = {0, 0, 0};
  box.hi = {14.46, 14.46, 14.46};
  box.periodic = {true, true, true};

  // Generate a regular grid of positions.
  std::vector<PositionVec> h_pos(static_cast<std::size_t>(N));
  double spacing = 14.46 / 8.0;  // ~6.3 atoms per axis = 6^3 < 256
  i32 idx = 0;
  for (i32 iz = 0; iz < 8 && idx < N; ++iz) {
    for (i32 iy = 0; iy < 8 && idx < N; ++iy) {
      for (i32 ix = 0; ix < 8 && idx < N; ++ix) {
        h_pos[static_cast<std::size_t>(idx)] = {
            (static_cast<double>(ix) + 0.5) * spacing,
            (static_cast<double>(iy) + 0.5) * spacing,
            (static_cast<double>(iz) + 0.5) * spacing};
        ++idx;
      }
    }
  }

  real r_list = real{4.0};

  // CPU cell list.
  tdmd::domain::CellList cpu_cl;
  // Need to include the CPU header too.
  cpu_cl.build(h_pos.data(), N, box, r_list);

  // GPU cell list.
  DeviceBuffer<PositionVec> d_pos(static_cast<std::size_t>(N));
  d_pos.copy_from_host(h_pos.data(), static_cast<std::size_t>(N));

  domain::DeviceCellList gpu_cl;
  gpu_cl.build(d_pos.data(), N, box, r_list);

  // Grid dimensions should match.
  EXPECT_EQ(gpu_cl.ncells_x(), cpu_cl.ncells_x());
  EXPECT_EQ(gpu_cl.ncells_y(), cpu_cl.ncells_y());
  EXPECT_EQ(gpu_cl.ncells_z(), cpu_cl.ncells_z());

  // Cell counts should match.
  i32 ncells = gpu_cl.ncells_total();
  std::vector<i32> gpu_counts(static_cast<std::size_t>(ncells));
  TDMD_CUDA_CHECK(cudaMemcpy(gpu_counts.data(), gpu_cl.d_cell_counts(),
                             static_cast<std::size_t>(ncells) * sizeof(i32),
                             cudaMemcpyDeviceToHost));

  for (i32 c = 0; c < ncells; ++c) {
    EXPECT_EQ(gpu_counts[static_cast<std::size_t>(c)], cpu_cl.count(c))
        << "cell count mismatch at cell " << c;
  }
}

// RD-3 / ADR 0010: when TDMD_DETERMINISTIC_REDUCE is ON, two back-to-back
// builds on the same input must produce bit-identical cell_atoms arrays (the
// scatter must be ID-ordered, not atomic-race-ordered). The default OFF build
// allows ties to fall either way, so this test only enforces the contract in
// the deterministic mode.
//
// Additionally, the deterministic ordering must match "atoms-in-each-cell
// appear in ascending atom-ID order" — that is the canonical ordering ADR 0010
// chose so the ordering is defined by the input alone, not by the scheduler.
TEST(DeviceCellListDeterminism, TwoBuildsBitIdentical) {
  if constexpr (!kDeterministicReduce) {
    GTEST_SKIP() << "enable -DTDMD_DETERMINISTIC_REDUCE=ON to run";
  }

  // Pack 2048 atoms into a box small enough that many atoms land in the
  // same cell (cell size ~= r_list = 3.5 A, box = 14 A ⇒ 4^3 = 64 cells,
  // average 32 atoms per cell — lots of ties for the scatter to break).
  constexpr i64 N = 2048;
  Box box;
  box.lo = {0, 0, 0};
  box.hi = {14, 14, 14};
  box.periodic = {true, true, true};

  std::vector<PositionVec> h_pos(static_cast<std::size_t>(N));
  // Deterministic pseudo-random placement (fixed LCG seed so the test is
  // reproducible regardless of the host RNG).
  std::uint64_t s = 0x9E3779B97F4A7C15ULL;
  auto rand01 = [&]() {
    s = s * 6364136223846793005ULL + 1442695040888963407ULL;
    return static_cast<double>(s >> 11) / static_cast<double>(1ULL << 53);
  };
  for (i64 i = 0; i < N; ++i) {
    h_pos[static_cast<std::size_t>(i)] = {
        rand01() * 14.0, rand01() * 14.0, rand01() * 14.0};
  }

  DeviceBuffer<PositionVec> d_pos(static_cast<std::size_t>(N));
  d_pos.copy_from_host(h_pos.data(), static_cast<std::size_t>(N));

  auto snapshot = [&](const domain::DeviceCellList& cl) {
    i32 nc = cl.ncells_total();
    std::vector<i32> atoms(static_cast<std::size_t>(N));
    std::vector<i32> offsets(static_cast<std::size_t>(nc));
    std::vector<i32> counts(static_cast<std::size_t>(nc));
    TDMD_CUDA_CHECK(cudaMemcpy(atoms.data(), cl.d_cell_atoms(),
                               static_cast<std::size_t>(N) * sizeof(i32),
                               cudaMemcpyDeviceToHost));
    TDMD_CUDA_CHECK(cudaMemcpy(offsets.data(), cl.d_cell_offsets(),
                               static_cast<std::size_t>(nc) * sizeof(i32),
                               cudaMemcpyDeviceToHost));
    TDMD_CUDA_CHECK(cudaMemcpy(counts.data(), cl.d_cell_counts(),
                               static_cast<std::size_t>(nc) * sizeof(i32),
                               cudaMemcpyDeviceToHost));
    return std::make_tuple(atoms, offsets, counts);
  };

  domain::DeviceCellList cl1;
  cl1.build(d_pos.data(), N, box, real{3.5});
  auto [atoms1, offsets1, counts1] = snapshot(cl1);

  domain::DeviceCellList cl2;
  cl2.build(d_pos.data(), N, box, real{3.5});
  auto [atoms2, offsets2, counts2] = snapshot(cl2);

  // Bit-identical between two builds.
  ASSERT_EQ(atoms1, atoms2);
  ASSERT_EQ(offsets1, offsets2);
  ASSERT_EQ(counts1, counts2);

  // And the canonical ordering: within each cell, atom IDs ascend.
  i32 ncells = cl1.ncells_total();
  for (i32 c = 0; c < ncells; ++c) {
    i32 off = offsets1[static_cast<std::size_t>(c)];
    i32 cnt = counts1[static_cast<std::size_t>(c)];
    for (i32 k = 1; k < cnt; ++k) {
      EXPECT_LT(atoms1[static_cast<std::size_t>(off + k - 1)],
                atoms1[static_cast<std::size_t>(off + k)])
          << "cell " << c << " slot " << k << " not in ascending atom-ID order";
    }
  }
}
