// SPDX-License-Identifier: Apache-2.0
// test_device_neighbor_list.cu — GPU neighbor list tests.
#include <gtest/gtest.h>

#include <algorithm>
#include <set>
#include <vector>

#include "core/device_buffer.cuh"
#include "core/types.hpp"
#include "neighbors/device_neighbor_list.cuh"
#include "neighbors/neighbor_list.hpp"

using namespace tdmd;

TEST(DeviceNeighborList, FullListSymmetry) {
  // Full-list: if j is neighbor of i, then i must be neighbor of j.
  constexpr i64 N = 32;
  Box box;
  box.lo = {0, 0, 0};
  box.hi = {10, 10, 10};
  box.periodic = {true, true, true};

  // 4x4x2 grid.
  std::vector<Vec3> h_pos(static_cast<std::size_t>(N));
  real sp = real{2.5};
  i32 idx = 0;
  for (i32 iz = 0; iz < 2; ++iz) {
    for (i32 iy = 0; iy < 4; ++iy) {
      for (i32 ix = 0; ix < 4; ++ix) {
        h_pos[static_cast<std::size_t>(idx)] = {
            static_cast<real>(ix) * sp + real{0.5},
            static_cast<real>(iy) * sp + real{0.5},
            static_cast<real>(iz) * sp + real{0.5}};
        ++idx;
      }
    }
  }

  DeviceBuffer<Vec3> d_pos(static_cast<std::size_t>(N));
  d_pos.copy_from_host(h_pos.data(), static_cast<std::size_t>(N));

  neighbors::DeviceNeighborList dnl;
  dnl.build(d_pos.data(), N, box, real{3.0}, real{0.5});

  // Download.
  auto un = static_cast<std::size_t>(N);
  std::vector<i32> h_counts(un), h_offsets(un);
  TDMD_CUDA_CHECK(cudaMemcpy(h_counts.data(), dnl.d_counts(),
                             un * sizeof(i32), cudaMemcpyDeviceToHost));
  TDMD_CUDA_CHECK(cudaMemcpy(h_offsets.data(), dnl.d_offsets(),
                             un * sizeof(i32), cudaMemcpyDeviceToHost));

  i32 total = h_offsets[un - 1] + h_counts[un - 1];
  std::vector<i32> h_nbrs(static_cast<std::size_t>(total));
  TDMD_CUDA_CHECK(cudaMemcpy(h_nbrs.data(), dnl.d_neighbors(),
                             static_cast<std::size_t>(total) * sizeof(i32),
                             cudaMemcpyDeviceToHost));

  // Build adjacency sets.
  std::vector<std::set<i32>> adj(un);
  for (std::size_t i = 0; i < un; ++i) {
    for (i32 k = 0; k < h_counts[i]; ++k) {
      adj[i].insert(h_nbrs[static_cast<std::size_t>(h_offsets[i] + k)]);
    }
  }

  // Check symmetry.
  for (std::size_t i = 0; i < un; ++i) {
    for (i32 j : adj[i]) {
      EXPECT_TRUE(adj[static_cast<std::size_t>(j)].count(static_cast<i32>(i)))
          << "full-list asymmetry: " << i << " has neighbor " << j
          << " but not vice versa";
    }
  }
}

TEST(DeviceNeighborList, MatchesCPUHalfListPairCount) {
  // GPU full-list should have exactly 2x the pairs of CPU half-list.
  constexpr i64 N = 256;
  Box box;
  box.lo = {0, 0, 0};
  box.hi = {14.46, 14.46, 14.46};
  box.periodic = {true, true, true};

  std::vector<Vec3> h_pos(static_cast<std::size_t>(N));
  real spacing = real{14.46} / real{8};
  i32 idx = 0;
  for (i32 iz = 0; iz < 8 && idx < N; ++iz) {
    for (i32 iy = 0; iy < 8 && idx < N; ++iy) {
      for (i32 ix = 0; ix < 8 && idx < N; ++ix) {
        h_pos[static_cast<std::size_t>(idx)] = {
            (static_cast<real>(ix) + real{0.5}) * spacing,
            (static_cast<real>(iy) + real{0.5}) * spacing,
            (static_cast<real>(iz) + real{0.5}) * spacing};
        ++idx;
      }
    }
  }

  real r_cut = real{4.0};
  real r_skin = real{0.5};

  // CPU half-list.
  neighbors::NeighborList cpu_nl;
  cpu_nl.build(h_pos.data(), N, box, r_cut, r_skin);
  i64 cpu_pairs = 0;
  for (i64 i = 0; i < N; ++i) {
    cpu_pairs += cpu_nl.count(i);
  }

  // GPU full-list.
  DeviceBuffer<Vec3> d_pos(static_cast<std::size_t>(N));
  d_pos.copy_from_host(h_pos.data(), static_cast<std::size_t>(N));

  neighbors::DeviceNeighborList dnl;
  dnl.build(d_pos.data(), N, box, r_cut, r_skin);

  auto un = static_cast<std::size_t>(N);
  std::vector<i32> h_counts(un);
  TDMD_CUDA_CHECK(cudaMemcpy(h_counts.data(), dnl.d_counts(),
                             un * sizeof(i32), cudaMemcpyDeviceToHost));
  i64 gpu_pairs = 0;
  for (auto c : h_counts) gpu_pairs += c;

  // Full-list = 2 * half-list.
  EXPECT_EQ(gpu_pairs, 2 * cpu_pairs)
      << "GPU full-list pair count should be 2x CPU half-list";
}
