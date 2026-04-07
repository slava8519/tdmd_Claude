// SPDX-License-Identifier: Apache-2.0
// device_neighbor_list.cu — GPU full neighbor list implementation.

#include "device_neighbor_list.cuh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

#include "../core/device_buffer.cuh"
#include "../core/error.hpp"
#include "../core/math.hpp"

namespace tdmd::neighbors {

namespace {

/// @brief Kernel: count neighbors per atom and optionally write them.
///
/// When `d_neighbors` is nullptr, only counts are written (sizing pass).
/// When `d_neighbors` is provided, neighbors are written (fill pass).
__global__ void build_nlist_kernel(
    const Vec3* __restrict__ positions, i32 natoms,
    const i32* __restrict__ cell_atoms, const i32* __restrict__ cell_offsets,
    const i32* __restrict__ cell_counts, i32 ncx, i32 ncy, i32 ncz,
    Vec3 box_lo, Vec3 box_size, bool pbc_x, bool pbc_y, bool pbc_z,
    real r_list_sq, i32* __restrict__ d_counts,
    i32* __restrict__ d_neighbors, const i32* __restrict__ d_offsets,
    i32 max_neighbors, i32* __restrict__ d_overflow) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= natoms) return;

  Vec3 pi = positions[i];

  // Determine cell of atom i.
  // Use inverse cell size derived from box_size / ncells.
  real inv_cx = static_cast<real>(ncx) / box_size.x;
  real inv_cy = static_cast<real>(ncy) / box_size.y;
  real inv_cz = static_cast<real>(ncz) / box_size.z;

  auto cix = static_cast<i32>((pi.x - box_lo.x) * inv_cx);
  auto ciy = static_cast<i32>((pi.y - box_lo.y) * inv_cy);
  auto ciz = static_cast<i32>((pi.z - box_lo.z) * inv_cz);
  if (cix >= ncx) cix = ncx - 1;
  if (ciy >= ncy) ciy = ncy - 1;
  if (ciz >= ncz) ciz = ncz - 1;
  if (cix < 0) cix = 0;
  if (ciy < 0) ciy = 0;
  if (ciz < 0) ciz = 0;

  i32 count = 0;
  i32 offset = d_neighbors ? d_offsets[i] : 0;

  // Iterate over 27 neighbor cells.
  for (i32 dz = -1; dz <= 1; ++dz) {
    for (i32 dy = -1; dy <= 1; ++dy) {
      for (i32 dx = -1; dx <= 1; ++dx) {
        i32 nx = cix + dx;
        i32 ny = ciy + dy;
        i32 nz = ciz + dz;

        // PBC wrap.
        if (nx < 0) nx += ncx;
        else if (nx >= ncx) nx -= ncx;
        if (ny < 0) ny += ncy;
        else if (ny >= ncy) ny -= ncy;
        if (nz < 0) nz += ncz;
        else if (nz >= ncz) nz -= ncz;

        i32 cell_id = nz * ncx * ncy + ny * ncx + nx;
        i32 cell_start = cell_offsets[cell_id];
        i32 cell_count = cell_counts[cell_id];

        for (i32 k = 0; k < cell_count; ++k) {
          i32 j = cell_atoms[cell_start + k];
          if (j == i) continue;

          // Minimum image distance.
          real dx_ij = pi.x - positions[j].x;
          real dy_ij = pi.y - positions[j].y;
          real dz_ij = pi.z - positions[j].z;

          if (pbc_x) {
            if (dx_ij > box_size.x * real{0.5})
              dx_ij -= box_size.x;
            else if (dx_ij < -box_size.x * real{0.5})
              dx_ij += box_size.x;
          }
          if (pbc_y) {
            if (dy_ij > box_size.y * real{0.5})
              dy_ij -= box_size.y;
            else if (dy_ij < -box_size.y * real{0.5})
              dy_ij += box_size.y;
          }
          if (pbc_z) {
            if (dz_ij > box_size.z * real{0.5})
              dz_ij -= box_size.z;
            else if (dz_ij < -box_size.z * real{0.5})
              dz_ij += box_size.z;
          }

          real r2 = dx_ij * dx_ij + dy_ij * dy_ij + dz_ij * dz_ij;
          if (r2 < r_list_sq) {
            if (d_neighbors) {
              if (count < max_neighbors) {
                d_neighbors[offset + count] = j;
              } else {
                atomicMax(d_overflow, count + 1);
              }
            }
            ++count;
          }
        }
      }
    }
  }

  d_counts[i] = count;
}

}  // namespace

void DeviceNeighborList::build(const Vec3* d_positions, i64 natoms,
                               const Box& box, real r_cut, real r_skin) {
  TDMD_ASSERT(r_cut > real{0}, "r_cut must be positive");
  TDMD_ASSERT(r_skin >= real{0}, "r_skin must be non-negative");
  if (natoms == 0) return;

  r_cut_ = r_cut;
  r_skin_ = r_skin;
  real r_list = r_cut + r_skin;
  real r_list_sq = r_list * r_list;

  auto un = static_cast<std::size_t>(natoms);

  // Build cell list.
  cell_list_.build(d_positions, natoms, box, r_list);

  // Allocate counts.
  counts_.resize(un);
  counts_.zero();

  Vec3 box_size = box.size();

  constexpr int kBlock = 256;
  int grid = (static_cast<int>(natoms) + kBlock - 1) / kBlock;

  // Pass 1: count neighbors only (d_neighbors = nullptr).
  build_nlist_kernel<<<grid, kBlock>>>(
      d_positions, static_cast<i32>(natoms), cell_list_.d_cell_atoms(),
      cell_list_.d_cell_offsets(), cell_list_.d_cell_counts(),
      cell_list_.ncells_x(), cell_list_.ncells_y(), cell_list_.ncells_z(),
      box.lo, box_size, box.periodic[0], box.periodic[1], box.periodic[2],
      r_list_sq, counts_.data(), nullptr, nullptr, 0, nullptr);
  TDMD_CUDA_CHECK(cudaGetLastError());
  TDMD_CUDA_CHECK(cudaDeviceSynchronize());

  // Download counts, compute offsets on host (prefix sum).
  std::vector<i32> h_counts(un);
  counts_.copy_to_host(h_counts.data(), un);

  std::vector<i32> h_offsets(un);
  h_offsets[0] = 0;
  for (std::size_t a = 1; a < un; ++a) {
    h_offsets[a] = h_offsets[a - 1] + h_counts[a - 1];
  }
  i32 total_pairs = h_offsets[un - 1] + h_counts[un - 1];

  // Find max neighbors for overflow check.
  i32 max_nbrs = 0;
  for (auto c : h_counts) {
    if (c > max_nbrs) max_nbrs = c;
  }
  max_neighbors_ = max_nbrs;

  // Upload offsets.
  offsets_.resize(un);
  offsets_.copy_from_host(h_offsets.data(), un);

  // Allocate neighbor storage.
  if (total_pairs > 0) {
    neighbors_.resize(static_cast<std::size_t>(total_pairs));
  }

  // Reset counts for pass 2.
  counts_.zero();

  // Overflow flag (device).
  DeviceBuffer<i32> d_overflow(1);
  d_overflow.zero();

  // Pass 2: fill neighbors.
  build_nlist_kernel<<<grid, kBlock>>>(
      d_positions, static_cast<i32>(natoms), cell_list_.d_cell_atoms(),
      cell_list_.d_cell_offsets(), cell_list_.d_cell_counts(),
      cell_list_.ncells_x(), cell_list_.ncells_y(), cell_list_.ncells_z(),
      box.lo, box_size, box.periodic[0], box.periodic[1], box.periodic[2],
      r_list_sq, counts_.data(), neighbors_.data(), offsets_.data(),
      max_neighbors_, d_overflow.data());
  TDMD_CUDA_CHECK(cudaGetLastError());
  TDMD_CUDA_CHECK(cudaDeviceSynchronize());
}

}  // namespace tdmd::neighbors
