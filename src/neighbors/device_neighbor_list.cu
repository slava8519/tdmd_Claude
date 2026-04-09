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

  // Position load in double for relative-coordinate trick (ADR 0007/0008).
  pos_t pix = positions[i].x;
  pos_t piy = positions[i].y;
  pos_t piz = positions[i].z;

  // Cell index determination (can stay in real — no precision concern).
  real inv_cx = static_cast<real>(ncx) / box_size.x;
  real inv_cy = static_cast<real>(ncy) / box_size.y;
  real inv_cz = static_cast<real>(ncz) / box_size.z;

  auto cix = static_cast<i32>((positions[i].x - box_lo.x) * inv_cx);
  auto ciy = static_cast<i32>((positions[i].y - box_lo.y) * inv_cy);
  auto ciz = static_cast<i32>((positions[i].z - box_lo.z) * inv_cz);
  if (cix >= ncx) cix = ncx - 1;
  if (ciy >= ncy) ciy = ncy - 1;
  if (ciz >= ncz) ciz = ncz - 1;
  if (cix < 0) cix = 0;
  if (ciy < 0) ciy = 0;
  if (ciz < 0) ciz = 0;

  // Box dimensions and cutoff to real for distance computation.
  // In mixed mode (real=float): relative-coordinate trick gives float distance.
  // In fp64 mode (real=double): identity, full double precision.
  const real bsx = static_cast<real>(box_size.x);
  const real bsy = static_cast<real>(box_size.y);
  const real bsz = static_cast<real>(box_size.z);
  const real bsx_half = real{0.5} * bsx;
  const real bsy_half = real{0.5} * bsy;
  const real bsz_half = real{0.5} * bsz;
  const real r_list_sq_r = static_cast<real>(r_list_sq);

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

          // Relative-coordinate trick: one double subtract, then cast to real.
          real dx_ij = static_cast<real>(pix - positions[j].x);
          real dy_ij = static_cast<real>(piy - positions[j].y);
          real dz_ij = static_cast<real>(piz - positions[j].z);

          if (pbc_x) {
            if (dx_ij > bsx_half)
              dx_ij -= bsx;
            else if (dx_ij < -bsx_half)
              dx_ij += bsx;
          }
          if (pbc_y) {
            if (dy_ij > bsy_half)
              dy_ij -= bsy;
            else if (dy_ij < -bsy_half)
              dy_ij += bsy;
          }
          if (pbc_z) {
            if (dz_ij > bsz_half)
              dz_ij -= bsz;
            else if (dz_ij < -bsz_half)
              dz_ij += bsz;
          }

          real r2 = dx_ij * dx_ij + dy_ij * dy_ij + dz_ij * dz_ij;
          if (r2 < r_list_sq_r) {
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
                               const Box& box, real r_cut, real r_skin,
                               cudaStream_t stream) {
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
  build_nlist_kernel<<<grid, kBlock, 0, stream>>>(
      d_positions, static_cast<i32>(natoms), cell_list_.d_cell_atoms(),
      cell_list_.d_cell_offsets(), cell_list_.d_cell_counts(),
      cell_list_.ncells_x(), cell_list_.ncells_y(), cell_list_.ncells_z(),
      box.lo, box_size, box.periodic[0], box.periodic[1], box.periodic[2],
      r_list_sq, counts_.data(), nullptr, nullptr, 0, nullptr);
  TDMD_CUDA_CHECK(cudaGetLastError());
  TDMD_CUDA_CHECK(cudaStreamSynchronize(stream));

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
  build_nlist_kernel<<<grid, kBlock, 0, stream>>>(
      d_positions, static_cast<i32>(natoms), cell_list_.d_cell_atoms(),
      cell_list_.d_cell_offsets(), cell_list_.d_cell_counts(),
      cell_list_.ncells_x(), cell_list_.ncells_y(), cell_list_.ncells_z(),
      box.lo, box_size, box.periodic[0], box.periodic[1], box.periodic[2],
      r_list_sq, counts_.data(), neighbors_.data(), offsets_.data(),
      max_neighbors_, d_overflow.data());
  TDMD_CUDA_CHECK(cudaGetLastError());
  TDMD_CUDA_CHECK(cudaStreamSynchronize(stream));
}

}  // namespace tdmd::neighbors
