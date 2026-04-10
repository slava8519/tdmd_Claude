// SPDX-License-Identifier: Apache-2.0
// device_cell_list.cu — GPU cell list implementation.

#include "device_cell_list.cuh"

#include <algorithm>
#include <cmath>
#include <vector>

#include "../core/device_buffer.cuh"
#include "../core/error.hpp"

namespace tdmd::domain {

namespace {

/// Kernel: assign each atom to a cell and atomically increment cell counts.
__global__ void assign_cells_kernel(
    const PositionVec* __restrict__ positions,
    i32* __restrict__ atom_cells, i32* __restrict__ cell_counts, i32 natoms,
    Vec3D lo, Vec3D inv_cell_size, i32 ncx, i32 ncy, i32 ncz) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= natoms) return;

  PositionVec p = positions[idx];
  auto ix = static_cast<i32>((p.x - lo.x) * inv_cell_size.x);
  auto iy = static_cast<i32>((p.y - lo.y) * inv_cell_size.y);
  auto iz = static_cast<i32>((p.z - lo.z) * inv_cell_size.z);

  // Clamp to valid range.
  if (ix >= ncx) ix = ncx - 1;
  if (iy >= ncy) iy = ncy - 1;
  if (iz >= ncz) iz = ncz - 1;
  if (ix < 0) ix = 0;
  if (iy < 0) iy = 0;
  if (iz < 0) iz = 0;

  i32 cell_id = iz * ncx * ncy + iy * ncx + ix;
  atom_cells[idx] = cell_id;
  atomicAdd(&cell_counts[cell_id], 1);
}

/// Kernel: scatter atom indices into sorted order using offsets.
__global__ void scatter_kernel(const i32* __restrict__ atom_cells,
                               const i32* __restrict__ cell_offsets,
                               i32* __restrict__ cell_placed,
                               i32* __restrict__ cell_atoms, i32 natoms) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= natoms) return;

  i32 cell_id = atom_cells[idx];
  i32 slot = cell_offsets[cell_id] + atomicAdd(&cell_placed[cell_id], 1);
  cell_atoms[slot] = idx;
}

}  // namespace

void DeviceCellList::build(const PositionVec* d_positions, i64 natoms,
                           const Box& box, real r_list) {
  TDMD_ASSERT(r_list > real{0}, "r_list must be positive");
  TDMD_ASSERT(natoms >= 0, "natoms must be non-negative");
  if (natoms == 0) return;

  const Vec3D box_size = box.size();

  // Determine grid dimensions (same logic as CPU).
  ncx_ = std::max(3, static_cast<i32>(std::floor(box_size.x / r_list)));
  ncy_ = std::max(3, static_cast<i32>(std::floor(box_size.y / r_list)));
  ncz_ = std::max(3, static_cast<i32>(std::floor(box_size.z / r_list)));

  cell_size_ = {box_size.x / static_cast<double>(ncx_),
                box_size.y / static_cast<double>(ncy_),
                box_size.z / static_cast<double>(ncz_)};

  Vec3D inv_cell_size = {1.0 / cell_size_.x, 1.0 / cell_size_.y,
                         1.0 / cell_size_.z};

  i32 ncells = ncx_ * ncy_ * ncz_;
  auto un = static_cast<std::size_t>(natoms);
  auto ucells = static_cast<std::size_t>(ncells);

  // Allocate / resize device buffers.
  cell_counts_.resize(ucells);
  cell_offsets_.resize(ucells);
  cell_atoms_.resize(un);
  atom_cells_.resize(un);

  // Zero cell counts.
  cell_counts_.zero();

  // Step 1: assign cells + count.
  constexpr int kBlock = 256;
  int grid = (static_cast<int>(natoms) + kBlock - 1) / kBlock;
  assign_cells_kernel<<<grid, kBlock>>>(
      d_positions, atom_cells_.data(), cell_counts_.data(),
      static_cast<i32>(natoms), box.lo, inv_cell_size, ncx_, ncy_, ncz_);
  TDMD_CUDA_CHECK(cudaGetLastError());

  // Step 2: exclusive prefix sum on host (ncells is small, typically <1000).
  std::vector<i32> h_counts(ucells);
  std::vector<i32> h_offsets(ucells);
  cell_counts_.copy_to_host(h_counts.data(), ucells);

  h_offsets[0] = 0;
  for (i32 c = 1; c < ncells; ++c) {
    h_offsets[static_cast<std::size_t>(c)] =
        h_offsets[static_cast<std::size_t>(c - 1)] +
        h_counts[static_cast<std::size_t>(c - 1)];
  }
  cell_offsets_.copy_from_host(h_offsets.data(), ucells);

  // Step 3: scatter atoms into sorted order.
  // Use a temporary "placed" counter (same as cell_counts but zeroed).
  DeviceBuffer<i32> cell_placed(ucells);
  cell_placed.zero();

  scatter_kernel<<<grid, kBlock>>>(atom_cells_.data(), cell_offsets_.data(),
                                   cell_placed.data(), cell_atoms_.data(),
                                   static_cast<i32>(natoms));
  TDMD_CUDA_CHECK(cudaGetLastError());
  TDMD_CUDA_CHECK(cudaDeviceSynchronize());
}

}  // namespace tdmd::domain
