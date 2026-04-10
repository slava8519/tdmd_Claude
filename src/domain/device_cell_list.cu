// SPDX-License-Identifier: Apache-2.0
// device_cell_list.cu — GPU cell list implementation.

#include "device_cell_list.cuh"

#include <algorithm>
#include <cmath>
#include <vector>

#include "../core/determinism.hpp"
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
  if constexpr (kDeterministicReduce) {
    // Deterministic path (ADR 0010): place atoms in strict ID order.
    // The default GPU path uses atomicAdd on a per-cell slot counter, so
    // two atoms in the same cell can land in either order depending on
    // warp scheduling. That ordering propagates into the neighbor-list
    // walk and ultimately into float atomicAdds in the force kernels,
    // which is exactly the non-reproducibility we want to eliminate here.
    // We shuttle atom_cells_ through host memory and do the scatter
    // sequentially. Cell-list build runs every ~10 MD steps and is not
    // the hot path, so a 2 * N-int PCIe round trip is acceptable; the
    // force kernel path is unaffected.
    std::vector<i32> h_atom_cells(un);
    std::vector<i32> h_cell_atoms(un);
    atom_cells_.copy_to_host(h_atom_cells.data(), un);
    std::vector<i32> h_placed(ucells, 0);
    for (i64 i = 0; i < natoms; ++i) {
      i32 cell_id = h_atom_cells[static_cast<std::size_t>(i)];
      i32 slot = h_offsets[static_cast<std::size_t>(cell_id)] +
                 h_placed[static_cast<std::size_t>(cell_id)]++;
      h_cell_atoms[static_cast<std::size_t>(slot)] = static_cast<i32>(i);
    }
    cell_atoms_.copy_from_host(h_cell_atoms.data(), un);
    TDMD_CUDA_CHECK(cudaDeviceSynchronize());
  } else {
    // Fast default path: atomicAdd slot reservation. Non-deterministic
    // under warp scheduling ties, but ~2 orders of magnitude faster than
    // the host round trip above.
    DeviceBuffer<i32> cell_placed(ucells);
    cell_placed.zero();

    scatter_kernel<<<grid, kBlock>>>(atom_cells_.data(), cell_offsets_.data(),
                                     cell_placed.data(), cell_atoms_.data(),
                                     static_cast<i32>(natoms));
    TDMD_CUDA_CHECK(cudaGetLastError());
    TDMD_CUDA_CHECK(cudaDeviceSynchronize());
  }
}

}  // namespace tdmd::domain
