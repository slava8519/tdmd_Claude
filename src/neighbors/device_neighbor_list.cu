// SPDX-License-Identifier: Apache-2.0
// device_neighbor_list.cu — GPU full neighbor list implementation.

#include "device_neighbor_list.cuh"

#include <cub/device/device_reduce.cuh>
#include <cub/device/device_scan.cuh>

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
    const PositionVec* __restrict__ positions, i32 natoms,
    const i32* __restrict__ cell_atoms, const i32* __restrict__ cell_offsets,
    const i32* __restrict__ cell_counts, i32 ncx, i32 ncy, i32 ncz,
    Vec3D box_lo, Vec3D box_size, bool pbc_x, bool pbc_y, bool pbc_z,
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
  real inv_cx = static_cast<real>(ncx) / static_cast<real>(box_size.x);
  real inv_cy = static_cast<real>(ncy) / static_cast<real>(box_size.y);
  real inv_cz = static_cast<real>(ncz) / static_cast<real>(box_size.z);

  auto cix = static_cast<i32>((static_cast<real>(positions[i].x) -
                               static_cast<real>(box_lo.x)) * inv_cx);
  auto ciy = static_cast<i32>((static_cast<real>(positions[i].y) -
                               static_cast<real>(box_lo.y)) * inv_cy);
  auto ciz = static_cast<i32>((static_cast<real>(positions[i].z) -
                               static_cast<real>(box_lo.z)) * inv_cz);
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

/// @brief Tiny helper kernel: pack total_pairs and max_neighbors into one
/// 2-element buffer for a single coalesced D2H copy.
///
/// total_pairs = d_offsets[n-1] + d_counts[n-1] (exclusive-scan identity).
/// d_max is already written by cub::DeviceReduce::Max. We just gather both
/// into d_meta[0], d_meta[1] so the host only needs one cudaMemcpyAsync.
__global__ void pack_meta_kernel(const i32* __restrict__ d_offsets,
                                 const i32* __restrict__ d_counts,
                                 const i32* __restrict__ d_max, i32 n,
                                 i32* __restrict__ d_meta) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  d_meta[0] = d_offsets[n - 1] + d_counts[n - 1];  // total_pairs
  d_meta[1] = *d_max;                              // max_neighbors
}

}  // namespace

void DeviceNeighborList::build(const PositionVec* d_positions, i64 natoms,
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

  // Allocate counts + offsets.
  counts_.resize(un);
  counts_.zero();
  offsets_.resize(un);

  Vec3D box_size = box.size();

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

  // --- Device-resident prefix sum (replaces host D2H + CPU scan + H2D) ---
  //
  // OPT-1 (ADR 0005 follow-up): the old implementation did
  //   D2H counts → CPU prefix sum → H2D offsets
  // with two cudaStreamSynchronize calls, ~4*N bytes of PCIe traffic, and
  // serialised the entire scheduler. We now chain three CUB device primitives
  // on `stream`:
  //   1. cub::DeviceScan::ExclusiveSum(counts → offsets)
  //   2. cub::DeviceReduce::Max(counts → d_max_scalar)
  //   3. pack_meta_kernel(offsets[n-1] + counts[n-1], d_max → d_meta[0..1])
  // and finish with a single 8-byte D2H. One sync, bandwidth ~negligible,
  // the entire prefix sum stays on the GPU.

  d_meta_.resize(2);
  d_max_scalar_.resize(1);

  // Query + grow scan temp storage.
  std::size_t scan_bytes = 0;
  cub::DeviceScan::ExclusiveSum(nullptr, scan_bytes, counts_.data(),
                                offsets_.data(), static_cast<int>(natoms),
                                stream);

  // Query + grow reduce-max temp storage. CUB requires separate temp buffers
  // for DeviceScan and DeviceReduce — sharing is not supported.
  std::size_t reduce_bytes = 0;
  cub::DeviceReduce::Max(nullptr, reduce_bytes, counts_.data(),
                         d_max_scalar_.data(), static_cast<int>(natoms),
                         stream);

  // Keep a single byte buffer sized to the larger of the two. Callers rebuild
  // with the same natoms over and over, so the grow is amortised to zero
  // after the first build.
  std::size_t needed = std::max(scan_bytes, reduce_bytes);
  if (d_cub_temp_.size() < needed) {
    d_cub_temp_.resize(needed);
  }

  cub::DeviceScan::ExclusiveSum(d_cub_temp_.data(), scan_bytes, counts_.data(),
                                offsets_.data(), static_cast<int>(natoms),
                                stream);

  cub::DeviceReduce::Max(d_cub_temp_.data(), reduce_bytes, counts_.data(),
                         d_max_scalar_.data(), static_cast<int>(natoms),
                         stream);

  pack_meta_kernel<<<1, 1, 0, stream>>>(offsets_.data(), counts_.data(),
                                        d_max_scalar_.data(),
                                        static_cast<i32>(natoms),
                                        d_meta_.data());
  TDMD_CUDA_CHECK(cudaGetLastError());

  // One 8-byte D2H, then sync. This is the only host round-trip in build().
  i32 h_meta[2] = {0, 0};
  TDMD_CUDA_CHECK(cudaMemcpyAsync(h_meta, d_meta_.data(), sizeof(h_meta),
                                  cudaMemcpyDeviceToHost, stream));
  TDMD_CUDA_CHECK(cudaStreamSynchronize(stream));

  i32 total_pairs = h_meta[0];
  max_neighbors_ = h_meta[1];

  // Allocate neighbor storage.
  if (total_pairs > 0) {
    neighbors_.resize(static_cast<std::size_t>(total_pairs));
  }

  // Reset counts for pass 2 (the kernel re-counts as it writes).
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
  // NOTE: no stream sync here. The caller is expected to sync the stream
  // before reading any nlist data on the host (which, in the scheduler hot
  // path, happens only at run_until() end).
}

}  // namespace tdmd::neighbors
