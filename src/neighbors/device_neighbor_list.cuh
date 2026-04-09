// SPDX-License-Identifier: Apache-2.0
// device_neighbor_list.cuh — GPU Verlet neighbor list (full-list).
#pragma once

#include <cuda_runtime.h>

#include "../core/box.hpp"
#include "../core/device_buffer.cuh"
#include "../core/types.hpp"
#include "../domain/device_cell_list.cuh"

namespace tdmd::neighbors {

/// @brief GPU Verlet neighbor list in CSR format (full-list).
///
/// Full-list stores both (i,j) and (j,i), so each atom accumulates its own
/// forces without atomics.  Built using a DeviceCellList.
class DeviceNeighborList {
 public:
  /// @brief Build the neighbor list on GPU.
  /// @param d_positions Device pointer to Vec3 positions.
  /// @param natoms Number of atoms.
  /// @param box Simulation box (host-side).
  /// @param r_cut Force cutoff radius.
  /// @param r_skin Skin distance.
  /// @param stream CUDA stream (default 0 = legacy default stream).
  void build(const Vec3* d_positions, i64 natoms, const Box& box, real r_cut,
             real r_skin, cudaStream_t stream = 0);

  /// @brief Device pointer to neighbor indices (flat CSR).
  [[nodiscard]] const i32* d_neighbors() const noexcept {
    return neighbors_.data();
  }
  /// @brief Device pointer to per-atom neighbor counts.
  [[nodiscard]] const i32* d_counts() const noexcept { return counts_.data(); }
  /// @brief Device pointer to per-atom offsets into neighbor array.
  [[nodiscard]] const i32* d_offsets() const noexcept {
    return offsets_.data();
  }

  [[nodiscard]] real cutoff() const noexcept { return r_cut_; }
  [[nodiscard]] real skin() const noexcept { return r_skin_; }
  [[nodiscard]] i32 max_neighbors() const noexcept { return max_neighbors_; }

 private:
  real r_cut_{0};
  real r_skin_{0};
  i32 max_neighbors_{128};  // initial estimate, grows if needed

  domain::DeviceCellList cell_list_;

  DeviceBuffer<i32> neighbors_;  // flat CSR
  DeviceBuffer<i32> counts_;     // per-atom count
  DeviceBuffer<i32> offsets_;    // per-atom offset
};

}  // namespace tdmd::neighbors
