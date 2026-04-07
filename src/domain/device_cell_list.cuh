// SPDX-License-Identifier: Apache-2.0
// device_cell_list.cuh — GPU cell list via counting sort + prefix sum.
#pragma once

#include "../core/box.hpp"
#include "../core/device_buffer.cuh"
#include "../core/types.hpp"

namespace tdmd::domain {

/// @brief GPU-resident cell list built with counting sort and exclusive prefix sum.
///
/// Mirrors the CPU CellList interface.  Owns all device memory.
/// Build steps: (1) assign each atom to a cell, (2) prefix-sum cell counts,
/// (3) scatter atom indices into sorted order.
class DeviceCellList {
 public:
  /// @brief Build cell list on the GPU.
  /// @param d_positions Device pointer to Vec3 array (length natoms).
  /// @param natoms Number of atoms.
  /// @param box Simulation box (host).
  /// @param r_list Cutoff + skin.
  void build(const Vec3* d_positions, i64 natoms, const Box& box, real r_list);

  [[nodiscard]] i32 ncells_x() const noexcept { return ncx_; }
  [[nodiscard]] i32 ncells_y() const noexcept { return ncy_; }
  [[nodiscard]] i32 ncells_z() const noexcept { return ncz_; }
  [[nodiscard]] i32 ncells_total() const noexcept { return ncx_ * ncy_ * ncz_; }

  /// @brief Device pointer to cell offsets (length ncells_total).
  [[nodiscard]] const i32* d_cell_offsets() const noexcept {
    return cell_offsets_.data();
  }
  /// @brief Device pointer to cell counts (length ncells_total).
  [[nodiscard]] const i32* d_cell_counts() const noexcept {
    return cell_counts_.data();
  }
  /// @brief Device pointer to sorted atom indices (length natoms).
  [[nodiscard]] const i32* d_cell_atoms() const noexcept {
    return cell_atoms_.data();
  }

  [[nodiscard]] Vec3 cell_size() const noexcept { return cell_size_; }

 private:
  i32 ncx_{0}, ncy_{0}, ncz_{0};
  Vec3 cell_size_{0, 0, 0};

  DeviceBuffer<i32> cell_counts_;   // length = ncells
  DeviceBuffer<i32> cell_offsets_;  // length = ncells (exclusive prefix sum)
  DeviceBuffer<i32> cell_atoms_;    // length = natoms, sorted by cell
  DeviceBuffer<i32> atom_cells_;    // length = natoms, cell id per atom
};

}  // namespace tdmd::domain
