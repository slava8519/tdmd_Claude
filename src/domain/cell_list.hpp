// SPDX-License-Identifier: Apache-2.0
// cell_list.hpp — regular grid cell list for neighbor search acceleration.
#pragma once

#include <vector>

#include "../core/box.hpp"
#include "../core/math.hpp"
#include "../core/types.hpp"

namespace tdmd::domain {

/// @brief Regular 3D grid of cells for O(N) neighbor list construction.
///
/// Cell side >= r_list so that neighbors of any atom are in the same cell
/// or the 26 adjacent cells.
class CellList {
 public:
  /// @brief Build the cell list from positions.
  /// @param positions Per-atom positions (length natoms).
  /// @param natoms Number of atoms.
  /// @param box Simulation box.
  /// @param r_list Cutoff + skin distance. Cells will have side >= r_list.
  void build(const Vec3* positions, i64 natoms, const Box& box, real r_list);

  /// Number of cells along each axis.
  [[nodiscard]] i32 ncells_x() const noexcept { return ncx_; }
  [[nodiscard]] i32 ncells_y() const noexcept { return ncy_; }
  [[nodiscard]] i32 ncells_z() const noexcept { return ncz_; }
  [[nodiscard]] i32 ncells_total() const noexcept { return ncx_ * ncy_ * ncz_; }

  /// Cell size along each axis.
  [[nodiscard]] Vec3 cell_size() const noexcept { return cell_size_; }

  /// Number of atoms in cell `cell_id`.
  [[nodiscard]] i32 count(i32 cell_id) const noexcept {
    return cell_counts_[static_cast<std::size_t>(cell_id)];
  }

  /// Pointer to the first atom index in cell `cell_id`.
  [[nodiscard]] const i32* atoms_in_cell(i32 cell_id) const noexcept {
    return cell_atoms_.data() +
           cell_offsets_[static_cast<std::size_t>(cell_id)];
  }

  /// Flat cell index from (ix, iy, iz) with PBC wrapping.
  [[nodiscard]] i32 cell_index(i32 ix, i32 iy, i32 iz) const noexcept {
    // Wrap with PBC.
    if (ix < 0) ix += ncx_;
    else if (ix >= ncx_) ix -= ncx_;
    if (iy < 0) iy += ncy_;
    else if (iy >= ncy_) iy -= ncy_;
    if (iz < 0) iz += ncz_;
    else if (iz >= ncz_) iz -= ncz_;
    return iz * ncx_ * ncy_ + iy * ncx_ + ix;
  }

  /// Cell index for a given position.
  /// `lo` is Vec3D (geometry stored in double per ADR 0007); the subtraction
  /// promotes to double then casts to i32.
  [[nodiscard]] i32 cell_of(Vec3 pos, Vec3D lo) const noexcept {
    auto ix = static_cast<i32>((static_cast<double>(pos.x) - lo.x) /
                               static_cast<double>(cell_size_.x));
    auto iy = static_cast<i32>((static_cast<double>(pos.y) - lo.y) /
                               static_cast<double>(cell_size_.y));
    auto iz = static_cast<i32>((static_cast<double>(pos.z) - lo.z) /
                               static_cast<double>(cell_size_.z));
    // Clamp to valid range (handles edge cases at boundary).
    if (ix >= ncx_) ix = ncx_ - 1;
    if (iy >= ncy_) iy = ncy_ - 1;
    if (iz >= ncz_) iz = ncz_ - 1;
    if (ix < 0) ix = 0;
    if (iy < 0) iy = 0;
    if (iz < 0) iz = 0;
    return iz * ncx_ * ncy_ + iy * ncx_ + ix;
  }

 private:
  i32 ncx_{0}, ncy_{0}, ncz_{0};
  Vec3 cell_size_{0, 0, 0};
  std::vector<i32> cell_counts_;   // length = ncells_total
  std::vector<i32> cell_offsets_;  // length = ncells_total (prefix sum)
  std::vector<i32> cell_atoms_;    // length = natoms, sorted by cell
};

}  // namespace tdmd::domain
