// SPDX-License-Identifier: Apache-2.0
// neighbor_list.hpp — Verlet neighbor list with skin.
//
// CPU half-list at M1, GPU full-list at M2.
#pragma once

#include <vector>

#include "../core/box.hpp"
#include "../core/types.hpp"
#include "../domain/cell_list.hpp"

namespace tdmd::neighbors {

/// @brief Verlet neighbor list (half-list, CPU).
///
/// Built using a CellList for O(N) construction.
/// Stores neighbors in CSR layout: for atom i, neighbors are
///   neighbors[offsets[i] .. offsets[i] + counts[i]).
/// Half-list: only stores pair (i,j) once where i < j.
class NeighborList {
 public:
  /// @brief Build the neighbor list from scratch.
  /// @param positions Per-atom positions.
  /// @param natoms Number of atoms.
  /// @param box Simulation box.
  /// @param r_cut Force cutoff radius.
  /// @param r_skin Skin distance (list stores pairs within r_cut + r_skin).
  void build(const Vec3* positions, i64 natoms, const Box& box,
             real r_cut, real r_skin);

  /// Check if any atom has moved more than r_skin/2 since last build.
  /// If so, the list needs rebuilding.
  [[nodiscard]] bool needs_rebuild(const Vec3* positions, i64 natoms) const;

  /// Number of neighbors of atom i.
  [[nodiscard]] i32 count(i64 i) const noexcept {
    return counts_[static_cast<std::size_t>(i)];
  }

  /// Pointer to neighbor indices of atom i.
  [[nodiscard]] const i32* neighbors_of(i64 i) const noexcept {
    return neighbors_.data() + offsets_[static_cast<std::size_t>(i)];
  }

  [[nodiscard]] real cutoff() const noexcept { return r_cut_; }
  [[nodiscard]] real skin() const noexcept { return r_skin_; }

 private:
  real r_cut_{0};
  real r_skin_{0};
  std::vector<i32> neighbors_;
  std::vector<i32> offsets_;
  std::vector<i32> counts_;
  std::vector<Vec3> build_positions_;  // snapshot at build time for skin check
  domain::CellList cell_list_;
};

}  // namespace tdmd::neighbors
