// SPDX-License-Identifier: Apache-2.0
// spatial_decomp.hpp — 1D spatial decomposition along Y-axis.
//
// Each spatial rank owns a contiguous Y-slab of the simulation box.
// Ghost atoms are atoms within r_ghost of subdomain boundaries (including PBC).
#pragma once

#include <vector>

#include "../core/box.hpp"
#include "../core/types.hpp"

namespace tdmd::domain {

/// @brief 1D spatial decomposition along the Y-axis for M6 hybrid parallelism.
///
/// Splits the box into P_space equal slabs along Y. Each rank owns atoms
/// in its Y-slab. Ghost atoms from neighboring slabs within r_ghost of the
/// boundary are identified for halo exchange.
class SpatialDecomp {
 public:
  /// @brief Build the spatial decomposition.
  /// @param box Simulation box.
  /// @param n_spatial Number of spatial ranks.
  /// @param my_spatial My spatial rank index (0-based).
  /// @param r_ghost Ghost cutoff distance (typically r_cut + r_skin).
  void build(const Box& box, i32 n_spatial, i32 my_spatial, real r_ghost);

  /// @brief Partition atoms into owned and ghost sets.
  ///
  /// Reorders arrays so that owned atoms come first (indices [0, n_owned)),
  /// then returns which atoms should be sent as ghosts to prev/next neighbor.
  ///
  /// @param positions  Per-atom positions (reordered in-place: owned first).
  /// @param velocities Per-atom velocities (reordered in-place).
  /// @param forces     Per-atom forces (reordered in-place).
  /// @param types      Per-atom types (reordered in-place).
  /// @param ids        Per-atom ids (reordered in-place).
  /// @param natoms     Total number of atoms.
  /// @return Number of owned atoms (first n_owned elements after reorder).
  i32 partition_atoms(Vec3* positions, Vec3* velocities, Vec3* forces,
                      i32* types, i32* ids, i64 natoms);

  /// @brief Identify owned atoms that are ghosts for the prev/next spatial neighbor.
  ///
  /// Call after partition_atoms. Indices are into the owned atom range [0, n_owned).
  void identify_send_ghosts(const Vec3* positions, i32 n_owned,
                            std::vector<i32>& send_to_prev,
                            std::vector<i32>& send_to_next) const;

  [[nodiscard]] real y_lo() const noexcept { return y_lo_; }
  [[nodiscard]] real y_hi() const noexcept { return y_hi_; }
  [[nodiscard]] real r_ghost() const noexcept { return r_ghost_; }
  [[nodiscard]] i32 n_spatial() const noexcept { return n_spatial_; }
  [[nodiscard]] i32 my_spatial() const noexcept { return my_spatial_; }
  [[nodiscard]] i32 prev_rank() const noexcept { return prev_rank_; }
  [[nodiscard]] i32 next_rank() const noexcept { return next_rank_; }

  /// @brief Check if atom at y-coordinate belongs to this spatial rank.
  [[nodiscard]] bool owns(real y) const noexcept;

 private:
  Box box_;
  i32 n_spatial_{1};
  i32 my_spatial_{0};
  i32 prev_rank_{0};
  i32 next_rank_{0};
  real y_lo_{0};
  real y_hi_{0};
  real r_ghost_{0};
  real slab_width_{0};
};

}  // namespace tdmd::domain
