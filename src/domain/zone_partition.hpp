// SPDX-License-Identifier: Apache-2.0
// zone_partition.hpp — partition box into zones along one axis.
#pragma once

#include <vector>

#include "../core/box.hpp"
#include "../core/types.hpp"
#include "../scheduler/zone.hpp"

namespace tdmd::domain {

/// @brief 1D zone partition along the X-axis.
///
/// Divides the simulation box into `n_zones` equal slabs along X.
/// Each zone has width >= r_c (enforced). Assigns atoms to zones based
/// on their X coordinate.
class ZonePartition {
 public:
  /// @brief Create a zone partition.
  /// @param box Simulation box.
  /// @param r_cut Force cutoff.
  /// @param n_zones Number of zones (0 = auto from box size / r_cut).
  void build(const Box& box, real r_cut, i32 n_zones = 0);

  /// @brief Assign atoms to zones and sort SystemState arrays by zone.
  ///
  /// After this call, atoms in zone k occupy indices
  /// [zone_offset(k), zone_offset(k) + zone_count(k)).
  /// Also sets Zone::atom_offset and Zone::natoms_in_zone.
  ///
  /// @param positions Per-atom positions (reordered in-place).
  /// @param velocities Per-atom velocities (reordered in-place).
  /// @param forces Per-atom forces (reordered in-place).
  /// @param types Per-atom types (reordered in-place).
  /// @param ids Per-atom ids (reordered in-place).
  /// @param natoms Number of atoms.
  void assign_atoms(PositionVec* positions, VelocityVec* velocities,
                    ForceVec* forces, i32* types, i32* ids, i64 natoms);

  [[nodiscard]] i32 n_zones() const noexcept {
    return static_cast<i32>(zones_.size());
  }
  [[nodiscard]] const std::vector<scheduler::Zone>& zones() const noexcept {
    return zones_;
  }
  [[nodiscard]] std::vector<scheduler::Zone>& zones() noexcept {
    return zones_;
  }

  /// @brief Zone index for a given x position.
  [[nodiscard]] i32 zone_of(real x) const noexcept;

  /// @brief Get precomputed neighbor zone indices for zone z_id.
  /// Neighbor zones are those within r_cut + r_skin distance along X.
  [[nodiscard]] const std::vector<i32>& zone_neighbors(i32 z_id) const {
    return zone_neighbors_[static_cast<std::size_t>(z_id)];
  }

  /// @brief Precompute zone neighbor lists.
  /// @param r_list Cutoff + skin.
  void build_zone_neighbors(real r_list);

  /// @brief Assign zones to MPI ranks (contiguous blocks).
  /// @param n_ranks Total number of MPI ranks.
  void assign_to_ranks(i32 n_ranks);

  /// @brief Which rank owns zone z_id.
  [[nodiscard]] i32 owner_rank(i32 z_id) const noexcept {
    return zones_[static_cast<std::size_t>(z_id)].owner_rank;
  }

  /// @brief Is zone z_id owned by my_rank?
  [[nodiscard]] bool is_local(i32 z_id, i32 my_rank) const noexcept {
    return owner_rank(z_id) == my_rank;
  }

  /// @brief First zone owned by rank r.
  [[nodiscard]] i32 first_zone_of_rank(i32 r) const noexcept;

  /// @brief Number of zones owned by rank r.
  [[nodiscard]] i32 n_zones_of_rank(i32 r) const noexcept;

  /// @brief Ghost zones: non-local zones that are neighbors of any local zone.
  [[nodiscard]] std::vector<i32> ghost_zones(i32 my_rank) const;

 private:
  real zone_width_{0};
  real box_lo_x_{0};
  i32 n_zones_{0};
  std::vector<scheduler::Zone> zones_;
  std::vector<std::vector<i32>> zone_neighbors_;
};

}  // namespace tdmd::domain
