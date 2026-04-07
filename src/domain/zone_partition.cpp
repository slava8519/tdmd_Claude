// SPDX-License-Identifier: Apache-2.0
// zone_partition.cpp — zone partitioning implementation.

#include "zone_partition.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

#include "../core/error.hpp"

namespace tdmd::domain {

void ZonePartition::build(const Box& box, real r_cut, i32 n_zones) {
  Vec3 box_size = box.size();
  box_lo_x_ = box.lo.x;

  if (n_zones <= 0) {
    n_zones = std::max(3, static_cast<i32>(std::floor(box_size.x / r_cut)));
  }
  TDMD_ASSERT(n_zones >= 3, "need at least 3 zones for PBC");

  n_zones_ = n_zones;
  zone_width_ = box_size.x / static_cast<real>(n_zones);
  // zone_width_ < r_cut is allowed: build_zone_neighbors handles larger spans.

  zones_.resize(static_cast<std::size_t>(n_zones));
  for (i32 i = 0; i < n_zones; ++i) {
    auto si = static_cast<std::size_t>(i);
    zones_[si].id = i;
    zones_[si].lattice_index = {i, 0, 0};
    zones_[si].bbox.lo = {box.lo.x + static_cast<real>(i) * zone_width_,
                          box.lo.y, box.lo.z};
    zones_[si].bbox.hi = {box.lo.x + static_cast<real>(i + 1) * zone_width_,
                          box.hi.y, box.hi.z};
    zones_[si].state = scheduler::ZoneState::Free;
    zones_[si].time_step = 0;
    zones_[si].owner_rank = 0;
  }
}

i32 ZonePartition::zone_of(real x) const noexcept {
  auto z = static_cast<i32>((x - box_lo_x_) / zone_width_);
  if (z < 0) z = 0;
  if (z >= n_zones_) z = n_zones_ - 1;
  return z;
}

void ZonePartition::assign_atoms(Vec3* positions, Vec3* velocities,
                                 Vec3* forces, i32* types, i32* ids,
                                 i64 natoms) {
  auto n = static_cast<std::size_t>(natoms);

  // Determine zone for each atom.
  std::vector<i32> atom_zone(n);
  for (std::size_t i = 0; i < n; ++i) {
    atom_zone[i] = zone_of(positions[i].x);
  }

  // Count atoms per zone.
  auto nz = static_cast<std::size_t>(n_zones_);
  std::vector<i32> zone_count(nz, 0);
  for (std::size_t i = 0; i < n; ++i) {
    ++zone_count[static_cast<std::size_t>(atom_zone[i])];
  }

  // Compute zone offsets (prefix sum).
  std::vector<i32> zone_offset(nz, 0);
  for (std::size_t z = 1; z < nz; ++z) {
    zone_offset[z] = zone_offset[z - 1] + zone_count[z - 1];
  }

  // Build sort permutation (counting sort by zone).
  std::vector<i32> perm(n);
  std::vector<i32> placed(nz, 0);
  for (std::size_t i = 0; i < n; ++i) {
    auto z = static_cast<std::size_t>(atom_zone[i]);
    perm[static_cast<std::size_t>(zone_offset[z] + placed[z])] =
        static_cast<i32>(i);
    ++placed[z];
  }

  // Apply permutation to all arrays.
  std::vector<Vec3> tmp_v3(n);
  std::vector<i32> tmp_i32(n);

  // Positions.
  for (std::size_t i = 0; i < n; ++i) {
    tmp_v3[i] = positions[static_cast<std::size_t>(perm[i])];
  }
  std::copy(tmp_v3.begin(), tmp_v3.end(), positions);

  // Velocities.
  for (std::size_t i = 0; i < n; ++i) {
    tmp_v3[i] = velocities[static_cast<std::size_t>(perm[i])];
  }
  std::copy(tmp_v3.begin(), tmp_v3.end(), velocities);

  // Forces.
  for (std::size_t i = 0; i < n; ++i) {
    tmp_v3[i] = forces[static_cast<std::size_t>(perm[i])];
  }
  std::copy(tmp_v3.begin(), tmp_v3.end(), forces);

  // Types.
  for (std::size_t i = 0; i < n; ++i) {
    tmp_i32[i] = types[static_cast<std::size_t>(perm[i])];
  }
  std::copy(tmp_i32.begin(), tmp_i32.end(), types);

  // IDs.
  for (std::size_t i = 0; i < n; ++i) {
    tmp_i32[i] = ids[static_cast<std::size_t>(perm[i])];
  }
  std::copy(tmp_i32.begin(), tmp_i32.end(), ids);

  // Set zone metadata.
  for (std::size_t z = 0; z < nz; ++z) {
    zones_[z].atom_offset = zone_offset[z];
    zones_[z].natoms_in_zone = zone_count[z];
  }
}

void ZonePartition::build_zone_neighbors(real r_list) {
  auto nz = static_cast<std::size_t>(n_zones_);
  zone_neighbors_.resize(nz);

  // How many zone widths does r_list span?
  i32 span = static_cast<i32>(std::ceil(r_list / zone_width_));

  for (i32 z = 0; z < n_zones_; ++z) {
    auto sz = static_cast<std::size_t>(z);
    zone_neighbors_[sz].clear();

    for (i32 dz = -span; dz <= span; ++dz) {
      i32 nz_id = z + dz;
      // PBC wrap.
      if (nz_id < 0) nz_id += n_zones_;
      else if (nz_id >= n_zones_) nz_id -= n_zones_;
      zone_neighbors_[sz].push_back(nz_id);
    }

    // Remove duplicates (can happen if span >= n_zones/2).
    auto& v = zone_neighbors_[sz];
    std::sort(v.begin(), v.end());
    v.erase(std::unique(v.begin(), v.end()), v.end());
  }
}

}  // namespace tdmd::domain
