// SPDX-License-Identifier: Apache-2.0
// spatial_decomp.cpp — 1D spatial decomposition along Y-axis.

#include "spatial_decomp.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

#include "../core/error.hpp"

namespace tdmd::domain {

void SpatialDecomp::build(const Box& box, i32 n_spatial, i32 my_spatial,
                          real r_ghost) {
  TDMD_ASSERT(n_spatial >= 1, "need at least 1 spatial rank");
  TDMD_ASSERT(my_spatial >= 0 && my_spatial < n_spatial,
              "spatial rank out of range");

  box_ = box;
  n_spatial_ = n_spatial;
  my_spatial_ = my_spatial;
  r_ghost_ = r_ghost;

  Vec3D box_size = box.size();
  slab_width_ = static_cast<real>(box_size.y / n_spatial);

  // My Y-subdomain boundaries.
  y_lo_ = static_cast<real>(box.lo.y) + static_cast<real>(my_spatial) * slab_width_;
  y_hi_ = y_lo_ + slab_width_;

  // PBC neighbors in ring topology.
  prev_rank_ = (my_spatial - 1 + n_spatial) % n_spatial;
  next_rank_ = (my_spatial + 1) % n_spatial;
}

bool SpatialDecomp::owns(real y) const noexcept {
  return y >= y_lo_ && y < y_hi_;
}

i32 SpatialDecomp::partition_atoms(Vec3* positions, Vec3* velocities,
                                   Vec3* forces, i32* types, i32* ids,
                                   i64 natoms) {
  auto n = static_cast<std::size_t>(natoms);

  // Classify: owned (in my Y-slab) vs not-owned.
  std::vector<i32> owned_idx;
  owned_idx.reserve(n);
  for (std::size_t i = 0; i < n; ++i) {
    if (owns(positions[i].y)) {
      owned_idx.push_back(static_cast<i32>(i));
    }
  }

  i32 n_owned = static_cast<i32>(owned_idx.size());

  // Build permutation: owned atoms first (preserving relative order).
  std::vector<i32> perm;
  perm.reserve(n);
  for (i32 idx : owned_idx) perm.push_back(idx);
  // Non-owned atoms after (not needed on this rank, but keep for consistent size).
  for (std::size_t i = 0; i < n; ++i) {
    if (!owns(positions[i].y)) {
      perm.push_back(static_cast<i32>(i));
    }
  }

  // Apply permutation.
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

  return n_owned;
}

void SpatialDecomp::identify_send_ghosts(const Vec3* positions, i32 n_owned,
                                         std::vector<i32>& send_to_prev,
                                         std::vector<i32>& send_to_next) const {
  send_to_prev.clear();
  send_to_next.clear();

  real box_ly = static_cast<real>(box_.hi.y - box_.lo.y);

  for (i32 i = 0; i < n_owned; ++i) {
    real y = positions[static_cast<std::size_t>(i)].y;

    // Distance from atom to lower boundary of this subdomain.
    real dist_lo = y - y_lo_;
    // Distance from atom to upper boundary.
    real dist_hi = y_hi_ - y;

    // Near lower boundary → send to prev (with PBC wrapping).
    if (dist_lo < r_ghost_) {
      send_to_prev.push_back(i);
    }
    // For PBC: if this is the first slab and atom is near the box upper boundary.
    if (my_spatial_ == 0 && (box_ly - dist_lo) < r_ghost_) {
      // This shouldn't happen (atom is owned → y >= y_lo_),
      // but PBC wrap: prev sees atoms near the top of the box.
      // Actually handled from the perspective of the last rank.
    }

    // Near upper boundary → send to next.
    if (dist_hi < r_ghost_) {
      send_to_next.push_back(i);
    }
  }
}

}  // namespace tdmd::domain
