// SPDX-License-Identifier: Apache-2.0
// neighbor_list.cpp — Verlet neighbor list implementation (CPU, half-list).

#include "neighbor_list.hpp"

#include <cmath>
#include <vector>

#include "../core/error.hpp"
#include "../core/math.hpp"

namespace tdmd::neighbors {

void NeighborList::build(const PositionVec* positions, i64 natoms, const Box& box,
                         real r_cut, real r_skin) {
  r_cut_ = r_cut;
  r_skin_ = r_skin;
  const real r_list = r_cut + r_skin;
  const real r_list_sq = r_list * r_list;
  const Vec3D box_size = box.size();

  // Build cell list.
  cell_list_.build(positions, natoms, box, r_list);

  // Allocate per-atom arrays.
  auto n = static_cast<std::size_t>(natoms);
  counts_.assign(n, 0);
  offsets_.resize(n);

  // First pass: count neighbors per atom.
  // We use a temporary per-atom vector of neighbor lists, then flatten to CSR.
  std::vector<std::vector<i32>> per_atom(n);

  const i32 ncx = cell_list_.ncells_x();
  const i32 ncy = cell_list_.ncells_y();

  for (i64 i = 0; i < natoms; ++i) {
    const PositionVec pi = positions[static_cast<std::size_t>(i)];
    const i32 ci = cell_list_.cell_of(pi, box.lo);

    // Decompose flat cell index to (ix, iy, iz).
    const i32 iz = ci / (ncx * ncy);
    const i32 iy = (ci - iz * ncx * ncy) / ncx;
    const i32 ix = ci - iz * ncx * ncy - iy * ncx;

    // Loop over 27 neighbor cells (including self).
    for (i32 dz = -1; dz <= 1; ++dz) {
      for (i32 dy = -1; dy <= 1; ++dy) {
        for (i32 dx = -1; dx <= 1; ++dx) {
          const i32 nci = cell_list_.cell_index(ix + dx, iy + dy, iz + dz);
          const i32 cnt = cell_list_.count(nci);
          const i32* atoms = cell_list_.atoms_in_cell(nci);

          for (i32 k = 0; k < cnt; ++k) {
            const i32 j = atoms[k];
            if (j <= static_cast<i32>(i)) continue;  // half-list: only i < j

            Vec3D delta = positions[static_cast<std::size_t>(j)] - pi;
            delta = minimum_image(delta, box_size, box.periodic);
            const real r2 = static_cast<real>(length_sq(delta));

            if (r2 < r_list_sq) {
              per_atom[static_cast<std::size_t>(i)].push_back(j);
            }
          }
        }
      }
    }
  }

  // Flatten to CSR.
  i32 total = 0;
  for (std::size_t i = 0; i < n; ++i) {
    offsets_[i] = total;
    counts_[i] = static_cast<i32>(per_atom[i].size());
    total += counts_[i];
  }

  neighbors_.resize(static_cast<std::size_t>(total));
  for (std::size_t i = 0; i < n; ++i) {
    auto off = static_cast<std::size_t>(offsets_[i]);
    for (std::size_t k = 0; k < per_atom[i].size(); ++k) {
      neighbors_[off + k] = per_atom[i][k];
    }
  }

  // Save positions snapshot for skin check.
  build_positions_.assign(positions, positions + natoms);
}

bool NeighborList::needs_rebuild(const PositionVec* positions, i64 natoms) const {
  const double half_skin_sq =
      0.25 * static_cast<double>(r_skin_) * static_cast<double>(r_skin_);
  for (i64 i = 0; i < natoms; ++i) {
    auto si = static_cast<std::size_t>(i);
    Vec3D delta = positions[si] - build_positions_[si];
    if (length_sq(delta) > half_skin_sq) return true;
  }
  return false;
}

}  // namespace tdmd::neighbors
