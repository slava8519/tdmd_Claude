// SPDX-License-Identifier: Apache-2.0
// cell_list.cpp — CellList implementation.

#include "cell_list.hpp"

#include <algorithm>
#include <cmath>

#include "../core/error.hpp"

namespace tdmd::domain {

void CellList::build(const Vec3* positions, i64 natoms, const Box& box,
                     real r_list) {
  TDMD_ASSERT(r_list > real{0}, "r_list must be positive");
  TDMD_ASSERT(natoms >= 0, "natoms must be non-negative");

  const Vec3D box_size = box.size();

  // Determine number of cells: at least 3 per axis for PBC to work correctly.
  ncx_ = std::max(3, static_cast<i32>(std::floor(box_size.x / r_list)));
  ncy_ = std::max(3, static_cast<i32>(std::floor(box_size.y / r_list)));
  ncz_ = std::max(3, static_cast<i32>(std::floor(box_size.z / r_list)));

  cell_size_ = {static_cast<real>(box_size.x / ncx_),
                static_cast<real>(box_size.y / ncy_),
                static_cast<real>(box_size.z / ncz_)};

  const i32 ncells = ncx_ * ncy_ * ncz_;

  // Counting sort.
  cell_counts_.assign(static_cast<std::size_t>(ncells), 0);

  for (i64 i = 0; i < natoms; ++i) {
    i32 c = cell_of(positions[static_cast<std::size_t>(i)], box.lo);
    ++cell_counts_[static_cast<std::size_t>(c)];
  }

  // Prefix sum for offsets.
  cell_offsets_.resize(static_cast<std::size_t>(ncells));
  cell_offsets_[0] = 0;
  for (i32 c = 1; c < ncells; ++c) {
    cell_offsets_[static_cast<std::size_t>(c)] =
        cell_offsets_[static_cast<std::size_t>(c - 1)] +
        cell_counts_[static_cast<std::size_t>(c - 1)];
  }

  // Place atoms into sorted array.
  cell_atoms_.resize(static_cast<std::size_t>(natoms));
  // Temporary counter for placement.
  std::vector<i32> placed(static_cast<std::size_t>(ncells), 0);

  for (i64 i = 0; i < natoms; ++i) {
    i32 c = cell_of(positions[static_cast<std::size_t>(i)], box.lo);
    auto sc = static_cast<std::size_t>(c);
    i32 slot = cell_offsets_[sc] + placed[sc];
    cell_atoms_[static_cast<std::size_t>(slot)] = static_cast<i32>(i);
    ++placed[sc];
  }
}

}  // namespace tdmd::domain
