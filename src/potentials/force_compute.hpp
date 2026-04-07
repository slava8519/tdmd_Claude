// SPDX-License-Identifier: Apache-2.0
// force_compute.hpp — top-level force evaluation for pair potentials.
#pragma once

#include "../core/system_state.hpp"
#include "../core/types.hpp"
#include "../neighbors/neighbor_list.hpp"
#include "ipair.hpp"

namespace tdmd::potentials {

/// @brief Compute forces and potential energy for a pair potential using a half-list.
///
/// Zeroes the force array, then accumulates forces using Newton's 3rd law.
/// @param state SystemState (forces will be overwritten).
/// @param nlist Neighbor list (half-list).
/// @param pair Pair potential.
/// @return Total potential energy.
[[nodiscard]] real compute_pair_forces(SystemState& state,
                                      const neighbors::NeighborList& nlist,
                                      const IPairStyle& pair);

}  // namespace tdmd::potentials
