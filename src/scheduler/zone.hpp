// SPDX-License-Identifier: Apache-2.0
// zone.hpp — Zone data structure and state machine.
//
// Spec: docs/01-theory/zone-state-machine.md
// All transitions in this header MUST match that spec exactly.
// Tests: tests/unit/test_zone_state_machine.cpp (added at M3).
#pragma once

#include <array>
#include <cstdint>

#include "../core/error.hpp"
#include "../core/types.hpp"

namespace tdmd::scheduler {

/// Lifecycle states of a Zone. See docs/01-theory/zone-state-machine.md.
enum class ZoneState : std::uint8_t {
  Free      = 0,
  Receiving = 1,
  Received  = 2,
  Ready     = 3,
  Computing = 4,
  Done      = 5,
  Sending   = 6,
};

/// Returns true if the transition (from -> to) is allowed by the state machine.
constexpr bool is_legal_transition(ZoneState from, ZoneState to) noexcept {
  switch (from) {
    case ZoneState::Free:      return to == ZoneState::Receiving;
    case ZoneState::Receiving: return to == ZoneState::Received;
    case ZoneState::Received:  return to == ZoneState::Ready;
    case ZoneState::Ready:     return to == ZoneState::Computing;
    case ZoneState::Computing: return to == ZoneState::Done;
    case ZoneState::Done:      return to == ZoneState::Sending;
    case ZoneState::Sending:   return to == ZoneState::Free;
  }
  return false;
}

/// Zone — the unit of work in TD.
/// Holds bookkeeping; the actual atom data lives in SystemState, indexed by [atom_offset, atom_offset+natoms_in_zone).
struct Zone {
  i32                  id{-1};
  std::array<i32, 3>   lattice_index{0, 0, 0};
  Aabb                 bbox{};
  i32                  atom_offset{0};
  i32                  natoms_in_zone{0};
  i32                  time_step{0};
  ZoneState            state{ZoneState::Free};
  i32                  owner_rank{-1};

  /// Transition this zone to a new state. Asserts legality.
  /// Increments time_step on Computing -> Done.
  void transition_to(ZoneState new_state) {
    TDMD_ASSERT(is_legal_transition(state, new_state),
                "illegal zone transition");
    state = new_state;
    if (new_state == ZoneState::Done) {
      ++time_step;
    }
  }
};

}  // namespace tdmd::scheduler
