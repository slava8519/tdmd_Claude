// SPDX-License-Identifier: Apache-2.0
// box.hpp — orthorhombic simulation box with periodic boundary conditions.
//
// At M0 this is a stub. M1 fills in the math. The interface is what matters here.
#pragma once

#include "types.hpp"

namespace tdmd {

/// Orthorhombic simulation box. Triclinic comes later (post-M7).
struct Box {
  Vec3 lo{0, 0, 0};
  Vec3 hi{0, 0, 0};
  std::array<bool, 3> periodic{true, true, true};

  /// Box edge length along each axis.
  Vec3 size() const noexcept {
    return Vec3{hi.x - lo.x, hi.y - lo.y, hi.z - lo.z};
  }
};

}  // namespace tdmd
