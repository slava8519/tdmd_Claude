// SPDX-License-Identifier: Apache-2.0
// box.hpp — orthorhombic simulation box with periodic boundary conditions.
//
// At M0 this is a stub. M1 fills in the math. The interface is what matters here.
#pragma once

#include "types.hpp"

namespace tdmd {

/// Orthorhombic simulation box. Triclinic comes later (post-M7).
///
/// Geometry is stored in double precision regardless of build mode
/// (ADR 0007). Box dimensions are constants set at simulation start
/// — no precision benefit to float storage, and double avoids
/// precision-mode-dependent geometry bugs.
struct Box {
  Vec3D lo{0, 0, 0};
  Vec3D hi{0, 0, 0};
  std::array<bool, 3> periodic{true, true, true};

  /// Box edge length along each axis.
  Vec3D size() const noexcept {
    return Vec3D{hi.x - lo.x, hi.y - lo.y, hi.z - lo.z};
  }
};

}  // namespace tdmd
