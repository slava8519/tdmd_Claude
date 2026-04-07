// SPDX-License-Identifier: Apache-2.0
// math.hpp — Vec3 arithmetic and PBC helpers.
#pragma once

#include <cmath>

#include "types.hpp"

namespace tdmd {

// ---- Vec3 arithmetic ----

inline constexpr Vec3 operator+(Vec3 a, Vec3 b) noexcept {
  return {a.x + b.x, a.y + b.y, a.z + b.z};
}

inline constexpr Vec3 operator-(Vec3 a, Vec3 b) noexcept {
  return {a.x - b.x, a.y - b.y, a.z - b.z};
}

inline constexpr Vec3 operator*(real s, Vec3 v) noexcept {
  return {s * v.x, s * v.y, s * v.z};
}

inline constexpr Vec3 operator*(Vec3 v, real s) noexcept {
  return {v.x * s, v.y * s, v.z * s};
}

inline constexpr Vec3& operator+=(Vec3& a, Vec3 b) noexcept {
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
  return a;
}

inline constexpr Vec3& operator-=(Vec3& a, Vec3 b) noexcept {
  a.x -= b.x;
  a.y -= b.y;
  a.z -= b.z;
  return a;
}

inline constexpr real dot(Vec3 a, Vec3 b) noexcept {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline constexpr real length_sq(Vec3 v) noexcept {
  return dot(v, v);
}

inline real length(Vec3 v) noexcept {
  return std::sqrt(length_sq(v));
}

// ---- Periodic boundary conditions ----

/// Apply minimum-image convention: adjust delta so |delta[d]| <= half box size.
/// `box_size` = hi - lo for each dimension.
/// `periodic` = per-axis periodicity flag.
inline Vec3 minimum_image(Vec3 delta, Vec3 box_size,
                          const std::array<bool, 3>& periodic) noexcept {
  if (periodic[0]) {
    if (delta.x > real{0.5} * box_size.x)
      delta.x -= box_size.x;
    else if (delta.x < real{-0.5} * box_size.x)
      delta.x += box_size.x;
  }
  if (periodic[1]) {
    if (delta.y > real{0.5} * box_size.y)
      delta.y -= box_size.y;
    else if (delta.y < real{-0.5} * box_size.y)
      delta.y += box_size.y;
  }
  if (periodic[2]) {
    if (delta.z > real{0.5} * box_size.z)
      delta.z -= box_size.z;
    else if (delta.z < real{-0.5} * box_size.z)
      delta.z += box_size.z;
  }
  return delta;
}

/// Wrap position into the box [lo, hi).
inline Vec3 wrap_position(Vec3 pos, Vec3 lo, Vec3 box_size,
                          const std::array<bool, 3>& periodic) noexcept {
  if (periodic[0]) {
    real rel = pos.x - lo.x;
    rel -= std::floor(rel / box_size.x) * box_size.x;
    pos.x = lo.x + rel;
  }
  if (periodic[1]) {
    real rel = pos.y - lo.y;
    rel -= std::floor(rel / box_size.y) * box_size.y;
    pos.y = lo.y + rel;
  }
  if (periodic[2]) {
    real rel = pos.z - lo.z;
    rel -= std::floor(rel / box_size.z) * box_size.z;
    pos.z = lo.z + rel;
  }
  return pos;
}

}  // namespace tdmd
