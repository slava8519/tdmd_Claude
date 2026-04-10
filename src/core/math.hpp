// SPDX-License-Identifier: Apache-2.0
// math.hpp — Vec3 arithmetic and PBC helpers.
#pragma once

#include <cmath>

#include "types.hpp"

namespace tdmd {

// ---- Vec3T<T> arithmetic (host-only) ----
//
// Templated on the element type so the same operators work for Vec3
// (= Vec3T<real>), Vec3D (= Vec3T<double>) and Vec3F (= Vec3T<float>).

template<class T>
inline constexpr Vec3T<T> operator+(Vec3T<T> a, Vec3T<T> b) noexcept {
  return {a.x + b.x, a.y + b.y, a.z + b.z};
}

template<class T>
inline constexpr Vec3T<T> operator-(Vec3T<T> a, Vec3T<T> b) noexcept {
  return {a.x - b.x, a.y - b.y, a.z - b.z};
}

template<class T>
inline constexpr Vec3T<T> operator*(T s, Vec3T<T> v) noexcept {
  return {s * v.x, s * v.y, s * v.z};
}

template<class T>
inline constexpr Vec3T<T> operator*(Vec3T<T> v, T s) noexcept {
  return {v.x * s, v.y * s, v.z * s};
}

template<class T>
inline constexpr Vec3T<T>& operator+=(Vec3T<T>& a, Vec3T<T> b) noexcept {
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
  return a;
}

template<class T>
inline constexpr Vec3T<T>& operator-=(Vec3T<T>& a, Vec3T<T> b) noexcept {
  a.x -= b.x;
  a.y -= b.y;
  a.z -= b.z;
  return a;
}

template<class T>
inline constexpr T dot(Vec3T<T> a, Vec3T<T> b) noexcept {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

template<class T>
inline constexpr T length_sq(Vec3T<T> v) noexcept {
  return dot(v, v);
}

template<class T>
inline T length(Vec3T<T> v) noexcept {
  return std::sqrt(length_sq(v));
}

// ---- Periodic boundary conditions ----

/// Apply minimum-image convention: adjust delta so |delta[d]| <= half box size.
/// `box_size` = hi - lo for each dimension.
/// `periodic` = per-axis periodicity flag.
template<class T>
inline Vec3T<T> minimum_image(Vec3T<T> delta, Vec3T<T> box_size,
                              const std::array<bool, 3>& periodic) noexcept {
  if (periodic[0]) {
    if (delta.x > T{0.5} * box_size.x)
      delta.x -= box_size.x;
    else if (delta.x < T{-0.5} * box_size.x)
      delta.x += box_size.x;
  }
  if (periodic[1]) {
    if (delta.y > T{0.5} * box_size.y)
      delta.y -= box_size.y;
    else if (delta.y < T{-0.5} * box_size.y)
      delta.y += box_size.y;
  }
  if (periodic[2]) {
    if (delta.z > T{0.5} * box_size.z)
      delta.z -= box_size.z;
    else if (delta.z < T{-0.5} * box_size.z)
      delta.z += box_size.z;
  }
  return delta;
}

/// Wrap position into the box [lo, hi).
template<class T>
inline Vec3T<T> wrap_position(Vec3T<T> pos, Vec3T<T> lo, Vec3T<T> box_size,
                              const std::array<bool, 3>& periodic) noexcept {
  if (periodic[0]) {
    T rel = pos.x - lo.x;
    rel -= std::floor(rel / box_size.x) * box_size.x;
    pos.x = lo.x + rel;
  }
  if (periodic[1]) {
    T rel = pos.y - lo.y;
    rel -= std::floor(rel / box_size.y) * box_size.y;
    pos.y = lo.y + rel;
  }
  if (periodic[2]) {
    T rel = pos.z - lo.z;
    rel -= std::floor(rel / box_size.z) * box_size.z;
    pos.z = lo.z + rel;
  }
  return pos;
}

}  // namespace tdmd
