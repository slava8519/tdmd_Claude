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
//
// minimum_image and wrap_position take two template types:
//   TP — type of the position/delta (build-mode precision in CPU code)
//   TB — type of the box (always double after Stage 1: Box stores Vec3D)
// Box values are cast to TP for arithmetic so the result type matches
// the input position/delta type.

/// Apply minimum-image convention: adjust delta so |delta[d]| <= half box size.
/// `box_size` = hi - lo for each dimension.
/// `periodic` = per-axis periodicity flag.
template<class TP, class TB>
inline Vec3T<TP> minimum_image(Vec3T<TP> delta, Vec3T<TB> box_size,
                               const std::array<bool, 3>& periodic) noexcept {
  if (periodic[0]) {
    const TP bs = static_cast<TP>(box_size.x);
    if (delta.x > TP{0.5} * bs)
      delta.x -= bs;
    else if (delta.x < TP{-0.5} * bs)
      delta.x += bs;
  }
  if (periodic[1]) {
    const TP bs = static_cast<TP>(box_size.y);
    if (delta.y > TP{0.5} * bs)
      delta.y -= bs;
    else if (delta.y < TP{-0.5} * bs)
      delta.y += bs;
  }
  if (periodic[2]) {
    const TP bs = static_cast<TP>(box_size.z);
    if (delta.z > TP{0.5} * bs)
      delta.z -= bs;
    else if (delta.z < TP{-0.5} * bs)
      delta.z += bs;
  }
  return delta;
}

/// Wrap position into the box [lo, hi).
template<class TP, class TB>
inline Vec3T<TP> wrap_position(Vec3T<TP> pos, Vec3T<TB> lo, Vec3T<TB> box_size,
                               const std::array<bool, 3>& periodic) noexcept {
  if (periodic[0]) {
    const TP lox = static_cast<TP>(lo.x);
    const TP bsx = static_cast<TP>(box_size.x);
    TP rel = pos.x - lox;
    rel -= std::floor(rel / bsx) * bsx;
    pos.x = lox + rel;
  }
  if (periodic[1]) {
    const TP loy = static_cast<TP>(lo.y);
    const TP bsy = static_cast<TP>(box_size.y);
    TP rel = pos.y - loy;
    rel -= std::floor(rel / bsy) * bsy;
    pos.y = loy + rel;
  }
  if (periodic[2]) {
    const TP loz = static_cast<TP>(lo.z);
    const TP bsz = static_cast<TP>(box_size.z);
    TP rel = pos.z - loz;
    rel -= std::floor(rel / bsz) * bsz;
    pos.z = loz + rel;
  }
  return pos;
}

}  // namespace tdmd
