// SPDX-License-Identifier: Apache-2.0
// types.hpp — fundamental scalar and vector types used everywhere in TDMD.
#pragma once

#include <array>
#include <cstdint>

namespace tdmd {

// ---- precision selection ----
#if defined(TDMD_FP64)
using real = double;
#else
using real = float;
#endif

// ---- aliases ----
using i32 = std::int32_t;
using i64 = std::int64_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;

// ---- minimal vec3 ----
// Plain aggregate so it works on host and (later) device without ceremony.
// All ops live as free functions in core/math.hpp once we need them.
struct Vec3 {
  real x;
  real y;
  real z;
};

// ---- axis-aligned bounding box ----
struct Aabb {
  Vec3 lo;
  Vec3 hi;
};

}  // namespace tdmd
